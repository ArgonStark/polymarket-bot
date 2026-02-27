"""
Main trading bot module.

Orchestrates all components: data feeds, signal generation,
risk management, and order execution.
"""
from __future__ import annotations

import asyncio
import logging
import os
from collections import deque
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from src.config import BotConfig
from src.models import MarketState, ChainlinkPrice, Signal, OrderAction, Side, PositionState
from src.data import (
    ChainlinkFeed,
    CLOBFeed,
    GammaAPI,
    BinanceFeed,
    BinanceTradeFeed,
    fetch_all_historical_prices,
    prepopulate_price_histories,
    get_data_api,
    get_settlement_verifier,
)
from src.data.binance import BinancePrice
from src.execution import create_trading_client, OrderExecutor
from src.execution.client import get_account_balance, get_trades
from src.execution.orders import OrderSafetyGuard, get_safety_guard
from src.strategy import SignalGenerator, RiskManager
from src.strategy.regime import RegimeDetector
from src.strategy.meta import MultiStrategyMeta, StrategyWeights
from src.strategy.risk_sizing import RiskSizer, RiskSizingConfig
from src.strategy.strategies.context import StrategyContext
from src.monitoring.metrics import MetricsCollector, MetricsConfig
from src.execution.low_latency.router import SmartOrderRouter, SmartRouterConfig
from src.execution.paper import PaperOrderExecutor, PaperTradingConfig
from src.state import BotStateManager
from src.kill_switch import KillSwitch
from src.attribution import AttributionTracker
from src.capital_scaling import CapitalScaler
from src.monitoring.dashboard import MetricsDashboard
from monitoring.monitor_server import MonitorServer
from src.strategy.trade_history import get_trade_history
from src.utils import log_trade, shutdown_notification_executor, print_status_box, print_config_box, Colors
from src.utils.logging import resolve_execution_mode, _MODE_LABELS

# Mixin imports
from src.bot.positions import PositionMixin
from src.bot.settlement import SettlementMixin
from src.bot.exits import EarlyExitMixin
from src.bot.monitoring import MonitoringMixin
from src.bot.trading import TradingMixin

logger = logging.getLogger(__name__)


class TradingBot(
    TradingMixin,
    SettlementMixin,
    EarlyExitMixin,
    MonitoringMixin,
    PositionMixin,
):
    """
    Polymarket 15-minute crypto arbitrage trading bot.

    Core strategy: Exploit temporal arbitrage between real-time
    Chainlink prices and lagging market odds.

    Market Lifecycle:
    1. Discovery: New markets found via Gamma API
    2. Active Trading: Generate signals, execute trades
    3. Expiring: Market approaching settlement, stop new trades
    4. Settlement: Check resolution, close positions, record P&L
    """

    def __init__(self, config: BotConfig):
        """
        Initialize the trading bot.

        Args:
            config: Bot configuration
        """
        self.config = config

        # Data feeds
        self.chainlink_feed = ChainlinkFeed(
            config=config,
            on_price_update=self._on_chainlink_price,
        )
        self.clob_feed = CLOBFeed(
            config=config,
            on_orderbook_update=self._on_orderbook_update,
        )
        self.binance_feed = BinanceFeed(
            config=config,
            on_price_update=self._on_binance_price,
        )
        self.gamma_api = GammaAPI(config=config)

        # Binance aggTrade feed for OFI (Order Flow Imbalance)
        self.binance_trade_feed = BinanceTradeFeed(config=config) if config.ofi.enabled else None

        # Regime detector (ATR + Efficiency Ratio)
        self.regime_detector = RegimeDetector(
            binance_feed=self.binance_feed,
            er_trending_threshold=config.regime.er_trending_threshold,
            er_ranging_threshold=config.regime.er_ranging_threshold,
            atr_low_percentile=config.regime.atr_low_percentile,
            ranging_kelly_mult=config.regime.ranging_kelly_mult,
            _cache_ttl=config.regime.cache_ttl,
        ) if config.regime.enabled else None

        # Trading components
        self.client = None
        self.executor = None
        self.smart_router = None
        self.signal_generator = SignalGenerator(config=config)
        self.risk_manager = RiskManager(config=config)

        # Multi-strategy meta layer
        self.meta_strategy = None
        if config.strategy.multi_strategy_enabled:
            weights = StrategyWeights(
                trend=config.strategy.trend_weight,
                mean_reversion=config.strategy.mean_reversion_weight,
                momentum_spike=config.strategy.momentum_weight,
            )
            self.meta_strategy = MultiStrategyMeta(weights)
            logger.info("🧠 Multi-strategy meta-layer enabled")

        # Dynamic risk sizing
        self.risk_sizer = RiskSizer(
            RiskSizingConfig(
                enabled=config.risk_sizing.enabled,
                target_volatility=config.risk_sizing.target_volatility,
                kelly_cap=config.risk_sizing.kelly_cap,
                max_exposure_pct=config.risk_sizing.max_exposure_pct,
                min_trade_usd=config.risk_sizing.min_trade_usd,
            )
        )

        # Monitoring
        self.metrics = MetricsCollector(
            MetricsConfig(
                enabled=config.monitoring.enabled,
                output_path=config.monitoring.output_path,
                rolling_window=config.monitoring.rolling_window,
            )
        )

        # Settlement verifier (on-chain verification via The Graph)
        self.settlement_verifier = get_settlement_verifier()
        if self.settlement_verifier.is_enabled():
            logger.info("📊 On-chain settlement verification enabled (The Graph)")

        # Graph stats cache for position status display
        self._graph_stats_cache: Optional[dict] = None
        self._graph_stats_last_fetch: Optional[datetime] = None
        self._graph_stats_refresh_interval = 300.0  # Refresh every 5 minutes

        # Market state - three-stage lifecycle
        self.markets: dict[str, MarketState] = {}  # Active trading markets
        self.expiring_markets: dict[str, MarketState] = {}  # Pending settlement
        self.settled_markets: set[str] = set()  # Already settled (prevent re-processing)
        self._markets_lock = asyncio.Lock()  # Thread safety for market operations

        # Token-to-market mapping for fast lookups
        self._token_to_market: dict[str, str] = {}  # token_id -> condition_id

        # Track logged rejections to avoid spam (cleared periodically)
        self._logged_rejections: set[str] = set()
        self._last_rejection_clear: Optional[datetime] = None
        self._rejection_clear_interval = 30.0  # Clear rejections every 30s to re-log
        self._signal_log_ts: dict[str, float] = {}
        self._price_log_ts: dict[str, float] = {}
        self._order_submit_ts: deque[float] = deque()
        self._last_order_time_by_market: dict[tuple[str, str], float] = {}

        # Status logging throttle - log status every 30 seconds instead of randomly
        self._last_status_log: Optional[datetime] = None
        self._status_log_interval = 30.0  # Log status every 30 seconds

        # Cooldown tracking - prevent duplicate orders per (asset, variant) pair
        self._last_order_time: dict[str, datetime] = {}  # "ASSET:variant" -> last order time
        self._order_cooldown_seconds = config.trading.order_cooldown_seconds  # Configurable (default 15s)

        # Cached API positions (updated by _sync_existing_orders)
        self._api_positions: dict[str, dict] = {}  # asset -> position info

        # Pending orders tracking - orders placed but not yet filled
        # Key: order_id, Value: dict with signal, ml_data, placed_time
        self._pending_orders: dict[str, dict] = {}
        self._order_check_interval = 5.0  # Check pending orders every 5 seconds
        self.last_order_check = None

        # Lock for concurrent access to pending orders
        self._pending_orders_lock = asyncio.Lock()

        # Timing trackers
        self.last_market_refresh = None
        self.last_settlement_check = None
        self.last_balance_sync = None
        self.last_orders_sync = None

        # Period tracking for market transitions
        # When a period boundary is crossed, we need to refresh markets with new target prices
        self._current_period_ts: dict[str, int] = {}  # variant -> current period ts
        self._period_transition_wait_until: Optional[datetime] = None  # Wait for price data
        self._captured_period_prices: dict[str, float] = {}  # asset -> price captured at period boundary

        # Active market tracking per (asset, variant) - detect transitions and cancel stale orders
        # Key: "ASSET:variant" (e.g. "BTC:fifteen"), Value: condition_id
        self._active_market_per_asset: dict[str, str] = {}

        # Safety guard for order submission (initialized lazily via get_safety_guard())
        self.safety_guard: Optional[OrderSafetyGuard] = None

        # State persistence
        state_file = config.paper_trading.state_file if config.paper_trading.enabled else "bot_state.json"
        self.state_manager = BotStateManager(state_file)
        self._last_state_save: Optional[datetime] = None
        self._state_save_interval = 60.0  # Save state every 60 seconds

        # Period boundary deduplication - prevent concurrent transitions
        self._period_boundary_lock = asyncio.Lock()
        self._last_handled_period_ts: int = 0  # Last period we completed transition for

        # Settlement backoff: per-market next-check time + attempt counter
        # Prevents log spam when API resolution is slow
        self._settle_next_check: dict[str, float] = {}  # market_id -> monotonic time
        self._settle_attempts: dict[str, int] = {}       # market_id -> retry count

        # Intervals (seconds)
        self.settlement_check_interval = 5.0  # Check settlements frequently
        self.market_discovery_interval = 15.0  # Discover new markets every 15s
        self.balance_sync_interval = 30.0  # Sync balance every 30s
        self.orders_sync_interval = 60.0  # Sync open orders every 60s
        self.period_transition_delay = 2.0  # Reduced: only wait 2s for real-time price capture
        self.position_log_interval = 30.0  # Log position status every 30s
        self.last_position_log = None  # Track last position log time

        # Compact position table throttling
        self._last_compact_table_ts: float = 0.0
        self._compact_table_interval: float = 10.0
        self._recent_settlements: deque = deque(maxlen=10)

        # Signal-processing diagnostics (reset each tick, displayed in compact table)
        self._tick_block_reasons: dict[str, int] = {}  # reason -> count
        self._tick_markets_processed: int = 0
        self._tick_signals_generated: int = 0
        self._tick_trades_executed: int = 0
        self._last_trade_ts: Optional[float] = None  # monotonic time of last successful trade

        # ORDER_BLOCK log throttling - prevent spam for repeated blocks
        # Key: "asset:reason" -> last log timestamp
        self._order_block_log_ts: dict[str, float] = {}
        self._order_block_log_interval = 5.0  # Only log same block once per 5s

        # Kill switch (hard safety gate)
        self.kill_switch = KillSwitch()

        # Trade attribution tracker
        self.attribution = AttributionTracker()

        # Capital scaling (drawdown-based sizing multiplier)
        self.capital_scaler = CapitalScaler()

        # Metrics dashboard (periodic snapshot)
        dashboard_path = os.getenv("METRICS_DASHBOARD_FILE", "")
        dashboard_interval = float(os.getenv("METRICS_DASHBOARD_INTERVAL", "30"))
        self.dashboard = MetricsDashboard(
            output_path=dashboard_path,
            interval=dashboard_interval,
        )

        # Real-time monitoring server (WebSocket + HTTP dashboard)
        self.monitor: Optional[MonitorServer] = None
        if config.monitoring.enabled:
            self.monitor = MonitorServer(port=config.monitoring.ws_port)

        # Control flags
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Warm-up / observation tracking
        self._startup_time: Optional[datetime] = None  # When bot started
        self._market_first_seen: dict[str, datetime] = {}  # market_id -> first observation time
        self._warmup_complete = False  # True after warm-up period ends

    @property
    def _execution_mode(self) -> str:
        """Derive execution mode — delegates to shared resolve_execution_mode()."""
        return resolve_execution_mode(self.config)

    async def initialize(self) -> bool:
        """
        Initialize bot components.

        Returns:
            True if initialization successful
        """
        logger.info("Initializing trading bot...")

        # Validate configuration
        is_valid, errors = self.config.validate()
        if not is_valid:
            for error in errors:
                logger.error(f"Config error: {error}")
            return False

        if self.config.paper_trading.enabled:
            logger.info("📝 Paper trading enabled - using simulated execution")
            self.client = None
            self.executor = PaperOrderExecutor(
                PaperTradingConfig(
                    enabled=True,
                    initial_balance=self.config.paper_trading.initial_balance,
                    maker_fee_bps=self.config.paper_trading.maker_fee_bps,
                    taker_fee_bps=self.config.paper_trading.taker_fee_bps,
                    slippage_bps=self.config.paper_trading.slippage_bps,
                ),
                self.clob_feed,
            )
        else:
            # Always try to create trading client to fetch account info
            # (even in dry-run mode, we want to show real balance if configured)
            self.client = create_trading_client(self.config)

            if not self.config.dry_run:
                if not self.client:
                    logger.error("Failed to create trading client")
                    return False

                self.executor = OrderExecutor(
                    client=self.client,
                    config=self.config,
                )
            else:
                logger.info("Running in DRY RUN mode - no actual trades")
                # Create dummy executor for dry run
                self.executor = OrderExecutor(
                    client=None,
                    config=self.config,
                )

        # Smart order router
        self.smart_router = SmartOrderRouter(
            SmartRouterConfig(
                enabled=self.config.execution.smart_router_enabled,
                maker_edge_threshold=self.config.execution.maker_edge_threshold,
                taker_edge_threshold=self.config.execution.taker_edge_threshold,
                max_spread=self.config.execution.max_spread,
                stale_order_seconds=self.config.execution.stale_order_seconds,
            ),
            self.executor,
            self.clob_feed,
        )

        # Log executor path for debugging
        executor_class = type(self.executor).__name__
        executor_module = type(self.executor).__module__
        router_status = "enabled" if self.config.execution.smart_router_enabled else "disabled"
        logger.info(
            "EXECUTOR_PATH class=%s module=%s mode=%s smart_router=%s",
            executor_class, executor_module, self._execution_mode, router_status
        )

        # Initialize safety guard for order submission limits
        self.safety_guard = get_safety_guard()

        # Initialize risk manager with actual bankroll
        initial_bankroll = await self._get_initial_bankroll()
        if initial_bankroll <= 0:
            logger.error("Cannot start: No bankroll available")
            return False
        self.risk_manager.initialize(initial_bankroll)

        # Restore persisted state (overwrites defaults if saved state exists)
        # Must happen BEFORE _display_startup_info so Account Status shows correct balance
        saved_state = self.state_manager.load()
        current_mode = self._execution_mode
        if saved_state:
            mode_changed = (
                saved_state.execution_mode
                and saved_state.execution_mode != current_mode
            )
            self.state_manager.restore_risk_manager(
                self.risk_manager, saved_state, current_mode=current_mode,
            )
            if self.config.paper_trading.enabled and hasattr(self.executor, 'account') and not mode_changed:
                self.state_manager.restore_paper_executor(self.executor, saved_state)
            # Restore settled_markets set
            if saved_state.settled_markets:
                self.settled_markets = set(saved_state.settled_markets)
                logger.info("Restored %d settled market IDs", len(self.settled_markets))

            logger.info(
                "STATE_RESTORED bankroll=%.2f peak=%.2f positions=%d",
                self.risk_manager.current_bankroll,
                self.risk_manager.peak_bankroll,
                len(self.risk_manager.positions),
            )
            # Restore kill-switch state if it was active at shutdown (skip on mode switch)
            ks_data = getattr(saved_state, 'kill_switch', None)
            if ks_data and not mode_changed:
                self.kill_switch = KillSwitch.from_dict(ks_data)
            elif ks_data and mode_changed:
                logger.info("STATE_MODE_SWITCH clearing kill_switch from %s mode", saved_state.execution_mode)

            # Clean up positions in markets that expired while bot was down
            self._cleanup_orphaned_positions()

        # Display startup info AFTER state restore so balance/positions are correct
        await self._display_startup_info()

        # Start real-time monitoring server
        if self.monitor:
            self.monitor.start()
            logger.info(
                "Monitor dashboard available at http://localhost:%d",
                self.monitor.http_port,
            )

        # Sync existing orders to prevent duplicates
        await self._sync_existing_orders(force=True)

        # Fetch historical price data to pre-populate price histories
        await self._fetch_historical_prices()



        logger.info("Trading bot initialized successfully")
        return True

    async def _sync_existing_orders(self, force: bool = False):
        """
        Sync existing orders and positions from Polymarket.

        This is critical to:
        1. Prevent duplicate orders when bot restarts
        2. Track real positions from the exchange
        3. Accurately reflect bankroll
        """
        now = datetime.now(timezone.utc)

        # Check if sync is needed (unless forced)
        if not force and self.last_orders_sync:
            elapsed = (now - self.last_orders_sync).total_seconds()
            if elapsed < self.orders_sync_interval:
                return

        if self.config.paper_trading.enabled and isinstance(self.executor, PaperOrderExecutor):
            balance = self.executor.get_balance()
            self.risk_manager.sync_bankroll(balance)
            self.last_balance_sync = now
            return

        if self.client is None:
            return

        try:
            from src.execution.client import get_open_orders, get_active_positions

            self.last_orders_sync = now
            synced_assets = set()
            token_to_asset = {"btc": "BTC", "eth": "ETH", "sol": "SOL", "xrp": "XRP"}

            # 1. Sync open orders (unfilled limit orders)
            open_orders = get_open_orders(self.client)
            open_order_count = len(open_orders)

            for order in open_orders:
                asset_id = order.get("asset_id", "").lower()
                market = order.get("market", "").lower()

                for key, asset in token_to_asset.items():
                    if key in asset_id or key in market:
                        synced_assets.add(asset)
                        # Set cooldown for all active variants (API doesn't tell us variant)
                        for _v in self.config.trading.trading_variants:
                            self._last_order_time[f"{asset}:{_v}"] = now
                        break

            # 2. Sync active positions (filled trades in active 5m/15m markets)
            positions_raw = get_active_positions(self.client)
            position_count = len(positions_raw)

            # Build asset-keyed cache for duplicate-trade prevention
            # (legacy format: keyed by asset name)
            asset_cache: dict[str, dict] = {}
            for _pk, pos_info in positions_raw.items():
                _a = pos_info.get("asset", _pk)
                if _a not in asset_cache:
                    asset_cache[_a] = pos_info
            self._update_cached_positions(asset_cache)

            for _pk, pos_info in positions_raw.items():
                asset = pos_info.get("asset", _pk.split(":")[0] if ":" in _pk else _pk)
                synced_assets.add(asset)
                variant = pos_info.get("variant", "fifteen")
                # Set cooldown so we don't try to trade this asset again
                self._last_order_time[f"{asset}:{variant}"] = now

                # Check if we already track this position internally
                condition_id = pos_info.get("conditionId", "")
                if not condition_id:
                    continue

                # Already tracked by risk manager → skip
                if condition_id in self.risk_manager.positions:
                    continue

                cost = pos_info.get("cost", 0)
                size = pos_info.get("size", 0)
                if cost <= 0 or size <= 0:
                    continue

                # Try to import into risk_manager.positions
                self._import_api_position(condition_id, pos_info)

            # 3. Log summary
            if force or synced_assets:
                bankroll = self.risk_manager.current_bankroll
                logger.info(
                    f"📋 SYNC: {open_order_count} open orders | "
                    f"{position_count} positions | "
                    f"Assets: {', '.join(synced_assets) if synced_assets else 'none'} | "
                    f"Bankroll: ${bankroll:.2f}"
                )

        except Exception as e:
            logger.warning(f"Could not sync orders/positions: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _import_api_position(self, condition_id: str, pos_info: dict):
        """
        Import an API-discovered position into risk_manager.positions.

        Called when _sync_existing_orders() finds a position on the Data API
        that isn't tracked internally (e.g., fill detection missed it).
        """
        from src.models import Position, Side

        asset = pos_info.get("asset", "???")
        variant = pos_info.get("variant", "fifteen")
        size = pos_info.get("size", 0)
        avg_price = pos_info.get("price", 0)
        held_token = pos_info.get("token_id", "")
        outcome = pos_info.get("outcome", "")
        cost = pos_info.get("cost", size * avg_price)

        # Find the matching MarketState
        market = self.markets.get(condition_id)
        if market is None:
            # Also check expiring markets (keyed by condition_id)
            market = getattr(self, 'expiring_markets', {}).get(condition_id)
        if market is None:
            logger.info(
                "POSITION_IMPORT_SKIP asset=%s condition=%s reason=no_matching_market",
                asset, condition_id[:16],
            )
            return

        # Determine side from token matching (primary) or outcome (fallback)
        side = None
        if held_token and market.up_token_id and held_token == market.up_token_id:
            side = Side.UP
        elif held_token and market.down_token_id and held_token == market.down_token_id:
            side = Side.DOWN
        elif outcome:
            # Fallback: "Yes" = UP, "No" = DOWN
            if outcome.lower() in ("yes", "up"):
                side = Side.UP
            elif outcome.lower() in ("no", "down"):
                side = Side.DOWN

        if side is None:
            logger.warning(
                "POSITION_IMPORT_SKIP asset=%s condition=%s reason=cannot_determine_side "
                "token=%s outcome=%s up_token=%s down_token=%s",
                asset, condition_id[:16],
                held_token[:16] if held_token else "none",
                outcome,
                market.up_token_id[:16] if market.up_token_id else "none",
                market.down_token_id[:16] if market.down_token_id else "none",
            )
            return

        # Set held token from market if not available from API
        if not held_token:
            held_token = market.up_token_id if side == Side.UP else market.down_token_id

        # Create Position and insert into risk manager
        position = Position(
            market=market,
            side=side,
            token_id=held_token,
            entry_price=avg_price,
            shares=size,
            entry_time=datetime.now(timezone.utc),
            market_id=condition_id,
            yes_token_id=market.up_token_id,
            no_token_id=market.down_token_id,
            held_token_id=held_token,
            total_cost=cost,
        )
        self.risk_manager.positions[condition_id] = position
        self.risk_manager.current_bankroll -= cost

        # Also credit paper executor if in paper mode
        if self.config.paper_trading.enabled and hasattr(self.executor, 'account'):
            self.executor.account.balance -= cost

        logger.info(
            "POSITION_IMPORTED asset=%s variant=%s side=%s shares=%.2f "
            "price=%.4f cost=%.2f condition=%s token=%s",
            asset, variant, side.value, size,
            avg_price, cost, condition_id[:16], held_token[:16],
        )

    async def _fetch_historical_prices(self):
        """
        Fetch historical price data at startup to pre-populate price histories.

        This allows the bot to:
        1. Have immediate context about recent price movements
        2. Calculate volatility estimates without waiting
        3. Make better trading decisions from the start
        """
        try:
            # Fetch last 1 hour of price data for all supported assets
            historical_data = await fetch_all_historical_prices(
                assets=self.config.supported_assets,
                hours=1,
            )

            # Pre-populate the signal generator's price histories
            prepopulate_price_histories(
                signal_generator=self.signal_generator,
                historical_data=historical_data,
            )

            # Log summary
            total_samples = sum(len(p) for p in historical_data.values())
            if total_samples > 0:
                logger.info(
                    f"📊 Historical data loaded: {total_samples} price points | "
                    f"Assets: {', '.join(self.config.supported_assets)}"
                )
            else:
                logger.warning(
                    "Could not fetch historical prices - will collect during observation"
                )

        except Exception as e:
            logger.warning(f"Failed to fetch historical prices: {e}")
            # Non-fatal - bot will collect data during observation period

    async def _sync_balance(self):
        """Periodically sync bankroll with actual Polymarket balance."""
        now = datetime.now(timezone.utc)

        # Check if sync is needed
        if self.last_balance_sync:
            elapsed = (now - self.last_balance_sync).total_seconds()
            if elapsed < self.balance_sync_interval:
                return

        if self.client is None:
            return

        try:
            balance = get_account_balance(self.client)
            if balance is not None:
                self.risk_manager.sync_bankroll(balance)
                self.last_balance_sync = now
        except Exception as e:
            logger.debug(f"Balance sync failed: {e}")

    def _refresh_paper_quotes(self):
        """
        Feed current market quotes into the paper executor's quote cache.

        Must run every tick (not just when pending orders exist) so that
        mark-to-market valuation always has fresh prices. Cache is keyed by
        token_id so two markets for the same asset cannot collide.
        """
        if not self.config.paper_trading.enabled:
            return
        if not self.executor or not hasattr(self.executor, 'update_quotes'):
            return

        for mkt in list(self.markets.values()) + list(self.expiring_markets.values()):
            if mkt.best_bid and mkt.best_ask:
                # YES (UP) token quotes come directly from MarketState
                self.executor.update_quotes(
                    mkt.up_token_id, mkt.best_bid, mkt.best_ask,
                    market_id=mkt.condition_id, source="market_state_up",
                )
                # NO (DOWN) token quotes are derived (complement of UP book)
                no_bid = 1.0 - mkt.best_ask  # best we can sell NO at
                no_ask = 1.0 - mkt.best_bid  # best we can buy NO at
                self.executor.update_quotes(
                    mkt.down_token_id, no_bid, no_ask,
                    market_id=mkt.condition_id, source="market_state_down_derived",
                )

    async def _check_pending_orders(self):
        """
        Check pending orders for fills and update positions accordingly.

        This ensures we only record positions when orders are actually filled,
        not just when they're placed.
        """
        now = datetime.now(timezone.utc)

        # Check if we need to run
        if self.last_order_check:
            elapsed = (now - self.last_order_check).total_seconds()
            if elapsed < self._order_check_interval:
                return

        self.last_order_check = now

        async with self._pending_orders_lock:
            if not self._pending_orders:
                return
            pending_items = list(self._pending_orders.items())

        if self.executor is None:
            return

        # For paper trading, feed market quotes into executor, then run fill simulation
        is_paper = self.config.paper_trading.enabled
        self._refresh_paper_quotes()
        if is_paper and hasattr(self.executor, 'check_pending_orders'):
            filled_paper_orders = self.executor.check_pending_orders()
            if filled_paper_orders:
                logger.info(
                    "PAPER_FILL_CHECK filled=%d pending_executor=%d",
                    len(filled_paper_orders),
                    self.executor.get_pending_count() if hasattr(self.executor, 'get_pending_count') else 0
                )

        # Log that we're checking pending orders
        logger.debug(f"Checking {len(pending_items)} pending orders... (paper={is_paper})")

        orders_to_remove = []

        for order_id, order_data in pending_items:
            try:
                asset = order_data.get("asset", "???")
                _variant = order_data.get("variant", "fifteen")
                _av_key = f"{asset}:{_variant}"
                placed_time = order_data.get("placed_time")
                wait_time = (now - placed_time).total_seconds() if placed_time else 0

                # Check order status
                order_info = self.executor.get_order_status(order_id)

                if order_info is None:
                    # Order not found - might have been filled or cancelled
                    if wait_time > 60:  # 1 minute timeout (reduced from 2)
                        # Check if API shows a position for this asset (order likely filled
                        # but was archived before we could check status)
                        signal = order_data.get("signal")
                        market_id = order_data.get("market_id", "")
                        api_has_position = (
                            hasattr(self, '_api_positions')
                            and asset in self._api_positions
                        )
                        if api_has_position and signal and market_id not in self.risk_manager.positions:
                            api_pos = self._api_positions[asset]
                            api_size = float(api_pos.get("size", 0)) if isinstance(api_pos, dict) else 0
                            api_price = float(api_pos.get("price", 0)) if isinstance(api_pos, dict) else 0
                            if api_size > 0:
                                logger.warning(
                                    "PENDING_ORDER_RESCUED order=%s asset=%s: API shows position "
                                    "(%s shares @ %.4f), recording fill",
                                    order_id[:8], asset, api_size, api_price,
                                )
                                self.risk_manager.record_position_open(
                                    signal=signal,
                                    entry_price=api_price or order_data.get("price", 0),
                                    shares=api_size,
                                )
                                orders_to_remove.append(order_id)
                                continue

                        logger.warning(
                            f"Order {order_id[:8]}... ({_av_key}) not found after {wait_time:.0f}s - removing"
                        )
                        orders_to_remove.append(order_id)
                        # Restore cooldown to allow new order
                        if _av_key in self._last_order_time:
                            del self._last_order_time[_av_key]
                    else:
                        logger.debug(f"Order {order_id[:8]}... ({asset}) status unknown, waiting {wait_time:.0f}s")
                    continue

                signal = order_data.get("signal")
                asset = order_data.get("asset", signal.market.asset if signal else "???")

                # Log the status we received
                side_val = order_data.get("side", "???")
                market_id_val = order_data.get("market_id", "????????")
                limit_price_val = order_data.get("price", 0)
                remaining = order_info.size - order_info.filled_size if order_info.size else 0
                logger.info(
                    f"📋 Order {order_id[:8]}... market={market_id_val[:8]} side={side_val} "
                    f"limit={limit_price_val:.4f} status={order_info.status.value} "
                    f"filled={order_info.filled_size:.2f} remaining={remaining:.2f} "
                    f"age={wait_time:.0f}s"
                )

                if order_info.status.value == "FILLED":
                    # Order filled - record the position
                    logger.info(
                        f"✅ ORDER FILLED: {asset} | "
                        f"Shares: {order_info.filled_size:.2f} @ {order_info.price:.4f}"
                    )

                    if signal:
                        # Record position with risk manager
                        logger.info(f"📍 Recording position for {asset} {signal.side.value}")
                        self.risk_manager.record_position_open(
                            signal=signal,
                            entry_price=order_info.price,
                            shares=order_info.filled_size,
                        )

                        # Record in trade history
                        trade_history = get_trade_history()
                        current_price = self.signal_generator.get_price(signal.market.asset)
                        trade_history.record_open(
                            asset=asset,
                            side=signal.side.value,
                            entry_price=order_info.price,
                            shares=order_info.filled_size,
                            target_price=signal.market.target_price,
                            chainlink_price=current_price,
                            market_id=signal.market.condition_id,
                            predicted_prob=None,
                            arb_type="none",
                            edge=signal.edge,
                            variant=signal.market.variant,
                        )

                    else:
                        logger.warning(f"Order filled but signal is None - cannot record position for {asset}")

                    orders_to_remove.append(order_id)

                elif order_info.status.value == "PARTIAL":
                    # Partially filled - check if essentially complete
                    fill_pct = order_info.fill_pct * 100
                    original_size = order_data.get("size", 0)

                    # If 95%+ filled, treat as FILLED (blockchain confirmation pending)
                    if fill_pct >= 95 or (original_size > 0 and order_info.filled_size >= original_size * 0.95):
                        logger.info(
                            f"✅ ORDER FILLED (via PARTIAL 100%): {asset} | "
                            f"Shares: {order_info.filled_size:.2f} @ {order_info.price:.4f}"
                        )

                        if signal:
                            # Record position with risk manager
                            logger.info(f"📍 Recording position for {asset} {signal.side.value}")
                            self.risk_manager.record_position_open(
                                signal=signal,
                                entry_price=order_info.price,
                                shares=order_info.filled_size,
                            )

                            # Record in trade history
                            trade_history = get_trade_history()
                            current_price = self.signal_generator.get_price(signal.market.asset)
                            trade_history.record_open(
                                asset=asset,
                                side=signal.side.value,
                                entry_price=order_info.price,
                                shares=order_info.filled_size,
                                target_price=signal.market.target_price,
                                chainlink_price=current_price,
                                market_id=signal.market.condition_id,
                                predicted_prob=None,
                                arb_type="none",
                                edge=signal.edge,
                                variant=signal.market.variant,
                            )

                        orders_to_remove.append(order_id)
                    else:
                        # Actually partial - log progress
                        logger.info(
                            f"⏳ ORDER PARTIAL: {asset} | "
                            f"Filled: {fill_pct:.0f}% ({order_info.filled_size:.2f} shares)"
                        )

                elif order_info.status.value in ["CANCELLED", "EXPIRED", "REJECTED"]:
                    # Order failed - remove and allow retry
                    logger.warning(
                        f"❌ ORDER {order_info.status.value}: {asset} | "
                        f"Order ID: {order_id[:16]}..."
                    )
                    orders_to_remove.append(order_id)
                    # Clear cooldown to allow immediate retry
                    if _av_key in self._last_order_time:
                        del self._last_order_time[_av_key]

                elif order_info.status.value in ["OPEN", "PENDING"]:
                    # Still open/pending - check if we should cancel
                    should_cancel = False
                    cancel_reason = ""

                    if signal:
                        market_id = signal.market.condition_id

                        # Cancel if market is expiring soon (60s buffer for safety)
                        if signal.market.time_remaining < 60:
                            should_cancel = True
                            cancel_reason = f"market expiring ({signal.market.time_remaining:.0f}s left)"

                        # Cancel if market is in expiring/settled queue (already expired)
                        elif market_id in self.expiring_markets or market_id in self.settled_markets:
                            should_cancel = True
                            cancel_reason = "market already expired"

                        # Cancel if market is no longer in active markets (period transitioned)
                        elif market_id not in self.markets:
                            should_cancel = True
                            cancel_reason = "market no longer active"

                    # Also cancel if order is too old (over 90 seconds)
                    if wait_time > 90:
                        should_cancel = True
                        cancel_reason = f"order timeout ({wait_time:.0f}s)"

                    if should_cancel:
                        logger.warning(
                            f"Cancelling unfilled order for {_av_key} - {cancel_reason}"
                        )
                        try:
                            self.executor.cancel_order(order_id)
                        except Exception as cancel_err:
                            logger.warning(f"Failed to cancel order: {cancel_err}")
                        orders_to_remove.append(order_id)
                        if _av_key in self._last_order_time:
                            del self._last_order_time[_av_key]
                    else:
                        logger.debug(f"Order {order_id[:8]}... ({_av_key}) still {order_info.status.value}, waiting {wait_time:.0f}s")

                else:
                    # Unknown status - log and apply timeout
                    logger.warning(f"Unknown order status '{order_info.status.value}' for {_av_key}")
                    if wait_time > 90:
                        logger.warning(f"Removing stale order for {_av_key} with status {order_info.status.value}")
                        orders_to_remove.append(order_id)
                        if _av_key in self._last_order_time:
                            del self._last_order_time[_av_key]

            except Exception as e:
                logger.error(f"Error checking order {order_id[:8]}...: {e}")
                # On error, still remove stale orders
                if wait_time > 120:
                    logger.warning(f"Removing errored order {order_id[:8]}... after {wait_time:.0f}s")
                    orders_to_remove.append(order_id)
                    if _av_key in self._last_order_time:
                        del self._last_order_time[_av_key]

        # Remove processed orders
        if orders_to_remove:
            async with self._pending_orders_lock:
                for order_id in orders_to_remove:
                    self._pending_orders.pop(order_id, None)

        # Log pending orders count
        async with self._pending_orders_lock:
            if self._pending_orders:
                logger.debug(f"Pending orders: {len(self._pending_orders)}")

    async def _display_startup_info(self):
        """
        Display account information at startup.

        Shows balance, trading mode, and key configuration parameters
        regardless of whether running in simulation or live mode.
        """
        # Derive mode from executor (single source of truth)
        mode = _MODE_LABELS.get(self._execution_mode, "UNKNOWN")
        trading_mode = self.config.trading.mode.upper()

        balance = None
        open_orders_count = 0
        positions_count = len(self.risk_manager.positions)  # Reflects restored positions

        if isinstance(self.executor, PaperOrderExecutor):
            balance = self.executor.get_balance()
        # Try to fetch real account info if client is available and configured
        elif self.client is not None:
            try:
                # Fetch balance using SDK
                balance = get_account_balance(self.client)

                # Fetch open orders count using SDK
                from src.execution.client import get_open_orders, get_active_positions
                open_orders = get_open_orders(self.client)
                open_orders_count = len(open_orders)

                # Fetch active positions
                active_positions = get_active_positions(self.client)
                positions_count = len(active_positions)

            except Exception as e:
                logger.warning(f"Could not fetch account info: {e}")

        # If no balance fetched and in dry run, use simulated
        if balance is None and self._execution_mode == "DRY":
            balance = (
                self.config.trading.base_position_size
                * self.config.trading.max_concurrent_positions
                * 5
            )

        # Daily stats from risk manager (may be restored from state)
        daily_pnl = self.risk_manager.daily_stats.total_pnl if self.risk_manager.daily_stats else 0.0

        # Print colorful status box
        print_status_box(
            mode=f"{mode} ({trading_mode})",
            balance=balance,
            open_orders=open_orders_count,
            positions=positions_count,
            daily_pnl=daily_pnl,
            win_rate=0.0,
        )

        # Print config box
        variants_str = ", ".join(self.config.trading.trading_variants)
        print_config_box({
            "Min Edge": self.config.trading.min_edge,
            "Min Time Remaining": f"{self.config.trading.min_time_remaining}s",
            "Base Position Size": self.config.trading.base_position_size,
            "Max Position %": self.config.trading.max_position_pct,
            "Max Concurrent": self.config.trading.max_concurrent_positions,
            "Daily Loss Limit": self.config.trading.daily_loss_limit,
            "Supported Assets": ", ".join(self.config.supported_assets),
            "Variants": variants_str,
        })

        # Display trade history summary
        await self._display_trade_history_summary()

    async def _display_trade_history_summary(self):
        """Display trade history performance summary at startup."""
        trade_history = get_trade_history()
        summary = trade_history.get_performance_summary()

        if not summary.get("has_data"):
            logger.info(
                f"{Colors.DIM}📊 No trade history yet - will learn from your trades{Colors.RESET}"
            )
            return

        # Build summary display
        completed = summary["completed_trades"]
        win_rate = summary["win_rate"]
        total_pnl = summary["total_pnl"]
        pred_accuracy = summary["prediction_accuracy"]

        # Color based on performance
        win_color = Colors.BRIGHT_GREEN if win_rate >= 0.5 else Colors.BRIGHT_YELLOW if win_rate >= 0.4 else Colors.BRIGHT_RED
        pnl_color = Colors.BRIGHT_GREEN if total_pnl > 0 else Colors.BRIGHT_RED

        logger.info(
            f"{Colors.BRIGHT_CYAN}📊 TRADE HISTORY REVIEW{Colors.RESET}"
        )
        logger.info(
            f"   Completed: {completed} trades | "
            f"{win_color}Win Rate: {win_rate:.0%}{Colors.RESET} | "
            f"{pnl_color}P&L: ${total_pnl:+.2f}{Colors.RESET}"
        )

        # Show recent performance
        recent_rate = summary.get("recent_win_rate", 0)
        recent_pnl = summary.get("recent_pnl", 0)
        recent_count = summary.get("recent_trades", 0)
        if recent_count > 0:
            recent_color = Colors.BRIGHT_GREEN if recent_rate >= 0.5 else Colors.BRIGHT_YELLOW if recent_rate >= 0.4 else Colors.BRIGHT_RED
            logger.info(
                f"   Recent ({recent_count}): {recent_color}{recent_rate:.0%} win{Colors.RESET} | "
                f"${recent_pnl:+.2f}"
            )

        # Show ML prediction accuracy if available
        if summary.get("predictions_made", 0) > 0:
            pred_color = Colors.BRIGHT_GREEN if pred_accuracy >= 0.5 else Colors.BRIGHT_YELLOW if pred_accuracy >= 0.45 else Colors.BRIGHT_RED
            logger.info(
                f"   ML Predictions: {pred_color}{pred_accuracy:.0%} accurate{Colors.RESET} "
                f"({summary['predictions_made']} predictions)"
            )

        # Show blocked assets/sides based on history
        blocked = []
        for asset in self.config.supported_assets:
            should_trade, reason = trade_history.should_trade_asset(
                asset,
                min_trades=self.config.trading.history_min_trades,
                min_win_rate=self.config.trading.history_min_win_rate,
            )
            if not should_trade:
                blocked.append(f"{asset} ({reason.split()[3]})")  # Extract win rate

        if blocked:
            logger.warning(
                f"   {Colors.BRIGHT_RED}⚠️ Blocked assets: {', '.join(blocked)}{Colors.RESET}"
            )

    async def start(self):
        """Start the trading bot."""
        if not await self.initialize():
            logger.error("Failed to initialize bot")
            return

        self._running = True

        # Startup banner
        logger.info("╔════════════════════════════════════════════════════════════════╗")
        logger.info("║     POLYMARKET 15-MIN CRYPTO ARBITRAGE BOT                     ║")
        logger.info("║     Made by Argon Stark                                        ║")
        logger.info("╚════════════════════════════════════════════════════════════════╝")

        # Mode-specific banner (derived from executor, not config flags)
        exec_mode = self._execution_mode
        if exec_mode == "LIVE":
            logger.warning("╔════════════════════════════════════════════════════════════════╗")
            logger.warning("║  ⚠️  LIVE TRADING MODE - REAL MONEY AT RISK ⚠️                  ║")
            logger.warning("╠════════════════════════════════════════════════════════════════╣")
            logger.warning(f"║  Max positions: {self.config.trading.max_concurrent_positions}                                              ║")
            max_exp = getattr(self.config.trading, 'max_total_exposure_pct', 0.25)
            logger.warning(f"║  Max exposure: {max_exp:.0%} of bankroll                                 ║")
            logger.warning(f"║  Bankroll: ${self.risk_manager.current_bankroll:.2f}                                       ║")
            logger.warning("║                                                                ║")
            logger.warning("║  Use --dry-run to test without real trades                     ║")
            logger.warning("╚════════════════════════════════════════════════════════════════╝")
        elif exec_mode == "PAPER":
            logger.info("╔════════════════════════════════════════════════════════════════╗")
            logger.info("║  📝  PAPER TRADING MODE - No real money at risk               ║")
            logger.info("╠════════════════════════════════════════════════════════════════╣")
            logger.info(f"║  Bankroll: ${self.risk_manager.current_bankroll:.2f}                                       ║")
            logger.info("╚════════════════════════════════════════════════════════════════╝")
        else:
            logger.info("🔸 DRY RUN MODE - No real trades will be placed")

        logger.info("Starting trading bot...")

        try:
            # Build list of tasks to run
            tasks = [
                self._run_chainlink_feed(),
                self._run_clob_feed(),
                self._run_trading_loop(),
                self._run_settlement_loop(),
            ]

            # Only add direct Binance feed if enabled
            # Otherwise, Binance prices come via Polymarket's WebSocket (bundled in ChainlinkFeed)
            if self.config.endpoints.binance_direct_enabled:
                logger.info("Using DIRECT Binance WebSocket connection")
                tasks.append(self._run_binance_feed())
            else:
                logger.info("Using Polymarket's bundled Binance prices (BINANCE_DIRECT=false)")

            # Binance aggTrade feed for Order Flow Imbalance
            if self.binance_trade_feed is not None:
                logger.info("Starting Binance aggTrade feed for OFI")
                tasks.append(self._run_binance_trade_feed())

            # Run all components concurrently
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            logger.info("Bot cancelled")
        except Exception as e:
            logger.error(f"Bot error: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """
        Gracefully shutdown the bot.

        Ordering is critical:
        1. Stop trading loop (no new orders)
        2. Cancel open orders
        3. Let settlement loop drain
        4. Disconnect feeds (prevents stale updates after state save)
        5. Persist state (final snapshot — no in-flight writes)
        """
        t0 = time.monotonic()
        logger.info("SHUTDOWN_PHASE phase=begin")

        # Phase 1: Stop all loops — prevents new order placement
        self._running = False
        self._shutdown_event.set()
        t1 = time.monotonic()
        logger.info("SHUTDOWN_PHASE phase=loops_stopped elapsed=%.3fs", t1 - t0)

        # Phase 2: Cancel open orders
        if self.executor and hasattr(self.executor, 'cancel_all_orders'):
            self.executor.cancel_all_orders()
        t2 = time.monotonic()
        logger.info("SHUTDOWN_PHASE phase=orders_cancelled elapsed=%.3fs", t2 - t0)

        # Phase 3: Small drain window — let any in-flight coroutines finish
        await asyncio.sleep(0.1)
        t3 = time.monotonic()
        logger.info("SHUTDOWN_PHASE phase=drain_complete elapsed=%.3fs", t3 - t0)

        # Phase 4: Disconnect data feeds BEFORE saving state.
        # This prevents feed callbacks from mutating state between save and exit.
        self.chainlink_feed.disconnect()
        self.clob_feed.disconnect()
        if self.config.endpoints.binance_direct_enabled:
            self.binance_feed.disconnect()
        if self.binance_trade_feed is not None:
            self.binance_trade_feed.disconnect()
        self.gamma_api.close()

        # Stop monitoring server
        if self.monitor:
            self.monitor.stop()

        # Shutdown notification thread pool
        shutdown_notification_executor()

        t4 = time.monotonic()
        logger.info("SHUTDOWN_PHASE phase=feeds_disconnected elapsed=%.3fs", t4 - t0)

        # Phase 5: Persist state (final snapshot — feeds are down, no more mutations)
        if self.state_manager:
            self.state_manager.save(
                self.risk_manager, self.markets,
                kill_switch=self.kill_switch,
                execution_mode=self._execution_mode,
                settled_markets=self.settled_markets,
            )
            self._check_state_invariants()
        t5 = time.monotonic()
        logger.info(
            "SHUTDOWN_PHASE phase=complete total=%.3fs "
            "(loops=%.3fs cancel=%.3fs drain=%.3fs disconnect=%.3fs save=%.3fs)",
            t5 - t0, t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4,
        )

    def _check_state_invariants(self):
        """
        Log-level assertions on state consistency after every STATE_SAVED.

        Checks:
        1. Paper mode: bankroll == paper_executor.balance
        2. Sum of open position costs <= bankroll + epsilon
        3. Serialized position count == risk_manager.positions count

        Never crashes — mismatches are logged as ERROR + STATE_DIFF block.
        """
        eps = 0.02  # rounding tolerance
        diffs = []

        # 1. Paper mode: bankroll must match paper executor balance
        if (
            self.config.paper_trading.enabled
            and isinstance(self.executor, PaperOrderExecutor)
        ):
            paper_bal = self.executor.get_balance()
            rm_bal = self.risk_manager.current_bankroll
            if abs(paper_bal - rm_bal) > eps:
                diffs.append(
                    f"bankroll={rm_bal:.2f} != paper_balance={paper_bal:.2f} "
                    f"(delta={rm_bal - paper_bal:+.2f})"
                )

        # 2. Sum of open position costs <= bankroll + epsilon
        total_cost = sum(
            p.cost_basis for p in self.risk_manager.positions.values()
        )
        bankroll = self.risk_manager.current_bankroll
        if total_cost > bankroll + eps:
            diffs.append(
                f"position_costs={total_cost:.2f} > bankroll={bankroll:.2f} "
                f"(over by {total_cost - bankroll:.2f})"
            )

        # 3. Position count consistency
        rm_count = len(self.risk_manager.positions)
        logger.info(
            "STATE_CHECK bankroll=%.2f positions=%d total_cost=%.2f ok=%s",
            bankroll, rm_count, total_cost, len(diffs) == 0,
        )

        if diffs:
            logger.error(
                "STATE_DIFF invariant_violations=%d:\n  %s",
                len(diffs), "\n  ".join(diffs),
            )

    def _get_binance_prices(self) -> dict[str, float]:
        """
        Get Binance prices from appropriate source based on config.

        If BINANCE_DIRECT=true: Uses direct Binance WebSocket (binance_feed)
        If BINANCE_DIRECT=false: Uses Polymarket's bundled Binance prices (chainlink_feed)

        Returns:
            Dict of asset -> price (e.g., {"BTC": 104000.50, "ETH": 3200.25})
        """
        if self.config.endpoints.binance_direct_enabled:
            return self.binance_feed.get_all_prices()
        else:
            return self.chainlink_feed.get_all_binance_prices()

    async def _get_initial_bankroll(self) -> float:
        """
        Get initial bankroll from exchange or config fallback.

        Returns:
            USDC balance, or fallback amount in dry run mode
        """
        if self.config.paper_trading.enabled:
            logger.info(f"[PAPER] Using virtual bankroll: ${self.config.paper_trading.initial_balance:.2f}")
            return self.config.paper_trading.initial_balance

        if self.config.dry_run:
            # Use configured fallback for dry run
            fallback = (
                self.config.trading.base_position_size
                * self.config.trading.max_concurrent_positions
                * 5  # 5x buffer for testing
            )
            logger.info(f"[DRY RUN] Using simulated bankroll: ${fallback:.2f}")
            return fallback

        # Try to fetch actual balance from exchange
        if self.client:
            try:
                balance = get_account_balance(self.client)
                if balance is not None and balance > 0:
                    logger.info(f"Fetched account balance: ${balance:.2f} USDC")
                    return balance
                else:
                    logger.warning("Account balance is zero or unavailable")
            except Exception as e:
                logger.error(f"Failed to fetch account balance: {e}")

        # Fallback: require manual configuration
        logger.error(
            "Could not fetch balance. Please ensure your wallet is funded "
            "and API credentials are correct."
        )
        return 0.0

    async def _run_chainlink_feed(self):
        """Run Chainlink price feed in background."""
        try:
            await self.chainlink_feed.connect_async()
        except Exception as e:
            logger.error(f"Chainlink feed error: {e}")

    async def _run_clob_feed(self):
        """Run CLOB order book feed in background."""
        try:
            await self.clob_feed.connect_async()
        except Exception as e:
            logger.error(f"CLOB feed error: {e}")

    async def _run_binance_feed(self):
        """
        Run Binance price feed in background.

        Binance provides faster price updates (~50ms) than Chainlink,
        used as a leading indicator to predict where Chainlink will go.
        """
        try:
            await self.binance_feed.connect_async()
        except Exception as e:
            logger.error(f"Binance feed error: {e}")

    async def _run_binance_trade_feed(self):
        """Run Binance aggTrade feed for Order Flow Imbalance."""
        try:
            await self.binance_trade_feed.connect_async()
        except Exception as e:
            logger.error(f"Binance aggTrade feed error: {e}")

    def _check_data_health(self) -> tuple[bool, str]:
        """
        Check if data feeds are healthy enough for trading.

        Returns:
            Tuple of (is_healthy, reason_if_not).
            is_healthy=True means all feeds are fresh and connected.
        """
        max_stale_sec = 60.0  # Max seconds before data is considered stale

        # Check circuit breakers
        chainlink_ok = not self.chainlink_feed._circuit_open
        clob_ok = not self.clob_feed._circuit_open

        if not chainlink_ok and not clob_ok:
            reason = "Both Chainlink and CLOB circuit breakers open"
            logger.warning("DATA_HEALTH FAIL: %s", reason)
            return (False, reason)

        if not chainlink_ok:
            reason = "Chainlink circuit breaker open"
            logger.warning("DATA_HEALTH DEGRADED: %s", reason)
            return (False, reason)

        if not clob_ok:
            reason = "CLOB circuit breaker open"
            logger.warning("DATA_HEALTH DEGRADED: %s", reason)
            return (False, reason)

        # Check connection state
        if not self.chainlink_feed.is_connected:
            reason = "Chainlink feed disconnected"
            logger.warning("DATA_HEALTH FAIL: %s", reason)
            return (False, reason)

        if not self.clob_feed.is_connected:
            reason = "CLOB feed disconnected"
            logger.warning("DATA_HEALTH FAIL: %s", reason)
            return (False, reason)

        # Check CLOB warmup after reconnect
        if not self.clob_feed.is_warmed_up:
            reason = "CLOB feed warming up after reconnect"
            logger.info("DATA_HEALTH WARMUP: %s", reason)
            return (False, reason)

        # Check CLOB orderbook freshness for active markets
        now = datetime.now(timezone.utc)
        for market in self.markets.values():
            ob = self.clob_feed.get_orderbook(market.up_token_id)
            if ob and ob.timestamp:
                age = (now - ob.timestamp).total_seconds()
                if age > max_stale_sec:
                    reason = (
                        f"CLOB orderbook stale for {market.asset} "
                        f"({age:.0f}s old, max {max_stale_sec:.0f}s)"
                    )
                    logger.warning("DATA_HEALTH STALE: %s", reason)
                    return (False, reason)

        # Check Chainlink price freshness (settlement-critical)
        chainlink_max_stale = 120.0  # Chainlink updates less frequently
        for asset in self.config.supported_assets:
            cl_age = self.chainlink_feed.get_chainlink_price_age(asset)
            if cl_age is not None and cl_age > chainlink_max_stale:
                reason = (
                    f"Chainlink price stale for {asset} "
                    f"({cl_age:.0f}s old, max {chainlink_max_stale:.0f}s)"
                )
                logger.warning("DATA_HEALTH STALE: %s", reason)
                return (False, reason)

        # Check Binance price freshness (if direct feed enabled)
        if self.config.endpoints.binance_direct_enabled:
            binance_max_stale = 30.0  # Binance should update every few seconds
            for asset in self.config.supported_assets:
                bn_age = self.chainlink_feed.get_binance_price_age(asset)
                if bn_age is not None and bn_age > binance_max_stale:
                    reason = (
                        f"Binance price stale for {asset} "
                        f"({bn_age:.0f}s old, max {binance_max_stale:.0f}s)"
                    )
                    logger.warning("DATA_HEALTH STALE: %s", reason)
                    return (False, reason)

        # aggTrade feed is non-blocking — log warning but don't gate trading
        if self.binance_trade_feed is not None and not self.binance_trade_feed.is_connected:
            logger.debug("DATA_HEALTH INFO: Binance aggTrade feed not connected (OFI unavailable)")

        return (True, "")

    async def _run_trading_loop(self):
        """Main trading loop - handles market discovery and signal execution."""
        logger.info("Starting trading loop...")

        # Record startup time for warm-up period
        self._startup_time = datetime.now(timezone.utc)
        warmup_seconds = self.config.trading.warmup_period_seconds

        logger.info(
            f"👀 OBSERVATION MODE: Watching prices for {warmup_seconds}s before trading..."
        )

        # Wait for data feeds to connect
        await asyncio.sleep(2.0)

        while self._running:
            try:
                # Check warm-up period
                if not self._warmup_complete:
                    elapsed = (datetime.now(timezone.utc) - self._startup_time).total_seconds()
                    if elapsed < warmup_seconds:
                        remaining = warmup_seconds - elapsed
                        # Log progress every 10 seconds
                        if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                            chainlink_prices = self.chainlink_feed.get_all_prices()
                            binance_prices = self._get_binance_prices()

                            # Build price comparison string
                            price_info = []
                            for asset in self.config.supported_assets:
                                cl_price = chainlink_prices.get(asset)
                                bn_price = binance_prices.get(asset)
                                if cl_price and bn_price:
                                    lead = bn_price - cl_price
                                    lead_pct = (lead / cl_price) * 100 if cl_price else 0
                                    price_info.append(
                                        f"{asset}: CL=${cl_price:,.0f} BN=${bn_price:,.0f} ({lead_pct:+.2f}%)"
                                    )
                                elif cl_price:
                                    price_info.append(f"{asset}: CL=${cl_price:,.0f}")

                            logger.info(
                                f"👀 OBSERVING: {remaining:.0f}s remaining | "
                                f"Markets: {len(self.markets)} | "
                                f"{' | '.join(price_info) if price_info else 'Waiting for prices...'}"
                            )
                        # Still discover markets and collect data, but don't trade
                        await self._refresh_markets()
                        await self._sync_balance()
                        await asyncio.sleep(self.config.loop_interval)
                        continue
                    else:
                        # Warm-up complete - verify data feeds
                        chainlink_prices = self.chainlink_feed.get_all_prices()
                        binance_prices = self._get_binance_prices()

                        # Check data availability
                        missing_chainlink = []
                        missing_binance = []
                        for asset in self.config.supported_assets:
                            if not chainlink_prices.get(asset):
                                missing_chainlink.append(asset)
                            if not binance_prices.get(asset):
                                missing_binance.append(asset)

                        # Warn about missing Binance data (but don't block)
                        if missing_binance:
                            logger.warning(
                                f"⚠️ Binance prices missing for: {', '.join(missing_binance)} - "
                                f"trading without confirmation signal for these assets"
                            )

                        # Block if Chainlink data missing (required for settlement)
                        if missing_chainlink:
                            logger.warning(
                                f"⚠️ Chainlink prices missing for: {', '.join(missing_chainlink)} - "
                                f"extending warm-up..."
                            )
                            await asyncio.sleep(5.0)
                            continue

                        # Warm-up complete!
                        self._warmup_complete = True

                        # Build final price summary with comparison
                        price_summary = []
                        for asset in self.config.supported_assets:
                            cl_price = chainlink_prices.get(asset)
                            bn_price = binance_prices.get(asset)
                            if cl_price and bn_price:
                                lead_pct = ((bn_price - cl_price) / cl_price) * 100
                                price_summary.append(f"{asset}=${cl_price:,.0f} (Binance {lead_pct:+.2f}%)")
                            elif cl_price:
                                price_summary.append(f"{asset}=${cl_price:,.0f} (no Binance)")

                        logger.info(
                            f"✅ WARM-UP COMPLETE: Now trading! | "
                            f"Markets: {len(self.markets)} | "
                            f"{' | '.join(price_summary)}"
                        )

                # Check data feed health before trading
                data_ok, data_reason = self._check_data_health()
                if not data_ok:
                    logger.warning("DATA_HEALTH gate: %s — skipping trade cycle", data_reason)
                    # Still run settlement and expiry checks even with degraded data
                    await self._check_expiring_markets()
                    await asyncio.sleep(10.0)
                    continue

                # Binance feed is optional (confirmation only) - just log if down
                # Only check circuit breaker when using direct Binance connection
                if self.config.endpoints.binance_direct_enabled and self.binance_feed._circuit_open:
                    logger.debug("Binance feed circuit breaker open - trading without confirmation signal")

                # Discover and refresh markets
                await self._refresh_markets()

                # Sync balance and open orders periodically
                await self._sync_balance()
                await self._sync_existing_orders()

                # Check pending orders for fills
                await self._check_pending_orders()

                # Check positions for early exit (take-profit / stop-loss)
                if self.config.trading.early_exit_enabled:
                    await self._check_early_exits()

                # Move expiring markets out of active trading
                await self._check_expiring_markets()

                # Refresh paper quote cache every tick so mark-to-market is fresh
                self._refresh_paper_quotes()

                # Calculate equity (cash + unrealized position value)
                # This prevents false drawdown triggers when positions are open
                equity = self._calculate_equity()

                # Update peak equity (only increases, never decreases)
                self.risk_manager.update_peak_equity(equity)

                # Apply gradual peak decay if active (after drawdown cooloff)
                self.risk_manager.apply_peak_decay()

                # Kill switch evaluation (hard safety — checked every tick)
                daily_loss_hit = (
                    self.risk_manager.daily_stats.hit_loss_limit_at(
                        self.config.trading.daily_loss_limit
                    )
                    if self.risk_manager.daily_stats
                    else False
                )
                self.kill_switch.evaluate(
                    peak_bankroll=self.risk_manager.peak_bankroll,
                    current_equity=equity,
                    daily_loss_hit=daily_loss_hit,
                    consecutive_losses=self.risk_manager.consecutive_losses,
                )

                # Metrics dashboard snapshot (every N seconds)
                realized_today = (
                    self.risk_manager.daily_stats.total_pnl
                    if self.risk_manager.daily_stats
                    else 0.0
                )
                unrealized = equity - self.risk_manager.current_bankroll
                self.dashboard.maybe_emit(
                    bankroll=self.risk_manager.current_bankroll,
                    peak_bankroll=self.risk_manager.peak_bankroll,
                    open_positions=len(self.risk_manager.positions),
                    total_exposure=self.risk_manager.get_total_exposure(),
                    realized_pnl_today=realized_today,
                    unrealized_pnl=unrealized,
                )

                # Push real-time monitoring snapshot
                if self.monitor:
                    self.monitor.push(self._build_monitor_snapshot(
                        equity=equity,
                        realized_today=realized_today,
                        unrealized=unrealized,
                    ))

                # Check if trading is allowed (use equity for drawdown check)
                can_trade, reason = self.risk_manager.can_trade(equity=equity)
                if not can_trade:
                    logger.warning(f"Trading paused: {reason}")
                    await asyncio.sleep(60.0)
                    continue

                # Periodically clear rejection log to re-show blocking reasons
                now = datetime.now(timezone.utc)
                if (self._last_rejection_clear is None or
                    (now - self._last_rejection_clear).total_seconds() > self._rejection_clear_interval):
                    if self._logged_rejections:
                        logger.debug(f"Clearing {len(self._logged_rejections)} logged rejections for re-logging")
                    self._logged_rejections.clear()
                    self._last_rejection_clear = now

                    # Prune tracking dicts — remove entries for markets no longer active/expiring
                    active_ids = set(self.markets.keys()) | set(self.expiring_markets.keys())
                    stale_first_seen = [k for k in self._market_first_seen if k not in active_ids]
                    for k in stale_first_seen:
                        del self._market_first_seen[k]
                    stale_order_ts = [k for k in self._last_order_time_by_market if k[0] not in active_ids]
                    for k in stale_order_ts:
                        del self._last_order_time_by_market[k]
                    if stale_first_seen or stale_order_ts:
                        logger.debug(
                            "DICT_PRUNE first_seen=%d order_ts=%d remaining_first_seen=%d remaining_order_ts=%d",
                            len(stale_first_seen), len(stale_order_ts),
                            len(self._market_first_seen), len(self._last_order_time_by_market),
                        )

                # Reset tick-level diagnostics before processing markets
                self._tick_block_reasons.clear()
                self._tick_markets_processed = 0
                self._tick_signals_generated = 0
                self._tick_trades_executed = 0

                # Sort markets by asset priority (BTC/ETH first for better liquidity)
                priority = self.config.trading.asset_priority
                sorted_markets = sorted(
                    self.markets.values(),
                    key=lambda m: priority.index(m.asset) if m.asset in priority else 99
                )

                # Process markets - parallel or sequential based on config
                if self.config.trading.parallel_execution:
                    # Parallel execution - process all markets simultaneously
                    results = await asyncio.gather(
                        *[self._process_market(m) for m in sorted_markets],
                        return_exceptions=True
                    )
                    # Log any exceptions that were silently returned
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            market = sorted_markets[i] if i < len(sorted_markets) else None
                            asset = market.asset if market else "UNKNOWN"
                            logger.error(f"[{asset}] Market processing error: {result}")
                else:
                    # Sequential execution
                    for market in sorted_markets:
                        await self._process_market(market)

                # ── POST-PROCESSING: Status & position display ──
                # Placed AFTER signal processing so position info isn't buried
                # under per-market ML/ORDER_BLOCK/PREFLIGHT noise.
                now_status = datetime.now(timezone.utc)
                if (self._last_status_log is None or
                    (now_status - self._last_status_log).total_seconds() > self._status_log_interval):
                    self._log_status_line()
                    self._last_status_log = now_status

                self._log_compact_table()

                # Detailed position status (every 30s) — mark prices, win prob, etc.
                self._log_position_status()

                # Periodic state save (crash protection)
                now_save = datetime.now(timezone.utc)
                if (
                    self._last_state_save is None
                    or (now_save - self._last_state_save).total_seconds() >= self._state_save_interval
                ):
                    self.state_manager.save(
                        self.risk_manager, self.markets,
                        kill_switch=self.kill_switch,
                        execution_mode=self._execution_mode,
                        settled_markets=self.settled_markets,
                    )
                    self._check_state_invariants()
                    self._last_state_save = now_save

                # Wait before next iteration
                await asyncio.sleep(self.config.loop_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(5.0)

        logger.info("Trading loop stopped")

    async def _run_settlement_loop(self):
        """
        Settlement loop - checks for market resolutions and closes positions.

        Runs independently from the trading loop to ensure timely
        settlement processing.
        """
        logger.info("Starting settlement loop...")

        # Wait for initial market discovery
        await asyncio.sleep(5.0)

        while self._running:
            try:
                await self._check_settlements()

                await asyncio.sleep(self.settlement_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Settlement loop error: {e}")
                await asyncio.sleep(5.0)

        logger.info("Settlement loop stopped")

    def _capture_period_boundary_prices(self, period_ts: int):
        """
        Capture current Chainlink prices at period boundary.

        At the exact moment a new 15-minute period starts, the current
        Chainlink price becomes the target price for that period's markets.
        This allows real-time transitions without waiting for the API.

        Args:
            period_ts: Unix timestamp of the new period start
        """
        self._captured_period_prices.clear()

        for asset in self.config.supported_assets:
            price = self.signal_generator.get_price(asset)
            if price and price > 0:
                self._captured_period_prices[asset] = price
                logger.info(
                    f"📍 Captured {asset} price at period boundary: ${price:,.2f}"
                )
            else:
                logger.warning(f"⚠️ No Chainlink price available for {asset} at boundary")

        if self._captured_period_prices:
            logger.info(
                f"✅ Captured {len(self._captured_period_prices)} prices for period {period_ts}"
            )

    def _inject_captured_prices_to_gamma(self, period_ts: int):
        """
        Inject captured prices into Gamma API cache.

        This allows immediate market creation without waiting for
        the Polymarket API to update with the new target prices.

        Args:
            period_ts: Unix timestamp of the period
        """
        if not self._captured_period_prices:
            logger.debug("No captured prices to inject")
            return

        injected = 0
        for asset, price in self._captured_period_prices.items():
            # Inject for all active variants so both 5m and 15m can use captured price
            for variant in self.config.trading.trading_variants:
                cache_key = f"{asset}:{variant}:{period_ts}"
                self.gamma_api._target_price_cache[cache_key] = price
                injected += 1
                logger.debug(f"Injected {asset}/{variant} target price ${price:,.2f} for period {period_ts}")

        if injected > 0:
            logger.info(f"Injected {injected} target prices into cache for instant trading")

    async def _refresh_markets(self):
        """Refresh list of active markets for all configured variants."""
        now = datetime.now(timezone.utc)
        current_ts = int(now.timestamp())

        # Detect period boundary crossing for each active variant
        # 5-min boundaries: every 300s, 15-min boundaries: every 900s
        _VARIANT_PERIODS = {"five": 300, "fifteen": 900}
        boundary_crossed = False

        for variant in self.config.trading.trading_variants:
            period = _VARIANT_PERIODS.get(variant, 900)
            current_period_ts = (current_ts // period) * period
            prev = self._current_period_ts.get(variant, 0)

            if prev > 0 and current_period_ts != prev:
                boundary_crossed = True
                logger.info(
                    "PERIOD_BOUNDARY variant=%s from=%d to=%d",
                    variant, prev, current_period_ts,
                )

            self._current_period_ts[variant] = current_period_ts

        if boundary_crossed:
            # Use lock to ensure only one transition runs at a time
            async with self._period_boundary_lock:
                # Capture current Chainlink prices as target prices for new period
                ref_ts = self._current_period_ts.get("fifteen", self._current_period_ts.get("five", current_ts))
                if self._last_handled_period_ts >= ref_ts:
                    pass  # already handled
                else:
                    self._last_handled_period_ts = ref_ts
                    self._capture_period_boundary_prices(ref_ts)

                    # Brief wait to ensure price capture is stable
                    if self._period_transition_wait_until is None:
                        wait_seconds = self.period_transition_delay
                        self._period_transition_wait_until = now + timedelta(seconds=wait_seconds)
                        logger.debug(f"Brief {wait_seconds}s stabilization wait...")

                    if now < self._period_transition_wait_until:
                        return

                    logger.info("Period transition complete - using captured prices")
                    self._period_transition_wait_until = None

                    # Clear stale target price cache
                    self.gamma_api.clear_stale_price_cache(ref_ts)

                    # Inject captured prices into gamma API cache for immediate use
                    self._inject_captured_prices_to_gamma(ref_ts)

                    # Move expired markets to expiring queue for settlement
                    async with self._markets_lock:
                        old_markets = list(self.markets.keys())
                        for market_id in old_markets:
                            market = self.markets.get(market_id)
                            if market and market.time_remaining <= 0:
                                # Skip if already in expiring or settled
                                if market_id in self.expiring_markets or market_id in self.settled_markets:
                                    self.markets.pop(market_id, None)
                                    logger.debug(f"Removed expired market (already tracked): {market.asset}/{market.variant}")
                                    continue

                                self.markets.pop(market_id, None)

                                # Snapshot Chainlink price for local resolution
                                if market.expiry_chainlink_price is None:
                                    try:
                                        snap_price = self.signal_generator.get_price(market.asset)
                                        market.expiry_chainlink_price = snap_price
                                        logger.info(
                                            "EXPIRY_SNAPSHOT market=%s asset=%s "
                                            "chainlink=%.2f target=%.2f time_remaining=%.1fs "
                                            "(period_transition)",
                                            market_id[:8], market.asset,
                                            snap_price or 0, market.target_price,
                                            market.time_remaining,
                                        )
                                    except Exception as e:
                                        logger.warning(f"Failed to snapshot expiry price for {market.asset}: {e}")

                                self.expiring_markets[market_id] = market
                                logger.info(f"Period transition: moved expired market to expiring queue: {market.asset}/{market.variant}")

                    # Force refresh by clearing last_market_refresh
                    self.last_market_refresh = None

        # Check if refresh is needed
        if self.last_market_refresh:
            elapsed = (now - self.last_market_refresh).total_seconds()
            if elapsed < self.market_discovery_interval:
                return

        logger.debug("Refreshing active markets...")

        # Collect stale order cancellations to run outside the lock
        stale_cancellations: list[tuple[str, str]] = []  # [(asset, old_market_id), ...]

        try:
            # Fetch active crypto markets for all configured variants
            new_markets = self.gamma_api.get_crypto_markets(
                variants=self.config.trading.trading_variants,
            )

            # Update markets dict with lock
            async with self._markets_lock:
                new_market_ids = set()
                for market in new_markets:
                    market_id = market.condition_id
                    new_market_ids.add(market_id)

                    # Skip already settled markets
                    if market_id in self.settled_markets:
                        continue

                    # Skip markets already in expiring queue
                    if market_id in self.expiring_markets:
                        continue

                    if market_id not in self.markets:
                        # New market discovered - subscribe to order book
                        self.markets[market_id] = market

                        # Update token-to-market mapping for O(1) lookups
                        self._token_to_market[market.up_token_id] = market_id
                        self._token_to_market[market.down_token_id] = market_id

                        self.clob_feed.subscribe(market.up_token_id)
                        self.clob_feed.subscribe(market.down_token_id)

                        # Check for market transition - collect for cancellation outside lock
                        asset = market.asset
                        av_key = f"{asset}:{market.variant}"
                        old_market_id = self._active_market_per_asset.get(av_key)
                        if old_market_id and old_market_id != market_id:
                            # Market transition detected - queue for cancellation
                            stale_cancellations.append((asset, old_market_id))

                        # Update active market tracking (both local and safety guard)
                        self._active_market_per_asset[av_key] = market_id
                        guard = self.safety_guard or get_safety_guard()
                        if guard:
                            guard.set_active_market(asset, market_id, variant=market.variant)

                        # Target price is now fetched from Polymarket API in gamma.py
                        logger.info(
                            "NEW_MARKET asset=%s variant=%s target=$%.2f "
                            "ends=%s time=%.0fs",
                            market.asset, market.variant,
                            market.target_price,
                            market.end_time.strftime('%H:%M:%S'),
                            market.time_remaining,
                        )
                    else:
                        # Update existing market state
                        existing = self.markets[market_id]
                        existing.best_bid = market.best_bid
                        existing.best_ask = market.best_ask
                        existing.last_updated = now

                        # IMPORTANT: Update target price if it changed (period transition)
                        if market.target_price != existing.target_price:
                            logger.info(
                                f"TARGET PRICE UPDATE {market.asset}: "
                                f"${existing.target_price:,.2f} → ${market.target_price:,.2f}"
                            )
                            existing.target_price = market.target_price

            self.last_market_refresh = now

            # Run stale order cancellations outside the lock (once per transition)
            for asset, old_market_id in stale_cancellations:
                await self._cancel_stale_orders(asset, old_market_id)

            # Clear old rejection logs for settled markets
            self._logged_rejections = {
                k for k in self._logged_rejections
                if k.split(":")[0] in self.markets or k.split(":")[0] in self.expiring_markets
            }

            logger.debug(
                f"Markets: {len(self.markets)} active, "
                f"{len(self.expiring_markets)} expiring"
            )

        except Exception as e:
            logger.error(f"Failed to refresh markets: {e}")

    async def _check_expiring_markets(self):
        """
        Move markets approaching expiry to the expiring queue.

        Markets are moved when time_remaining < min_time_remaining,
        which prevents new trades but keeps them for settlement tracking.
        Also cancels any pending orders for expiring markets.
        Uses per-variant min_time_remaining (5-min markets have tighter window).
        """
        min_time_15m = self.config.trading.min_time_remaining
        min_time_5m = self.config.trading.min_time_remaining_5m

        async with self._markets_lock:
            # Find markets that should stop trading
            markets_to_expire = []
            for market_id, market in self.markets.items():
                min_time = min_time_5m if market.variant == "five" else min_time_15m
                if market.time_remaining <= min_time:
                    markets_to_expire.append(market_id)

            # Move them to expiring queue
            for market_id in markets_to_expire:
                market = self.markets.pop(market_id)

                # Snapshot Chainlink price at expiry for local resolution
                # Only set once (guard against duplicate snapshots on re-entry)
                if market.expiry_chainlink_price is None:
                    try:
                        snap_price = self.signal_generator.get_price(market.asset)
                        market.expiry_chainlink_price = snap_price
                        tr = market.time_remaining
                        if tr < -5.0:
                            logger.warning(
                                "EXPIRY_SNAPSHOT_LATE market=%s asset=%s "
                                "chainlink=%.2f target=%.2f time_remaining=%.1fs "
                                "(snapshot taken >5s after expiry)",
                                market_id[:8], market.asset,
                                snap_price or 0, market.target_price, tr,
                            )
                        else:
                            logger.info(
                                "EXPIRY_SNAPSHOT market=%s asset=%s "
                                "chainlink=%.2f target=%.2f time_remaining=%.1fs",
                                market_id[:8], market.asset,
                                snap_price or 0, market.target_price, tr,
                            )
                    except Exception as e:
                        logger.warning(f"Failed to snapshot expiry price for {market.asset}: {e}")

                self.expiring_markets[market_id] = market

                # Cancel any pending orders for this market's asset
                asset = market.asset
                await self._cancel_pending_orders_for_asset(asset)

                # Clear cooldown so new market for this (asset, variant) can trade
                av_key = f"{asset}:{market.variant}"
                if av_key in self._last_order_time:
                    del self._last_order_time[av_key]
                    logger.debug("Cleared cooldown for %s (market expiring)", av_key)

                # Clear from cached API positions
                if hasattr(self, '_api_positions') and asset in self._api_positions:
                    del self._api_positions[asset]

                logger.info(
                    "EXPIRING asset=%s variant=%s time=%.0fs target=$%.2f",
                    market.asset, market.variant,
                    market.time_remaining, market.target_price,
                )

    async def _cancel_pending_orders_for_asset(self, asset: str):
        """
        Cancel all pending orders for a specific asset.

        Called when a market expires to ensure we can trade
        the new market for that asset.

        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP)
        """
        async with self._pending_orders_lock:
            if not self._pending_orders:
                return
            pending_items = list(self._pending_orders.items())

        orders_to_cancel = []
        for order_id, order_data in pending_items:
            if order_data.get("asset") == asset:
                orders_to_cancel.append(order_id)

        if orders_to_cancel:
            async with self._pending_orders_lock:
                for order_id in orders_to_cancel:
                    try:
                        if self.executor:
                            logger.info(f"Cancelling pending order for {asset} (market expiring)")
                            self.executor.cancel_order(order_id)
                    except Exception as e:
                        logger.warning(f"Failed to cancel order {order_id[:16]}...: {e}")
                    finally:
                        # Remove from pending regardless of cancel success
                        self._pending_orders.pop(order_id, None)

    async def _cancel_stale_orders(self, asset: str, old_market_id: str):
        """
        Cancel all orders for an old market when a new market becomes active.

        Called during market transitions (period rollovers) to prevent stale orders
        from the old market from getting filled incorrectly.

        This method is wrapped in try/except to prevent exceptions from propagating
        and causing "Task exception was never retrieved" errors.

        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP)
            old_market_id: condition_id of the old market being replaced
        """
        try:
            await self._cancel_stale_orders_impl(asset, old_market_id)
        except Exception as e:
            logger.error(
                f"ORDER_CANCEL_ERROR asset={asset} market={old_market_id[:8]} "
                f"error={str(e)[:100]}"
            )
            logger.debug("Stale order cancellation traceback:", exc_info=True)

    async def _cancel_stale_orders_impl(self, asset: str, old_market_id: str):
        """Internal implementation of stale order cancellation."""
        cancelled = 0
        already_filled = 0
        not_found = 0
        failed = 0
        orders_found = []

        is_paper = self.config.paper_trading.enabled

        # 0. Log paper order registry state for diagnostics
        if is_paper and self.executor and hasattr(self.executor, '_pending_orders'):
            with self.executor._orders_lock:
                executor_pending = list(self.executor._pending_orders.keys())
                executor_filled = list(self.executor._filled_orders.keys())
                stale_in_executor = [
                    oid for oid, order in self.executor._pending_orders.items()
                    if order.asset == asset and order.market_id == old_market_id
                ]
            logger.info(
                f"PAPER_ORDER_REGISTRY_DUMP asset={asset} old_market={old_market_id[:8]} "
                f"executor_pending={len(executor_pending)} executor_filled={len(executor_filled)} "
                f"stale_in_executor={len(stale_in_executor)} stale_ids={[o[:12] for o in stale_in_executor]}"
            )

        # 1. Collect orders from local pending tracking
        async with self._pending_orders_lock:
            pending_items = list(self._pending_orders.items())

        for order_id, order_data in pending_items:
            order_market_id = order_data.get("market_id", "")
            if order_data.get("asset") == asset and order_market_id == old_market_id:
                orders_found.append((order_id, order_data.get("side", "UNKNOWN")))

        # 2. Also check safety guard's open orders tracking (with defensive None check)
        guard = self.safety_guard or get_safety_guard()
        if guard:
            with guard._lock:
                for (a, m, s), oid in list(guard._open_orders.items()):
                    if a == asset and m == old_market_id:
                        if (oid, s) not in orders_found:
                            orders_found.append((oid, s))

        order_count = len(orders_found)
        if order_count == 0:
            logger.debug(
                f"ORDER_CANCEL_REQUEST asset={asset} market={old_market_id[:8]} order_count=0 "
                f"reason=market_transition (no orders to cancel)"
            )
            return

        # Log ORDER_CANCEL_REQUEST
        logger.info(
            f"ORDER_CANCEL_REQUEST asset={asset} market={old_market_id[:8]} order_count={order_count} "
            f"reason=market_transition"
        )

        # 3. Cancel each order (both in executor and local tracking)
        for order_id, side in orders_found:
            try:
                # Cancel in executor (works for both paper and live)
                if self.executor and hasattr(self.executor, 'cancel_order'):
                    result = self.executor.cancel_order(order_id)
                    # Handle both old (bool) and new (tuple) return types
                    if isinstance(result, tuple):
                        success, reason = result
                    else:
                        success, reason = result, "ok" if result else "failed"
                else:
                    success, reason = True, "no_executor"

                if success:
                    if reason == "cancelled":
                        cancelled += 1
                    elif reason == "already_filled":
                        already_filled += 1
                    elif reason == "not_found":
                        not_found += 1
                    else:
                        cancelled += 1  # Generic success

                    logger.info(
                        f"ORDER_CANCEL_RESULT asset={asset} market={old_market_id[:8]} "
                        f"order_id={order_id[:12]} side={side} status=success reason={reason}"
                    )
                else:
                    failed += 1
                    logger.warning(
                        f"ORDER_CANCEL_RESULT asset={asset} market={old_market_id[:8]} "
                        f"order_id={order_id[:12]} side={side} status=failed reason={reason}"
                    )
            except Exception as e:
                failed += 1
                logger.warning(
                    f"ORDER_CANCEL_RESULT asset={asset} market={old_market_id[:8]} "
                    f"order_id={order_id[:12]} side={side} status=error error={str(e)[:50]}"
                )

        # 4. Clear from local tracking regardless of cancel success
        async with self._pending_orders_lock:
            for order_id, _ in orders_found:
                self._pending_orders.pop(order_id, None)

        # 5. Clear from safety guard (with defensive None check)
        guard = self.safety_guard or get_safety_guard()
        if guard:
            with guard._lock:
                keys_to_remove = [
                    k for k in guard._open_orders
                    if k[0] == asset and k[1] == old_market_id
                ]
                for key in keys_to_remove:
                    del guard._open_orders[key]

        # Log ORDER_CANCEL_SUMMARY with detailed breakdown
        mode = "paper" if is_paper else "live"
        logger.info(
            f"ORDER_CANCEL_SUMMARY asset={asset} market={old_market_id[:8]} "
            f"cancelled={cancelled} already_filled={already_filled} not_found={not_found} failed={failed} mode={mode}"
        )

    def _on_chainlink_price(self, price: ChainlinkPrice):
        """Handle Chainlink price update."""
        # Update signal generator with new price (used for probability calculations)
        self.signal_generator.update_price(price.symbol, price.price)

    def _on_binance_price(self, price: BinancePrice):
        """
        Handle Binance price update.

        Binance prices are faster than Chainlink (~50ms vs ~500ms).
        Used as a leading indicator to predict Chainlink movements.
        """
        # Update signal generator with Binance price (used for confirmation)
        self.signal_generator.update_binance_price(price.asset, price.price)

    def _on_orderbook_update(self, token_id: str, orderbook):
        """Handle order book update."""
        # Use token-to-market mapping for O(1) lookup instead of O(n) iteration
        market_id = self._token_to_market.get(token_id)
        if not market_id:
            return

        # Find market in active or expiring dict
        market = self.markets.get(market_id) or self.expiring_markets.get(market_id)
        if not market:
            return

        # Only update prices for UP token (which determines market price)
        if token_id == market.up_token_id:
            if orderbook.best_bid is not None:
                market.best_bid = orderbook.best_bid
            if orderbook.best_ask is not None:
                market.best_ask = orderbook.best_ask
            market.last_updated = datetime.now(timezone.utc)


async def main(argv=None) -> None:
    if argv is not None:
        import sys
        sys.argv = [sys.argv[0], *argv]
    from src.bot.run import main as run_main
    await run_main()
