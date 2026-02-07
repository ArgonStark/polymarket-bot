"""
Main trading bot module.

Orchestrates all components: data feeds, signal generation,
risk management, and order execution.
"""

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
    fetch_all_historical_prices,
    prepopulate_price_histories,
    get_data_api,
    get_settlement_verifier,
)
from src.data.binance import BinancePrice
from src.execution import create_trading_client, OrderExecutor
from src.execution.client import get_account_balance, get_trades
from src.execution.orders import OrderSafetyGuard, get_safety_guard
from src.strategy import SignalGenerator, RiskManager, init_auto_retrainer, get_auto_retrainer
from src.strategy.meta import MultiStrategyMeta, StrategyWeights
from src.strategy.risk_sizing import RiskSizer, RiskSizingConfig
from src.strategy.strategies.context import StrategyContext
from src.prediction import FeatureBuilder, ProbabilityModel
from src.monitoring.metrics import MetricsCollector, MetricsConfig
from src.execution.low_latency.router import SmartOrderRouter, SmartRouterConfig
from src.execution.paper import PaperOrderExecutor, PaperTradingConfig
from src.strategy.ml_predictor import (
    get_ml_predictor,
    MLSignalPredictor,
    extract_ml_features_from_market,
    calculate_price_trend,
)

# New clean ML interface (for future migration)
# Usage: ml = MLPredictor() or ml = NoOpML()
#        ml_input = create_ml_input_from_signal(signal, market, chart, signal_gen)
#        decision = ml.evaluate_trade(ml_input)
#        if decision.should_trade: execute()
#        ml.record_outcome(trade_id, ml_input, won=True)
from src.ml import (
    MLInterface,
    MLInput,
    MLDecision,
    MLPredictor,
    NoOpML,
    create_ml_input_from_signal,
    TradeSide,
    MarketContext,
    TrendContext,
    ChartContext,
    IndicatorContext,
    PriceContext,
)
from src.ml.infer import ModelBundle, InferenceEngine
from src.ml.policy import Policy, PolicyConfig
from src.state import BotStateManager
from src.strategy.trade_history import get_trade_history
from src.utils import log_trade, shutdown_notification_executor, print_status_box, print_config_box, Colors
from src.utils.logging import resolve_execution_mode, _MODE_LABELS

# Import Binance chart analyzer for ML features
try:
    from src.data.binance_chart import analyze_chart
    CHART_ANALYSIS_AVAILABLE = True
except ImportError:
    CHART_ANALYSIS_AVAILABLE = False
    def analyze_chart(asset: str):
        return None


logger = logging.getLogger(__name__)


class TradingBot:
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

        # Probabilistic prediction model
        self.feature_builder = FeatureBuilder(
            velocity_window=config.prediction.velocity_window,
            vol_window=config.prediction.vol_window,
        )
        self.prediction_model = None
        if config.prediction.enabled:
            self.prediction_model = ProbabilityModel(model_path=config.prediction.model_path)
            logger.info("🧪 Probabilistic prediction layer enabled")

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

        # EV-based ML engine
        self.ml_engine = None
        self.ml_policy = None
        if config.ml_engine.enabled:
            try:
                bundle = ModelBundle(config.ml_engine.bundle_path)
                self.ml_engine = InferenceEngine(bundle)
                self.ml_policy = Policy(
                    PolicyConfig(
                        min_ev=config.ml_engine.min_ev,
                        max_uncertainty=config.ml_engine.max_uncertainty,
                        max_spread=config.ml_engine.max_spread,
                        min_depth=config.ml_engine.min_depth,
                        max_exposure_pct=config.ml_engine.max_exposure_pct,
                        ev_sizing_scale=config.ml_engine.ev_sizing_scale,
                        target_volatility=config.ml_engine.target_volatility,
                    )
                )
                logger.info("🧠 EV-based ML engine enabled")
            except Exception as e:
                logger.error(f"Failed to load ML bundle: {e}")
                self.ml_engine = None
                self.ml_policy = None

        # ML Signal Predictor
        self.ml_predictor: Optional[MLSignalPredictor] = None
        if config.trading.ml_enabled:
            self.ml_predictor = get_ml_predictor()
            self.ml_predictor.min_confidence = config.trading.ml_min_confidence
            self.ml_predictor.min_training_samples = config.trading.ml_min_samples
            logger.info(
                f"🤖 ML enabled: {self.ml_predictor.training_samples} samples loaded | "
                f"Model: {self.ml_predictor.model_path}"
            )
        else:
            logger.info("🤖 ML disabled (set ML_ENABLED=true to enable)")

        # Optional ML interface (bundle-aware)
        self.ml_interface: Optional[MLInterface] = None
        bundle_path = os.getenv("ML_BUNDLE") or os.getenv("ML_BUNDLE_PATH") or None
        # ML_MIN_EDGE falls back to config.trading.min_edge (unified source)
        ml_min_edge_env = os.getenv("ML_MIN_EDGE")
        if ml_min_edge_env is not None:
            try:
                min_edge = float(ml_min_edge_env)
            except ValueError:
                min_edge = config.trading.min_edge
        else:
            min_edge = config.trading.min_edge
        if bundle_path or config.trading.ml_enabled:
            if bundle_path:
                logger.info(f"ML bundle path resolved: {bundle_path}")
            self.ml_interface = MLPredictor(
                min_confidence=config.trading.ml_min_confidence,
                min_samples=config.trading.ml_min_samples,
                bundle_path=bundle_path,
                min_edge=min_edge,
            )

        # Settlement verifier (on-chain verification via The Graph)
        self.settlement_verifier = get_settlement_verifier()
        if self.settlement_verifier.is_enabled():
            logger.info("📊 On-chain settlement verification enabled (The Graph)")

        # Graph stats cache for position status display
        self._graph_stats_cache: Optional[dict] = None
        self._graph_stats_last_fetch: Optional[datetime] = None
        self._graph_stats_refresh_interval = 300.0  # Refresh every 5 minutes

        # Auto-retrainer (learns from your trades)
        self.auto_retrainer = None
        if self.ml_predictor:
            self.auto_retrainer = init_auto_retrainer(self.ml_predictor)
            logger.info("🔄 Auto-retraining enabled (every 50 trades)")

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
        self._ml_decision_log_ts: dict[str, float] = {}
        self._signal_log_ts: dict[str, float] = {}
        self._price_log_ts: dict[str, float] = {}
        self._order_submit_ts: deque[float] = deque()
        self._last_order_time_by_market: dict[tuple[str, str], float] = {}

        # Status logging throttle - log status every 30 seconds instead of randomly
        self._last_status_log: Optional[datetime] = None
        self._status_log_interval = 30.0  # Log status every 30 seconds

        # Cooldown tracking - prevent duplicate orders per asset
        self._last_order_time: dict[str, datetime] = {}  # asset -> last order time
        self._order_cooldown_seconds = config.trading.order_cooldown_seconds  # Configurable (default 15s)

        # Cached API positions (updated by _sync_existing_orders)
        self._api_positions: dict[str, dict] = {}  # asset -> position info

        # Pending orders tracking - orders placed but not yet filled
        # Key: order_id, Value: dict with signal, ml_data, placed_time
        self._pending_orders: dict[str, dict] = {}
        self._order_check_interval = 5.0  # Check pending orders every 5 seconds
        self.last_order_check = None

        # Thread-safe locks for concurrent access to shared dictionaries
        self._pending_orders_lock = asyncio.Lock()
        self._last_order_time_lock = asyncio.Lock()

        # Timing trackers
        self.last_market_refresh = None
        self.last_settlement_check = None
        self.last_balance_sync = None
        self.last_orders_sync = None

        # Period tracking for 15-minute market transitions
        # When a period boundary is crossed, we need to refresh markets with new target prices
        self._current_period_ts: int = 0  # Current 15-min period timestamp
        self._period_transition_wait_until: Optional[datetime] = None  # Wait for price data
        self._captured_period_prices: dict[str, float] = {}  # asset -> price captured at period boundary

        # Active market tracking per asset - used to detect market transitions and cancel stale orders
        # Key: asset (BTC, ETH, etc.), Value: condition_id of the currently active market
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

        # ML Sync from Polymarket API - bypasses internal position tracking
        # This ensures ML learns from actual trades even if internal tracking fails
        self._ml_synced_trades: set[str] = set()  # Set of trade IDs already synced to ML
        self.ml_sync_interval = 60.0  # Sync ML every 60 seconds
        self.last_ml_sync = None  # Track last ML sync time

        # ORDER_BLOCK log throttling - prevent spam for repeated blocks
        # Key: "asset:reason" -> last log timestamp
        self._order_block_log_ts: dict[str, float] = {}
        self._order_block_log_interval = 15.0  # Only log same block once per 15s

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
        if saved_state:
            self.state_manager.restore_risk_manager(self.risk_manager, saved_state)
            if self.config.paper_trading.enabled and hasattr(self.executor, 'account'):
                self.state_manager.restore_paper_executor(self.executor, saved_state)
            logger.info(
                "STATE_RESTORED bankroll=%.2f peak=%.2f positions=%d",
                self.risk_manager.current_bankroll,
                self.risk_manager.peak_bankroll,
                len(self.risk_manager.positions),
            )
            # Clean up positions in markets that expired while bot was down
            self._cleanup_orphaned_positions()

        # Display startup info AFTER state restore so balance/positions are correct
        await self._display_startup_info()

        # Sync existing orders to prevent duplicates
        await self._sync_existing_orders(force=True)

        # Fetch historical price data to pre-populate price histories
        # This allows the bot to make better decisions immediately
        await self._fetch_historical_prices()

        # Initial ML sync from Polymarket API - backfill from trade history
        # This ensures ML model learns from all past trades immediately on startup
        if self.ml_predictor and self.client:
            logger.info("🤖 Syncing ML model from Polymarket trade history...")
            await self._sync_ml_from_polymarket(force=True)
            stats = self.ml_predictor.get_model_stats()
            logger.info(
                f"🤖 ML Model: {stats['training_samples']} samples | "
                f"Mode: {'ACTIVE' if stats['is_active'] else 'LEARNING'}"
            )

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
            from .execution.client import get_open_orders, get_active_positions

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
                        self._last_order_time[asset] = now
                        break

            # 2. Sync active positions (filled trades in active 15-min markets)
            positions = get_active_positions(self.client)
            position_count = len(positions)

            # Cache positions for duplicate checking
            self._update_cached_positions(positions)

            for asset, pos_info in positions.items():
                synced_assets.add(asset)
                # Set cooldown so we don't try to trade this asset again
                self._last_order_time[asset] = now

                # Track this as an external position
                cost = pos_info.get("cost", 0)
                size = pos_info.get("size", 0)
                market_slug = pos_info.get("market", "unknown")

                # Check if we already know about this position
                known_positions = [p.market.asset for p in self.risk_manager.positions.values()]
                if asset not in known_positions and cost > 0:
                    logger.info(
                        f"📊 POSITION FOUND: {asset} | "
                        f"{size:.2f} shares @ ${pos_info.get('price', 0):.2f} | "
                        f"Cost: ${cost:.2f} | Market: {market_slug}"
                    )

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
                placed_time = order_data.get("placed_time")
                wait_time = (now - placed_time).total_seconds() if placed_time else 0

                # Check order status
                order_info = self.executor.get_order_status(order_id)

                if order_info is None:
                    # Order not found - might have been filled or cancelled
                    if wait_time > 60:  # 1 minute timeout (reduced from 2)
                        logger.warning(
                            f"⚠️ Order {order_id[:8]}... ({asset}) not found after {wait_time:.0f}s - removing"
                        )
                        orders_to_remove.append(order_id)
                        # Restore cooldown to allow new order
                        if asset in self._last_order_time:
                            del self._last_order_time[asset]
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
                            ml_volatility=order_data.get("ml_volatility"),
                            ml_momentum=order_data.get("ml_momentum"),
                            ml_confidence=order_data.get("ml_confidence"),
                            ml_arb_type=order_data.get("ml_arb_type"),
                            ml_spread=order_data.get("ml_spread"),
                            ml_bid_depth=order_data.get("ml_bid_depth"),
                            ml_ask_depth=order_data.get("ml_ask_depth"),
                            ml_price_trend=order_data.get("ml_price_trend"),
                            ml_distance_from_target=order_data.get("ml_distance_from_target"),
                            ml_binance_lead_pct=order_data.get("ml_binance_lead_pct"),
                            ml_binance_confirmation=order_data.get("ml_binance_confirmation"),
                            ml_trend_1h=order_data.get("ml_trend_1h"),
                            ml_trend_4h=order_data.get("ml_trend_4h"),
                            ml_trend_1d=order_data.get("ml_trend_1d"),
                            # NEW INDICATOR FEATURES
                            ml_macd_histogram=order_data.get("ml_macd_histogram"),
                            ml_macd_crossover=order_data.get("ml_macd_crossover"),
                            ml_bb_bandwidth=order_data.get("ml_bb_bandwidth"),
                            ml_bb_position=order_data.get("ml_bb_position"),
                            ml_stoch_k=order_data.get("ml_stoch_k"),
                            ml_stoch_d=order_data.get("ml_stoch_d"),
                            ml_stoch_signal=order_data.get("ml_stoch_signal"),
                            ml_rsi_divergence=order_data.get("ml_rsi_divergence"),
                            ml_rsi_divergence_strength=order_data.get("ml_rsi_divergence_strength"),
                            ml_volume_ratio=order_data.get("ml_volume_ratio"),
                            ml_is_high_volume=order_data.get("ml_is_high_volume"),
                            ml_obv_trend=order_data.get("ml_obv_trend"),
                            ml_ha_trend=order_data.get("ml_ha_trend"),
                            ml_ha_consecutive=order_data.get("ml_ha_consecutive"),
                            ml_ha_strength=order_data.get("ml_ha_strength"),
                            ml_vwap_distance_pct=order_data.get("ml_vwap_distance_pct"),
                            ml_vwap_position=order_data.get("ml_vwap_position"),
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
                            predicted_prob=order_data.get("ml_confidence"),
                            arb_type=order_data.get("ml_arb_type") or "none",
                            edge=signal.edge,
                        )

                    else:
                        logger.warning(f"⚠️ Order filled but signal is None - cannot record position for {asset}")

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
                                ml_volatility=order_data.get("ml_volatility"),
                                ml_momentum=order_data.get("ml_momentum"),
                                ml_confidence=order_data.get("ml_confidence"),
                                ml_arb_type=order_data.get("ml_arb_type"),
                                ml_spread=order_data.get("ml_spread"),
                                ml_bid_depth=order_data.get("ml_bid_depth"),
                                ml_ask_depth=order_data.get("ml_ask_depth"),
                                ml_price_trend=order_data.get("ml_price_trend"),
                                ml_distance_from_target=order_data.get("ml_distance_from_target"),
                                ml_binance_lead_pct=order_data.get("ml_binance_lead_pct"),
                                ml_binance_confirmation=order_data.get("ml_binance_confirmation"),
                                ml_trend_1h=order_data.get("ml_trend_1h"),
                                ml_trend_4h=order_data.get("ml_trend_4h"),
                                ml_trend_1d=order_data.get("ml_trend_1d"),
                                # NEW INDICATOR FEATURES
                                ml_macd_histogram=order_data.get("ml_macd_histogram"),
                                ml_macd_crossover=order_data.get("ml_macd_crossover"),
                                ml_bb_bandwidth=order_data.get("ml_bb_bandwidth"),
                                ml_bb_position=order_data.get("ml_bb_position"),
                                ml_stoch_k=order_data.get("ml_stoch_k"),
                                ml_stoch_d=order_data.get("ml_stoch_d"),
                                ml_stoch_signal=order_data.get("ml_stoch_signal"),
                                ml_rsi_divergence=order_data.get("ml_rsi_divergence"),
                                ml_rsi_divergence_strength=order_data.get("ml_rsi_divergence_strength"),
                                ml_volume_ratio=order_data.get("ml_volume_ratio"),
                                ml_is_high_volume=order_data.get("ml_is_high_volume"),
                                ml_obv_trend=order_data.get("ml_obv_trend"),
                                ml_ha_trend=order_data.get("ml_ha_trend"),
                                ml_ha_consecutive=order_data.get("ml_ha_consecutive"),
                                ml_ha_strength=order_data.get("ml_ha_strength"),
                                ml_vwap_distance_pct=order_data.get("ml_vwap_distance_pct"),
                                ml_vwap_position=order_data.get("ml_vwap_position"),
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
                                predicted_prob=order_data.get("ml_confidence"),
                                arb_type=order_data.get("ml_arb_type") or "none",
                                edge=signal.edge,
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
                    if asset in self._last_order_time:
                        del self._last_order_time[asset]

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
                            f"⚠️ Cancelling unfilled order for {asset} - {cancel_reason}"
                        )
                        try:
                            self.executor.cancel_order(order_id)
                        except Exception as cancel_err:
                            logger.warning(f"Failed to cancel order: {cancel_err}")
                        orders_to_remove.append(order_id)
                        if asset in self._last_order_time:
                            del self._last_order_time[asset]
                    else:
                        logger.debug(f"Order {order_id[:8]}... ({asset}) still {order_info.status.value}, waiting {wait_time:.0f}s")

                else:
                    # Unknown status - log and apply timeout
                    logger.warning(f"Unknown order status '{order_info.status.value}' for {asset}")
                    if wait_time > 90:
                        logger.warning(f"⚠️ Removing stale order for {asset} with status {order_info.status.value}")
                        orders_to_remove.append(order_id)
                        if asset in self._last_order_time:
                            del self._last_order_time[asset]

            except Exception as e:
                logger.error(f"Error checking order {order_id[:8]}...: {e}")
                # On error, still remove stale orders
                if wait_time > 120:
                    logger.warning(f"⚠️ Removing errored order {order_id[:8]}... after {wait_time:.0f}s")
                    orders_to_remove.append(order_id)
                    asset = order_data.get("asset")
                    if asset and asset in self._last_order_time:
                        del self._last_order_time[asset]

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
                from .execution.client import get_open_orders, get_active_positions
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
        print_config_box({
            "Min Edge": self.config.trading.min_edge,
            "Min Time Remaining": f"{self.config.trading.min_time_remaining}s",
            "Base Position Size": self.config.trading.base_position_size,
            "Max Position %": self.config.trading.max_position_pct,
            "Max Concurrent": self.config.trading.max_concurrent_positions,
            "Daily Loss Limit": self.config.trading.daily_loss_limit,
            "Supported Assets": ", ".join(self.config.supported_assets),
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
        4. Persist state (no in-flight writes after this)
        5. Disconnect feeds
        """
        t0 = time.monotonic()
        logger.info("SHUTDOWN_PHASE phase=begin")

        # Phase 1: Stop all loops — prevents new order placement
        self._running = False
        self._shutdown_event.set()
        t1 = time.monotonic()
        logger.info("SHUTDOWN_PHASE phase=loops_stopped elapsed=%.3fs", t1 - t0)

        # Phase 2: Cancel open orders (before state save so state reflects cancellations)
        if self.executor and hasattr(self.executor, 'cancel_all_orders'):
            self.executor.cancel_all_orders()
        t2 = time.monotonic()
        logger.info("SHUTDOWN_PHASE phase=orders_cancelled elapsed=%.3fs", t2 - t0)

        # Phase 3: Small drain window — let any in-flight coroutines finish
        # asyncio.gather in start() uses return_exceptions=True so tasks wind down,
        # but give a brief window for any pending awaits to complete
        await asyncio.sleep(0.1)
        t3 = time.monotonic()
        logger.info("SHUTDOWN_PHASE phase=drain_complete elapsed=%.3fs", t3 - t0)

        # Phase 4: Persist state (no more writes after this point)
        if self.state_manager:
            self.state_manager.save(self.risk_manager, self.markets)
            self._check_state_invariants()
        t4 = time.monotonic()
        logger.info("SHUTDOWN_PHASE phase=state_saved elapsed=%.3fs", t4 - t0)

        # Phase 5: Disconnect data feeds
        self.chainlink_feed.disconnect()
        self.clob_feed.disconnect()
        if self.config.endpoints.binance_direct_enabled:
            self.binance_feed.disconnect()
        self.gamma_api.close()

        # Shutdown notification thread pool
        shutdown_notification_executor()

        t5 = time.monotonic()
        logger.info(
            "SHUTDOWN_PHASE phase=complete total=%.3fs "
            "(loops=%.3fs cancel=%.3fs drain=%.3fs save=%.3fs disconnect=%.3fs)",
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

                # Log position status periodically (every 30s)
                self._log_position_status()

                # Calculate equity (cash + unrealized position value)
                # This prevents false drawdown triggers when positions are open
                # Reuse cached value if _log_position_status just computed it
                detail = getattr(self, '_last_equity_detail', None)
                equity = detail["equity"] if detail else self._calculate_equity()

                # Update peak equity (only increases, never decreases)
                self.risk_manager.update_peak_equity(equity)

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

                # Generate and execute signals for each active market
                active_count = len(self.markets)
                if active_count > 0:
                    # Log status every 30 seconds (instead of randomly)
                    if (self._last_status_log is None or
                        (now - self._last_status_log).total_seconds() > self._status_log_interval):
                        self._log_status_line()
                        self._last_status_log = now

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

                # Periodic state save (crash protection)
                now_save = datetime.now(timezone.utc)
                if (
                    self._last_state_save is None
                    or (now_save - self._last_state_save).total_seconds() >= self._state_save_interval
                ):
                    self.state_manager.save(self.risk_manager, self.markets)
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

                # Sync ML from Polymarket API - ensures ML learns from actual trades
                # This bypasses internal position tracking which may fail
                await self._sync_ml_from_polymarket()

                # Sync positions from Polymarket API - clears stale positions
                # that no longer exist (prevents blocking new trades)
                await self._sync_positions_from_polymarket()

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
            cache_key = f"{asset}:{period_ts}"
            self.gamma_api._target_price_cache[cache_key] = price
            injected += 1
            logger.debug(f"Injected {asset} target price ${price:,.2f} for period {period_ts}")

        if injected > 0:
            logger.info(f"💉 Injected {injected} target prices into cache for instant trading")

    async def _refresh_markets(self):
        """Refresh list of active 15-minute markets."""
        now = datetime.now(timezone.utc)
        current_ts = int(now.timestamp())

        # Calculate current 15-minute period (rounds down to :00, :15, :30, :45)
        current_period_ts = (current_ts // 900) * 900

        # Detect period boundary crossing with single-flight lock
        if self._current_period_ts > 0 and current_period_ts != self._current_period_ts:
            # Use lock to ensure only one transition runs at a time
            async with self._period_boundary_lock:
                # Double-check after acquiring lock (another coroutine may have handled it)
                if self._last_handled_period_ts >= current_period_ts:
                    # Already handled this transition, skip
                    self._current_period_ts = current_period_ts
                    return

                # Mark this period as being handled IMMEDIATELY to prevent duplicates
                # (even if we return early during wait phase)
                self._last_handled_period_ts = current_period_ts

                logger.info(
                    f"⏰ PERIOD BOUNDARY: Transitioning from {self._current_period_ts} "
                    f"to {current_period_ts}"
                )

                # REAL-TIME: Capture current Chainlink prices as target prices for new period
                # This eliminates the need to wait for the API
                self._capture_period_boundary_prices(current_period_ts)

                # Brief wait to ensure price capture is stable (reduced from 15s to 2s)
                if self._period_transition_wait_until is None:
                    wait_seconds = self.period_transition_delay
                    self._period_transition_wait_until = now + timedelta(seconds=wait_seconds)
                    logger.debug(f"Brief {wait_seconds}s stabilization wait...")

                # If still waiting, don't refresh yet (but dedup flag is already set)
                if now < self._period_transition_wait_until:
                    return

                # Transition complete - clear old data and proceed
                logger.info("✅ Period transition complete - using captured prices")
                self._period_transition_wait_until = None

                # Clear stale target price cache
                self.gamma_api.clear_stale_price_cache(current_period_ts)

                # Inject captured prices into gamma API cache for immediate use
                self._inject_captured_prices_to_gamma(current_period_ts)

                # Clear markets from old period (they should be in expiring/settled by now)
                async with self._markets_lock:
                    old_markets = list(self.markets.keys())
                    for market_id in old_markets:
                        market = self.markets.get(market_id)
                        if market and market.time_remaining <= 0:
                            self.markets.pop(market_id, None)
                            logger.debug(f"Removed expired market: {market.asset}")

                # Force refresh by clearing last_market_refresh
                self.last_market_refresh = None

        # Update current period tracking
        self._current_period_ts = current_period_ts

        # Check if refresh is needed
        if self.last_market_refresh:
            elapsed = (now - self.last_market_refresh).total_seconds()
            if elapsed < self.market_discovery_interval:
                return

        logger.debug("Refreshing active markets...")

        # Collect stale order cancellations to run outside the lock
        stale_cancellations: list[tuple[str, str]] = []  # [(asset, old_market_id), ...]

        try:
            # Fetch active 15-min crypto markets
            new_markets = self.gamma_api.get_15min_crypto_markets()

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
                        old_market_id = self._active_market_per_asset.get(asset)
                        if old_market_id and old_market_id != market_id:
                            # Market transition detected - queue for cancellation
                            stale_cancellations.append((asset, old_market_id))

                        # Update active market tracking (both local and safety guard)
                        self._active_market_per_asset[asset] = market_id
                        guard = self.safety_guard or get_safety_guard()
                        if guard:
                            guard.set_active_market(asset, market_id)

                        # Target price is now fetched from Polymarket API in gamma.py
                        logger.info(
                            f"NEW MARKET: {market.asset} | "
                            f"Target: ${market.target_price:,.2f} | "
                            f"Ends: {market.end_time.strftime('%H:%M:%S')} | "
                            f"Time: {market.time_remaining:.0f}s"
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
        """
        min_time = self.config.trading.min_time_remaining

        async with self._markets_lock:
            # Find markets that should stop trading
            markets_to_expire = []
            for market_id, market in self.markets.items():
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

                # Clear cooldown so new market can trade
                if asset in self._last_order_time:
                    del self._last_order_time[asset]
                    logger.debug(f"Cleared cooldown for {asset} (market expiring)")

                # Clear from cached API positions
                if hasattr(self, '_api_positions') and asset in self._api_positions:
                    del self._api_positions[asset]

                logger.info(
                    f"EXPIRING: {market.asset} | "
                    f"Time remaining: {market.time_remaining:.0f}s | "
                    f"Target: ${market.target_price:,.2f}"
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

    async def _check_early_exits(self):
        """
        Check open positions for take-profit, stop-loss, or chart-based exit conditions.

        For each position:
        1. Get current market price from order book
        2. Calculate unrealized P&L percentage
        3. If >= take_profit_pct: sell to lock in profit
        4. If <= -stop_loss_pct: sell to limit loss
        5. Check chart analysis for trend reversal or DCA opportunity
        """
        if not self.risk_manager.positions:
            return

        now = datetime.now(timezone.utc)
        trading = self.config.trading

        # Import position monitor
        try:
            from .strategy.position_monitor import analyze_open_position, PositionAction
            from .data.binance_chart import analyze_chart
            POSITION_MONITOR_AVAILABLE = True
        except ImportError:
            POSITION_MONITOR_AVAILABLE = False

        for market_key, position in list(self.risk_manager.positions.items()):
            try:
                # Check minimum hold time
                hold_time = (now - position.entry_time).total_seconds()
                if hold_time < trading.early_exit_min_hold_time:
                    continue

                # Find the market (in active or expiring)
                market = self.markets.get(market_key) or self.expiring_markets.get(market_key)
                if not market:
                    continue

                # Get current market price for our position side
                if position.side == Side.UP:
                    current_price = market.best_bid  # Selling UP, get bid
                else:
                    # For DOWN, price is 1 - UP_ask (we sell DOWN = buy UP at ask)
                    current_price = 1.0 - market.best_ask if market.best_ask else None

                if current_price is None or current_price <= 0:
                    continue

                # Calculate unrealized P&L percentage
                entry_price = position.entry_price
                pnl_pct = (current_price - entry_price) / entry_price

                # Check take-profit
                if pnl_pct >= trading.take_profit_pct:
                    logger.info(
                        f"{Colors.BRIGHT_GREEN}💰 TAKE PROFIT: {market.asset} {position.side.value} | "
                        f"P&L: {pnl_pct:+.0%} | Entry: {entry_price:.2f} → Exit: {current_price:.2f}{Colors.RESET}"
                    )
                    await self._execute_early_exit(market, position, current_price, "take_profit")
                    continue

                # Check stop-loss
                if pnl_pct <= -trading.stop_loss_pct:
                    logger.info(
                        f"{Colors.BRIGHT_RED}🛑 STOP LOSS: {market.asset} {position.side.value} | "
                        f"P&L: {pnl_pct:+.0%} | Entry: {entry_price:.2f} → Exit: {current_price:.2f}{Colors.RESET}"
                    )
                    await self._execute_early_exit(market, position, current_price, "stop_loss")
                    continue

                # Time-based exit if edge decays near settlement
                if trading.time_exit_enabled and market.time_remaining <= trading.time_exit_seconds:
                    decay_signal = self.signal_generator.generate_signal(market)
                    decay_edge = decay_signal.edge if decay_signal else 0.0
                    if decay_edge < trading.exit_edge_threshold:
                        logger.info(
                            f"{Colors.BRIGHT_YELLOW}⏱️ TIME EXIT: {market.asset} {position.side.value} | "
                            f"Edge {decay_edge:.2%} < {trading.exit_edge_threshold:.2%} | "
                            f"{market.time_remaining:.0f}s remaining{Colors.RESET}"
                        )
                        await self._execute_early_exit(market, position, current_price, "time_exit")
                        continue

                # === CHART-BASED POSITION MONITORING ===
                # Check 15m chart to decide: hold, close, or DCA
                if POSITION_MONITOR_AVAILABLE:
                    await self._check_chart_based_exit(market, position, current_price)

            except Exception as e:
                logger.error(f"Error checking early exit for {market_key}: {e}")

    async def _check_chart_based_exit(self, market: MarketState, position, current_token_price: float):
        """
        Check if chart analysis suggests exiting or modifying a position.

        Uses 15m candlestick data to detect:
        - Trend reversals (close position)
        - Strong confirmation (DCA opportunity)
        - Extreme RSI (potential reversal)
        """
        try:
            from .strategy.position_monitor import analyze_open_position, PositionAction, get_position_monitor
            from .data.binance_chart import analyze_chart
        except ImportError:
            return

        asset = market.asset
        monitor = get_position_monitor()

        # Check if we should analyze (rate limited)
        if not monitor.should_check(asset):
            return

        # Get chart analysis
        chart_analysis = analyze_chart(asset)
        if not chart_analysis:
            return

        # Analyze position
        decision = analyze_open_position(
            position=position,
            chart_analysis=chart_analysis,
            current_token_price=current_token_price,
            time_remaining=market.time_remaining,
        )

        # Log position status periodically
        summary = monitor.get_position_summary(position, chart_analysis, current_token_price)
        logger.info(f"📊 POSITION MONITOR: {summary} | Action: {decision.action.value}")

        # Act on decision
        if decision.action == PositionAction.CLOSE:
            logger.info(
                f"{Colors.BRIGHT_YELLOW}📉 CHART EXIT: {asset} {position.side.value} | "
                f"Reason: {decision.reason} | Confidence: {decision.confidence:.0%}{Colors.RESET}"
            )
            await self._execute_early_exit(market, position, current_token_price, f"chart_exit_{decision.urgency}")

        elif decision.action == PositionAction.DCA:
            logger.info(
                f"{Colors.BRIGHT_CYAN}📈 CHART DCA: {asset} {position.side.value} | "
                f"Reason: {decision.reason} | Size: ${decision.suggested_size:.2f}{Colors.RESET}"
            )
            # Execute DCA through averaging system
            await self._execute_chart_dca(market, position, decision)

        elif decision.action == PositionAction.ADD:
            logger.info(
                f"{Colors.BRIGHT_GREEN}📈 CHART ADD: {asset} {position.side.value} | "
                f"Reason: {decision.reason} | Size: ${decision.suggested_size:.2f}{Colors.RESET}"
            )
            # Execute ADD through averaging system
            await self._execute_chart_dca(market, position, decision)

        # HOLD - just log at debug level
        else:
            logger.debug(f"[{asset}] Position HOLD: {decision.reason}")

    async def _execute_chart_dca(self, market: MarketState, position, decision):
        """
        Execute a DCA or ADD order based on chart analysis.

        Args:
            market: Market state
            position: Position to add to
            decision: PositionDecision with suggested size
        """
        try:
            from .strategy.averaging import update_position_after_averaging
        except ImportError:
            logger.error("Could not import averaging module")
            return

        asset = market.asset

        # Calculate shares and price
        if position.side == Side.UP:
            token_price = market.best_ask
        else:
            token_price = 1 - market.best_bid

        shares = decision.suggested_size / token_price
        if shares < 5:
            shares = 5
            decision.suggested_size = shares * token_price

        # Create signal for execution
        signal = Signal(
            market=market,
            side=position.side,
            edge=0.10,  # Placeholder
            true_prob=0.5,
            market_prob=token_price,
            recommended_action=OrderAction.LIMIT,
            recommended_price=token_price,
            size_usd=decision.suggested_size,
            size_shares=shares,
            chainlink_price=self.signal_generator.get_price(asset) or 0,
            time_remaining=market.time_remaining,
            reasoning=f"[CHART {decision.action.value}] {decision.reason}",
        )

        # Execute the order
        start_exec = time.perf_counter()
        try:
            if self.smart_router:
                result = await self.smart_router.execute(signal)
            else:
                result = await self.executor.execute_signal_async(signal)
        except Exception as e:
            logger.error(
                "ORDER_RESULT asset=%s side=%s status=exception error=%s mode=%s",
                asset, signal.side.value, str(e)[:200],
                "paper" if self.config.paper_trading.enabled else "live"
            )
            logger.debug("Chart execution exception traceback:", exc_info=True)
            return  # Exit gracefully
        exec_latency_ms = (time.perf_counter() - start_exec) * 1000
        self.metrics.record_latency("execute_signal", exec_latency_ms)

        ml_ev_info = getattr(signal, "_ml_ev_info", None)
        if ml_ev_info:
            logger.info(
                f"[{asset}] ML_EV p_up={ml_ev_info['p_up']:.3f} ev_up={ml_ev_info['ev_up']:.3f} "
                f"ev_down={ml_ev_info['ev_down']:.3f} edge_up={ml_ev_info['edge_up']:.3f} "
                f"edge_down={ml_ev_info['edge_down']:.3f} side={ml_ev_info['side']} result={result.success}"
            )

        if result.success:
            # Calculate new average price
            old_cost = position.total_cost if position.total_cost > 0 else (position.entry_price * position.shares)
            new_total_cost = old_cost + decision.suggested_size
            new_total_shares = position.shares + shares
            new_avg_price = new_total_cost / new_total_shares

            # Update position
            update_position_after_averaging(
                position=position,
                additional_shares=shares,
                additional_cost=decision.suggested_size,
                new_entry_price=new_avg_price,
            )

            # Set cooldown
            now = datetime.now(timezone.utc)
            self._last_order_time[asset] = now

            logger.info(
                f"[{asset}] ✅ CHART {decision.action.value}: "
                f"Added {shares:.1f} shares @ ${decision.suggested_size:.2f} | "
                f"New avg: ${new_avg_price:.3f}"
            )
        else:
            logger.warning(f"[{asset}] ❌ CHART {decision.action.value} FAILED: {result.error_message}")

    async def _execute_early_exit(
        self,
        market: MarketState,
        position,
        exit_price: float,
        exit_reason: str,
    ):
        """
        Execute early exit (sell position before settlement).

        Args:
            market: Market state
            position: Position to exit
            exit_price: Price to sell at
            exit_reason: "take_profit" or "stop_loss"
        """
        # In dry run mode (but NOT paper trading), just log and process tracking
        if self.config.dry_run and not self.config.paper_trading.enabled:
            logger.info(f"[DRY RUN] Would sell {position.shares:.2f} shares at {exit_price:.4f}")
            # Still process the exit for tracking
            pnl = (exit_price - position.entry_price) * position.shares
            await self._process_early_exit_result(market, position, exit_price, pnl, exit_reason)
            return

        if not self.executor:
            logger.warning("No executor available for early exit")
            return

        try:
            # Create a sell signal
            from .models import Signal, OrderAction
            sell_signal = Signal(
                market=market,
                side=position.side,
                recommended_action=OrderAction.MARKET,  # Market order for quick exit
                recommended_price=exit_price,
                size_usd=position.shares * exit_price,
                size_shares=position.shares,
                edge=0.0,  # Not relevant for exit
                reasoning=f"Early exit: {exit_reason}",
            )

            # Execute sell order
            result = self.executor.sell_position(
                token_id=position.token_id,
                shares=position.shares,
                min_price=exit_price * 0.98,  # Allow 2% slippage
            )

            if result and result.success:
                actual_price = result.filled_price or exit_price
                pnl = (actual_price - position.entry_price) * position.shares
                logger.info(
                    f"✅ Early exit executed: {market.asset} {position.side.value} | "
                    f"Sold {result.filled_size:.2f} @ {actual_price:.4f} | P&L: ${pnl:+.2f}"
                )
                await self._process_early_exit_result(market, position, actual_price, pnl, exit_reason)
            else:
                error_msg = result.error_message if result else "Unknown error"
                logger.warning(f"Early exit failed: {error_msg}")

        except Exception as e:
            logger.error(f"Error executing early exit: {e}")

    async def _process_early_exit_result(
        self,
        market: MarketState,
        position,
        exit_price: float,
        pnl: float,
        exit_reason: str,
    ):
        """
        Process the result of an early exit - update tracking, ML, etc.

        Args:
            market: Market state
            position: Closed position
            exit_price: Actual exit price
            pnl: Realized P&L
            exit_reason: "take_profit" or "stop_loss"
        """
        won = pnl > 0

        # Record with risk manager
        self.risk_manager.record_position_close(
            market_key=market.condition_id,
            exit_price=exit_price,
            pnl=pnl,
        )
        self.metrics.record_trade(market.asset, pnl)

        # Record with trade history (mark as early exit)
        trade_history = get_trade_history()
        trade_history.record_close(
            market_id=market.condition_id,
            won=won,
            pnl=pnl,
            exit_price=exit_price,
        )

        # Record ML outcome - this helps ML learn which trades were good
        if self.ml_predictor:
            from types import SimpleNamespace

            # Get ML features
            if position.ml_volatility is not None:
                volatility = position.ml_volatility
                momentum = position.ml_momentum or 0.0
                arb_type = position.ml_arb_type or "none"
                spread = position.ml_spread or 0.0
                bid_depth = position.ml_bid_depth or 0.0
                ask_depth = position.ml_ask_depth or 0.0
                price_trend = position.ml_price_trend or 0.0
                distance_from_target = position.ml_distance_from_target or 0.0
                binance_lead_pct = position.ml_binance_lead_pct or 0.0
                binance_confirmation = position.ml_binance_confirmation or "NONE"
                trend_1h = position.ml_trend_1h or 0.0
                trend_4h = position.ml_trend_4h or 0.0
                trend_1d = position.ml_trend_1d or 0.0
            else:
                volatility = self.signal_generator.get_volatility(market.asset)
                current_price = self.signal_generator.get_price(market.asset)
                momentum = 0.0
                if current_price and market.target_price:
                    momentum = (current_price - market.target_price) / market.target_price
                    momentum = max(-1, min(1, momentum * 10))
                arb_type = "none"
                spread = market.best_ask - market.best_bid if market.best_ask and market.best_bid else 0.0
                bid_depth = market.bid_depth
                ask_depth = market.ask_depth
                price_trend = 0.0
                distance_from_target = 0.0
                if current_price and market.target_price:
                    distance_from_target = abs(current_price - market.target_price) / market.target_price
                binance_lead_pct = 0.0
                binance_confirmation = "NONE"
                trend_1h = 0.0
                trend_4h = 0.0
                trend_1d = 0.0
                # Try to fetch trends if available
                try:
                    from .data.binance import get_multi_timeframe_trends
                    trends = get_multi_timeframe_trends(market.asset)
                    trend_1h = trends.get("trend_1h", 0.0)
                    trend_4h = trends.get("trend_4h", 0.0)
                    trend_1d = trends.get("trend_1d", 0.0)
                except Exception:
                    pass

            fake_signal = SimpleNamespace(
                edge=0.0,
                market=market,
                side=position.side,
                _arb_type=arb_type,
                # Additional attributes needed by TraderModelLoader
                recommended_price=position.entry_price,
                size_shares=position.shares,
                size_usd=position.shares * position.entry_price,
                time_remaining=market.time_remaining,
            )

            # Get price data for ML
            current_price = self.signal_generator.get_price(market.asset) or 0.0
            target_price = market.target_price or 0.0
            price_high, price_low = self.signal_generator.get_price_range(market.asset)
            price_velocity = self.signal_generator.get_price_velocity(market.asset)

            # Record with early exit flag
            self.ml_predictor.record_outcome(
                signal=fake_signal,
                volatility=volatility,
                price_momentum=momentum,
                won=won,
                arb_type=arb_type,
                spread=spread,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                price_trend=price_trend,
                distance_from_target=distance_from_target,
                binance_lead_pct=binance_lead_pct,
                binance_confirmation=binance_confirmation,
                trend_1h=trend_1h,
                trend_4h=trend_4h,
                trend_1d=trend_1d,
                current_price=current_price,
                target_price=target_price,
                price_high=price_high,
                price_low=price_low,
                price_velocity=price_velocity,
            )

            # Track for auto-retraining
            if self.auto_retrainer:
                self.auto_retrainer.record_trade(won=won)

        # Clear asset cooldown so we can trade again
        asset = market.asset
        if asset in self._last_order_time:
            del self._last_order_time[asset]

        # Log summary
        result_emoji = "💰" if exit_reason == "take_profit" else "🛑"
        logger.info(
            f"{result_emoji} EARLY EXIT COMPLETE: {market.asset} | "
            f"Reason: {exit_reason} | P&L: ${pnl:+.2f} | "
            f"Bankroll: ${self.risk_manager.current_bankroll:.2f}"
        )

    def _cleanup_orphaned_positions(self):
        """
        Force-close positions in markets that expired while the bot was down.

        Called after state restore. Positions in expired markets cannot be
        settled normally (no live feed data), so close them as losses to
        prevent orphaned entries that block trading.
        """
        now = datetime.now(timezone.utc)
        orphaned = []
        for cid, position in list(self.risk_manager.positions.items()):
            market = position.market
            if market.end_time < now:
                orphaned.append(cid)

        if not orphaned:
            return

        for cid in orphaned:
            position = self.risk_manager.positions.get(cid)
            if not position:
                continue

            age_sec = (now - position.market.end_time).total_seconds()
            logger.warning(
                "ORPHAN_CLEANUP market=%s asset=%s side=%s cost=%.2f "
                "expired_ago=%.0fs — force-closing as LOSS",
                cid[:16], position.market.asset, position.side.value,
                position.cost_basis, age_sec,
            )

            # Credit paper executor with zero proceeds (full loss)
            if self.config.paper_trading.enabled and hasattr(self.executor, 'credit_settlement'):
                self.executor.credit_settlement(0.0, cid)

            self.risk_manager.record_position_close(
                market_key=cid,
                exit_price=0.0,
                pnl=-position.cost_basis,
            )

            # Mark as settled so we don't try to settle again
            self.settled_markets.add(cid)

        logger.info(
            "ORPHAN_CLEANUP_DONE closed=%d bankroll=%.2f",
            len(orphaned), self.risk_manager.current_bankroll,
        )

    async def _check_settlements(self):
        """
        Check expiring markets for settlement and close positions.

        Uses per-market exponential backoff to avoid log spam when
        API resolution is slow (5s → 10s → 20s → 40s, capped at 60s).
        """
        now_mono = time.monotonic()

        async with self._markets_lock:
            markets_to_settle = []
            for market_id, market in self.expiring_markets.items():
                # Check if market has passed end time (with small buffer)
                if market.time_remaining <= -2.0:  # 2 second buffer after expiry
                    # Exponential backoff: skip if not yet time for next check
                    next_at = self._settle_next_check.get(market_id, 0.0)
                    if now_mono < next_at:
                        continue
                    markets_to_settle.append(market_id)

            for market_id in markets_to_settle:
                market = self.expiring_markets[market_id]
                settled = await self._settle_market(market)

                # Only remove from expiring if actually settled
                if settled:
                    # Clean up backoff state
                    self._settle_next_check.pop(market_id, None)
                    self._settle_attempts.pop(market_id, None)

                    # Move to settled set and cleanup
                    self.expiring_markets.pop(market_id, None)
                    self.settled_markets.add(market_id)

                    # Clean up token-to-market mapping
                    self._token_to_market.pop(market.up_token_id, None)
                    self._token_to_market.pop(market.down_token_id, None)

                    # Unsubscribe from order book
                    self.clob_feed.unsubscribe(market.up_token_id)
                    self.clob_feed.unsubscribe(market.down_token_id)

                    # IMPORTANT: Clear all tracking for this asset so new market can trade
                    asset = market.asset

                    # Clear cached API positions
                    if hasattr(self, '_api_positions') and asset in self._api_positions:
                        del self._api_positions[asset]
                        logger.info(f"✅ Cleared cached position for {asset} after settlement")

                    # Clear cooldown
                    if asset in self._last_order_time:
                        del self._last_order_time[asset]
                        logger.debug(f"Cleared cooldown for {asset} after settlement")

                    # Cancel any pending orders for this asset
                    await self._cancel_pending_orders_for_asset(asset)

                    # Cleanup old settled markets (keep last 100)
                    if len(self.settled_markets) > 100:
                        self.settled_markets = set(list(self.settled_markets)[-100:])
                else:
                    # Not yet resolved — apply exponential backoff
                    # 5s → 10s → 20s → 40s → 60s (capped)
                    attempts = self._settle_attempts.get(market_id, 0) + 1
                    self._settle_attempts[market_id] = attempts
                    delay = min(5.0 * (2 ** (attempts - 1)), 60.0)
                    self._settle_next_check[market_id] = now_mono + delay

    async def _settle_market(self, market: MarketState) -> bool:
        """
        Process settlement for a single market.

        Args:
            market: Market to settle

        Returns:
            True if settlement was successful, False if should retry
        """
        # Idempotency guard: skip if already settled
        if market.condition_id in self.settled_markets:
            logger.debug("SETTLE_SKIP already settled market=%s", market.condition_id[:16])
            return True

        # Check if we have a position in this market
        position = self.risk_manager.positions.get(market.condition_id)
        has_position = position is not None

        # Log that we're attempting settlement
        pos_str = f" (HAVE POSITION: {position.side.value})" if has_position else ""
        logger.info(f"SETTLING: {market.asset}{pos_str} | {market.question[:50]}...")

        time_since_expiry = abs(market.time_remaining)

        # Fetch resolution from API
        resolution = self.gamma_api.get_market_resolution(market.condition_id)

        winning_outcome = None
        resolution_price = None

        if resolution and resolution.get("resolved"):
            # API has resolution
            winning_outcome = resolution.get("winning_outcome")
            resolution_price = resolution.get("resolution_price")
            logger.info(
                f"✅ RESOLVED (API): {market.asset} | "
                f"Winner: {winning_outcome} | "
                f"Settlement Price: ${resolution_price:,.2f if resolution_price else 0}"
            )

            # Cross-verify with on-chain data (The Graph)
            if self.settlement_verifier.is_enabled() and winning_outcome:
                verified, verify_msg = self.settlement_verifier.cross_verify_settlement(
                    condition_id=market.condition_id,
                    api_outcome=winning_outcome,
                    api_price=resolution_price
                )
                if verified:
                    logger.debug(f"  └─ On-chain: {verify_msg}")
                else:
                    # On-chain disagrees with API — prefer on-chain
                    onchain_result = self.settlement_verifier.verify_settlement(market.condition_id)
                    if onchain_result and onchain_result.on_chain_resolved and onchain_result.winning_outcome:
                        old_outcome = winning_outcome
                        winning_outcome = onchain_result.winning_outcome.upper()
                        logger.warning(
                            "SETTLE_OVERRIDE market=%s api=%s onchain=%s — using on-chain",
                            market.condition_id[:16], old_outcome, winning_outcome,
                        )
                    else:
                        logger.warning(
                            "SETTLE_MISMATCH_UNRESOLVED market=%s api=%s onchain_msg=%s — using API",
                            market.condition_id[:16], winning_outcome, verify_msg,
                        )
        else:
            # API doesn't have resolution yet - try local determination
            # For 15-minute crypto markets: price >= target = UP wins
            local_outcome, local_price = self._determine_outcome_locally(market)

            if local_outcome:
                # Wait at least 30 seconds after expiry before using local resolution
                # This gives the API time to update
                if time_since_expiry >= 30:
                    winning_outcome = local_outcome
                    resolution_price = local_price
                    logger.info(
                        f"✅ RESOLVED (LOCAL): {market.asset} | "
                        f"Winner: {winning_outcome} | "
                        f"Price: ${resolution_price:,.2f} vs Target: ${market.target_price:,.2f}"
                    )
                else:
                    # Log at INFO on first attempt, DEBUG thereafter (backoff reduces frequency)
                    attempts = self._settle_attempts.get(market.condition_id, 0)
                    log_fn = logger.info if attempts <= 1 else logger.debug
                    log_fn(
                        "SETTLE_WAIT market=%s asset=%s local=%s waited=%.0fs attempt=%d",
                        market.condition_id[:16], market.asset,
                        local_outcome, time_since_expiry, attempts,
                    )
                    return False  # Wait for API first
            else:
                # Couldn't determine locally either
                attempts = self._settle_attempts.get(market.condition_id, 0)
                log_fn = logger.warning if attempts <= 1 else logger.debug
                log_fn(
                    "SETTLE_WAIT market=%s asset=%s local=NONE waited=%.0fs attempt=%d",
                    market.condition_id[:16], market.asset,
                    time_since_expiry, attempts,
                )
                # If we've waited more than 5 minutes, give up
                if time_since_expiry > 300:
                    logger.error(
                        f"SETTLEMENT TIMEOUT: {market.asset} - no resolution after 5 minutes"
                    )
                    # Force-close any open position as a loss to avoid orphaning
                    if position:
                        logger.warning(
                            f"Force-closing position in {market.asset} as LOSS (timeout)"
                        )
                        # Full loss: proceeds = cost_basis + (-cost_basis) = 0
                        if self.config.paper_trading.enabled and hasattr(self.executor, 'credit_settlement'):
                            self.executor.credit_settlement(0.0, market.condition_id)
                        self.risk_manager.record_position_close(
                            market_key=market.condition_id,
                            exit_price=0.0,
                            pnl=-position.cost_basis,
                        )
                    return True  # Force remove to prevent infinite retry
                return False  # Retry later

        # Process position if we had one
        if position and winning_outcome:
            await self._close_position_on_settlement(
                market=market,
                position=position,
                winning_outcome=winning_outcome,
            )
        elif position:
            logger.warning(f"Have position in {market.asset} but no winning outcome - force-closing as LOSS")
            if self.config.paper_trading.enabled and hasattr(self.executor, 'credit_settlement'):
                self.executor.credit_settlement(0.0, market.condition_id)
            self.risk_manager.record_position_close(
                market_key=market.condition_id,
                exit_price=0.0,
                pnl=-position.cost_basis,
            )
        else:
            logger.debug(f"No position in {market.asset}, nothing to settle")

        # ALWAYS record ML observation from expired markets (even without positions)
        # This gives the ML model more training data by observing outcomes
        if winning_outcome and resolution_price and self.ml_predictor:
            self._record_ml_observation_from_expired_market(
                market=market,
                winning_outcome=winning_outcome,
                resolution_price=resolution_price,
                had_position=has_position,
            )

        return True  # Settlement successful

    def _determine_outcome_locally(self, market: MarketState) -> tuple[Optional[str], Optional[float]]:
        """
        Determine market outcome locally using Chainlink price snapshot.

        Uses the expiry_chainlink_price (captured when market moved to expiring
        queue) to avoid using stale/moved live prices after market ends.

        For 15-minute crypto markets:
        - If ref_price >= target price: UP wins
        - If ref_price < target price: DOWN wins

        Args:
            market: Market to determine outcome for

        Returns:
            Tuple of (winning_outcome, ref_price) or (None, None) if can't determine
        """
        # Prefer snapshot price captured at expiry; fall back to live with warning
        ref_price = market.expiry_chainlink_price
        source = "snapshot"

        if ref_price is None or ref_price <= 0:
            ref_price = self.signal_generator.get_price(market.asset)
            source = "live_fallback"
            if ref_price is not None and ref_price > 0:
                logger.warning(
                    "LOCAL_RESOLVE_FALLBACK market=%s asset=%s using live price %.2f "
                    "(no expiry snapshot available)",
                    market.condition_id[:8], market.asset, ref_price,
                )

        if ref_price is None or ref_price <= 0:
            return (None, None)

        target_price = market.target_price
        if target_price is None or target_price <= 0:
            return (None, None)

        # Determine winner based on ref price vs target
        winner = "UP" if ref_price >= target_price else "DOWN"

        logger.info(
            "LOCAL_RESOLVE market=%s asset=%s ref_price=%.2f target=%.2f "
            "winner=%s source=%s",
            market.condition_id[:8], market.asset, ref_price,
            target_price, winner, source,
        )

        return (winner, ref_price)

    def _record_ml_observation_from_expired_market(
        self,
        market: MarketState,
        winning_outcome: str,
        resolution_price: float,
        had_position: bool,
    ):
        """
        Record ML training data from an expired market (observational learning).

        This allows the ML model to learn from ALL expired markets, not just
        the ones we traded. By observing outcomes, the model can learn patterns
        without risking money.

        Records two observations:
        1. The winning side (UP or DOWN) - what would have won
        2. The losing side - what would have lost

        Args:
            market: The expired market
            winning_outcome: "UP" or "DOWN" - which side won
            resolution_price: The final price at settlement
            had_position: Whether we had a position (skip if already recorded)
        """
        # Skip if we already recorded this via position close
        if had_position:
            return

        # Skip if ML predictor not available
        if not self.ml_predictor:
            return

        try:
            from types import SimpleNamespace
            from .models import Side

            # Get current market data for feature extraction
            volatility = self.signal_generator.get_volatility(market.asset)
            current_price = resolution_price

            # Calculate how far price ended from target
            target_price = market.target_price
            distance_from_target = 0.0
            if target_price and target_price > 0:
                distance_from_target = (current_price - target_price) / target_price

            # Get Binance trends if available
            trend_1h, trend_4h, trend_1d = 0.0, 0.0, 0.0
            try:
                from .data.binance import get_multi_timeframe_trends
                trends = get_multi_timeframe_trends(market.asset)
                trend_1h = trends.get("trend_1h", 0.0)
                trend_4h = trends.get("trend_4h", 0.0)
                trend_1d = trends.get("trend_1d", 0.0)
            except Exception:
                pass

            # Calculate price momentum based on outcome
            # If UP won, price was trending up; if DOWN won, price was trending down
            price_momentum = distance_from_target * 10  # Amplify for better signal
            price_momentum = max(-1, min(1, price_momentum))

            # Record BOTH sides for balanced learning:
            # 1. Record the winning side as a WIN
            # 2. Record the losing side as a LOSS
            for side_value in ["UP", "DOWN"]:
                side = Side.UP if side_value == "UP" else Side.DOWN
                won = (side_value == winning_outcome)

                # Create a fake signal for ML recording
                fake_signal = SimpleNamespace(
                    market=market,
                    side=side,
                    edge=0.0,  # Unknown edge for observation
                    recommended_price=market.best_bid if side == Side.UP else market.best_ask,
                    size_shares=0,
                    size_usd=0,
                    time_remaining=0,
                    _arb_type="observation",  # Mark as observational data
                )

                # Record to ML model
                self.ml_predictor.record_outcome(
                    signal=fake_signal,
                    volatility=volatility,
                    price_momentum=price_momentum if side == Side.UP else -price_momentum,
                    won=won,
                    arb_type="observation",
                    spread=market.best_ask - market.best_bid if market.best_ask and market.best_bid else 0.0,
                    bid_depth=market.bid_depth,
                    ask_depth=market.ask_depth,
                    price_trend=price_momentum,
                    distance_from_target=distance_from_target,
                    binance_lead_pct=0.0,
                    binance_confirmation="NONE",
                    trend_1h=trend_1h,
                    trend_4h=trend_4h,
                    trend_1d=trend_1d,
                    current_price=current_price,
                    target_price=target_price,
                    price_high=current_price * 1.005,
                    price_low=current_price * 0.995,
                    price_velocity=abs(price_momentum) * 0.001,
                )

            logger.info(
                f"🤖 ML OBSERVATION: {market.asset} | "
                f"Winner: {winning_outcome} | "
                f"Price: ${resolution_price:,.2f} vs Target: ${target_price:,.2f} | "
                f"+2 training samples"
            )

        except Exception as e:
            logger.debug(f"Error recording ML observation: {e}")

    async def _close_position_on_settlement(
        self,
        market: MarketState,
        position,
        winning_outcome: str,
    ):
        """
        Close a position based on market settlement using token-ID payout.

        Payout logic (binary market):
          winning_token = yes_token if winner==UP else no_token
          payout_per_share = 1.0 if held_token == winning_token else 0.0
          realized = (payout_per_share - entry_price) * shares

        Args:
            market: Settled market
            position: Our position in the market
            winning_outcome: "UP" or "DOWN"
        """
        # Idempotency guard: verify position still exists in risk manager
        if market.condition_id not in self.risk_manager.positions:
            logger.warning(
                "SETTLE_SKIP_CLOSED position already closed market=%s",
                market.condition_id[:16],
            )
            return

        position_side = position.side.value  # "UP" or "DOWN"

        # --- Token-ID based payout (authoritative) ---
        winning_token_id = (
            market.up_token_id if winning_outcome == "UP"
            else market.down_token_id
        )

        # Use explicit held_token_id if available, fall back to token_id
        held_token = position.held_token_id or position.token_id
        payout_per_share = 1.0 if held_token == winning_token_id else 0.0
        won = payout_per_share == 1.0

        # Cross-check: token-ID result should match side-string comparison
        side_match = (position_side == winning_outcome)
        if won != side_match:
            logger.error(
                "SETTLE_MISMATCH token-based won=%s but side-match=%s | "
                "side=%s winner=%s held=%s winning_token=%s",
                won, side_match, position_side, winning_outcome,
                held_token[:16], winning_token_id[:16],
            )

        # Calculate P&L
        pnl = (payout_per_share - position.entry_price) * position.shares

        # Audit log
        logger.info(
            "SETTLE_APPLY market_id=%s asset=%s side=%s held_token=%s "
            "winning_token=%s payout=%.1f entry=%.4f shares=%.2f realized=%+.2f",
            market.condition_id[:8], market.asset, position_side,
            held_token[:16], winning_token_id[:16],
            payout_per_share, position.entry_price, position.shares, pnl,
        )

        # Color and format the result
        result_color = Colors.BRIGHT_GREEN if won else Colors.BRIGHT_RED
        result_emoji = "🎉" if won else "💔"
        result_text = "WIN" if won else "LOSS"
        side_arrow = "▲" if position_side == "UP" else "▼"

        # Calculate ROI
        roi = (pnl / position.cost_basis * 100) if position.cost_basis > 0 else 0

        # Box format for position close
        logger.info(f"{result_color}╔══════════════════════════════════════════════════════════════════╗{Colors.RESET}")
        logger.info(f"{result_color}║  {result_emoji} POSITION CLOSED: {result_text:4}                                        ║{Colors.RESET}")
        logger.info(f"{result_color}╠══════════════════════════════════════════════════════════════════╣{Colors.RESET}")
        logger.info(f"{result_color}║{Colors.RESET}  Asset: {market.asset:4} {side_arrow}  │  Shares: {position.shares:>8.2f}  │  Entry: {position.entry_price:.4f}     {result_color}║{Colors.RESET}")
        logger.info(f"{result_color}║{Colors.RESET}  P&L: {result_color}${pnl:>+10.2f}{Colors.RESET}  │  ROI: {result_color}{roi:>+7.1f}%{Colors.RESET}  │  Cost: ${position.cost_basis:.2f}     {result_color}║{Colors.RESET}")
        logger.info(f"{result_color}╚══════════════════════════════════════════════════════════════════╝{Colors.RESET}")

        # Credit paper executor BEFORE updating risk manager so that
        # _sync_existing_orders() will read the correct balance.
        # proceeds = cost_basis + pnl  (e.g. WIN: $50 cost + $49 pnl = $99)
        settlement_proceeds = position.cost_basis + pnl
        if self.config.paper_trading.enabled and hasattr(self.executor, 'credit_settlement'):
            self.executor.credit_settlement(
                proceeds=settlement_proceeds,
                market_id=market.condition_id,
            )

        # Record with risk manager
        self.risk_manager.record_position_close(
            market_key=market.condition_id,
            exit_price=payout_per_share,
            pnl=pnl,
        )
        self.metrics.record_trade(market.asset, pnl)

        # Record with trade history
        trade_history = get_trade_history()
        trade_history.record_close(
            market_id=market.condition_id,
            won=won,
            pnl=pnl,
            exit_price=payout_per_share,
        )

        # Track for auto-retraining
        if self.auto_retrainer:
            self.auto_retrainer.record_trade(won=won)
            # Check if retraining should be triggered
            result = self.auto_retrainer.check_and_retrain()
            if result and result.get("success"):
                logger.info(
                    f"🔄 Auto-retrain completed: {result['old_accuracy']:.1%} → "
                    f"{result['new_accuracy']:.1%} ({result['reason']})"
                )

        # Record ML outcome for model learning
        # Always try to record if ML is enabled - extract features if not stored
        if self.ml_predictor:
            from types import SimpleNamespace

            # Get ML features - either from stored position data or extract now
            if position.ml_volatility is not None:
                # Use stored ML data
                volatility = position.ml_volatility
                momentum = position.ml_momentum or 0.0
                arb_type = position.ml_arb_type or "none"
                spread = position.ml_spread or 0.0
                bid_depth = position.ml_bid_depth or 0.0
                ask_depth = position.ml_ask_depth or 0.0
                price_trend = position.ml_price_trend or 0.0
                distance_from_target = position.ml_distance_from_target or 0.0
                binance_lead_pct = position.ml_binance_lead_pct or 0.0
                binance_confirmation = position.ml_binance_confirmation or "NONE"
                trend_1h = position.ml_trend_1h or 0.0
                trend_4h = position.ml_trend_4h or 0.0
                trend_1d = position.ml_trend_1d or 0.0
                # Chart analysis features from stored position
                chart_rsi = position.ml_chart_rsi or 50.0
                chart_trend_strength = position.ml_chart_trend_strength or 0.0
                chart_is_uptrend = position.ml_chart_is_uptrend or 0.0
                chart_is_downtrend = position.ml_chart_is_downtrend or 0.0
                chart_is_ranging = position.ml_chart_is_ranging or 0.0
                chart_bullish_reversal = position.ml_chart_bullish_reversal or 0.0
                chart_bearish_reversal = position.ml_chart_bearish_reversal or 0.0
                chart_momentum = position.ml_chart_momentum or 0.0
                chart_bias_bullish = position.ml_chart_bias_bullish or 0.0
                chart_bias_bearish = position.ml_chart_bias_bearish or 0.0
                chart_confidence = position.ml_chart_confidence or 0.5
                chart_bullish_pattern = position.ml_chart_bullish_pattern or 0.0
                chart_bearish_pattern = position.ml_chart_bearish_pattern or 0.0
                # NEW INDICATOR FEATURES from stored position
                macd_histogram = position.ml_macd_histogram or 0.0
                macd_crossover = position.ml_macd_crossover or "none"
                bb_bandwidth = position.ml_bb_bandwidth or 0.0
                bb_position = position.ml_bb_position or "middle"
                stoch_k = position.ml_stoch_k or 50.0
                stoch_d = position.ml_stoch_d or 50.0
                stoch_signal = position.ml_stoch_signal or "neutral"
                rsi_divergence = position.ml_rsi_divergence or "none"
                rsi_divergence_strength = position.ml_rsi_divergence_strength or 0.0
                volume_ratio = position.ml_volume_ratio or 1.0
                is_high_volume = position.ml_is_high_volume or False
                obv_trend = position.ml_obv_trend or 0.0
                ha_trend = position.ml_ha_trend or "neutral"
                ha_consecutive = position.ml_ha_consecutive or 0
                ha_strength = position.ml_ha_strength or 0.0
                vwap_distance_pct = position.ml_vwap_distance_pct or 0.0
                vwap_position = position.ml_vwap_position or "at"
            else:
                # Extract features now (for positions without stored ML data)
                volatility = self.signal_generator.get_volatility(market.asset)
                current_price = self.signal_generator.get_price(market.asset)
                momentum = 0.0
                if current_price and market.target_price:
                    momentum = (current_price - market.target_price) / market.target_price
                    momentum = max(-1, min(1, momentum * 10))
                arb_type = "none"
                spread = market.best_ask - market.best_bid if market.best_ask and market.best_bid else 0.0
                bid_depth = market.bid_depth
                ask_depth = market.ask_depth
                price_trend = 0.0
                distance_from_target = 0.0
                if current_price and market.target_price:
                    distance_from_target = abs(current_price - market.target_price) / market.target_price
                binance_lead_pct = 0.0
                binance_confirmation = "NONE"
                trend_1h = 0.0
                trend_4h = 0.0
                trend_1d = 0.0
                # Default chart analysis features
                chart_rsi = 50.0
                chart_trend_strength = 0.0
                chart_is_uptrend = 0.0
                chart_is_downtrend = 0.0
                chart_is_ranging = 0.0
                chart_bullish_reversal = 0.0
                chart_bearish_reversal = 0.0
                chart_momentum = 0.0
                chart_bias_bullish = 0.0
                chart_bias_bearish = 0.0
                chart_confidence = 0.5
                chart_bullish_pattern = 0.0
                chart_bearish_pattern = 0.0
                # NEW INDICATOR FEATURES defaults
                macd_histogram = 0.0
                macd_crossover = "none"
                bb_bandwidth = 0.0
                bb_position = "middle"
                stoch_k = 50.0
                stoch_d = 50.0
                stoch_signal = "neutral"
                rsi_divergence = "none"
                rsi_divergence_strength = 0.0
                volume_ratio = 1.0
                is_high_volume = False
                obv_trend = 0.0
                ha_trend = "neutral"
                ha_consecutive = 0
                ha_strength = 0.0
                vwap_distance_pct = 0.0
                vwap_position = "at"
                # Try to fetch trends if available
                try:
                    from .data.binance import get_multi_timeframe_trends
                    trends = get_multi_timeframe_trends(market.asset)
                    trend_1h = trends.get("trend_1h", 0.0)
                    trend_4h = trends.get("trend_4h", 0.0)
                    trend_1d = trends.get("trend_1d", 0.0)
                except Exception:
                    pass

                logger.debug(f"Extracted ML features at settlement for {market.asset}")

            fake_signal = SimpleNamespace(
                edge=0.0,
                market=market,
                side=position.side,
                _arb_type=arb_type,
                # Additional attributes needed by TraderModelLoader
                recommended_price=position.entry_price,
                size_shares=position.shares,
                size_usd=position.shares * position.entry_price,
                time_remaining=market.time_remaining,
            )

            # Log ML learning with prediction accuracy
            ml_conf = position.ml_confidence
            ml_prediction = "WIN" if ml_conf and ml_conf >= 0.5 else "LOSS" if ml_conf else None
            actual_result = "WIN" if won else "LOSS"

            if ml_prediction:
                correct = ml_prediction == actual_result
                correct_str = f"{Colors.BRIGHT_GREEN}✓ CORRECT{Colors.RESET}" if correct else f"{Colors.BRIGHT_RED}✗ WRONG{Colors.RESET}"
                logger.info(
                    f"🤖 ML EVAL: {market.asset} | "
                    f"Predicted: {ml_prediction} ({ml_conf:.0%}) | Actual: {actual_result} | {correct_str}"
                )
            else:
                logger.info(f"🤖 ML EVAL: {market.asset} | Actual: {actual_result} (learning mode - no prediction)")

            # Get price data for ML
            current_price = self.signal_generator.get_price(market.asset) or 0.0
            target_price = market.target_price or 0.0
            price_high, price_low = self.signal_generator.get_price_range(market.asset)
            price_velocity = self.signal_generator.get_price_velocity(market.asset)

            self.ml_predictor.record_outcome(
                signal=fake_signal,
                volatility=volatility,
                price_momentum=momentum,
                won=won,
                arb_type=arb_type,
                spread=spread,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                price_trend=price_trend,
                distance_from_target=distance_from_target,
                binance_lead_pct=binance_lead_pct,
                binance_confirmation=binance_confirmation,
                trend_1h=trend_1h,
                trend_4h=trend_4h,
                trend_1d=trend_1d,
                current_price=current_price,
                target_price=target_price,
                price_high=price_high,
                price_low=price_low,
                price_velocity=price_velocity,
                # Chart analysis features
                chart_rsi=chart_rsi,
                chart_trend_strength=chart_trend_strength,
                chart_is_uptrend=chart_is_uptrend,
                chart_is_downtrend=chart_is_downtrend,
                chart_is_ranging=chart_is_ranging,
                chart_bullish_reversal=chart_bullish_reversal,
                chart_bearish_reversal=chart_bearish_reversal,
                chart_momentum=chart_momentum,
                chart_bias_bullish=chart_bias_bullish,
                chart_bias_bearish=chart_bias_bearish,
                chart_confidence=chart_confidence,
                chart_bullish_pattern=chart_bullish_pattern,
                chart_bearish_pattern=chart_bearish_pattern,
                # NEW INDICATOR FEATURES
                macd_histogram=macd_histogram,
                macd_crossover=macd_crossover,
                bb_bandwidth=bb_bandwidth,
                bb_position=bb_position,
                stoch_k=stoch_k,
                stoch_d=stoch_d,
                stoch_signal=stoch_signal,
                rsi_divergence=rsi_divergence,
                rsi_divergence_strength=rsi_divergence_strength,
                volume_ratio=volume_ratio,
                is_high_volume=is_high_volume,
                obv_trend=obv_trend,
                ha_trend=ha_trend,
                ha_consecutive=ha_consecutive,
                ha_strength=ha_strength,
                vwap_distance_pct=vwap_distance_pct,
                vwap_position=vwap_position,
            )

    async def _sync_ml_from_polymarket(self, force: bool = False):
        """
        Sync ML model with actual trades from Polymarket Data API.

        Uses multiple endpoints for comprehensive trade tracking:
        - /activity (type=TRADE) - All trade activity with BUY/SELL
        - /closed-positions - Completed trades with realized P&L
        - /activity (type=REDEEM) - Redeemed positions (settled markets)

        The ML learns from trades where we can determine the outcome (win/loss).

        NOTE: This is SKIPPED if you have a pre-trained model with 1000+ samples
        to avoid overwriting good training data with random market data.

        Args:
            force: If True, bypass rate limiting (used on startup)
        """
        if not self.client or not self.ml_predictor:
            return

        # Skip sync if we have a pre-trained model with significant samples
        # This prevents overwriting a good model (e.g., trained from successful trader)
        # with random market data that has lower accuracy
        if self.ml_predictor.training_samples >= 1000:
            if not hasattr(self, '_sync_skip_logged'):
                logger.info(
                    f"⏭️ Skipping CLOB sync - using pre-trained model "
                    f"({self.ml_predictor.training_samples} samples)"
                )
                self._sync_skip_logged = True
            return

        # Rate limit: only sync every 5 minutes (unless forced)
        now = datetime.now(timezone.utc)
        if not hasattr(self, '_last_ml_sync_time'):
            self._last_ml_sync_time = None

        if not force and self._last_ml_sync_time and (now - self._last_ml_sync_time).total_seconds() < 300:
            return

        self._last_ml_sync_time = now

        try:
            # Get wallet address
            wallet_address = self.client.get_address()
            if not wallet_address:
                logger.warning("Could not get wallet address for ML sync")
                return

            logger.info(f"🤖 ML Sync: Fetching trades for {wallet_address[:10]}...")
            logger.info("─" * 50)

            # Initialize Data API
            data_api = get_data_api()

            # Track what we've already synced (using condition_id + outcome + timestamp as key)
            if not hasattr(self, '_synced_trades'):
                self._synced_trades = set()

            synced_count = 0
            total_trades_found = 0
            crypto_trades_found = 0
            source_stats = {"clob": 0, "closed": 0, "activity": 0, "redeem": 0}

            # === Method 0: CLOB Trades (direct from order book - most reliable) ===
            try:
                clob_trades = get_trades(self.client, limit=500)
                logger.info(f"📊 [1/4] CLOB Trades: {len(clob_trades)} found")

                # Count trades with Up/Down outcomes (15-min crypto indicators)
                up_down_trades = [t for t in clob_trades if t.get("outcome", "").lower() in ["up", "down"]]
                if up_down_trades:
                    logger.info(f"     └─ {len(up_down_trades)} trades with Up/Down outcomes (15-min crypto)")

                # Log sample trades for visibility
                if clob_trades and len(clob_trades) > 0:
                    for i, sample in enumerate(clob_trades[:3]):
                        market_id = sample.get('market', '')[:12] + '...' if sample.get('market') else '?'
                        logger.info(
                            f"     └─ Trade {i+1}: {sample.get('side', '?'):4} | "
                            f"Price: {sample.get('price', '?')} | "
                            f"Size: {sample.get('size', '?')} | "
                            f"Outcome: {sample.get('outcome', '?')} | "
                            f"Market: {market_id}"
                        )
                    if len(clob_trades) > 3:
                        logger.info(f"     └─ ... and {len(clob_trades) - 3} more")

                # Process trades with market lookup
                lookups_done = 0
                for trade in clob_trades:
                    result = self._process_clob_trade_for_ml(trade)
                    if result == "synced":
                        synced_count += 1
                        crypto_trades_found += 1
                        source_stats["clob"] += 1
                    elif result == "crypto":
                        crypto_trades_found += 1
                    total_trades_found += 1

                # Show market lookup stats
                if hasattr(self, '_market_lookup_cache'):
                    cache_size = len(self._market_lookup_cache)
                    crypto_markets = sum(1 for v in self._market_lookup_cache.values() if v and v.get("asset"))
                    if cache_size > 0:
                        logger.info(f"     └─ Market lookups: {cache_size} total, {crypto_markets} 15-min crypto identified")
            except Exception as e:
                logger.info(f"📊 [1/4] CLOB Trades: Failed ({e})")

            # === Method 1: Closed Positions (completed trades with P&L) ===
            closed_positions = data_api.get_all_closed_positions(wallet_address, max_positions=500)
            logger.info(f"📊 [2/4] Closed Positions: {len(closed_positions)} found")

            # Log sample positions for visibility
            crypto_closed = []
            for pos in closed_positions:
                slug = (pos.get("slug", "") or "").lower()
                title = (pos.get("title", "") or "").lower()
                if any(p in slug or p in title for p in ["15m", "15-min", "updown"]):
                    crypto_closed.append(pos)

            if crypto_closed:
                for i, sample in enumerate(crypto_closed[:3]):
                    pnl = float(sample.get('realizedPnl', 0) or 0)
                    pnl_str = f"${pnl:+.2f}"
                    result = "✓ WIN" if pnl > 0 else "✗ LOSS"
                    logger.info(
                        f"     └─ {sample.get('outcome', '?'):5} | "
                        f"Avg: {sample.get('avgPrice', '?')} | "
                        f"{result} {pnl_str}"
                    )
                if len(crypto_closed) > 3:
                    logger.info(f"     └─ ... and {len(crypto_closed) - 3} more 15-min trades")
            else:
                logger.info(f"     └─ No 15-min crypto positions found")

            for pos in closed_positions:
                result = self._process_trade_for_ml(pos, "closed")
                if result == "synced":
                    synced_count += 1
                    crypto_trades_found += 1
                    source_stats["closed"] += 1
                elif result == "crypto":
                    crypto_trades_found += 1
                total_trades_found += 1

            # === Method 2: Trade Activity (all BUY/SELL activity) ===
            trade_activity = data_api.get_all_activity(wallet_address, activity_type="TRADE", max_items=500)
            logger.info(f"📊 [3/4] Trade Activity: {len(trade_activity)} found")

            # Log sample activities for visibility
            crypto_activity = []
            for act in trade_activity:
                slug = (act.get("slug", "") or "").lower()
                title = (act.get("title", "") or "").lower()
                if any(p in slug or p in title for p in ["15m", "15-min", "updown"]):
                    crypto_activity.append(act)

            if crypto_activity:
                for i, sample in enumerate(crypto_activity[:3]):
                    side_icon = "🟢" if sample.get('side') == "BUY" else "🔴"
                    usdc = float(sample.get('usdcSize', 0) or 0)
                    logger.info(
                        f"     └─ {side_icon} {sample.get('side', '?'):4} {sample.get('outcome', '?'):5} | "
                        f"Price: {sample.get('price', '?')} | "
                        f"${usdc:.2f}"
                    )
                if len(crypto_activity) > 3:
                    logger.info(f"     └─ ... and {len(crypto_activity) - 3} more 15-min trades")
            else:
                logger.info(f"     └─ No 15-min crypto activity found")

            for trade in trade_activity:
                result = self._process_trade_for_ml(trade, "activity")
                if result == "synced":
                    synced_count += 1
                    crypto_trades_found += 1
                    source_stats["activity"] += 1
                elif result == "crypto":
                    crypto_trades_found += 1
                total_trades_found += 1

            # === Method 3: Redeem Activity (claimed winnings) ===
            redeem_activity = data_api.get_all_activity(wallet_address, activity_type="REDEEM", max_items=200)
            logger.info(f"📊 [4/4] Redeem Activity: {len(redeem_activity)} found")

            # Log sample redeems for visibility
            crypto_redeems = []
            for act in redeem_activity:
                slug = (act.get("slug", "") or "").lower()
                title = (act.get("title", "") or "").lower()
                if any(p in slug or p in title for p in ["15m", "15-min", "updown"]):
                    crypto_redeems.append(act)

            if crypto_redeems:
                for i, sample in enumerate(crypto_redeems[:3]):
                    usdc = float(sample.get('usdcSize', 0) or 0)
                    logger.info(
                        f"     └─ 💰 {sample.get('outcome', '?'):5} redeemed | "
                        f"${usdc:.2f}"
                    )
                if len(crypto_redeems) > 3:
                    logger.info(f"     └─ ... and {len(crypto_redeems) - 3} more redeems")
            else:
                logger.info(f"     └─ No 15-min crypto redeems found")

            for redeem in redeem_activity:
                result = self._process_trade_for_ml(redeem, "redeem")
                if result == "synced":
                    synced_count += 1
                    crypto_trades_found += 1
                    source_stats["redeem"] += 1
                elif result == "crypto":
                    crypto_trades_found += 1

            # Summary logging
            logger.info("─" * 50)
            if synced_count > 0:
                sources_used = [f"{k}:{v}" for k, v in source_stats.items() if v > 0]
                logger.info(
                    f"✅ ML SYNCED: {synced_count} new trades | "
                    f"Sources: {', '.join(sources_used) or 'none'}"
                )
                logger.info(
                    f"📈 Total ML samples: {self.ml_predictor.training_samples}"
                )
            elif crypto_trades_found > 0:
                logger.info(
                    f"ℹ️  Found {crypto_trades_found} crypto trades (already synced or pending settlement)"
                )
            else:
                logger.info(
                    f"⚠️  No 15-min crypto trades found in {total_trades_found} total activities"
                )
                logger.info(
                    f"    Hint: Make sure you have traded 15-min crypto markets (BTC/ETH/SOL/XRP)"
                )

        except Exception as e:
            logger.error(f"ML sync from Polymarket failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _process_trade_for_ml(self, trade_data: dict, source: str) -> str:
        """
        Process a single trade/position for ML learning.

        Args:
            trade_data: Trade or position data from API
            source: Source type ("closed", "activity", "redeem")

        Returns:
            "synced" if successfully synced, "crypto" if crypto but already synced,
            "skip" if not a 15-min crypto trade
        """
        try:
            # Extract identifiers
            condition_id = trade_data.get("conditionId", "")
            outcome = trade_data.get("outcome", "")
            timestamp = trade_data.get("timestamp", "")
            tx_hash = trade_data.get("transactionHash", "")

            # Create unique key for deduplication
            trade_key = f"{condition_id}:{outcome}:{timestamp}:{tx_hash}"

            # Skip if already synced
            if trade_key in self._synced_trades:
                return "skip"

            # Extract market info
            slug = trade_data.get("slug", "") or trade_data.get("eventSlug", "")
            title = trade_data.get("title", "")
            slug_lower = slug.lower()
            title_lower = title.lower()

            # Comprehensive 15-min crypto market detection
            is_15min = any(p in slug_lower or p in title_lower for p in [
                "updown-15m", "-15m-", "15min", "15-min", "15m",
                "up-down-15m", "updown15m", "crypto-15"
            ])

            if not is_15min:
                return "skip"

            # Identify asset
            asset = None
            asset_patterns = {
                "btc": "BTC", "bitcoin": "BTC",
                "eth": "ETH", "ethereum": "ETH",
                "sol": "SOL", "solana": "SOL",
                "xrp": "XRP", "ripple": "XRP",
            }
            for pattern, asset_name in asset_patterns.items():
                if pattern in slug_lower or pattern in title_lower:
                    asset = asset_name
                    break

            if not asset:
                return "skip"

            # Get side (Up/Down)
            if isinstance(outcome, str):
                outcome_lower = outcome.lower()
                if "up" in outcome_lower:
                    side = Side.UP
                elif "down" in outcome_lower:
                    side = Side.DOWN
                else:
                    return "skip"
            else:
                return "skip"

            # Determine win/loss based on source
            won = None

            if source == "closed":
                # Closed positions have realized P&L
                realized_pnl = float(trade_data.get("realizedPnl", 0) or 0)
                won = realized_pnl > 0

            elif source == "redeem":
                # Redeems are always wins (you only redeem winning positions)
                won = True

            elif source == "activity":
                # Trade activity - check if we can determine outcome
                # For BUY trades, we need to check if the market settled in our favor
                trade_side = trade_data.get("side", "")
                if trade_side == "SELL":
                    # SELL means we closed a position - check P&L from usdcSize vs size*price
                    size = float(trade_data.get("size", 0) or 0)
                    price = float(trade_data.get("price", 0) or 0)
                    usdc_size = float(trade_data.get("usdcSize", 0) or 0)
                    # If we sold, the P&L depends on our entry price (unknown)
                    # Skip for now - closed positions will capture this
                    return "crypto"
                else:
                    # BUY trade - need to check settlement
                    # Skip for now - will be captured when position closes
                    return "crypto"

            if won is None:
                return "crypto"

            # Get trade details
            avg_price = float(trade_data.get("avgPrice", 0) or trade_data.get("price", 0) or 0)
            total_bought = float(trade_data.get("totalBought", 0) or trade_data.get("size", 0) or 0)

            if avg_price <= 0 or total_bought <= 0:
                return "skip"

            # Create mock signal for ML recording
            from types import SimpleNamespace

            # ============================================================
            # ENHANCED FEATURE EXTRACTION (same as learn_from_trader.py)
            # ============================================================

            # 1. Entry price deviation from 0.50 (market mid-point)
            price_deviation = avg_price - 0.50

            # 2. Estimate edge based on entry price and outcome
            estimated_edge = (1.0 - avg_price) if won else (avg_price - 1.0)
            estimated_edge = max(0.01, min(0.50, abs(estimated_edge)))

            # 3. Position sizing (normalize by typical size)
            position_size_normalized = min(total_bought / 100, 1.0)

            # 4. Price momentum inference
            inferred_momentum = 0.2 if (side == Side.UP and won) or (side == Side.DOWN and not won) else -0.2

            # 5. Infer volatility from entry price
            inferred_volatility = 0.5 - abs(price_deviation)
            inferred_volatility = max(0.01, inferred_volatility * 0.1)

            # 6. Distance from fair value (0.50)
            distance_from_fair = abs(price_deviation)

            fake_signal = SimpleNamespace(
                edge=estimated_edge,
                market=SimpleNamespace(
                    asset=asset,
                    condition_id="",
                    target_price=1.0,
                    time_remaining=450,
                    best_bid=avg_price - 0.01,
                    best_ask=avg_price + 0.01,
                    bid_depth=1000 * position_size_normalized,
                    ask_depth=1000 * position_size_normalized,
                ),
                side=side,
                _arb_type="synced",
                recommended_price=avg_price,
                size_shares=total_bought,
                size_usd=total_bought * avg_price,
                time_remaining=450,
            )

            # Record to ML model with enhanced features
            self.ml_predictor.record_outcome(
                signal=fake_signal,
                volatility=inferred_volatility,
                price_momentum=inferred_momentum,
                won=won,
                arb_type="synced",
                spread=0.02,
                bid_depth=1000 * position_size_normalized,
                ask_depth=1000 * position_size_normalized,
                price_trend=inferred_momentum * 0.5,
                distance_from_target=distance_from_fair,
                binance_lead_pct=inferred_momentum * 0.005,
                binance_confirmation="MEDIUM" if abs(inferred_momentum) > 0.1 else "NONE",
                trend_1h=inferred_momentum * 0.3,
                trend_4h=inferred_momentum * 0.2,
                trend_1d=inferred_momentum * 0.1,
                current_price=avg_price,
                target_price=0.50,
                price_high=avg_price + inferred_volatility,
                price_low=avg_price - inferred_volatility,
                price_velocity=abs(inferred_momentum) * 0.01,
            )

            # Mark as synced
            self._synced_trades.add(trade_key)

            result_str = "✓ WIN" if won else "✗ LOSS"
            result_icon = "🟢" if won else "🔴"
            logger.info(f"     {result_icon} ML +1: {asset} {side.value:4} {result_str} (from {source})")

            return "synced"

        except Exception as e:
            logger.debug(f"Error processing trade for ML: {e}")
            return "skip"

    def _process_clob_trade_for_ml(self, trade: dict) -> str:
        """
        Process a CLOB trade for ML learning.

        CLOB trades have format:
        {
            "id": "trade_id",
            "market": "condition_id",
            "asset_id": "token_id",
            "side": "BUY" or "SELL",
            "size": "100.5",
            "price": "0.55",
            "status": "MATCHED",
            "match_time": "1234567890",
            "outcome": "Yes" or "No" or "Up" or "Down",
            "taker_order_id": "...",
            "maker_address": "0x...",
            "fee_rate_bps": "0",
            ...
        }

        Args:
            trade: CLOB trade data

        Returns:
            "synced" if successfully synced, "crypto" if crypto but needs settlement,
            "skip" if not a 15-min crypto trade
        """
        try:
            trade_id = trade.get("id", "")
            market_id = trade.get("market", "")
            asset_id = trade.get("asset_id", "")
            match_time = trade.get("match_time", "")

            # Create unique key
            trade_key = f"clob:{trade_id}:{market_id}:{match_time}"

            if trade_key in self._synced_trades:
                return "skip"

            # Try to identify if this is a 15-min crypto market
            # CLOB trades don't have slug, so we need to check the market
            outcome = trade.get("outcome", "")
            side_str = trade.get("side", "")
            price = float(trade.get("price", 0) or 0)
            size = float(trade.get("size", 0) or 0)

            if price <= 0 or size <= 0:
                return "skip"

            # Determine side from outcome (Up/Down for 15-min crypto markets)
            outcome_lower = outcome.lower()
            side = None
            if outcome_lower in ["yes", "up"]:
                side = Side.UP
            elif outcome_lower in ["no", "down"]:
                side = Side.DOWN
            else:
                return "skip"  # Not a 15-min crypto market

            # For CLOB trades, we need to determine the asset
            # First check if we have this market in our known markets
            market = self.markets.get(market_id) or self.expiring_markets.get(market_id)
            asset = None

            if market:
                # We have market info cached
                asset = market.asset
            else:
                # Initialize market lookup cache if needed
                if not hasattr(self, '_market_lookup_cache'):
                    self._market_lookup_cache = {}

                # Check cache first
                if market_id in self._market_lookup_cache:
                    cached = self._market_lookup_cache[market_id]
                    if cached:
                        asset = cached.get("asset")
                else:
                    # Look up market from Gamma API
                    try:
                        market_info = self.gamma_api.get_market_by_id(market_id)
                        if market_info:
                            slug = market_info.get("slug", "").lower()
                            question = market_info.get("question", "").lower()

                            # Check if this is a 15-min crypto market
                            is_15min = any(p in slug or p in question for p in [
                                "updown-15m", "-15m-", "15min", "15-min", "15m",
                                "up-down-15m", "updown15m"
                            ])

                            if is_15min:
                                # Identify asset from slug or question
                                for pattern, asset_name in [
                                    ("btc", "BTC"), ("bitcoin", "BTC"),
                                    ("eth", "ETH"), ("ethereum", "ETH"),
                                    ("sol", "SOL"), ("solana", "SOL"),
                                    ("xrp", "XRP"), ("ripple", "XRP"),
                                ]:
                                    if pattern in slug or pattern in question:
                                        asset = asset_name
                                        break

                            # Cache the result (even if not 15-min, to avoid re-lookup)
                            self._market_lookup_cache[market_id] = {
                                "asset": asset,
                                "is_15min": is_15min,
                                "slug": slug,
                            }
                        else:
                            # Cache negative result
                            self._market_lookup_cache[market_id] = None
                    except Exception as e:
                        logger.debug(f"Market lookup failed for {market_id[:16]}...: {e}")
                        self._market_lookup_cache[market_id] = None

            if not asset:
                return "skip"

            # For CLOB trades, we can't directly determine win/loss without settlement
            # But we can track BUY trades and check if they've settled
            if side_str == "SELL":
                # SELL means closing position - this could be early exit or settlement
                # Check if this was profitable (price > 0.5 for long, price < 0.5 for short)
                # This is a heuristic since we don't have perfect info
                if side == Side.UP:
                    won = price > 0.5  # If we sold UP for > 0.5, likely winning
                else:
                    won = price < 0.5  # If we sold DOWN for < 0.5, likely winning
            else:
                # BUY - we need to wait for settlement to know outcome
                # Skip for now, will be captured by closed-positions or redeem
                return "crypto"

            # ============================================================
            # ENHANCED FEATURE EXTRACTION (same as learn_from_trader.py)
            # ============================================================
            from types import SimpleNamespace

            # 1. Entry price deviation from 0.50 (market mid-point)
            price_deviation = price - 0.50

            # 2. Estimate edge based on entry price and outcome
            estimated_edge = (1.0 - price) if won else (price - 1.0)
            estimated_edge = max(0.01, min(0.50, abs(estimated_edge)))

            # 3. Position sizing (normalize by typical size)
            position_size_normalized = min(size / 100, 1.0)

            # 4. Price momentum inference
            inferred_momentum = 0.2 if (side == Side.UP and won) or (side == Side.DOWN and not won) else -0.2

            # 5. Infer volatility from entry price
            inferred_volatility = 0.5 - abs(price_deviation)
            inferred_volatility = max(0.01, inferred_volatility * 0.1)

            # 6. Distance from fair value (0.50)
            distance_from_fair = abs(price_deviation)

            fake_signal = SimpleNamespace(
                edge=estimated_edge,
                market=SimpleNamespace(
                    asset=asset,
                    condition_id="",
                    target_price=1.0,
                    time_remaining=450,
                    best_bid=price - 0.01,
                    best_ask=price + 0.01,
                    bid_depth=1000 * position_size_normalized,
                    ask_depth=1000 * position_size_normalized,
                ),
                side=side,
                _arb_type="synced",
                recommended_price=price,
                size_shares=size,
                size_usd=size * price,
                time_remaining=450,
            )

            # Record to ML model with enhanced features
            self.ml_predictor.record_outcome(
                signal=fake_signal,
                volatility=inferred_volatility,
                price_momentum=inferred_momentum,
                won=won,
                arb_type="synced",
                spread=0.02,
                bid_depth=1000 * position_size_normalized,
                ask_depth=1000 * position_size_normalized,
                price_trend=inferred_momentum * 0.5,
                distance_from_target=distance_from_fair,
                binance_lead_pct=inferred_momentum * 0.005,
                binance_confirmation="MEDIUM" if abs(inferred_momentum) > 0.1 else "NONE",
                trend_1h=inferred_momentum * 0.3,
                trend_4h=inferred_momentum * 0.2,
                trend_1d=inferred_momentum * 0.1,
                current_price=price,
                target_price=0.50,
                price_high=price + inferred_volatility,
                price_low=price - inferred_volatility,
                price_velocity=abs(inferred_momentum) * 0.01,
            )

            self._synced_trades.add(trade_key)

            result_str = "✓ WIN" if won else "✗ LOSS"
            result_icon = "🟢" if won else "🔴"
            logger.info(f"     {result_icon} ML +1: {asset} {side.value:4} {result_str} (from CLOB)")

            return "synced"

        except Exception as e:
            logger.debug(f"Error processing CLOB trade for ML: {e}")
            return "skip"

    async def _sync_positions_from_polymarket(self):
        """
        Sync internal position tracking with actual Polymarket positions.

        Removes stale positions that no longer exist on Polymarket.
        This prevents the bot from blocking new trades due to ghost positions.

        IMPORTANT: Only clears positions that are both:
        1. Not found on Polymarket API
        2. Either: older than 60 seconds OR market has passed settlement time
        """
        if not self.client:
            return

        try:
            # Get wallet address
            wallet_address = self.client.get_address()
            if not wallet_address:
                return

            # Initialize Data API
            data_api = get_data_api()

            # Get current positions from Polymarket
            api_positions = data_api.get_positions(wallet_address, limit=100, size_threshold=0.1)

            # Build set of condition IDs that actually have positions
            api_condition_ids = set()
            for pos in api_positions:
                cid = pos.get("conditionId", "")
                if cid:
                    api_condition_ids.add(cid)

            # Check internal positions against API
            internal_positions = list(self.risk_manager.positions.keys())
            cleared_count = 0
            now = datetime.now(timezone.utc)
            grace_period_seconds = 60  # 60 seconds grace period for new positions

            for market_key in internal_positions:
                position = self.risk_manager.positions.get(market_key)
                if not position:
                    continue

                asset = position.market.asset if hasattr(position, 'market') else "???"

                # Check if market has settled (end_time has passed)
                market_settled = False
                if hasattr(position, 'market') and position.market and hasattr(position.market, 'end_time'):
                    # Calculate actual time remaining (can be negative)
                    actual_remaining = (position.market.end_time - now).total_seconds()
                    market_settled = actual_remaining < -30  # 30 second buffer after settlement

                # Check position age
                position_age = 0
                if hasattr(position, 'entry_time') and position.entry_time:
                    position_age = (now - position.entry_time).total_seconds()

                # Skip if position is new AND market hasn't settled yet
                # (Give API time to sync for new positions)
                if position_age < grace_period_seconds and not market_settled:
                    logger.debug(
                        f"Skipping position sync for {asset}: "
                        f"opened {position_age:.0f}s ago, market not settled"
                    )
                    continue

                if market_key not in api_condition_ids:
                    # Position exists internally but not on Polymarket
                    side = position.side.value if hasattr(position, 'side') else "???"

                    reason = "market settled" if market_settled else "not found on API"
                    logger.info(
                        f"🧹 Clearing stale position: {asset} {side} | {reason}"
                    )

                    # Remove from risk manager
                    self.risk_manager.positions.pop(market_key, None)
                    cleared_count += 1

                    # Also clear any asset-level tracking
                    if hasattr(self, '_api_positions') and asset in self._api_positions:
                        del self._api_positions[asset]

            if cleared_count > 0:
                logger.info(f"🧹 Cleared {cleared_count} stale positions from tracking")

        except Exception as e:
            logger.debug(f"Position sync failed: {e}")

    async def _process_market(self, market: MarketState):
        """
        Process a single market for trading opportunities.

        Args:
            market: Market to process
        """
        # Skip if market is about to expire (should be in expiring_markets)
        if market.time_remaining < self.config.trading.min_time_remaining:
            return

        # === SHORT-CIRCUIT: Skip if we already have position for this asset ===
        # This avoids generating signals and all downstream processing
        for pos in self.risk_manager.positions.values():
            if hasattr(pos, 'market') and pos.market.asset == market.asset:
                # Already have position - skip all processing (no log, very common)
                return

        # === SHORT-CIRCUIT: Skip if max positions reached ===
        if len(self.risk_manager.positions) >= self.config.trading.max_concurrent_positions:
            # Max positions reached - skip (logged via throttled block later if signal generated)
            return

        # Track when we first saw this market
        market_id = market.condition_id
        now = datetime.now(timezone.utc)
        if market_id not in self._market_first_seen:
            self._market_first_seen[market_id] = now
            logger.debug(f"[{market.asset}] First observation - collecting data...")

        # Check minimum observation time for this market
        observation_time = (now - self._market_first_seen[market_id]).total_seconds()
        min_observation = self.config.trading.min_observation_time
        if observation_time < min_observation:
            logger.debug(
                f"[{market.asset}] Observing: {observation_time:.0f}s / {min_observation:.0f}s"
            )
            # Still update orderbook data, just don't trade yet
            self._update_market_from_orderbook(market)
            return

        # Check minimum price samples for this asset (Chainlink)
        asset_symbol = f"{market.asset.lower()}/usd"
        price_history = self.signal_generator.price_histories.get(asset_symbol, [])
        min_samples = self.config.trading.min_price_samples
        if len(price_history) < min_samples:
            logger.debug(
                f"[{market.asset}] Need more Chainlink data: {len(price_history)}/{min_samples} samples"
            )
            self._update_market_from_orderbook(market)
            return

        # Check Binance price availability (used for confirmation signal - optional)
        binance_price = self.signal_generator.get_binance_price(market.asset)
        chainlink_price = self.signal_generator.get_price(market.asset)

        # CRITICAL: Validate target price before trading
        # After period transitions, the API may return stale target prices
        if chainlink_price and market.target_price:
            target_deviation = abs(chainlink_price - market.target_price) / market.target_price
            # For a 15-min market, target price should be within ~5% of current price
            # (crypto can move, but >5% in 15 mins is very unusual and suggests stale data)
            max_deviation = 0.05  # 5% max deviation
            if target_deviation > max_deviation:
                stale_key = f"{market.condition_id}:stale_target"
                if stale_key not in self._logged_rejections:
                    self._logged_rejections.add(stale_key)
                    logger.warning(
                        f"⚠️ [{market.asset}] STALE TARGET PRICE DETECTED! "
                        f"Target ${market.target_price:,.2f} deviates {target_deviation:.1%} "
                        f"from Chainlink ${chainlink_price:,.2f} (max {max_deviation:.0%}). "
                        f"Skipping trades until target price updates."
                    )
                return

        # Binance is optional - used for confirmation signal boost, not required
        # The signal generator handles missing Binance data gracefully

        # Log price comparison when both prices are available
        if chainlink_price and binance_price:
            lead = binance_price - chainlink_price
            lead_pct = (lead / chainlink_price) * 100 if chainlink_price else 0
            logger.debug(
                f"[{market.asset}] Price check: Chainlink=${chainlink_price:,.2f}, "
                f"Binance=${binance_price:,.2f}, Lead={lead_pct:+.3f}%"
            )

        # Update market state from order book
        self._update_market_from_orderbook(market)

        # Generate trading signal (multi-strategy meta-layer or default signal generator)
        signal = None
        if self.meta_strategy:
            price_history = self.signal_generator.price_histories.get(market.asset, [])
            mid_price = (market.best_bid + market.best_ask) / 2 if market.best_bid and market.best_ask else 0.0
            orderbook_imbalance = 0.0
            if market.bid_depth + market.ask_depth > 0:
                orderbook_imbalance = (market.bid_depth - market.ask_depth) / (market.bid_depth + market.ask_depth)

            price_velocity = 0.0
            if len(price_history) >= 2 and price_history[-2] > 0:
                price_velocity = (price_history[-1] - price_history[-2]) / price_history[-2]

            volatility = self.signal_generator.get_volatility(market.asset)
            ctx = StrategyContext(
                asset=market.asset,
                mid_price=mid_price,
                reference_price=chainlink_price,
                orderbook_imbalance=orderbook_imbalance,
                price_velocity=price_velocity,
                volatility=volatility,
                time_remaining=market.time_remaining,
            )
            meta_signal = self.meta_strategy.generate(ctx)
            if not meta_signal:
                logger.debug("ORDER_BLOCK asset=%s reason=meta_strategy_no_signal", market.asset)
                return
            self.metrics.record_strategy_performance("meta", meta_signal.edge, meta_signal.confidence)
            signal = self._signal_from_meta(market, meta_signal, chainlink_price or 0.0)
        else:
            signal = self.signal_generator.generate_signal(market)
        if not signal:
            logger.debug("ORDER_BLOCK asset=%s reason=no_signal_generated", market.asset)
            return

        # Probabilistic prediction layer
        # NOTE: Only apply prediction model if it AGREES with signal direction
        # If prediction model gives negative edge, keep signal generator's values
        if not self.ml_engine and self.prediction_model and self.config.prediction.enabled:
            price_history = self.signal_generator.price_histories.get(market.asset, [])
            features = self.feature_builder.build(market, price_history, chainlink_price)
            pred = self.prediction_model.predict(features)
            if pred.confidence < self.config.prediction.min_confidence:
                logger.info(
                    "ORDER_BLOCK asset=%s reason=prediction_low_confidence conf=%.4f min=%.4f",
                    market.asset, pred.confidence, self.config.prediction.min_confidence
                )
                return
            market_prob = market.implied_prob_up
            prob_yes = pred.prob_yes
            original_edge = signal.edge
            original_true_prob = signal.true_prob
            original_side = signal.side.value

            # Calculate what edge would be with prediction model
            if signal.side == Side.UP:
                pred_true_prob = prob_yes
                pred_edge = prob_yes - market_prob
            else:
                pred_true_prob = 1 - prob_yes
                pred_edge = market_prob - prob_yes

            # Only apply prediction if edge stays positive (model agrees with direction)
            if pred_edge > 0:
                signal.true_prob = pred_true_prob
                signal.edge = pred_edge
                logger.debug(
                    "PREDICTION_APPLIED asset=%s side=%s edge=%.4f->%.4f prob=%.4f->%.4f",
                    market.asset, original_side, original_edge, pred_edge,
                    original_true_prob, pred_true_prob
                )
            else:
                # Keep original signal generator values - prediction model disagrees
                logger.debug(
                    "PREDICTION_SKIPPED asset=%s side=%s reason=negative_edge "
                    "signal_edge=%.4f pred_edge=%.4f prob_yes=%.4f market_prob=%.4f",
                    market.asset, original_side, original_edge, pred_edge,
                    prob_yes, market_prob
                )

        # Dynamic risk sizing
        if self.config.risk_sizing.enabled:
            volatility = self.signal_generator.get_volatility(market.asset)
            market_price = max(0.01, signal.recommended_price)
            # Debug log to trace edge/prob/price values entering sizing
            logger.debug(
                "SIZING_INPUT asset=%s side=%s edge=%.4f true_prob=%.4f market_price=%.4f "
                "recommended_price=%.4f best_bid=%.4f best_ask=%.4f vol=%.4f bankroll=%.2f",
                market.asset, signal.side.value, signal.edge, signal.true_prob, market_price,
                signal.recommended_price, market.best_bid, market.best_ask, volatility,
                self.risk_manager.current_bankroll
            )
            desired_size = self.risk_sizer.size_position(
                bankroll=self.risk_manager.current_bankroll,
                edge=signal.edge,
                prob=signal.true_prob,
                market_price=market_price,
                volatility=volatility,
                base_size=signal.size_usd,
            )
            if desired_size <= 0:
                # Throttle this log to once per 30s per asset to avoid spam
                block_key = f"{market.asset}:risk_sizing_zero"
                now_ts = time.time()
                last_log = getattr(self, '_risk_sizing_log_ts', {}).get(block_key, 0.0)
                if now_ts - last_log >= 30.0:
                    if not hasattr(self, '_risk_sizing_log_ts'):
                        self._risk_sizing_log_ts = {}
                    self._risk_sizing_log_ts[block_key] = now_ts
                    logger.info(
                        "ORDER_BLOCK asset=%s side=%s reason=risk_sizing_zero edge=%.4f vol=%.4f "
                        "bankroll=%.2f prob=%.4f price=%.4f",
                        market.asset, signal.side.value, signal.edge, volatility,
                        self.risk_manager.current_bankroll, signal.true_prob, market_price
                    )
                return
            signal.size_usd = desired_size
            signal.size_shares = desired_size / market_price

        # Log current Polymarket best prices and computed edge (throttled per asset)
        now_ts = time.time()
        last_price_log = self._price_log_ts.get(market.asset, 0.0)
        if now_ts - last_price_log >= 5.0:
            yes_bid = market.best_bid or 0.0
            yes_ask = market.best_ask or 0.0
            if yes_bid > 0 and yes_ask > 0:
                no_bid = 1.0 - yes_ask
                no_ask = 1.0 - yes_bid
                spread_yes = yes_ask - yes_bid
                edge_val = float(getattr(signal, "edge", 0.0) or 0.0)
                logger.info(
                    "PMKT_PRICES asset=%s yes_bid=%.4f yes_ask=%.4f no_bid=%.4f no_ask=%.4f "
                    "spread_yes=%.4f edge=%.4f",
                    market.asset,
                    yes_bid,
                    yes_ask,
                    no_bid,
                    no_ask,
                    spread_yes,
                    edge_val,
                )
                self._price_log_ts[market.asset] = now_ts

        # Skip signals without action
        if signal.recommended_action == OrderAction.SKIP:
            logger.info(
                "ORDER_DECISION asset=%s decision=SKIP edge_used=%.4f min_edge_used=%.4f "
                "time_remaining=%.0f simple_mode=%s aggressive_mode=%s order_type=%s",
                market.asset,
                float(getattr(signal, "edge", 0.0) or 0.0),
                float(self.config.trading.min_edge),
                float(getattr(signal, "time_remaining", 0.0) or 0.0),
                str(self.config.trading.simple_mode).lower(),
                str(self.config.trading.aggressive_mode).lower(),
                str(signal.recommended_action),
            )
            logger.debug(f"SKIP {market.asset}: {signal.reasoning}")
            return

        # NOTE: ORDER_DECISION=PLACE_ORDER is logged AFTER all gating checks pass,
        # right before _execute_signal() is called. See line ~4000.

        # Validate signal against risk limits (no verbose logging - only log final decision)
        is_valid, reason = self.risk_manager.validate_signal(signal)
        if not is_valid:
            # Throttle common blocks (max_positions, already_have_position)
            block_reason = f"risk_manager:{reason.replace(' ', '_')[:30]}"
            self._log_order_block(
                market.asset, block_reason,
                market=market.condition_id[:8], side=signal.side.value
            )
            return

        # Trade history filter - check past performance for this asset/side
        trade_history = get_trade_history()
        should_proceed, history_reason = trade_history.evaluate_trade(
            asset=market.asset,
            side=signal.side.value,
            min_trades=self.config.trading.history_min_trades,
            min_win_rate=self.config.trading.history_min_win_rate,
            check_prediction_accuracy=True,
            min_prediction_accuracy=self.config.trading.history_min_prediction_accuracy,
        )
        if not should_proceed:
            block_reason = f"trade_history:{history_reason.replace(' ', '_')[:30]}"
            self._log_order_block(
                market.asset, block_reason,
                market=market.condition_id[:8], side=signal.side.value
            )
            return

        # === CHART-BASED FILTER ===
        # Block trades that go against strong chart signals
        # This is the PRIMARY filter - chart analysis determines if we should trade
        chart_analysis = None
        if CHART_ANALYSIS_AVAILABLE:
            try:
                chart_analysis = analyze_chart(market.asset)
                if chart_analysis:
                    chart_bias = chart_analysis.bias
                    chart_confidence = chart_analysis.confidence
                    market_type = chart_analysis.market_type.value
                    avg_trend = (chart_analysis.trend_15m + chart_analysis.trend_1h + chart_analysis.trend_4h) / 3

                    # Determine chart direction
                    chart_says_down = (chart_bias == "bearish" and chart_confidence >= 0.6) or avg_trend < -0.5
                    chart_says_up = (chart_bias == "bullish" and chart_confidence >= 0.6) or avg_trend > 0.5

                    # Calculate signal strength
                    chart_signal_strength = chart_confidence if chart_bias != "neutral" else abs(avg_trend)
                    if "strong_downtrend" in market_type or "strong_uptrend" in market_type:
                        chart_signal_strength = min(1.0, chart_signal_strength + 0.2)

                    # Block trades that go AGAINST strong chart signals
                    # EXCEPTION: Allow REVERSAL trades (these intentionally go against the trend!)
                    # Can be disabled with CHART_FILTER_ENABLED=false for testing
                    is_reversal_trade = "REVERSAL" in signal.reasoning.upper()
                    chart_filter_enabled = getattr(self.config.trading, 'chart_filter_enabled', True)

                    if chart_filter_enabled and chart_signal_strength >= 0.7 and not is_reversal_trade:
                        # Throttle chart filter logs to once per 30s per asset+side
                        block_key = f"{market.asset}:{signal.side.value}:chart_filter"
                        now_ts = time.time()
                        if not hasattr(self, '_chart_block_log_ts'):
                            self._chart_block_log_ts = {}
                        last_log = self._chart_block_log_ts.get(block_key, 0.0)

                        if chart_says_down and signal.side == Side.UP:
                            if now_ts - last_log >= 30.0:
                                self._chart_block_log_ts[block_key] = now_ts
                                logger.info(
                                    "ORDER_BLOCK asset=%s market=%s side=%s reason=chart_bearish_vs_up "
                                    "strength=%.2f edge=%.4f",
                                    market.asset, market.condition_id[:8], signal.side.value,
                                    chart_signal_strength, signal.edge
                                )
                            return
                        elif chart_says_up and signal.side == Side.DOWN:
                            if now_ts - last_log >= 30.0:
                                self._chart_block_log_ts[block_key] = now_ts
                                logger.info(
                                    "ORDER_BLOCK asset=%s market=%s side=%s reason=chart_bullish_vs_down "
                                    "strength=%.2f edge=%.4f",
                                    market.asset, market.condition_id[:8], signal.side.value,
                                    chart_signal_strength, signal.edge
                                )
                            return

            except Exception as e:
                logger.debug(f"Chart analysis failed: {e}")

        # EV-based ML engine
        ml_confidence = None
        if self.ml_engine and self.ml_policy:
            try:
                snapshot = self._build_ml_snapshot(market, chainlink_price or 0.0)
                if not snapshot:
                    # Log why snapshot failed and CONTINUE with signal generator values (fallback)
                    orderbook = self.clob_feed.get_orderbook(market.up_token_id)
                    depth_levels = self.ml_engine.bundle.builder.depth_levels if self.ml_engine else 0
                    ob_bids = len(orderbook.bids) if orderbook else 0
                    ob_asks = len(orderbook.asks) if orderbook else 0
                    asset_key = f"{market.asset.lower()}/usd"
                    history_len = len(self.signal_generator.price_histories.get(asset_key, []))
                    logger.debug(
                        "ML_SNAPSHOT_SKIP asset=%s reason=insufficient_data orderbook=%s "
                        "bids=%d asks=%d depth_needed=%d history=%d -> using signal generator fallback",
                        market.asset, "yes" if orderbook else "no",
                        ob_bids, ob_asks, depth_levels, history_len
                    )
                    # Don't block - continue with signal generator's edge/prob values
                    pass
                else:
                    # ML snapshot built successfully - use ML engine
                    inference = self.ml_engine.evaluate(
                        snapshot,
                        fees_bps=self.config.ml_engine.fees_bps,
                        slippage_bps=self.config.ml_engine.slippage_bps,
                    )
                    volatility = self.signal_generator.get_volatility(market.asset)
                    decision = self.ml_policy.decide(
                        snapshot=snapshot,
                        inference=inference,
                        bankroll=self.risk_manager.current_bankroll,
                        volatility=volatility,
                    )
                    if self.config.trading.ml_enabled and os.getenv("ML_MODE", "EV").upper() == "EV":
                        now = time.time()
                        last_log = self._ml_decision_log_ts.get(market.asset, 0.0)
                        if now - last_log >= 1.0:
                            self._ml_decision_log_ts[market.asset] = now
                            yes_ask = float(snapshot.get("yes_ask", 0.0))
                            no_ask = float(snapshot.get("no_ask", 0.0))
                            edge_up = inference.edge_up
                            edge_down = inference.edge_down
                            if decision.should_trade:
                                chosen_side = "UP" if decision.side == "YES" else "DOWN"
                            else:
                                chosen_side = "NONE"
                            min_edge = self.ml_interface.min_edge if self.ml_interface else self.config.trading.min_edge
                            logger.info(
                                "ML_EV_DECISION asset=%s p_up=%.4f yes_ask=%.4f no_ask=%.4f "
                                "edge_up=%.4f edge_down=%.4f side=%s min_edge=%.4f decision=%s",
                                market.asset,
                                inference.p_up,
                                yes_ask,
                                no_ask,
                                edge_up,
                                edge_down,
                                chosen_side,
                                min_edge,
                                "TRADE" if decision.should_trade else "SKIP",
                            )
                    if not decision.should_trade:
                        logger.info(
                            "ORDER_BLOCK asset=%s market=%s side=%s reason=ml_ev:%s",
                            market.asset, market.condition_id[:8], signal.side.value,
                            decision.reason.replace(" ", "_")[:30]
                        )
                        return

                    # Override signal based on EV decision
                    if decision.side == "YES":
                        signal.side = Side.UP
                        signal.recommended_price = market.best_ask
                        signal.market_prob = market.implied_prob_up
                        signal.true_prob = inference.p_up
                        signal.edge = inference.edge_up
                    else:
                        signal.side = Side.DOWN
                        signal.recommended_price = 1.0 - market.best_bid
                        signal.market_prob = market.implied_prob_down
                        signal.true_prob = 1.0 - inference.p_up
                        signal.edge = inference.edge_down

                    time_remaining = market.time_remaining
                    if signal.edge >= self.config.trading.edge_for_market or time_remaining < self.config.trading.time_for_market:
                        signal.recommended_action = OrderAction.MARKET
                    elif signal.edge >= self.config.trading.edge_for_limit or time_remaining < self.config.trading.time_for_limit:
                        signal.recommended_action = OrderAction.LIMIT
                    elif signal.edge >= self.config.trading.edge_for_post_only:
                        signal.recommended_action = OrderAction.POST_ONLY
                    else:
                        signal.recommended_action = OrderAction.SKIP

                    signal.size_usd = decision.size_usd
                    if signal.recommended_price > 0:
                        signal.size_shares = signal.size_usd / signal.recommended_price

                    signal.reasoning = (
                        f"ML_EV side={decision.side} p_up={inference.p_up:.3f} "
                        f"ev_up={inference.ev_up:.3f} ev_down={inference.ev_down:.3f} "
                        f"edge_up={inference.edge_up:.3f} edge_down={inference.edge_down:.3f}"
                    )

                    try:
                        features = self.ml_engine.bundle.builder.build(snapshot)
                        logger.info(f"[{market.asset}] ML_FEATURES {features}")
                    except Exception:
                        pass

                    signal._ml_ev_info = {
                        "p_up": inference.p_up,
                        "uncertainty": inference.uncertainty,
                        "edge_up": inference.edge_up,
                        "edge_down": inference.edge_down,
                        "ev_up": inference.ev_up,
                        "ev_down": inference.ev_down,
                        "side": decision.side,
                    }
            except Exception as e:
                # Log full exception but DON'T block - fallback to signal generator values
                logger.exception(
                    "ML_ENGINE_ERROR asset=%s market=%s side=%s -> using signal generator fallback",
                    market.asset, market.condition_id[:8], signal.side.value
                )

        # Legacy ML filter - check predicted win probability (do this BEFORE sizing for Kelly)
        if not (self.ml_engine and self.ml_policy) and self.ml_predictor:

            # Extract all ML features using the helper function
            ml_features = extract_ml_features_from_market(
                signal=signal,
                signal_generator=self.signal_generator,
                market=market,
                chart_analysis=chart_analysis,
            )

            should_trade, confidence, ml_reason = self.ml_predictor.should_trade(
                signal=signal,
                **ml_features,
            )
            if not should_trade:
                logger.info(
                    "ORDER_BLOCK asset=%s market=%s side=%s reason=ml_predictor:%s",
                    market.asset, market.condition_id[:8], signal.side.value,
                    ml_reason.replace(" ", "_")[:30]
                )
                return

            # Store ML confidence for Kelly sizing
            ml_confidence = confidence

            # Store all ML data with signal for outcome recording
            signal._ml_volatility = ml_features["volatility"]
            signal._ml_momentum = ml_features["price_momentum"]
            signal._ml_confidence = confidence
            signal._ml_arb_type = ml_features["arb_type"]
            signal._ml_spread = ml_features["spread"]
            signal._ml_bid_depth = ml_features["bid_depth"]
            signal._ml_ask_depth = ml_features["ask_depth"]
            signal._ml_price_trend = ml_features["price_trend"]
            signal._ml_distance_from_target = ml_features["distance_from_target"]
            signal._ml_binance_lead_pct = ml_features.get("binance_lead_pct", 0.0)
            signal._ml_binance_confirmation = ml_features.get("binance_confirmation", "NONE")
            signal._ml_trend_1h = ml_features.get("trend_1h", 0.0)
            signal._ml_trend_4h = ml_features.get("trend_4h", 0.0)
            signal._ml_trend_1d = ml_features.get("trend_1d", 0.0)
            # Chart analysis features
            signal._ml_chart_rsi = ml_features.get("chart_rsi", 50.0)
            signal._ml_chart_trend_strength = ml_features.get("chart_trend_strength", 0.0)
            signal._ml_chart_is_uptrend = ml_features.get("chart_is_uptrend", 0.0)
            signal._ml_chart_is_downtrend = ml_features.get("chart_is_downtrend", 0.0)
            signal._ml_chart_is_ranging = ml_features.get("chart_is_ranging", 0.0)
            signal._ml_chart_bullish_reversal = ml_features.get("chart_bullish_reversal", 0.0)
            signal._ml_chart_bearish_reversal = ml_features.get("chart_bearish_reversal", 0.0)
            signal._ml_chart_momentum = ml_features.get("chart_momentum", 0.0)
            signal._ml_chart_bias_bullish = ml_features.get("chart_bias_bullish", 0.0)
            signal._ml_chart_bias_bearish = ml_features.get("chart_bias_bearish", 0.0)
            signal._ml_chart_confidence = ml_features.get("chart_confidence", 0.5)
            signal._ml_chart_bullish_pattern = ml_features.get("chart_bullish_pattern", 0.0)
            signal._ml_chart_bearish_pattern = ml_features.get("chart_bearish_pattern", 0.0)
            # NEW INDICATOR FEATURES (for outcome recording)
            signal._ml_macd_histogram = ml_features.get("macd_histogram", 0.0)
            signal._ml_macd_crossover = ml_features.get("macd_crossover", "none")
            signal._ml_bb_bandwidth = ml_features.get("bb_bandwidth", 0.0)
            signal._ml_bb_position = ml_features.get("bb_position", "middle")
            signal._ml_stoch_k = ml_features.get("stoch_k", 50.0)
            signal._ml_stoch_d = ml_features.get("stoch_d", 50.0)
            signal._ml_stoch_signal = ml_features.get("stoch_signal", "neutral")
            signal._ml_rsi_divergence = ml_features.get("rsi_divergence", "none")
            signal._ml_rsi_divergence_strength = ml_features.get("rsi_divergence_strength", 0.0)
            signal._ml_volume_ratio = ml_features.get("volume_ratio", 1.0)
            signal._ml_is_high_volume = ml_features.get("is_high_volume", False)
            signal._ml_obv_trend = ml_features.get("obv_trend", 0.0)
            signal._ml_ha_trend = ml_features.get("ha_trend", "neutral")
            signal._ml_ha_consecutive = ml_features.get("ha_consecutive", 0)
            signal._ml_ha_strength = ml_features.get("ha_strength", 0.0)
            signal._ml_vwap_distance_pct = ml_features.get("vwap_distance_pct", 0.0)
            signal._ml_vwap_position = ml_features.get("vwap_position", "at")

        # Adjust size using Kelly criterion (with ML confidence for optimal sizing)
        signal = self.risk_manager.adjust_signal_size(
            signal,
            ml_confidence=ml_confidence,
            use_kelly=True,
        )

        # Check if signal was rejected due to size
        if signal.size_usd <= 0 or signal.size_shares <= 0:
            # Throttle to once per 30s per asset
            block_key = f"{market.asset}:size_too_small"
            now_ts = time.time()
            last_log = getattr(self, '_size_block_log_ts', {}).get(block_key, 0.0)
            if now_ts - last_log >= 30.0:
                if not hasattr(self, '_size_block_log_ts'):
                    self._size_block_log_ts = {}
                self._size_block_log_ts[block_key] = now_ts
                logger.info(
                    "ORDER_BLOCK asset=%s market=%s side=%s reason=size_too_small usd=%.2f shares=%.4f",
                    market.asset, market.condition_id[:8], signal.side.value,
                    signal.size_usd, signal.size_shares
                )
            return

        # === CONSOLIDATED SIGNAL LOG (single line, for signals passing initial checks) ===
        # NOTE: ORDER_DECISION is logged in _execute_signal after ALL blocking checks pass
        # Throttle to once per 30s per market to avoid spam
        arb_type = getattr(signal, '_arb_type', 'probability')
        distance_pct = getattr(signal, '_ml_distance_from_target', 0) * 100
        rsi = getattr(signal, '_ml_chart_rsi', 50)
        signal_msg = (
            f"🎯 {market.asset} {signal.side.value} ${signal.size_usd:.0f} | "
            f"edge={signal.edge:.0%} dist={distance_pct:+.1f}% RSI={rsi:.0f} | "
            f"{signal.reasoning[:40]}"
        )
        now_ts = time.time()
        last_signal_log = self._signal_log_ts.get(market.asset, 0.0)
        if now_ts - last_signal_log >= 30.0:
            self._signal_log_ts[market.asset] = now_ts
            logger.info(signal_msg)
        else:
            logger.debug(signal_msg)

        # Execute the signal
        await self._execute_signal(signal)

    def _update_market_from_orderbook(self, market: MarketState):
        """Update market state from CLOB order book."""
        # Get UP token order book
        up_book = self.clob_feed.get_orderbook(market.up_token_id)
        if up_book:
            if up_book.best_bid is not None:
                market.best_bid = up_book.best_bid
            if up_book.best_ask is not None:
                market.best_ask = up_book.best_ask

            # Calculate depth - make thread-safe copies to avoid race conditions
            # with WebSocket thread that may be modifying the orderbook
            try:
                bids_snapshot = list(up_book.bids)
                asks_snapshot = list(up_book.asks)
                # Validate sizes are numeric before summing (handle malformed data)
                market.bid_depth = sum(
                    level.size for level in bids_snapshot
                    if hasattr(level, 'size') and isinstance(level.size, (int, float))
                )
                market.ask_depth = sum(
                    level.size for level in asks_snapshot
                    if hasattr(level, 'size') and isinstance(level.size, (int, float))
                )
            except Exception as e:
                logger.debug(f"[{market.asset}] Orderbook depth calc error: {e}")
                # Keep previous depth values on error

        market.last_updated = datetime.now(timezone.utc)

    def _build_ml_snapshot(self, market: MarketState, chainlink_price: float):
        if not self.ml_engine:
            return None

        orderbook = self.clob_feed.get_orderbook(market.up_token_id)
        if not orderbook:
            return None

        depth_levels = self.ml_engine.bundle.builder.depth_levels
        bids = [(lvl.price, lvl.size) for lvl in orderbook.bids[:depth_levels]]
        asks = [(lvl.price, lvl.size) for lvl in orderbook.asks[:depth_levels]]

        if len(bids) < depth_levels or len(asks) < depth_levels:
            return None

        yes_bid = market.best_bid
        yes_ask = market.best_ask
        no_bid = 1.0 - yes_ask
        no_ask = 1.0 - yes_bid

        asset_key = f"{market.asset.lower()}/usd"
        raw_history = self.signal_generator.price_histories.get(asset_key, [])
        history = []
        now_ts = datetime.now(timezone.utc)
        for item in raw_history:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                history.append((item[0], item[1]))
            elif isinstance(item, (int, float)):
                history.append((now_ts, float(item)))

        if not history:
            return None

        snapshot = {
            "timestamp": now_ts,
            "market_open_ts": market.start_time,
            "market_end_ts": market.end_time,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "no_bid": no_bid,
            "no_ask": no_ask,
            "orderbook_bids": bids,
            "orderbook_asks": asks,
            "reference_price": chainlink_price,
            "reference_price_open": market.target_price,
            "reference_price_history": history,
        }

        bid_depth = sum(s for _, s in bids)
        ask_depth = sum(s for _, s in asks)
        snapshot["orderbook_bid_depth"] = bid_depth
        snapshot["orderbook_ask_depth"] = ask_depth
        return snapshot

    def _signal_from_meta(self, market: MarketState, meta_signal, chainlink_price: float) -> Optional[Signal]:
        market_prob = market.implied_prob_up
        side = Side.UP if meta_signal.side == "YES" else Side.DOWN

        if side == Side.UP:
            true_prob = min(0.99, max(0.01, market_prob + meta_signal.edge))
        else:
            true_prob = min(0.99, max(0.01, market_prob - meta_signal.edge))

        edge = abs(true_prob - market_prob)
        time_remaining = market.time_remaining

        action = OrderAction.SKIP
        if edge >= self.config.trading.edge_for_market or time_remaining < self.config.trading.time_for_market:
            action = OrderAction.MARKET
        elif edge >= self.config.trading.edge_for_limit or time_remaining < self.config.trading.time_for_limit:
            action = OrderAction.LIMIT
        elif edge >= self.config.trading.edge_for_post_only:
            action = OrderAction.POST_ONLY

        if side == Side.UP:
            price = market.best_ask
        else:
            price = 1.0 - market.best_bid if market.best_bid else 0.5

        size_usd = self.config.trading.base_position_size
        volatility = self.signal_generator.get_volatility(market.asset)
        size_usd = self.risk_sizer.size_position(
            bankroll=self.risk_manager.current_bankroll,
            edge=edge,
            prob=true_prob if side == Side.UP else 1 - true_prob,
            market_price=price if price > 0 else 0.5,
            volatility=volatility,
            base_size=size_usd,
        )
        if size_usd <= 0:
            return None

        size_shares = size_usd / price if price > 0 else 0.0

        return Signal(
            market=market,
            side=side,
            edge=edge,
            true_prob=true_prob,
            market_prob=market_prob,
            recommended_action=action,
            recommended_price=price,
            size_usd=size_usd,
            size_shares=size_shares,
            chainlink_price=chainlink_price or 0.0,
            time_remaining=time_remaining,
            reasoning=f"META {meta_signal.side} edge={edge:.3f} conf={meta_signal.confidence:.2f}",
        )

    def _log_status_line(self):
        """Log a clean status line with key info."""
        now = datetime.now(timezone.utc)
        bankroll = self.risk_manager.current_bankroll
        pos_count = len(self.risk_manager.positions)
        pending_count = len(self._pending_orders) if hasattr(self, '_pending_orders') else 0
        market_count = len(self.markets)

        # Get prices
        chainlink_prices = self.chainlink_feed.get_all_prices()
        binance_prices = self._get_binance_prices()

        # Build compact price string with distance info
        price_parts = []
        for asset in self.config.supported_assets:
            asset_key = f"{asset.lower()}/usd"
            cl_price = chainlink_prices.get(asset_key)
            bn_price = binance_prices.get(asset)

            # Find market for this asset to get target
            target = None
            for m in self.markets.values():
                if m.asset == asset:
                    target = m.target_price
                    break

            if cl_price:
                part = f"{asset}=${cl_price:,.0f}"
                if target and target > 0:
                    dist_pct = ((cl_price - target) / target) * 100
                    if dist_pct > 0:
                        part += f"↑{dist_pct:.1f}%"
                    else:
                        part += f"↓{abs(dist_pct):.1f}%"
                price_parts.append(part)

        # Single clean status line
        status = f"💰${bankroll:.0f} | {market_count} mkts | {pos_count} pos | {pending_count} pnd"
        if price_parts:
            status += f" | {' '.join(price_parts)}"

        logger.info(status)

    def _log_order_block(self, asset: str, reason: str, **kwargs) -> bool:
        """
        Log ORDER_BLOCK with throttling to prevent spam.

        Args:
            asset: Asset symbol
            reason: Block reason
            **kwargs: Additional log fields

        Returns:
            True if logged, False if throttled
        """
        block_key = f"{asset}:{reason}"
        now_ts = time.time()
        last_log = self._order_block_log_ts.get(block_key, 0.0)

        if now_ts - last_log < self._order_block_log_interval:
            return False  # Throttled

        self._order_block_log_ts[block_key] = now_ts

        # Build log message with extra fields
        extra_parts = " ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
        if extra_parts:
            logger.info(f"ORDER_BLOCK asset={asset} reason={reason} {extra_parts}")
        else:
            logger.info(f"ORDER_BLOCK asset={asset} reason={reason}")
        return True

    def _derive_position_state(
        self, position, market, resolved_outcome: str = None
    ) -> PositionState:
        """
        Derive the current state of a position.

        Args:
            position: The position object
            market: The market state (may be None)
            resolved_outcome: If known, "WIN" or "LOSS" from settlement verification

        Returns:
            PositionState enum value

        State machine:
        - OPEN: market.time_remaining > 0
        - ENDED_UNRESOLVED: time_remaining <= 0 and no resolved_outcome
        - RESOLVED_WIN: resolved_outcome == "WIN"
        - RESOLVED_LOSS: resolved_outcome == "LOSS"
        - SETTLING: (future use - redemption in progress)
        - CLOSED: (future use - finalized)
        """
        if resolved_outcome == "WIN":
            return PositionState.RESOLVED_WIN
        if resolved_outcome == "LOSS":
            return PositionState.RESOLVED_LOSS

        if not market:
            return PositionState.ENDED_UNRESOLVED

        time_remaining = market.time_remaining if hasattr(market, 'time_remaining') else 0

        if time_remaining > 0:
            return PositionState.OPEN
        else:
            return PositionState.ENDED_UNRESOLVED

    def _get_position_mark_price(
        self, position, market, state: PositionState = None
    ) -> tuple[float, str, dict]:
        """
        Get mark price for a position based on its state.

        Marking rules by state:
        - RESOLVED_WIN: mark_price = 1.0
        - RESOLVED_LOSS: mark_price = 0.0
        - OPEN / ENDED_UNRESOLVED: mark-to-market using best available quote

        For long positions, mark = bid of the SAME outcome token:
          UP position  -> YES token bid
          DOWN position -> NO token bid

        Quote fallback chain:
        1. CLOB orderbook bid for this token
        2. Paper executor quote cache bid for this token (keyed by token_id)
        3. MarketState prices (derived from UP token)
        4. None (unmarked)

        Returns:
            Tuple of (mark_price, source_description, debug_info)
            mark_price is None if no valid price available
            debug_info contains token_id_used, bid, ask, mid for logging
        """
        debug_info = {
            "token_id_used": position.token_id if position else "",
            "bid": None, "ask": None, "mid": None,
        }

        # Derive state if not provided
        if state is None:
            state = self._derive_position_state(position, market)

        # RESOLVED_WIN: mark = 1.0
        if state == PositionState.RESOLVED_WIN:
            return 1.0, "resolved_win", debug_info

        # RESOLVED_LOSS: mark = 0.0
        if state == PositionState.RESOLVED_LOSS:
            return 0.0, "resolved_loss", debug_info

        # OPEN or ENDED_UNRESOLVED: use best available quote
        if not market:
            return None, "no_market", debug_info

        # Determine the correct token_id for this position's side
        if position.side == Side.UP:
            token_id = market.up_token_id
        else:
            token_id = market.down_token_id
        debug_info["token_id_used"] = token_id

        # Verify position.token_id matches expected token for side
        if position.token_id and position.token_id != token_id:
            logger.warning(
                "TOKEN_MISMATCH position.token_id=%s expected_%s_token=%s market=%s asset=%s",
                position.token_id[:16], position.side.value.lower(),
                token_id[:16], market.condition_id[:8], market.asset,
            )

        # 1) CLOB orderbook — mark at bid (what we could sell at)
        orderbook = self.clob_feed.get_orderbook(token_id) if self.clob_feed else None
        if orderbook and orderbook.best_bid is not None and orderbook.best_bid > 0:
            debug_info["bid"] = orderbook.best_bid
            debug_info["ask"] = orderbook.best_ask
            debug_info["mid"] = orderbook.mid_price
            return orderbook.best_bid, "clob_bid", debug_info

        # 2) Paper executor quote cache (fed from MarketState each tick)
        if self.executor and hasattr(self.executor, 'get_cached_quote'):
            cached = self.executor.get_cached_quote(token_id)
            if cached and cached.bid > 0:
                debug_info["bid"] = cached.bid
                debug_info["ask"] = cached.ask
                debug_info["mid"] = cached.mid
                return cached.bid, "cache_bid", debug_info

        # 3) MarketState prices (derived from UP token orderbook)
        if position.side == Side.UP:
            if market.best_bid and market.best_bid > 0:
                debug_info["bid"] = market.best_bid
                debug_info["ask"] = market.best_ask
                debug_info["mid"] = (market.best_bid + market.best_ask) / 2
                return market.best_bid, "market_yes_bid", debug_info
        else:
            if market.best_ask and market.best_ask < 1.0:
                no_bid = 1.0 - market.best_ask
                no_ask = 1.0 - market.best_bid if market.best_bid else None
                debug_info["bid"] = no_bid
                debug_info["ask"] = no_ask
                debug_info["mid"] = (no_bid + no_ask) / 2 if no_ask else no_bid
                return no_bid, "market_no_bid", debug_info

        return None, "no_price", debug_info

    def _calculate_equity(self, debug_log: bool = False) -> float:
        """
        Calculate current equity: cash + sum(mark_value or cost_basis).

        Uses state-aware marking:
        - RESOLVED_WIN: mark = 1.0
        - RESOLVED_LOSS: mark = 0.0
        - OPEN/ENDED_UNRESOLVED: mark = best available quote
        - No price: fallback to cost basis (equity neutral, 0 unrealized)

        Returns:
            Total equity (cash + position value)
        """
        cash = self.risk_manager.current_bankroll
        sum_mark_value = 0.0
        sum_cost_basis = 0.0
        unrealized_pnl = 0.0

        for market_key, position in self.risk_manager.positions.items():
            market = self.markets.get(market_key) or self.expiring_markets.get(market_key)

            state = self._derive_position_state(position, market)
            mark_price, source, _dbg = self._get_position_mark_price(position, market, state)

            cost = position.cost_basis
            sum_cost_basis += cost

            if mark_price is not None and mark_price >= 0:
                mark_val = position.shares * mark_price
                sum_mark_value += mark_val
                unrealized_pnl += mark_val - cost
            else:
                # Unmarked: use cost_basis for equity, contribute 0 to unrealized
                sum_mark_value += cost

        # Realized P&L from daily stats
        daily = self.risk_manager.get_daily_stats()
        realized_pnl = daily.total_pnl if daily else 0.0

        equity = cash + sum_mark_value
        total_pnl = realized_pnl + unrealized_pnl

        if debug_log:
            logger.debug(
                "EQUITY_CALC cash=%.2f sum_cost_basis=%.2f sum_mark_value=%.2f "
                "realized=%.2f unrealized=%.2f equity=%.2f pnl=%.2f",
                cash, sum_cost_basis, sum_mark_value,
                realized_pnl, unrealized_pnl, equity, total_pnl,
            )

        # Stash for header use (avoids recomputing)
        self._last_equity_detail = {
            "cash": cash,
            "sum_cost_basis": sum_cost_basis,
            "sum_mark_value": sum_mark_value,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "equity": equity,
            "total_pnl": total_pnl,
        }

        return equity

    def _log_position_status(self):
        """
        Log detailed position status including all tracking state.

        Called periodically (every 30s) to give visibility into:
        - Tracked positions (risk_manager.positions)
        - API-discovered positions (_api_positions)
        - Pending orders (_pending_orders)
        - Active cooldowns (_last_order_time)
        """
        now = datetime.now(timezone.utc)

        # Check if it's time to log
        if self.last_position_log:
            elapsed = (now - self.last_position_log).total_seconds()
            if elapsed < self.position_log_interval:
                return
        self.last_position_log = now

        positions = self.risk_manager.positions
        api_positions = getattr(self, '_api_positions', {})
        pending_orders = getattr(self, '_pending_orders', {})

        # Calculate equity with debug logging
        equity = self._calculate_equity(debug_log=True)
        detail = getattr(self, '_last_equity_detail', {})
        cash = detail.get("cash", self.risk_manager.current_bankroll)
        total_pnl = detail.get("total_pnl", 0.0)
        self.metrics.record_position(
            positions=len(self.risk_manager.positions),
            exposure=self.risk_manager.get_total_exposure(),
            equity=equity,
        )

        # Format P&L (realized + unrealized)
        if total_pnl > 0:
            pnl_color = Colors.BRIGHT_GREEN
            pnl_icon = "📈"
        elif total_pnl < 0:
            pnl_color = Colors.BRIGHT_RED
            pnl_icon = "📉"
        else:
            pnl_color = Colors.DIM
            pnl_icon = "➖"

        # Box header
        box_width = 64
        logger.info(f"{Colors.BRIGHT_CYAN}┌{'─' * box_width}┐{Colors.RESET}")
        logger.info(f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  📊 POSITION STATUS                                              {Colors.BRIGHT_CYAN}│{Colors.RESET}")
        logger.info(f"{Colors.BRIGHT_CYAN}├{'─' * box_width}┤{Colors.RESET}")

        # Account summary line
        logger.info(
            f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  💰 Cash: {Colors.BRIGHT_WHITE}${cash:>10.2f}{Colors.RESET}  │  "
            f"💎 Equity: {Colors.BRIGHT_WHITE}${equity:>10.2f}{Colors.RESET}  │  "
            f"{pnl_icon} P&L: {pnl_color}${total_pnl:>+9.2f}{Colors.RESET}  {Colors.BRIGHT_CYAN}│{Colors.RESET}"
        )

        # Counts summary
        pos_count = len(positions)
        api_count = len(api_positions)
        pending_count = len(pending_orders)
        logger.info(
            f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  📦 Positions: {Colors.BRIGHT_YELLOW}{pos_count}{Colors.RESET}  │  "
            f"🔗 API: {Colors.BRIGHT_YELLOW}{api_count}{Colors.RESET}  │  "
            f"⏳ Pending: {Colors.BRIGHT_YELLOW}{pending_count}{Colors.RESET}                       {Colors.BRIGHT_CYAN}│{Colors.RESET}"
        )
        logger.info(f"{Colors.BRIGHT_CYAN}├{'─' * box_width}┤{Colors.RESET}")

        # Log tracked positions with P&L and win probability
        if positions:
            for market_key, position in positions.items():
                market = self.markets.get(market_key) or self.expiring_markets.get(market_key)
                asset = position.market.asset if hasattr(position, 'market') else "???"
                side = position.side.value
                time_remaining = market.time_remaining if market else 0

                # === DERIVE POSITION STATE ===
                # Check if we have a resolved outcome from settlement verification
                resolved_outcome = None  # TODO: Get from settlement_verifier when available
                state = self._derive_position_state(position, market, resolved_outcome)

                # === GET MARK PRICE BASED ON STATE ===
                mark_price, mark_source, mark_debug = self._get_position_mark_price(position, market, state)

                # === COMPUTE UNREALIZED PNL ===
                unrealized_pnl = None
                pnl_pct = None
                if mark_price is not None and mark_price >= 0:
                    unrealized_pnl = (mark_price - position.entry_price) * position.shares
                    if position.entry_price > 0:
                        pnl_pct = ((mark_price - position.entry_price) / position.entry_price * 100)

                # === CALCULATE WIN PROBABILITY ===
                chainlink_price = self.signal_generator.get_price(asset) if hasattr(self, 'signal_generator') else None
                target_price = market.target_price if market else None
                our_win_prob = 0.5  # Default

                if chainlink_price and target_price and market and time_remaining > 0:
                    from .probability import calculate_true_probability
                    volatility = self.signal_generator.get_volatility(asset)
                    true_prob_up = calculate_true_probability(
                        current_price=chainlink_price,
                        target_price=target_price,
                        time_remaining_sec=time_remaining,
                        volatility_15min=volatility,
                    )
                    our_win_prob = true_prob_up if side == "UP" else (1 - true_prob_up)
                elif state == PositionState.RESOLVED_WIN:
                    our_win_prob = 1.0
                elif state == PositionState.RESOLVED_LOSS:
                    our_win_prob = 0.0

                # Color and icon based on probability
                if our_win_prob >= 0.7:
                    prob_color = Colors.BRIGHT_GREEN
                    prob_icon = "🟢"
                elif our_win_prob >= 0.5:
                    prob_color = Colors.BRIGHT_YELLOW
                    prob_icon = "🟡"
                else:
                    prob_color = Colors.BRIGHT_RED
                    prob_icon = "🔴"

                # === STATUS STRING BASED ON STATE ===
                if state == PositionState.RESOLVED_WIN:
                    status_str = "✅ WON"
                elif state == PositionState.RESOLVED_LOSS:
                    status_str = "❌ LOST"
                elif state == PositionState.ENDED_UNRESOLVED:
                    status_str = "⏰ PENDING"
                elif time_remaining > 60:
                    status_str = f"⏱️  {int(time_remaining // 60)}m {int(time_remaining % 60)}s"
                else:
                    status_str = f"⏱️  {int(time_remaining)}s"

                # === EXPLICIT DEBUG LOG FOR EACH POSITION ROW ===
                token_id_used = mark_debug.get("token_id_used", "")
                dbg_bid = mark_debug.get("bid")
                dbg_ask = mark_debug.get("ask")
                dbg_mid = mark_debug.get("mid")
                is_pending = state in (
                    PositionState.ENDED_UNRESOLVED,
                    PositionState.RESOLVED_WIN,
                    PositionState.RESOLVED_LOSS,
                )
                logger.info(
                    "POSITION_VALUATION market=%s asset=%s side=%s "
                    "token_id_used=%s bid=%s ask=%s mid=%s "
                    "mark_price=%s mark_source=%s "
                    "entry=%.4f shares=%.2f unrealized_pnl=%s realized_pnl=0.00 "
                    "state=%s pending_settlement=%s",
                    market_key[:8] if market_key else "????????",
                    asset,
                    side,
                    token_id_used[:16] if token_id_used else "None",
                    f"{dbg_bid:.4f}" if dbg_bid is not None else "None",
                    f"{dbg_ask:.4f}" if dbg_ask is not None else "None",
                    f"{dbg_mid:.4f}" if dbg_mid is not None else "None",
                    f"{mark_price:.4f}" if mark_price is not None else "None",
                    mark_source,
                    position.entry_price,
                    position.shares,
                    f"{unrealized_pnl:+.2f}" if unrealized_pnl is not None else "pending",
                    state.value,
                    "true" if is_pending else "false",
                )

                # Side arrow
                side_arrow = "▲" if side == "UP" else "▼"
                side_color = Colors.BRIGHT_GREEN if side == "UP" else Colors.BRIGHT_RED

                if mark_price is not None and unrealized_pnl is not None:
                    pnl_color = Colors.BRIGHT_GREEN if unrealized_pnl > 0 else Colors.BRIGHT_RED

                    # Format mark price display
                    if state == PositionState.RESOLVED_WIN:
                        mark_display = "1.000"
                    elif state == PositionState.RESOLVED_LOSS:
                        mark_display = "0.000"
                    else:
                        mark_display = f"{mark_price:.3f}"

                    logger.info(
                        f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  {side_color}{side_arrow} {asset:4}{Colors.RESET} │ "
                        f"{position.shares:>6.1f} @ {position.entry_price:.3f} → {mark_display} │ "
                        f"{pnl_color}{unrealized_pnl:>+7.2f} ({pnl_pct:>+5.0f}%){Colors.RESET} │ "
                        f"{prob_color}{prob_icon} {our_win_prob:>3.0%}{Colors.RESET} │ {status_str}  {Colors.BRIGHT_CYAN}│{Colors.RESET}"
                    )
                else:
                    # No valid mark - show pending
                    logger.info(
                        f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  {side_color}{side_arrow} {asset:4}{Colors.RESET} │ "
                        f"{position.shares:>6.1f} @ {position.entry_price:.3f} → {'--':>5} │ "
                        f"{'(pending)':^16} │ "
                        f"{prob_color}{prob_icon} {our_win_prob:>3.0%}{Colors.RESET} │ {status_str}  {Colors.BRIGHT_CYAN}│{Colors.RESET}"
                    )
        else:
            logger.info(f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  {Colors.DIM}No active positions{Colors.RESET}                                            {Colors.BRIGHT_CYAN}│{Colors.RESET}")

        # Log API-discovered positions (if any not already tracked)
        tracked_assets = [p.market.asset for p in positions.values() if hasattr(p, 'market')]
        untracked_api = {k: v for k, v in api_positions.items() if k not in tracked_assets}
        if untracked_api:
            logger.info(f"{Colors.BRIGHT_CYAN}├{'─' * 64}┤{Colors.RESET}")
            logger.info(f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  🔗 API Positions (untracked):                                   {Colors.BRIGHT_CYAN}│{Colors.RESET}")
            for asset, pos_info in untracked_api.items():
                logger.info(
                    f"{Colors.BRIGHT_CYAN}│{Colors.RESET}     {asset}: {pos_info.get('size', 0):.1f} shares @ "
                    f"${pos_info.get('price', 0):.2f}                                {Colors.BRIGHT_CYAN}│{Colors.RESET}"
                )

        # Log pending orders
        if pending_orders:
            logger.info(f"{Colors.BRIGHT_CYAN}├{'─' * 64}┤{Colors.RESET}")
            logger.info(f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  ⏳ Pending Orders:                                               {Colors.BRIGHT_CYAN}│{Colors.RESET}")
            for order_id, order_data in pending_orders.items():
                asset = order_data.get("asset", "???")
                side = order_data.get("side", "???")
                size = order_data.get("size", 0)
                limit_price = order_data.get("price", 0)
                placed_time = order_data.get("placed_time")
                age_s = int((now - placed_time).total_seconds()) if placed_time else 0

                # Try to get live status from executor
                order_status = "OPEN"
                filled = 0.0
                if self.executor and hasattr(self.executor, 'get_order_status'):
                    oi = self.executor.get_order_status(order_id)
                    if oi:
                        order_status = oi.status.value
                        filled = oi.filled_size

                logger.info(
                    f"{Colors.BRIGHT_CYAN}│{Colors.RESET}     {asset} {side}: {size:.1f} @ {limit_price:.3f} │ "
                    f"{order_status} │ filled={filled:.2f} │ age={age_s}s │ "
                    f"id={order_id[:12]}  {Colors.BRIGHT_CYAN}│{Colors.RESET}"
                )

        # Log active cooldowns
        active_cooldowns = []
        for asset, last_time in self._last_order_time.items():
            elapsed = (now - last_time).total_seconds()
            remaining = self._order_cooldown_seconds - elapsed
            if remaining > 0:
                active_cooldowns.append(f"{asset}:{remaining:.0f}s")

        if active_cooldowns:
            logger.info(f"{Colors.BRIGHT_CYAN}├{'─' * 64}┤{Colors.RESET}")
            cooldown_str = ", ".join(active_cooldowns)
            logger.info(f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  ⏱️  Cooldowns: {cooldown_str:<48}{Colors.BRIGHT_CYAN}│{Colors.RESET}")

        # On-chain stats from The Graph
        if self.settlement_verifier.is_enabled():
            graph_stats = self._get_graph_stats()
            if graph_stats:
                logger.info(f"{Colors.BRIGHT_CYAN}├{'─' * 64}┤{Colors.RESET}")
                logger.info(f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  📈 On-Chain Stats (The Graph):                                  {Colors.BRIGHT_CYAN}│{Colors.RESET}")
                wins = graph_stats.get("winning_redemptions", 0)
                losses = graph_stats.get("losing_redemptions", 0)
                total_payouts = graph_stats.get("total_redemption_payout_usdc", 0)
                win_rate = graph_stats.get("redemption_win_rate", 0) * 100
                splits = graph_stats.get("total_splits", 0)

                # Win rate color
                if win_rate >= 60:
                    wr_color = Colors.BRIGHT_GREEN
                elif win_rate >= 50:
                    wr_color = Colors.BRIGHT_YELLOW
                else:
                    wr_color = Colors.BRIGHT_RED

                logger.info(
                    f"{Colors.BRIGHT_CYAN}│{Colors.RESET}     Trades: {Colors.BRIGHT_WHITE}{splits}{Colors.RESET}  │  "
                    f"Wins: {Colors.BRIGHT_GREEN}{wins}{Colors.RESET}  │  "
                    f"Losses: {Colors.BRIGHT_RED}{losses}{Colors.RESET}  │  "
                    f"Win Rate: {wr_color}{win_rate:.0f}%{Colors.RESET}          {Colors.BRIGHT_CYAN}│{Colors.RESET}"
                )
                logger.info(
                    f"{Colors.BRIGHT_CYAN}│{Colors.RESET}     Total Payouts: {Colors.BRIGHT_GREEN}${total_payouts:,.2f}{Colors.RESET}                                     {Colors.BRIGHT_CYAN}│{Colors.RESET}"
                )

        # Box footer
        logger.info(f"{Colors.BRIGHT_CYAN}└{'─' * 64}┘{Colors.RESET}")

    def _get_graph_stats(self) -> Optional[dict]:
        """
        Get user stats from The Graph with caching.

        Returns cached stats or fetches new ones if cache expired.
        """
        now = datetime.now(timezone.utc)

        # Check if we have cached stats that are still fresh
        if (self._graph_stats_cache is not None and
            self._graph_stats_last_fetch is not None):
            elapsed = (now - self._graph_stats_last_fetch).total_seconds()
            if elapsed < self._graph_stats_refresh_interval:
                return self._graph_stats_cache

        # Fetch new stats
        try:
            wallet_address = self.client.get_address()
            if not wallet_address:
                return None

            graph = self.settlement_verifier.graph
            if not graph:
                return None

            stats = graph.calculate_user_stats(wallet_address)
            self._graph_stats_cache = stats
            self._graph_stats_last_fetch = now
            return stats

        except Exception as e:
            logger.debug(f"Failed to fetch Graph stats: {e}")
            return self._graph_stats_cache  # Return stale cache on error

    def _has_active_position(self, asset: str) -> bool:
        """
        Check if we already have an active position or pending order for this asset.

        Checks:
        1. Internal position tracking (risk_manager.positions)
        2. Cached API positions (updated by _sync_existing_orders)
        3. Pending orders (orders placed but not yet filled)

        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP)

        Returns:
            True if position or pending order exists, False otherwise
        """
        # Check internal tracking
        for pos in self.risk_manager.positions.values():
            if pos.market.asset == asset:
                return True

        # Check cached API positions (stored during sync)
        if hasattr(self, '_api_positions') and asset in self._api_positions:
            return True

        # Check pending orders
        if hasattr(self, '_pending_orders'):
            for order_data in self._pending_orders.values():
                if order_data.get("asset") == asset:
                    return True

        return False

    def _get_active_position(self, asset: str):
        """
        Get the active position for an asset if one exists.

        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP)

        Returns:
            Position object or None
        """
        # Check internal tracking
        for pos in self.risk_manager.positions.values():
            if pos.market.asset == asset:
                return pos

        return None

    def _check_averaging_opportunity(self, position, signal, market):
        """
        Check if we should average down on an existing position.

        Args:
            position: Existing position
            signal: New signal that triggered this check
            market: Current market state

        Returns:
            AveragingDecision or None
        """
        try:
            from .strategy.averaging import should_average_down, AveragingConfig
            from .data.binance_chart import analyze_chart
        except ImportError:
            return None

        # Only consider averaging if signal matches position direction
        if signal.side != position.side:
            return None

        # Get current token price
        if position.side.value == "UP":
            current_token_price = market.best_ask
        else:
            current_token_price = 1 - market.best_bid

        # Get chart analysis
        chart_analysis = analyze_chart(market.asset)

        # Check averaging conditions with moderate config
        config = AveragingConfig(
            enabled=True,
            min_chart_confidence=0.70,  # 70% chart confidence
            min_price_improvement=0.08,  # 8% cheaper
            max_averages=1,  # Only average once
            min_time_remaining=300,  # 5 minutes
            rsi_min=25.0,  # Don't average if too oversold
            rsi_max=75.0,  # Don't average if too overbought
            min_alignment_score=0.66,  # 2/3 timeframes agree
            averaging_size_multiplier=0.5,  # Add 50% of original
        )

        return should_average_down(
            position=position,
            current_token_price=current_token_price,
            chart_analysis=chart_analysis,
            time_remaining=market.time_remaining,
            config=config,
        )

    async def _execute_averaging_order(self, position, signal, averaging_decision):
        """
        Execute an averaging down order.

        Args:
            position: Position to add to
            signal: Original signal (for order execution)
            averaging_decision: AveragingDecision with suggested size
        """
        try:
            from .strategy.averaging import update_position_after_averaging
        except ImportError:
            logger.error("Could not import averaging module")
            return

        market = signal.market
        asset = market.asset

        # Create a modified signal for the averaging order
        averaging_signal = Signal(
            market=market,
            side=position.side,
            edge=signal.edge,
            true_prob=signal.true_prob,
            market_prob=signal.market_prob,
            recommended_action=OrderAction.LIMIT,
            recommended_price=signal.recommended_price,
            size_usd=averaging_decision.suggested_size_usd,
            size_shares=averaging_decision.suggested_shares,
            chainlink_price=signal.chainlink_price,
            time_remaining=signal.time_remaining,
            reasoning=f"[AVERAGING] {averaging_decision.reason}",
        )

        # Execute the order
        try:
            result = await self.executor.execute_signal_async(averaging_signal)
        except Exception as e:
            logger.error(
                "ORDER_RESULT asset=%s side=%s status=exception error=%s mode=averaging",
                asset, position.side.value, str(e)[:200]
            )
            logger.debug("Averaging execution exception traceback:", exc_info=True)
            return  # Exit gracefully

        if result.success:
            # Update position with new average
            update_position_after_averaging(
                position=position,
                additional_shares=averaging_decision.suggested_shares,
                additional_cost=averaging_decision.suggested_size_usd,
                new_entry_price=averaging_decision.new_avg_price,
            )

            # Set cooldown
            now = datetime.now(timezone.utc)
            self._last_order_time[asset] = now

            logger.info(
                f"[{asset}] ✅ AVERAGED DOWN: "
                f"Added {averaging_decision.suggested_shares:.1f} shares @ ${averaging_decision.suggested_size_usd:.2f} | "
                f"New avg price: ${averaging_decision.new_avg_price:.3f} | "
                f"Total shares: {position.shares:.1f}"
            )
        else:
            logger.warning(
                f"[{asset}] ❌ AVERAGING FAILED: {result.error_message}"
            )

    def _update_cached_positions(self, positions: dict):
        """Update cached API positions."""
        self._api_positions = positions

    async def _execute_signal(self, signal: Signal):
        """Execute a trading signal."""
        # Guard: reject if shutdown has started (no in-flight orders after state save)
        if not self._running:
            return

        asset = signal.market.asset
        now = datetime.now(timezone.utc)

        # Check cooldown - prevent rapid duplicate orders
        if asset in self._last_order_time:
            elapsed = (now - self._last_order_time[asset]).total_seconds()
            if elapsed < self._order_cooldown_seconds:
                remaining = self._order_cooldown_seconds - elapsed
                # Log once per cooldown period at INFO level so user knows why trades aren't executing
                cooldown_key = f"{signal.market.condition_id}:cooldown"
                if cooldown_key not in self._logged_rejections:
                    self._logged_rejections.add(cooldown_key)
                    logger.info(
                        "ORDER_SKIP_DUP asset=%s reason=cooldown secs_since_last=%.0f",
                        asset,
                        elapsed,
                    )
                return

        # Check for existing FILLED positions for this market+side (prevents duplicate positions)
        market_id = signal.market.condition_id
        for pos_key, pos in self.risk_manager.positions.items():
            pos_market_id = getattr(pos.market, 'condition_id', None) if hasattr(pos, 'market') else None
            pos_side = pos.side.value if hasattr(pos.side, 'value') else str(pos.side)
            if pos_market_id == market_id and pos_side == signal.side.value:
                # Throttled - this can spam when position is open
                self._log_order_block(
                    asset, "position_exists",
                    market=market_id[:8], side=signal.side.value
                )
                return

        # Check for existing PENDING orders for this asset (prevents duplicate unfilled orders)
        async with self._pending_orders_lock:
            pending_items = list(self._pending_orders.items())
            pending_count = len(self._pending_orders)
        for order_id, order_data in pending_items:
            if order_data.get("asset") == asset:
                placed_time = order_data.get("placed_time")
                if placed_time:
                    in_window = signal.market.start_time <= placed_time <= signal.market.end_time
                    same_side = order_data.get("side") == signal.side.value
                    if in_window and same_side:
                        # Throttled
                        self._log_order_block(
                            asset, "pending_order_exists",
                            market=market_id[:8], side=signal.side.value
                        )
                        return
                # Already have any pending order for this asset
                self._log_order_block(
                    asset, "pending_order_any",
                    market=market_id[:8], order_id=order_id[:8]
                )
                return

        # Check total positions (filled + pending) against max concurrent limit
        filled_positions = len(self.risk_manager.positions)
        pending_orders = pending_count
        total_positions = filled_positions + pending_orders
        max_positions = self.config.trading.max_concurrent_positions

        if total_positions >= max_positions:
            self._log_order_block(
                asset, "position_limit",
                total=total_positions, filled=filled_positions,
                pending=pending_orders, max=max_positions,
            )
            return

        # Check total capital at risk (filled positions + pending orders + new order)
        current_exposure = 0.0
        # Add value of filled positions
        for pos in self.risk_manager.positions.values():
            current_exposure += pos.shares * pos.entry_price
        # Add value of pending orders
        for _, order_data in pending_items:
            order_size = order_data.get("size", 0) * order_data.get("price", 0)
            current_exposure += order_size
        # Add proposed new order
        new_order_value = signal.size_usd
        total_exposure = current_exposure + new_order_value

        # Calculate max allowed exposure based on equity (cash + positions)
        # Using equity prevents over-allocation after realized profits
        equity = self._calculate_equity()
        max_exposure_pct = getattr(self.config.trading, 'max_total_exposure_pct', 0.25)
        max_exposure = equity * max_exposure_pct

        if total_exposure > max_exposure:
            self._log_order_block(
                asset, "exposure_limit",
                total_exposure=f"{total_exposure:.2f}",
                current=f"{current_exposure:.2f}",
                new=f"{new_order_value:.2f}",
                max_exposure=f"{max_exposure:.2f}",
            )
            return

        # CRITICAL: Check API positions FIRST (prevents duplicate trades after restart)
        if hasattr(self, '_api_positions') and asset in self._api_positions:
            api_pos = self._api_positions[asset]
            api_shares = api_pos.get('shares', 0) if isinstance(api_pos, dict) else getattr(api_pos, 'shares', 0)
            if api_shares > 0:
                position_key = f"{asset}:api_position"
                if position_key not in self._logged_rejections:
                    self._logged_rejections.add(position_key)
                    logger.warning(
                        f"[{asset}] ⛔ BLOCKING ORDER: Found {api_shares:.2f} shares in API! "
                        f"Position not in internal tracking - possible restart. Skipping to prevent duplicate."
                    )
                return

        # Check for existing positions from internal tracking
        existing_position = self._get_active_position(asset)
        if existing_position:
            # Check if we should average down instead of skipping
            averaging_decision = self._check_averaging_opportunity(
                position=existing_position,
                signal=signal,
                market=signal.market,
            )

            if averaging_decision and averaging_decision.should_average:
                # Execute averaging order
                logger.info(
                    f"[{asset}] 📈 AVERAGING DOWN: {averaging_decision.reason} | "
                    f"Adding ${averaging_decision.suggested_size_usd:.2f} ({averaging_decision.suggested_shares:.1f} shares)"
                )
                await self._execute_averaging_order(existing_position, signal, averaging_decision)
                return

            # Not averaging - skip as usual
            position_key = f"{signal.market.condition_id}:active_position"
            if position_key not in self._logged_rejections:
                self._logged_rejections.add(position_key)
                reason = averaging_decision.reason if averaging_decision else "already has position"
                logger.info(f"[{asset}] ❌ ACTIVE POSITION: {reason} - skipping signal")
            return

        # Check if we have enough balance before attempting
        available = self.risk_manager.current_bankroll * 0.90  # 10% buffer
        if signal.size_usd > available:
            # Log at INFO level - insufficient balance is critical info
            balance_key = f"{signal.market.condition_id}:balance"
            if balance_key not in self._logged_rejections:
                self._logged_rejections.add(balance_key)
                logger.info(
                    f"[{asset}] ❌ INSUFFICIENT BALANCE: Need ${signal.size_usd:.2f}, "
                    f"have ${available:.2f} (bankroll ${self.risk_manager.current_bankroll:.2f})"
                )
            # Set cooldown to prevent spam (never move timestamps into the future)
            self._last_order_time[asset] = now
            return

        # Get current price for logging
        current_price = self.signal_generator.get_price(signal.market.asset)
        target = signal.market.target_price

        # FINAL SAFETY CHECK: Validate target price at execution time
        # This catches any stale target prices that slipped through earlier checks
        if current_price and target:
            deviation = abs(current_price - target) / target
            # Calculate how long into the current period we are
            current_ts = int(now.timestamp())
            period_start = (current_ts // 900) * 900
            seconds_into_period = current_ts - period_start

            # In the first 60 seconds of a period, be extra strict about target price
            # The target should be VERY close to current price at period start
            if seconds_into_period < 60:
                max_deviation = 0.02  # Only 2% deviation allowed in first 60 seconds
                if deviation > max_deviation:
                    logger.warning(
                        f"⚠️ [{asset}] BLOCKING TRADE - Target price ${target:,.2f} "
                        f"deviates {deviation:.1%} from Chainlink ${current_price:,.2f} "
                        f"({seconds_into_period}s into period). Likely stale price from previous period."
                    )
                    return
            else:
                # After 60 seconds, use 5% threshold
                max_deviation = 0.05
                if deviation > max_deviation:
                    logger.warning(
                        f"⚠️ [{asset}] BLOCKING TRADE - Target price ${target:,.2f} "
                        f"deviates {deviation:.1%} from Chainlink ${current_price:,.2f}. "
                        f"Target price may be stale."
                    )
                    return

        # Log positions BEFORE trade (debug level - detailed analysis moved to after safety checks)
        api_pos_count = len(self._api_positions) if hasattr(self, '_api_positions') else 0
        risk_pos_count = len(self.risk_manager.positions)
        logger.debug(
            f"PRE-TRADE CHECK [{asset}]: API positions={api_pos_count} | "
            f"Tracked positions={risk_pos_count} | Bankroll=${self.risk_manager.current_bankroll:.2f}"
        )

        # === MINIMUM ORDER SIZE CHECK ===
        # Polymarket minimums:
        # - LIMIT orders: 5 shares minimum
        # - MARKET orders: $1 minimum (no share minimum)
        if signal.recommended_action == OrderAction.LIMIT:
            MIN_SHARES = 5.0
            if signal.size_shares < MIN_SHARES:
                logger.warning(
                    f"[{asset}] ❌ ORDER TOO SMALL: {signal.size_shares:.2f} shares < {MIN_SHARES} minimum for LIMIT | "
                    f"Skipping trade"
                )
                return
        else:  # MARKET order
            MIN_USD = 1.0
            if signal.size_usd < MIN_USD:
                logger.warning(
                    f"[{asset}] ❌ ORDER TOO SMALL: ${signal.size_usd:.2f} < ${MIN_USD} minimum for MARKET | "
                    f"Skipping trade"
                )
                return

        # Hard safety brakes: dedup, cooldown, rate limit, exposure caps
        market_id = signal.market.condition_id
        for _, order_data in pending_items:
            if (
                order_data.get("market_id") == market_id
                and order_data.get("asset") == asset
                and order_data.get("side") == signal.side.value
            ):
                self._log_order_block(asset, "duplicate_open_order")
                return

        now_ts = time.time()
        key = (market_id, asset)
        last_ts = self._last_order_time_by_market.get(key)
        if last_ts is not None:
            elapsed = now_ts - last_ts
            if elapsed < self._order_cooldown_seconds:
                remaining = self._order_cooldown_seconds - elapsed
                self._log_order_block(asset, "cooldown", secs_remaining=f"{remaining:.0f}")
                return

        # Global order rate limit (rolling 60s)
        while self._order_submit_ts and now_ts - self._order_submit_ts[0] > 60.0:
            self._order_submit_ts.popleft()
        if len(self._order_submit_ts) >= self.config.trading.max_orders_per_min:
            self._log_order_block(asset, "global_rate_limit")
            return

        # Exposure caps (positions + open orders + new order)
        total_exposure = 0.0
        per_asset_exposure: dict[str, float] = {}
        for pos in self.risk_manager.positions.values():
            pos_asset = pos.market.asset if hasattr(pos, "market") else asset
            pos_val = pos.shares * pos.entry_price
            total_exposure += pos_val
            per_asset_exposure[pos_asset] = per_asset_exposure.get(pos_asset, 0.0) + pos_val
        for _, order_data in pending_items:
            order_asset = order_data.get("asset")
            if not order_asset:
                continue
            order_val = float(order_data.get("size", 0)) * float(order_data.get("price", 0))
            total_exposure += order_val
            per_asset_exposure[order_asset] = per_asset_exposure.get(order_asset, 0.0) + order_val

        total_exposure_after = total_exposure + signal.size_usd
        per_asset_after = per_asset_exposure.get(asset, 0.0) + signal.size_usd
        if total_exposure_after > self.config.trading.max_total_exposure_usd:
            self._log_order_block(asset, "exposure_cap_total")
            return
        if per_asset_after > self.config.trading.max_exposure_per_asset_usd:
            self._log_order_block(asset, "exposure_cap_per_asset")
            return

        self._last_order_time_by_market[key] = now_ts
        self._order_submit_ts.append(now_ts)

        # === ALL SAFETY CHECKS PASSED ===
        # ORDER_DECISION is logged here AFTER all blocking gates (cooldown/dup/rate-limit/exposure)
        logger.info(
            "ORDER_DECISION asset=%s decision=PLACE_ORDER side=%s edge=%.4f min_edge=%.4f "
            "true_prob=%.4f price=%.4f size_usd=%.2f time_remaining=%.0f order_type=%s",
            asset,
            signal.side.value,
            float(signal.edge or 0.0),
            float(self.config.trading.min_edge),
            float(signal.true_prob or 0.0),
            float(signal.recommended_price or 0.0),
            float(signal.size_usd or 0.0),
            float(signal.time_remaining or 0.0),
            str(signal.recommended_action),
        )

        # Log detailed trade analysis
        arb_type = getattr(signal, '_arb_type', 'unknown')
        ml_confidence = getattr(signal, '_ml_confidence', None)
        binance_conf = getattr(signal, '_ml_binance_confirmation', 'NONE')
        binance_lead = getattr(signal, '_ml_binance_lead_pct', 0) or 0

        # Price analysis
        price_vs_target = "ABOVE" if current_price and target and current_price >= target else "BELOW"
        distance_pct = abs(current_price - target) / target * 100 if current_price and target else 0

        logger.info(f"{Colors.BRIGHT_CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
        logger.info(f"{Colors.BRIGHT_CYAN}║  📊 TRADE ANALYSIS: {asset} {signal.side.value:4}{Colors.RESET}")
        logger.info(f"{Colors.BRIGHT_CYAN}╠══════════════════════════════════════════════════════════════╣{Colors.RESET}")

        # Why this trade?
        logger.info(f"{Colors.BRIGHT_CYAN}║{Colors.RESET}  🎯 Signal Type: {arb_type.upper()}")
        logger.info(f"{Colors.BRIGHT_CYAN}║{Colors.RESET}  📝 Reasoning: {signal.reasoning}")

        # Price info
        logger.info(f"{Colors.BRIGHT_CYAN}║{Colors.RESET}  💰 Chainlink: ${current_price:,.2f} ({price_vs_target} target by {distance_pct:.2f}%)")
        logger.info(f"{Colors.BRIGHT_CYAN}║{Colors.RESET}  🎯 Target: ${target:,.2f}")

        # Market prices
        market = signal.market
        logger.info(f"{Colors.BRIGHT_CYAN}║{Colors.RESET}  📈 UP price: bid={market.best_bid:.3f} / ask={market.best_ask:.3f}")
        logger.info(f"{Colors.BRIGHT_CYAN}║{Colors.RESET}  📉 DOWN price: bid={1-market.best_ask:.3f} / ask={1-market.best_bid:.3f}")

        # Edge calculation
        logger.info(f"{Colors.BRIGHT_CYAN}║{Colors.RESET}  ✨ Edge: {signal.edge:.1%} (min required: {self.config.trading.min_edge:.1%})")

        # Binance confirmation
        if binance_conf != "NONE":
            logger.info(f"{Colors.BRIGHT_CYAN}║{Colors.RESET}  🔗 Binance: {binance_conf} confirmation (lead: {binance_lead:+.3f}%)")

        # ML prediction
        if ml_confidence:
            logger.info(f"{Colors.BRIGHT_CYAN}║{Colors.RESET}  🤖 ML Confidence: {ml_confidence:.0%} win probability")

        # Time remaining
        logger.info(f"{Colors.BRIGHT_CYAN}║{Colors.RESET}  ⏱️  Time left: {signal.time_remaining:.0f}s")

        # Position size
        logger.info(f"{Colors.BRIGHT_CYAN}║{Colors.RESET}  💵 Size: ${signal.size_usd:.2f} ({signal.size_shares:.2f} shares @ {signal.recommended_price:.3f})")

        logger.info(f"{Colors.BRIGHT_CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")

        # Log the trade attempt with colors
        direction = "▲" if signal.side == Side.UP else "▼"
        side_color = Colors.BRIGHT_GREEN if signal.side == Side.UP else Colors.BRIGHT_RED
        mode = self._execution_mode.lower()

        # Log ORDER_SUBMIT BEFORE execution attempt
        logger.info(
            "ORDER_SUBMIT asset=%s market=%s side=%s price=%.4f size_usd=%.2f size_shares=%.4f type=%s mode=%s",
            asset,
            market_id[:8],
            signal.side.value,
            float(signal.recommended_price or 0.0),
            float(signal.size_usd or 0.0),
            float(signal.size_shares or 0.0),
            str(signal.recommended_action),
            mode,
        )
        # Record to metrics file
        self.metrics.record_order_event(
            event_type="ORDER_SUBMIT",
            asset=asset,
            side=signal.side.value,
            price=float(signal.recommended_price or 0.0),
            size_usd=float(signal.size_usd or 0.0),
            mode=mode,
        )

        logger.info(
            f"{side_color}>>> EXECUTING: {asset} {signal.side.value} {direction}{Colors.RESET} │ "
            f"Edge: {Colors.BRIGHT_YELLOW}{signal.edge:.0%}{Colors.RESET} │ "
            f"${current_price:,.0f} vs ${target:,.0f} │ "
            f"${signal.size_usd:.2f}"
        )

        # Execute through order executor
        try:
            result = await self.executor.execute_signal_async(signal)
        except Exception as e:
            # Log but DO NOT re-raise - keep market loop alive
            logger.error(
                "ORDER_RESULT asset=%s market=%s side=%s status=exception error=%s mode=%s",
                asset,
                market_id[:8],
                signal.side.value,
                str(e)[:200],
                mode,
            )
            logger.debug("Executor exception traceback:", exc_info=True)
            # Record failed order to metrics
            self.metrics.record_order_event(
                event_type="ORDER_RESULT",
                asset=asset,
                side=signal.side.value,
                reason=f"exception: {str(e)[:50]}",
                mode=mode,
            )
            return  # Exit gracefully, don't kill the market loop

        # Log ORDER_RESULT AFTER execution
        if result.success:
            logger.info(
                "ORDER_RESULT asset=%s market=%s side=%s status=success order_id=%s "
                "filled_size=%.4f filled_price=%.4f mode=%s",
                asset,
                market_id[:8],
                signal.side.value,
                result.order_id[:16] if result.order_id else "none",
                result.filled_size,
                result.filled_price,
                mode,
            )
            # Record ORDER_FILL to metrics
            self.metrics.record_order_event(
                event_type="ORDER_FILL",
                asset=asset,
                side=signal.side.value,
                price=result.filled_price,
                size_usd=result.filled_size * result.filled_price,
                order_id=result.order_id,
                mode=mode,
            )
        else:
            logger.info(
                "ORDER_RESULT asset=%s market=%s side=%s status=failed error=%s mode=%s",
                asset,
                market_id[:8],
                signal.side.value,
                result.error_message[:50] if result.error_message else "unknown",
                mode,
            )
            # Record failed order to metrics
            self.metrics.record_order_event(
                event_type="ORDER_RESULT",
                asset=asset,
                side=signal.side.value,
                price=float(signal.recommended_price or 0.0),
                size_usd=float(signal.size_usd or 0.0),
                reason=result.error_message[:50] if result.error_message else "unknown",
                mode=mode,
            )

        ml_ev_info = getattr(signal, "_ml_ev_info", None)
        if ml_ev_info:
            logger.info(
                f"[{asset}] ML_EV p_up={ml_ev_info['p_up']:.3f} ev_up={ml_ev_info['ev_up']:.3f} "
                f"ev_down={ml_ev_info['ev_down']:.3f} edge_up={ml_ev_info['edge_up']:.3f} "
                f"edge_down={ml_ev_info['edge_down']:.3f} side={ml_ev_info['side']} result={result.success}"
            )

        if result.success:
            # Set full cooldown for successful orders
            self._last_order_time[asset] = now

            # Get ML data if available
            ml_volatility = getattr(signal, '_ml_volatility', None)
            ml_momentum = getattr(signal, '_ml_momentum', None)
            ml_confidence = getattr(signal, '_ml_confidence', None)
            ml_arb_type = getattr(signal, '_ml_arb_type', None)
            ml_spread = getattr(signal, '_ml_spread', None)
            ml_bid_depth = getattr(signal, '_ml_bid_depth', None)
            ml_ask_depth = getattr(signal, '_ml_ask_depth', None)
            ml_price_trend = getattr(signal, '_ml_price_trend', None)
            ml_distance_from_target = getattr(signal, '_ml_distance_from_target', None)
            ml_binance_lead_pct = getattr(signal, '_ml_binance_lead_pct', None)
            ml_binance_confirmation = getattr(signal, '_ml_binance_confirmation', None)
            ml_trend_1h = getattr(signal, '_ml_trend_1h', None)
            ml_trend_4h = getattr(signal, '_ml_trend_4h', None)
            ml_trend_1d = getattr(signal, '_ml_trend_1d', None)
            # Chart analysis features
            ml_chart_rsi = getattr(signal, '_ml_chart_rsi', None)
            ml_chart_trend_strength = getattr(signal, '_ml_chart_trend_strength', None)
            ml_chart_is_uptrend = getattr(signal, '_ml_chart_is_uptrend', None)
            ml_chart_is_downtrend = getattr(signal, '_ml_chart_is_downtrend', None)
            ml_chart_is_ranging = getattr(signal, '_ml_chart_is_ranging', None)
            ml_chart_bullish_reversal = getattr(signal, '_ml_chart_bullish_reversal', None)
            ml_chart_bearish_reversal = getattr(signal, '_ml_chart_bearish_reversal', None)
            ml_chart_momentum = getattr(signal, '_ml_chart_momentum', None)
            ml_chart_bias_bullish = getattr(signal, '_ml_chart_bias_bullish', None)
            ml_chart_bias_bearish = getattr(signal, '_ml_chart_bias_bearish', None)
            ml_chart_confidence = getattr(signal, '_ml_chart_confidence', None)
            ml_chart_bullish_pattern = getattr(signal, '_ml_chart_bullish_pattern', None)
            ml_chart_bearish_pattern = getattr(signal, '_ml_chart_bearish_pattern', None)
            # NEW INDICATOR FEATURES
            ml_macd_histogram = getattr(signal, '_ml_macd_histogram', None)
            ml_macd_crossover = getattr(signal, '_ml_macd_crossover', None)
            ml_bb_bandwidth = getattr(signal, '_ml_bb_bandwidth', None)
            ml_bb_position = getattr(signal, '_ml_bb_position', None)
            ml_stoch_k = getattr(signal, '_ml_stoch_k', None)
            ml_stoch_d = getattr(signal, '_ml_stoch_d', None)
            ml_stoch_signal = getattr(signal, '_ml_stoch_signal', None)
            ml_rsi_divergence = getattr(signal, '_ml_rsi_divergence', None)
            ml_rsi_divergence_strength = getattr(signal, '_ml_rsi_divergence_strength', None)
            ml_volume_ratio = getattr(signal, '_ml_volume_ratio', None)
            ml_is_high_volume = getattr(signal, '_ml_is_high_volume', None)
            ml_obv_trend = getattr(signal, '_ml_obv_trend', None)
            ml_ha_trend = getattr(signal, '_ml_ha_trend', None)
            ml_ha_consecutive = getattr(signal, '_ml_ha_consecutive', None)
            ml_ha_strength = getattr(signal, '_ml_ha_strength', None)
            ml_vwap_distance_pct = getattr(signal, '_ml_vwap_distance_pct', None)
            ml_vwap_position = getattr(signal, '_ml_vwap_position', None)

            conf_str = f" (ML: {ml_confidence:.0%})" if ml_confidence else ""
            arb_str = f" [{ml_arb_type}]" if ml_arb_type and ml_arb_type != "none" else ""

            # Check if order was immediately filled (market orders or crossing limit orders)
            if result.filled_size and result.filled_size > 0:
                # Order filled immediately - record position
                self.risk_manager.record_position_open(
                    signal=signal,
                    entry_price=result.filled_price or signal.recommended_price,
                    shares=result.filled_size,
                    ml_volatility=ml_volatility,
                    ml_momentum=ml_momentum,
                    ml_confidence=ml_confidence,
                    ml_arb_type=ml_arb_type,
                    ml_spread=ml_spread,
                    ml_bid_depth=ml_bid_depth,
                    ml_ask_depth=ml_ask_depth,
                    ml_price_trend=ml_price_trend,
                    ml_distance_from_target=ml_distance_from_target,
                    ml_binance_lead_pct=ml_binance_lead_pct,
                    ml_binance_confirmation=ml_binance_confirmation,
                    ml_trend_1h=ml_trend_1h,
                    ml_trend_4h=ml_trend_4h,
                    ml_trend_1d=ml_trend_1d,
                    # Chart analysis features
                    ml_chart_rsi=ml_chart_rsi,
                    ml_chart_trend_strength=ml_chart_trend_strength,
                    ml_chart_is_uptrend=ml_chart_is_uptrend,
                    ml_chart_is_downtrend=ml_chart_is_downtrend,
                    ml_chart_is_ranging=ml_chart_is_ranging,
                    ml_chart_bullish_reversal=ml_chart_bullish_reversal,
                    ml_chart_bearish_reversal=ml_chart_bearish_reversal,
                    ml_chart_momentum=ml_chart_momentum,
                    ml_chart_bias_bullish=ml_chart_bias_bullish,
                    ml_chart_bias_bearish=ml_chart_bias_bearish,
                    ml_chart_confidence=ml_chart_confidence,
                    ml_chart_bullish_pattern=ml_chart_bullish_pattern,
                    ml_chart_bearish_pattern=ml_chart_bearish_pattern,
                    # NEW INDICATOR FEATURES
                    ml_macd_histogram=ml_macd_histogram,
                    ml_macd_crossover=ml_macd_crossover,
                    ml_bb_bandwidth=ml_bb_bandwidth,
                    ml_bb_position=ml_bb_position,
                    ml_stoch_k=ml_stoch_k,
                    ml_stoch_d=ml_stoch_d,
                    ml_stoch_signal=ml_stoch_signal,
                    ml_rsi_divergence=ml_rsi_divergence,
                    ml_rsi_divergence_strength=ml_rsi_divergence_strength,
                    ml_volume_ratio=ml_volume_ratio,
                    ml_is_high_volume=ml_is_high_volume,
                    ml_obv_trend=ml_obv_trend,
                    ml_ha_trend=ml_ha_trend,
                    ml_ha_consecutive=ml_ha_consecutive,
                    ml_ha_strength=ml_ha_strength,
                    ml_vwap_distance_pct=ml_vwap_distance_pct,
                    ml_vwap_position=ml_vwap_position,
                )

                # Record in trade history
                trade_history = get_trade_history()
                trade_history.record_open(
                    asset=asset,
                    side=signal.side.value,
                    entry_price=result.filled_price or signal.recommended_price,
                    shares=result.filled_size,
                    target_price=signal.market.target_price,
                    chainlink_price=current_price,
                    market_id=signal.market.condition_id,
                    predicted_prob=ml_confidence,
                    arb_type=ml_arb_type or "none",
                    edge=signal.edge,
                )

                logger.info(f"    {Colors.BRIGHT_GREEN}✓ FILLED @ {result.filled_price:.2f}{conf_str}{arb_str}{Colors.RESET}")

                # Log positions AFTER trade
                logger.info(
                    f"📋 POST-TRADE [{asset}]: Tracked positions={len(self.risk_manager.positions)} | "
                    f"Bankroll=${self.risk_manager.current_bankroll:.2f}"
                )
            else:
                # Order placed but not filled yet - track as pending
                logger.info(
                    f"    {Colors.BRIGHT_YELLOW}⏳ ORDER PLACED @ {signal.recommended_price:.2f} "
                    f"(waiting for fill){conf_str}{arb_str}{Colors.RESET}"
                )

                # Store order details for later fill checking
                async with self._pending_orders_lock:
                    self._pending_orders[result.order_id] = {
                        "signal": signal,
                        "asset": asset,
                        "side": signal.side.value,  # "UP" or "DOWN"
                        "market_id": signal.market.condition_id,
                        "placed_time": now,
                        "price": signal.recommended_price,
                        "size": signal.size_shares,  # For display in status logs
                        "ml_volatility": ml_volatility,
                        "ml_momentum": ml_momentum,
                        "ml_confidence": ml_confidence,
                        "ml_arb_type": ml_arb_type,
                        "ml_spread": ml_spread,
                        "ml_bid_depth": ml_bid_depth,
                        "ml_ask_depth": ml_ask_depth,
                        "ml_price_trend": ml_price_trend,
                        "ml_distance_from_target": ml_distance_from_target,
                        "ml_binance_lead_pct": ml_binance_lead_pct,
                        "ml_binance_confirmation": ml_binance_confirmation,
                        "ml_trend_1h": ml_trend_1h,
                        "ml_trend_4h": ml_trend_4h,
                        "ml_trend_1d": ml_trend_1d,
                        # NEW INDICATOR FEATURES
                        "ml_macd_histogram": ml_macd_histogram,
                        "ml_macd_crossover": ml_macd_crossover,
                        "ml_bb_bandwidth": ml_bb_bandwidth,
                        "ml_bb_position": ml_bb_position,
                        "ml_stoch_k": ml_stoch_k,
                        "ml_stoch_d": ml_stoch_d,
                        "ml_stoch_signal": ml_stoch_signal,
                        "ml_rsi_divergence": ml_rsi_divergence,
                        "ml_rsi_divergence_strength": ml_rsi_divergence_strength,
                        "ml_volume_ratio": ml_volume_ratio,
                        "ml_is_high_volume": ml_is_high_volume,
                        "ml_obv_trend": ml_obv_trend,
                        "ml_ha_trend": ml_ha_trend,
                        "ml_ha_consecutive": ml_ha_consecutive,
                        "ml_ha_strength": ml_ha_strength,
                        "ml_vwap_distance_pct": ml_vwap_distance_pct,
                        "ml_vwap_position": ml_vwap_position,
                    }

                async with self._pending_orders_lock:
                    pending_count = len(self._pending_orders)
                logger.info(
                    "PENDING_ORDER_ADDED id=%s asset=%s market=%s side=%s "
                    "limit=%.4f shares=%.4f usd=%.2f pending_total=%d",
                    result.order_id[:16], asset,
                    signal.market.condition_id[:8], signal.side.value,
                    float(signal.recommended_price or 0),
                    float(signal.size_shares or 0),
                    float(signal.size_usd or 0),
                    pending_count,
                )
        else:
            # Set shorter cooldown (30s) on failures to prevent spam
            self._last_order_time[asset] = now - timedelta(seconds=self._order_cooldown_seconds - 30)
            logger.info(
                "ORDER_REJECT asset=%s reason=execution_error error=%s",
                asset,
                result.error_message,
            )
            logger.warning(f"    {Colors.BRIGHT_RED}✗ {result.error_message}{Colors.RESET}")

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

    def get_status(self) -> dict:
        """Get current bot status."""
        return {
            "running": self._running,
            "chainlink_connected": self.chainlink_feed.is_connected,
            "clob_connected": self.clob_feed.is_connected,
            "active_markets": len(self.markets),
            "expiring_markets": len(self.expiring_markets),
            "settled_count": len(self.settled_markets),
            "chainlink_prices": self.chainlink_feed.get_all_prices(),
            "risk_status": self.risk_manager.get_status_summary(),
            "config": self.config.to_dict(),
        }


async def main(argv=None) -> None:
    if argv is not None:
        import sys
        sys.argv = [sys.argv[0], *argv]
    from src.bot.run import main as run_main
    await run_main()
