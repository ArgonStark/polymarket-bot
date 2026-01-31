"""
Main trading bot module.

Orchestrates all components: data feeds, signal generation,
risk management, and order execution.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from .config import BotConfig
from .models import MarketState, ChainlinkPrice, Signal, OrderAction, Side
from .data import (
    ChainlinkFeed,
    CLOBFeed,
    GammaAPI,
    BinanceFeed,
    fetch_all_historical_prices,
    prepopulate_price_histories,
    get_data_api,
)
from .data.binance import BinancePrice
from .execution import create_trading_client, OrderExecutor
from .execution.client import get_account_balance, get_trades
from .strategy import SignalGenerator, RiskManager
from .strategy.ml_predictor import (
    get_ml_predictor,
    MLSignalPredictor,
    extract_ml_features_from_market,
    calculate_price_trend,
)
from .strategy.trade_history import get_trade_history
from .utils import log_trade, shutdown_notification_executor, print_status_box, print_config_box, Colors

# Import Binance chart analyzer for ML features
try:
    from .data.binance_chart import analyze_chart
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
        self.signal_generator = SignalGenerator(config=config)
        self.risk_manager = RiskManager(config=config)

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

        # Control flags
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Warm-up / observation tracking
        self._startup_time: Optional[datetime] = None  # When bot started
        self._market_first_seen: dict[str, datetime] = {}  # market_id -> first observation time
        self._warmup_complete = False  # True after warm-up period ends

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

        # Display startup info (balance, account status, etc.)
        await self._display_startup_info()

        # Initialize risk manager with actual bankroll
        initial_bankroll = await self._get_initial_bankroll()
        if initial_bankroll <= 0:
            logger.error("Cannot start: No bankroll available")
            return False
        self.risk_manager.initialize(initial_bankroll)

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

        if not self._pending_orders:
            return

        if self.client is None or self.executor is None:
            return

        # Log that we're checking pending orders
        logger.debug(f"Checking {len(self._pending_orders)} pending orders...")

        orders_to_remove = []

        for order_id, order_data in list(self._pending_orders.items()):
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
                logger.info(
                    f"📋 Order {order_id[:8]}... ({asset}): status={order_info.status.value} | "
                    f"filled={order_info.filled_size:.2f} | wait={wait_time:.0f}s"
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
        for order_id in orders_to_remove:
            self._pending_orders.pop(order_id, None)

        # Log pending orders count
        if self._pending_orders:
            logger.debug(f"Pending orders: {len(self._pending_orders)}")

    async def _display_startup_info(self):
        """
        Display account information at startup.

        Shows balance, trading mode, and key configuration parameters
        regardless of whether running in simulation or live mode.
        """
        mode = "SIMULATION" if self.config.dry_run else "LIVE TRADING"
        trading_mode = self.config.trading.mode.upper()

        balance = None
        open_orders_count = 0
        positions_count = 0

        # Try to fetch real account info if client is available and configured
        if self.client is not None:
            try:
                # Fetch balance using SDK
                balance = get_account_balance(self.client)

                # Fetch open orders count using SDK
                from .execution.client import get_open_orders, get_active_positions
                open_orders = get_open_orders(self.client)
                open_orders_count = len(open_orders)

                # Fetch active positions
                positions = get_active_positions(self.client)
                positions_count = len(positions)

            except Exception as e:
                logger.warning(f"Could not fetch account info: {e}")

        # If no balance fetched and in dry run, use simulated
        if balance is None and self.config.dry_run:
            balance = (
                self.config.trading.base_position_size
                * self.config.trading.max_concurrent_positions
                * 5
            )

        # Print colorful status box
        print_status_box(
            mode=f"{mode} ({trading_mode})",
            balance=balance,
            open_orders=open_orders_count,
            positions=positions_count,
            daily_pnl=0.0,
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
        """Gracefully shutdown the bot."""
        logger.info("Shutting down trading bot...")
        self._running = False
        self._shutdown_event.set()

        # Cancel any open orders
        if self.executor and not self.config.dry_run:
            self.executor.cancel_all_orders()

        # Disconnect data feeds
        self.chainlink_feed.disconnect()
        self.clob_feed.disconnect()
        if self.config.endpoints.binance_direct_enabled:
            self.binance_feed.disconnect()
        self.gamma_api.close()

        # Shutdown notification thread pool
        shutdown_notification_executor()

        logger.info("Trading bot shutdown complete")

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

                # Check if data feeds are healthy (circuit breakers)
                if self.chainlink_feed._circuit_open and self.clob_feed._circuit_open:
                    logger.critical(
                        "CRITICAL: Both data feeds have circuit breakers open. "
                        "Trading halted until feeds recover."
                    )
                    await asyncio.sleep(30.0)
                    continue
                elif self.chainlink_feed._circuit_open:
                    logger.warning("Chainlink feed circuit breaker open - trading with stale prices")
                elif self.clob_feed._circuit_open:
                    logger.warning("CLOB feed circuit breaker open - trading with stale order books")

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

                # Log position status periodically (every 30s)
                self._log_position_status()

                # Calculate equity (cash + unrealized position value)
                # This prevents false drawdown triggers when positions are open
                equity = self._calculate_equity()

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
                    # Periodically log colorful status (roughly every 30s)
                    import random
                    if random.random() < 0.02:
                        self._log_status_line()

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

        # Detect period boundary crossing
        if self._current_period_ts > 0 and current_period_ts != self._current_period_ts:
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

            # If still waiting, don't refresh yet
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
        if not self._pending_orders:
            return

        orders_to_cancel = []
        for order_id, order_data in self._pending_orders.items():
            if order_data.get("asset") == asset:
                orders_to_cancel.append(order_id)

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

    async def _check_early_exits(self):
        """
        Check open positions for take-profit or stop-loss conditions.

        For each position:
        1. Get current market price from order book
        2. Calculate unrealized P&L percentage
        3. If >= take_profit_pct: sell to lock in profit
        4. If <= -stop_loss_pct: sell to limit loss
        """
        if not self.risk_manager.positions:
            return

        now = datetime.now(timezone.utc)
        trading = self.config.trading

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

            except Exception as e:
                logger.error(f"Error checking early exit for {market_key}: {e}")

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
        if self.config.dry_run:
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

    async def _check_settlements(self):
        """
        Check expiring markets for settlement and close positions.

        For each market that has passed its end_time:
        1. Fetch resolution from API
        2. If resolved, calculate P&L and close position
        3. Move to settled set
        """
        async with self._markets_lock:
            markets_to_settle = []
            for market_id, market in self.expiring_markets.items():
                # Check if market has passed end time (with small buffer)
                if market.time_remaining <= -2.0:  # 2 second buffer after expiry
                    markets_to_settle.append(market_id)

            for market_id in markets_to_settle:
                market = self.expiring_markets[market_id]
                settled = await self._settle_market(market)

                # Only remove from expiring if actually settled
                if settled:
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

    async def _settle_market(self, market: MarketState) -> bool:
        """
        Process settlement for a single market.

        Args:
            market: Market to settle

        Returns:
            True if settlement was successful, False if should retry
        """
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
                    logger.info(
                        f"Market {market.asset} waiting for API resolution... "
                        f"(local: {local_outcome}, waited {time_since_expiry:.0f}s)"
                    )
                    return False  # Wait for API first
            else:
                # Couldn't determine locally either
                logger.warning(
                    f"Could not determine outcome for {market.asset} "
                    f"(waiting {time_since_expiry:.0f}s since expiry)"
                )
                # If we've waited more than 5 minutes, give up
                if time_since_expiry > 300:
                    logger.error(
                        f"SETTLEMENT TIMEOUT: {market.asset} - no resolution after 5 minutes"
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
            logger.warning(f"Have position in {market.asset} but no winning outcome!")
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
        Determine market outcome locally using Chainlink price.

        For 15-minute crypto markets:
        - If current price >= target price: UP wins
        - If current price < target price: DOWN wins

        Args:
            market: Market to determine outcome for

        Returns:
            Tuple of (winning_outcome, current_price) or (None, None) if can't determine
        """
        # Get current Chainlink price for this asset
        current_price = self.signal_generator.get_price(market.asset)

        if current_price is None or current_price <= 0:
            return (None, None)

        target_price = market.target_price
        if target_price is None or target_price <= 0:
            return (None, None)

        # Determine winner based on price vs target
        if current_price >= target_price:
            return ("UP", current_price)
        else:
            return ("DOWN", current_price)

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
        Close a position based on market settlement.

        Args:
            market: Settled market
            position: Our position in the market
            winning_outcome: "UP" or "DOWN"
        """
        # Determine if we won
        position_side = position.side.value  # "UP" or "DOWN"
        won = position_side == winning_outcome

        # Calculate P&L
        if won:
            # Winner pays out $1 per share
            payout = position.shares * 1.0
            pnl = payout - position.cost_basis
        else:
            # Loser gets nothing
            payout = 0.0
            pnl = -position.cost_basis

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

        # Record with risk manager
        self.risk_manager.record_position_close(
            market_key=market.condition_id,
            exit_price=1.0 if won else 0.0,
            pnl=pnl,
        )

        # Record with trade history
        trade_history = get_trade_history()
        trade_history.record_close(
            market_id=market.condition_id,
            won=won,
            pnl=pnl,
            exit_price=1.0 if won else 0.0,
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
            )

    async def _sync_ml_from_polymarket(self, force: bool = False):
        """
        Sync ML model with actual trades from Polymarket Data API.

        Uses multiple endpoints for comprehensive trade tracking:
        - /activity (type=TRADE) - All trade activity with BUY/SELL
        - /closed-positions - Completed trades with realized P&L
        - /activity (type=REDEEM) - Redeemed positions (settled markets)

        The ML learns from trades where we can determine the outcome (win/loss).

        Args:
            force: If True, bypass rate limiting (used on startup)
        """
        if not self.client or not self.ml_predictor:
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

        # Generate trading signal
        signal = self.signal_generator.generate_signal(market)
        if not signal:
            return

        # Skip signals without action
        if signal.recommended_action == OrderAction.SKIP:
            logger.debug(f"SKIP {market.asset}: {signal.reasoning}")
            return

        # Log signal details for debugging (only for actionable signals)
        arb_type = getattr(signal, '_arb_type', 'probability')
        logger.info(
            f"📡 SIGNAL [{market.asset}]: {signal.side.value} | "
            f"Edge: {signal.edge:.1%} | Type: {arb_type} | "
            f"Size: ${signal.size_usd:.2f} | Action: {signal.recommended_action.value}"
        )

        # Validate signal against risk limits
        logger.info(f"[{market.asset}] ✓1 Risk validation")
        is_valid, reason = self.risk_manager.validate_signal(signal)
        if not is_valid:
            # Always log the blocking reason (cleared every 30s to avoid spam)
            rejection_key = f"{market.condition_id}:{reason}"
            if rejection_key not in self._logged_rejections:
                self._logged_rejections.add(rejection_key)
                logger.info(
                    f"[{market.asset}] ❌ BLOCKED: {reason} | "
                    f"Bankroll: ${self.risk_manager.current_bankroll:.2f} | "
                    f"Positions: {len(self.risk_manager.positions)}"
                )
            else:
                logger.info(f"[{market.asset}] ⏸️ BLOCKED (repeat): {reason}")
            return

        # Trade history filter - check past performance for this asset/side
        logger.info(f"[{market.asset}] ✓2 History filter")
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
            rejection_key = f"{market.condition_id}:history:{signal.side.value}"
            if rejection_key not in self._logged_rejections:
                self._logged_rejections.add(rejection_key)
                logger.info(f"[{market.asset}] ❌ HISTORY BLOCK: {history_reason}")
            else:
                logger.info(f"[{market.asset}] ⏸️ HISTORY (repeat): {history_reason}")
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
                    if chart_signal_strength >= 0.7:  # Strong chart signal
                        if chart_says_down and signal.side == Side.UP:
                            logger.info(
                                f"[{market.asset}] ❌ CHART BLOCK: Taking UP against strong BEARISH chart "
                                f"({market_type}, {chart_bias} {chart_confidence:.0%}, trend={avg_trend:.2f})"
                            )
                            return
                        elif chart_says_up and signal.side == Side.DOWN:
                            logger.info(
                                f"[{market.asset}] ❌ CHART BLOCK: Taking DOWN against strong BULLISH chart "
                                f"({market_type}, {chart_bias} {chart_confidence:.0%}, trend={avg_trend:.2f})"
                            )
                            return

                    logger.info(f"[{market.asset}] ✓ Chart filter passed")

            except Exception as e:
                logger.debug(f"Chart analysis failed: {e}")

        # ML filter - check predicted win probability (do this BEFORE sizing for Kelly)
        logger.info(f"[{market.asset}] ✓3 ML filter")
        ml_confidence = None
        if self.ml_predictor:

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
                rejection_key = f"{market.condition_id}:ml"
                if rejection_key not in self._logged_rejections:
                    self._logged_rejections.add(rejection_key)
                    logger.info(f"[{market.asset}] ❌ ML BLOCK: {ml_reason} | Confidence: {confidence:.1%}")
                else:
                    logger.info(f"[{market.asset}] ⏸️ ML (repeat): {ml_reason} | Conf: {confidence:.1%}")
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

        # Adjust size using Kelly criterion (with ML confidence for optimal sizing)
        logger.info(f"[{market.asset}] ✓4 Kelly sizing (conf: {ml_confidence})")
        signal = self.risk_manager.adjust_signal_size(
            signal,
            ml_confidence=ml_confidence,
            use_kelly=True,
        )

        # Check if signal was rejected due to size
        logger.info(f"[{market.asset}] ✓5 Size: ${signal.size_usd:.2f} ({signal.size_shares:.1f} shares)")
        if signal.size_usd <= 0 or signal.size_shares <= 0:
            size_key = f"{market.condition_id}:size_too_small"
            if size_key not in self._logged_rejections:
                self._logged_rejections.add(size_key)
                max_size = self.risk_manager.current_bankroll * self.config.trading.max_position_pct
                logger.info(
                    f"[{market.asset}] ❌ SIZE TOO SMALL: "
                    f"Max position ${max_size:.2f} (bankroll ${self.risk_manager.current_bankroll:.2f} × "
                    f"{self.config.trading.max_position_pct:.0%}) < $3 minimum"
                )
            else:
                logger.info(f"[{market.asset}] ⏸️ SIZE (repeat): too small")
            return

        # Execute the signal
        logger.info(f"[{market.asset}] ✓6 EXECUTING TRADE!")
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

    def _log_status_line(self):
        """Log a colorful status line with positions and bankroll."""
        # Get current prices
        prices = self.chainlink_feed.get_all_prices()

        # Build price string
        price_parts = []
        for k, v in prices.items():
            asset = k.split('/')[0].upper()
            price_parts.append(f"{asset}: ${v:,.0f}")
        price_str = " | ".join(price_parts)

        # Get positions from cooldown (assets we're blocking)
        now = datetime.now(timezone.utc)
        active_positions = []
        for asset in ["BTC", "ETH", "SOL", "XRP"]:
            if asset in self._last_order_time:
                elapsed = (now - self._last_order_time[asset]).total_seconds()
                if elapsed < self._order_cooldown_seconds:
                    active_positions.append(asset)

        # Build status line
        bankroll = self.risk_manager.current_bankroll
        pos_count = len(self.risk_manager.positions)

        # Colorful output
        status_parts = [
            f"{Colors.BRIGHT_CYAN}💰 ${bankroll:.2f}{Colors.RESET}",
        ]

        if active_positions:
            pos_str = ", ".join(active_positions)
            status_parts.append(f"{Colors.BRIGHT_YELLOW}📊 {pos_str}{Colors.RESET}")

        # Show pending orders count
        pending_count = len(self._pending_orders) if hasattr(self, '_pending_orders') else 0
        if pending_count > 0:
            status_parts.append(f"{Colors.DIM}⏳ {pending_count} pending{Colors.RESET}")

        if price_str:
            status_parts.append(f"{Colors.DIM}{price_str}{Colors.RESET}")

        logger.info(" │ ".join(status_parts))

    def _calculate_equity(self) -> float:
        """
        Calculate current equity: cash + unrealized position value.

        For each open position, estimate its current value based on
        order book prices. This gives a more accurate picture of
        actual account value than just looking at cash.

        Returns:
            Total equity (cash + unrealized position value)
        """
        cash = self.risk_manager.current_bankroll
        unrealized_value = 0.0

        for market_key, position in self.risk_manager.positions.items():
            # Find the market
            market = self.markets.get(market_key) or self.expiring_markets.get(market_key)
            if not market:
                # If we can't find market, use cost basis as conservative estimate
                unrealized_value += position.cost_basis
                continue

            # Get current market price for our side
            if position.side == Side.UP:
                current_price = market.best_bid  # What we could sell for
            else:
                # For DOWN, value is 1 - UP_ask
                current_price = 1.0 - market.best_ask if market.best_ask else None

            if current_price and current_price > 0:
                # Current value = shares * current price
                unrealized_value += position.shares * current_price
            else:
                # Fallback to cost basis
                unrealized_value += position.cost_basis

        return cash + unrealized_value

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

        # Calculate equity
        equity = self._calculate_equity()
        cash = self.risk_manager.current_bankroll
        unrealized_total = equity - cash

        # Format P&L
        if unrealized_total > 0:
            pnl_color = Colors.BRIGHT_GREEN
            pnl_icon = "📈"
        elif unrealized_total < 0:
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
            f"{pnl_icon} P&L: {pnl_color}${unrealized_total:>+9.2f}{Colors.RESET}  {Colors.BRIGHT_CYAN}│{Colors.RESET}"
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

                # Get Chainlink price and calculate win probability
                chainlink_price = self.signal_generator.get_price(asset) if hasattr(self, 'signal_generator') else None
                target_price = market.target_price if market else None
                time_remaining = market.time_remaining if market else 0
                our_win_prob = 0.5  # Default

                if chainlink_price and target_price and market:
                    # Calculate real-time win probability
                    from .probability import calculate_true_probability
                    volatility = self.signal_generator.get_volatility(asset)
                    true_prob_up = calculate_true_probability(
                        current_price=chainlink_price,
                        target_price=target_price,
                        time_remaining_sec=market.time_remaining,
                        volatility_15min=volatility,
                    )
                    # Our win probability depends on our side
                    our_win_prob = true_prob_up if side == "UP" else (1 - true_prob_up)

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

                # Calculate P&L - use settlement price for expired markets
                current_market_price = None
                if market:
                    if time_remaining <= 0 and our_win_prob >= 0.99:
                        current_market_price = 1.0
                    elif time_remaining <= 0 and our_win_prob <= 0.01:
                        current_market_price = 0.0
                    else:
                        if position.side == Side.UP:
                            current_market_price = market.best_bid
                        else:
                            current_market_price = 1.0 - market.best_ask if market.best_ask else None

                if current_market_price is not None and current_market_price >= 0:
                    unrealized_pnl = (current_market_price - position.entry_price) * position.shares
                    pnl_pct = ((current_market_price - position.entry_price) / position.entry_price * 100) if position.entry_price > 0 else 0
                    pnl_color = Colors.BRIGHT_GREEN if unrealized_pnl > 0 else Colors.BRIGHT_RED

                    # Format time remaining
                    if time_remaining > 60:
                        time_str = f"{int(time_remaining // 60)}m {int(time_remaining % 60)}s"
                    else:
                        time_str = f"{int(time_remaining)}s"

                    # Settlement indicator
                    status_str = "⏰ SETTLING" if time_remaining <= 0 else f"⏱️  {time_str}"

                    # Side arrow
                    side_arrow = "▲" if side == "UP" else "▼"
                    side_color = Colors.BRIGHT_GREEN if side == "UP" else Colors.BRIGHT_RED

                    logger.info(
                        f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  {side_color}{side_arrow} {asset:4}{Colors.RESET} │ "
                        f"{position.shares:>6.1f} @ {position.entry_price:.3f} → {current_market_price:.3f} │ "
                        f"{pnl_color}{unrealized_pnl:>+7.2f} ({pnl_pct:>+5.0f}%){Colors.RESET} │ "
                        f"{prob_color}{prob_icon} {our_win_prob:>3.0%}{Colors.RESET} │ {status_str}  {Colors.BRIGHT_CYAN}│{Colors.RESET}"
                    )
                else:
                    side_arrow = "▲" if side == "UP" else "▼"
                    side_color = Colors.BRIGHT_GREEN if side == "UP" else Colors.BRIGHT_RED
                    logger.info(
                        f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  {side_color}{side_arrow} {asset:4}{Colors.RESET} │ "
                        f"{position.shares:>6.1f} @ {position.entry_price:.3f}        │ "
                        f"{'(no price)':^16} │ "
                        f"{prob_color}{prob_icon} {our_win_prob:>3.0%}{Colors.RESET} │ ...      {Colors.BRIGHT_CYAN}│{Colors.RESET}"
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
                logger.info(
                    f"{Colors.BRIGHT_CYAN}│{Colors.RESET}     {asset} {side}: {size:.1f} shares │ "
                    f"Order: {order_id[:12]}...                   {Colors.BRIGHT_CYAN}│{Colors.RESET}"
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

        # Box footer
        logger.info(f"{Colors.BRIGHT_CYAN}└{'─' * 64}┘{Colors.RESET}")

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
        # Check internal tracking first
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

    def _execute_averaging_order(self, position, signal, averaging_decision):
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
        result = self.executor.execute_signal(averaging_signal)

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
                    logger.info(f"[{asset}] ❌ COOLDOWN: {remaining:.0f}s remaining (last order {elapsed:.0f}s ago)")
                return

        # Check for existing positions from API (prevents duplicate trades)
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
                self._execute_averaging_order(existing_position, signal, averaging_decision)
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
            # Set a short cooldown to prevent spam
            self._last_order_time[asset] = now - timedelta(seconds=self._order_cooldown_seconds - 30)
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

        # Log positions BEFORE trade
        api_pos_count = len(self._api_positions) if hasattr(self, '_api_positions') else 0
        risk_pos_count = len(self.risk_manager.positions)
        logger.info(
            f"📋 PRE-TRADE CHECK [{asset}]: API positions={api_pos_count} | "
            f"Tracked positions={risk_pos_count} | Bankroll=${self.risk_manager.current_bankroll:.2f}"
        )

        # === DETAILED TRADE ANALYSIS LOG ===
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

        # === MINIMUM ORDER SIZE CHECK ===
        # Polymarket requires minimum 5 shares per order
        MIN_SHARES = 5.0
        if signal.size_shares < MIN_SHARES:
            logger.warning(
                f"[{asset}] ❌ ORDER TOO SMALL: {signal.size_shares:.2f} shares < {MIN_SHARES} minimum | "
                f"Skipping trade (graduated sizing reduced position too much)"
            )
            return

        # Log the trade attempt with colors
        direction = "▲" if signal.side == Side.UP else "▼"
        side_color = Colors.BRIGHT_GREEN if signal.side == Side.UP else Colors.BRIGHT_RED
        logger.info(
            f"{side_color}>>> EXECUTING: {asset} {signal.side.value} {direction}{Colors.RESET} │ "
            f"Edge: {Colors.BRIGHT_YELLOW}{signal.edge:.0%}{Colors.RESET} │ "
            f"${current_price:,.0f} vs ${target:,.0f} │ "
            f"${signal.size_usd:.2f}"
        )

        # Execute through order executor
        result = self.executor.execute_signal(signal)

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
                self._pending_orders[result.order_id] = {
                    "signal": signal,
                    "asset": asset,
                    "side": signal.side.value,  # "UP" or "DOWN"
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
                }

                logger.info(
                    f"📋 PENDING ORDER [{asset}]: {len(self._pending_orders)} orders waiting for fill"
                )
        else:
            # Set shorter cooldown (30s) on failures to prevent spam
            self._last_order_time[asset] = now - timedelta(seconds=self._order_cooldown_seconds - 30)
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
