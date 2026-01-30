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

        # Track logged rejections to avoid spam
        self._logged_rejections: set[str] = set()

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

        # Intervals (seconds)
        self.settlement_check_interval = 5.0  # Check settlements frequently
        self.market_discovery_interval = 15.0  # Discover new markets every 15s
        self.balance_sync_interval = 30.0  # Sync balance every 30s
        self.orders_sync_interval = 60.0  # Sync open orders every 60s
        self.period_transition_delay = 5.0  # Seconds to wait after period boundary for price data
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
                    await asyncio.gather(
                        *[self._process_market(m) for m in sorted_markets],
                        return_exceptions=True
                    )
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

    async def _refresh_markets(self):
        """Refresh list of active 15-minute markets."""
        now = datetime.now(timezone.utc)
        current_ts = int(now.timestamp())

        # Calculate current 15-minute period (rounds down to :00, :15, :30, :45)
        current_period_ts = (current_ts // 900) * 900

        # Detect period boundary crossing
        if self._current_period_ts > 0 and current_period_ts != self._current_period_ts:
            logger.info(
                f"PERIOD BOUNDARY: Transitioning from {self._current_period_ts} "
                f"to {current_period_ts}"
            )

            # Wait for new price data to become available
            # The API needs a few seconds after period boundary to have closePrice
            if self._period_transition_wait_until is None:
                wait_seconds = self.period_transition_delay
                self._period_transition_wait_until = now + timedelta(seconds=wait_seconds)
                logger.info(
                    f"Waiting {wait_seconds}s for new period price data..."
                )

            # If still waiting, don't refresh yet
            if now < self._period_transition_wait_until:
                logger.debug(
                    f"Still waiting for period transition "
                    f"({(self._period_transition_wait_until - now).total_seconds():.1f}s remaining)"
                )
                return

            # Wait period complete - clear old markets and reset transition state
            logger.info("Period transition complete - refreshing markets with new prices")
            self._period_transition_wait_until = None

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

        # Color the result
        result_color = Colors.BRIGHT_GREEN if won else Colors.BRIGHT_RED
        result_emoji = "🎉" if won else "💔"
        logger.info(
            f"{result_color}{result_emoji} POSITION CLOSED: {market.asset} {position_side} | "
            f"{'WIN' if won else 'LOSS'} | "
            f"Shares: {position.shares:.2f} | "
            f"Entry: {position.entry_price:.4f} | "
            f"P&L: ${pnl:+.2f}{Colors.RESET}"
        )

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
            )

    async def _sync_ml_from_polymarket(self):
        """
        Sync ML model with actual trades from Polymarket Data API.

        Uses three endpoints:
        - /positions - current open positions
        - /activity - all trade activity
        - /closed-positions - completed trades with P&L

        The ML learns from closed positions where we know the outcome (win/loss).
        """
        if not self.client or not self.ml_predictor:
            return

        # Rate limit: only sync every 5 minutes
        now = datetime.now(timezone.utc)
        if not hasattr(self, '_last_ml_sync_time'):
            self._last_ml_sync_time = None

        if self._last_ml_sync_time and (now - self._last_ml_sync_time).total_seconds() < 300:
            return

        self._last_ml_sync_time = now

        try:
            # Get wallet address
            wallet_address = self.client.get_address()
            if not wallet_address:
                logger.warning("Could not get wallet address for ML sync")
                return

            # Initialize Data API
            data_api = get_data_api()

            # Track what we've already synced (using condition_id + outcome as key)
            if not hasattr(self, '_synced_positions'):
                self._synced_positions = set()

            # Fetch closed positions (completed trades with realized P&L)
            closed_positions = data_api.get_all_closed_positions(wallet_address, max_positions=200)

            if not closed_positions:
                logger.debug("No closed positions found for ML sync")
                return

            synced_count = 0
            for pos in closed_positions:
                try:
                    # Create unique key for this position
                    condition_id = pos.get("conditionId", "")
                    outcome = pos.get("outcome", "")
                    pos_key = f"{condition_id}:{outcome}"

                    # Skip if already synced
                    if pos_key in self._synced_positions:
                        continue

                    # Check if this is a 15-min crypto market
                    slug = pos.get("slug", "") or pos.get("eventSlug", "")
                    title = pos.get("title", "")

                    if "updown-15m" not in slug.lower() and "15m" not in title.lower():
                        continue

                    # Identify asset
                    asset = None
                    for a in ["btc", "eth", "sol", "xrp"]:
                        if slug.lower().startswith(a) or a in slug.lower():
                            asset = a.upper()
                            break

                    if not asset:
                        continue

                    # Get outcome (Up/Down)
                    if isinstance(outcome, str):
                        outcome_lower = outcome.lower()
                        if "up" in outcome_lower:
                            side = Side.UP
                        elif "down" in outcome_lower:
                            side = Side.DOWN
                        else:
                            continue
                    else:
                        continue

                    # Determine win/loss from realizedPnl
                    realized_pnl = float(pos.get("realizedPnl", 0) or 0)
                    won = realized_pnl > 0

                    # Get trade details
                    avg_price = float(pos.get("avgPrice", 0) or 0)
                    total_bought = float(pos.get("totalBought", 0) or 0)

                    if avg_price <= 0 or total_bought <= 0:
                        continue

                    # Create mock signal for ML recording
                    from types import SimpleNamespace
                    fake_signal = SimpleNamespace(
                        edge=0.0,
                        market=SimpleNamespace(
                            asset=asset,
                            target_price=0,
                            time_remaining=0,
                            best_bid=avg_price,
                            best_ask=avg_price,
                            bid_depth=0,
                            ask_depth=0,
                        ),
                        side=side,
                        _arb_type="none",
                        recommended_price=avg_price,
                        size_shares=total_bought,
                        size_usd=total_bought * avg_price,
                        time_remaining=0,
                    )

                    # Record to ML model (historical trade - no real-time price data)
                    self.ml_predictor.record_outcome(
                        signal=fake_signal,
                        volatility=0.003,
                        price_momentum=0.0,
                        won=won,
                        # Price features not available for historical trades
                        current_price=0.0,
                        target_price=0.0,
                        price_high=0.0,
                        price_low=0.0,
                        price_velocity=0.0,
                    )

                    # Mark as synced
                    self._synced_positions.add(pos_key)
                    synced_count += 1

                except Exception as e:
                    logger.debug(f"Error processing position for ML sync: {e}")
                    continue

            if synced_count > 0:
                logger.info(
                    f"🤖 ML synced {synced_count} trades from Polymarket | "
                    f"Total samples: {self.ml_predictor.training_samples}"
                )

        except Exception as e:
            logger.error(f"ML sync from Polymarket failed: {e}")

    async def _sync_positions_from_polymarket(self):
        """
        Sync internal position tracking with actual Polymarket positions.

        Removes stale positions that no longer exist on Polymarket.
        This prevents the bot from blocking new trades due to ghost positions.
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

            for market_key in internal_positions:
                if market_key not in api_condition_ids:
                    # Position exists internally but not on Polymarket - it's been settled
                    position = self.risk_manager.positions.get(market_key)
                    if position:
                        asset = position.market.asset if hasattr(position, 'market') else "???"
                        side = position.side.value if hasattr(position, 'side') else "???"

                        logger.info(
                            f"🧹 Clearing stale position: {asset} {side} | "
                            f"Not found on Polymarket"
                        )

                        # Remove from risk manager (assume break-even if unknown)
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

        # Validate signal against risk limits
        is_valid, reason = self.risk_manager.validate_signal(signal)
        if not is_valid:
            # Only log each rejection reason once per market to avoid spam
            rejection_key = f"{market.condition_id}:{reason}"
            if rejection_key not in self._logged_rejections:
                self._logged_rejections.add(rejection_key)
                logger.info(f"[{market.asset}] Blocked: {reason}")
            return

        # Adjust size if needed
        signal = self.risk_manager.adjust_signal_size(signal)

        # Check if signal was rejected due to size
        if signal.size_usd <= 0 or signal.size_shares <= 0:
            size_key = f"{market.condition_id}:size_too_small"
            if size_key not in self._logged_rejections:
                self._logged_rejections.add(size_key)
                logger.info(f"[{market.asset}] 📏 Size too small after adjustment - check bankroll")
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
            rejection_key = f"{market.condition_id}:history:{signal.side.value}"
            if rejection_key not in self._logged_rejections:
                self._logged_rejections.add(rejection_key)
                logger.info(f"[{market.asset}] 📊 History block: {history_reason}")
            return

        # ML filter - check predicted win probability
        if self.ml_predictor:
            # Extract all ML features using the helper function
            ml_features = extract_ml_features_from_market(
                signal=signal,
                signal_generator=self.signal_generator,
                market=market,
            )

            should_trade, confidence, ml_reason = self.ml_predictor.should_trade(
                signal=signal,
                **ml_features,
            )
            if not should_trade:
                rejection_key = f"{market.condition_id}:ml"
                if rejection_key not in self._logged_rejections:
                    self._logged_rejections.add(rejection_key)
                    logger.info(f"[{market.asset}] 🤖 {ml_reason}")
                return

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

            # Calculate depth
            market.bid_depth = sum(level.size for level in up_book.bids)
            market.ask_depth = sum(level.size for level in up_book.asks)

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

        # Always log header with summary
        pnl_str = ""
        if unrealized_total != 0:
            pnl_color = Colors.BRIGHT_GREEN if unrealized_total > 0 else Colors.BRIGHT_RED
            pnl_str = f" | {pnl_color}Unrealized: ${unrealized_total:+.2f}{Colors.RESET}"

        logger.info(f"{Colors.BRIGHT_CYAN}📊 POSITION STATUS (every 30s){Colors.RESET}")
        logger.info(
            f"   Tracked: {len(positions)} | API: {len(api_positions)} | "
            f"Pending: {len(pending_orders)} | "
            f"Cash: ${cash:.2f} | Equity: ${equity:.2f}{pnl_str}"
        )

        # Log tracked positions with P&L and win probability
        for market_key, position in positions.items():
            market = self.markets.get(market_key) or self.expiring_markets.get(market_key)
            asset = position.market.asset if hasattr(position, 'market') else "???"
            side = position.side.value

            # Get Chainlink price and calculate win probability
            chainlink_price = self.signal_generator.get_price(asset) if hasattr(self, 'signal_generator') else None
            target_price = market.target_price if market else None
            time_remaining = market.time_remaining if market else 0
            win_prob_str = ""
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

                # Color based on probability
                if our_win_prob >= 0.7:
                    prob_color = Colors.BRIGHT_GREEN
                    prob_icon = "✅"
                elif our_win_prob >= 0.5:
                    prob_color = Colors.BRIGHT_YELLOW
                    prob_icon = "⚖️"
                else:
                    prob_color = Colors.BRIGHT_RED
                    prob_icon = "⚠️"

                win_prob_str = f" | {prob_color}{prob_icon} Win: {our_win_prob:.0%}{Colors.RESET}"

                # Add price context
                price_diff = chainlink_price - target_price
                price_direction = "above" if price_diff > 0 else "below"
                win_prob_str += f" (${chainlink_price:,.0f} {price_direction} target)"

            # Calculate P&L - use settlement price for expired markets
            current_market_price = None
            if market:
                # For expired markets with known outcome, use settlement price
                if time_remaining <= 0 and our_win_prob >= 0.99:
                    # We're winning - settlement pays $1 per share
                    current_market_price = 1.0
                elif time_remaining <= 0 and our_win_prob <= 0.01:
                    # We're losing - settlement pays $0
                    current_market_price = 0.0
                else:
                    # Active market - use current market price
                    if position.side == Side.UP:
                        current_market_price = market.best_bid
                    else:
                        current_market_price = 1.0 - market.best_ask if market.best_ask else None

            if current_market_price is not None and current_market_price >= 0:
                current_value = position.shares * current_market_price
                unrealized_pnl = current_value - position.cost_basis
                if position.entry_price > 0:
                    pnl_pct = (current_market_price - position.entry_price) / position.entry_price
                else:
                    pnl_pct = 0

                # Color based on P&L
                pnl_color = Colors.BRIGHT_GREEN if unrealized_pnl > 0 else Colors.BRIGHT_RED

                # ML confidence if stored
                ml_str = ""
                if position.ml_confidence:
                    ml_str = f" | ML: {position.ml_confidence:.0%}"

                # Show settlement indicator for expired markets
                settled_str = " [SETTLING]" if time_remaining <= 0 else ""

                logger.info(
                    f"   📍 {asset} {side}: {position.shares:.1f} @ {position.entry_price:.2f} → "
                    f"{current_market_price:.2f} | {pnl_color}P&L: ${unrealized_pnl:+.2f} ({pnl_pct:+.0%}){Colors.RESET} | "
                    f"Time: {time_remaining:.0f}s{settled_str}{win_prob_str}{ml_str}"
                )
            else:
                logger.info(
                    f"   📍 {asset} {side}: {position.shares:.1f} shares @ {position.entry_price:.2f} | "
                    f"(no price data){win_prob_str}"
                )

        # Log API-discovered positions (if any)
        for asset, pos_info in api_positions.items():
            # Check if already logged as tracked position
            tracked_assets = [p.market.asset for p in positions.values() if hasattr(p, 'market')]
            if asset not in tracked_assets:
                logger.info(
                    f"   🔍 {asset} (API): {pos_info.get('size', 0):.1f} shares @ "
                    f"${pos_info.get('price', 0):.2f} | Market: {pos_info.get('market', 'unknown')}"
                )

        # Log pending orders
        for order_id, order_data in pending_orders.items():
            asset = order_data.get("asset", "???")
            side = order_data.get("side", "???")
            size = order_data.get("size", 0)
            logger.info(f"   ⏳ {asset} {side}: PENDING | {size:.1f} shares | Order: {order_id[:8]}...")

        # Log active cooldowns
        active_cooldowns = []
        for asset, last_time in self._last_order_time.items():
            elapsed = (now - last_time).total_seconds()
            remaining = self._order_cooldown_seconds - elapsed
            if remaining > 0:
                active_cooldowns.append(f"{asset}:{remaining:.0f}s")

        if active_cooldowns:
            logger.info(f"   ⏱️  Cooldowns: {', '.join(active_cooldowns)}")

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
                    logger.info(f"[{asset}] ⏳ Cooldown active: {remaining:.0f}s remaining")
                return

        # Check for existing positions from API (prevents duplicate trades)
        if self._has_active_position(asset):
            # Log at INFO level - this is an important reason trades aren't executing
            position_key = f"{signal.market.condition_id}:active_position"
            if position_key not in self._logged_rejections:
                self._logged_rejections.add(position_key)
                logger.info(f"[{asset}] 📊 Already has active position - skipping signal")
            return

        # Check if we have enough balance before attempting
        available = self.risk_manager.current_bankroll * 0.90  # 10% buffer
        if signal.size_usd > available:
            # Log at INFO level - insufficient balance is critical info
            balance_key = f"{signal.market.condition_id}:balance"
            if balance_key not in self._logged_rejections:
                self._logged_rejections.add(balance_key)
                logger.info(f"[{asset}] 💰 Insufficient balance: need ${signal.size_usd:.2f}, have ${available:.2f}")
            # Set a short cooldown to prevent spam
            self._last_order_time[asset] = now - timedelta(seconds=self._order_cooldown_seconds - 30)
            return

        # Get current price for logging
        current_price = self.signal_generator.get_price(signal.market.asset)
        target = signal.market.target_price

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
