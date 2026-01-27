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
from .data import ChainlinkFeed, CLOBFeed, GammaAPI
from .execution import create_trading_client, OrderExecutor
from .execution.client import get_account_balance
from .strategy import SignalGenerator, RiskManager
from .strategy.ml_predictor import get_ml_predictor, MLSignalPredictor
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
        self._order_cooldown_seconds = 120  # 2 minutes cooldown per asset

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

        # Control flags
        self._running = False
        self._shutdown_event = asyncio.Event()

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

        logger.info("Trading bot initialized successfully")
        return True

    async def _sync_existing_orders(self, force: bool = False):
        """Sync existing orders and positions from Polymarket."""
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

            # 1. Sync open orders (unfilled)
            open_orders = get_open_orders(self.client)
            token_to_asset = {"btc": "BTC", "eth": "ETH", "sol": "SOL", "xrp": "XRP"}

            for order in open_orders:
                asset_id = order.get("asset_id", "").lower()
                market = order.get("market", "").lower()

                for key, asset in token_to_asset.items():
                    if key in asset_id or key in market:
                        synced_assets.add(asset)
                        self._last_order_time[asset] = now
                        break

            # 2. Sync active positions (filled trades in active markets)
            positions = get_active_positions(self.client)
            for asset, pos_info in positions.items():
                synced_assets.add(asset)
                self._last_order_time[asset] = now
                # Update bankroll to reflect held positions
                if asset not in [p.market.asset for p in self.risk_manager.positions.values()]:
                    cost = pos_info.get("cost", 0)
                    if cost > 0:
                        self.risk_manager.current_bankroll -= cost
                        logger.info(f"Found position: {asset} ${cost:.2f}")

            if synced_assets:
                logger.info(f"Active: {', '.join(synced_assets)} | Bankroll: ${self.risk_manager.current_bankroll:.2f}")

        except Exception as e:
            logger.warning(f"Could not sync orders/positions: {e}")

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

    async def start(self):
        """Start the trading bot."""
        if not await self.initialize():
            logger.error("Failed to initialize bot")
            return

        self._running = True
        logger.info("Starting trading bot...")

        try:
            # Run all components concurrently
            await asyncio.gather(
                self._run_chainlink_feed(),
                self._run_clob_feed(),
                self._run_trading_loop(),
                self._run_settlement_loop(),  # New: dedicated settlement checker
                return_exceptions=True,
            )
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
        self.gamma_api.close()

        # Shutdown notification thread pool
        shutdown_notification_executor()

        logger.info("Trading bot shutdown complete")

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

    async def _run_trading_loop(self):
        """Main trading loop - handles market discovery and signal execution."""
        logger.info("Starting trading loop...")

        # Wait for data feeds to connect
        await asyncio.sleep(2.0)

        while self._running:
            try:
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

                # Discover and refresh markets
                await self._refresh_markets()

                # Sync balance and open orders periodically
                await self._sync_balance()
                await self._sync_existing_orders()

                # Move expiring markets out of active trading
                await self._check_expiring_markets()

                # Check if trading is allowed
                can_trade, reason = self.risk_manager.can_trade()
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

                for market in list(self.markets.values()):
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
                logger.info(
                    f"EXPIRING: {market.asset} | "
                    f"Time remaining: {market.time_remaining:.0f}s | "
                    f"Target: ${market.target_price:,.2f}"
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
                await self._settle_market(market)

                # Move to settled set and cleanup
                self.expiring_markets.pop(market_id, None)
                self.settled_markets.add(market_id)

                # Clean up token-to-market mapping
                self._token_to_market.pop(market.up_token_id, None)
                self._token_to_market.pop(market.down_token_id, None)

                # Unsubscribe from order book
                self.clob_feed.unsubscribe(market.up_token_id)
                self.clob_feed.unsubscribe(market.down_token_id)

                # Cleanup old settled markets (keep last 100)
                if len(self.settled_markets) > 100:
                    self.settled_markets = set(list(self.settled_markets)[-100:])

    async def _settle_market(self, market: MarketState):
        """
        Process settlement for a single market.

        Args:
            market: Market to settle
        """
        logger.info(f"SETTLING: {market.asset} | {market.question[:50]}...")

        # Check if we have a position in this market
        position = self.risk_manager.positions.get(market.condition_id)

        # Fetch resolution from API
        resolution = self.gamma_api.get_market_resolution(market.condition_id)

        if not resolution:
            logger.warning(f"Could not fetch resolution for {market.condition_id}")
            # Retry later - don't remove from expiring yet
            return

        if not resolution.get("resolved"):
            logger.debug(f"Market {market.asset} not yet resolved, will retry...")
            return

        winning_outcome = resolution.get("winning_outcome")
        resolution_price = resolution.get("resolution_price")

        logger.info(
            f"RESOLVED: {market.asset} | "
            f"Winner: {winning_outcome} | "
            f"Settlement Price: ${resolution_price:,.2f if resolution_price else 0}"
        )

        # Process position if we had one
        if position:
            await self._close_position_on_settlement(
                market=market,
                position=position,
                winning_outcome=winning_outcome,
            )
        else:
            logger.debug(f"No position in {market.asset}, nothing to settle")

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

        # Record ML outcome for model learning
        if self.ml_predictor and position.ml_volatility is not None:
            # We need to recreate the signal to record the outcome
            # Create a minimal signal-like object for ML recording
            from types import SimpleNamespace
            fake_signal = SimpleNamespace(
                edge=0.0,  # Not needed for outcome
                market=market,
                side=position.side,
            )
            self.ml_predictor.record_outcome(
                signal=fake_signal,
                volatility=position.ml_volatility,
                price_momentum=position.ml_momentum or 0.0,
                won=won,
            )

    async def _process_market(self, market: MarketState):
        """
        Process a single market for trading opportunities.

        Args:
            market: Market to process
        """
        # Skip if market is about to expire (should be in expiring_markets)
        if market.time_remaining < self.config.trading.min_time_remaining:
            return

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
            logger.debug(f"Signal for {market.asset} rejected: size too small")
            return

        # ML filter - check predicted win probability
        if self.ml_predictor:
            volatility = self.config.volatility.get(market.asset)
            # Calculate simple momentum from Chainlink price
            price_momentum = 0.0
            current_price = self.signal_generator.get_price(market.asset)
            if current_price and market.target_price:
                # Positive if price > target, negative if below
                price_momentum = (current_price - market.target_price) / market.target_price
                price_momentum = max(-1, min(1, price_momentum * 10))  # Scale and clamp

            should_trade, confidence, ml_reason = self.ml_predictor.should_trade(
                signal, volatility, price_momentum
            )
            if not should_trade:
                rejection_key = f"{market.condition_id}:ml"
                if rejection_key not in self._logged_rejections:
                    self._logged_rejections.add(rejection_key)
                    logger.info(f"[{market.asset}] 🤖 {ml_reason}")
                return

            # Store ML data with signal for outcome recording
            signal._ml_volatility = volatility
            signal._ml_momentum = price_momentum
            signal._ml_confidence = confidence

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

        if price_str:
            status_parts.append(f"{Colors.DIM}{price_str}{Colors.RESET}")

        logger.info(" │ ".join(status_parts))

    async def _execute_signal(self, signal: Signal):
        """Execute a trading signal."""
        asset = signal.market.asset
        now = datetime.now(timezone.utc)

        # Check cooldown - prevent rapid duplicate orders
        if asset in self._last_order_time:
            elapsed = (now - self._last_order_time[asset]).total_seconds()
            if elapsed < self._order_cooldown_seconds:
                remaining = self._order_cooldown_seconds - elapsed
                logger.debug(f"[{asset}] Cooldown: {remaining:.0f}s remaining")
                return

        # Check if we have enough balance before attempting
        available = self.risk_manager.current_bankroll * 0.90  # 10% buffer
        if signal.size_usd > available:
            logger.debug(f"[{asset}] Insufficient balance: need ${signal.size_usd:.2f}, have ${available:.2f}")
            # Set a short cooldown to prevent spam
            self._last_order_time[asset] = now - timedelta(seconds=self._order_cooldown_seconds - 30)
            return

        # Get current price for logging
        current_price = self.signal_generator.get_price(signal.market.asset)
        target = signal.market.target_price

        # Log the trade attempt with colors
        direction = "▲" if signal.side == Side.UP else "▼"
        side_color = Colors.BRIGHT_GREEN if signal.side == Side.UP else Colors.BRIGHT_RED
        logger.info(
            f"{side_color}>>> {asset} {signal.side.value} {direction}{Colors.RESET} │ "
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

            # Record position with risk manager (including ML data for outcome tracking)
            self.risk_manager.record_position_open(
                signal=signal,
                entry_price=result.filled_price or signal.recommended_price,
                shares=result.filled_size or signal.size_shares,
                ml_volatility=ml_volatility,
                ml_momentum=ml_momentum,
                ml_confidence=ml_confidence,
            )
            conf_str = f" (ML: {ml_confidence:.0%})" if ml_confidence else ""
            logger.info(f"    {Colors.BRIGHT_GREEN}✓ FILLED @ {signal.recommended_price:.2f}{conf_str}{Colors.RESET}")
        else:
            # Set shorter cooldown (30s) on failures to prevent spam
            self._last_order_time[asset] = now - timedelta(seconds=self._order_cooldown_seconds - 30)
            logger.warning(f"    {Colors.BRIGHT_RED}✗ {result.error_message}{Colors.RESET}")

    def _on_chainlink_price(self, price: ChainlinkPrice):
        """Handle Chainlink price update."""
        # Update signal generator with new price (used for probability calculations)
        self.signal_generator.update_price(price.symbol, price.price)

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
