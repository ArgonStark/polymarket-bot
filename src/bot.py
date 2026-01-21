"""
Main trading bot module.

Orchestrates all components: data feeds, signal generation,
risk management, and order execution.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from .config import BotConfig
from .models import MarketState, ChainlinkPrice, Signal, OrderAction, Side
from .data import ChainlinkFeed, CLOBFeed, GammaAPI
from .execution import create_trading_client, OrderExecutor
from .strategy import SignalGenerator, RiskManager
from .utils import log_trade


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

        # Market state - three-stage lifecycle
        self.markets: dict[str, MarketState] = {}  # Active trading markets
        self.expiring_markets: dict[str, MarketState] = {}  # Pending settlement
        self.settled_markets: set[str] = set()  # Already settled (prevent re-processing)

        # Timing trackers
        self.last_market_refresh = None
        self.last_settlement_check = None

        # Intervals (seconds)
        self.settlement_check_interval = 5.0  # Check settlements frequently
        self.market_discovery_interval = 15.0  # Discover new markets every 15s

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

        # Create trading client
        if not self.config.dry_run:
            self.client = create_trading_client(self.config)
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

        # Initialize risk manager with starting bankroll
        # In production, fetch actual balance
        initial_bankroll = float(
            self.config.trading.base_position_size
            * self.config.trading.max_concurrent_positions
            * 3
        )
        self.risk_manager.initialize(initial_bankroll)

        logger.info("Trading bot initialized successfully")
        return True

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

        logger.info("Trading bot shutdown complete")

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
                # Discover and refresh markets
                await self._refresh_markets()

                # Move expiring markets out of active trading
                await self._check_expiring_markets()

                # Check if trading is allowed
                can_trade, reason = self.risk_manager.can_trade()
                if not can_trade:
                    logger.warning(f"Trading paused: {reason}")
                    await asyncio.sleep(60.0)
                    continue

                # Generate and execute signals for each active market
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

        # Check if refresh is needed
        if self.last_market_refresh:
            elapsed = (now - self.last_market_refresh).total_seconds()
            if elapsed < self.market_discovery_interval:
                return

        logger.debug("Refreshing active markets...")

        try:
            # Fetch active 15-min crypto markets
            new_markets = self.gamma_api.get_15min_crypto_markets()

            # Update markets dict
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
                    self.clob_feed.subscribe(market.up_token_id)
                    self.clob_feed.subscribe(market.down_token_id)
                    logger.info(
                        f"NEW MARKET: {market.asset} | "
                        f"Target: ${market.target_price:,.2f} | "
                        f"Ends: {market.end_time.strftime('%H:%M:%S')} | "
                        f"Time: {market.time_remaining:.0f}s"
                    )
                else:
                    # Update existing market state
                    self.markets[market_id].best_bid = market.best_bid
                    self.markets[market_id].best_ask = market.best_ask
                    self.markets[market_id].last_updated = now

            self.last_market_refresh = now
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
        now = datetime.now(timezone.utc)
        min_time = self.config.trading.min_time_remaining

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
        now = datetime.now(timezone.utc)

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

        logger.info(
            f"POSITION CLOSED: {market.asset} {position_side} | "
            f"{'WIN' if won else 'LOSS'} | "
            f"Shares: {position.shares:.2f} | "
            f"Entry: {position.entry_price:.4f} | "
            f"P&L: ${pnl:+.2f}"
        )

        # Record with risk manager
        self.risk_manager.record_position_close(
            market_key=market.condition_id,
            exit_price=1.0 if won else 0.0,
            pnl=pnl,
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
            logger.info(f"Signal rejected: {reason}")
            return

        # Adjust size if needed
        signal = self.risk_manager.adjust_signal_size(signal)

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

    async def _execute_signal(self, signal: Signal):
        """
        Execute a trading signal.

        Args:
            signal: Signal to execute
        """
        logger.info(
            f"EXECUTING {signal.recommended_action.value}: "
            f"{signal.side.value} {signal.market.asset} | "
            f"Edge: {signal.edge:.1%} | "
            f"Size: ${signal.size_usd:.2f} | "
            f"Time: {signal.time_remaining:.0f}s"
        )

        # Execute through order executor
        result = self.executor.execute_signal(signal)

        if result.success:
            # Record position with risk manager
            self.risk_manager.record_position_open(
                signal=signal,
                entry_price=result.filled_price or signal.recommended_price,
                shares=result.filled_size or signal.size_shares,
            )

            # Log trade
            log_trade(
                signal=signal,
                result=result,
                config=self.config,
            )
        else:
            logger.warning(f"Trade failed: {result.error_message}")

    def _on_chainlink_price(self, price: ChainlinkPrice):
        """Handle Chainlink price update."""
        # Update signal generator with new price
        self.signal_generator.update_price(price.symbol, price.price)

    def _on_orderbook_update(self, token_id: str, orderbook):
        """Handle order book update."""
        # Find market with this token (check both active and expiring)
        all_markets = {**self.markets, **self.expiring_markets}
        for market in all_markets.values():
            if market.up_token_id == token_id:
                if orderbook.best_bid is not None:
                    market.best_bid = orderbook.best_bid
                if orderbook.best_ask is not None:
                    market.best_ask = orderbook.best_ask
                market.last_updated = datetime.now(timezone.utc)
                break

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
