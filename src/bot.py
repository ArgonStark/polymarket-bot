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
from .models import MarketState, ChainlinkPrice, Signal, OrderAction
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

        # Market state
        self.markets: dict[str, MarketState] = {}
        self.last_market_refresh = None

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
        """Main trading loop."""
        logger.info("Starting trading loop...")

        # Wait for data feeds to connect
        await asyncio.sleep(2.0)

        while self._running:
            try:
                # Refresh markets periodically
                await self._refresh_markets()

                # Check if trading is allowed
                can_trade, reason = self.risk_manager.can_trade()
                if not can_trade:
                    logger.warning(f"Trading paused: {reason}")
                    await asyncio.sleep(60.0)  # Wait longer when paused
                    continue

                # Generate and execute signals for each market
                for market in self.markets.values():
                    await self._process_market(market)

                # Wait before next iteration
                await asyncio.sleep(self.config.loop_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(5.0)

        logger.info("Trading loop stopped")

    async def _refresh_markets(self):
        """Refresh list of active 15-minute markets."""
        now = datetime.now(timezone.utc)

        # Check if refresh is needed
        if self.last_market_refresh:
            elapsed = (now - self.last_market_refresh).total_seconds()
            if elapsed < self.config.market_refresh_interval:
                return

        logger.info("Refreshing active markets...")

        try:
            # Fetch active 15-min crypto markets
            new_markets = self.gamma_api.get_15min_crypto_markets()

            # Update markets dict
            new_market_ids = set()
            for market in new_markets:
                market_id = market.condition_id
                new_market_ids.add(market_id)

                if market_id not in self.markets:
                    # New market - subscribe to order book
                    self.markets[market_id] = market
                    self.clob_feed.subscribe(market.up_token_id)
                    self.clob_feed.subscribe(market.down_token_id)
                    logger.info(f"Tracking market: {market.asset} {market.question[:50]}...")
                else:
                    # Update existing market state
                    self.markets[market_id].best_bid = market.best_bid
                    self.markets[market_id].best_ask = market.best_ask
                    self.markets[market_id].last_updated = now

            # Remove expired markets
            expired = set(self.markets.keys()) - new_market_ids
            for market_id in expired:
                market = self.markets.pop(market_id)
                self.clob_feed.unsubscribe(market.up_token_id)
                self.clob_feed.unsubscribe(market.down_token_id)
                logger.info(f"Removed expired market: {market.asset}")

            self.last_market_refresh = now
            logger.info(f"Tracking {len(self.markets)} active markets")

        except Exception as e:
            logger.error(f"Failed to refresh markets: {e}")

    async def _process_market(self, market: MarketState):
        """
        Process a single market for trading opportunities.

        Args:
            market: Market to process
        """
        # Skip if market is about to expire
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
            f"{signal.side.value} {signal.market.asset} "
            f"Edge: {signal.edge:.1%} "
            f"Size: ${signal.size_usd:.2f}"
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
        # Find market with this token
        for market in self.markets.values():
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
            "chainlink_prices": self.chainlink_feed.get_all_prices(),
            "risk_status": self.risk_manager.get_status_summary(),
            "config": self.config.to_dict(),
        }
