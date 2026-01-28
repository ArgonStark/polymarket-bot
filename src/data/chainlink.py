"""
Chainlink RTDS (Real-Time Data Stream) WebSocket client.

Connects to Polymarket's Chainlink price feed for real-time
cryptocurrency prices used for market settlement.
"""

import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable, Optional
from dataclasses import dataclass, field

import websocket

from ..models import ChainlinkPrice, PriceHistory
from ..config import BotConfig


logger = logging.getLogger(__name__)


@dataclass
class ChainlinkFeed:
    """
    WebSocket client for Chainlink price data.

    Connects to wss://ws-live-data.polymarket.com and subscribes
    to the crypto_prices_chainlink topic for real-time price updates.
    """

    config: BotConfig
    on_price_update: Optional[Callable[[ChainlinkPrice], None]] = None

    # Internal state
    _ws: Optional[websocket.WebSocketApp] = None
    _prices: dict[str, float] = field(default_factory=dict)
    _price_history: dict[str, PriceHistory] = field(default_factory=dict)
    _connected: bool = False
    _reconnect_delay: float = field(init=False)
    _max_reconnect_delay: float = field(init=False)
    _max_retries: int = field(init=False)
    _ping_interval: int = field(init=False)
    _ping_timeout: int = field(init=False)
    _retry_count: int = 0
    _circuit_open: bool = False  # Circuit breaker state
    _shutdown: bool = False  # Graceful shutdown flag

    def __post_init__(self):
        """Initialize price history and WebSocket settings from config."""
        # Initialize WebSocket settings from config
        self._reconnect_delay = self.config.websocket.initial_reconnect_delay
        self._max_reconnect_delay = self.config.websocket.max_reconnect_delay
        self._max_retries = self.config.websocket.max_retries
        self._ping_interval = self.config.websocket.ping_interval
        self._ping_timeout = self.config.websocket.ping_timeout

        # Initialize price history for supported assets
        for asset in self.config.supported_assets:
            symbol = f"{asset.lower()}/usd"
            self._price_history[symbol] = PriceHistory(asset=asset)

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected

    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.

        Args:
            symbol: Symbol like "btc/usd" or "BTC"

        Returns:
            Current price or None if not available
        """
        # Normalize symbol
        symbol_lower = symbol.lower()
        if "/" not in symbol_lower:
            symbol_lower = f"{symbol_lower}/usd"

        return self._prices.get(symbol_lower)

    def get_price_history(self, symbol: str) -> list[float]:
        """
        Get price history for volatility calculation.

        Args:
            symbol: Symbol like "btc/usd" or "BTC"

        Returns:
            List of recent prices
        """
        symbol_lower = symbol.lower()
        if "/" not in symbol_lower:
            symbol_lower = f"{symbol_lower}/usd"

        history = self._price_history.get(symbol_lower)
        if history:
            return history.get_prices()
        return []

    def _on_open(self, ws):
        """Handle WebSocket connection opened."""
        logger.info("Chainlink WebSocket connected")
        self._connected = True
        self._reconnect_delay = self.config.websocket.initial_reconnect_delay  # Reset reconnect delay
        self._retry_count = 0  # Reset retry count on successful connection

        # Subscribe to crypto prices
        subscribe_msg = {
            "action": "subscribe",
            "subscriptions": [
                {
                    "topic": self.config.endpoints.chainlink_topic,
                    "type": "*",
                    "filters": "",  # All symbols
                }
            ],
        }
        try:
            ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to {self.config.endpoints.chainlink_topic}")
        except Exception as e:
            logger.error(f"Failed to send subscription message: {e}")

    def _on_message(self, ws, message: str):
        """Handle incoming WebSocket message."""
        # Skip empty messages (keepalive/ping)
        if not message or not message.strip():
            return

        try:
            data = json.loads(message)

            # Check if it's a price update
            topic = data.get("topic")
            if topic == self.config.endpoints.chainlink_topic:
                payload = data.get("payload", {})
                symbol = payload.get("symbol")  # e.g., "btc/usd"
                price = payload.get("value")
                timestamp_ms = payload.get("timestamp")

                if symbol and price is not None:
                    # Update current price
                    self._prices[symbol] = float(price)

                    # Parse timestamp
                    if timestamp_ms:
                        timestamp = datetime.fromtimestamp(
                            timestamp_ms / 1000, tz=timezone.utc
                        )
                    else:
                        timestamp = datetime.now(timezone.utc)

                    # Update price history
                    if symbol in self._price_history:
                        self._price_history[symbol].add_price(timestamp, float(price))

                    # Create price object
                    chainlink_price = ChainlinkPrice(
                        symbol=symbol,
                        price=float(price),
                        timestamp=timestamp,
                    )

                    logger.debug(f"Price update: {symbol} = ${price:,.2f}")

                    # Notify callback
                    if self.on_price_update:
                        self.on_price_update(chainlink_price)

            elif data.get("type") == "subscribed":
                logger.info(f"Successfully subscribed: {data}")

            elif data.get("type") == "error":
                logger.error(f"WebSocket error: {data}")

        except json.JSONDecodeError:
            # Silently ignore parse errors for keepalive/ping messages
            if message and message.strip() and message not in ('', 'ping', 'pong'):
                logger.debug(f"Non-JSON Chainlink message: {message[:50]}...")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        logger.error(f"Chainlink WebSocket error: {error}")
        self._connected = False

    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection closed."""
        logger.warning(
            f"Chainlink WebSocket closed: {close_status_code} - {close_msg}"
        )
        self._connected = False

    def connect(self):
        """
        Connect to Chainlink WebSocket (blocking).

        This method runs the WebSocket connection in a blocking manner.
        For async usage, use connect_async() instead.
        """
        url = self.config.endpoints.chainlink_rtds_url

        self._ws = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        logger.info(f"Connecting to Chainlink feed: {url}")
        self._ws.run_forever()

    async def connect_async(self):
        """
        Connect to Chainlink WebSocket asynchronously.

        Runs the blocking WebSocket in a thread pool executor.
        Handles automatic reconnection with exponential backoff and circuit breaker.
        """
        loop = asyncio.get_running_loop()

        while not self._circuit_open and not self._shutdown:
            try:
                url = self.config.endpoints.chainlink_rtds_url

                self._ws = websocket.WebSocketApp(
                    url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )

                logger.info(f"Connecting to Chainlink feed: {url}")

                # Run in thread pool to not block event loop
                ping_int = self._ping_interval
                ping_to = self._ping_timeout
                await loop.run_in_executor(
                    None,
                    lambda: self._ws.run_forever(ping_interval=ping_int, ping_timeout=ping_to),
                )

                # If we get here, connection was successful then closed
                # Reset retry count on successful connection
                if self._connected:
                    self._retry_count = 0
                    self._reconnect_delay = self.config.websocket.initial_reconnect_delay

            except asyncio.CancelledError:
                logger.info("Chainlink connection cancelled")
                return
            except Exception as e:
                logger.error(f"Chainlink connection error: {e}")

            # Check circuit breaker
            self._retry_count += 1
            if self._retry_count >= self._max_retries:
                self._circuit_open = True
                logger.error(
                    f"Chainlink circuit breaker OPEN after {self._retry_count} failures. "
                    f"Feed will not auto-reconnect. Manual intervention required."
                )
                return

            # Check shutdown before reconnecting
            if self._shutdown:
                logger.info("Chainlink shutdown requested, stopping reconnection")
                return

            # Reconnect with exponential backoff
            logger.warning(
                f"Chainlink reconnecting in {self._reconnect_delay:.1f}s... "
                f"(attempt {self._retry_count}/{self._max_retries})"
            )
            await asyncio.sleep(self._reconnect_delay)

            # Check shutdown again after sleep
            if self._shutdown:
                logger.info("Chainlink shutdown requested, stopping reconnection")
                return

            self._reconnect_delay = min(
                self._reconnect_delay * 2, self._max_reconnect_delay
            )

    def reset_circuit_breaker(self):
        """Reset circuit breaker to allow reconnection."""
        self._circuit_open = False
        self._retry_count = 0
        self._reconnect_delay = self.config.websocket.initial_reconnect_delay
        logger.info("Chainlink circuit breaker reset")

    def disconnect(self):
        """Close the WebSocket connection and stop reconnection attempts."""
        self._shutdown = True  # Prevent reconnection attempts
        if self._ws:
            self._ws.close()
            self._connected = False
            logger.info("Chainlink WebSocket disconnected")

    def get_all_prices(self) -> dict[str, float]:
        """
        Get all current prices mapped by asset symbol.

        Returns:
            Dict of asset -> price (e.g., {"BTC": 104000.50, "ETH": 3200.25})
        """
        result = {}
        for symbol, price in self._prices.items():
            # Convert "btc/usd" -> "BTC"
            if "/" in symbol:
                asset = symbol.split("/")[0].upper()
            else:
                asset = symbol.upper()
            result[asset] = price
        return result
