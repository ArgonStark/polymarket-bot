"""
Polymarket CLOB (Central Limit Order Book) WebSocket client.

Connects to Polymarket's order book WebSocket for real-time
bid/ask updates and trade executions.
"""

import json
import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Callable, Optional
from dataclasses import dataclass, field

import websocket

from ..models import OrderBook, OrderBookLevel
from ..config import BotConfig


logger = logging.getLogger(__name__)


@dataclass
class CLOBFeed:
    """
    WebSocket client for Polymarket CLOB order book data.

    Connects to wss://ws-subscriptions-clob.polymarket.com/ws/market
    and subscribes to market channels for real-time order book updates.
    """

    config: BotConfig
    on_orderbook_update: Optional[Callable[[str, OrderBook], None]] = None
    on_trade: Optional[Callable[[dict], None]] = None

    # Internal state
    _ws: Optional[websocket.WebSocketApp] = None
    _orderbooks: dict[str, OrderBook] = field(default_factory=dict)
    _subscribed_tokens: set[str] = field(default_factory=set)
    _connected: bool = False
    _reconnect_delay: float = field(init=False, default=1.0)
    _max_reconnect_delay: float = field(init=False, default=60.0)
    _max_retries: int = field(init=False, default=10)
    _ping_interval: int = field(init=False, default=30)
    _ping_timeout: int = field(init=False, default=10)
    _retry_count: int = 0
    _circuit_open: bool = False  # Circuit breaker state
    _shutdown: bool = False  # Graceful shutdown flag
    _orderbook_lock: threading.Lock = field(default_factory=threading.Lock)  # Thread-safe orderbook access

    def __post_init__(self):
        """Initialize WebSocket settings from config."""
        self._reconnect_delay = self.config.websocket.initial_reconnect_delay
        self._max_reconnect_delay = self.config.websocket.max_reconnect_delay
        self._max_retries = self.config.websocket.max_retries
        self._ping_interval = self.config.websocket.ping_interval
        self._ping_timeout = self.config.websocket.ping_timeout

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected

    def get_orderbook(self, token_id: str) -> Optional[OrderBook]:
        """
        Get current order book for a token.

        Returns a thread-safe snapshot of the orderbook.

        Args:
            token_id: Token ID to get order book for

        Returns:
            OrderBook or None if not available
        """
        with self._orderbook_lock:
            ob = self._orderbooks.get(token_id)
            if ob is None:
                return None
            # Return a snapshot copy to avoid race conditions with WebSocket thread
            return OrderBook(
                token_id=ob.token_id,
                bids=list(ob.bids),
                asks=list(ob.asks),
                timestamp=ob.timestamp,
            )

    def get_best_bid(self, token_id: str) -> Optional[float]:
        """Get best bid price for a token (thread-safe)."""
        with self._orderbook_lock:
            ob = self._orderbooks.get(token_id)
            return ob.best_bid if ob else None

    def get_best_ask(self, token_id: str) -> Optional[float]:
        """Get best ask price for a token (thread-safe)."""
        with self._orderbook_lock:
            ob = self._orderbooks.get(token_id)
            return ob.best_ask if ob else None

    def _on_open(self, ws):
        """Handle WebSocket connection opened."""
        logger.info("CLOB WebSocket connected")
        self._connected = True
        self._reconnect_delay = self.config.websocket.initial_reconnect_delay
        self._retry_count = 0  # Reset retry count on successful connection

        # Re-subscribe to any previously subscribed tokens
        for token_id in self._subscribed_tokens:
            self._send_subscribe(ws, token_id)

    def _send_subscribe(self, ws, token_id: str):
        """Send subscription message for a token."""
        subscribe_msg = {
            "type": "subscribe",
            "channel": "market",
            "assets_ids": [token_id],
        }
        try:
            ws.send(json.dumps(subscribe_msg))
            logger.debug(f"Subscribed to token: {token_id[:16]}...")
        except Exception as e:
            logger.error(f"Failed to subscribe to {token_id[:16]}: {e}")

    def subscribe(self, token_id: str):
        """
        Subscribe to order book updates for a token.

        Args:
            token_id: Token ID to subscribe to
        """
        self._subscribed_tokens.add(token_id)

        # Initialize empty order book (thread-safe)
        with self._orderbook_lock:
            if token_id not in self._orderbooks:
                self._orderbooks[token_id] = OrderBook(token_id=token_id)

        # If connected, send subscription immediately
        if self._ws and self._connected:
            self._send_subscribe(self._ws, token_id)

    def unsubscribe(self, token_id: str):
        """
        Unsubscribe from order book updates for a token.

        Args:
            token_id: Token ID to unsubscribe from
        """
        self._subscribed_tokens.discard(token_id)

        if self._ws and self._connected:
            unsubscribe_msg = {
                "type": "unsubscribe",
                "channel": "market",
                "assets_ids": [token_id],
            }
            try:
                self._ws.send(json.dumps(unsubscribe_msg))
                logger.info(f"Unsubscribed from token: {token_id[:16]}...")
            except Exception as e:
                logger.error(f"Failed to unsubscribe from {token_id[:16]}: {e}")

    def _on_message(self, ws, message: str):
        """Handle incoming WebSocket message."""
        # Skip empty messages
        if not message or not message.strip():
            return

        try:
            data = json.loads(message)

            # Handle list responses (batch messages)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        self._process_message(item)
                return

            # Handle single dict message
            if isinstance(data, dict):
                self._process_message(data)

        except json.JSONDecodeError:
            # Silently ignore parse errors for keepalive/ping messages
            if message not in ('', 'ping', 'pong'):
                logger.debug(f"Non-JSON CLOB message: {message[:50]}...")
        except Exception as e:
            logger.error(f"Error processing CLOB message: {e}")

    def _process_message(self, data: dict):
        """Process a single CLOB message."""
        msg_type = data.get("type")

        if msg_type == "book":
            # Full order book snapshot
            self._handle_book_snapshot(data)

        elif msg_type == "price_change":
            # Price level update
            self._handle_price_change(data)

        elif msg_type == "trade":
            # Trade execution
            self._handle_trade(data)

        elif msg_type == "subscribed":
            logger.debug(f"Subscription confirmed: {data}")

        elif msg_type == "error":
            logger.error(f"CLOB WebSocket error: {data}")

    def _handle_book_snapshot(self, data: dict):
        """Handle full order book snapshot."""
        asset_id = data.get("asset_id")
        if not asset_id:
            return

        bids = []
        asks = []

        for bid in data.get("bids", []):
            bids.append(OrderBookLevel(
                price=float(bid.get("price", 0)),
                size=float(bid.get("size", 0)),
            ))

        for ask in data.get("asks", []):
            asks.append(OrderBookLevel(
                price=float(ask.get("price", 0)),
                size=float(ask.get("size", 0)),
            ))

        orderbook = OrderBook(
            token_id=asset_id,
            bids=bids,
            asks=asks,
            timestamp=datetime.now(timezone.utc),
        )

        # Thread-safe assignment
        with self._orderbook_lock:
            self._orderbooks[asset_id] = orderbook

        logger.debug(
            f"Book snapshot {asset_id[:16]}: "
            f"bid={orderbook.best_bid}, ask={orderbook.best_ask}"
        )

        # Notify callback
        if self.on_orderbook_update:
            self.on_orderbook_update(asset_id, orderbook)

    def _handle_price_change(self, data: dict):
        """Handle price level update."""
        asset_id = data.get("asset_id")
        if not asset_id:
            return

        side = data.get("side")  # "BUY" or "SELL"
        price = float(data.get("price", 0))
        size = float(data.get("size", 0))

        # Thread-safe orderbook modification
        with self._orderbook_lock:
            if asset_id not in self._orderbooks:
                return

            orderbook = self._orderbooks[asset_id]

            if side == "BUY":
                # Update bids
                orderbook.bids = [b for b in orderbook.bids if b.price != price]
                if size > 0:
                    orderbook.bids.append(OrderBookLevel(price=price, size=size))
                    orderbook.bids.sort(key=lambda x: x.price, reverse=True)
            elif side == "SELL":
                # Update asks
                orderbook.asks = [a for a in orderbook.asks if a.price != price]
                if size > 0:
                    orderbook.asks.append(OrderBookLevel(price=price, size=size))
                    orderbook.asks.sort(key=lambda x: x.price)

            orderbook.timestamp = datetime.now(timezone.utc)

        # Notify callback (outside lock to avoid blocking)
        if self.on_orderbook_update:
            self.on_orderbook_update(asset_id, orderbook)

    def _handle_trade(self, data: dict):
        """Handle trade execution."""
        logger.debug(f"Trade: {data}")

        if self.on_trade:
            self.on_trade(data)

    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        logger.error(f"CLOB WebSocket error: {error}")
        self._connected = False

    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection closed."""
        logger.warning(
            f"CLOB WebSocket closed: {close_status_code} - {close_msg}"
        )
        self._connected = False

    def connect(self):
        """
        Connect to CLOB WebSocket (blocking).

        This method runs the WebSocket connection in a blocking manner.
        For async usage, use connect_async() instead.
        """
        url = self.config.endpoints.clob_ws_url

        self._ws = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        logger.info(f"Connecting to CLOB feed: {url}")
        self._ws.run_forever()

    async def connect_async(self):
        """
        Connect to CLOB WebSocket asynchronously.

        Runs the blocking WebSocket in a thread pool executor.
        Handles automatic reconnection with exponential backoff and circuit breaker.
        """
        loop = asyncio.get_running_loop()

        while not self._circuit_open and not self._shutdown:
            try:
                url = self.config.endpoints.clob_ws_url

                self._ws = websocket.WebSocketApp(
                    url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )

                logger.info(f"Connecting to CLOB feed: {url}")

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
                logger.info("CLOB connection cancelled")
                return
            except Exception as e:
                logger.error(f"CLOB connection error: {e}")

            # Check circuit breaker
            self._retry_count += 1
            if self._retry_count >= self._max_retries:
                self._circuit_open = True
                logger.error(
                    f"CLOB circuit breaker OPEN after {self._retry_count} failures. "
                    f"Feed will not auto-reconnect. Manual intervention required."
                )
                return

            # Check shutdown before reconnecting
            if self._shutdown:
                logger.info("CLOB shutdown requested, stopping reconnection")
                return

            # Reconnect with exponential backoff
            logger.warning(
                f"CLOB reconnecting in {self._reconnect_delay:.1f}s... "
                f"(attempt {self._retry_count}/{self._max_retries})"
            )
            await asyncio.sleep(self._reconnect_delay)

            # Check shutdown again after sleep
            if self._shutdown:
                logger.info("CLOB shutdown requested, stopping reconnection")
                return

            self._reconnect_delay = min(
                self._reconnect_delay * 2, self._max_reconnect_delay
            )

    def reset_circuit_breaker(self):
        """Reset circuit breaker to allow reconnection."""
        self._circuit_open = False
        self._retry_count = 0
        self._reconnect_delay = self.config.websocket.initial_reconnect_delay
        logger.info("CLOB circuit breaker reset")

    def disconnect(self):
        """Close the WebSocket connection and stop reconnection attempts."""
        self._shutdown = True  # Prevent reconnection attempts
        if self._ws:
            self._ws.close()
            self._connected = False
            logger.info("CLOB WebSocket disconnected")

    def get_all_orderbooks(self) -> dict[str, OrderBook]:
        """Get all current order books."""
        return self._orderbooks.copy()
