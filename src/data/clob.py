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
    _reconnected_at: Optional[datetime] = None  # When last reconnect completed
    _warmup_seconds: float = 5.0  # Seconds of fresh data required after reconnect

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

    @property
    def is_warmed_up(self) -> bool:
        """True when enough time has passed since reconnect for orderbooks to refresh."""
        if self._reconnected_at is None:
            return True  # First connection, no warmup needed
        elapsed = (datetime.now(timezone.utc) - self._reconnected_at).total_seconds()
        return elapsed >= self._warmup_seconds

    def _on_open(self, ws):
        """Handle WebSocket connection opened."""
        logger.info("CLOB WebSocket connected")
        self._connected = True
        self._reconnect_delay = self.config.websocket.initial_reconnect_delay

        # Track reconnect time for warmup gate (skip on first connect)
        if self._retry_count > 0:
            self._reconnected_at = datetime.now(timezone.utc)
            # Invalidate stale orderbook timestamps so data_health catches them
            with self._orderbook_lock:
                for ob in self._orderbooks.values():
                    ob.timestamp = datetime.min.replace(tzinfo=timezone.utc)
            logger.info(
                "CLOB_WARMUP started: invalidated %d orderbooks, warming up for %.0fs",
                len(self._orderbooks), self._warmup_seconds,
            )

        self._retry_count = 0  # Reset retry count on successful connection

        # Send initial subscription with ALL tokens in a single message
        # Polymarket initial subscribe: {"assets_ids": [...], "type": "market"}
        if self._subscribed_tokens:
            token_list = list(self._subscribed_tokens)
            subscribe_msg = {
                "assets_ids": token_list,
                "type": "market",
            }
            try:
                ws.send(json.dumps(subscribe_msg))
                logger.info(
                    "CLOB subscribed to %d tokens on connect",
                    len(token_list),
                )
            except Exception as e:
                logger.error(f"Failed to send initial subscription: {e}")

    def _send_subscribe(self, ws, token_ids: list[str]):
        """
        Send subscription for additional tokens after initial connect.

        Uses the Polymarket 'operation' format for adding tokens to an
        existing subscription.
        """
        if not token_ids:
            return
        subscribe_msg = {
            "assets_ids": token_ids,
            "operation": "subscribe",
        }
        try:
            ws.send(json.dumps(subscribe_msg))
            logger.debug(
                "CLOB subscribe add %d tokens: %s...",
                len(token_ids),
                token_ids[0][:16] if token_ids else "",
            )
        except Exception as e:
            logger.error(f"Failed to subscribe to tokens: {e}")

    def subscribe(self, token_id: str):
        """
        Subscribe to order book updates for a token.

        Args:
            token_id: Token ID to subscribe to
        """
        is_new = token_id not in self._subscribed_tokens
        self._subscribed_tokens.add(token_id)

        # Initialize empty order book (thread-safe)
        with self._orderbook_lock:
            if token_id not in self._orderbooks:
                self._orderbooks[token_id] = OrderBook(token_id=token_id)

        # If connected and this is a new token, send add-subscription
        if is_new and self._ws and self._connected:
            self._send_subscribe(self._ws, [token_id])

    def unsubscribe(self, token_id: str):
        """
        Unsubscribe from order book updates for a token.

        Args:
            token_id: Token ID to unsubscribe from
        """
        self._subscribed_tokens.discard(token_id)

        if self._ws and self._connected:
            unsubscribe_msg = {
                "assets_ids": [token_id],
                "operation": "unsubscribe",
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
        # Polymarket uses "event_type" for market channel messages
        msg_type = data.get("event_type") or data.get("type")

        if msg_type == "book":
            # Full order book snapshot
            self._handle_book_snapshot(data)

        elif msg_type == "price_change":
            # Price level update (contains price_changes array)
            self._handle_price_change(data)

        elif msg_type == "last_trade_price":
            # Trade execution
            self._handle_trade(data)

        elif msg_type == "best_bid_ask":
            # Best bid/ask update — lightweight, use to keep MarketState fresh
            self._handle_best_bid_ask(data)

        elif msg_type == "market_resolved":
            logger.info("CLOB market_resolved: %s", data.get("market", "?"))

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
        """
        Handle price_change event.

        Polymarket format:
        {
            "event_type": "price_change",
            "market": "<condition_id>",
            "timestamp": "...",
            "price_changes": [
                {"asset_id": "...", "price": "0.5", "size": "200",
                 "side": "BUY", "best_bid": "0.5", "best_ask": "1"},
                ...
            ]
        }
        """
        changes = data.get("price_changes", [])

        # Track which orderbooks were modified so we can notify callbacks
        modified_books = {}

        with self._orderbook_lock:
            for change in changes:
                asset_id = change.get("asset_id")
                if not asset_id or asset_id not in self._orderbooks:
                    continue

                side = change.get("side")  # "BUY" or "SELL"
                price = float(change.get("price", 0))
                size = float(change.get("size", 0))

                orderbook = self._orderbooks[asset_id]

                if side == "BUY":
                    orderbook.bids = [b for b in orderbook.bids if b.price != price]
                    if size > 0:
                        orderbook.bids.append(OrderBookLevel(price=price, size=size))
                        orderbook.bids.sort(key=lambda x: x.price, reverse=True)
                elif side == "SELL":
                    orderbook.asks = [a for a in orderbook.asks if a.price != price]
                    if size > 0:
                        orderbook.asks.append(OrderBookLevel(price=price, size=size))
                        orderbook.asks.sort(key=lambda x: x.price)

                orderbook.timestamp = datetime.now(timezone.utc)
                modified_books[asset_id] = orderbook

        # Notify callbacks outside lock
        if self.on_orderbook_update:
            for asset_id, orderbook in modified_books.items():
                self.on_orderbook_update(asset_id, orderbook)

    def _handle_best_bid_ask(self, data: dict):
        """
        Handle best_bid_ask event — lightweight price update.

        Format: {"event_type": "best_bid_ask", "asset_id": "...",
                 "best_bid": "0.73", "best_ask": "0.77", ...}
        """
        asset_id = data.get("asset_id")
        if not asset_id:
            return

        best_bid = float(data.get("best_bid", 0))
        best_ask = float(data.get("best_ask", 0))

        with self._orderbook_lock:
            if asset_id not in self._orderbooks:
                return

            orderbook = self._orderbooks[asset_id]

            # Update top-of-book: ensure at least the best level exists
            if best_bid > 0:
                # Remove stale best bid levels above the new best_bid, add new one
                orderbook.bids = [b for b in orderbook.bids if b.price <= best_bid]
                if not orderbook.bids or orderbook.bids[0].price != best_bid:
                    # Synthetic level (size unknown but non-zero)
                    orderbook.bids = [OrderBookLevel(price=best_bid, size=1.0)] + [
                        b for b in orderbook.bids if b.price < best_bid
                    ]
            if best_ask > 0:
                orderbook.asks = [a for a in orderbook.asks if a.price >= best_ask]
                if not orderbook.asks or orderbook.asks[0].price != best_ask:
                    orderbook.asks = [OrderBookLevel(price=best_ask, size=1.0)] + [
                        a for a in orderbook.asks if a.price > best_ask
                    ]

            orderbook.timestamp = datetime.now(timezone.utc)

        if self.on_orderbook_update:
            self.on_orderbook_update(asset_id, orderbook)

    def _handle_trade(self, data: dict):
        """Handle trade execution (last_trade_price event)."""
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
