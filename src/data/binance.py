"""
Binance Real-Time Price WebSocket client.

Connects to Binance's WebSocket for faster price updates (~50ms latency).
Used as a "leading indicator" to predict where Chainlink will go.

Strategy:
- Binance prices move faster than Chainlink (which aggregates from multiple sources)
- When Binance crosses a target before Chainlink, it signals Chainlink will likely follow
- Used as CONFIRMATION signal, not primary trading signal

Signal strength:
- Chainlink crossed target → STRONG (existing behavior)
- Chainlink near target + Binance crossed → MEDIUM (boost confidence)
- Only Binance crossed → WEAK (don't trade, just watch)
"""

import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable, Optional
from dataclasses import dataclass, field

import websocket

from ..config import BotConfig


logger = logging.getLogger(__name__)


# Binance symbol mapping: our asset -> Binance symbol
BINANCE_SYMBOLS = {
    "BTC": "btcusdt",
    "ETH": "ethusdt",
    "SOL": "solusdt",
    "XRP": "xrpusdt",
}

# Reverse mapping for lookups
SYMBOL_TO_ASSET = {v: k for k, v in BINANCE_SYMBOLS.items()}


@dataclass
class BinancePrice:
    """Real-time price from Binance."""

    symbol: str          # e.g., "btcusdt"
    price: float         # Current price
    timestamp: datetime  # Price timestamp

    @property
    def asset(self) -> str:
        """Extract asset symbol (e.g., 'BTC' from 'btcusdt')."""
        return SYMBOL_TO_ASSET.get(self.symbol.lower(), self.symbol.upper())


@dataclass
class BinanceFeed:
    """
    WebSocket client for Binance real-time price data.

    Connects to wss://stream.binance.com:9443/ws and subscribes
    to mini ticker streams for BTC, ETH, SOL, XRP.

    Updates are extremely fast (~50ms) compared to Chainlink (~500ms+).
    """

    config: BotConfig
    on_price_update: Optional[Callable[[BinancePrice], None]] = None

    # Internal state
    _ws: Optional[websocket.WebSocketApp] = None
    _prices: dict[str, float] = field(default_factory=dict)
    _last_update: dict[str, datetime] = field(default_factory=dict)
    _connected: bool = False
    _reconnect_delay: float = field(init=False)
    _max_reconnect_delay: float = field(init=False)
    _max_retries: int = field(init=False)
    _retry_count: int = 0
    _circuit_open: bool = False  # Circuit breaker state
    _shutdown: bool = False  # Graceful shutdown flag

    def __post_init__(self):
        """Initialize WebSocket settings from config."""
        self._reconnect_delay = self.config.websocket.initial_reconnect_delay
        self._max_reconnect_delay = self.config.websocket.max_reconnect_delay
        self._max_retries = self.config.websocket.max_retries

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected

    def get_price(self, asset: str) -> Optional[float]:
        """
        Get current Binance price for an asset.

        Args:
            asset: Asset symbol like "BTC", "ETH", etc.

        Returns:
            Current price or None if not available
        """
        symbol = BINANCE_SYMBOLS.get(asset.upper())
        if symbol:
            return self._prices.get(symbol)
        return None

    def get_all_prices(self) -> dict[str, float]:
        """
        Get all current prices mapped by asset.

        Returns:
            Dict of asset -> price (e.g., {"BTC": 104000.50, "ETH": 3200.25})
        """
        return {
            SYMBOL_TO_ASSET.get(symbol, symbol.upper()): price
            for symbol, price in self._prices.items()
        }

    def get_price_age(self, asset: str) -> Optional[float]:
        """
        Get how old the price data is in seconds.

        Args:
            asset: Asset symbol like "BTC"

        Returns:
            Age in seconds, or None if no price available
        """
        symbol = BINANCE_SYMBOLS.get(asset.upper())
        if symbol and symbol in self._last_update:
            age = (datetime.now(timezone.utc) - self._last_update[symbol]).total_seconds()
            return age
        return None

    def _build_stream_url(self) -> str:
        """Build the combined stream URL for all assets."""
        # Use combined stream for all assets in one connection
        streams = [f"{symbol}@miniTicker" for symbol in BINANCE_SYMBOLS.values()]
        stream_path = "/".join(streams)
        return f"{self.config.endpoints.binance_ws_url}/stream?streams={stream_path}"

    def _on_open(self, ws):
        """Handle WebSocket connection opened."""
        logger.info("Binance WebSocket connected")
        self._connected = True
        self._reconnect_delay = self.config.websocket.initial_reconnect_delay
        self._retry_count = 0

    def _on_message(self, ws, message: str):
        """Handle incoming WebSocket message."""
        if not message or not message.strip():
            return

        try:
            data = json.loads(message)

            # Combined stream format: {"stream": "btcusdt@miniTicker", "data": {...}}
            if "stream" in data and "data" in data:
                ticker_data = data["data"]
            else:
                # Direct ticker format (fallback)
                ticker_data = data

            # Extract price from mini ticker
            # Format: {"e": "24hrMiniTicker", "s": "BTCUSDT", "c": "104000.50", ...}
            event_type = ticker_data.get("e")
            if event_type == "24hrMiniTicker":
                symbol = ticker_data.get("s", "").lower()  # e.g., "btcusdt"
                close_price = ticker_data.get("c")  # Current close price
                event_time = ticker_data.get("E")  # Event time in ms

                if symbol in SYMBOL_TO_ASSET and close_price:
                    price = float(close_price)
                    self._prices[symbol] = price

                    # Parse timestamp
                    if event_time:
                        timestamp = datetime.fromtimestamp(
                            event_time / 1000, tz=timezone.utc
                        )
                    else:
                        timestamp = datetime.now(timezone.utc)

                    self._last_update[symbol] = timestamp

                    # Create price object
                    binance_price = BinancePrice(
                        symbol=symbol,
                        price=price,
                        timestamp=timestamp,
                    )

                    logger.debug(
                        f"Binance price: {SYMBOL_TO_ASSET[symbol]} = ${price:,.2f}"
                    )

                    # Notify callback
                    if self.on_price_update:
                        self.on_price_update(binance_price)

        except json.JSONDecodeError:
            # Silently ignore non-JSON messages (pings, etc.)
            pass
        except Exception as e:
            logger.error(f"Error processing Binance message: {e}")

    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        logger.error(f"Binance WebSocket error: {error}")
        self._connected = False

    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection closed."""
        logger.warning(
            f"Binance WebSocket closed: {close_status_code} - {close_msg}"
        )
        self._connected = False

    def connect(self):
        """
        Connect to Binance WebSocket (blocking).

        This method runs the WebSocket connection in a blocking manner.
        For async usage, use connect_async() instead.
        """
        url = self._build_stream_url()

        self._ws = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        logger.info(f"Connecting to Binance feed: {url}")
        self._ws.run_forever()

    async def connect_async(self):
        """
        Connect to Binance WebSocket asynchronously.

        Runs the blocking WebSocket in a thread pool executor.
        Handles automatic reconnection with exponential backoff and circuit breaker.
        """
        loop = asyncio.get_running_loop()

        while not self._circuit_open and not self._shutdown:
            try:
                url = self._build_stream_url()

                self._ws = websocket.WebSocketApp(
                    url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )

                logger.info(f"Connecting to Binance feed: {url}")

                # Run in thread pool to not block event loop
                await loop.run_in_executor(
                    None,
                    lambda: self._ws.run_forever(
                        ping_interval=self.config.websocket.ping_interval,
                        ping_timeout=self.config.websocket.ping_timeout,
                    ),
                )

                # If we get here, connection was successful then closed
                if self._connected:
                    self._retry_count = 0
                    self._reconnect_delay = self.config.websocket.initial_reconnect_delay

            except asyncio.CancelledError:
                logger.info("Binance connection cancelled")
                return
            except Exception as e:
                logger.error(f"Binance connection error: {e}")

            # Check circuit breaker
            self._retry_count += 1
            if self._retry_count >= self._max_retries:
                self._circuit_open = True
                logger.error(
                    f"Binance circuit breaker OPEN after {self._retry_count} failures. "
                    f"Feed will not auto-reconnect. Manual intervention required."
                )
                return

            # Check shutdown before reconnecting
            if self._shutdown:
                logger.info("Binance shutdown requested, stopping reconnection")
                return

            # Reconnect with exponential backoff
            logger.warning(
                f"Binance reconnecting in {self._reconnect_delay:.1f}s... "
                f"(attempt {self._retry_count}/{self._max_retries})"
            )
            await asyncio.sleep(self._reconnect_delay)

            # Check shutdown again after sleep
            if self._shutdown:
                logger.info("Binance shutdown requested, stopping reconnection")
                return

            self._reconnect_delay = min(
                self._reconnect_delay * 2, self._max_reconnect_delay
            )

    def reset_circuit_breaker(self):
        """Reset circuit breaker to allow reconnection."""
        self._circuit_open = False
        self._retry_count = 0
        self._reconnect_delay = self.config.websocket.initial_reconnect_delay
        logger.info("Binance circuit breaker reset")

    def disconnect(self):
        """Close the WebSocket connection and stop reconnection attempts."""
        self._shutdown = True
        if self._ws:
            self._ws.close()
            self._connected = False
            logger.info("Binance WebSocket disconnected")


# ============================================================================
# Multi-Timeframe Trend Analysis (REST API)
# ============================================================================

import requests
from datetime import timedelta

# Cache for klines data to avoid repeated API calls
_klines_cache: dict = {}
_cache_expiry: dict = {}
CACHE_DURATION = timedelta(minutes=5)  # Refresh every 5 minutes


def fetch_klines(
    asset: str,
    interval: str = "1h",
    limit: int = 2,
) -> list[dict]:
    """
    Fetch historical klines (candlestick) data from Binance REST API.

    Args:
        asset: Asset symbol (BTC, ETH, SOL, XRP)
        interval: Kline interval (1h, 4h, 1d)
        limit: Number of candles to fetch

    Returns:
        List of kline dicts with open, high, low, close, volume
    """
    symbol = BINANCE_SYMBOLS.get(asset.upper())
    if not symbol:
        return []

    cache_key = f"{symbol}:{interval}"
    now = datetime.now(timezone.utc)

    # Check cache
    if cache_key in _klines_cache:
        expiry = _cache_expiry.get(cache_key)
        if expiry and now < expiry:
            return _klines_cache[cache_key]

    # Fetch from Binance API
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        raw_klines = response.json()

        # Parse klines into dict format
        # [open_time, open, high, low, close, volume, close_time, ...]
        klines = []
        for k in raw_klines:
            klines.append({
                "open_time": k[0],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": k[6],
            })

        # Cache the result
        _klines_cache[cache_key] = klines
        _cache_expiry[cache_key] = now + CACHE_DURATION

        return klines

    except Exception as e:
        logger.debug(f"Failed to fetch klines for {asset} {interval}: {e}")
        return []


def calculate_trend_from_klines(klines: list[dict]) -> float:
    """
    Calculate trend direction from klines data.

    Returns:
        Trend value between -1 (strong downtrend) and +1 (strong uptrend)
        Based on price change percentage, capped at ±5%
    """
    if not klines or len(klines) < 1:
        return 0.0

    # Use most recent completed candle
    latest = klines[-1]
    open_price = latest["open"]
    close_price = latest["close"]

    if open_price <= 0:
        return 0.0

    # Calculate percentage change
    pct_change = (close_price - open_price) / open_price

    # Normalize to -1 to +1 range (±5% = ±1.0)
    # This means 5% move = max trend strength
    trend = pct_change / 0.05
    trend = max(-1.0, min(1.0, trend))

    return trend


def get_multi_timeframe_trends(asset: str) -> dict:
    """
    Get trend data across multiple timeframes for an asset.

    Args:
        asset: Asset symbol (BTC, ETH, SOL, XRP)

    Returns:
        Dict with trend_1h, trend_4h, trend_1d values (-1 to +1)
    """
    result = {
        "trend_1h": 0.0,
        "trend_4h": 0.0,
        "trend_1d": 0.0,
    }

    # Fetch klines for each timeframe
    for interval, key in [("1h", "trend_1h"), ("4h", "trend_4h"), ("1d", "trend_1d")]:
        klines = fetch_klines(asset, interval, limit=2)
        if klines:
            result[key] = calculate_trend_from_klines(klines)

    return result


def get_trend_alignment(trends: dict) -> str:
    """
    Determine if multiple timeframes are aligned.

    Args:
        trends: Dict with trend_1h, trend_4h, trend_1d

    Returns:
        "BULLISH" if all positive, "BEARISH" if all negative,
        "MIXED" if conflicting signals
    """
    t1h = trends.get("trend_1h", 0)
    t4h = trends.get("trend_4h", 0)
    t1d = trends.get("trend_1d", 0)

    positive = sum(1 for t in [t1h, t4h, t1d] if t > 0.1)
    negative = sum(1 for t in [t1h, t4h, t1d] if t < -0.1)

    if positive >= 2 and negative == 0:
        return "BULLISH"
    elif negative >= 2 and positive == 0:
        return "BEARISH"
    else:
        return "MIXED"
