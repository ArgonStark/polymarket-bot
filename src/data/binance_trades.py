"""
Binance aggTrade WebSocket for real-time Order Flow Imbalance (OFI).

Connects to Binance's aggTrade stream for BTC, ETH, SOL, XRP.
aggTrade provides individual trade events with buyer/seller maker flag,
enabling calculation of Order Flow Imbalance — the #1 short-term
predictive feature for crypto price direction (53% accuracy OOS).

OFI = (buy_volume - sell_volume) / (buy_volume + sell_volume)
Range: [-1, +1] where +1 = all buy pressure, -1 = all sell pressure.
"""

import json
import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import websocket

from ..config import BotConfig

logger = logging.getLogger(__name__)

# Reuse symbol mapping from binance.py
BINANCE_SYMBOLS = {
    "BTC": "btcusdt",
    "ETH": "ethusdt",
    "SOL": "solusdt",
    "XRP": "xrpusdt",
    "DOGE": "dogeusdt",
    "BNB": "bnbusdt",
    # HYPE intentionally absent — not on Binance spot (OFI neutral for it)
}
SYMBOL_TO_ASSET = {v.upper(): k for k, v in BINANCE_SYMBOLS.items()}


@dataclass
class _Trade:
    """Single aggregated trade."""
    timestamp_ms: int
    price: float
    qty_usd: float  # quantity * price
    is_buy: bool


@dataclass
class TradeAccumulator:
    """Per-symbol rolling trade buffer."""
    trades: deque = field(default_factory=lambda: deque(maxlen=2000))
    last_update_ms: int = 0

    def add(self, trade: _Trade):
        self.trades.append(trade)
        self.last_update_ms = trade.timestamp_ms

    def _window(self, window_seconds: float) -> list[_Trade]:
        """Return trades within the last `window_seconds`."""
        cutoff_ms = int((time.time() - window_seconds) * 1000)
        return [t for t in self.trades if t.timestamp_ms >= cutoff_ms]

    def ofi(self, window_seconds: float = 30.0) -> float:
        """Order Flow Imbalance in [-1, +1]."""
        trades = self._window(window_seconds)
        if not trades:
            return 0.0
        buy_vol = sum(t.qty_usd for t in trades if t.is_buy)
        sell_vol = sum(t.qty_usd for t in trades if not t.is_buy)
        total = buy_vol + sell_vol
        if total == 0:
            return 0.0
        return (buy_vol - sell_vol) / total

    def ofi_acceleration(self) -> float:
        """Recent 10s OFI minus previous 10s OFI."""
        recent = self.ofi(10.0)
        # OFI over 10-20s ago window
        now_ms = int(time.time() * 1000)
        cutoff_recent = now_ms - 10_000
        cutoff_old = now_ms - 20_000
        old_trades = [t for t in self.trades if cutoff_old <= t.timestamp_ms < cutoff_recent]
        if not old_trades:
            return recent  # No old data, acceleration = current OFI
        buy_vol = sum(t.qty_usd for t in old_trades if t.is_buy)
        sell_vol = sum(t.qty_usd for t in old_trades if not t.is_buy)
        total = buy_vol + sell_vol
        old_ofi = (buy_vol - sell_vol) / total if total > 0 else 0.0
        return recent - old_ofi

    def trade_rate(self, window_seconds: float = 10.0) -> float:
        """Trades per second over the window."""
        trades = self._window(window_seconds)
        if not trades or window_seconds <= 0:
            return 0.0
        return len(trades) / window_seconds

    def data_age_seconds(self) -> Optional[float]:
        """Seconds since last trade, or None if no trades."""
        if self.last_update_ms == 0:
            return None
        return (time.time() * 1000 - self.last_update_ms) / 1000.0


@dataclass
class BinanceTradeFeed:
    """
    WebSocket client for Binance aggTrade stream.

    Provides real-time Order Flow Imbalance per asset.
    Follows the same pattern as BinanceFeed (connect_async, circuit breaker, etc.).
    """

    config: BotConfig

    # Internal state
    _ws: Optional[websocket.WebSocketApp] = field(default=None, repr=False)
    _accumulators: dict[str, TradeAccumulator] = field(default_factory=dict)
    _connected: bool = False
    _reconnect_delay: float = field(init=False)
    _max_reconnect_delay: float = field(init=False)
    _max_retries: int = field(init=False)
    _retry_count: int = 0
    _circuit_open: bool = False
    _shutdown: bool = False

    def __post_init__(self):
        self._reconnect_delay = self.config.websocket.initial_reconnect_delay
        self._max_reconnect_delay = self.config.websocket.max_reconnect_delay
        self._max_retries = self.config.websocket.max_retries
        # Pre-create accumulators
        for asset, symbol in BINANCE_SYMBOLS.items():
            self._accumulators[symbol.upper()] = TradeAccumulator()

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_ofi(self, asset: str, window_seconds: float = 30.0) -> float:
        """Get Order Flow Imbalance for an asset. Returns 0.0 if no data."""
        symbol = BINANCE_SYMBOLS.get(asset.upper(), "").upper()
        acc = self._accumulators.get(symbol)
        if acc is None:
            return 0.0
        return acc.ofi(window_seconds)

    def get_ofi_acceleration(self, asset: str) -> float:
        """Recent 10s OFI minus previous 10s OFI."""
        symbol = BINANCE_SYMBOLS.get(asset.upper(), "").upper()
        acc = self._accumulators.get(symbol)
        if acc is None:
            return 0.0
        return acc.ofi_acceleration()

    def get_trade_rate(self, asset: str, window_seconds: float = 10.0) -> float:
        """Trades per second."""
        symbol = BINANCE_SYMBOLS.get(asset.upper(), "").upper()
        acc = self._accumulators.get(symbol)
        if acc is None:
            return 0.0
        return acc.trade_rate(window_seconds)

    def get_data_age(self, asset: str) -> Optional[float]:
        """Seconds since last trade for this asset."""
        symbol = BINANCE_SYMBOLS.get(asset.upper(), "").upper()
        acc = self._accumulators.get(symbol)
        if acc is None:
            return None
        return acc.data_age_seconds()

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    def _build_stream_url(self) -> str:
        streams = [f"{symbol}@aggTrade" for symbol in BINANCE_SYMBOLS.values()]
        stream_path = "/".join(streams)
        return f"{self.config.endpoints.binance_ws_url}/stream?streams={stream_path}"

    def _on_open(self, ws):
        logger.info("Binance aggTrade WebSocket connected")
        self._connected = True
        self._reconnect_delay = self.config.websocket.initial_reconnect_delay
        self._retry_count = 0

    def _on_message(self, ws, message: str):
        if not message or not message.strip():
            return
        try:
            data = json.loads(message)

            # Combined stream format: {"stream": "btcusdt@aggTrade", "data": {...}}
            if "stream" in data and "data" in data:
                trade_data = data["data"]
            else:
                trade_data = data

            event_type = trade_data.get("e")
            if event_type != "aggTrade":
                return

            symbol = trade_data.get("s", "")  # "BTCUSDT"
            price_str = trade_data.get("p")
            qty_str = trade_data.get("q")
            is_maker_buyer = trade_data.get("m", False)
            event_time = trade_data.get("T", 0)  # Trade time in ms

            if not symbol or not price_str or not qty_str:
                return

            price = float(price_str)
            qty = float(qty_str)
            qty_usd = price * qty
            # m=true means buyer was maker → taker was SELLER → sell trade
            is_buy = not is_maker_buyer

            acc = self._accumulators.get(symbol)
            if acc is None:
                return

            acc.add(_Trade(
                timestamp_ms=event_time,
                price=price,
                qty_usd=qty_usd,
                is_buy=is_buy,
            ))

        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.error(f"Error processing aggTrade message: {e}")

    def _on_error(self, ws, error):
        logger.error(f"Binance aggTrade WebSocket error: {error}")
        self._connected = False

    def _on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"Binance aggTrade WebSocket closed: {close_status_code} - {close_msg}")
        self._connected = False

    async def connect_async(self):
        """Connect with automatic reconnection and circuit breaker."""
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
                logger.info(f"Connecting to Binance aggTrade feed: {url}")
                await loop.run_in_executor(
                    None,
                    lambda: self._ws.run_forever(
                        ping_interval=self.config.websocket.ping_interval,
                        ping_timeout=self.config.websocket.ping_timeout,
                    ),
                )
                if self._connected:
                    self._retry_count = 0
                    self._reconnect_delay = self.config.websocket.initial_reconnect_delay

            except asyncio.CancelledError:
                logger.info("Binance aggTrade connection cancelled")
                return
            except Exception as e:
                logger.error(f"Binance aggTrade connection error: {e}")

            self._retry_count += 1
            if self._retry_count >= self._max_retries:
                self._circuit_open = True
                logger.error(
                    f"Binance aggTrade circuit breaker OPEN after {self._retry_count} failures."
                )
                return

            if self._shutdown:
                return

            logger.warning(
                f"Binance aggTrade reconnecting in {self._reconnect_delay:.1f}s "
                f"(attempt {self._retry_count}/{self._max_retries})"
            )
            await asyncio.sleep(self._reconnect_delay)

            if self._shutdown:
                return

            self._reconnect_delay = min(
                self._reconnect_delay * 2, self._max_reconnect_delay
            )

    def disconnect(self):
        self._shutdown = True
        if self._ws:
            self._ws.close()
            self._connected = False
            logger.info("Binance aggTrade WebSocket disconnected")

    def reset_circuit_breaker(self):
        self._circuit_open = False
        self._retry_count = 0
        self._reconnect_delay = self.config.websocket.initial_reconnect_delay
        logger.info("Binance aggTrade circuit breaker reset")
