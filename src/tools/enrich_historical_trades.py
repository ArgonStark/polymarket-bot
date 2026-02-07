"""
Historical Trade Data Enrichment Tool

Fetches historical closed positions from Polymarket and enriches them with
market context from Binance for ML training.

Usage:
    python -m src.tools.enrich_historical_trades --wallet 0x63ce342161250d705dc0b16df89036c8e5f9ba9a

Data Sources:
    - Polymarket closed-positions API: Trade outcomes with P&L
    - Binance historical klines API: OHLCV data for indicators

Performance:
    With --binance-batch-days (default), Binance klines are fetched once per day
    per symbol, reducing API calls from ~N_trades to ~N_days * N_symbols.
    Target: 500+ trades/minute after preloading.
"""

import json
import time
import math
import logging
import argparse
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Set
from pathlib import Path

import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
BINANCE_API = "https://api.binance.com/api/v3"
POLYMARKET_DATA_API = "https://data-api.polymarket.com"

# Asset detection patterns
ASSET_PATTERNS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
    "XRP": ["xrp", "ripple"],
}

# Asset to Binance symbol mapping
ASSET_TO_SYMBOL = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
}

# Rate limiting
BINANCE_RATE_LIMIT = 0.12  # seconds between requests
POLYMARKET_RATE_LIMIT = 0.25

# Binance API limits
BINANCE_MAX_KLINES_PER_REQUEST = 1000  # Max klines per request
MS_PER_MINUTE = 60_000
MS_PER_DAY = 86_400_000


@dataclass
class EnrichmentStats:
    """Track enrichment progress and performance metrics."""
    processed: int = 0
    new_count: int = 0
    skipped_existing: int = 0
    skipped_reason: int = 0
    errors: int = 0
    binance_requests: int = 0
    start_time: float = field(default_factory=time.time)
    skip_reasons: Dict[str, int] = field(default_factory=dict)

    def record_skip(self, reason: str) -> None:
        self.skipped_reason += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def avg_per_100(self) -> float:
        if self.processed == 0:
            return 0.0
        return (self.elapsed() / self.processed) * 100

    def trades_per_minute(self) -> float:
        elapsed = self.elapsed()
        if elapsed == 0:
            return 0.0
        return (self.processed / elapsed) * 60


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:
            ts = ts / 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _isoformat(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_ts(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, str):
        try:
            return datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    if isinstance(x, (int, float)):
        ts = float(x)
        if ts > 1e12:
            ts = ts / 1000.0
        return ts
    return None


@dataclass
class EnrichedTrade:
    """A trade enriched with market context for ML training."""

    # Basic trade info
    trade_id: str
    timestamp: str
    timestamp_ms: int
    asset: str
    side: str  # "UP" or "DOWN" (what they bet on)
    outcome: str  # "WIN" or "LOSS"
    entry_price: float
    size_usd: float
    realized_pnl: float

    # Price context at entry
    asset_price: float = 0.0
    btc_price: float = 0.0
    eth_price: float = 0.0
    sol_price: float = 0.0

    # Technical indicators
    rsi_14: float = 50.0
    rsi_7: float = 50.0
    macd_histogram: float = 0.0
    macd_signal: str = "none"

    # Bollinger Bands
    bb_bandwidth: float = 0.0
    bb_position: str = "middle"

    # Stochastic
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    stoch_signal: str = "neutral"

    # Moving averages
    sma_20: float = 0.0
    sma_50: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    price_vs_sma20: float = 0.0
    price_vs_sma50: float = 0.0

    # Trend
    trend_1h: float = 0.0
    trend_4h: float = 0.0
    trend_1d: float = 0.0
    trend_direction: str = "neutral"

    # Volatility
    volatility_1h: float = 0.0
    volatility_24h: float = 0.0
    atr_14: float = 0.0

    # Volume
    volume_ratio: float = 1.0
    volume_24h: float = 0.0

    # Time context
    hour_utc: int = 0
    day_of_week: int = 0
    is_weekend: bool = False
    is_asia_session: bool = False
    is_europe_session: bool = False
    is_us_session: bool = False

    # Market structure
    higher_high: bool = False
    lower_low: bool = False
    consolidating: bool = False


class KlineCache:
    """LRU cache for Binance kline windows."""

    def __init__(self, max_size: int = 5000):
        self.max_size = max_size
        self._store: OrderedDict[Tuple[str, str, int, int], List[Dict[str, Any]]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _normalize_key(symbol: str, interval: str, start_ms: int, end_ms: int) -> Tuple[str, str, int, int]:
        norm_start = int(start_ms // 60000) * 60000
        norm_end = int(end_ms // 60000) * 60000
        return symbol, interval, norm_start, norm_end

    def get(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> Optional[List[Dict[str, Any]]]:
        key = self._normalize_key(symbol, interval, start_ms, end_ms)
        if key in self._store:
            self._store.move_to_end(key)
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def set(self, symbol: str, interval: str, start_ms: int, end_ms: int, value: List[Dict[str, Any]]) -> None:
        key = self._normalize_key(symbol, interval, start_ms, end_ms)
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def stats(self) -> Tuple[int, int, float]:
        total = self.hits + self.misses
        rate = (self.hits / total * 100.0) if total > 0 else 0.0
        return self.hits, self.misses, rate


class BinanceHistoricalData:
    """Fetches historical OHLCV data from Binance with day-level batching."""

    def __init__(
        self,
        max_cache_windows: int = 5000,
        batch_days: bool = True,
        interval: str = "1m",
        timeout: int = 15,
    ):
        self.cache: Dict[str, List[Dict]] = {}
        self.last_request = 0
        self.price_cache: Dict[str, float] = {}
        self.window_cache = KlineCache(max_cache_windows)
        self.batch_days = batch_days
        self.interval = interval
        self.timeout = timeout
        # day_cache[symbol][day_start_ms][minute_ms] = close_price
        self.day_cache: Dict[str, Dict[int, Dict[int, float]]] = {}
        self.request_count = 0
        self.prefetch_request_count = 0

    def _rate_limit(self):
        elapsed = time.time() - self.last_request
        if elapsed < BINANCE_RATE_LIMIT:
            time.sleep(BINANCE_RATE_LIMIT - elapsed)
        self.last_request = time.time()

    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
        limit: int = 500
    ) -> List[Dict]:
        """Fetch historical klines from Binance."""
        cache_key = f"{symbol}_{interval}_{start_time}_{end_time}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        cached = self.window_cache.get(symbol, interval, start_time, end_time)
        if cached is not None:
            return cached

        self._rate_limit()

        try:
            self.request_count += 1
            response = requests.get(
                f"{BINANCE_API}/klines",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": start_time,
                    "endTime": end_time,
                    "limit": limit,
                },
                timeout=self.timeout
            )
            response.raise_for_status()

            raw_klines = response.json()
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

            self.cache[cache_key] = klines
            self.window_cache.set(symbol, interval, start_time, end_time, klines)
            return klines

        except Exception as e:
            logger.debug(f"Failed to fetch klines for {symbol}: {e}")
            return []

    def get_price_at_time(self, symbol: str, timestamp_ms: int) -> Optional[float]:
        """Get the price at a specific timestamp."""
        cache_key = f"{symbol}_{timestamp_ms // 60000}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]

        if self.batch_days:
            price = self._get_price_from_day_cache(symbol, timestamp_ms)
            if price is not None:
                self.price_cache[cache_key] = price
                return price

        klines = self.get_klines(
            symbol=symbol,
            interval="1m",
            start_time=timestamp_ms - 60000,
            end_time=timestamp_ms + 60000,
            limit=3,
        )

        if klines:
            price = klines[0]["close"]
            self.price_cache[cache_key] = price
            return price
        return None

    def get_ohlcv_context(
        self,
        symbol: str,
        timestamp_ms: int,
        lookback_hours: int = 48
    ) -> List[Dict]:
        """Get OHLCV data for context around a timestamp."""
        start_time = timestamp_ms - (lookback_hours * 60 * 60 * 1000)
        return self.get_klines(
            symbol=symbol,
            interval="1h",
            start_time=start_time,
            end_time=timestamp_ms,
            limit=lookback_hours
        )

    def _get_day_store(self, symbol: str) -> Dict[int, Dict[int, float]]:
        store = self.day_cache.get(symbol)
        if store is None:
            store = {}
            self.day_cache[symbol] = store
        return store

    def _get_price_from_day_cache(self, symbol: str, timestamp_ms: int) -> Optional[float]:
        day_start_ms = (timestamp_ms // MS_PER_DAY) * MS_PER_DAY
        minute_ms = (timestamp_ms // MS_PER_MINUTE) * MS_PER_MINUTE
        day_store = self.day_cache.get(symbol, {}).get(day_start_ms, {})
        return day_store.get(minute_ms)

    def _fetch_day_klines(self, symbol: str, day_start_ms: int) -> Dict[int, float]:
        """Fetch all 1m klines for a single day. Returns minute_ms -> close price."""
        day_end_ms = day_start_ms + MS_PER_DAY - 1
        klines: List[Dict[str, Any]] = []

        # A day has 1440 minutes; need 2 requests of 1000 each (or 720 each)
        cursor = day_start_ms
        while cursor <= day_end_ms:
            chunk_end = min(cursor + BINANCE_MAX_KLINES_PER_REQUEST * MS_PER_MINUTE - 1, day_end_ms)

            self._rate_limit()
            self.request_count += 1
            self.prefetch_request_count += 1

            try:
                response = requests.get(
                    f"{BINANCE_API}/klines",
                    params={
                        "symbol": symbol,
                        "interval": self.interval,
                        "startTime": cursor,
                        "endTime": chunk_end,
                        "limit": BINANCE_MAX_KLINES_PER_REQUEST,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                raw_klines = response.json()

                for k in raw_klines:
                    klines.append({
                        "open_time": k[0],
                        "close": float(k[4]),
                    })
            except Exception as e:
                logger.warning(f"Failed to fetch klines for {symbol} day {day_start_ms}: {e}")

            cursor = chunk_end + 1

        minute_prices: Dict[int, float] = {}
        for k in klines:
            minute_prices[int(k["open_time"])] = k["close"]
        return minute_prices

    def prefetch_days(
        self,
        symbol: str,
        day_starts_ms: List[int],
        max_days: Optional[int] = None,
    ) -> int:
        """
        Prefetch all 1m klines for the given days.

        Args:
            symbol: Binance symbol (e.g., BTCUSDT)
            day_starts_ms: List of day start timestamps in milliseconds
            max_days: Optional limit on number of days to prefetch (most recent first)

        Returns:
            Number of days actually prefetched
        """
        if not self.batch_days:
            return 0
        if not day_starts_ms:
            return 0

        day_store = self._get_day_store(symbol)

        # Filter to days not already cached
        days_to_fetch = [d for d in day_starts_ms if d not in day_store]

        # Sort descending (most recent first) and apply max_days limit
        days_to_fetch.sort(reverse=True)
        if max_days is not None and max_days > 0:
            days_to_fetch = days_to_fetch[:max_days]

        # Re-sort ascending for sequential fetching
        days_to_fetch.sort()

        prefetched = 0
        for day_start in days_to_fetch:
            minute_prices = self._fetch_day_klines(symbol, day_start)
            day_store[day_start] = minute_prices
            prefetched += 1

            if prefetched % 10 == 0:
                day_dt = datetime.fromtimestamp(day_start / 1000, tz=timezone.utc)
                logger.info(f"  Prefetched {prefetched}/{len(days_to_fetch)} days for {symbol} (current: {day_dt.date()})")

        return prefetched

    def get_price_history(
        self,
        symbol: str,
        end_time_ms: int,
        points: int,
        interval: str = "1m",
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Get a rolling price history ending at the given timestamp.

        Returns:
            Tuple of (history list, error reason or None)
            History points must end at or before entry timestamp.
        """
        if points <= 0:
            return [], "invalid points"

        start_time = end_time_ms - (points * MS_PER_MINUTE)

        if interval != "1m" or not self.batch_days:
            # Fallback to per-request fetching
            klines = self.get_klines(
                symbol=symbol,
                interval=interval if interval != "1m" else self.interval,
                start_time=start_time,
                end_time=end_time_ms,
                limit=min(points * 2, BINANCE_MAX_KLINES_PER_REQUEST),
            )
            history = []
            for k in klines:
                open_time = k["open_time"]
                if open_time > end_time_ms:
                    continue
                history.append({
                    "ts": datetime.fromtimestamp(open_time / 1000, tz=timezone.utc).isoformat(),
                    "price": float(k["close"]),
                })

            if len(history) < points:
                return [], f"insufficient data from API ({len(history)}/{points})"
            return history[-points:], None

        # Batch mode: slice from preloaded day cache
        klines = []
        end_minute_ms = (end_time_ms // MS_PER_MINUTE) * MS_PER_MINUTE
        needed = points
        current_ms = end_minute_ms
        days_checked = set()

        while needed > 0 and current_ms > 0:
            day_start_ms = (current_ms // MS_PER_DAY) * MS_PER_DAY
            days_checked.add(day_start_ms)

            day_prices = self.day_cache.get(symbol, {}).get(day_start_ms)
            if day_prices is None:
                # Day not prefetched - can't continue
                break

            while current_ms >= day_start_ms and needed > 0:
                price = day_prices.get(current_ms)
                if price is not None:
                    klines.append({"open_time": current_ms, "close": price})
                    needed -= 1
                current_ms -= MS_PER_MINUTE

            if current_ms < day_start_ms:
                current_ms = day_start_ms - MS_PER_MINUTE

        klines = list(reversed(klines))

        history = []
        for k in klines:
            open_time = k["open_time"]
            if open_time > end_time_ms:
                continue
            history.append({
                "ts": datetime.fromtimestamp(open_time / 1000, tz=timezone.utc).isoformat(),
                "price": float(k["close"]),
            })

        if len(history) < points:
            return [], f"insufficient history from cache ({len(history)}/{points}, days checked: {len(days_checked)})"

        return history[-points:], None


class TechnicalIndicators:
    """Calculate technical indicators from OHLCV data."""

    @staticmethod
    def calculate_rsi(closes: List[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0

        changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [c if c > 0 else 0 for c in changes]
        losses = [-c if c < 0 else 0 for c in changes]

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 2)

    @staticmethod
    def calculate_sma(values: List[float], period: int) -> float:
        if len(values) < period:
            return values[-1] if values else 0.0
        return sum(values[-period:]) / period

    @staticmethod
    def calculate_ema(values: List[float], period: int) -> float:
        if len(values) < period:
            return values[-1] if values else 0.0

        multiplier = 2 / (period + 1)
        ema = sum(values[:period]) / period

        for price in values[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    @staticmethod
    def calculate_macd(closes: List[float]) -> Tuple[float, str]:
        if len(closes) < 26:
            return 0.0, "none"

        ema12 = TechnicalIndicators.calculate_ema(closes, 12)
        ema26 = TechnicalIndicators.calculate_ema(closes, 26)
        macd_line = ema12 - ema26

        # Previous MACD for crossover detection
        if len(closes) > 27:
            prev_ema12 = TechnicalIndicators.calculate_ema(closes[:-1], 12)
            prev_ema26 = TechnicalIndicators.calculate_ema(closes[:-1], 26)
            prev_macd = prev_ema12 - prev_ema26

            if macd_line > 0 and prev_macd <= 0:
                signal = "bullish_cross"
            elif macd_line < 0 and prev_macd >= 0:
                signal = "bearish_cross"
            else:
                signal = "bullish" if macd_line > 0 else "bearish"
        else:
            signal = "bullish" if macd_line > 0 else "bearish"

        return round(macd_line, 6), signal

    @staticmethod
    def calculate_bollinger_bands(closes: List[float], period: int = 20) -> Tuple[float, str]:
        if len(closes) < period:
            return 0.0, "middle"

        sma = sum(closes[-period:]) / period
        variance = sum((p - sma) ** 2 for p in closes[-period:]) / period
        std = variance ** 0.5

        upper = sma + (2 * std)
        lower = sma - (2 * std)
        bandwidth = (upper - lower) / sma if sma > 0 else 0.0

        current_price = closes[-1]
        if current_price > upper:
            position = "above"
        elif current_price < lower:
            position = "below"
        else:
            position = "middle"

        return round(bandwidth, 4), position

    @staticmethod
    def calculate_stochastic(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Tuple[float, float, str]:
        if len(closes) < period:
            return 50.0, 50.0, "neutral"

        highest_high = max(highs[-period:])
        lowest_low = min(lows[-period:])

        if highest_high == lowest_low:
            k = 50.0
        else:
            k = 100 * (closes[-1] - lowest_low) / (highest_high - lowest_low)

        d = k  # Simplified

        if k > 80:
            signal = "overbought"
        elif k < 20:
            signal = "oversold"
        else:
            signal = "neutral"

        return round(k, 2), round(d, 2), signal

    @staticmethod
    def calculate_volatility(closes: List[float], period: int = 24) -> float:
        if len(closes) < period + 1:
            return 0.0

        returns = [(closes[i] - closes[i-1]) / closes[i-1]
                   for i in range(1, len(closes)) if closes[i-1] != 0]

        if len(returns) < period:
            return 0.0

        recent_returns = returns[-period:]
        mean_return = sum(recent_returns) / len(recent_returns)
        variance = sum((r - mean_return) ** 2 for r in recent_returns) / len(recent_returns)

        return round(variance ** 0.5, 6)

    @staticmethod
    def calculate_trend(closes: List[float], period: int) -> float:
        if len(closes) < period or closes[-period] == 0:
            return 0.0
        return round((closes[-1] - closes[-period]) / closes[-period] * 100, 4)

    @staticmethod
    def calculate_volume_ratio(volumes: List[float], period: int = 20) -> float:
        if len(volumes) < period + 1:
            return 1.0
        avg_volume = sum(volumes[-period-1:-1]) / period
        if avg_volume == 0:
            return 1.0
        return round(volumes[-1] / avg_volume, 2)

    @staticmethod
    def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 0.0

        true_ranges = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            true_ranges.append(tr)

        if len(true_ranges) < period:
            return 0.0

        return round(sum(true_ranges[-period:]) / period, 4)


class PolymarketTradesFetcher:
    """Fetches historical closed positions from Polymarket."""

    def __init__(self, wallet_address: str):
        self.wallet_address = wallet_address.lower()
        self.last_request = 0

    def _rate_limit(self):
        elapsed = time.time() - self.last_request
        if elapsed < POLYMARKET_RATE_LIMIT:
            time.sleep(POLYMARKET_RATE_LIMIT - elapsed)
        self.last_request = time.time()

    def fetch_closed_positions(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Fetch closed positions with P&L data."""
        self._rate_limit()

        try:
            response = requests.get(
                f"{POLYMARKET_DATA_API}/closed-positions",
                params={
                    "user": self.wallet_address,
                    "limit": limit,
                    "offset": offset,
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch closed positions: {e}")
            return []

    def fetch_all_closed_positions(self, max_positions: int = 20000) -> List[Dict]:
        """Fetch all historical closed positions."""
        all_positions = []
        seen_ids = set()
        offset = 0
        limit = 50  # API seems to cap at 50

        while len(all_positions) < max_positions:
            logger.info(f"Fetching closed positions... offset={offset}, total={len(all_positions)}")

            positions = self.fetch_closed_positions(limit=limit, offset=offset)

            if not positions:
                break

            # Deduplicate by conditionId
            new_count = 0
            for pos in positions:
                cid = pos.get("conditionId", "")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    all_positions.append(pos)
                    new_count += 1

            logger.info(f"  Got {len(positions)} positions, {new_count} new (total unique: {len(all_positions)})")

            # If we got no new positions, we've hit the end
            if new_count == 0:
                offset += limit  # Try next page anyway
                if offset > len(all_positions) + 1000:  # Safety limit
                    break
            else:
                offset += limit

            if len(positions) < limit:
                break

        logger.info(f"Fetched {len(all_positions)} unique closed positions")
        return all_positions


class TradeEnricher:
    """Enriches trades with market context."""

    def __init__(
        self,
        max_cache_windows: int = 5000,
        batch_days: bool = True,
        binance_interval: str = "1m",
        binance_timeout: int = 15,
        fetch_market_info: bool = False,
    ):
        self.binance = BinanceHistoricalData(
            max_cache_windows=max_cache_windows,
            batch_days=batch_days,
            interval=binance_interval,
            timeout=binance_timeout,
        )
        self.indicators = TechnicalIndicators()
        self._market_cache: Dict[str, Dict[str, Any]] = {}
        self._market_last_request = 0.0
        self._fetch_market_info_enabled = fetch_market_info

    def _detect_asset(self, title: str) -> Optional[str]:
        """Detect asset from market title."""
        title_lower = title.lower()
        for asset, patterns in ASSET_PATTERNS.items():
            for pattern in patterns:
                if pattern in title_lower:
                    return asset
        return None

    def _parse_side(self, outcome: str) -> Optional[str]:
        """Parse side from outcome field."""
        outcome_lower = outcome.lower()
        if "up" in outcome_lower or "yes" in outcome_lower or "higher" in outcome_lower:
            return "UP"
        elif "down" in outcome_lower or "no" in outcome_lower or "lower" in outcome_lower:
            return "DOWN"
        return None

    def _detect_market_direction(self, title: str) -> Optional[str]:
        title_lower = title.lower()
        if "up" in title_lower or "higher" in title_lower or "above" in title_lower:
            return "UP"
        if "down" in title_lower or "lower" in title_lower or "below" in title_lower:
            return "DOWN"
        return None

    def _parse_outcome_token(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        v = value.lower()
        if "yes" in v or v == "y":
            return "YES"
        if "no" in v or v == "n":
            return "NO"
        if v in ("1", "true"):
            return "YES"
        if v in ("0", "false"):
            return "NO"
        if "up" in v or "higher" in v or "above" in v:
            return "UP"
        if "down" in v or "lower" in v or "below" in v:
            return "DOWN"
        return None

    def _opposite_token(self, token: str) -> str:
        return {
            "YES": "NO",
            "NO": "YES",
            "UP": "DOWN",
            "DOWN": "UP",
        }[token]

    def _rate_limit_market(self) -> None:
        elapsed = time.time() - self._market_last_request
        if elapsed < POLYMARKET_RATE_LIMIT:
            time.sleep(POLYMARKET_RATE_LIMIT - elapsed)
        self._market_last_request = time.time()

    def _fetch_market_info(self, condition_id: str) -> Optional[Dict[str, Any]]:
        if not condition_id:
            return None
        if condition_id in self._market_cache:
            return self._market_cache[condition_id]

        self._rate_limit_market()
        try:
            response = requests.get(
                f"{POLYMARKET_DATA_API}/markets/{condition_id}",
                timeout=30,
            )
            if response.status_code == 404:
                response = requests.get(
                    f"{POLYMARKET_DATA_API}/markets",
                    params={"conditionId": condition_id},
                    timeout=30,
                )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                data = data[0] if data else None
            if data:
                self._market_cache[condition_id] = data
            return data
        except Exception as e:
            logger.debug(f"Failed to fetch market info for {condition_id}: {e}")
            return None

    def _extract_ts_from(self, data: Dict[str, Any], keys: List[str]) -> Optional[datetime]:
        for key in keys:
            if key in data and data[key]:
                parsed = _parse_timestamp(data[key])
                if parsed:
                    return parsed
        return None

    def _resolve_market_times(
        self,
        position: Dict[str, Any],
        market_info: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        open_keys = [
            "marketOpenTimestamp",
            "openTimestamp",
            "openTime",
            "openingTime",
            "startTime",
            "startDate",
            "startTimestamp",
        ]
        end_keys = [
            "marketEndTimestamp",
            "closeTimestamp",
            "closeTime",
            "closingTime",
            "endTime",
            "endDate",
            "resolutionTime",
            "resolutionDate",
            "resolvedAt",
        ]

        open_ts = self._extract_ts_from(position, open_keys)
        end_ts = self._extract_ts_from(position, end_keys)

        if market_info:
            open_ts = open_ts or self._extract_ts_from(market_info, open_keys)
            end_ts = end_ts or self._extract_ts_from(market_info, end_keys)

        if open_ts and not end_ts:
            end_ts = open_ts + timedelta(minutes=15)
        if end_ts and not open_ts:
            open_ts = end_ts - timedelta(minutes=15)

        return open_ts, end_ts

    def _infer_yes_mid(
        self,
        entry_price: float,
        outcome_token: Optional[str],
    ) -> Optional[float]:
        if outcome_token in ("YES", "UP"):
            return entry_price
        if outcome_token in ("NO", "DOWN"):
            return 1.0 - entry_price
        return None

    def _synthetic_orderbook(
        self,
        yes_bid: float,
        yes_ask: float,
        depth_levels: int,
        tick: float = 0.001,
        base_size: float = 100.0,
        size_step: float = 20.0,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        bids: List[List[float]] = []
        asks: List[List[float]] = []
        for i in range(depth_levels):
            bid_price = max(yes_bid - tick * i, 0.0001)
            ask_price = max(yes_ask + tick * i, 0.0001)
            size = max(base_size - size_step * i, 1.0)
            bids.append([round(bid_price, 6), float(size)])
            asks.append([round(ask_price, 6), float(size)])
        return bids, asks

    def _resolve_resolution(
        self,
        position: Dict[str, Any],
        market_info: Optional[Dict[str, Any]],
        market_direction: Optional[str],
        outcome_token: Optional[str],
    ) -> Optional[str]:
        candidates = []
        for src in (position, market_info or {}):
            candidates.extend(
                [
                    src.get("resolvedOutcome"),
                    src.get("resolvedOutcomeId"),
                    src.get("resolution"),
                    src.get("result"),
                    src.get("finalOutcome"),
                ]
            )

        resolved_token = None
        for cand in candidates:
            token = self._parse_outcome_token(str(cand)) if cand is not None else None
            if token:
                resolved_token = token
                break

        if resolved_token is None:
            cur_price = position.get("curPrice")
            cur_price = _to_float(cur_price)
            if cur_price is None or cur_price not in (0.0, 1.0):
                return None
            if outcome_token is None:
                return None
            resolved_token = outcome_token if cur_price == 1.0 else self._opposite_token(outcome_token)

        if resolved_token in ("UP", "DOWN"):
            return resolved_token

        if resolved_token in ("YES", "NO") and market_direction:
            if market_direction == "UP":
                return "UP" if resolved_token == "YES" else "DOWN"
            return "DOWN" if resolved_token == "YES" else "UP"

        return None

    def _get_session_info(self, timestamp: datetime) -> Tuple[bool, bool, bool, bool]:
        """Determine trading session based on UTC hour."""
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5
        is_asia = 0 <= hour < 8
        is_europe = 7 <= hour < 16
        is_us = 13 <= hour < 22
        return is_weekend, is_asia, is_europe, is_us

    def enrich_position(self, position: Dict) -> Optional[EnrichedTrade]:
        """Enrich a closed position with market context."""
        try:
            # Parse timestamp (seconds to datetime)
            timestamp_sec = position.get("timestamp")
            if not timestamp_sec:
                return None

            timestamp = datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)
            timestamp_ms = timestamp_sec * 1000

            # Detect asset
            title = position.get("title", "")
            asset = self._detect_asset(title)
            if not asset:
                logger.debug(f"Could not detect asset from: {title}")
                return None

            # Parse side (what they bet on)
            outcome = position.get("outcome", "")
            side = self._parse_side(outcome)
            if not side:
                logger.debug(f"Could not parse side from: {outcome}")
                return None

            # Determine win/loss from curPrice (1 = won, 0 = lost)
            cur_price = position.get("curPrice", 0)
            realized_pnl = position.get("realizedPnl", 0)

            if cur_price == 1:
                trade_outcome = "WIN"
            elif cur_price == 0:
                trade_outcome = "LOSS"
            elif realized_pnl > 0:
                trade_outcome = "WIN"
            elif realized_pnl < 0:
                trade_outcome = "LOSS"
            else:
                # Can't determine outcome
                return None

            # Get trade details
            entry_price = float(position.get("avgPrice", 0.5))
            total_shares = float(position.get("totalBought", 0))

            # Cost = shares * entry price
            cost_usd = total_shares * entry_price

            # API realizedPnl shows theoretical P&L if held to settlement
            # Real P&L is lower due to early exits (~$946K vs ~$16M theoretical)
            # Scale factor: 946327 / 16267903 ≈ 0.058
            # This gives a more realistic P&L estimate
            PNL_SCALE_FACTOR = 0.058
            api_pnl = float(realized_pnl)
            calculated_pnl = api_pnl * PNL_SCALE_FACTOR

            # Create enriched trade
            enriched = EnrichedTrade(
                trade_id=position.get("conditionId", str(timestamp_ms)),
                timestamp=timestamp.isoformat(),
                timestamp_ms=timestamp_ms,
                asset=asset,
                side=side,
                outcome=trade_outcome,
                entry_price=entry_price,
                size_usd=cost_usd,
                realized_pnl=calculated_pnl,  # Use our calculated P&L
            )

            # Fetch historical OHLCV data from Binance
            symbol = ASSET_TO_SYMBOL.get(asset, "BTCUSDT")
            ohlcv = self.binance.get_ohlcv_context(symbol, timestamp_ms, lookback_hours=48)

            if not ohlcv or len(ohlcv) < 5:
                logger.debug(f"Insufficient OHLCV data for {asset} at {timestamp}")
                # Still return with basic info, just no indicators
                return enriched

            # Extract price arrays
            closes = [k["close"] for k in ohlcv]
            highs = [k["high"] for k in ohlcv]
            lows = [k["low"] for k in ohlcv]
            volumes = [k["volume"] for k in ohlcv]

            # Asset price
            enriched.asset_price = closes[-1] if closes else 0.0

            # Other asset prices (with caching)
            enriched.btc_price = self.binance.get_price_at_time("BTCUSDT", timestamp_ms) or 0.0
            if asset != "ETH":
                enriched.eth_price = self.binance.get_price_at_time("ETHUSDT", timestamp_ms) or 0.0
            if asset != "SOL":
                enriched.sol_price = self.binance.get_price_at_time("SOLUSDT", timestamp_ms) or 0.0

            # Calculate indicators
            enriched.rsi_14 = self.indicators.calculate_rsi(closes, 14)
            enriched.rsi_7 = self.indicators.calculate_rsi(closes, 7)

            enriched.macd_histogram, enriched.macd_signal = self.indicators.calculate_macd(closes)
            enriched.bb_bandwidth, enriched.bb_position = self.indicators.calculate_bollinger_bands(closes)
            enriched.stoch_k, enriched.stoch_d, enriched.stoch_signal = self.indicators.calculate_stochastic(highs, lows, closes)

            # Moving averages
            enriched.sma_20 = self.indicators.calculate_sma(closes, 20)
            enriched.sma_50 = self.indicators.calculate_sma(closes, min(50, len(closes)))
            enriched.ema_12 = self.indicators.calculate_ema(closes, 12)
            enriched.ema_26 = self.indicators.calculate_ema(closes, 26)

            current_price = closes[-1]
            if enriched.sma_20 > 0:
                enriched.price_vs_sma20 = round((current_price - enriched.sma_20) / enriched.sma_20 * 100, 2)
            if enriched.sma_50 > 0:
                enriched.price_vs_sma50 = round((current_price - enriched.sma_50) / enriched.sma_50 * 100, 2)

            # Trends
            enriched.trend_1h = self.indicators.calculate_trend(closes, min(1, len(closes)-1))
            enriched.trend_4h = self.indicators.calculate_trend(closes, min(4, len(closes)-1))
            enriched.trend_1d = self.indicators.calculate_trend(closes, min(24, len(closes)-1))

            if enriched.trend_1d > 1:
                enriched.trend_direction = "up"
            elif enriched.trend_1d < -1:
                enriched.trend_direction = "down"
            else:
                enriched.trend_direction = "neutral"

            # Volatility
            enriched.volatility_1h = self.indicators.calculate_volatility(closes, min(1, len(closes)-1))
            enriched.volatility_24h = self.indicators.calculate_volatility(closes, min(24, len(closes)-1))
            enriched.atr_14 = self.indicators.calculate_atr(highs, lows, closes, min(14, len(closes)-1))

            # Volume
            enriched.volume_ratio = self.indicators.calculate_volume_ratio(volumes)
            enriched.volume_24h = sum(volumes[-24:]) if len(volumes) >= 24 else sum(volumes)

            # Time context
            enriched.hour_utc = timestamp.hour
            enriched.day_of_week = timestamp.weekday()
            is_weekend, is_asia, is_europe, is_us = self._get_session_info(timestamp)
            enriched.is_weekend = is_weekend
            enriched.is_asia_session = is_asia
            enriched.is_europe_session = is_europe
            enriched.is_us_session = is_us

            # Market structure
            if len(highs) >= 3:
                enriched.higher_high = highs[-1] > highs[-2] > highs[-3]
                enriched.lower_low = lows[-1] < lows[-2] < lows[-3]
                price_range = (max(highs[-3:]) - min(lows[-3:])) / current_price if current_price > 0 else 0
                enriched.consolidating = price_range < 0.02

            return enriched

        except Exception as e:
            logger.debug(f"Error enriching position: {e}")
            return None

    def enrich_position_v2(
        self,
        position: Dict[str, Any],
        depth_levels: int,
        history_points: int,
        window_tolerance_seconds: int,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            ts_raw = position.get("timestamp") or position.get("ts")
            entry_dt = _parse_timestamp(ts_raw)
            if not entry_dt:
                return None, "missing timestamp"

            entry_ts = normalize_ts(_isoformat(entry_dt))
            if entry_ts is None:
                return None, "missing timestamp"

            entry_ms = int(entry_dt.timestamp() * 1000)

            title = position.get("title") or position.get("question") or ""
            asset = self._detect_asset(title)
            if not asset:
                return None, "asset not detected"

            symbol = ASSET_TO_SYMBOL.get(asset)
            if not symbol:
                return None, "missing symbol"

            market_direction = self._detect_market_direction(title)
            if not market_direction:
                return None, "market direction not detected"

            condition_id = position.get("conditionId") or position.get("marketId") or position.get("id")
            market_info = None
            if self._fetch_market_info_enabled and condition_id:
                market_info = self._fetch_market_info(str(condition_id))

            market_open_ts = math.floor(entry_ts / 900.0) * 900.0
            market_end_ts = market_open_ts + 900.0
            market_open_dt = datetime.fromtimestamp(market_open_ts, tz=timezone.utc)
            market_end_dt = datetime.fromtimestamp(market_end_ts, tz=timezone.utc)
            tol = float(window_tolerance_seconds)
            if entry_ts < (market_open_ts - tol) or entry_ts > (market_end_ts + tol):
                return None, "entry timestamp outside market window"

            entry_price = _to_float(position.get("avgPrice") or position.get("entryPrice") or position.get("price"))
            if entry_price is None or entry_price <= 0 or entry_price >= 1:
                return None, "invalid entry price"

            outcome_token = self._parse_outcome_token(
                position.get("outcome")
                or position.get("positionOutcome")
                or position.get("outcomeTitle")
            )

            yes_bid = _to_float(position.get("yesBid") or position.get("yes_bid") or position.get("bestYesBid"))
            yes_ask = _to_float(position.get("yesAsk") or position.get("yes_ask") or position.get("bestYesAsk"))

            if yes_bid is None or yes_ask is None:
                yes_mid = self._infer_yes_mid(entry_price, outcome_token)
                if yes_mid is None:
                    return None, "cannot infer yes price"
                yes_bid = yes_mid if yes_bid is None else yes_bid
                yes_ask = yes_mid if yes_ask is None else yes_ask

            if yes_bid <= 0 or yes_ask <= 0:
                return None, "invalid yes bid/ask"

            no_bid = _to_float(position.get("noBid") or position.get("no_bid") or position.get("bestNoBid"))
            no_ask = _to_float(position.get("noAsk") or position.get("no_ask") or position.get("bestNoAsk"))
            if no_bid is None:
                no_bid = 1.0 - yes_ask
            if no_ask is None:
                no_ask = 1.0 - yes_bid

            if no_bid <= 0 or no_ask <= 0:
                return None, "invalid no bid/ask"

            orderbook_bids, orderbook_asks = self._synthetic_orderbook(yes_bid, yes_ask, depth_levels)

            history, history_err = self.binance.get_price_history(symbol, entry_ms, history_points)
            if history_err:
                return None, f"insufficient reference price history: {history_err}"

            reference_price = self.binance.get_price_at_time(symbol, entry_ms)
            if reference_price is None:
                reference_price = history[-1]["price"]
            reference_price_open = self.binance.get_price_at_time(
                symbol, int(market_open_dt.timestamp() * 1000)
            )
            if reference_price is None or reference_price_open is None:
                return None, "missing reference prices"

            resolution = self._resolve_resolution(position, market_info, market_direction, outcome_token)
            if resolution is None:
                return None, "missing resolution"

            row = {
                "timestamp": _isoformat(entry_dt),
                "market_open_ts": _isoformat(market_open_dt),
                "market_end_ts": _isoformat(market_end_dt),
                "polymarket_yes_bid": float(yes_bid),
                "polymarket_yes_ask": float(yes_ask),
                "polymarket_no_bid": float(no_bid),
                "polymarket_no_ask": float(no_ask),
                "orderbook_bids": orderbook_bids,
                "orderbook_asks": orderbook_asks,
                "reference_price": float(reference_price),
                "reference_price_open": float(reference_price_open),
                "reference_price_history": history,
                "resolution": resolution,
                "trade_id": str(condition_id) if condition_id else "",
                "asset": asset,
            }

            return row, None
        except Exception as e:
            return None, f"error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Enrich historical trades with market data")
    parser.add_argument("--wallet", required=True, help="Wallet address to fetch trades for")
    parser.add_argument("--output", default="data/enriched_trades_v2.jsonl", help="Output file path")
    parser.add_argument("--depth-levels", type=int, default=5, help="Synthetic orderbook depth levels")
    parser.add_argument("--history-points", type=int, default=20, help="Reference price history points")
    parser.add_argument(
        "--window-tolerance-seconds",
        type=int,
        default=60,
        help="Tolerance window in seconds for entry timestamp vs market window",
    )
    parser.add_argument(
        "--binance-batch-days",
        type=lambda v: str(v).lower() in ("1", "true", "yes"),
        default=True,
        help="Batch Binance 1m klines by day (default true)",
    )
    parser.add_argument(
        "--binance-interval",
        type=str,
        default="1m",
        help="Binance kline interval (default 1m)",
    )
    parser.add_argument(
        "--binance-timeout",
        type=int,
        default=15,
        help="Binance API request timeout in seconds (default 15)",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="Only preload this many most-recent days for debugging (default: all)",
    )
    parser.add_argument(
        "--max-cache-windows",
        type=int,
        default=5000,
        help="Max cached kline windows (LRU)",
    )
    parser.add_argument(
        "--fetch-market-info",
        type=lambda v: str(v).lower() in ("1", "true", "yes"),
        default=False,
        help="Fetch additional market info from Polymarket API (slow, default false)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Stop after N trades")
    parser.add_argument("--max-positions", type=int, default=20000, help="Maximum positions to fetch")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing data if resuming
    existing_ids = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    trade_id = row.get("trade_id")
                    if trade_id:
                        existing_ids.add(trade_id)
                except json.JSONDecodeError:
                    continue
        logger.info(f"Resuming with {len(existing_ids)} existing enriched trades")

    # Fetch closed positions
    logger.info(f"Fetching closed positions for wallet: {args.wallet}")
    fetcher = PolymarketTradesFetcher(args.wallet)
    positions = fetcher.fetch_all_closed_positions(max_positions=args.max_positions)
    logger.info(f"Found {len(positions)} closed positions")

    # Filter to crypto markets only
    crypto_positions = []
    for pos in positions:
        title = pos.get("title", "").lower()
        if any(pattern in title for patterns in ASSET_PATTERNS.values() for pattern in patterns):
            crypto_positions.append(pos)

    logger.info(f"Filtered to {len(crypto_positions)} crypto positions")
    logger.info("Orderbook snapshots unavailable; using synthetic orderbooks")

    # Enrich positions
    enricher = TradeEnricher(
        max_cache_windows=args.max_cache_windows,
        batch_days=args.binance_batch_days,
        binance_interval=args.binance_interval,
        binance_timeout=args.binance_timeout,
        fetch_market_info=args.fetch_market_info,
    )

    stats = EnrichmentStats()
    total = len(crypto_positions)
    mode = "a" if args.resume and output_path.exists() else "w"

    # Prefetch Binance data by day if batching enabled
    if args.binance_batch_days:
        logger.info("Extracting trade timestamps and computing required days...")
        symbol_days: Dict[str, Set[int]] = {}

        for pos in crypto_positions:
            ts_raw = pos.get("timestamp") or pos.get("ts")
            entry_ts = normalize_ts(ts_raw)
            if entry_ts is None:
                continue
            title = pos.get("title") or pos.get("question") or ""
            asset = enricher._detect_asset(title)
            if not asset:
                continue
            symbol = ASSET_TO_SYMBOL.get(asset)
            if not symbol:
                continue

            # Compute day start in milliseconds
            day_start_sec = int(entry_ts // 86400) * 86400
            day_start_ms = day_start_sec * 1000
            symbol_days.setdefault(symbol, set()).add(day_start_ms)

            # If trade is near start of day, also need previous day for history
            minutes_into_day = (entry_ts - day_start_sec) / 60.0
            if minutes_into_day < args.history_points:
                prev_day_ms = (day_start_sec - 86400) * 1000
                symbol_days.setdefault(symbol, set()).add(prev_day_ms)

        # Log summary and prefetch
        total_days = sum(len(days) for days in symbol_days.values())
        logger.info(f"Need to prefetch {total_days} symbol-days across {len(symbol_days)} symbols")

        for symbol, days in sorted(symbol_days.items()):
            sorted_days = sorted(days)
            date_range = ""
            if sorted_days:
                first_dt = datetime.fromtimestamp(sorted_days[0] / 1000, tz=timezone.utc)
                last_dt = datetime.fromtimestamp(sorted_days[-1] / 1000, tz=timezone.utc)
                date_range = f" ({first_dt.date()} to {last_dt.date()})"

            logger.info(f"Prefetching {len(days)} days for {symbol}{date_range}...")
            prefetched = enricher.binance.prefetch_days(symbol, sorted_days, max_days=args.max_days)
            logger.info(f"  Completed: {prefetched} days, {enricher.binance.prefetch_request_count} API requests")

        logger.info(f"Prefetch complete. Total Binance requests: {enricher.binance.request_count}")
        logger.info("-" * 60)

    def log_progress(stats: EnrichmentStats, idx: int, total: int, enricher: TradeEnricher) -> None:
        """Log progress stats."""
        elapsed = stats.elapsed()
        avg_per_100 = stats.avg_per_100()
        tpm = stats.trades_per_minute()

        hits, misses, cache_rate = enricher.binance.window_cache.stats()

        logger.info(
            f"[{idx}/{total}] processed={stats.processed} | new={stats.new_count} | "
            f"skipped={stats.skipped_existing + stats.skipped_reason} | errors={stats.errors}"
        )
        logger.info(
            f"  Binance requests: {enricher.binance.request_count} | "
            f"Cache hit rate: {cache_rate:.1f}% ({hits}/{hits + misses})"
        )
        logger.info(
            f"  Elapsed: {elapsed:.1f}s | Avg: {avg_per_100:.2f}s/100 trades | Rate: {tpm:.1f} trades/min"
        )

    logger.info(f"Starting enrichment of {total} crypto positions...")

    with open(output_path, mode) as f:
        last_log_time = time.time()

        for i, position in enumerate(crypto_positions):
            if args.limit is not None and stats.processed >= args.limit:
                break

            trade_id = position.get("conditionId", "")

            # Skip already processed
            if trade_id in existing_ids:
                stats.skipped_existing += 1
                continue

            stats.processed += 1

            # Log every 500 trades OR every 30 seconds (whichever comes first)
            now = time.time()
            if stats.processed % 500 == 0 or (now - last_log_time) >= 30:
                log_progress(stats, i + 1, total, enricher)
                last_log_time = now

            row, reason = enricher.enrich_position_v2(
                position,
                depth_levels=args.depth_levels,
                history_points=args.history_points,
                window_tolerance_seconds=args.window_tolerance_seconds,
            )

            if row:
                f.write(json.dumps(row) + "\n")
                f.flush()
                if trade_id:
                    existing_ids.add(trade_id)
                stats.new_count += 1
            else:
                if reason:
                    stats.record_skip(reason)
                    logger.debug(f"Skipping {trade_id or 'unknown'}: {reason}")
                else:
                    stats.errors += 1

    # Final summary
    elapsed = stats.elapsed()
    logger.info("=" * 60)
    logger.info("ENRICHMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total processed:        {stats.processed}")
    logger.info(f"New trades enriched:    {stats.new_count}")
    logger.info(f"Skipped (existing):     {stats.skipped_existing}")
    logger.info(f"Skipped (invalid):      {stats.skipped_reason}")
    logger.info(f"Errors:                 {stats.errors}")
    logger.info("-" * 40)
    logger.info(f"Binance API requests:   {enricher.binance.request_count}")
    logger.info(f"  (prefetch requests):  {enricher.binance.prefetch_request_count}")
    hits, misses, cache_rate = enricher.binance.window_cache.stats()
    logger.info(f"Cache hit rate:         {cache_rate:.1f}% ({hits}/{hits + misses})")
    logger.info("-" * 40)
    logger.info(f"Elapsed time:           {elapsed:.1f}s")
    if stats.processed > 0:
        logger.info(f"Throughput:             {stats.trades_per_minute():.1f} trades/min")
    logger.info("-" * 40)

    # Log skip reasons breakdown
    if stats.skip_reasons:
        logger.info("Skip reasons breakdown:")
        for reason, count in sorted(stats.skip_reasons.items(), key=lambda x: -x[1]):
            logger.info(f"  {reason}: {count}")

    logger.info(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
