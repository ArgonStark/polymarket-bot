"""
Binance Chart Analysis Module.

Fetches candlestick data from Binance and performs technical analysis:
- Trend detection (EMA/SMA crossovers)
- Trend change detection
- Range vs trending market identification
- Bullish/bearish pattern recognition

API Reference: https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict
from enum import Enum
import requests

logger = logging.getLogger(__name__)


# Binance API endpoints
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

# Symbol mapping
BINANCE_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
}

# Available intervals
INTERVALS = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
    "3d": "3d",
    "1w": "1w",
}


class MarketType(Enum):
    """Market condition type."""
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    WEAK_UPTREND = "weak_uptrend"
    RANGING = "ranging"
    WEAK_DOWNTREND = "weak_downtrend"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


class TrendChange(Enum):
    """Trend change signals."""
    BULLISH_REVERSAL = "bullish_reversal"
    BEARISH_REVERSAL = "bearish_reversal"
    BULLISH_CONTINUATION = "bullish_continuation"
    BEARISH_CONTINUATION = "bearish_continuation"
    NO_CHANGE = "no_change"


@dataclass
class Candle:
    """Single candlestick data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int = 0

    @property
    def body_size(self) -> float:
        """Absolute body size."""
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        """Total candle range."""
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        """True if close > open."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """True if close < open."""
        return self.close < self.open

    @property
    def body_percentage(self) -> float:
        """Body as percentage of range."""
        if self.range == 0:
            return 0.0
        return self.body_size / self.range

    @property
    def upper_wick(self) -> float:
        """Upper wick size."""
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        """Lower wick size."""
        return min(self.open, self.close) - self.low


@dataclass
class ChartAnalysis:
    """Complete chart analysis result."""
    asset: str
    timestamp: datetime

    # Current price info
    current_price: float
    price_change_1h: float = 0.0
    price_change_4h: float = 0.0
    price_change_24h: float = 0.0

    # Trend indicators
    trend_1m: float = 0.0  # -1 to +1
    trend_5m: float = 0.0
    trend_15m: float = 0.0
    trend_1h: float = 0.0
    trend_4h: float = 0.0

    # Moving averages
    ema_9: float = 0.0
    ema_21: float = 0.0
    ema_50: float = 0.0
    sma_20: float = 0.0
    sma_50: float = 0.0

    # Market type
    market_type: MarketType = MarketType.RANGING
    trend_strength: float = 0.0  # 0 to 1

    # Trend change
    trend_change: TrendChange = TrendChange.NO_CHANGE
    trend_change_confidence: float = 0.0

    # Volatility
    volatility: float = 0.0
    atr: float = 0.0  # Average True Range

    # Support/Resistance
    support_level: float = 0.0
    resistance_level: float = 0.0

    # RSI
    rsi_14: float = 50.0

    # Momentum
    momentum: float = 0.0  # -1 to +1
    momentum_increasing: bool = False

    # Pattern detection
    is_bullish_pattern: bool = False
    is_bearish_pattern: bool = False
    pattern_name: str = ""

    # Market uncertainty (trend change detection)
    is_uncertain: bool = False  # True if market is in transition
    uncertainty_score: float = 0.0  # 0 to 1
    uncertainty_reason: str = ""  # Why market is uncertain

    # Recommendation
    bias: str = "neutral"  # "bullish", "bearish", "neutral"
    confidence: float = 0.5  # 0 to 1

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "asset": self.asset,
            "timestamp": self.timestamp.isoformat(),
            "current_price": self.current_price,
            "trend_1h": self.trend_1h,
            "trend_4h": self.trend_4h,
            "market_type": self.market_type.value,
            "trend_strength": self.trend_strength,
            "trend_change": self.trend_change.value,
            "rsi_14": self.rsi_14,
            "momentum": self.momentum,
            "bias": self.bias,
            "confidence": self.confidence,
            "is_uncertain": self.is_uncertain,
            "uncertainty_score": self.uncertainty_score,
            "uncertainty_reason": self.uncertainty_reason,
        }


class BinanceChartAnalyzer:
    """
    Analyzes Binance price charts for trading signals.

    Fetches candlestick data and performs technical analysis to determine:
    - Current trend direction and strength
    - Trend changes and reversals
    - Market type (trending vs ranging)
    - Bullish/bearish patterns
    """

    def __init__(self, cache_ttl: int = 30):
        """
        Initialize analyzer.

        Args:
            cache_ttl: Cache time-to-live in seconds
        """
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, tuple] = {}  # key -> (data, timestamp)
        self._kline_cache: Dict[str, tuple] = {}  # key -> (candles, timestamp)

    def fetch_klines(
        self,
        asset: str,
        interval: str = "15m",
        limit: int = 100,
    ) -> List[Candle]:
        """
        Fetch candlestick data from Binance.

        Args:
            asset: Asset symbol (BTC, ETH, etc.)
            interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to fetch (max 1000)

        Returns:
            List of Candle objects
        """
        symbol = BINANCE_SYMBOLS.get(asset.upper())
        if not symbol:
            logger.warning(f"Unknown asset: {asset}")
            return []

        cache_key = f"{symbol}_{interval}_{limit}"
        now = time.time()

        # Check cache
        if cache_key in self._kline_cache:
            cached_data, cached_time = self._kline_cache[cache_key]
            if now - cached_time < self.cache_ttl:
                return cached_data

        try:
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
            }

            response = requests.get(
                BINANCE_KLINES_URL,
                params=params,
                timeout=10,
            )

            if response.status_code != 200:
                logger.warning(f"Binance API error: {response.status_code}")
                return []

            data = response.json()
            candles = []

            for kline in data:
                candle = Candle(
                    timestamp=datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc),
                    open=float(kline[1]),
                    high=float(kline[2]),
                    low=float(kline[3]),
                    close=float(kline[4]),
                    volume=float(kline[5]),
                    trades=int(kline[8]),
                )
                candles.append(candle)

            # Cache result
            self._kline_cache[cache_key] = (candles, now)

            return candles

        except Exception as e:
            logger.error(f"Failed to fetch klines for {asset}: {e}")
            return []

    def calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return prices[-1] if prices else 0.0

        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period  # SMA for first period

        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def calculate_sma(self, prices: List[float], period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return sum(prices) / len(prices) if prices else 0.0
        return sum(prices[-period:]) / period

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0

        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        if len(gains) < period:
            return 50.0

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_atr(self, candles: List[Candle], period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(candles) < period + 1:
            return 0.0

        true_ranges = []
        for i in range(1, len(candles)):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i - 1].close

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            true_ranges.append(tr)

        if len(true_ranges) < period:
            return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0

        return sum(true_ranges[-period:]) / period

    def detect_trend(self, candles: List[Candle]) -> float:
        """
        Detect trend direction and strength.

        Returns:
            Float from -1 (strong downtrend) to +1 (strong uptrend)
        """
        if len(candles) < 20:
            return 0.0

        closes = [c.close for c in candles]

        # Calculate EMAs
        ema_9 = self.calculate_ema(closes, 9)
        ema_21 = self.calculate_ema(closes, 21)

        current_price = closes[-1]

        # Price position relative to EMAs
        above_ema_9 = current_price > ema_9
        above_ema_21 = current_price > ema_21
        ema_9_above_21 = ema_9 > ema_21

        # Calculate trend strength based on multiple factors
        trend_score = 0.0

        # EMA alignment
        if above_ema_9 and above_ema_21 and ema_9_above_21:
            trend_score += 0.4  # Bullish alignment
        elif not above_ema_9 and not above_ema_21 and not ema_9_above_21:
            trend_score -= 0.4  # Bearish alignment

        # Price momentum (last 5 candles)
        if len(closes) >= 5:
            momentum = (closes[-1] - closes[-5]) / closes[-5]
            trend_score += max(-0.3, min(0.3, momentum * 10))

        # Higher highs / lower lows
        if len(candles) >= 10:
            recent_highs = [c.high for c in candles[-10:]]
            recent_lows = [c.low for c in candles[-10:]]

            # Check for higher highs
            if recent_highs[-1] > max(recent_highs[:-3]):
                trend_score += 0.15
            elif recent_highs[-1] < min(recent_highs[:-3]):
                trend_score -= 0.15

            # Check for higher lows / lower lows
            if recent_lows[-1] > min(recent_lows[:-3]):
                trend_score += 0.15
            elif recent_lows[-1] < min(recent_lows[:-3]):
                trend_score -= 0.15

        return max(-1.0, min(1.0, trend_score))

    def detect_market_type(
        self,
        candles: List[Candle],
        trend: float,
    ) -> tuple[MarketType, float]:
        """
        Detect market type (trending vs ranging).

        Returns:
            Tuple of (MarketType, strength from 0 to 1)
        """
        if len(candles) < 20:
            return MarketType.RANGING, 0.0

        closes = [c.close for c in candles]

        # Calculate ATR as percentage of price
        atr = self.calculate_atr(candles)
        atr_pct = atr / closes[-1] if closes[-1] > 0 else 0

        # Calculate price range
        high = max(c.high for c in candles[-20:])
        low = min(c.low for c in candles[-20:])
        range_pct = (high - low) / low if low > 0 else 0

        # Ranging market: low ATR and narrow range
        is_ranging = atr_pct < 0.005 and range_pct < 0.02

        if is_ranging:
            return MarketType.RANGING, 1.0 - abs(trend)

        # Determine trend type based on trend score
        strength = abs(trend)

        if trend > 0.6:
            return MarketType.STRONG_UPTREND, strength
        elif trend > 0.3:
            return MarketType.UPTREND, strength
        elif trend > 0.1:
            return MarketType.WEAK_UPTREND, strength
        elif trend < -0.6:
            return MarketType.STRONG_DOWNTREND, strength
        elif trend < -0.3:
            return MarketType.DOWNTREND, strength
        elif trend < -0.1:
            return MarketType.WEAK_DOWNTREND, strength
        else:
            return MarketType.RANGING, 1.0 - strength

    def detect_trend_change(
        self,
        candles: List[Candle],
        current_trend: float,
    ) -> tuple[TrendChange, float]:
        """
        Detect potential trend changes.

        Returns:
            Tuple of (TrendChange, confidence from 0 to 1)
        """
        if len(candles) < 30:
            return TrendChange.NO_CHANGE, 0.0

        closes = [c.close for c in candles]

        # Calculate short and long EMAs
        ema_9 = self.calculate_ema(closes, 9)
        ema_21 = self.calculate_ema(closes, 21)

        # Calculate previous EMAs (5 candles ago)
        prev_closes = closes[:-5]
        if len(prev_closes) >= 21:
            prev_ema_9 = self.calculate_ema(prev_closes, 9)
            prev_ema_21 = self.calculate_ema(prev_closes, 21)
        else:
            return TrendChange.NO_CHANGE, 0.0

        # Check for EMA crossovers
        current_cross = ema_9 - ema_21
        prev_cross = prev_ema_9 - prev_ema_21

        # Bullish crossover: EMA9 crosses above EMA21
        if current_cross > 0 and prev_cross <= 0:
            confidence = min(1.0, abs(current_cross / ema_21) * 100)
            return TrendChange.BULLISH_REVERSAL, confidence

        # Bearish crossover: EMA9 crosses below EMA21
        if current_cross < 0 and prev_cross >= 0:
            confidence = min(1.0, abs(current_cross / ema_21) * 100)
            return TrendChange.BEARISH_REVERSAL, confidence

        # Continuation patterns
        if current_trend > 0.3 and current_cross > prev_cross:
            return TrendChange.BULLISH_CONTINUATION, abs(current_trend)
        elif current_trend < -0.3 and current_cross < prev_cross:
            return TrendChange.BEARISH_CONTINUATION, abs(current_trend)

        return TrendChange.NO_CHANGE, 0.0

    def detect_market_uncertainty(self, candles: List[Candle]) -> tuple[bool, float, str]:
        """
        Detect if the market is in an uncertain/transitional state.

        Signs of uncertainty:
        1. EMAs are very close together (about to cross or just crossed)
        2. Recent volatility spike
        3. Conflicting timeframe signals
        4. Price oscillating around EMAs

        Returns:
            Tuple of (is_uncertain, uncertainty_score 0-1, reason)
        """
        if len(candles) < 30:
            return False, 0.0, ""

        closes = [c.close for c in candles]
        current_price = closes[-1]

        # Calculate EMAs
        ema_9 = self.calculate_ema(closes, 9)
        ema_21 = self.calculate_ema(closes, 21)
        ema_50 = self.calculate_ema(closes, 50) if len(closes) >= 50 else ema_21

        # 1. Check if EMAs are converging (about to cross)
        ema_spread = abs(ema_9 - ema_21) / ema_21
        emas_converging = ema_spread < 0.002  # Within 0.2%

        # 2. Check for volatility spike
        atr = self.calculate_atr(candles)
        avg_atr = sum(c.range for c in candles[-20:]) / 20
        volatility_spike = atr > avg_atr * 1.5

        # 3. Check if price is oscillating around EMAs
        price_above_ema9 = current_price > ema_9
        price_above_ema21 = current_price > ema_21
        ema9_above_ema21 = ema_9 > ema_21

        # Mixed signals = uncertainty
        signals_mixed = (price_above_ema9 != price_above_ema21) or \
                        (price_above_ema9 != ema9_above_ema21)

        # 4. Check recent price action for whipsaws
        recent_closes = closes[-10:]
        crosses_ema = 0
        for i in range(1, len(recent_closes)):
            if (recent_closes[i] > ema_21 and recent_closes[i-1] < ema_21) or \
               (recent_closes[i] < ema_21 and recent_closes[i-1] > ema_21):
                crosses_ema += 1
        whipsaw = crosses_ema >= 2  # 2+ crosses in last 10 candles

        # Calculate uncertainty score
        uncertainty_score = 0.0
        reasons = []

        if emas_converging:
            uncertainty_score += 0.35
            reasons.append("EMAs converging")

        if volatility_spike:
            uncertainty_score += 0.25
            reasons.append("volatility spike")

        if signals_mixed:
            uncertainty_score += 0.25
            reasons.append("mixed signals")

        if whipsaw:
            uncertainty_score += 0.30
            reasons.append("whipsaw detected")

        uncertainty_score = min(1.0, uncertainty_score)
        is_uncertain = uncertainty_score >= 0.5

        reason = ", ".join(reasons) if reasons else "stable"

        return is_uncertain, uncertainty_score, reason

    def detect_patterns(self, candles: List[Candle]) -> tuple[bool, bool, str]:
        """
        Detect bullish/bearish candlestick patterns.

        Returns:
            Tuple of (is_bullish, is_bearish, pattern_name)
        """
        if len(candles) < 3:
            return False, False, ""

        last = candles[-1]
        prev = candles[-2]
        prev2 = candles[-3]

        # Bullish Engulfing
        if (prev.is_bearish and last.is_bullish and
            last.open < prev.close and last.close > prev.open):
            return True, False, "bullish_engulfing"

        # Bearish Engulfing
        if (prev.is_bullish and last.is_bearish and
            last.open > prev.close and last.close < prev.open):
            return False, True, "bearish_engulfing"

        # Hammer (bullish)
        if (last.is_bullish and
            last.lower_wick > last.body_size * 2 and
            last.upper_wick < last.body_size * 0.5):
            return True, False, "hammer"

        # Shooting Star (bearish)
        if (last.is_bearish and
            last.upper_wick > last.body_size * 2 and
            last.lower_wick < last.body_size * 0.5):
            return False, True, "shooting_star"

        # Three White Soldiers (bullish)
        if (prev2.is_bullish and prev.is_bullish and last.is_bullish and
            prev.close > prev2.close and last.close > prev.close):
            return True, False, "three_white_soldiers"

        # Three Black Crows (bearish)
        if (prev2.is_bearish and prev.is_bearish and last.is_bearish and
            prev.close < prev2.close and last.close < prev.close):
            return False, True, "three_black_crows"

        # Morning Star (bullish)
        if (prev2.is_bearish and prev.body_percentage < 0.3 and last.is_bullish and
            last.close > (prev2.open + prev2.close) / 2):
            return True, False, "morning_star"

        # Evening Star (bearish)
        if (prev2.is_bullish and prev.body_percentage < 0.3 and last.is_bearish and
            last.close < (prev2.open + prev2.close) / 2):
            return False, True, "evening_star"

        return False, False, ""

    def find_support_resistance(
        self,
        candles: List[Candle],
    ) -> tuple[float, float]:
        """
        Find support and resistance levels.

        Returns:
            Tuple of (support, resistance)
        """
        if len(candles) < 20:
            return 0.0, 0.0

        # Get recent highs and lows
        highs = [c.high for c in candles[-50:]]
        lows = [c.low for c in candles[-50:]]

        # Simple approach: recent swing high/low
        resistance = max(highs[-20:])
        support = min(lows[-20:])

        return support, resistance

    def analyze(self, asset: str) -> Optional[ChartAnalysis]:
        """
        Perform comprehensive chart analysis.

        Args:
            asset: Asset symbol (BTC, ETH, etc.)

        Returns:
            ChartAnalysis object or None if analysis fails
        """
        cache_key = f"analysis_{asset}"
        now = time.time()

        # Check cache
        if cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if now - cached_time < self.cache_ttl:
                return cached_data

        try:
            # Fetch candles for multiple timeframes
            candles_1m = self.fetch_klines(asset, "1m", 60)
            candles_5m = self.fetch_klines(asset, "5m", 60)
            candles_15m = self.fetch_klines(asset, "15m", 100)
            candles_1h = self.fetch_klines(asset, "1h", 100)
            candles_4h = self.fetch_klines(asset, "4h", 50)

            if not candles_15m:
                logger.warning(f"No candle data for {asset}")
                return None

            current_price = candles_15m[-1].close

            # Calculate trends for each timeframe
            trend_1m = self.detect_trend(candles_1m) if candles_1m else 0.0
            trend_5m = self.detect_trend(candles_5m) if candles_5m else 0.0
            trend_15m = self.detect_trend(candles_15m)
            trend_1h = self.detect_trend(candles_1h) if candles_1h else 0.0
            trend_4h = self.detect_trend(candles_4h) if candles_4h else 0.0

            # Calculate price changes
            price_change_1h = 0.0
            price_change_4h = 0.0
            price_change_24h = 0.0

            if candles_1h and len(candles_1h) >= 2:
                price_change_1h = (current_price - candles_1h[-2].close) / candles_1h[-2].close

            if candles_4h and len(candles_4h) >= 2:
                price_change_4h = (current_price - candles_4h[-2].close) / candles_4h[-2].close

            if candles_1h and len(candles_1h) >= 24:
                price_change_24h = (current_price - candles_1h[-24].close) / candles_1h[-24].close

            # Calculate moving averages
            closes_15m = [c.close for c in candles_15m]
            ema_9 = self.calculate_ema(closes_15m, 9)
            ema_21 = self.calculate_ema(closes_15m, 21)
            ema_50 = self.calculate_ema(closes_15m, 50) if len(closes_15m) >= 50 else 0.0
            sma_20 = self.calculate_sma(closes_15m, 20)
            sma_50 = self.calculate_sma(closes_15m, 50) if len(closes_15m) >= 50 else 0.0

            # Calculate RSI
            rsi_14 = self.calculate_rsi(closes_15m, 14)

            # Calculate ATR
            atr = self.calculate_atr(candles_15m)

            # Detect market type
            avg_trend = (trend_15m + trend_1h + trend_4h) / 3
            market_type, trend_strength = self.detect_market_type(candles_15m, avg_trend)

            # Detect trend changes
            trend_change, change_confidence = self.detect_trend_change(candles_15m, avg_trend)

            # Detect patterns
            is_bullish, is_bearish, pattern_name = self.detect_patterns(candles_15m)

            # Find support/resistance
            support, resistance = self.find_support_resistance(candles_15m)

            # Calculate momentum
            momentum = (trend_1m + trend_5m + trend_15m) / 3
            momentum_increasing = trend_1m > trend_5m > trend_15m if trend_15m > 0 else trend_1m < trend_5m < trend_15m

            # Calculate volatility
            volatility = atr / current_price if current_price > 0 else 0.0

            # Determine bias
            bullish_signals = 0
            bearish_signals = 0

            if trend_1h > 0.2:
                bullish_signals += 1
            elif trend_1h < -0.2:
                bearish_signals += 1

            if trend_4h > 0.2:
                bullish_signals += 1
            elif trend_4h < -0.2:
                bearish_signals += 1

            if rsi_14 < 30:
                bullish_signals += 1  # Oversold
            elif rsi_14 > 70:
                bearish_signals += 1  # Overbought

            if current_price > ema_21:
                bullish_signals += 1
            else:
                bearish_signals += 1

            if is_bullish:
                bullish_signals += 1
            if is_bearish:
                bearish_signals += 1

            if trend_change == TrendChange.BULLISH_REVERSAL:
                bullish_signals += 2
            elif trend_change == TrendChange.BEARISH_REVERSAL:
                bearish_signals += 2

            total_signals = bullish_signals + bearish_signals
            if total_signals == 0:
                bias = "neutral"
                confidence = 0.5
            elif bullish_signals > bearish_signals:
                bias = "bullish"
                confidence = bullish_signals / total_signals
            elif bearish_signals > bullish_signals:
                bias = "bearish"
                confidence = bearish_signals / total_signals
            else:
                bias = "neutral"
                confidence = 0.5

            # Detect market uncertainty (trend changes, transitions)
            is_uncertain, uncertainty_score, uncertainty_reason = self.detect_market_uncertainty(candles_15m)

            # Also flag as uncertain if we just had a reversal
            if trend_change in [TrendChange.BULLISH_REVERSAL, TrendChange.BEARISH_REVERSAL]:
                is_uncertain = True
                uncertainty_score = max(uncertainty_score, 0.7)
                if uncertainty_reason:
                    uncertainty_reason += f", {trend_change.value}"
                else:
                    uncertainty_reason = trend_change.value

            analysis = ChartAnalysis(
                asset=asset,
                timestamp=datetime.now(timezone.utc),
                current_price=current_price,
                price_change_1h=price_change_1h,
                price_change_4h=price_change_4h,
                price_change_24h=price_change_24h,
                trend_1m=trend_1m,
                trend_5m=trend_5m,
                trend_15m=trend_15m,
                trend_1h=trend_1h,
                trend_4h=trend_4h,
                ema_9=ema_9,
                ema_21=ema_21,
                ema_50=ema_50,
                sma_20=sma_20,
                sma_50=sma_50,
                market_type=market_type,
                trend_strength=trend_strength,
                trend_change=trend_change,
                trend_change_confidence=change_confidence,
                volatility=volatility,
                atr=atr,
                support_level=support,
                resistance_level=resistance,
                rsi_14=rsi_14,
                momentum=momentum,
                momentum_increasing=momentum_increasing,
                is_bullish_pattern=is_bullish,
                is_bearish_pattern=is_bearish,
                pattern_name=pattern_name,
                is_uncertain=is_uncertain,
                uncertainty_score=uncertainty_score,
                uncertainty_reason=uncertainty_reason,
                bias=bias,
                confidence=confidence,
            )

            # Cache result
            self._cache[cache_key] = (analysis, now)

            return analysis

        except Exception as e:
            logger.error(f"Chart analysis failed for {asset}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None


# Global instance
_chart_analyzer: Optional[BinanceChartAnalyzer] = None


def get_chart_analyzer() -> BinanceChartAnalyzer:
    """Get global chart analyzer instance."""
    global _chart_analyzer
    if _chart_analyzer is None:
        _chart_analyzer = BinanceChartAnalyzer()
    return _chart_analyzer


def analyze_chart(asset: str) -> Optional[ChartAnalysis]:
    """
    Analyze chart for an asset.

    Args:
        asset: Asset symbol (BTC, ETH, etc.)

    Returns:
        ChartAnalysis or None
    """
    return get_chart_analyzer().analyze(asset)


def get_chart_bias(asset: str) -> tuple[str, float]:
    """
    Get chart bias for an asset.

    Args:
        asset: Asset symbol

    Returns:
        Tuple of (bias, confidence)
        bias: "bullish", "bearish", or "neutral"
        confidence: 0.0 to 1.0
    """
    analysis = analyze_chart(asset)
    if analysis:
        return analysis.bias, analysis.confidence
    return "neutral", 0.5


def get_market_type(asset: str) -> tuple[MarketType, float]:
    """
    Get market type for an asset.

    Returns:
        Tuple of (MarketType, strength)
    """
    analysis = analyze_chart(asset)
    if analysis:
        return analysis.market_type, analysis.trend_strength
    return MarketType.RANGING, 0.0


def is_trend_changing(asset: str) -> tuple[bool, TrendChange, float]:
    """
    Check if trend is changing for an asset.

    Returns:
        Tuple of (is_changing, direction, confidence)
    """
    analysis = analyze_chart(asset)
    if analysis:
        is_changing = analysis.trend_change in [
            TrendChange.BULLISH_REVERSAL,
            TrendChange.BEARISH_REVERSAL,
        ]
        return is_changing, analysis.trend_change, analysis.trend_change_confidence
    return False, TrendChange.NO_CHANGE, 0.0


def is_market_uncertain(asset: str) -> tuple[bool, float, str]:
    """
    Check if the market is in an uncertain/transitional state.

    This is used to pause trading during trend changes.

    Returns:
        Tuple of (is_uncertain, uncertainty_score, reason)
    """
    analysis = analyze_chart(asset)
    if analysis:
        return analysis.is_uncertain, analysis.uncertainty_score, analysis.uncertainty_reason
    return False, 0.0, ""


def should_pause_trading(asset: str, min_uncertainty: float = 0.5) -> tuple[bool, str]:
    """
    Determine if trading should be paused for an asset.

    Reasons to pause:
    1. Market uncertainty is high (trend change in progress)
    2. EMAs converging (about to cross)
    3. Recent reversal detected

    Args:
        asset: Asset symbol
        min_uncertainty: Minimum uncertainty score to trigger pause (default 0.5)

    Returns:
        Tuple of (should_pause, reason)
    """
    analysis = analyze_chart(asset)
    if not analysis:
        return False, ""

    reasons = []

    # Check uncertainty
    if analysis.is_uncertain and analysis.uncertainty_score >= min_uncertainty:
        reasons.append(f"market uncertain ({analysis.uncertainty_score:.0%}): {analysis.uncertainty_reason}")

    # Check for recent reversal
    if analysis.trend_change in [TrendChange.BULLISH_REVERSAL, TrendChange.BEARISH_REVERSAL]:
        reasons.append(f"trend reversal: {analysis.trend_change.value}")

    # Check if bias is neutral with low confidence (indecisive market)
    if analysis.bias == "neutral" and analysis.confidence < 0.55:
        reasons.append(f"indecisive market (bias={analysis.bias}, conf={analysis.confidence:.0%})")

    if reasons:
        return True, "; ".join(reasons)

    return False, ""
