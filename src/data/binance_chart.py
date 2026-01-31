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
class SupplyDemandZone:
    """
    A supply or demand zone on the chart.

    Supply zones (resistance): Areas where price was rejected down
    Demand zones (support): Areas where price bounced up
    """
    zone_type: str  # "supply" or "demand"
    price_low: float  # Lower boundary of zone
    price_high: float  # Upper boundary of zone
    strength: int  # Number of times price reacted at this zone
    timeframe: str  # "15m", "1h", "4h"
    last_tested: datetime  # When zone was last tested
    broken: bool = False  # True if price has decisively broken through

    @property
    def midpoint(self) -> float:
        """Get the zone midpoint price."""
        return (self.price_low + self.price_high) / 2

    @property
    def width(self) -> float:
        """Get zone width as a price."""
        return self.price_high - self.price_low

    def contains_price(self, price: float) -> bool:
        """Check if price is within this zone."""
        return self.price_low <= price <= self.price_high

    def distance_from_price(self, price: float) -> float:
        """Get distance from zone (negative if inside)."""
        if price < self.price_low:
            return self.price_low - price
        elif price > self.price_high:
            return price - self.price_high
        else:
            return 0.0  # Inside zone


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

    # Graduated position sizing (replaces binary pause)
    position_size_multiplier: float = 1.0  # 0.0 to 1.0 based on uncertainty

    # Multi-timeframe confirmation
    timeframes_aligned: bool = True  # True if 15m, 1h trends agree
    alignment_score: float = 1.0  # 0.0 (disagreement) to 1.0 (full alignment)
    higher_timeframe_bias: str = "neutral"  # 1h trend direction

    # Trend break detection
    trend_strength_dropping: bool = False  # True if strength dropped >30% recently
    ema_distance_pct: float = 0.0  # Distance between EMA9 and EMA21 as %

    # Resume confirmation tracking
    consecutive_candles_same_dir: int = 0  # Count of candles in same direction
    resume_ready: bool = True  # False during pause, True when safe to resume
    resume_confidence: float = 1.0  # Confidence in resuming (0-1)

    # Supply/Demand Zones
    supply_zones: List[SupplyDemandZone] = field(default_factory=list)  # Resistance areas
    demand_zones: List[SupplyDemandZone] = field(default_factory=list)  # Support areas
    nearest_supply: Optional[float] = None  # Nearest supply zone price
    nearest_demand: Optional[float] = None  # Nearest demand zone price
    in_supply_zone: bool = False  # True if price is in a supply zone
    in_demand_zone: bool = False  # True if price is in a demand zone
    supply_zone_strength: int = 0  # Strength of nearest supply zone
    demand_zone_strength: int = 0  # Strength of nearest demand zone

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
            # Graduated position sizing
            "position_size_multiplier": self.position_size_multiplier,
            # Multi-timeframe
            "timeframes_aligned": self.timeframes_aligned,
            "alignment_score": self.alignment_score,
            "higher_timeframe_bias": self.higher_timeframe_bias,
            # Trend break
            "trend_strength_dropping": self.trend_strength_dropping,
            "ema_distance_pct": self.ema_distance_pct,
            # Resume
            "consecutive_candles_same_dir": self.consecutive_candles_same_dir,
            "resume_ready": self.resume_ready,
            "resume_confidence": self.resume_confidence,
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
        Detect trend direction and strength using CONTINUOUS values.

        Uses actual price distances from EMAs rather than binary flags
        to provide more granular trend readings.

        Returns:
            Float from -1 (strong downtrend) to +1 (strong uptrend)
        """
        if len(candles) < 20:
            return 0.0

        closes = [c.close for c in candles]
        current_price = closes[-1]

        # Calculate EMAs
        ema_9 = self.calculate_ema(closes, 9)
        ema_21 = self.calculate_ema(closes, 21)
        ema_50 = self.calculate_ema(closes, min(50, len(closes)))

        trend_score = 0.0

        # === 1. PRICE DISTANCE FROM EMAs (continuous, up to ±0.35) ===
        # How far is price from EMA21? Further = stronger trend
        ema21_distance = (current_price - ema_21) / ema_21
        # Scale: 0.5% distance = 0.1 score, cap at 1.5% = 0.3
        ema_score = max(-0.35, min(0.35, ema21_distance * 20))
        trend_score += ema_score

        # === 2. EMA SEPARATION (continuous, up to ±0.25) ===
        # How far apart are EMA9 and EMA21? Wider = stronger trend
        ema_separation = (ema_9 - ema_21) / ema_21
        # Scale: 0.3% separation = 0.1 score, cap at 0.75% = 0.25
        separation_score = max(-0.25, min(0.25, ema_separation * 33))
        trend_score += separation_score

        # === 3. MOMENTUM (last 5 candles, up to ±0.20) ===
        if len(closes) >= 5:
            momentum = (closes[-1] - closes[-5]) / closes[-5]
            # Scale: 1% move in 5 candles = 0.1 score
            momentum_score = max(-0.20, min(0.20, momentum * 10))
            trend_score += momentum_score

        # === 4. CANDLE DIRECTION (last 3 candles, up to ±0.10) ===
        if len(candles) >= 3:
            bullish_count = sum(1 for c in candles[-3:] if c.is_bullish)
            bearish_count = sum(1 for c in candles[-3:] if c.is_bearish)
            # 3 bullish = +0.10, 3 bearish = -0.10
            candle_score = (bullish_count - bearish_count) / 3 * 0.10
            trend_score += candle_score

        # === 5. SLOPE OF EMA21 (recent direction, up to ±0.10) ===
        if len(closes) >= 25:
            # Compare current EMA21 vs EMA21 from 5 candles ago
            old_closes = closes[:-5]
            old_ema_21 = self.calculate_ema(old_closes, 21)
            ema_slope = (ema_21 - old_ema_21) / old_ema_21
            slope_score = max(-0.10, min(0.10, ema_slope * 50))
            trend_score += slope_score

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

    def calculate_graduated_position_size(
        self,
        uncertainty_score: float,
        alignment_score: float,
        trend_strength_dropping: bool,
    ) -> float:
        """
        Calculate graduated position size multiplier based on market conditions.

        Instead of binary pause (trade or don't trade), we scale position size:
        - Full confidence (1.0): 100% position
        - Moderate uncertainty: 50-75% position
        - High uncertainty: 25-50% position
        - Extreme uncertainty: 0% (full pause)

        Args:
            uncertainty_score: Market uncertainty (0-1)
            alignment_score: Timeframe alignment (0-1)
            trend_strength_dropping: True if trend is weakening

        Returns:
            Position size multiplier (0.0 to 1.0)
        """
        # Start with full position
        multiplier = 1.0

        # Reduce based on uncertainty (graduated, not binary)
        if uncertainty_score < 0.3:
            # Low uncertainty: full position
            uncertainty_penalty = 0.0
        elif uncertainty_score < 0.5:
            # Moderate: 75% position
            uncertainty_penalty = 0.25
        elif uncertainty_score < 0.7:
            # High: 50% position
            uncertainty_penalty = 0.50
        elif uncertainty_score < 0.85:
            # Very high: 25% position
            uncertainty_penalty = 0.75
        else:
            # Extreme: full pause
            uncertainty_penalty = 1.0

        multiplier -= uncertainty_penalty

        # Reduce based on timeframe misalignment
        if alignment_score < 0.5:
            # Major disagreement: reduce by 25%
            multiplier *= 0.75
        elif alignment_score < 0.75:
            # Moderate disagreement: reduce by 10%
            multiplier *= 0.90

        # Reduce if trend is breaking
        if trend_strength_dropping:
            multiplier *= 0.75  # Extra 25% reduction

        return max(0.0, min(1.0, multiplier))

    def calculate_timeframe_alignment(
        self,
        trend_15m: float,
        trend_1h: float,
        trend_4h: float,
    ) -> tuple[bool, float, str]:
        """
        Calculate multi-timeframe alignment score.

        Alignment is higher when all timeframes agree on direction.
        Perfect alignment: all positive or all negative.
        Misalignment: mixed signals.

        Returns:
            Tuple of (is_aligned, alignment_score 0-1, higher_timeframe_bias)
        """
        # Determine higher timeframe bias (1h is most important)
        if trend_1h > 0.2:
            higher_tf_bias = "bullish"
        elif trend_1h < -0.2:
            higher_tf_bias = "bearish"
        else:
            higher_tf_bias = "neutral"

        # Check if all trends agree on direction
        all_positive = trend_15m > 0 and trend_1h > 0 and trend_4h > 0
        all_negative = trend_15m < 0 and trend_1h < 0 and trend_4h < 0

        # Calculate alignment score
        if all_positive or all_negative:
            # Full alignment
            alignment_score = 1.0
            is_aligned = True
        else:
            # Calculate how aligned they are
            # Count how many are in each direction
            positive_count = sum(1 for t in [trend_15m, trend_1h, trend_4h] if t > 0.1)
            negative_count = sum(1 for t in [trend_15m, trend_1h, trend_4h] if t < -0.1)
            neutral_count = 3 - positive_count - negative_count

            # Alignment is higher when more agree
            max_agreement = max(positive_count, negative_count)
            alignment_score = max_agreement / 3.0

            # 15m disagreeing with 1h is particularly bad
            if (trend_15m > 0.2 and trend_1h < -0.2) or (trend_15m < -0.2 and trend_1h > 0.2):
                alignment_score *= 0.5  # Heavy penalty

            is_aligned = alignment_score >= 0.66  # At least 2/3 agree

        return is_aligned, alignment_score, higher_tf_bias

    def detect_trend_break(
        self,
        candles: List[Candle],
        current_trend_strength: float,
    ) -> tuple[bool, float, float]:
        """
        Detect if trend is breaking down.

        Signs of trend break:
        1. EMA distance collapsing (EMAs converging)
        2. Rapid drop in trend strength
        3. Price whipsawing through EMAs

        Returns:
            Tuple of (is_breaking, ema_distance_pct, strength_change)
        """
        if len(candles) < 30:
            return False, 0.0, 0.0

        closes = [c.close for c in candles]

        # Calculate current EMAs
        ema_9 = self.calculate_ema(closes, 9)
        ema_21 = self.calculate_ema(closes, 21)

        # EMA distance as percentage
        ema_distance_pct = abs(ema_9 - ema_21) / ema_21 * 100

        # Calculate trend strength 5 candles ago for comparison
        if len(closes) > 5:
            closes_5_ago = closes[:-5]
            old_trend = self.detect_trend(candles[:-5])
            strength_change = abs(current_trend_strength) - abs(old_trend)
        else:
            strength_change = 0.0

        # Detect trend break conditions
        is_breaking = False

        # EMAs very close (about to cross or just crossed)
        if ema_distance_pct < 0.15:  # Within 0.15%
            is_breaking = True

        # Rapid strength drop (>30% drop in 5 candles)
        if strength_change < -0.3:
            is_breaking = True

        return is_breaking, ema_distance_pct, strength_change

    def calculate_resume_confidence(
        self,
        candles: List[Candle],
        rsi: float,
        uncertainty_score: float,
    ) -> tuple[bool, int, float]:
        """
        Calculate confidence in resuming trading after a pause.

        Requirements to resume with full confidence:
        1. 3+ consecutive candles in same direction
        2. RSI decisive (not in 40-60 neutral zone)
        3. Uncertainty has dropped below 50%

        Returns:
            Tuple of (resume_ready, consecutive_candles, confidence)
        """
        if len(candles) < 5:
            return True, 0, 1.0

        # Count consecutive candles in same direction
        consecutive = 0
        last_direction = None

        for candle in reversed(candles[-10:]):
            if candle.is_bullish:
                direction = "up"
            elif candle.is_bearish:
                direction = "down"
            else:
                direction = "neutral"

            if last_direction is None:
                last_direction = direction
                consecutive = 1
            elif direction == last_direction and direction != "neutral":
                consecutive += 1
            else:
                break

        # Check RSI decisiveness
        rsi_decisive = rsi < 40 or rsi > 60

        # Calculate resume confidence
        confidence = 0.0

        # Consecutive candles (max 0.4)
        confidence += min(consecutive / 5.0, 0.4)  # 5 candles = max 0.4

        # RSI decisiveness (max 0.3)
        if rsi_decisive:
            confidence += 0.3
        elif rsi < 45 or rsi > 55:
            confidence += 0.15

        # Low uncertainty (max 0.3)
        if uncertainty_score < 0.3:
            confidence += 0.3
        elif uncertainty_score < 0.5:
            confidence += 0.15

        # Ready to resume if confidence >= 0.6 and not highly uncertain
        resume_ready = confidence >= 0.6 and uncertainty_score < 0.6

        return resume_ready, consecutive, confidence

    def detect_patterns(self, candles: List[Candle]) -> tuple[bool, bool, str]:
        """
        Detect bullish/bearish candlestick patterns for SHORT-TERM trading.

        For 15-minute Polymarket markets, we want to catch:
        - Bullish bounces even in a downtrend (dead cat bounces are tradeable)
        - Bearish pullbacks even in an uptrend
        - Any short-term reversal signal

        Returns:
            Tuple of (is_bullish, is_bearish, pattern_name)
        """
        if len(candles) < 3:
            return False, False, ""

        last = candles[-1]
        prev = candles[-2]
        prev2 = candles[-3]

        # === BULLISH PATTERNS (detect bounces, even in downtrend) ===

        # Bullish Engulfing - strong short-term reversal signal
        if (prev.is_bearish and last.is_bullish and
            last.open < prev.close and last.close > prev.open):
            return True, False, "bullish_engulfing"

        # Hammer (bullish) - rejection of lower prices
        if (last.is_bullish and
            last.lower_wick > last.body_size * 2 and
            last.upper_wick < last.body_size * 0.5):
            return True, False, "hammer"

        # Three White Soldiers (bullish)
        if (prev2.is_bullish and prev.is_bullish and last.is_bullish and
            prev.close > prev2.close and last.close > prev.close):
            return True, False, "three_white_soldiers"

        # Morning Star (bullish reversal)
        if (prev2.is_bearish and prev.body_percentage < 0.3 and last.is_bullish and
            last.close > (prev2.open + prev2.close) / 2):
            return True, False, "morning_star"

        # Bullish Harami - potential bounce
        if (prev.is_bearish and last.is_bullish and
            last.open > prev.close and last.close < prev.open and
            last.body_size < prev.body_size * 0.5):
            return True, False, "bullish_harami"

        # Double bottom hint (two consecutive lows at similar level)
        if len(candles) >= 5:
            recent_lows = [c.low for c in candles[-5:]]
            min_low = min(recent_lows)
            if (last.is_bullish and
                abs(last.low - min_low) / min_low < 0.002 and  # Within 0.2%
                last.close > last.open):
                return True, False, "double_bottom_hint"

        # === BEARISH PATTERNS (detect pullbacks, even in uptrend) ===

        # Bearish Engulfing
        if (prev.is_bullish and last.is_bearish and
            last.open > prev.close and last.close < prev.open):
            return False, True, "bearish_engulfing"

        # Shooting Star (bearish)
        if (last.is_bearish and
            last.upper_wick > last.body_size * 2 and
            last.lower_wick < last.body_size * 0.5):
            return False, True, "shooting_star"

        # Three Black Crows (bearish)
        if (prev2.is_bearish and prev.is_bearish and last.is_bearish and
            prev.close < prev2.close and last.close < prev.close):
            return False, True, "three_black_crows"

        # Evening Star (bearish)
        if (prev2.is_bullish and prev.body_percentage < 0.3 and last.is_bearish and
            last.close < (prev2.open + prev2.close) / 2):
            return False, True, "evening_star"

        # Bearish Harami - potential pullback
        if (prev.is_bullish and last.is_bearish and
            last.open < prev.close and last.close > prev.open and
            last.body_size < prev.body_size * 0.5):
            return False, True, "bearish_harami"

        # Double top hint (two consecutive highs at similar level)
        if len(candles) >= 5:
            recent_highs = [c.high for c in candles[-5:]]
            max_high = max(recent_highs)
            if (last.is_bearish and
                abs(last.high - max_high) / max_high < 0.002 and  # Within 0.2%
                last.close < last.open):
                return False, True, "double_top_hint"

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

    def find_supply_demand_zones(
        self,
        candles: List[Candle],
        timeframe: str = "15m",
        min_touches: int = 2,
    ) -> tuple[List[SupplyDemandZone], List[SupplyDemandZone]]:
        """
        Find supply (resistance) and demand (support) zones.

        Supply zones: Areas where price dropped after testing
        Demand zones: Areas where price bounced after testing

        Args:
            candles: List of candles to analyze
            timeframe: Timeframe label for zones
            min_touches: Minimum touches to confirm a zone

        Returns:
            Tuple of (supply_zones, demand_zones)
        """
        if len(candles) < 30:
            return [], []

        supply_zones = []
        demand_zones = []
        current_price = candles[-1].close
        now = candles[-1].timestamp

        # Calculate ATR for zone width
        atr = self.calculate_atr(candles)
        zone_width = atr * 0.5  # Zone is half ATR wide

        # Find swing highs (potential supply zones)
        # A swing high is a candle whose high is higher than neighbors
        for i in range(5, len(candles) - 5):
            candle = candles[i]

            # Check if this is a swing high
            is_swing_high = all(
                candle.high >= candles[j].high
                for j in range(i - 3, i + 4)
                if j != i and 0 <= j < len(candles)
            )

            if is_swing_high:
                zone_high = candle.high
                zone_low = candle.high - zone_width

                # Count how many times price touched this zone and reversed
                touches = 0
                for j in range(i + 1, len(candles)):
                    test_candle = candles[j]
                    # Price entered zone
                    if test_candle.high >= zone_low and test_candle.high <= zone_high + zone_width:
                        # And then price fell (bearish candle or next candle lower)
                        if test_candle.close < test_candle.open or (j + 1 < len(candles) and candles[j + 1].close < test_candle.close):
                            touches += 1

                if touches >= min_touches:
                    # Check if zone is broken
                    broken = current_price > zone_high + zone_width

                    supply_zones.append(SupplyDemandZone(
                        zone_type="supply",
                        price_low=zone_low,
                        price_high=zone_high,
                        strength=touches,
                        timeframe=timeframe,
                        last_tested=candles[-1].timestamp,
                        broken=broken,
                    ))

            # Check if this is a swing low (potential demand zone)
            is_swing_low = all(
                candle.low <= candles[j].low
                for j in range(i - 3, i + 4)
                if j != i and 0 <= j < len(candles)
            )

            if is_swing_low:
                zone_low = candle.low
                zone_high = candle.low + zone_width

                # Count how many times price touched this zone and bounced
                touches = 0
                for j in range(i + 1, len(candles)):
                    test_candle = candles[j]
                    # Price entered zone
                    if test_candle.low <= zone_high and test_candle.low >= zone_low - zone_width:
                        # And then price rose (bullish candle or next candle higher)
                        if test_candle.close > test_candle.open or (j + 1 < len(candles) and candles[j + 1].close > test_candle.close):
                            touches += 1

                if touches >= min_touches:
                    # Check if zone is broken
                    broken = current_price < zone_low - zone_width

                    demand_zones.append(SupplyDemandZone(
                        zone_type="demand",
                        price_low=zone_low,
                        price_high=zone_high,
                        strength=touches,
                        timeframe=timeframe,
                        last_tested=candles[-1].timestamp,
                        broken=broken,
                    ))

        # Remove duplicate/overlapping zones, keeping strongest
        supply_zones = self._merge_overlapping_zones(supply_zones)
        demand_zones = self._merge_overlapping_zones(demand_zones)

        # Sort by distance from current price
        supply_zones.sort(key=lambda z: z.midpoint - current_price if z.midpoint > current_price else float('inf'))
        demand_zones.sort(key=lambda z: current_price - z.midpoint if z.midpoint < current_price else float('inf'))

        # Keep only top 3 closest zones
        supply_zones = supply_zones[:3]
        demand_zones = demand_zones[:3]

        return supply_zones, demand_zones

    def _merge_overlapping_zones(self, zones: List[SupplyDemandZone]) -> List[SupplyDemandZone]:
        """Merge overlapping zones, keeping the strongest."""
        if not zones:
            return []

        # Sort by price_low
        zones = sorted(zones, key=lambda z: z.price_low)
        merged = [zones[0]]

        for zone in zones[1:]:
            last = merged[-1]
            # Check if zones overlap
            if zone.price_low <= last.price_high:
                # Merge: extend the zone and combine strength
                if zone.strength > last.strength:
                    # Replace with stronger zone
                    merged[-1] = SupplyDemandZone(
                        zone_type=zone.zone_type,
                        price_low=min(last.price_low, zone.price_low),
                        price_high=max(last.price_high, zone.price_high),
                        strength=zone.strength + last.strength,
                        timeframe=zone.timeframe,
                        last_tested=zone.last_tested,
                        broken=zone.broken and last.broken,
                    )
                else:
                    merged[-1] = SupplyDemandZone(
                        zone_type=last.zone_type,
                        price_low=min(last.price_low, zone.price_low),
                        price_high=max(last.price_high, zone.price_high),
                        strength=zone.strength + last.strength,
                        timeframe=last.timeframe,
                        last_tested=last.last_tested,
                        broken=zone.broken and last.broken,
                    )
            else:
                merged.append(zone)

        return merged

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

            # Find supply/demand zones (multi-timeframe)
            supply_zones_15m, demand_zones_15m = self.find_supply_demand_zones(candles_15m, "15m")
            supply_zones_1h, demand_zones_1h = [], []
            supply_zones_4h, demand_zones_4h = [], []
            if candles_1h:
                supply_zones_1h, demand_zones_1h = self.find_supply_demand_zones(candles_1h, "1h", min_touches=1)
            if candles_4h:
                supply_zones_4h, demand_zones_4h = self.find_supply_demand_zones(candles_4h, "4h", min_touches=1)

            # Combine all zones
            all_supply_zones = supply_zones_15m + supply_zones_1h + supply_zones_4h
            all_demand_zones = demand_zones_15m + demand_zones_1h + demand_zones_4h

            # Find nearest zones to current price
            nearest_supply = None
            nearest_demand = None
            in_supply_zone = False
            in_demand_zone = False
            supply_zone_strength = 0
            demand_zone_strength = 0

            for zone in all_supply_zones:
                if not zone.broken:
                    if zone.contains_price(current_price):
                        in_supply_zone = True
                        supply_zone_strength = max(supply_zone_strength, zone.strength)
                    elif zone.midpoint > current_price:
                        if nearest_supply is None or zone.midpoint < nearest_supply:
                            nearest_supply = zone.midpoint
                            supply_zone_strength = zone.strength

            for zone in all_demand_zones:
                if not zone.broken:
                    if zone.contains_price(current_price):
                        in_demand_zone = True
                        demand_zone_strength = max(demand_zone_strength, zone.strength)
                    elif zone.midpoint < current_price:
                        if nearest_demand is None or zone.midpoint > nearest_demand:
                            nearest_demand = zone.midpoint
                            demand_zone_strength = zone.strength

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

            # === MULTI-TIMEFRAME ALIGNMENT ===
            timeframes_aligned, alignment_score, higher_tf_bias = self.calculate_timeframe_alignment(
                trend_15m, trend_1h, trend_4h
            )

            # === TREND BREAK DETECTION ===
            trend_strength_dropping, ema_distance_pct, strength_change = self.detect_trend_break(
                candles_15m, trend_strength
            )

            # === GRADUATED POSITION SIZING ===
            position_size_multiplier = self.calculate_graduated_position_size(
                uncertainty_score, alignment_score, trend_strength_dropping
            )

            # === RESUME CONFIDENCE ===
            resume_ready, consecutive_candles, resume_confidence = self.calculate_resume_confidence(
                candles_15m, rsi_14, uncertainty_score
            )

            # If currently paused (high uncertainty), check if we can resume
            if is_uncertain and uncertainty_score >= 0.6:
                if not resume_ready:
                    # Still in pause mode
                    position_size_multiplier = 0.0
                else:
                    # Resuming - use reduced position initially
                    position_size_multiplier = min(position_size_multiplier, 0.5)  # Cooldown: 50% max

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
                # Graduated position sizing
                position_size_multiplier=position_size_multiplier,
                # Multi-timeframe confirmation
                timeframes_aligned=timeframes_aligned,
                alignment_score=alignment_score,
                higher_timeframe_bias=higher_tf_bias,
                # Trend break detection
                trend_strength_dropping=trend_strength_dropping,
                ema_distance_pct=ema_distance_pct,
                # Resume confirmation
                consecutive_candles_same_dir=consecutive_candles,
                resume_ready=resume_ready,
                resume_confidence=resume_confidence,
                # Supply/Demand zones
                supply_zones=all_supply_zones,
                demand_zones=all_demand_zones,
                nearest_supply=nearest_supply,
                nearest_demand=nearest_demand,
                in_supply_zone=in_supply_zone,
                in_demand_zone=in_demand_zone,
                supply_zone_strength=supply_zone_strength,
                demand_zone_strength=demand_zone_strength,
                # Bias
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


# Cooldown tracking for post-uncertainty trading
# Maps asset -> {"trades_since_resume": int, "last_uncertainty_time": datetime, "in_cooldown": bool}
_cooldown_tracker: Dict[str, dict] = {}


def get_position_size_multiplier(asset: str) -> tuple[float, str]:
    """
    Get the recommended position size multiplier for an asset.

    This implements graduated position sizing instead of binary pause:
    - 1.0: Full position (low uncertainty, aligned timeframes)
    - 0.75: Reduced position (moderate uncertainty)
    - 0.5: Half position (high uncertainty or cooldown)
    - 0.25: Quarter position (very high uncertainty)
    - 0.0: Full pause (extreme uncertainty, not ready to resume)

    Returns:
        Tuple of (multiplier 0-1, reason)
    """
    analysis = analyze_chart(asset)
    if not analysis:
        return 1.0, "no chart data"

    multiplier = analysis.position_size_multiplier
    reasons = []

    if analysis.is_uncertain:
        reasons.append(f"uncertain ({analysis.uncertainty_score:.0%})")

    if not analysis.timeframes_aligned:
        reasons.append(f"timeframes misaligned ({analysis.alignment_score:.0%})")

    if analysis.trend_strength_dropping:
        reasons.append("trend breaking")

    if not analysis.resume_ready and analysis.is_uncertain:
        reasons.append(f"waiting for resume ({analysis.resume_confidence:.0%})")

    # Check cooldown
    cooldown_info = _cooldown_tracker.get(asset.upper(), {})
    if cooldown_info.get("in_cooldown", False):
        trades_since = cooldown_info.get("trades_since_resume", 0)
        if trades_since < 2:
            multiplier = min(multiplier, 0.5)
            reasons.append(f"cooldown ({trades_since}/2 trades)")

    reason = ", ".join(reasons) if reasons else "full confidence"
    return multiplier, reason


def record_trade_for_cooldown(asset: str, uncertainty_was_high: bool = False):
    """
    Record a trade for cooldown tracking.

    Call this after each trade to track cooldown progress.
    """
    asset_upper = asset.upper()

    if asset_upper not in _cooldown_tracker:
        _cooldown_tracker[asset_upper] = {
            "trades_since_resume": 0,
            "last_uncertainty_time": None,
            "in_cooldown": False,
        }

    tracker = _cooldown_tracker[asset_upper]

    if uncertainty_was_high:
        # Just came out of high uncertainty - start cooldown
        tracker["in_cooldown"] = True
        tracker["trades_since_resume"] = 0
        tracker["last_uncertainty_time"] = datetime.now(timezone.utc)
    elif tracker["in_cooldown"]:
        # In cooldown - count trades
        tracker["trades_since_resume"] += 1
        if tracker["trades_since_resume"] >= 2:
            # Cooldown complete
            tracker["in_cooldown"] = False


def get_multi_timeframe_decision(asset: str) -> tuple[str, float, bool]:
    """
    Get trading decision based on multi-timeframe analysis.

    Returns:
        Tuple of (recommended_side "UP"/"DOWN"/"NONE", confidence, should_trade)
    """
    analysis = analyze_chart(asset)
    if not analysis:
        return "NONE", 0.5, True

    # Use higher timeframe bias as primary direction
    if analysis.higher_timeframe_bias == "bullish" and analysis.alignment_score >= 0.66:
        return "UP", analysis.alignment_score, True
    elif analysis.higher_timeframe_bias == "bearish" and analysis.alignment_score >= 0.66:
        return "DOWN", analysis.alignment_score, True
    elif analysis.alignment_score < 0.5:
        # Significant disagreement - be cautious
        return "NONE", analysis.alignment_score, False
    else:
        return "NONE", analysis.alignment_score, True


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
