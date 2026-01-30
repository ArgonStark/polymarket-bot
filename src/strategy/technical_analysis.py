"""
Technical Analysis module for smart trend-following trading.

This module provides professional-grade technical analysis for 15-minute
crypto prediction markets on Polymarket.

Strategy Overview:
- Multi-timeframe candle analysis (1m, 5m, 15m, 1h)
- Momentum indicators (EMA crossovers, RSI)
- Trend strength scoring
- Smart entry signals that FOLLOW the trend

Key Insight for 15-min markets:
- If trend is UP and price is BELOW target → BET YES (price will rise)
- If trend is DOWN and price is ABOVE target → BET NO (price will fall)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
from enum import Enum
import requests

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Trend direction."""
    STRONG_UP = "strong_up"
    UP = "up"
    NEUTRAL = "neutral"
    DOWN = "down"
    STRONG_DOWN = "strong_down"


class CandlePattern(Enum):
    """Candle patterns that indicate direction."""
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    DOJI = "doji"
    STRONG_BULLISH = "strong_bullish"
    STRONG_BEARISH = "strong_bearish"
    NONE = "none"


@dataclass
class Candle:
    """OHLCV candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def body_size(self) -> float:
        """Size of candle body (close - open)."""
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        """Size of upper wick."""
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        """Size of lower wick."""
        return min(self.open, self.close) - self.low

    @property
    def range(self) -> float:
        """Total candle range (high - low)."""
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        """True if candle closed higher than opened."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """True if candle closed lower than opened."""
        return self.close < self.open

    @property
    def body_pct(self) -> float:
        """Body as percentage of total range."""
        if self.range == 0:
            return 0.0
        return self.body_size / self.range


@dataclass
class TimeframeAnalysis:
    """Analysis results for a single timeframe."""
    timeframe: str  # "1m", "5m", "15m", "1h"
    trend: TrendDirection
    momentum: float  # -1 to 1
    ema_fast: float
    ema_slow: float
    rsi: float
    pattern: CandlePattern
    candles: list[Candle] = field(default_factory=list)


@dataclass
class TechnicalSignal:
    """Complete technical analysis signal."""
    asset: str
    timestamp: datetime

    # Multi-timeframe analysis
    tf_1m: Optional[TimeframeAnalysis] = None
    tf_5m: Optional[TimeframeAnalysis] = None
    tf_15m: Optional[TimeframeAnalysis] = None
    tf_1h: Optional[TimeframeAnalysis] = None

    # Aggregate scores
    trend_score: float = 0.0  # -1 (strong down) to 1 (strong up)
    momentum_score: float = 0.0  # -1 to 1
    confidence: float = 0.0  # 0 to 1

    # Trading recommendation
    recommended_side: str = "NONE"  # "UP", "DOWN", "NONE"
    edge_adjustment: float = 0.0  # How much to adjust edge

    @property
    def all_timeframes_aligned(self) -> bool:
        """Check if all timeframes agree on direction."""
        trends = []
        for tf in [self.tf_1m, self.tf_5m, self.tf_15m, self.tf_1h]:
            if tf is not None:
                if tf.trend in [TrendDirection.UP, TrendDirection.STRONG_UP]:
                    trends.append("UP")
                elif tf.trend in [TrendDirection.DOWN, TrendDirection.STRONG_DOWN]:
                    trends.append("DOWN")
                else:
                    trends.append("NEUTRAL")

        if not trends:
            return False

        # All must be same direction (ignoring neutrals)
        non_neutral = [t for t in trends if t != "NEUTRAL"]
        if not non_neutral:
            return False

        return len(set(non_neutral)) == 1


class TechnicalAnalyzer:
    """
    Technical analysis engine for crypto assets.

    Uses Binance API to fetch candle data and performs multi-timeframe analysis.
    """

    # Binance API endpoints
    BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

    # Symbol mapping
    SYMBOL_MAP = {
        "BTC": "BTCUSDT",
        "ETH": "ETHUSDT",
        "SOL": "SOLUSDT",
        "XRP": "XRPUSDT",
    }

    # EMA periods
    EMA_FAST = 9
    EMA_SLOW = 21

    # RSI period
    RSI_PERIOD = 14

    def __init__(self):
        """Initialize the technical analyzer."""
        self._cache: dict[str, tuple[datetime, list[Candle]]] = {}
        self._cache_ttl = timedelta(seconds=5)  # Cache candles for 5 seconds

    def analyze(self, asset: str) -> Optional[TechnicalSignal]:
        """
        Perform complete technical analysis for an asset.

        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP)

        Returns:
            TechnicalSignal with analysis results, or None if analysis fails
        """
        asset = asset.upper()
        if asset not in self.SYMBOL_MAP:
            logger.warning(f"Unsupported asset: {asset}")
            return None

        symbol = self.SYMBOL_MAP[asset]
        now = datetime.now(timezone.utc)

        signal = TechnicalSignal(
            asset=asset,
            timestamp=now,
        )

        # Analyze each timeframe
        try:
            signal.tf_1m = self._analyze_timeframe(symbol, "1m", limit=50)
            signal.tf_5m = self._analyze_timeframe(symbol, "5m", limit=30)
            signal.tf_15m = self._analyze_timeframe(symbol, "15m", limit=20)
            signal.tf_1h = self._analyze_timeframe(symbol, "1h", limit=20)
        except Exception as e:
            logger.error(f"Failed to analyze {asset}: {e}")
            return None

        # Calculate aggregate scores
        self._calculate_aggregate_scores(signal)

        return signal

    def _analyze_timeframe(
        self,
        symbol: str,
        interval: str,
        limit: int = 50,
    ) -> TimeframeAnalysis:
        """
        Analyze a single timeframe.

        Args:
            symbol: Binance symbol (e.g., "BTCUSDT")
            interval: Candle interval (e.g., "1m", "5m")
            limit: Number of candles to fetch

        Returns:
            TimeframeAnalysis with results
        """
        # Fetch candles
        candles = self._fetch_candles(symbol, interval, limit)

        if len(candles) < 20:
            return TimeframeAnalysis(
                timeframe=interval,
                trend=TrendDirection.NEUTRAL,
                momentum=0.0,
                ema_fast=0.0,
                ema_slow=0.0,
                rsi=50.0,
                pattern=CandlePattern.NONE,
                candles=candles,
            )

        # Calculate indicators
        closes = [c.close for c in candles]

        ema_fast = self._calculate_ema(closes, self.EMA_FAST)
        ema_slow = self._calculate_ema(closes, self.EMA_SLOW)
        rsi = self._calculate_rsi(closes, self.RSI_PERIOD)

        # Determine trend
        trend = self._determine_trend(candles, ema_fast, ema_slow, rsi)

        # Calculate momentum
        momentum = self._calculate_momentum(candles, ema_fast, ema_slow)

        # Detect candle pattern
        pattern = self._detect_pattern(candles[-3:]) if len(candles) >= 3 else CandlePattern.NONE

        return TimeframeAnalysis(
            timeframe=interval,
            trend=trend,
            momentum=momentum,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            rsi=rsi,
            pattern=pattern,
            candles=candles[-10:],  # Keep last 10 for analysis
        )

    def _fetch_candles(
        self,
        symbol: str,
        interval: str,
        limit: int,
    ) -> list[Candle]:
        """
        Fetch candles from Binance API.

        Uses caching to avoid excessive API calls.
        """
        cache_key = f"{symbol}_{interval}"
        now = datetime.now(timezone.utc)

        # Check cache
        if cache_key in self._cache:
            cache_time, candles = self._cache[cache_key]
            if now - cache_time < self._cache_ttl:
                return candles

        try:
            response = requests.get(
                self.BINANCE_KLINES_URL,
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "limit": limit,
                },
                timeout=5,
            )
            response.raise_for_status()
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
                )
                candles.append(candle)

            # Update cache
            self._cache[cache_key] = (now, candles)

            return candles

        except Exception as e:
            logger.error(f"Failed to fetch candles for {symbol} {interval}: {e}")
            # Return cached data if available (even if stale)
            if cache_key in self._cache:
                return self._cache[cache_key][1]
            return []

    def _calculate_ema(self, prices: list[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return prices[-1] if prices else 0.0

        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period  # Start with SMA

        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _calculate_rsi(self, prices: list[float], period: int = 14) -> float:
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

        # Calculate initial averages
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        # Smooth with subsequent values
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _determine_trend(
        self,
        candles: list[Candle],
        ema_fast: float,
        ema_slow: float,
        rsi: float,
    ) -> TrendDirection:
        """Determine trend direction from indicators."""
        if not candles:
            return TrendDirection.NEUTRAL

        current_price = candles[-1].close

        # Score based on multiple factors
        score = 0

        # EMA crossover
        if ema_fast > ema_slow:
            score += 1
            if (ema_fast - ema_slow) / ema_slow > 0.001:  # Strong crossover
                score += 1
        elif ema_fast < ema_slow:
            score -= 1
            if (ema_slow - ema_fast) / ema_slow > 0.001:
                score -= 1

        # Price vs EMAs
        if current_price > ema_fast > ema_slow:
            score += 1
        elif current_price < ema_fast < ema_slow:
            score -= 1

        # RSI
        if rsi > 70:
            score += 1  # Overbought but still bullish momentum
        elif rsi > 55:
            score += 0.5
        elif rsi < 30:
            score -= 1  # Oversold but still bearish momentum
        elif rsi < 45:
            score -= 0.5

        # Recent candles direction
        recent = candles[-5:]
        bullish_count = sum(1 for c in recent if c.is_bullish)
        if bullish_count >= 4:
            score += 1
        elif bullish_count <= 1:
            score -= 1

        # Determine trend from score
        if score >= 3:
            return TrendDirection.STRONG_UP
        elif score >= 1:
            return TrendDirection.UP
        elif score <= -3:
            return TrendDirection.STRONG_DOWN
        elif score <= -1:
            return TrendDirection.DOWN
        else:
            return TrendDirection.NEUTRAL

    def _calculate_momentum(
        self,
        candles: list[Candle],
        ema_fast: float,
        ema_slow: float,
    ) -> float:
        """
        Calculate momentum score from -1 to 1.

        Positive = bullish momentum
        Negative = bearish momentum
        """
        if not candles or len(candles) < 5:
            return 0.0

        # EMA momentum
        if ema_slow > 0:
            ema_momentum = (ema_fast - ema_slow) / ema_slow
        else:
            ema_momentum = 0.0

        # Price change momentum (last 5 candles)
        price_start = candles[-5].close
        price_end = candles[-1].close
        if price_start > 0:
            price_momentum = (price_end - price_start) / price_start
        else:
            price_momentum = 0.0

        # Combine (weighted)
        momentum = ema_momentum * 0.6 + price_momentum * 0.4

        # Clamp to -1 to 1
        return max(-1.0, min(1.0, momentum * 100))

    def _detect_pattern(self, candles: list[Candle]) -> CandlePattern:
        """Detect candle patterns in the last few candles."""
        if len(candles) < 2:
            return CandlePattern.NONE

        last = candles[-1]
        prev = candles[-2]

        # Strong bullish candle
        if last.is_bullish and last.body_pct > 0.7 and last.body_size > prev.body_size:
            return CandlePattern.STRONG_BULLISH

        # Strong bearish candle
        if last.is_bearish and last.body_pct > 0.7 and last.body_size > prev.body_size:
            return CandlePattern.STRONG_BEARISH

        # Bullish engulfing
        if (prev.is_bearish and last.is_bullish and
            last.close > prev.open and last.open < prev.close):
            return CandlePattern.BULLISH_ENGULFING

        # Bearish engulfing
        if (prev.is_bullish and last.is_bearish and
            last.open > prev.close and last.close < prev.open):
            return CandlePattern.BEARISH_ENGULFING

        # Hammer (bullish reversal)
        if (last.lower_wick > last.body_size * 2 and
            last.upper_wick < last.body_size * 0.5):
            return CandlePattern.HAMMER

        # Shooting star (bearish reversal)
        if (last.upper_wick > last.body_size * 2 and
            last.lower_wick < last.body_size * 0.5):
            return CandlePattern.SHOOTING_STAR

        # Doji
        if last.body_pct < 0.1 and last.range > 0:
            return CandlePattern.DOJI

        return CandlePattern.NONE

    def _calculate_aggregate_scores(self, signal: TechnicalSignal):
        """Calculate aggregate trend and momentum scores."""
        trend_scores = []
        momentum_scores = []
        weights = {
            "1m": 0.15,   # Short-term timing
            "5m": 0.25,   # Short momentum
            "15m": 0.35,  # Main trend (matches market duration)
            "1h": 0.25,   # Larger context
        }

        for tf_name, tf_analysis in [
            ("1m", signal.tf_1m),
            ("5m", signal.tf_5m),
            ("15m", signal.tf_15m),
            ("1h", signal.tf_1h),
        ]:
            if tf_analysis is None:
                continue

            weight = weights[tf_name]

            # Convert trend to score
            trend_map = {
                TrendDirection.STRONG_UP: 1.0,
                TrendDirection.UP: 0.5,
                TrendDirection.NEUTRAL: 0.0,
                TrendDirection.DOWN: -0.5,
                TrendDirection.STRONG_DOWN: -1.0,
            }
            trend_scores.append(trend_map[tf_analysis.trend] * weight)
            momentum_scores.append(tf_analysis.momentum * weight)

        if trend_scores:
            signal.trend_score = sum(trend_scores)
            signal.momentum_score = sum(momentum_scores)

        # Calculate confidence based on alignment
        if signal.all_timeframes_aligned:
            signal.confidence = 0.8 + abs(signal.trend_score) * 0.2
        elif abs(signal.trend_score) > 0.3:
            signal.confidence = 0.5 + abs(signal.trend_score) * 0.3
        else:
            signal.confidence = 0.3

        # Determine recommended side
        if signal.trend_score > 0.3 and signal.confidence > 0.5:
            signal.recommended_side = "UP"
            signal.edge_adjustment = signal.trend_score * 0.03  # Up to 3% boost
        elif signal.trend_score < -0.3 and signal.confidence > 0.5:
            signal.recommended_side = "DOWN"
            signal.edge_adjustment = abs(signal.trend_score) * 0.03
        else:
            signal.recommended_side = "NONE"
            signal.edge_adjustment = 0.0


@dataclass
class SmartTradingDecision:
    """Decision from smart trend-following strategy."""
    should_trade: bool
    side: str  # "UP", "DOWN", "NONE"
    edge_boost: float
    confidence: float
    reason: str
    technical_signal: Optional[TechnicalSignal] = None


class SmartTrendFollower:
    """
    Smart trend-following strategy for 15-minute markets.

    Key Logic (BOOST ONLY - no blocking):
    - Analyzes multi-timeframe trends (1m, 5m, 15m, 1h)
    - BOOSTS trades that align with the trend
    - ALLOWS trades against trend (no boost, reduced confidence)
    - Does NOT block trades - that's handled by TrendProtection (1h/4h/1d)

    NOTE: This system uses short-term analysis (1m-1h) while TrendProtection
    uses longer-term analysis (1h-1d). They can show different trends!
    To avoid contradictions, this system only boosts, never blocks.
    """

    def __init__(self):
        """Initialize the smart trend follower."""
        self.analyzer = TechnicalAnalyzer()

    def evaluate(
        self,
        asset: str,
        current_price: float,
        target_price: float,
        time_remaining: float,
        original_side: str,  # Side from arbitrage/probability calculation
        original_edge: float,
    ) -> SmartTradingDecision:
        """
        Evaluate whether to trade and in which direction.

        Args:
            asset: Asset symbol (BTC, ETH, etc.)
            current_price: Current Chainlink price
            target_price: Market target price
            time_remaining: Seconds until market closes
            original_side: Side from original calculation ("UP" or "DOWN")
            original_edge: Edge from original calculation

        Returns:
            SmartTradingDecision with recommendation
        """
        # Get technical analysis
        tech_signal = self.analyzer.analyze(asset)

        if tech_signal is None:
            return SmartTradingDecision(
                should_trade=True,  # Allow original signal if analysis fails
                side=original_side,
                edge_boost=0.0,
                confidence=0.5,
                reason="Technical analysis unavailable - using original signal",
            )

        # Calculate price position relative to target
        if target_price > 0:
            price_vs_target = (current_price - target_price) / target_price
        else:
            price_vs_target = 0.0

        price_below_target = current_price < target_price
        price_above_target = current_price > target_price

        # Determine if trend supports the trade
        trend_is_bullish = tech_signal.trend_score > 0.2
        trend_is_bearish = tech_signal.trend_score < -0.2
        trend_is_neutral = not trend_is_bullish and not trend_is_bearish

        # === SMART TRADING LOGIC ===

        # Scenario 1: Strong bullish trend + price below target = BET YES
        if trend_is_bullish and price_below_target:
            # Trend will push price UP towards/past target
            if original_side == "UP":
                # Original signal agrees with trend - BOOST
                return SmartTradingDecision(
                    should_trade=True,
                    side="UP",
                    edge_boost=tech_signal.edge_adjustment,
                    confidence=tech_signal.confidence,
                    reason=f"Strong UP trend ({tech_signal.trend_score:.2f}) + price below target = HIGH confidence YES",
                    technical_signal=tech_signal,
                )
            else:
                # Original says DOWN but short-term trend is UP - allow but no boost
                # (TrendProtection handles long-term blocking)
                return SmartTradingDecision(
                    should_trade=True,
                    side=original_side,
                    edge_boost=0.0,
                    confidence=0.3,
                    reason=f"Short-term UP trend ({tech_signal.trend_score:.2f}) - no boost for DOWN",
                    technical_signal=tech_signal,
                )

        # Scenario 2: Strong bearish trend + price above target = BET NO
        if trend_is_bearish and price_above_target:
            # Trend will push price DOWN towards/past target
            if original_side == "DOWN":
                # Original signal agrees with trend - BOOST
                return SmartTradingDecision(
                    should_trade=True,
                    side="DOWN",
                    edge_boost=tech_signal.edge_adjustment,
                    confidence=tech_signal.confidence,
                    reason=f"Strong DOWN trend ({tech_signal.trend_score:.2f}) + price above target = HIGH confidence NO",
                    technical_signal=tech_signal,
                )
            else:
                # Original says UP but short-term trend is DOWN - allow but no boost
                # (TrendProtection handles long-term blocking)
                return SmartTradingDecision(
                    should_trade=True,
                    side=original_side,
                    edge_boost=0.0,
                    confidence=0.3,
                    reason=f"Short-term DOWN trend ({tech_signal.trend_score:.2f}) - no boost for UP",
                    technical_signal=tech_signal,
                )

        # Scenario 3: Trading AGAINST the trend - allow but reduce confidence (no boost)
        # Note: TrendProtection handles long-term blocking, we only handle short-term boosts
        if trend_is_bullish and original_side == "DOWN":
            return SmartTradingDecision(
                should_trade=True,
                side=original_side,
                edge_boost=0.0,
                confidence=0.3,
                reason=f"Against short-term UP trend ({tech_signal.trend_score:.2f}) - no boost",
                technical_signal=tech_signal,
            )

        if trend_is_bearish and original_side == "UP":
            return SmartTradingDecision(
                should_trade=True,
                side=original_side,
                edge_boost=0.0,
                confidence=0.3,
                reason=f"Against short-term DOWN trend ({tech_signal.trend_score:.2f}) - no boost",
                technical_signal=tech_signal,
            )

        # Scenario 4: Neutral trend - allow original signal but no boost
        if trend_is_neutral:
            return SmartTradingDecision(
                should_trade=True,
                side=original_side,
                edge_boost=0.0,
                confidence=0.5,
                reason=f"Neutral trend ({tech_signal.trend_score:.2f}) - using original signal without boost",
                technical_signal=tech_signal,
            )

        # Scenario 5: Trend aligns with price position and original signal
        if original_side in ["UP", "DOWN"]:
            # All signals align
            if tech_signal.all_timeframes_aligned:
                return SmartTradingDecision(
                    should_trade=True,
                    side=original_side,
                    edge_boost=tech_signal.edge_adjustment * 1.5,  # Extra boost for alignment
                    confidence=tech_signal.confidence,
                    reason=f"All timeframes aligned ({tech_signal.trend_score:.2f}) - HIGH confidence",
                    technical_signal=tech_signal,
                )
            else:
                return SmartTradingDecision(
                    should_trade=True,
                    side=original_side,
                    edge_boost=tech_signal.edge_adjustment * 0.5,
                    confidence=tech_signal.confidence * 0.8,
                    reason=f"Partial alignment ({tech_signal.trend_score:.2f}) - MEDIUM confidence",
                    technical_signal=tech_signal,
                )

        # Default: Allow with no adjustment
        return SmartTradingDecision(
            should_trade=True,
            side=original_side,
            edge_boost=0.0,
            confidence=0.5,
            reason="No clear trend signal - using original",
            technical_signal=tech_signal,
        )


# Global instance for easy access
smart_trend_follower = SmartTrendFollower()


def get_smart_trading_decision(
    asset: str,
    current_price: float,
    target_price: float,
    time_remaining: float,
    original_side: str,
    original_edge: float,
) -> SmartTradingDecision:
    """
    Get smart trading decision based on technical analysis.

    This is the main entry point for the smart trend-following strategy.
    """
    return smart_trend_follower.evaluate(
        asset=asset,
        current_price=current_price,
        target_price=target_price,
        time_remaining=time_remaining,
        original_side=original_side,
        original_edge=original_edge,
    )
