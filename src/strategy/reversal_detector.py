"""
Reversal Detection System

Detects when price is likely to reverse direction using:
1. RSI oversold/overbought
2. Bollinger Band touches
3. Candlestick patterns
4. Support/Resistance bounces
5. Volume exhaustion

This helps catch bounces - when price goes below target but will likely
come back up (or vice versa).
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum

logger = logging.getLogger(__name__)


class ReversalType(Enum):
    BULLISH = "bullish"  # Price likely to go UP
    BEARISH = "bearish"  # Price likely to go DOWN
    NONE = "none"        # No reversal signal


@dataclass
class ReversalSignal:
    """Result of reversal analysis."""
    reversal_type: ReversalType
    confidence: float  # 0.0 to 1.0
    reasons: List[str]

    # Key indicators
    rsi: float
    bb_position: float  # -1 (lower band) to +1 (upper band)
    candle_pattern: str
    near_support: bool
    near_resistance: bool


@dataclass
class CandleData:
    """Single candle data."""
    open: float
    high: float
    low: float
    close: float
    volume: float


def calculate_rsi(closes: List[float], period: int = 14) -> float:
    """Calculate RSI from closing prices."""
    if len(closes) < period + 1:
        return 50.0  # Neutral if not enough data

    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]

    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_bollinger_position(
    price: float,
    closes: List[float],
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[float, float, float, float]:
    """
    Calculate Bollinger Band position.

    Returns:
        (position, lower_band, middle_band, upper_band)
        position: -1 = at lower band, 0 = at middle, +1 = at upper band
    """
    if len(closes) < period:
        return 0.0, price * 0.98, price, price * 1.02

    recent = closes[-period:]
    middle = sum(recent) / period

    variance = sum((x - middle) ** 2 for x in recent) / period
    std = variance ** 0.5

    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)

    # Calculate position (-1 to +1)
    band_width = upper - lower
    if band_width > 0:
        position = (price - middle) / (band_width / 2)
        position = max(-1.0, min(1.0, position))
    else:
        position = 0.0

    return position, lower, middle, upper


def detect_candle_pattern(candles: List[CandleData]) -> str:
    """
    Detect candlestick patterns.

    Returns pattern name or "none".
    """
    if len(candles) < 3:
        return "none"

    current = candles[-1]
    prev = candles[-2]

    body = abs(current.close - current.open)
    upper_wick = current.high - max(current.close, current.open)
    lower_wick = min(current.close, current.open) - current.low
    total_range = current.high - current.low

    if total_range == 0:
        return "doji"

    body_ratio = body / total_range

    # Doji - very small body
    if body_ratio < 0.1:
        return "doji"

    # Hammer - small body at top, long lower wick (BULLISH)
    if (lower_wick > body * 2 and
        upper_wick < body * 0.5 and
        current.close > current.open):
        return "hammer"

    # Inverted Hammer (BULLISH at bottom)
    if (upper_wick > body * 2 and
        lower_wick < body * 0.5 and
        current.close > current.open):
        return "inverted_hammer"

    # Shooting Star - small body at bottom, long upper wick (BEARISH)
    if (upper_wick > body * 2 and
        lower_wick < body * 0.5 and
        current.close < current.open):
        return "shooting_star"

    # Bullish Engulfing
    if (prev.close < prev.open and  # Previous red
        current.close > current.open and  # Current green
        current.open < prev.close and  # Opens below prev close
        current.close > prev.open):  # Closes above prev open
        return "bullish_engulfing"

    # Bearish Engulfing
    if (prev.close > prev.open and  # Previous green
        current.close < current.open and  # Current red
        current.open > prev.close and  # Opens above prev close
        current.close < prev.open):  # Closes below prev open
        return "bearish_engulfing"

    return "none"


def find_support_resistance(
    candles: List[CandleData],
    current_price: float,
    threshold_pct: float = 0.002  # 0.2%
) -> Tuple[bool, bool, Optional[float], Optional[float]]:
    """
    Find if price is near support or resistance levels.

    Returns:
        (near_support, near_resistance, support_level, resistance_level)
    """
    if len(candles) < 10:
        return False, False, None, None

    # Find recent swing lows (support) and highs (resistance)
    lows = [c.low for c in candles[-20:]]
    highs = [c.high for c in candles[-20:]]

    # Simple: use recent min/max
    support = min(lows)
    resistance = max(highs)

    # Check if near support (within threshold)
    near_support = abs(current_price - support) / support < threshold_pct
    near_resistance = abs(current_price - resistance) / resistance < threshold_pct

    return near_support, near_resistance, support, resistance


def detect_reversal(
    current_price: float,
    target_price: float,
    closes: List[float],
    candles: List[CandleData],
    volumes: Optional[List[float]] = None,
) -> ReversalSignal:
    """
    Main reversal detection function.

    Analyzes multiple indicators to detect potential reversals.

    Args:
        current_price: Current price
        target_price: Target price for the market
        closes: List of recent closing prices (newest last)
        candles: List of recent candles (newest last)
        volumes: Optional list of volumes

    Returns:
        ReversalSignal with type, confidence, and reasons
    """
    reasons = []
    bullish_score = 0.0
    bearish_score = 0.0

    # 1. RSI Analysis
    rsi = calculate_rsi(closes)

    if rsi < 30:
        bullish_score += 0.3
        reasons.append(f"RSI oversold ({rsi:.0f})")
    elif rsi < 40:
        bullish_score += 0.1
        reasons.append(f"RSI low ({rsi:.0f})")
    elif rsi > 70:
        bearish_score += 0.3
        reasons.append(f"RSI overbought ({rsi:.0f})")
    elif rsi > 60:
        bearish_score += 0.1
        reasons.append(f"RSI high ({rsi:.0f})")

    # 2. Bollinger Band Analysis
    bb_position, bb_lower, bb_middle, bb_upper = calculate_bollinger_position(
        current_price, closes
    )

    if bb_position < -0.8:  # Near lower band
        bullish_score += 0.25
        reasons.append(f"Near lower Bollinger Band")
    elif bb_position < -0.5:
        bullish_score += 0.1
    elif bb_position > 0.8:  # Near upper band
        bearish_score += 0.25
        reasons.append(f"Near upper Bollinger Band")
    elif bb_position > 0.5:
        bearish_score += 0.1

    # 3. Candlestick Pattern
    candle_pattern = detect_candle_pattern(candles)

    bullish_patterns = ["hammer", "inverted_hammer", "bullish_engulfing", "doji"]
    bearish_patterns = ["shooting_star", "bearish_engulfing"]

    if candle_pattern in bullish_patterns:
        # Only count as bullish if price is below target or falling
        if current_price <= target_price:
            bullish_score += 0.2
            reasons.append(f"Bullish pattern: {candle_pattern}")
    elif candle_pattern in bearish_patterns:
        if current_price >= target_price:
            bearish_score += 0.2
            reasons.append(f"Bearish pattern: {candle_pattern}")

    # 4. Support/Resistance
    near_support, near_resistance, support, resistance = find_support_resistance(
        candles, current_price
    )

    if near_support and current_price < target_price:
        bullish_score += 0.15
        reasons.append(f"Near support level ${support:.2f}")

    if near_resistance and current_price > target_price:
        bearish_score += 0.15
        reasons.append(f"Near resistance level ${resistance:.2f}")

    # 5. Price position vs target
    distance_pct = (current_price - target_price) / target_price

    # If price is significantly below target but showing bullish signals
    if distance_pct < -0.002 and bullish_score > 0.3:  # 0.2% below
        bullish_score += 0.1
        reasons.append(f"Below target with bullish signals")

    # If price is significantly above target but showing bearish signals
    if distance_pct > 0.002 and bearish_score > 0.3:  # 0.2% above
        bearish_score += 0.1
        reasons.append(f"Above target with bearish signals")

    # 6. Volume analysis (if available)
    if volumes and len(volumes) >= 5:
        recent_vol = sum(volumes[-3:]) / 3
        avg_vol = sum(volumes[-10:]) / 10

        if recent_vol > avg_vol * 1.5:
            # High volume can indicate exhaustion/reversal
            if bullish_score > bearish_score:
                bullish_score += 0.1
                reasons.append("High volume (potential exhaustion)")
            elif bearish_score > bullish_score:
                bearish_score += 0.1
                reasons.append("High volume (potential exhaustion)")

    # Determine reversal type
    if bullish_score > bearish_score and bullish_score >= 0.4:
        reversal_type = ReversalType.BULLISH
        confidence = min(0.85, bullish_score)
    elif bearish_score > bullish_score and bearish_score >= 0.4:
        reversal_type = ReversalType.BEARISH
        confidence = min(0.85, bearish_score)
    else:
        reversal_type = ReversalType.NONE
        confidence = 0.0

    return ReversalSignal(
        reversal_type=reversal_type,
        confidence=confidence,
        reasons=reasons,
        rsi=rsi,
        bb_position=bb_position,
        candle_pattern=candle_pattern,
        near_support=near_support,
        near_resistance=near_resistance,
    )


def get_reversal_recommendation(
    current_price: float,
    target_price: float,
    reversal: ReversalSignal,
    time_remaining: float,
) -> Tuple[str, float, str]:
    """
    Get trading recommendation based on reversal analysis.

    Returns:
        (direction, confidence, reason)
        direction: "UP", "DOWN", or "SKIP"
    """
    distance_pct = (current_price - target_price) / target_price

    # Case 1: Price below target + Bullish reversal → bet UP (expecting bounce)
    if distance_pct < 0 and reversal.reversal_type == ReversalType.BULLISH:
        confidence = reversal.confidence * 0.8  # Slightly reduce for reversal trades
        if time_remaining > 300:  # More than 5 min left
            confidence *= 0.9  # More time = more uncertainty
        reason = f"REVERSAL UP: {', '.join(reversal.reasons[:2])}"
        return "UP", confidence, reason

    # Case 2: Price above target + Bearish reversal → bet DOWN (expecting drop)
    if distance_pct > 0 and reversal.reversal_type == ReversalType.BEARISH:
        confidence = reversal.confidence * 0.8
        if time_remaining > 300:
            confidence *= 0.9
        reason = f"REVERSAL DOWN: {', '.join(reversal.reasons[:2])}"
        return "DOWN", confidence, reason

    # Case 3: Price below target + No reversal → bet DOWN (continue trend)
    if distance_pct < -0.001 and reversal.reversal_type == ReversalType.NONE:
        confidence = 0.55 + abs(distance_pct) * 10  # Higher confidence if further
        confidence = min(0.75, confidence)
        reason = f"Below target ({distance_pct:.2%}), no reversal signal"
        return "DOWN", confidence, reason

    # Case 4: Price above target + No reversal → bet UP (continue trend)
    if distance_pct > 0.001 and reversal.reversal_type == ReversalType.NONE:
        confidence = 0.55 + abs(distance_pct) * 10
        confidence = min(0.75, confidence)
        reason = f"Above target ({distance_pct:.2%}), no reversal signal"
        return "UP", confidence, reason

    # Case 5: At target, no clear signal → SKIP
    return "SKIP", 0.50, "At target, no clear signal"


# Example usage
if __name__ == "__main__":
    # Test data
    closes = [100, 101, 100.5, 99, 98, 97, 96.5, 96, 95.5, 95, 94.5, 94, 93.5, 93, 92.5]
    candles = [
        CandleData(100, 101, 99, 100.5, 1000),
        CandleData(100.5, 101, 98, 99, 1200),
        CandleData(99, 99.5, 96, 96.5, 1500),
        CandleData(96.5, 97, 94, 94.5, 1800),
        CandleData(94.5, 95, 92, 93, 2000),  # Current candle
    ]

    reversal = detect_reversal(
        current_price=93,
        target_price=95,
        closes=closes,
        candles=candles,
    )

    print(f"Reversal Type: {reversal.reversal_type.value}")
    print(f"Confidence: {reversal.confidence:.2f}")
    print(f"RSI: {reversal.rsi:.0f}")
    print(f"BB Position: {reversal.bb_position:.2f}")
    print(f"Candle Pattern: {reversal.candle_pattern}")
    print(f"Reasons: {reversal.reasons}")

    direction, conf, reason = get_reversal_recommendation(
        current_price=93,
        target_price=95,
        reversal=reversal,
        time_remaining=600,
    )
    print(f"\nRecommendation: {direction} ({conf:.0%}) - {reason}")
