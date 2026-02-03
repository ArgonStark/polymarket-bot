"""
Simple Signal Generator for Polymarket 15-Minute Markets.

Decision Hierarchy:
1. DISTANCE from target (PRIMARY) - If price far from target, trust it
2. TREND direction (SECONDARY) - If price near target, follow the trend
3. MOMENTUM (CONFIRMATION) - Confirms or warns about the signal

Why this works:
- Price far from target rarely crosses in 15 minutes
- Trends persist: downtrend → likely closes DOWN, uptrend → likely closes UP
- Ranging markets: use distance from target
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class Conviction(Enum):
    HIGH = "high"      # >80% confident
    MEDIUM = "medium"  # 65-80% confident
    LOW = "low"        # 55-65% confident


class Trend(Enum):
    STRONG_UP = "strong_up"      # Clear uptrend
    UP = "up"                    # Moderate uptrend
    RANGING = "ranging"          # No clear trend
    DOWN = "down"                # Moderate downtrend
    STRONG_DOWN = "strong_down"  # Clear downtrend


@dataclass
class SimpleSignal:
    direction: str           # "UP" or "DOWN"
    conviction: Conviction
    win_probability: float   # Our estimated win rate
    edge: float             # vs market odds
    reason: str
    size_multiplier: float  # 0.2 to 0.8

    # Key data
    distance_pct: float     # Distance from target
    trend: Trend            # Market trend
    momentum: float         # Short-term movement
    time_remaining: float   # Seconds left


def detect_trend(
    trend_15m: float,
    trend_1h: float,
    trend_4h: float,
) -> Tuple[Trend, float]:
    """
    Detect market trend from multi-timeframe data.

    Args:
        trend_15m: 15-minute trend (-1 to +1)
        trend_1h: 1-hour trend (-1 to +1)
        trend_4h: 4-hour trend (-1 to +1)

    Returns:
        (Trend enum, strength 0-1)
    """
    # Weight: 15m=40%, 1h=35%, 4h=25%
    weighted = trend_15m * 0.40 + trend_1h * 0.35 + trend_4h * 0.25
    strength = abs(weighted)

    # Agreement bonus: if all timeframes agree, stronger signal
    all_bullish = trend_15m > 0.05 and trend_1h > 0.05 and trend_4h > 0
    all_bearish = trend_15m < -0.05 and trend_1h < -0.05 and trend_4h < 0

    if all_bullish:
        strength = min(1.0, strength * 1.3)
    elif all_bearish:
        strength = min(1.0, strength * 1.3)

    # Classify trend
    if weighted > 0.25 or (all_bullish and weighted > 0.15):
        return Trend.STRONG_UP, strength
    elif weighted > 0.10:
        return Trend.UP, strength
    elif weighted < -0.25 or (all_bearish and weighted < -0.15):
        return Trend.STRONG_DOWN, strength
    elif weighted < -0.10:
        return Trend.DOWN, strength
    else:
        return Trend.RANGING, strength


def generate_simple_signal(
    asset: str,
    current_price: float,
    target_price: float,
    time_remaining: float,
    momentum: float = 0.0,      # -1 to +1 (from Binance velocity)
    trend_15m: float = 0.0,     # 15-minute trend
    trend_1h: float = 0.0,      # 1-hour trend
    trend_4h: float = 0.0,      # 4-hour trend
    market_odds_up: float = 0.50,
    market_odds_down: float = 0.50,
) -> SimpleSignal:
    """
    Generate signal using:
    1. Distance from target (PRIMARY)
    2. Trend direction (SECONDARY)
    3. Momentum (CONFIRMATION)

    Logic:
    ┌────────────────────────────────────────────────────────────────────┐
    │ Scenario                          │ Signal │ Conviction │ Win %   │
    ├────────────────────────────────────────────────────────────────────┤
    │ Far above target (>0.3%)          │ UP     │ HIGH       │ 85%+    │
    │ Far below target (>0.3%)          │ DOWN   │ HIGH       │ 85%+    │
    │ Near target + Strong Downtrend    │ DOWN   │ HIGH       │ 75-80%  │
    │ Near target + Strong Uptrend      │ UP     │ HIGH       │ 75-80%  │
    │ Near target + Moderate Trend      │ Follow │ MEDIUM     │ 65-70%  │
    │ Near target + Ranging             │ By pos │ LOW        │ 55-60%  │
    └────────────────────────────────────────────────────────────────────┘
    """

    # ==========================================================================
    # STEP 1: Calculate key metrics
    # ==========================================================================
    distance_pct = (current_price - target_price) / target_price
    distance_abs = abs(distance_pct)
    is_above = distance_pct > 0
    is_below = distance_pct < 0

    # Time factors
    time_minutes = time_remaining / 60
    is_final_5min = time_minutes < 5
    is_final_2min = time_minutes < 2

    # Detect trend
    trend, trend_strength = detect_trend(trend_15m, trend_1h, trend_4h)

    # Momentum checks
    momentum_up = momentum > 0.15
    momentum_down = momentum < -0.15
    momentum_strong = abs(momentum) > 0.35

    # ==========================================================================
    # STEP 2: Make decision
    # ==========================================================================
    direction = None
    win_prob = 0.50
    conviction = Conviction.LOW
    reason = ""

    # -------------------------------------------------------------------------
    # CASE A: Price FAR from target (>0.3%) - Distance is PRIMARY
    # -------------------------------------------------------------------------
    if distance_abs > 0.003:
        if is_above:
            direction = "UP"
            win_prob = 0.78 + min(0.12, distance_abs * 15)
            reason = f"{distance_abs:.2%} above target"
        else:
            direction = "DOWN"
            win_prob = 0.78 + min(0.12, distance_abs * 15)
            reason = f"{distance_abs:.2%} below target"

        # Time bonus
        if is_final_5min:
            win_prob += 0.03

        # Momentum confirmation/warning
        if (direction == "UP" and momentum_down and momentum_strong):
            win_prob -= 0.05
            reason += " (momentum against)"
        elif (direction == "DOWN" and momentum_up and momentum_strong):
            win_prob -= 0.05
            reason += " (momentum against)"
        elif (direction == "UP" and momentum_up):
            win_prob += 0.02
            reason += " + momentum"
        elif (direction == "DOWN" and momentum_down):
            win_prob += 0.02
            reason += " + momentum"

        conviction = Conviction.HIGH if win_prob >= 0.80 else Conviction.MEDIUM

    # -------------------------------------------------------------------------
    # CASE B: Price MODERATELY from target (0.15%-0.3%) - Consider trend
    # -------------------------------------------------------------------------
    elif distance_abs > 0.0015:
        # Start with distance-based direction
        if is_above:
            base_direction = "UP"
            base_prob = 0.62 + min(0.08, distance_abs * 25)
        else:
            base_direction = "DOWN"
            base_prob = 0.62 + min(0.08, distance_abs * 25)

        # Check if trend overrides
        if trend in [Trend.STRONG_DOWN, Trend.DOWN]:
            if base_direction == "DOWN":
                # Trend confirms distance
                direction = "DOWN"
                win_prob = base_prob + 0.08
                reason = f"{distance_abs:.2%} below + downtrend"
                conviction = Conviction.HIGH if win_prob >= 0.75 else Conviction.MEDIUM
            else:
                # Conflict: above target but downtrend
                # Trust distance if significant, else trust trend
                if distance_abs > 0.002:
                    direction = "UP"
                    win_prob = base_prob - 0.05  # Reduced confidence
                    reason = f"{distance_abs:.2%} above, downtrend warning"
                    conviction = Conviction.MEDIUM
                else:
                    direction = "DOWN"
                    win_prob = 0.65 + trend_strength * 0.10
                    reason = f"Near target, following downtrend"
                    conviction = Conviction.MEDIUM

        elif trend in [Trend.STRONG_UP, Trend.UP]:
            if base_direction == "UP":
                # Trend confirms distance
                direction = "UP"
                win_prob = base_prob + 0.08
                reason = f"{distance_abs:.2%} above + uptrend"
                conviction = Conviction.HIGH if win_prob >= 0.75 else Conviction.MEDIUM
            else:
                # Conflict: below target but uptrend
                if distance_abs > 0.002:
                    direction = "DOWN"
                    win_prob = base_prob - 0.05
                    reason = f"{distance_abs:.2%} below, uptrend warning"
                    conviction = Conviction.MEDIUM
                else:
                    direction = "UP"
                    win_prob = 0.65 + trend_strength * 0.10
                    reason = f"Near target, following uptrend"
                    conviction = Conviction.MEDIUM

        else:  # Ranging
            direction = base_direction
            win_prob = base_prob
            reason = f"{distance_abs:.2%} {'above' if is_above else 'below'}, ranging"
            conviction = Conviction.MEDIUM

    # -------------------------------------------------------------------------
    # CASE C: Price AT target (<0.15%) - TREND is PRIMARY
    # -------------------------------------------------------------------------
    else:
        if trend == Trend.STRONG_DOWN:
            direction = "DOWN"
            win_prob = 0.72 + trend_strength * 0.08
            reason = f"At target, strong downtrend"
            conviction = Conviction.HIGH if win_prob >= 0.75 else Conviction.MEDIUM

        elif trend == Trend.DOWN:
            direction = "DOWN"
            win_prob = 0.65 + trend_strength * 0.08
            reason = f"At target, downtrend"
            conviction = Conviction.MEDIUM

        elif trend == Trend.STRONG_UP:
            direction = "UP"
            win_prob = 0.72 + trend_strength * 0.08
            reason = f"At target, strong uptrend"
            conviction = Conviction.HIGH if win_prob >= 0.75 else Conviction.MEDIUM

        elif trend == Trend.UP:
            direction = "UP"
            win_prob = 0.65 + trend_strength * 0.08
            reason = f"At target, uptrend"
            conviction = Conviction.MEDIUM

        else:  # Ranging - use momentum or micro-distance
            if momentum_strong:
                direction = "UP" if momentum_up else "DOWN"
                win_prob = 0.58
                reason = f"Ranging, strong momentum {'up' if momentum_up else 'down'}"
                conviction = Conviction.MEDIUM if momentum_strong else Conviction.LOW
            else:
                # True coin flip
                direction = "UP" if distance_pct > 0 else "DOWN"
                win_prob = 0.52
                reason = f"Ranging, slight {'up' if distance_pct > 0 else 'down'} bias"
                conviction = Conviction.LOW

    # ==========================================================================
    # STEP 3: Calculate edge
    # ==========================================================================
    if direction == "UP":
        edge = win_prob - market_odds_up
    else:
        edge = win_prob - market_odds_down

    # Caps
    win_prob = min(0.92, max(0.50, win_prob))
    edge = min(0.15, max(-0.10, edge))

    # ==========================================================================
    # STEP 4: Position sizing
    # ==========================================================================
    if win_prob >= 0.80:
        size_mult = 0.80
    elif win_prob >= 0.70:
        size_mult = 0.60
    elif win_prob >= 0.60:
        size_mult = 0.40
    else:
        size_mult = 0.25

    # Time adjustments
    if is_final_2min:
        size_mult *= 0.7

    # Negative edge = reduce size
    if edge < 0:
        size_mult *= 0.5

    size_mult = max(0.20, min(0.80, size_mult))

    return SimpleSignal(
        direction=direction,
        conviction=conviction,
        win_probability=win_prob,
        edge=edge,
        reason=reason,
        size_multiplier=size_mult,
        distance_pct=distance_pct,
        trend=trend,
        momentum=momentum,
        time_remaining=time_remaining,
    )


def should_skip_market(time_remaining: float, spread: float) -> Tuple[bool, str]:
    """Skip only extreme cases."""
    if time_remaining < 20:
        return True, "< 20s remaining"
    if spread > 0.25:
        return True, f"Spread {spread:.0%} too wide"
    return False, ""


def log_simple_signal(
    asset: str,
    signal: SimpleSignal,
    target_price: float,
    current_price: float
):
    """Clean, informative logging."""

    arrow = "📈" if signal.direction == "UP" else "📉"

    stars = {
        Conviction.HIGH: "★★★",
        Conviction.MEDIUM: "★★☆",
        Conviction.LOW: "★☆☆"
    }[signal.conviction]

    # Win indicator
    if signal.win_probability >= 0.80:
        win_ind = "🟢"
    elif signal.win_probability >= 0.65:
        win_ind = "🟡"
    else:
        win_ind = "🟠"

    # Trend indicator
    trend_str = {
        Trend.STRONG_UP: "⬆️⬆️",
        Trend.UP: "⬆️",
        Trend.RANGING: "↔️",
        Trend.DOWN: "⬇️",
        Trend.STRONG_DOWN: "⬇️⬇️",
    }[signal.trend]

    mins = int(signal.time_remaining // 60)
    secs = int(signal.time_remaining % 60)

    logger.info(
        f"🎯 [{asset}] {arrow} {signal.direction} {stars} | "
        f"Win: {win_ind} {signal.win_probability:.0%} | "
        f"${current_price:,.2f} vs ${target_price:,.2f} ({signal.distance_pct:+.2%}) | "
        f"Trend: {trend_str} | Size: {signal.size_multiplier:.0%} | "
        f"{mins}m{secs}s | {signal.reason}"
    )


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 80)
    print("TEST SCENARIOS - Simple Signal Generator with Trend")
    print("=" * 80)

    # Test 1: Far above target (distance dominates)
    print("\n1. BTC 0.4% above target (distance should dominate):")
    signal = generate_simple_signal(
        asset="BTC",
        current_price=97900,
        target_price=97500,
        time_remaining=300,
        momentum=0.1,
        trend_15m=-0.2, trend_1h=-0.3, trend_4h=-0.5,  # Downtrend!
        market_odds_up=0.55, market_odds_down=0.48,
    )
    log_simple_signal("BTC", signal, 97500, 97900)
    print(f"   Expected: UP (distance > trend)")

    # Test 2: Near target with strong downtrend
    print("\n2. ETH near target with strong downtrend:")
    signal = generate_simple_signal(
        asset="ETH",
        current_price=2648,
        target_price=2650,
        time_remaining=400,
        momentum=-0.3,
        trend_15m=-0.3, trend_1h=-0.4, trend_4h=-0.6,  # Strong downtrend
        market_odds_up=0.50, market_odds_down=0.52,
    )
    log_simple_signal("ETH", signal, 2650, 2648)
    print(f"   Expected: DOWN (trend dominates near target)")

    # Test 3: At target with uptrend
    print("\n3. SOL at target with uptrend:")
    signal = generate_simple_signal(
        asset="SOL",
        current_price=185.01,
        target_price=185.00,
        time_remaining=300,
        momentum=0.2,
        trend_15m=0.25, trend_1h=0.20, trend_4h=0.15,  # Uptrend
        market_odds_up=0.50, market_odds_down=0.52,
    )
    log_simple_signal("SOL", signal, 185.00, 185.01)
    print(f"   Expected: UP (following uptrend)")

    # Test 4: Ranging market at target
    print("\n4. XRP ranging at target:")
    signal = generate_simple_signal(
        asset="XRP",
        current_price=2.501,
        target_price=2.500,
        time_remaining=200,
        momentum=0.0,
        trend_15m=0.02, trend_1h=-0.03, trend_4h=0.01,  # Ranging
        market_odds_up=0.50, market_odds_down=0.50,
    )
    log_simple_signal("XRP", signal, 2.500, 2.501)
    print(f"   Expected: LOW conviction (coin flip)")

    # Test 5: Conflict - below target but uptrend
    print("\n5. BTC below target but strong uptrend:")
    signal = generate_simple_signal(
        asset="BTC",
        current_price=97400,
        target_price=97500,
        time_remaining=300,
        momentum=0.4,
        trend_15m=0.3, trend_1h=0.35, trend_4h=0.4,  # Strong uptrend
        market_odds_up=0.48, market_odds_down=0.54,
    )
    log_simple_signal("BTC", signal, 97500, 97400)
    print(f"   Expected: UP (uptrend + momentum can overcome small distance)")
