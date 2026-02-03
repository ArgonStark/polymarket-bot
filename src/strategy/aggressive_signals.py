"""
Aggressive Trading Strategy for Polymarket 15-Minute Crypto Markets.

Philosophy: Trade EVERY market. Always have a position. Don't overthink.

The successful trader approach:
- 96 periods per day × 4 assets = 384 opportunities
- Pick a side on EACH one
- Win more than you lose
- Compound the edge

Key insight: You don't need 85% win rate.
At even odds (0.50), winning 55% of the time = profitable.
At 0.45 odds, winning 52% = profitable.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class Conviction(Enum):
    """How confident are we in this trade?"""
    HIGH = "high"        # Strong signal, full size
    MEDIUM = "medium"    # Decent signal, 70% size
    LOW = "low"          # Weak signal, 50% size


@dataclass
class AggressiveSignal:
    """Simple, clear trading signal."""
    direction: str  # "UP" or "DOWN"
    conviction: Conviction
    edge: float  # Estimated edge (can be negative - we still trade)
    reason: str

    # Position sizing
    size_multiplier: float  # 0.5 to 1.0

    # For logging
    price_vs_target: float  # Current price distance from target (%)
    momentum: float  # Recent price movement
    trend_alignment: bool  # Does momentum match our direction?


def generate_aggressive_signal(
    asset: str,
    current_price: float,
    target_price: float,
    time_remaining: float,
    binance_trend: float,  # -1 to +1, recent momentum
    volatility: float,  # 15-min volatility
    market_odds_up: float,  # Current market ask for UP
    market_odds_down: float,  # Current 1-bid for DOWN
) -> AggressiveSignal:
    """
    Generate an aggressive trading signal.

    ALWAYS returns a signal. We trade EVERY market.

    Args:
        asset: BTC, ETH, SOL, XRP
        current_price: Chainlink price
        target_price: Market target
        time_remaining: Seconds until settlement
        binance_trend: -1 (dumping) to +1 (pumping)
        volatility: Expected 15-min price movement (as decimal, e.g., 0.003 = 0.3%)
        market_odds_up: Best ask for UP token
        market_odds_down: 1 - best bid (cost to buy DOWN)

    Returns:
        AggressiveSignal with direction and conviction
    """

    # ==========================================================================
    # STEP 1: Where is price relative to target?
    # ==========================================================================

    distance_pct = (current_price - target_price) / target_price
    is_above = distance_pct > 0
    is_below = distance_pct < 0
    distance_abs = abs(distance_pct)

    # ==========================================================================
    # STEP 2: What is the momentum saying?
    # ==========================================================================

    momentum_bullish = binance_trend > 0.1
    momentum_bearish = binance_trend < -0.1
    momentum_strong = abs(binance_trend) > 0.3

    # ==========================================================================
    # STEP 3: Calculate time factor
    # ==========================================================================

    # How much can price move in remaining time?
    time_factor = (time_remaining / 900) ** 0.5  # sqrt decay
    expected_move = volatility * time_factor

    # How many "moves" away is the target?
    moves_to_target = distance_abs / expected_move if expected_move > 0 else 10

    # ==========================================================================
    # STEP 4: MAKE THE DECISION
    # ==========================================================================

    # Primary signal: Current position relative to target
    # Secondary signal: Momentum confirmation

    direction = None
    conviction = Conviction.MEDIUM
    reason = ""

    # ----------------------------------------------------------------------
    # CASE 1: Price clearly above target → Lean UP
    # ----------------------------------------------------------------------
    if is_above and distance_abs > 0.001:  # >0.1% above
        direction = "UP"

        if momentum_bullish:
            # Price above AND rising = HIGH conviction
            conviction = Conviction.HIGH
            reason = f"Price {distance_abs:.2%} above target, momentum confirms (+{binance_trend:.1%})"
        elif momentum_bearish and momentum_strong:
            # Price above but STRONGLY falling = still UP but LOW conviction
            conviction = Conviction.LOW
            reason = f"Price {distance_abs:.2%} above target, but momentum against ({binance_trend:.1%})"
        else:
            # Price above, neutral/weak momentum
            conviction = Conviction.MEDIUM
            reason = f"Price {distance_abs:.2%} above target, momentum neutral"

    # ----------------------------------------------------------------------
    # CASE 2: Price clearly below target → Lean DOWN
    # ----------------------------------------------------------------------
    elif is_below and distance_abs > 0.001:  # >0.1% below
        direction = "DOWN"

        if momentum_bearish:
            # Price below AND falling = HIGH conviction
            conviction = Conviction.HIGH
            reason = f"Price {distance_abs:.2%} below target, momentum confirms ({binance_trend:.1%})"
        elif momentum_bullish and momentum_strong:
            # Price below but STRONGLY rising = still DOWN but LOW conviction
            conviction = Conviction.LOW
            reason = f"Price {distance_abs:.2%} below target, but momentum against (+{binance_trend:.1%})"
        else:
            # Price below, neutral/weak momentum
            conviction = Conviction.MEDIUM
            reason = f"Price {distance_abs:.2%} below target, momentum neutral"

    # ----------------------------------------------------------------------
    # CASE 3: Price AT target (within 0.1%) → Use momentum to decide
    # ----------------------------------------------------------------------
    else:
        # This is the hardest case - price is right at target
        # Use momentum as the primary signal

        if momentum_bullish:
            direction = "UP"
            conviction = Conviction.MEDIUM if momentum_strong else Conviction.LOW
            reason = f"Price at target, momentum bullish (+{binance_trend:.1%})"
        elif momentum_bearish:
            direction = "DOWN"
            conviction = Conviction.MEDIUM if momentum_strong else Conviction.LOW
            reason = f"Price at target, momentum bearish ({binance_trend:.1%})"
        else:
            # No clear signal - pick based on micro-distance
            if distance_pct > 0:
                direction = "UP"
                reason = f"Price barely above target ({distance_pct:.3%}), slight UP bias"
            else:
                direction = "DOWN"
                reason = f"Price barely below target ({distance_pct:.3%}), slight DOWN bias"
            conviction = Conviction.LOW

    # ==========================================================================
    # STEP 5: Calculate edge
    # ==========================================================================

    # Estimate true probability
    # Simple model: further from target = higher probability of staying
    if direction == "UP":
        # Probability price stays above (or rises to) target
        if is_above:
            base_prob = 0.5 + min(0.4, moves_to_target * 0.1)
        else:
            base_prob = 0.5 - min(0.3, moves_to_target * 0.08)

        # Momentum adjustment
        if momentum_bullish:
            base_prob += 0.05
        elif momentum_bearish:
            base_prob -= 0.05

        true_prob = max(0.15, min(0.85, base_prob))
        market_prob = market_odds_up
        edge = true_prob - market_prob

    else:  # DOWN
        if is_below:
            base_prob = 0.5 + min(0.4, moves_to_target * 0.1)
        else:
            base_prob = 0.5 - min(0.3, moves_to_target * 0.08)

        if momentum_bearish:
            base_prob += 0.05
        elif momentum_bullish:
            base_prob -= 0.05

        true_prob = max(0.15, min(0.85, base_prob))
        market_prob = market_odds_down
        edge = true_prob - market_prob

    # ==========================================================================
    # STEP 6: Position sizing
    # ==========================================================================

    size_multiplier = {
        Conviction.HIGH: 1.0,
        Conviction.MEDIUM: 0.7,
        Conviction.LOW: 0.5,
    }[conviction]

    # Boost size if edge is particularly good
    if edge > 0.08:
        size_multiplier = min(1.0, size_multiplier * 1.2)

    # Reduce size if edge is negative (but still trade!)
    if edge < 0:
        size_multiplier *= 0.7

    # Reduce size in final minute (less time to be right)
    if time_remaining < 60:
        size_multiplier *= 0.5

    # ==========================================================================
    # STEP 7: Return signal
    # ==========================================================================

    return AggressiveSignal(
        direction=direction,
        conviction=conviction,
        edge=edge,
        reason=reason,
        size_multiplier=size_multiplier,
        price_vs_target=distance_pct,
        momentum=binance_trend,
        trend_alignment=(
            (direction == "UP" and momentum_bullish) or
            (direction == "DOWN" and momentum_bearish)
        ),
    )


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def should_skip_market(time_remaining: float, spread: float) -> Tuple[bool, str]:
    """
    Check if we should skip this market entirely.

    We're aggressive, but not stupid. Skip only when:
    - Market is about to settle (< 30 seconds)
    - Spread is insane (> 15%)

    Returns:
        (should_skip, reason)
    """
    if time_remaining < 30:
        return True, "Market settling in < 30s"

    if spread > 0.15:
        return True, f"Spread too wide: {spread:.1%}"

    return False, ""


def log_aggressive_signal(asset: str, signal: AggressiveSignal):
    """Log the signal in a clear, readable format."""

    conviction_emoji = {
        Conviction.HIGH: "🔥",
        Conviction.MEDIUM: "✓",
        Conviction.LOW: "?",
    }[signal.conviction]

    direction_emoji = "📈" if signal.direction == "UP" else "📉"
    alignment = "✓" if signal.trend_alignment else "✗"

    edge_str = f"+{signal.edge:.1%}" if signal.edge > 0 else f"{signal.edge:.1%}"

    logger.info(
        f"🎯 AGGRESSIVE [{asset}] {direction_emoji} {signal.direction} "
        f"{conviction_emoji} {signal.conviction.value.upper()} | "
        f"Edge: {edge_str} | "
        f"Size: {signal.size_multiplier:.0%} | "
        f"Trend: {alignment} | "
        f"{signal.reason}"
    )


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    # Test the signal generator
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Scenario 1: BTC clearly above target, momentum confirms
    signal = generate_aggressive_signal(
        asset="BTC",
        current_price=97650,
        target_price=97500,
        time_remaining=600,
        binance_trend=0.4,
        volatility=0.003,
        market_odds_up=0.55,
        market_odds_down=0.48,
    )
    log_aggressive_signal("BTC", signal)

    # Scenario 2: ETH below target, momentum against
    signal = generate_aggressive_signal(
        asset="ETH",
        current_price=2640,
        target_price=2650,
        time_remaining=400,
        binance_trend=0.3,  # Bullish but we're below target
        volatility=0.004,
        market_odds_up=0.52,
        market_odds_down=0.51,
    )
    log_aggressive_signal("ETH", signal)

    # Scenario 3: SOL right at target
    signal = generate_aggressive_signal(
        asset="SOL",
        current_price=185.02,
        target_price=185.00,
        time_remaining=300,
        binance_trend=-0.15,
        volatility=0.005,
        market_odds_up=0.50,
        market_odds_down=0.52,
    )
    log_aggressive_signal("SOL", signal)
