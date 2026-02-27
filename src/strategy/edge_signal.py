"""
Unified edge signal generator — replaces all 6 existing signal frameworks.

Three-signal pipeline: Distance + OFI + Regime
- Distance from strike (Chainlink price vs target) — PRIMARY
- Order Flow Imbalance (Binance aggTrade) — CONFIRMATION
- Regime detection (ATR + Efficiency Ratio) — GATE

Signal hierarchy:
  HIGH:   distance > high AND OFI confirms AND regime TRENDING → 70% Kelly
  MEDIUM: distance > med AND OFI not contradicting → 40% Kelly
          OR distance > half_med AND OFI confirms → 40% Kelly (combined)
  LOW:    OFI (>0.25) near strike with any distance → 25% Kelly
  SKIP:   OFI contradicts distance, OR regime CHOPPY, OR late + close to strike
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.models import Side
from src.strategy.regime import RegimeState, RegimeType

logger = logging.getLogger(__name__)


class Conviction(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    SKIP = "SKIP"


@dataclass
class EdgeSignal:
    """Output of the unified signal generator."""
    direction: Side             # UP or DOWN
    conviction: Conviction
    edge: float                 # Estimated edge (capped at max_edge)
    win_probability: float      # Conservative estimate
    size_fraction: float        # Kelly-derived fraction of bankroll
    reason: str                 # Human-readable explanation
    metadata: dict = field(default_factory=dict)

    @property
    def should_trade(self) -> bool:
        return self.conviction != Conviction.SKIP


@dataclass
class EdgeSignalConfig:
    """Configuration for edge signal generation."""
    enabled: bool = True

    # Distance thresholds (fraction, e.g. 0.003 = 0.3%)
    high_distance_threshold: float = 0.003
    medium_distance_threshold: float = 0.002

    # Complete-set edge (bid_UP + bid_DOWN < 1 - gap)
    complete_set_min_gap: float = 0.015  # 1.5%

    # Kelly fractions per conviction level
    kelly_high: float = 0.70
    kelly_medium: float = 0.40
    kelly_low: float = 0.25

    # OFI thresholds
    min_ofi_threshold: float = 0.15     # Below this = "neutral"
    strong_ofi_threshold: float = 0.25  # Above this = "strong" (lowered from 0.40 for BTC/ETH)
    ofi_confirmation_threshold: float = 0.10  # OFI must exceed this to confirm direction

    # Win probability estimates (conservative)
    win_prob_high: float = 0.58
    win_prob_medium: float = 0.55
    win_prob_low: float = 0.53

    # Maximum edge cap (no more fabricated 30% edges)
    max_edge: float = 0.12

    # Time-phase thresholds (fraction of total market duration)
    early_phase_fraction: float = 0.25   # First 25% of market
    late_phase_fraction: float = 0.80    # Last 20% of market
    late_skip_distance: float = 0.001    # Skip late trades if distance < 0.1%


def generate_edge_signal(
    *,
    asset: str,
    chainlink_price: float,
    target_price: float,
    best_bid_up: float,
    best_ask_up: float,
    best_bid_down: float,
    best_ask_down: float,
    ofi: float,
    ofi_acceleration: float,
    trade_rate: float,
    regime: RegimeState,
    time_remaining: float,
    total_duration: float,
    config: EdgeSignalConfig,
) -> EdgeSignal:
    """
    Generate a unified edge signal from the three-signal pipeline.

    Returns EdgeSignal with conviction level and sizing recommendation.
    """
    # --- Step 1: Distance from strike ---
    if target_price <= 0 or chainlink_price <= 0:
        return _skip("no_price_data")

    distance = (chainlink_price - target_price) / target_price
    abs_distance = abs(distance)
    price_above = distance > 0  # True = price above target → UP likely to win

    # --- Step 1b: Duration-aware threshold scaling ---
    # 5-min markets have less price movement → lower distance thresholds
    # Scale: sqrt(duration / 900) — e.g., 5min → 0.577x, 15min → 1.0x
    duration_scale = (total_duration / 900.0) ** 0.5 if total_duration > 0 else 1.0
    duration_scale = max(0.4, min(1.0, duration_scale))  # Clamp [0.4, 1.0]
    high_dist = config.high_distance_threshold * duration_scale
    med_dist = config.medium_distance_threshold * duration_scale

    # Direction threshold: much lower than conviction thresholds.
    # Any measurable distance (>0.03%) is enough to determine direction.
    # Conviction hierarchy decides if it's worth trading.
    min_direction_dist = med_dist * 0.25  # ~0.03% for 5m, ~0.05% for 15m

    # --- Step 2: Complete-set edge (structural) ---
    complete_set_gap = 1.0 - (best_bid_up + best_bid_down)
    has_complete_set_edge = complete_set_gap >= config.complete_set_min_gap

    # --- Step 3: Determine direction ---
    # Use a generous threshold for direction — hierarchy decides quality
    if abs_distance >= min_direction_dist:
        direction = Side.UP if price_above else Side.DOWN
    elif abs(ofi) >= config.ofi_confirmation_threshold:
        # Very near strike but OFI gives a directional hint
        direction = Side.UP if ofi > 0 else Side.DOWN
    else:
        return _skip(f"near_strike distance={abs_distance:.4f} ofi={ofi:.2f}")

    # --- Step 4: Check OFI confirmation/contradiction ---
    ofi_confirms = False
    ofi_contradicts = False
    if direction == Side.UP:
        ofi_confirms = ofi >= config.ofi_confirmation_threshold
        ofi_contradicts = ofi <= -config.min_ofi_threshold
    else:
        ofi_confirms = ofi <= -config.ofi_confirmation_threshold
        ofi_contradicts = ofi >= config.min_ofi_threshold

    # --- Step 5: Regime gate ---
    if not regime.should_trade:
        return _skip(f"regime_choppy er={regime.efficiency_ratio:.2f} atr={regime.atr_percentile:.0f}")

    # --- Step 6: Contradiction gate (must run BEFORE hierarchy) ---
    # If OFI strongly contradicts the distance-derived direction, SKIP
    # unless distance is overwhelming (>= high threshold)
    if ofi_contradicts and abs_distance < high_dist:
        return _skip(
            f"ofi_contradicts dist={abs_distance:.3%} ofi={ofi:+.2f} "
            f"dir={direction.value}"
        )

    # --- Step 7: Signal hierarchy ---
    conviction = Conviction.SKIP
    reason_parts = []

    # HIGH: distance > high_dist AND OFI confirms AND regime TRENDING
    if (abs_distance >= high_dist
            and ofi_confirms
            and regime.regime in (RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN)):
        conviction = Conviction.HIGH
        reason_parts.append(f"dist={abs_distance:.3%}")
        reason_parts.append(f"ofi={ofi:+.2f}")
        reason_parts.append(f"regime={regime.regime.value}")

    # MEDIUM: distance > med_dist AND (OFI neutral or confirms)
    elif (abs_distance >= med_dist
            and not ofi_contradicts
            and regime.should_trade):
        conviction = Conviction.MEDIUM
        reason_parts.append(f"dist={abs_distance:.3%}")
        if ofi_confirms:
            reason_parts.append(f"ofi_confirm={ofi:+.2f}")
        reason_parts.append(f"regime={regime.regime.value}")

    # MEDIUM (combined): moderate distance + OFI confirms together
    # Catches BTC/ETH where distance alone < med_dist but OFI confirms direction
    elif (abs_distance >= min_direction_dist
            and ofi_confirms
            and abs(ofi) >= config.strong_ofi_threshold
            and regime.should_trade):
        conviction = Conviction.MEDIUM
        reason_parts.append(f"dist+ofi={abs_distance:.3%}")
        reason_parts.append(f"ofi={ofi:+.2f}")
        reason_parts.append(f"regime={regime.regime.value}")

    # LOW: OFI-driven with any distance signal
    elif (abs(ofi) >= config.strong_ofi_threshold
            and not ofi_contradicts
            and regime.should_trade):
        conviction = Conviction.LOW
        reason_parts.append(f"ofi={ofi:+.2f}")
        reason_parts.append(f"dist={abs_distance:.3%}")

    # LOW (distance): moderate distance without OFI confirmation
    # For assets where OFI is weak but distance is meaningful
    elif (abs_distance >= med_dist * 0.6
            and not ofi_contradicts
            and regime.should_trade):
        conviction = Conviction.LOW
        reason_parts.append(f"dist_only={abs_distance:.3%}")
        reason_parts.append(f"ofi={ofi:+.2f}")

    else:
        return _skip(
            f"no_conviction dist={abs_distance:.3%} ofi={ofi:+.2f} "
            f"regime={regime.regime.value}"
        )

    # --- Step 8: Time-phase adjustment ---
    time_fraction = 1.0 - (time_remaining / total_duration) if total_duration > 0 else 0.5

    if time_fraction < config.early_phase_fraction:
        # Early: reduce slightly (less data, price may revert)
        time_mult = 0.8
        reason_parts.append("phase=early")
    elif time_fraction > config.late_phase_fraction:
        # Late: boost if distance confirmed, skip if close
        if abs_distance < config.late_skip_distance:
            return _skip(f"late_near_strike t_frac={time_fraction:.2f} dist={abs_distance:.3%}")
        time_mult = 1.2
        reason_parts.append("phase=late")
    else:
        time_mult = 1.0

    # --- Step 9: Compute edge and win probability ---
    # Base edge from distance + OFI contribution
    base_edge = abs_distance
    if ofi_confirms:
        # OFI contribution scales with OFI strength (not just a flat bonus)
        base_edge += abs(ofi) * 0.03
    if has_complete_set_edge:
        base_edge += complete_set_gap * 0.5  # Half the complete-set gap
        reason_parts.append(f"cs_gap={complete_set_gap:.3f}")

    edge = min(base_edge * time_mult, config.max_edge)

    # Floor check: skip if computed edge is trivially small
    # Scale floor by duration: 5-min → 0.23%, 15-min → 0.4%
    edge_floor = 0.004 * duration_scale
    edge_floor = max(0.002, edge_floor)  # Never below 0.2%
    if edge < edge_floor:
        return _skip(f"edge_too_low edge={edge:.4f} floor={edge_floor:.4f} dist={abs_distance:.3%} conviction={conviction.value}")

    # Win probability (conservative, conviction-dependent)
    if conviction == Conviction.HIGH:
        win_prob = config.win_prob_high
    elif conviction == Conviction.MEDIUM:
        win_prob = config.win_prob_medium
    else:
        win_prob = config.win_prob_low

    # --- Step 10: Kelly-based size fraction ---
    if conviction == Conviction.HIGH:
        kelly_frac = config.kelly_high
    elif conviction == Conviction.MEDIUM:
        kelly_frac = config.kelly_medium
    else:
        kelly_frac = config.kelly_low

    # Apply regime multiplier
    size_fraction = kelly_frac * regime.kelly_multiplier * time_mult

    # Compute Kelly: f* = (bp - q) / b where b = 1/price - 1
    # For limit orders at our price, approximate cost = 1 - win_prob
    market_price = best_ask_up if direction == Side.UP else best_ask_down
    if 0 < market_price < 1:
        b = (1.0 / market_price) - 1.0
        if b > 0:
            full_kelly = (b * win_prob - (1.0 - win_prob)) / b
            if full_kelly > 0:
                size_fraction = min(size_fraction, full_kelly)

    # Clamp
    size_fraction = max(0.02, min(0.20, size_fraction))

    reason = f"{conviction.value} {direction.value} | " + " ".join(reason_parts)

    metadata = {
        "distance_pct": distance,
        "abs_distance": abs_distance,
        "ofi": ofi,
        "ofi_acceleration": ofi_acceleration,
        "trade_rate": trade_rate,
        "regime": regime.regime.value,
        "efficiency_ratio": regime.efficiency_ratio,
        "atr_percentile": regime.atr_percentile,
        "complete_set_gap": complete_set_gap,
        "time_fraction": time_fraction,
        "time_mult": time_mult,
        "raw_edge": base_edge,
        "high_dist": high_dist,
        "med_dist": med_dist,
        "duration_scale": duration_scale,
    }

    return EdgeSignal(
        direction=direction,
        conviction=conviction,
        edge=edge,
        win_probability=win_prob,
        size_fraction=size_fraction,
        reason=reason,
        metadata=metadata,
    )


def _skip(reason: str) -> EdgeSignal:
    """Helper to create a SKIP signal."""
    return EdgeSignal(
        direction=Side.NONE,
        conviction=Conviction.SKIP,
        edge=0.0,
        win_probability=0.5,
        size_fraction=0.0,
        reason=f"SKIP | {reason}",
    )
