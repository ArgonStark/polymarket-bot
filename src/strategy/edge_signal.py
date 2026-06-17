"""
Unified edge signal generator — model-based EV pricing.

Pipeline: Probability model + OFI + Regime
- Win probability from driftless diffusion: p_up = Φ(distance / σ√τ)
  where distance = (chainlink - strike)/strike and σ√τ is the expected
  price dispersion over the REMAINING window (per-asset 15-min vol scaled
  by sqrt of time left). This is the standard binary-option probability.
- Edge = model probability − executable ask of that side. A big distance
  is NOT an edge if the market already prices it (ask 0.99); a modest
  distance IS an edge late in the window if the ask lags the model.
- Order Flow Imbalance (Binance aggTrade) — small drift adjustment + gate
- Regime detection (ATR + Efficiency Ratio) — gate + size multiplier

Conviction from edge size (EV after paying the ask):
  HIGH:   edge >= high_edge_ev AND OFI confirms → 70% Kelly
  MEDIUM: edge >= medium_edge_ev → 40% Kelly
  LOW:    edge >= min_edge_ev → 25% Kelly
  SKIP:   edge below floor, OFI contradicts, regime CHOPPY, late + near strike
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.models import Side
from src.strategy.regime import RegimeState, RegimeType

logger = logging.getLogger(__name__)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via erf (no scipy needed)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


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

    # EV thresholds: edge = model win probability − executable ask.
    # These are real expected values per dollar staked (before fill risk).
    min_edge_ev: float = 0.015     # Floor — below this, spread/noise eats it
    medium_edge_ev: float = 0.035
    high_edge_ev: float = 0.07

    # Max OFI-driven probability adjustment (order flow predicts short drift)
    ofi_prob_shift_max: float = 0.03

    # Max Binance-lead drift adjustment to the distance numerator (fraction).
    # Binance leads Chainlink by ~0.5-2s; the fresh Binance return predicts
    # the next Chainlink print. Clamped hard — it's a nudge, not a signal.
    max_drift_adj: float = 0.002

    # Taker fee at price 0.50 (worst case). Polymarket 15-min crypto markets
    # charge up to ~1.56% at 50c, tapering to ~0 at the extremes:
    #   fee(price) ≈ peak * min(price, 1-price) / 0.5
    # Set 0 to disable for fee-free markets.
    taker_fee_peak: float = 0.0156

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
    vol_15m: float = 0.005,
    drift: float = 0.0,
) -> EdgeSignal:
    """
    Generate a unified edge signal: model-based win probability vs the
    executable ask, gated by OFI and regime.

    Args:
        vol_15m: Per-asset price volatility over a 15-minute window
                 (fraction, e.g. 0.005 = 0.5%). Used to scale the
                 probability model to the remaining time.
        drift: Short-horizon Binance return (fraction) — the leading
               indicator for the next Chainlink print. Clamped to
               config.max_drift_adj before entering the model.

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

    # --- Step 7: Time-phase gates (before pricing) ---
    time_fraction = 1.0 - (time_remaining / total_duration) if total_duration > 0 else 0.5

    if time_fraction > config.late_phase_fraction and abs_distance < config.late_skip_distance:
        # Late + glued to the strike = coin flip with settlement risk
        return _skip(f"late_near_strike t_frac={time_fraction:.2f} dist={abs_distance:.3%}")

    # Size multiplier only — the probability model handles time itself
    if time_fraction < config.early_phase_fraction:
        time_mult = 0.8  # early: thin information, price may revert
    elif time_fraction > config.late_phase_fraction:
        time_mult = 1.2  # late: model is sharpest here
    else:
        time_mult = 1.0

    # --- Step 8: Model win probability (diffusion + clamped drift) ---
    # σ over the remaining window from per-asset 15-minute volatility:
    # σ(τ) = vol_15m * sqrt(τ / 900). p_up = Φ((distance + drift) / σ(τ)).
    # drift = fresh Binance lead return — where Chainlink is about to print.
    tau = max(1.0, time_remaining)
    sigma_tau = max(1e-5, vol_15m * math.sqrt(tau / 900.0))
    drift_adj = max(-config.max_drift_adj, min(config.max_drift_adj, drift))
    z = (distance + drift_adj) / sigma_tau
    p_up = _norm_cdf(z)
    win_prob = p_up if direction == Side.UP else 1.0 - p_up

    # Order flow predicts short-horizon drift: nudge probability, capped
    if ofi_confirms:
        win_prob += min(config.ofi_prob_shift_max, abs(ofi) * 0.06)
    win_prob = min(0.98, max(0.02, win_prob))

    # --- Step 9: Edge = model probability − executable price ---
    # This is the actual EV per dollar at the price we can trade NOW.
    # A huge distance with ask=0.99 is NO edge; a modest distance with a
    # lagging ask is a real one.
    exec_ask = best_ask_up if direction == Side.UP else best_ask_down
    if not (0 < exec_ask < 1):
        return _skip(f"no_executable_ask dir={direction.value} ask={exec_ask}")

    # Taker fee tapers from peak at 0.50 toward 0 at the extremes.
    # Subtracting it here makes the EV threshold a true after-fee number
    # (maker fills pay no fee — they just get a bonus we didn't count).
    taker_fee = config.taker_fee_peak * (min(exec_ask, 1.0 - exec_ask) / 0.5)

    edge = win_prob - exec_ask - taker_fee

    reason_parts = []
    if has_complete_set_edge:
        # Structural mispricing: both sides cheap. Credit half the gap.
        edge += complete_set_gap * 0.5
        reason_parts.append(f"cs_gap={complete_set_gap:.3f}")

    edge = min(edge, config.max_edge)

    if edge < config.min_edge_ev:
        return _skip(
            f"edge_too_low ev={edge:+.4f} p={win_prob:.3f} ask={exec_ask:.3f} "
            f"dist={abs_distance:.3%} z={z:+.2f}"
        )

    # --- Conviction from EV size + confirmations ---
    if (edge >= config.high_edge_ev
            and ofi_confirms
            and regime.regime in (RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN)):
        conviction = Conviction.HIGH
    elif edge >= config.medium_edge_ev:
        conviction = Conviction.MEDIUM
    else:
        conviction = Conviction.LOW

    reason_parts.insert(0, f"p={win_prob:.2f} ask={exec_ask:.2f} ev={edge:+.3f}")
    reason_parts.append(f"dist={abs_distance:.3%} z={z:+.2f}")
    if ofi_confirms:
        reason_parts.append(f"ofi={ofi:+.2f}")
    reason_parts.append(f"regime={regime.regime.value}")
    if time_mult != 1.0:
        reason_parts.append("phase=early" if time_mult < 1.0 else "phase=late")

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
        "z_score": z,
        "sigma_tau": sigma_tau,
        "vol_15m": vol_15m,
        "model_prob": win_prob,
        "exec_ask": exec_ask,
        "taker_fee": taker_fee,
        "drift_adj": drift_adj,
        "ofi": ofi,
        "ofi_acceleration": ofi_acceleration,
        "trade_rate": trade_rate,
        "regime": regime.regime.value,
        "efficiency_ratio": regime.efficiency_ratio,
        "atr_percentile": regime.atr_percentile,
        "complete_set_gap": complete_set_gap,
        "time_fraction": time_fraction,
        "time_mult": time_mult,
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
