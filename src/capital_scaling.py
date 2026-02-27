"""
Capital scaling — deterministic position-sizing multiplier.

Graduated drawdown tiers (never blocks — the kill switch handles halts):

  drawdown > severe  →  0.25  (minimum-viable sizing)
  drawdown > moderate →  0.50
  drawdown == 0      →  1.0  (or 1.2 if last 50 trades EV-positive)
  else               →  1.0

Severe/moderate thresholds default to fractions of MAX_DRAWDOWN_PCT so they
stay in sync with the kill switch and risk manager.

The multiplier is applied to the signal's ``size_usd`` BEFORE risk-manager
caps.  The final size never exceeds ``max_position_pct * bankroll`` or any
other configured cap.

All changes are logged with ``CAPITAL_SCALING``.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


def _default_severe() -> float:
    """Severe threshold: SCALE_SEVERE_DD_PCT env, else 75% of MAX_DRAWDOWN_PCT."""
    raw = os.getenv("SCALE_SEVERE_DD_PCT")
    if raw:
        return float(raw)
    max_dd = float(os.getenv("MAX_DRAWDOWN_PCT", "0.40"))
    return max_dd * 0.75  # 30% for default 40% max DD


def _default_moderate() -> float:
    """Moderate threshold: SCALE_MODERATE_DD_PCT env, else 50% of MAX_DRAWDOWN_PCT."""
    raw = os.getenv("SCALE_MODERATE_DD_PCT")
    if raw:
        return float(raw)
    max_dd = float(os.getenv("MAX_DRAWDOWN_PCT", "0.40"))
    return max_dd * 0.50  # 20% for default 40% max DD


@dataclass
class CapitalScaler:
    """Compute a sizing multiplier from drawdown + recent performance."""

    # Drawdown tiers (configurable via env, defaults derived from MAX_DRAWDOWN_PCT)
    tier_severe_pct: float = field(default_factory=_default_severe)
    tier_moderate_pct: float = field(default_factory=_default_moderate)
    multiplier_severe: float = field(
        default_factory=lambda: float(os.getenv("SCALE_SEVERE_MULT", "0.25"))
    )
    multiplier_moderate: float = field(
        default_factory=lambda: float(os.getenv("SCALE_MODERATE_MULT", "0.5"))
    )
    multiplier_peak_bonus: float = field(
        default_factory=lambda: float(os.getenv("SCALE_PEAK_BONUS", "1.2"))
    )

    def compute_multiplier(
        self,
        peak_bankroll: float,
        current_equity: float,
        recent_pnls: list[float],
        *,
        max_position_pct: float = 0.10,
        bankroll: float = 0.0,
        current_size_usd: float = 0.0,
    ) -> tuple[float, str]:
        """
        Return (multiplier, reason).

        The multiplier is applied to signal size *before* risk-manager caps.
        Caller must still enforce max_position_pct and other hard limits.
        """
        if peak_bankroll <= 0:
            return 1.0, "no_peak"

        dd = (peak_bankroll - current_equity) / peak_bankroll

        # Severe drawdown → minimum-viable sizing (kill switch handles full halt)
        if dd >= self.tier_severe_pct:
            # Throttle: only log once per 30s to avoid spam
            import time as _time
            _now = _time.monotonic()
            _last = getattr(self, '_last_severe_log', 0.0)
            if _now - _last >= 30.0:
                self._last_severe_log = _now
                logger.info(
                    "CAPITAL_SCALING multiplier=%.2f reason=severe_drawdown dd=%.4f threshold=%.4f",
                    self.multiplier_severe, dd, self.tier_severe_pct,
                )
            return self.multiplier_severe, f"severe_drawdown_{dd:.1%}"

        # Moderate drawdown → reduce size
        if dd >= self.tier_moderate_pct:
            logger.info(
                "CAPITAL_SCALING multiplier=%.2f reason=moderate_drawdown dd=%.4f threshold=%.4f",
                self.multiplier_moderate, dd, self.tier_moderate_pct,
            )
            return self.multiplier_moderate, f"moderate_drawdown_{dd:.1%}"

        # At or near peak with positive recent EV → slight bonus
        if dd <= 0.001 and len(recent_pnls) >= 10:
            avg_pnl = sum(recent_pnls) / len(recent_pnls)
            if avg_pnl > 0:
                mult = self.multiplier_peak_bonus
                # Cap check: scaled size must not exceed max_position_pct * bankroll
                if bankroll > 0 and current_size_usd > 0:
                    max_size = bankroll * max_position_pct
                    if current_size_usd * mult > max_size:
                        mult = max_size / current_size_usd
                        mult = max(1.0, mult)  # Never reduce below 1.0 in bonus path
                logger.info(
                    "CAPITAL_SCALING multiplier=%.2f reason=peak_bonus avg_pnl=%.4f trades=%d",
                    mult, avg_pnl, len(recent_pnls),
                )
                return mult, "peak_bonus"

        # Normal — no adjustment
        return 1.0, "normal"

    def apply(
        self,
        size_usd: float,
        peak_bankroll: float,
        current_equity: float,
        recent_pnls: list[float],
        max_position_pct: float,
        bankroll: float,
    ) -> tuple[float, float, str]:
        """
        Convenience: apply multiplier and enforce cap.

        Returns (adjusted_size_usd, multiplier, reason).
        """
        mult, reason = self.compute_multiplier(
            peak_bankroll=peak_bankroll,
            current_equity=current_equity,
            recent_pnls=recent_pnls,
            max_position_pct=max_position_pct,
            bankroll=bankroll,
            current_size_usd=size_usd,
        )

        adjusted = size_usd * mult

        # Hard cap: never exceed max_position_pct * bankroll
        cap = bankroll * max_position_pct
        if cap > 0 and adjusted > cap:
            adjusted = cap

        return adjusted, mult, reason
