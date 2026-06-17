"""Dynamic risk and position sizing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional


logger = logging.getLogger(__name__)


@dataclass
class RiskSizingConfig:
    """Configuration for dynamic risk sizing."""

    enabled: bool
    target_volatility: float
    kelly_cap: float
    max_exposure_pct: float
    min_trade_usd: float
    kelly_fraction: float = 0.25  # fraction of full Kelly to bet (quarter Kelly)


class RiskSizer:
    """Volatility-adjusted sizing with Kelly fraction cap."""

    def __init__(self, config: RiskSizingConfig):
        self.config = config

    def size_position(
        self,
        bankroll: float,
        edge: float,
        prob: float,
        market_price: float,
        volatility: float,
        base_size: float,
    ) -> float:
        """Single source of truth for position size — fractional Kelly, reduce-only.

        Pipeline (each step can only SHRINK the bet, never grow it):
          full_kelly  = (b·p − q) / b      where b = 1/price − 1
          fraction    = kelly_fraction · full_kelly        (e.g. quarter Kelly)
          fraction   ·= min(1, target_vol / volatility)    (vol REDUCES only)
          fraction    = min(fraction, kelly_cap)            (hard ceiling)
          size        = bankroll · fraction
          size        = min(size, bankroll · max_exposure_pct)   (per-position cap)
          size        = max(size, min_trade_usd) if there is edge (floor to viable)

        Returns 0.0 when there is no edge (full Kelly ≤ 0). ``base_size`` is only
        used as a passthrough when dynamic sizing is disabled.
        """
        if not self.config.enabled:
            return base_size

        if bankroll <= 0 or not (0.0 < market_price < 1.0):
            logger.debug(
                "RISK_SIZING_ZERO reason=invalid_inputs bankroll=%.2f market_price=%.4f",
                bankroll, market_price
            )
            return 0.0

        # No edge → no bet. (edge is the caller's estimate; full Kelly is the
        # authoritative check below.)
        if edge <= 0:
            return 0.0

        full_kelly = self._kelly_fraction(prob, market_price)
        if full_kelly <= 0:
            logger.debug(
                "RISK_SIZING_ZERO reason=kelly_nonpositive prob=%.4f price=%.4f",
                prob, market_price,
            )
            return 0.0

        # Fractional Kelly (safety) — only ever scales DOWN.
        fraction = self.config.kelly_fraction * full_kelly

        # Volatility scaling: high vol shrinks the bet; low vol leaves it at
        # full fractional Kelly. Never amplifies above 1.0.
        vol_adjust = 1.0
        if volatility > 0 and self.config.target_volatility > 0:
            vol_adjust = min(1.0, self.config.target_volatility / volatility)
        fraction *= vol_adjust

        # Hard ceiling on the bet fraction.
        fraction = min(fraction, self.config.kelly_cap)

        size = bankroll * fraction

        # Per-position cap.
        size = min(size, bankroll * self.config.max_exposure_pct)

        # Floor to the minimum viable trade when there is genuine edge, but only
        # if the per-position cap leaves room (don't exceed it).
        if 0.0 < size < self.config.min_trade_usd:
            size = min(self.config.min_trade_usd, bankroll * self.config.max_exposure_pct)

        logger.debug(
            "RISK_SIZING prob=%.4f price=%.4f full_kelly=%.4f frac=%.4f "
            "vol_adj=%.2f size=%.2f bankroll=%.2f cap=%.2f",
            prob, market_price, full_kelly, fraction, vol_adjust, size,
            bankroll, bankroll * self.config.max_exposure_pct,
        )
        return max(0.0, size)

    @staticmethod
    def _kelly_fraction(prob: float, price: float) -> float:
        prob = min(max(prob, 1e-6), 1 - 1e-6)
        price = min(max(price, 1e-6), 1 - 1e-6)
        b = (1 / price) - 1
        q = 1 - prob
        fraction = (b * prob - q) / b if b > 0 else 0.0
        return max(0.0, fraction)
