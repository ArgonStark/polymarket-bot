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
        if not self.config.enabled:
            return base_size

        if bankroll <= 0 or market_price <= 0:
            logger.debug(
                "RISK_SIZING_ZERO reason=invalid_inputs bankroll=%.2f market_price=%.4f",
                bankroll, market_price
            )
            return 0.0

        kelly_fraction = self._kelly_fraction(prob, market_price)
        kelly_fraction_raw = kelly_fraction
        kelly_fraction = max(0.0, min(self.config.kelly_cap, kelly_fraction))

        vol_adjust = 1.0
        if volatility > 0:
            vol_adjust = min(2.0, max(0.2, self.config.target_volatility / volatility))

        size = bankroll * kelly_fraction * vol_adjust
        size_before_clamp = size
        size = max(self.config.min_trade_usd, min(size, bankroll * self.config.max_exposure_pct))

        if edge <= 0:
            logger.info(
                "RISK_SIZING_DETAIL reason=negative_edge edge=%.4f prob=%.4f price=%.4f "
                "kelly_raw=%.4f kelly_capped=%.4f vol_adjust=%.2f size_pre_clamp=%.2f "
                "min_trade=%.2f max_exp_pct=%.2f bankroll=%.2f",
                edge, prob, market_price, kelly_fraction_raw, kelly_fraction,
                vol_adjust, size_before_clamp, self.config.min_trade_usd,
                self.config.max_exposure_pct, bankroll
            )
            return 0.0

        return max(size, base_size)

    @staticmethod
    def _kelly_fraction(prob: float, price: float) -> float:
        prob = min(max(prob, 1e-6), 1 - 1e-6)
        price = min(max(price, 1e-6), 1 - 1e-6)
        b = (1 / price) - 1
        q = 1 - prob
        fraction = (b * prob - q) / b if b > 0 else 0.0
        return max(0.0, fraction)
