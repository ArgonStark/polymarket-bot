"""Mean reversion strategy."""

from __future__ import annotations

from typing import Optional

from .base import StrategySignal
from .context import StrategyContext


class MeanReversionStrategy:
    """Mean reversion against reference price deviation."""

    def __init__(self, min_distance: float = 0.003, min_confidence: float = 0.05):
        self.min_distance = min_distance
        self.min_confidence = min_confidence

    def generate(self, ctx: StrategyContext) -> Optional[StrategySignal]:
        if ctx.reference_price is None or ctx.reference_price <= 0:
            return None

        deviation = (ctx.mid_price - ctx.reference_price) / ctx.reference_price
        if abs(deviation) < self.min_distance:
            return None

        side = "NO" if deviation > 0 else "YES"
        edge = abs(deviation) * 0.6 + max(0.0, -ctx.price_velocity) * 0.1
        confidence = min(1.0, abs(deviation) + (1 - abs(ctx.orderbook_imbalance)) * 0.2)

        if confidence < self.min_confidence:
            return None

        return StrategySignal(side=side, edge=edge, confidence=confidence)
