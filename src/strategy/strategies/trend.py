"""Trend continuation strategy."""

from __future__ import annotations

from typing import Optional

from .base import StrategySignal
from .context import StrategyContext


class TrendContinuationStrategy:
    """Trend continuation based on velocity and reference deviation."""

    def __init__(self, min_velocity: float = 0.002, min_confidence: float = 0.05):
        self.min_velocity = min_velocity
        self.min_confidence = min_confidence

    def generate(self, ctx: StrategyContext) -> Optional[StrategySignal]:
        if abs(ctx.price_velocity) < self.min_velocity:
            return None

        side = "YES" if ctx.price_velocity > 0 else "NO"
        edge = abs(ctx.price_velocity) * 0.5 + abs(ctx.orderbook_imbalance) * 0.2
        confidence = min(1.0, abs(ctx.price_velocity) + abs(ctx.orderbook_imbalance))

        if confidence < self.min_confidence:
            return None

        return StrategySignal(side=side, edge=edge, confidence=confidence)
