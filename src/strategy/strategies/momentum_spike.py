"""Momentum spike detection strategy."""

from __future__ import annotations

from typing import Optional

from .base import StrategySignal
from .context import StrategyContext


class MomentumSpikeStrategy:
    """Detects short-term momentum spikes."""

    def __init__(self, spike_threshold: float = 0.006, min_confidence: float = 0.08):
        self.spike_threshold = spike_threshold
        self.min_confidence = min_confidence

    def generate(self, ctx: StrategyContext) -> Optional[StrategySignal]:
        if abs(ctx.price_velocity) < self.spike_threshold:
            return None

        side = "YES" if ctx.price_velocity > 0 else "NO"
        edge = abs(ctx.price_velocity) * 0.8 + abs(ctx.orderbook_imbalance) * 0.1
        confidence = min(1.0, abs(ctx.price_velocity) * 1.5)

        if confidence < self.min_confidence:
            return None

        return StrategySignal(side=side, edge=edge, confidence=confidence)
