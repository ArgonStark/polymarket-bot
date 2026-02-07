"""Multi-strategy meta-layer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .strategies.base import StrategySignal
from .strategies.context import StrategyContext
from .strategies.mean_reversion import MeanReversionStrategy
from .strategies.momentum_spike import MomentumSpikeStrategy
from .strategies.trend import TrendContinuationStrategy


logger = logging.getLogger(__name__)


@dataclass
class StrategyWeights:
    trend: float = 0.4
    mean_reversion: float = 0.3
    momentum_spike: float = 0.3


class MultiStrategyMeta:
    """Combine multiple strategies with weighted voting."""

    def __init__(self, weights: StrategyWeights):
        self.weights = weights
        self.trend = TrendContinuationStrategy()
        self.mean_reversion = MeanReversionStrategy()
        self.momentum_spike = MomentumSpikeStrategy()

    def generate(self, ctx: StrategyContext) -> Optional[StrategySignal]:
        votes = []

        trend_sig = self.trend.generate(ctx)
        if trend_sig:
            votes.append((trend_sig, self.weights.trend))

        mr_sig = self.mean_reversion.generate(ctx)
        if mr_sig:
            votes.append((mr_sig, self.weights.mean_reversion))

        spike_sig = self.momentum_spike.generate(ctx)
        if spike_sig:
            votes.append((spike_sig, self.weights.momentum_spike))

        if not votes:
            return None

        score_yes = 0.0
        score_no = 0.0
        edge_sum = 0.0
        conf_sum = 0.0
        weight_sum = 0.0

        for sig, weight in votes:
            weight_sum += weight
            edge_sum += sig.edge * weight
            conf_sum += sig.confidence * weight
            if sig.side == "YES":
                score_yes += weight
            else:
                score_no += weight

        if weight_sum <= 0:
            return None

        side = "YES" if score_yes >= score_no else "NO"
        edge = edge_sum / weight_sum
        confidence = conf_sum / weight_sum

        if edge <= 0 or confidence <= 0:
            return None

        return StrategySignal(side=side, edge=edge, confidence=confidence)
