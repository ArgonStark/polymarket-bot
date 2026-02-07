"""EV-based trade policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .infer import InferenceOutput


@dataclass
class PolicyConfig:
    min_ev: float
    max_uncertainty: float
    max_spread: float
    min_depth: float
    max_exposure_pct: float
    ev_sizing_scale: float
    target_volatility: float


@dataclass
class PolicyDecision:
    should_trade: bool
    side: Optional[str]
    size_usd: float
    reason: str


class Policy:
    """Converts model outputs into trade decisions."""

    def __init__(self, config: PolicyConfig):
        self.config = config

    def decide(
        self,
        snapshot: Dict[str, Any],
        inference: InferenceOutput,
        bankroll: float,
        volatility: float,
    ) -> PolicyDecision:
        yes_bid = float(snapshot["yes_bid"])
        yes_ask = float(snapshot["yes_ask"])
        spread = yes_ask - yes_bid

        bid_depth = float(snapshot.get("orderbook_bid_depth", 0.0))
        ask_depth = float(snapshot.get("orderbook_ask_depth", 0.0))

        if spread > self.config.max_spread:
            return PolicyDecision(False, None, 0.0, "spread")
        if bid_depth < self.config.min_depth or ask_depth < self.config.min_depth:
            return PolicyDecision(False, None, 0.0, "depth")
        if inference.uncertainty > self.config.max_uncertainty:
            return PolicyDecision(False, None, 0.0, "uncertainty")

        if inference.ev_up <= 0 and inference.ev_down <= 0:
            return PolicyDecision(False, None, 0.0, "ev")

        side = "YES" if inference.ev_up >= inference.ev_down else "NO"
        ev = inference.ev_up if side == "YES" else inference.ev_down

        if ev < self.config.min_ev:
            return PolicyDecision(False, None, 0.0, "min_ev")

        vol_adj = 1.0
        if volatility > 0:
            vol_adj = min(1.0, self.config.target_volatility / volatility)

        size_pct = min(self.config.max_exposure_pct, ev / max(self.config.ev_sizing_scale, 1e-6))
        size_usd = bankroll * size_pct * vol_adj

        return PolicyDecision(True, side, size_usd, "ok")
