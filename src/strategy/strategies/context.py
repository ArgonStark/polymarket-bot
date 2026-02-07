"""Strategy context container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class StrategyContext:
    """Input data for strategy evaluation."""

    asset: str
    mid_price: float
    reference_price: Optional[float]
    orderbook_imbalance: float
    price_velocity: float
    volatility: float
    time_remaining: float
