"""Feature engineering for probabilistic prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from ..models import MarketState


@dataclass
class PredictionFeatures:
    """Feature vector for probabilistic prediction."""

    orderbook_imbalance: float
    price_velocity: float
    ref_distance: float
    volatility: float
    spread: float
    mid_price: float

    def as_vector(self) -> list[float]:
        return [
            self.orderbook_imbalance,
            self.price_velocity,
            self.ref_distance,
            self.volatility,
            self.spread,
            self.mid_price,
        ]


class FeatureBuilder:
    """Builds prediction features from market state and price history."""

    def __init__(self, velocity_window: int = 20, vol_window: int = 60):
        self.velocity_window = max(2, velocity_window)
        self.vol_window = max(5, vol_window)

    def build(
        self,
        market: MarketState,
        price_history: Iterable[float],
        reference_price: Optional[float],
    ) -> PredictionFeatures:
        prices = [p for p in price_history if p is not None and p > 0]
        mid_price = market.mid_price or ((market.best_bid + market.best_ask) / 2 if market.best_bid and market.best_ask else 0.0)

        orderbook_imbalance = 0.0
        if market.bid_depth + market.ask_depth > 0:
            orderbook_imbalance = (market.bid_depth - market.ask_depth) / (market.bid_depth + market.ask_depth)

        price_velocity = 0.0
        if len(prices) >= 2:
            window = prices[-self.velocity_window :]
            if len(window) >= 2 and window[0] > 0:
                price_velocity = (window[-1] - window[0]) / window[0]

        ref_distance = 0.0
        if reference_price and reference_price > 0 and mid_price > 0:
            ref_distance = (mid_price - reference_price) / reference_price

        volatility = 0.0
        if len(prices) >= 2:
            window = prices[-self.vol_window :]
            if len(window) >= 2:
                mean = sum(window) / len(window)
                var = sum((p - mean) ** 2 for p in window) / (len(window) - 1)
                volatility = (var ** 0.5) / mean if mean > 0 else 0.0

        spread = 0.0
        if market.best_bid and market.best_ask:
            spread = max(0.0, market.best_ask - market.best_bid)

        return PredictionFeatures(
            orderbook_imbalance=orderbook_imbalance,
            price_velocity=price_velocity,
            ref_distance=ref_distance,
            volatility=volatility,
            spread=spread,
            mid_price=mid_price or 0.0,
        )
