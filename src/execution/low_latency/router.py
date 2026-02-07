"""Smart order routing based on live orderbook."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from ...models import Signal, OrderAction


logger = logging.getLogger(__name__)


@dataclass
class SmartRouterConfig:
    enabled: bool
    maker_edge_threshold: float
    taker_edge_threshold: float
    max_spread: float
    stale_order_seconds: float


class SmartOrderRouter:
    """Route orders as maker or taker based on edge and spread."""

    def __init__(self, config: SmartRouterConfig, executor, clob_feed):
        self.config = config
        self.executor = executor
        self.clob_feed = clob_feed

    async def execute(self, signal: Signal) -> Optional[object]:
        if not self.config.enabled:
            return await self.executor.execute_signal_async(signal)

        token_id = signal.market.up_token_id if signal.side.value == "UP" else signal.market.down_token_id
        orderbook = self.clob_feed.get_orderbook(token_id) if self.clob_feed else None

        if not orderbook or orderbook.best_bid is None or orderbook.best_ask is None:
            return await self.executor.execute_signal_async(signal)

        spread = orderbook.best_ask - orderbook.best_bid

        action = signal.recommended_action
        if signal.edge >= self.config.taker_edge_threshold:
            action = OrderAction.MARKET
        elif spread <= self.config.max_spread and signal.edge >= self.config.maker_edge_threshold:
            action = OrderAction.POST_ONLY
        else:
            action = OrderAction.LIMIT

        routed_signal = Signal(
            market=signal.market,
            side=signal.side,
            edge=signal.edge,
            true_prob=signal.true_prob,
            market_prob=signal.market_prob,
            recommended_action=action,
            recommended_price=signal.recommended_price,
            size_usd=signal.size_usd,
            size_shares=signal.size_shares,
            chainlink_price=signal.chainlink_price,
            time_remaining=signal.time_remaining,
            reasoning=signal.reasoning,
        )

        return await self.executor.execute_signal_async(routed_signal)
