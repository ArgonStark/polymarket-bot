"""Backtest execution simulation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from .models import BacktestOrder, BacktestFill, OrderBookSnapshot, TradeTick


logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    maker_fee_bps: float
    taker_fee_bps: float
    latency_ms: int
    slippage_bps: float


class ExecutionSimulator:
    """Simulates order fills with latency, fees, and queue position."""

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.open_orders: dict[str, BacktestOrder] = {}
        self._order_seq = 0
        self.last_orderbook: OrderBookSnapshot | None = None

    def update_orderbook(self, snapshot: OrderBookSnapshot):
        self.last_orderbook = snapshot

    def place_order(self, side: str, price: float, size: float, is_maker: bool, ts: datetime) -> BacktestOrder:
        self._order_seq += 1
        order_id = f"bt_{self._order_seq}"

        queue_remaining = 0.0
        if is_maker and self.last_orderbook:
            queue_remaining = self._estimate_queue(side, price, self.last_orderbook)

        order = BacktestOrder(
            order_id=order_id,
            side=side,
            price=price,
            size=size,
            is_maker=is_maker,
            created_at=ts,
            remaining=size,
            queue_remaining=queue_remaining,
        )
        self.open_orders[order_id] = order
        return order

    def on_trade(self, trade: TradeTick) -> list[BacktestFill]:
        fills: list[BacktestFill] = []
        for order in list(self.open_orders.values()):
            if not order.is_maker:
                continue

            if not self._trade_hits_order(order, trade):
                continue

            queue_before = order.queue_remaining
            if queue_before > 0:
                order.queue_remaining = max(0.0, queue_before - trade.size)
                if order.queue_remaining > 0:
                    continue

            fill_size = min(order.remaining, trade.size)
            if fill_size <= 0:
                continue

            fee = self._fee(order.price * fill_size, taker=False)
            fills.append(BacktestFill(order.order_id, trade.ts, order.price, fill_size, fee, order.side))
            order.remaining -= fill_size
            if order.remaining <= 0:
                self.open_orders.pop(order.order_id, None)

        return fills

    def fill_taker(self, side: str, size: float, ts: datetime) -> list[BacktestFill]:
        if not self.last_orderbook:
            return []

        levels = self.last_orderbook.asks if side == "BUY" else self.last_orderbook.bids
        if side == "SELL":
            levels = list(sorted(levels, key=lambda x: x[0], reverse=True))
        else:
            levels = list(sorted(levels, key=lambda x: x[0]))

        remaining = size
        fills: list[BacktestFill] = []
        for price, qty in levels:
            if remaining <= 0:
                break
            take = min(remaining, qty)
            fill_price = _apply_slippage(price, self.config.slippage_bps)
            fee = self._fee(fill_price * take, taker=True)
            fills.append(BacktestFill("taker", ts, fill_price, take, fee, side))
            remaining -= take

        return fills

    def _estimate_queue(self, side: str, price: float, snapshot: OrderBookSnapshot) -> float:
        if side == "BUY":
            levels = sorted(snapshot.bids, key=lambda x: x[0], reverse=True)
            ahead = sum(q for p, q in levels if p > price)
            same = sum(q for p, q in levels if p == price)
        else:
            levels = sorted(snapshot.asks, key=lambda x: x[0])
            ahead = sum(q for p, q in levels if p < price)
            same = sum(q for p, q in levels if p == price)
        return ahead + same * 0.5

    def _trade_hits_order(self, order: BacktestOrder, trade: TradeTick) -> bool:
        if order.side == "BUY" and trade.side == "SELL":
            return trade.price <= order.price
        if order.side == "SELL" and trade.side == "BUY":
            return trade.price >= order.price
        return False

    def _fee(self, notional: float, taker: bool) -> float:
        bps = self.config.taker_fee_bps if taker else self.config.maker_fee_bps
        return notional * (bps / 10000.0)


def _apply_slippage(price: float, bps: float) -> float:
    return price * (1 + bps / 10000.0) if price else price
