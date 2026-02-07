"""Backtest data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class OrderBookSnapshot:
    """Orderbook snapshot at a point in time."""

    ts: datetime
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    reference_price: Optional[float] = None


@dataclass
class TradeTick:
    """Executed trade from the tape."""

    ts: datetime
    price: float
    size: float
    side: str  # "BUY" or "SELL"


@dataclass
class BacktestOrder:
    """Backtest order state."""

    order_id: str
    side: str  # "BUY" or "SELL"
    price: float
    size: float
    is_maker: bool
    created_at: datetime
    remaining: float
    queue_remaining: float = 0.0


@dataclass
class BacktestFill:
    """Fill event for backtest order."""

    order_id: str
    ts: datetime
    price: float
    size: float
    fee: float
    side: str  # \"BUY\" or \"SELL\"


@dataclass
class BacktestPosition:
    """Position tracked during backtest."""

    side: str
    entry_price: float
    shares: float
    entry_time: datetime
    last_update: datetime

    def update(self, fill_price: float, fill_size: float, ts: datetime):
        total_cost = self.entry_price * self.shares
        total_cost += fill_price * fill_size
        self.shares += fill_size
        self.entry_price = total_cost / self.shares if self.shares > 0 else self.entry_price
        self.last_update = ts


@dataclass
class BacktestMetrics:
    """Summary metrics for a backtest run."""

    win_rate: float
    expectancy: float
    max_drawdown: float
    sharpe: float
    profit_factor: float
    total_trades: int
    pnl: float
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
