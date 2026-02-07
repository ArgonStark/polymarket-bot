"""Backtesting engine for Polymarket strategies."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from .data_loader import merge_events
from .execution import ExecutionConfig, ExecutionSimulator
from .metrics import expectancy, max_drawdown, profit_factor, sharpe_ratio, win_rate
from .models import BacktestMetrics, BacktestPosition, OrderBookSnapshot, TradeTick
from ..strategy.meta import MultiStrategyMeta, StrategyWeights
from ..strategy.risk_sizing import RiskSizer, RiskSizingConfig
from ..strategy.strategies.context import StrategyContext


logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    initial_balance: float
    min_edge: float
    exit_edge: float
    exit_seconds_before_end: int
    market_duration_seconds: int


class BacktestEngine:
    """Replays historical orderbooks and trades to simulate fills."""

    def __init__(
        self,
        config: BacktestConfig,
        exec_config: ExecutionConfig,
        risk_config: RiskSizingConfig,
        strategy_weights: Optional[StrategyWeights] = None,
    ):
        self.config = config
        self.execution = ExecutionSimulator(exec_config)
        self.risk_sizer = RiskSizer(risk_config)
        self.strategy = MultiStrategyMeta(strategy_weights or StrategyWeights())
        self.balance = config.initial_balance
        self.position: Optional[BacktestPosition] = None
        self.trade_pnls: list[float] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.pending_entries: list[tuple[datetime, str, float, float, bool]] = []

    def run(self, orderbooks: list[OrderBookSnapshot], trades: list[TradeTick]) -> BacktestMetrics:
        if not orderbooks:
            return BacktestMetrics(0, 0, 0, 0, 0, 0, 0, [])

        start_ts = orderbooks[0].ts
        end_ts = start_ts + timedelta(seconds=self.config.market_duration_seconds)

        price_history: list[float] = []
        events = merge_events(orderbooks, trades)

        for ts, event_type, payload in events:
            if ts > end_ts:
                break

            self._drain_pending(ts)

            if event_type == "orderbook":
                snapshot = payload
                self.execution.update_orderbook(snapshot)
                mid = _mid_price(snapshot)
                if mid > 0:
                    price_history.append(mid)

                ctx = self._build_context(snapshot, price_history, end_ts)
                signal = self.strategy.generate(ctx)
                if signal:
                    self._maybe_trade(signal, snapshot, ts)

                self._maybe_exit(signal, snapshot, ts, end_ts)

            elif event_type == "trade":
                fills = self.execution.on_trade(payload)
                if fills:
                    self._apply_fills(fills)

            self._record_equity(ts)

        if self.position:
            self._close_position(orderbooks[-1].ts, _mid_price(orderbooks[-1]))

        equity_values = [v for _, v in self.equity_curve]
        metrics = BacktestMetrics(
            win_rate=win_rate(self.trade_pnls),
            expectancy=expectancy(self.trade_pnls),
            max_drawdown=max_drawdown(equity_values),
            sharpe=sharpe_ratio(self.trade_pnls),
            profit_factor=profit_factor(self.trade_pnls),
            total_trades=len(self.trade_pnls),
            pnl=sum(self.trade_pnls),
            equity_curve=self.equity_curve,
        )
        return metrics

    def _drain_pending(self, ts: datetime):
        ready = [p for p in self.pending_entries if p[0] <= ts]
        if not ready:
            return
        self.pending_entries = [p for p in self.pending_entries if p[0] > ts]

        for _, side, price, size, is_maker in ready:
            if is_maker:
                self.execution.place_order(side, price, size, True, ts)
            else:
                fills = self.execution.fill_taker(side, size, ts)
                if fills:
                    self._apply_fills(fills)

    def _build_context(
        self,
        snapshot: OrderBookSnapshot,
        price_history: list[float],
        end_ts: datetime,
    ) -> StrategyContext:
        mid = _mid_price(snapshot)
        orderbook_imbalance = _imbalance(snapshot)
        price_velocity = _velocity(price_history)
        volatility = _volatility(price_history)
        time_remaining = max(0.0, (end_ts - snapshot.ts).total_seconds())

        return StrategyContext(
            asset="BTC",
            mid_price=mid,
            reference_price=snapshot.reference_price,
            orderbook_imbalance=orderbook_imbalance,
            price_velocity=price_velocity,
            volatility=volatility,
            time_remaining=time_remaining,
        )

    def _maybe_trade(self, signal, snapshot: OrderBookSnapshot, ts: datetime):
        if signal.edge < self.config.min_edge:
            return

        if self.position:
            return

        price = _mid_price(snapshot)
        if price <= 0:
            return

        bankroll = self.balance
        size_usd = self.risk_sizer.size_position(
            bankroll=bankroll,
            edge=signal.edge,
            prob=0.5 + signal.edge,
            market_price=price,
            volatility=_volatility_from_snapshot(snapshot),
            base_size=bankroll * 0.05,
        )

        size_shares = size_usd / price if price > 0 else 0
        if size_shares <= 0:
            return

        is_maker = signal.edge < self.config.min_edge * 2
        side = "BUY" if signal.side == "YES" else "SELL"
        exec_at = ts + timedelta(milliseconds=self.execution.config.latency_ms)
        self.pending_entries.append((exec_at, side, price, size_shares, is_maker))

    def _maybe_exit(self, signal, snapshot: OrderBookSnapshot, ts: datetime, end_ts: datetime):
        if not self.position:
            return

        time_remaining = (end_ts - ts).total_seconds()
        if time_remaining <= self.config.exit_seconds_before_end:
            self._close_position(ts, _mid_price(snapshot))
            return

        if signal and signal.edge < self.config.exit_edge:
            self._close_position(ts, _mid_price(snapshot))

    def _apply_fills(self, fills):
        if not fills:
            return

        fee_total = sum(f.fee for f in fills)
        fill_price = _weighted_price(fills)
        fill_size = sum(f.size for f in fills)

        if not self.position:
            side = "YES" if fills[0].side == "BUY" else "NO"
            self.position = BacktestPosition(
                side=side,
                entry_price=fill_price,
                shares=fill_size,
                entry_time=fills[-1].ts,
                last_update=fills[-1].ts,
            )
            self.balance -= fill_price * fill_size + fee_total
        else:
            self.position.update(fill_price, fill_size, fills[-1].ts)
            self.balance -= fill_price * fill_size + fee_total

    def _close_position(self, ts: datetime, price: float):
        if not self.position or price <= 0:
            return
        pnl = (price - self.position.entry_price) * self.position.shares
        if self.position.side == "NO":
            pnl = -pnl
        self.balance += self.position.shares * price + pnl
        self.trade_pnls.append(pnl)
        self.position = None

    def _record_equity(self, ts: datetime):
        equity = self.balance
        if self.position:
            equity += self.position.shares * self.position.entry_price
        self.equity_curve.append((ts, equity))


def _mid_price(snapshot: OrderBookSnapshot) -> float:
    if snapshot.bids and snapshot.asks:
        return (snapshot.bids[0][0] + snapshot.asks[0][0]) / 2
    if snapshot.bids:
        return snapshot.bids[0][0]
    if snapshot.asks:
        return snapshot.asks[0][0]
    return 0.0


def _imbalance(snapshot: OrderBookSnapshot) -> float:
    bid_depth = sum(q for _, q in snapshot.bids)
    ask_depth = sum(q for _, q in snapshot.asks)
    if bid_depth + ask_depth == 0:
        return 0.0
    return (bid_depth - ask_depth) / (bid_depth + ask_depth)


def _velocity(prices: list[float]) -> float:
    if len(prices) < 2:
        return 0.0
    return (prices[-1] - prices[-2]) / prices[-2] if prices[-2] > 0 else 0.0


def _volatility(prices: list[float]) -> float:
    if len(prices) < 5:
        return 0.0
    window = prices[-20:]
    mean = sum(window) / len(window)
    var = sum((p - mean) ** 2 for p in window) / max(1, len(window) - 1)
    return (var ** 0.5) / mean if mean > 0 else 0.0


def _volatility_from_snapshot(snapshot: OrderBookSnapshot) -> float:
    mid = _mid_price(snapshot)
    if mid <= 0:
        return 0.0
    spread = 0.0
    if snapshot.bids and snapshot.asks:
        spread = snapshot.asks[0][0] - snapshot.bids[0][0]
    return spread / mid if mid > 0 else 0.0


def _weighted_price(fills) -> float:
    total = sum(f.size * f.price for f in fills)
    size = sum(f.size for f in fills)
    return total / size if size > 0 else 0.0
