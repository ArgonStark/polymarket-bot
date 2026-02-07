"""Backtest metrics calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def win_rate(pnls: Iterable[float]) -> float:
    pnls = list(pnls)
    if not pnls:
        return 0.0
    wins = sum(1 for p in pnls if p > 0)
    return wins / len(pnls)


def expectancy(pnls: Iterable[float]) -> float:
    pnls = list(pnls)
    return sum(pnls) / len(pnls) if pnls else 0.0


def max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


def sharpe_ratio(pnls: Iterable[float], risk_free: float = 0.0) -> float:
    pnls = list(pnls)
    if len(pnls) < 2:
        return 0.0
    mean = sum(pnls) / len(pnls)
    var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
    std = var ** 0.5
    return ((mean - risk_free) / std) if std > 0 else 0.0


def profit_factor(pnls: Iterable[float]) -> float:
    gains = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p < 0)
    if losses == 0:
        return 0.0
    return gains / losses
