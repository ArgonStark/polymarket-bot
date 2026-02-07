"""Monitoring and metrics collection."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    enabled: bool
    output_path: str
    rolling_window: int = 50


@dataclass
class TradeStats:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0


class MetricsCollector:
    """Collects and emits JSONL metrics for dashboards."""

    def __init__(self, config: MetricsConfig):
        self.config = config
        self.stats = TradeStats()
        self.recent_pnls: list[float] = []
        if self.config.enabled:
            Path(self.config.output_path).parent.mkdir(parents=True, exist_ok=True)

    def record_trade(self, asset: str, pnl: float, latency_ms: Optional[float] = None):
        if not self.config.enabled:
            return

        self.stats.trades += 1
        if pnl > 0:
            self.stats.wins += 1
        else:
            self.stats.losses += 1
        self.stats.pnl += pnl

        self.recent_pnls.append(pnl)
        if len(self.recent_pnls) > self.config.rolling_window:
            self.recent_pnls.pop(0)

        payload = {
            "ts": _now_iso(),
            "type": "trade",
            "asset": asset,
            "pnl": pnl,
            "win_rate": self.win_rate,
            "rolling_win_rate": self.rolling_win_rate,
        }
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms

        self._write(payload)

    def record_position(self, positions: int, exposure: float, equity: float):
        if not self.config.enabled:
            return
        payload = {
            "ts": _now_iso(),
            "type": "positions",
            "open_positions": positions,
            "exposure": exposure,
            "equity": equity,
        }
        self._write(payload)

    def record_latency(self, stage: str, latency_ms: float):
        if not self.config.enabled:
            return
        payload = {
            "ts": _now_iso(),
            "type": "latency",
            "stage": stage,
            "latency_ms": latency_ms,
        }
        self._write(payload)

    def record_strategy_performance(self, name: str, edge: float, confidence: float):
        if not self.config.enabled:
            return
        payload = {
            "ts": _now_iso(),
            "type": "strategy",
            "strategy": name,
            "edge": edge,
            "confidence": confidence,
        }
        self._write(payload)

    def record_order_event(
        self,
        event_type: str,
        asset: str,
        side: str,
        price: float = 0.0,
        size_usd: float = 0.0,
        order_id: Optional[str] = None,
        reason: Optional[str] = None,
        mode: str = "paper",
    ):
        """Record ORDER_SUBMIT, ORDER_RESULT, ORDER_FILL, ORDER_BLOCK events."""
        if not self.config.enabled:
            return
        payload = {
            "ts": _now_iso(),
            "type": event_type,
            "asset": asset,
            "side": side,
            "price": price,
            "size_usd": size_usd,
            "mode": mode,
        }
        if order_id:
            payload["order_id"] = order_id
        if reason:
            payload["reason"] = reason
        self._write(payload)

    @property
    def win_rate(self) -> float:
        return self.stats.wins / self.stats.trades if self.stats.trades > 0 else 0.0

    @property
    def rolling_win_rate(self) -> float:
        if not self.recent_pnls:
            return 0.0
        wins = sum(1 for p in self.recent_pnls if p > 0)
        return wins / len(self.recent_pnls)

    def _write(self, payload: dict):
        try:
            with open(self.config.output_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write metrics: {e}")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
