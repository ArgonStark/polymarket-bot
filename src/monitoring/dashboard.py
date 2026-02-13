"""
Lightweight metrics dashboard — periodic structured log line + JSON snapshot.

No external dependencies (no Prometheus/Grafana).  Emits:
  * ``METRICS_SNAPSHOT`` log line every *interval* seconds
  * ``metrics.json`` file (atomic write) with the full snapshot

Metrics included:
  - bankroll, peak_bankroll, drawdown_pct
  - open positions count + total notional exposure
  - realized pnl today, unrealized pnl
  - win rate (rolling 50 trades and session)
  - average edge of executed trades (rolling 50)
  - edge decay (predicted edge at entry vs current mark)
  - number of vetoed trades by reason in last 5 minutes
"""

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EdgeRecord:
    """Tracks predicted edge at entry vs mark at close/current."""
    predicted_edge: float
    current_mark_edge: float  # for open: live, for closed: exit mark
    closed: bool = False


@dataclass
class VetoRecord:
    """Single veto event."""
    reason: str
    timestamp: float  # monotonic


class MetricsDashboard:
    """Collects and periodically emits a metrics snapshot."""

    def __init__(
        self,
        output_path: str = "",
        interval: float = 30.0,
        rolling_window: int = 50,
    ):
        if not output_path:
            project_root = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
            output_path = os.path.join(project_root, "metrics.json")

        self.output_path = output_path
        self.interval = interval
        self.rolling_window = rolling_window

        # Rolling trade history for win-rate and avg-edge
        self._trade_pnls: deque[float] = deque(maxlen=rolling_window)
        self._trade_edges: deque[float] = deque(maxlen=rolling_window)

        # Session totals
        self._session_trades: int = 0
        self._session_wins: int = 0

        # Edge decay tracking: last 20 closed + all open
        self._closed_edge_records: deque[EdgeRecord] = deque(maxlen=20)
        self._open_edge_records: dict[str, EdgeRecord] = {}  # market_key -> EdgeRecord

        # Veto tracking (last 5 min window)
        self._veto_window = 300.0  # 5 minutes
        self._vetoes: deque[VetoRecord] = deque()

        # Timing
        self._last_emit: float = 0.0

    # ── data ingestion ──────────────────────────────────────────

    def record_trade(self, pnl: float, edge_at_entry: float = 0.0):
        """Called after a trade closes."""
        self._trade_pnls.append(pnl)
        self._trade_edges.append(edge_at_entry)
        self._session_trades += 1
        if pnl > 0:
            self._session_wins += 1

    def record_open_position(self, market_key: str, predicted_edge: float):
        """Called when a new position is opened."""
        self._open_edge_records[market_key] = EdgeRecord(
            predicted_edge=predicted_edge,
            current_mark_edge=predicted_edge,
        )

    def update_open_mark(self, market_key: str, current_mark_edge: float):
        """Called each tick to update the live mark edge of an open position."""
        rec = self._open_edge_records.get(market_key)
        if rec:
            rec.current_mark_edge = current_mark_edge

    def close_position(self, market_key: str, final_mark_edge: float = 0.0):
        """Move position from open to closed edge-decay tracking."""
        rec = self._open_edge_records.pop(market_key, None)
        if rec:
            rec.current_mark_edge = final_mark_edge
            rec.closed = True
            self._closed_edge_records.append(rec)

    def record_veto(self, reason: str):
        """Record a blocked/vetoed trade decision."""
        self._vetoes.append(VetoRecord(reason=reason, timestamp=time.monotonic()))

    # ── snapshot emission ───────────────────────────────────────

    def maybe_emit(
        self,
        bankroll: float,
        peak_bankroll: float,
        open_positions: int,
        total_exposure: float,
        realized_pnl_today: float,
        unrealized_pnl: float,
    ) -> Optional[dict]:
        """
        Emit snapshot if *interval* seconds have elapsed since last emit.

        Returns the snapshot dict if emitted, else None.
        """
        now = time.monotonic()
        if now - self._last_emit < self.interval:
            return None
        self._last_emit = now

        snapshot = self._build_snapshot(
            bankroll=bankroll,
            peak_bankroll=peak_bankroll,
            open_positions=open_positions,
            total_exposure=total_exposure,
            realized_pnl_today=realized_pnl_today,
            unrealized_pnl=unrealized_pnl,
        )

        # Structured log line
        logger.info(
            "METRICS_SNAPSHOT bankroll=%.2f peak=%.2f drawdown=%.4f "
            "positions=%d exposure=%.2f realized_pnl=%.2f unrealized_pnl=%.2f "
            "win_rate_rolling=%.4f win_rate_session=%.4f avg_edge=%.4f "
            "edge_decay_open=%.4f edge_decay_closed=%.4f vetoes_5m=%d",
            snapshot["bankroll"],
            snapshot["peak_bankroll"],
            snapshot["drawdown_pct"],
            snapshot["open_positions"],
            snapshot["total_exposure"],
            snapshot["realized_pnl_today"],
            snapshot["unrealized_pnl"],
            snapshot["win_rate_rolling"],
            snapshot["win_rate_session"],
            snapshot["avg_edge_rolling"],
            snapshot["edge_decay_open"],
            snapshot["edge_decay_closed"],
            snapshot["vetoes_5m_total"],
        )

        # Atomic JSON write
        self._write_json(snapshot)
        return snapshot

    # ── internals ───────────────────────────────────────────────

    def _build_snapshot(
        self,
        bankroll: float,
        peak_bankroll: float,
        open_positions: int,
        total_exposure: float,
        realized_pnl_today: float,
        unrealized_pnl: float,
    ) -> dict:
        drawdown = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0.0

        # Rolling win rate
        win_rate_rolling = 0.0
        if self._trade_pnls:
            wins = sum(1 for p in self._trade_pnls if p > 0)
            win_rate_rolling = wins / len(self._trade_pnls)

        # Session win rate
        win_rate_session = 0.0
        if self._session_trades > 0:
            win_rate_session = self._session_wins / self._session_trades

        # Average edge (rolling)
        avg_edge = 0.0
        if self._trade_edges:
            avg_edge = sum(self._trade_edges) / len(self._trade_edges)

        # Edge decay — open positions
        edge_decay_open = self._compute_edge_decay(self._open_edge_records.values())

        # Edge decay — last 20 closed
        edge_decay_closed = self._compute_edge_decay(self._closed_edge_records)

        # Vetoes in last 5 minutes
        cutoff = time.monotonic() - self._veto_window
        # Prune old vetoes
        while self._vetoes and self._vetoes[0].timestamp < cutoff:
            self._vetoes.popleft()
        veto_counts: dict[str, int] = {}
        for v in self._vetoes:
            veto_counts[v.reason] = veto_counts.get(v.reason, 0) + 1

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bankroll": bankroll,
            "peak_bankroll": peak_bankroll,
            "drawdown_pct": round(drawdown, 6),
            "open_positions": open_positions,
            "total_exposure": round(total_exposure, 2),
            "realized_pnl_today": round(realized_pnl_today, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "win_rate_rolling": round(win_rate_rolling, 4),
            "win_rate_session": round(win_rate_session, 4),
            "avg_edge_rolling": round(avg_edge, 6),
            "edge_decay_open": round(edge_decay_open, 6),
            "edge_decay_closed": round(edge_decay_closed, 6),
            "vetoes_5m_total": sum(veto_counts.values()),
            "vetoes_5m_by_reason": veto_counts,
            "rolling_window": self.rolling_window,
            "session_trades": self._session_trades,
        }

    @staticmethod
    def _compute_edge_decay(records) -> float:
        """Mean (predicted - current) across records.  Positive = edge decayed."""
        items = list(records)
        if not items:
            return 0.0
        total = sum(r.predicted_edge - r.current_mark_edge for r in items)
        return total / len(items)

    def _write_json(self, snapshot: dict):
        try:
            tmp = self.output_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(snapshot, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.output_path)
        except Exception as e:
            logger.warning("METRICS_WRITE_FAILED: %s", e)
