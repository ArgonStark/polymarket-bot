"""
Tests for the MetricsDashboard.

Validates:
1. Metrics file written with required keys
2. Rolling windows don't crash on small sample sizes
3. Edge decay computation
4. Veto counting within 5-minute window
5. Snapshot emission respects interval
"""

import json
import os
import tempfile
import time

import pytest

from src.monitoring.dashboard import MetricsDashboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dashboard(interval=0.0, rolling_window=50):
    """Dashboard with zero interval (emit every call) and temp output."""
    path = os.path.join(tempfile.gettempdir(), f"metrics_test_{os.getpid()}.json")
    if os.path.exists(path):
        os.remove(path)
    return MetricsDashboard(output_path=path, interval=interval, rolling_window=rolling_window), path


def _default_emit_args(**overrides):
    args = dict(
        bankroll=1000.0,
        peak_bankroll=1000.0,
        open_positions=0,
        total_exposure=0.0,
        realized_pnl_today=0.0,
        unrealized_pnl=0.0,
    )
    args.update(overrides)
    return args


# ---------------------------------------------------------------------------
# Tests: file output and required keys
# ---------------------------------------------------------------------------

class TestDashboardOutput:
    def test_writes_json_with_required_keys(self):
        dash, path = _make_dashboard()
        snap = dash.maybe_emit(**_default_emit_args())
        assert snap is not None

        with open(path) as f:
            data = json.load(f)

        required = {
            "timestamp", "bankroll", "peak_bankroll", "drawdown_pct",
            "open_positions", "total_exposure", "realized_pnl_today",
            "unrealized_pnl", "win_rate_rolling", "win_rate_session",
            "avg_edge_rolling", "edge_decay_open", "edge_decay_closed",
            "vetoes_5m_total", "vetoes_5m_by_reason",
        }
        assert required.issubset(set(data.keys()))

    def test_drawdown_calculation(self):
        dash, _ = _make_dashboard()
        snap = dash.maybe_emit(**_default_emit_args(
            bankroll=900.0,
            peak_bankroll=1000.0,
        ))
        assert abs(snap["drawdown_pct"] - 0.1) < 0.001

    def test_zero_peak_no_crash(self):
        dash, _ = _make_dashboard()
        snap = dash.maybe_emit(**_default_emit_args(
            bankroll=0.0,
            peak_bankroll=0.0,
        ))
        assert snap["drawdown_pct"] == 0.0


# ---------------------------------------------------------------------------
# Tests: rolling windows on small samples
# ---------------------------------------------------------------------------

class TestRollingWindows:
    def test_empty_trades(self):
        dash, _ = _make_dashboard()
        snap = dash.maybe_emit(**_default_emit_args())
        assert snap["win_rate_rolling"] == 0.0
        assert snap["avg_edge_rolling"] == 0.0
        assert snap["win_rate_session"] == 0.0

    def test_single_trade(self):
        dash, _ = _make_dashboard()
        dash.record_trade(pnl=5.0, edge_at_entry=0.05)
        snap = dash.maybe_emit(**_default_emit_args())
        assert snap["win_rate_rolling"] == 1.0
        assert snap["win_rate_session"] == 1.0
        assert abs(snap["avg_edge_rolling"] - 0.05) < 0.001

    def test_mixed_trades(self):
        dash, _ = _make_dashboard()
        dash.record_trade(pnl=5.0, edge_at_entry=0.10)
        dash.record_trade(pnl=-3.0, edge_at_entry=0.02)
        dash.record_trade(pnl=2.0, edge_at_entry=0.08)
        snap = dash.maybe_emit(**_default_emit_args())
        # 2 wins out of 3
        assert abs(snap["win_rate_rolling"] - 2 / 3) < 0.01
        assert abs(snap["avg_edge_rolling"] - (0.10 + 0.02 + 0.08) / 3) < 0.001

    def test_rolling_window_limit(self):
        dash, _ = _make_dashboard(rolling_window=5)
        for i in range(10):
            dash.record_trade(pnl=1.0 if i >= 5 else -1.0, edge_at_entry=0.05)
        snap = dash.maybe_emit(**_default_emit_args())
        # Only last 5 trades (all wins)
        assert snap["win_rate_rolling"] == 1.0


# ---------------------------------------------------------------------------
# Tests: edge decay
# ---------------------------------------------------------------------------

class TestEdgeDecay:
    def test_no_positions(self):
        dash, _ = _make_dashboard()
        snap = dash.maybe_emit(**_default_emit_args())
        assert snap["edge_decay_open"] == 0.0
        assert snap["edge_decay_closed"] == 0.0

    def test_open_edge_decay(self):
        dash, _ = _make_dashboard()
        dash.record_open_position("mkt1", predicted_edge=0.10)
        dash.update_open_mark("mkt1", current_mark_edge=0.05)
        snap = dash.maybe_emit(**_default_emit_args())
        # decay = 0.10 - 0.05 = 0.05
        assert abs(snap["edge_decay_open"] - 0.05) < 0.001

    def test_closed_edge_decay(self):
        dash, _ = _make_dashboard()
        dash.record_open_position("mkt1", predicted_edge=0.10)
        dash.close_position("mkt1", final_mark_edge=0.02)
        snap = dash.maybe_emit(**_default_emit_args())
        assert snap["edge_decay_open"] == 0.0  # no open positions
        assert abs(snap["edge_decay_closed"] - 0.08) < 0.001


# ---------------------------------------------------------------------------
# Tests: veto counting
# ---------------------------------------------------------------------------

class TestVetoCounting:
    def test_veto_count(self):
        dash, _ = _make_dashboard()
        dash.record_veto("chart_bearish")
        dash.record_veto("chart_bearish")
        dash.record_veto("risk_sizing_zero")
        snap = dash.maybe_emit(**_default_emit_args())
        assert snap["vetoes_5m_total"] == 3
        assert snap["vetoes_5m_by_reason"]["chart_bearish"] == 2
        assert snap["vetoes_5m_by_reason"]["risk_sizing_zero"] == 1


# ---------------------------------------------------------------------------
# Tests: interval gating
# ---------------------------------------------------------------------------

class TestIntervalGating:
    def test_respects_interval(self):
        dash, _ = _make_dashboard(interval=999.0)
        # First call always emits (last_emit starts at 0)
        snap1 = dash.maybe_emit(**_default_emit_args())
        assert snap1 is not None
        # Second call within interval returns None
        snap2 = dash.maybe_emit(**_default_emit_args())
        assert snap2 is None
