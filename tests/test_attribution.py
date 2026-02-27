"""
Tests for the trade attribution module.

Validates:
1. Trade record includes source, veto_reason, decision_context
2. Execution records are correctly categorised
3. Veto records capture reason
4. Summary shows breakdown counts (executed by source, blocked by reason)
5. determine_source() returns correct labels
"""

import pytest

from src.attribution import AttributionTracker, DecisionRecord, determine_source


# ---------------------------------------------------------------------------
# Tests: determine_source
# ---------------------------------------------------------------------------

class TestDetermineSource:
    def test_always_rules(self):
        assert determine_source() == "RULES"


# ---------------------------------------------------------------------------
# Tests: recording
# ---------------------------------------------------------------------------

class TestAttributionRecording:
    def test_execution_record(self):
        tracker = AttributionTracker()
        rec = tracker.record_execution(
            asset="BTC",
            market_id="cond_abc123",
            side="UP",
            source="ML_EV",
            decision_context={"edge": 0.05, "min_edge": 0.02},
        )
        assert rec.executed is True
        assert rec.source == "ML_EV"
        assert rec.decision_context["edge"] == 0.05
        assert rec.veto_reason == ""
        assert rec.asset == "BTC"

    def test_veto_record(self):
        tracker = AttributionTracker()
        rec = tracker.record_veto(
            asset="ETH",
            market_id="cond_def456",
            side="DOWN",
            source="RULES",
            veto_reason="chart_bearish_vs_up",
            decision_context={"time_remaining": 120},
        )
        assert rec.executed is False
        assert rec.veto_reason == "chart_bearish_vs_up"
        assert rec.source == "RULES"

    def test_to_dict(self):
        tracker = AttributionTracker()
        rec = tracker.record_execution(
            asset="BTC",
            market_id="cond_1",
            side="UP",
            source="RULES",
        )
        d = rec.to_dict()
        assert d["asset"] == "BTC"
        assert d["executed"] is True
        assert "timestamp" in d


# ---------------------------------------------------------------------------
# Tests: summary
# ---------------------------------------------------------------------------

class TestAttributionSummary:
    def test_empty_summary(self):
        tracker = AttributionTracker()
        summary = tracker.get_summary()
        assert summary["total_executed"] == 0
        assert summary["total_blocked"] == 0

    def test_mixed_summary(self):
        tracker = AttributionTracker()
        # 3 ML_EV executions, 2 RULES executions
        for _ in range(3):
            tracker.record_execution("BTC", "m1", "UP", "ML_EV")
        for _ in range(2):
            tracker.record_execution("ETH", "m2", "DOWN", "RULES")
        # 4 vetoes of various reasons
        tracker.record_veto("BTC", "m3", "UP", "ML_EV", "chart_bearish_vs_up")
        tracker.record_veto("BTC", "m4", "UP", "RULES", "risk_sizing_zero")
        tracker.record_veto("BTC", "m5", "UP", "RULES", "chart_bearish_vs_up")
        tracker.record_veto("BTC", "m6", "DOWN", "RULES", "kill_switch")

        summary = tracker.get_summary()
        assert summary["total_executed"] == 5
        assert summary["executed_by_source"]["ML_EV"] == 3
        assert summary["executed_by_source"]["RULES"] == 2
        assert summary["total_blocked"] == 4
        # Top veto reason
        assert summary["blocked_by_reason"]["chart_bearish_vs_up"] == 2

    def test_top5_truncation(self):
        tracker = AttributionTracker()
        # 7 different veto reasons — only top 5 should appear
        for i in range(7):
            for j in range(i + 1):
                tracker.record_veto("BTC", f"m{i}", "UP", "RULES", f"reason_{i}")
        summary = tracker.get_summary()
        assert len(summary["blocked_by_reason"]) == 5

    def test_recent_decisions(self):
        tracker = AttributionTracker()
        tracker.record_execution("BTC", "m1", "UP", "RULES")
        tracker.record_veto("ETH", "m2", "DOWN", "ML_EV", "low_confidence")
        recent = tracker.get_recent_decisions(n=5)
        assert len(recent) == 2
        assert recent[0]["executed"] is True
        assert recent[1]["executed"] is False


# ---------------------------------------------------------------------------
# Tests: history cap
# ---------------------------------------------------------------------------

class TestHistoryCap:
    def test_cap_enforced(self):
        tracker = AttributionTracker(max_history=10)
        for i in range(20):
            tracker.record_execution("BTC", f"m{i}", "UP", "RULES")
        assert len(tracker._decisions) == 10
        # Most recent should be m19
        assert tracker._decisions[-1].market_id == "m19"
