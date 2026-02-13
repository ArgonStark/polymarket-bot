"""
Tests for the KillSwitch component.

Validates:
1. Kill switch blocks order submission but allows settlement/state save
2. Drawdown trigger trips the switch
3. Daily loss trigger trips the switch
4. Consecutive losses trigger trips the switch
5. Manual file flag triggers the switch
6. Env var triggers the switch
7. Persistence: save/restore round-trip
8. Reset clears the switch
"""

import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.kill_switch import KillSwitch, KillSwitchState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_ks(**kwargs):
    """Create a KillSwitch with a temp flag path to avoid touching real files."""
    flag = os.path.join(tempfile.gettempdir(), f"ks_test_{os.getpid()}.flag")
    # Ensure flag doesn't exist
    if os.path.exists(flag):
        os.remove(flag)
    return KillSwitch(flag_path=flag, **kwargs)


# ---------------------------------------------------------------------------
# Tests: blocking behaviour
# ---------------------------------------------------------------------------

class TestKillSwitchBlocking:
    def test_inactive_allows_orders(self):
        ks = _fresh_ks()
        blocked, reason = ks.should_block_order()
        assert not blocked
        assert reason == ""

    def test_active_blocks_orders(self):
        ks = _fresh_ks()
        ks._trip("test reason")
        blocked, reason = ks.should_block_order()
        assert blocked
        assert "test reason" in reason

    def test_settlement_not_blocked(self):
        """Kill switch has no method that blocks settlement — verify by design."""
        ks = _fresh_ks()
        ks._trip("drawdown")
        # Settlement and state save are not gated by KillSwitch.
        # The bot calls ks.should_block_order() only in _execute_signal().
        # Verify that is_active() returns True (so callers can check),
        # but there is no "should_block_settlement" method.
        assert ks.is_active()
        assert not hasattr(ks, "should_block_settlement")


# ---------------------------------------------------------------------------
# Tests: triggers
# ---------------------------------------------------------------------------

class TestKillSwitchTriggers:
    def test_drawdown_trigger(self):
        ks = _fresh_ks(dd_pct=0.10)
        # 12% drawdown → should trip
        result = ks.evaluate(
            peak_bankroll=1000.0,
            current_equity=880.0,
            daily_loss_hit=False,
            consecutive_losses=0,
        )
        assert result is True
        assert ks.active
        assert "drawdown" in ks.reason

    def test_drawdown_below_threshold_no_trip(self):
        ks = _fresh_ks(dd_pct=0.10)
        # 5% drawdown → should not trip
        result = ks.evaluate(
            peak_bankroll=1000.0,
            current_equity=950.0,
            daily_loss_hit=False,
            consecutive_losses=0,
        )
        assert result is False
        assert not ks.active

    def test_daily_loss_trigger(self):
        ks = _fresh_ks()
        result = ks.evaluate(
            peak_bankroll=1000.0,
            current_equity=950.0,
            daily_loss_hit=True,
            consecutive_losses=0,
        )
        assert result is True
        assert "daily loss" in ks.reason

    def test_consecutive_losses_trigger(self):
        ks = _fresh_ks(consec_losses=5)
        result = ks.evaluate(
            peak_bankroll=1000.0,
            current_equity=950.0,
            daily_loss_hit=False,
            consecutive_losses=5,
        )
        assert result is True
        assert "consecutive losses" in ks.reason

    def test_consecutive_losses_below_threshold(self):
        ks = _fresh_ks(consec_losses=5)
        result = ks.evaluate(
            peak_bankroll=1000.0,
            current_equity=950.0,
            daily_loss_hit=False,
            consecutive_losses=4,
        )
        assert result is False

    def test_file_flag_trigger(self):
        ks = _fresh_ks()
        # Create the flag file
        with open(ks.flag_path, "w") as f:
            f.write("kill")
        try:
            result = ks.evaluate(
                peak_bankroll=1000.0,
                current_equity=1000.0,
                daily_loss_hit=False,
                consecutive_losses=0,
            )
            assert result is True
            assert "file flag" in ks.reason
        finally:
            os.remove(ks.flag_path)

    def test_env_var_trigger(self):
        ks = _fresh_ks()
        with patch.dict(os.environ, {"KILL_SWITCH": "1"}):
            result = ks.evaluate(
                peak_bankroll=1000.0,
                current_equity=1000.0,
                daily_loss_hit=False,
                consecutive_losses=0,
            )
        assert result is True
        assert "env var" in ks.reason

    def test_already_active_stays_active(self):
        ks = _fresh_ks()
        ks._trip("first trip")
        # Second evaluate should return True without changing reason
        result = ks.evaluate(
            peak_bankroll=1000.0,
            current_equity=1000.0,
            daily_loss_hit=False,
            consecutive_losses=0,
        )
        assert result is True
        assert ks.reason == "first trip"


# ---------------------------------------------------------------------------
# Tests: persistence
# ---------------------------------------------------------------------------

class TestKillSwitchPersistence:
    def test_round_trip_active(self):
        ks = _fresh_ks()
        ks._trip("drawdown 15%")
        data = ks.to_dict()

        ks2 = KillSwitch.from_dict(data, flag_path=ks.flag_path)
        assert ks2.active is True
        assert ks2.reason == "drawdown 15%"
        assert ks2.triggered_at is not None

    def test_round_trip_inactive(self):
        ks = _fresh_ks()
        data = ks.to_dict()

        ks2 = KillSwitch.from_dict(data, flag_path=ks.flag_path)
        assert ks2.active is False
        assert ks2.reason == ""

    def test_reset(self):
        ks = _fresh_ks()
        ks._trip("test")
        assert ks.active
        ks.reset("manual review")
        assert not ks.active
        assert ks.reason == ""
        assert ks.triggered_at is None


# ---------------------------------------------------------------------------
# Tests: zero-peak edge case
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_peak_no_crash(self):
        ks = _fresh_ks()
        result = ks.evaluate(
            peak_bankroll=0.0,
            current_equity=0.0,
            daily_loss_hit=False,
            consecutive_losses=0,
        )
        assert result is False
