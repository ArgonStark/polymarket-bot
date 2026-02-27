"""
Tests for the CapitalScaler.

Validates:
1. Multiplier reduces sizing during drawdown
2. Multiplier never allows size beyond cap
3. Peak bonus applied when at peak with positive EV
4. Severe drawdown returns minimum-viable multiplier (0.25)
5. Normal (no drawdown) returns 1.0
6. Edge cases: zero peak, empty PnL history
"""

import pytest

from src.capital_scaling import CapitalScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scaler(**kwargs):
    defaults = dict(
        tier_severe_pct=0.10,
        tier_moderate_pct=0.05,
        multiplier_severe=0.25,
        multiplier_moderate=0.5,
        multiplier_peak_bonus=1.2,
    )
    defaults.update(kwargs)
    return CapitalScaler(**defaults)


# ---------------------------------------------------------------------------
# Tests: drawdown tiers
# ---------------------------------------------------------------------------

class TestDrawdownTiers:
    def test_severe_drawdown_returns_minimum(self):
        scaler = _make_scaler()
        mult, reason = scaler.compute_multiplier(
            peak_bankroll=1000.0,
            current_equity=890.0,  # 11% drawdown
            recent_pnls=[],
        )
        assert mult == 0.25
        assert "severe" in reason

    def test_moderate_drawdown_returns_half(self):
        scaler = _make_scaler()
        mult, reason = scaler.compute_multiplier(
            peak_bankroll=1000.0,
            current_equity=940.0,  # 6% drawdown
            recent_pnls=[],
        )
        assert mult == 0.5
        assert "moderate" in reason

    def test_no_drawdown_returns_one(self):
        scaler = _make_scaler()
        mult, reason = scaler.compute_multiplier(
            peak_bankroll=1000.0,
            current_equity=1000.0,
            recent_pnls=[],  # not enough for bonus
        )
        assert mult == 1.0
        assert reason == "normal"

    def test_small_drawdown_returns_one(self):
        scaler = _make_scaler()
        mult, reason = scaler.compute_multiplier(
            peak_bankroll=1000.0,
            current_equity=980.0,  # 2% drawdown - below moderate threshold
            recent_pnls=[],
        )
        assert mult == 1.0


# ---------------------------------------------------------------------------
# Tests: peak bonus
# ---------------------------------------------------------------------------

class TestPeakBonus:
    def test_peak_bonus_with_positive_ev(self):
        scaler = _make_scaler()
        # At peak with positive PnL history
        mult, reason = scaler.compute_multiplier(
            peak_bankroll=1000.0,
            current_equity=1000.0,
            recent_pnls=[1.0, 2.0, -0.5, 1.5, 3.0, 0.8, 1.2, -0.3, 2.1, 0.5],
        )
        assert mult == 1.2
        assert "peak_bonus" in reason

    def test_no_bonus_with_negative_ev(self):
        scaler = _make_scaler()
        mult, reason = scaler.compute_multiplier(
            peak_bankroll=1000.0,
            current_equity=1000.0,
            recent_pnls=[-1.0, -2.0, -0.5, -1.5, -3.0, -0.8, -1.2, -0.3, -2.1, -0.5],
        )
        assert mult == 1.0
        assert reason == "normal"

    def test_no_bonus_insufficient_history(self):
        scaler = _make_scaler()
        mult, reason = scaler.compute_multiplier(
            peak_bankroll=1000.0,
            current_equity=1000.0,
            recent_pnls=[1.0, 2.0],  # need 10+
        )
        assert mult == 1.0


# ---------------------------------------------------------------------------
# Tests: cap enforcement via apply()
# ---------------------------------------------------------------------------

class TestCapEnforcement:
    def test_apply_respects_max_position_pct(self):
        scaler = _make_scaler()
        adjusted, mult, reason = scaler.apply(
            size_usd=200.0,
            peak_bankroll=1000.0,
            current_equity=1000.0,
            recent_pnls=[1.0] * 15,  # positive EV → would get 1.2x bonus
            max_position_pct=0.10,
            bankroll=1000.0,
        )
        # 200 * 1.2 = 240, but cap = 1000 * 0.10 = 100
        assert adjusted <= 100.0

    def test_apply_scales_down_in_drawdown(self):
        scaler = _make_scaler()
        adjusted, mult, reason = scaler.apply(
            size_usd=50.0,
            peak_bankroll=1000.0,
            current_equity=930.0,  # 7% drawdown → moderate
            recent_pnls=[],
            max_position_pct=0.10,
            bankroll=930.0,
        )
        # 50 * 0.5 = 25
        assert adjusted == 25.0
        assert mult == 0.5

    def test_apply_severe_returns_reduced(self):
        scaler = _make_scaler()
        adjusted, mult, reason = scaler.apply(
            size_usd=50.0,
            peak_bankroll=1000.0,
            current_equity=880.0,  # 12% drawdown → severe
            recent_pnls=[],
            max_position_pct=0.10,
            bankroll=880.0,
        )
        # 50 * 0.25 = 12.50
        assert adjusted == 12.50
        assert mult == 0.25

    def test_normal_no_change(self):
        scaler = _make_scaler()
        adjusted, mult, reason = scaler.apply(
            size_usd=50.0,
            peak_bankroll=1000.0,
            current_equity=970.0,  # 3% drawdown — below moderate
            recent_pnls=[],
            max_position_pct=0.10,
            bankroll=970.0,
        )
        assert adjusted == 50.0
        assert mult == 1.0


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_peak(self):
        scaler = _make_scaler()
        mult, reason = scaler.compute_multiplier(
            peak_bankroll=0.0,
            current_equity=0.0,
            recent_pnls=[],
        )
        assert mult == 1.0
        assert reason == "no_peak"

    def test_peak_bonus_capped_by_position_limit(self):
        """If bonus would exceed max_position_pct, clamp the multiplier."""
        scaler = _make_scaler(multiplier_peak_bonus=2.0)
        mult, reason = scaler.compute_multiplier(
            peak_bankroll=1000.0,
            current_equity=1000.0,
            recent_pnls=[1.0] * 15,
            max_position_pct=0.10,
            bankroll=1000.0,
            current_size_usd=80.0,  # 80 * 2.0 = 160 > 100 cap
        )
        # Multiplier should be capped to 100/80 = 1.25
        assert mult == pytest.approx(1.25, abs=0.01)
        assert "peak_bonus" in reason
