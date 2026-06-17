"""Tests for the model-based edge signal: p = Φ(distance/σ√τ), edge = p − ask."""
import pytest

from src.models import Side
from src.strategy.edge_signal import (
    generate_edge_signal, EdgeSignalConfig, Conviction, _norm_cdf,
)
from src.strategy.regime import RegimeState, RegimeType


def _regime(rtype=RegimeType.RANGING, trade=True, mult=1.0):
    return RegimeState(
        regime=rtype, efficiency_ratio=0.5, atr_percentile=50.0,
        should_trade=trade, kelly_multiplier=mult,
    )


def _signal(*, price=100_000.0, target=100_000.0, bid_up=0.50, ask_up=0.52,
            ofi=0.0, time_remaining=450.0, duration=900.0, vol=0.005,
            regime=None, cfg=None):
    return generate_edge_signal(
        asset="BTC", chainlink_price=price, target_price=target,
        best_bid_up=bid_up, best_ask_up=ask_up,
        best_bid_down=1 - ask_up, best_ask_down=1 - bid_up,
        ofi=ofi, ofi_acceleration=0.0, trade_rate=10.0,
        regime=regime or _regime(), time_remaining=time_remaining,
        total_duration=duration, config=cfg or EdgeSignalConfig(),
        vol_15m=vol,
    )


class TestProbabilityModel:
    def test_norm_cdf_basics(self):
        assert _norm_cdf(0.0) == pytest.approx(0.5)
        assert _norm_cdf(10.0) == pytest.approx(1.0, abs=1e-9)
        assert _norm_cdf(-10.0) == pytest.approx(0.0, abs=1e-9)

    def test_no_edge_when_market_prices_it(self):
        """Big distance + expensive ask = old code's false 'edge'. Now SKIP."""
        # 0.75% above target, UP ask already 0.99 → p≈0.98 < 0.99 → no edge
        sig = _signal(price=100_750, ask_up=0.99, bid_up=0.97, time_remaining=300)
        assert not sig.should_trade
        assert "edge_too_low" in sig.reason

    def test_edge_when_ask_lags_model(self):
        """Late window, clear lead, cheap ask → the trade the old code missed."""
        # 0.2% above with 60s left: σ=0.005*sqrt(60/900)=0.00129, z≈1.55, p≈0.94
        sig = _signal(price=100_200, ask_up=0.85, bid_up=0.83, time_remaining=60)
        assert sig.should_trade
        assert sig.direction == Side.UP
        assert sig.win_probability > 0.90
        assert sig.edge >= 0.035  # at least MEDIUM EV
        assert sig.conviction in (Conviction.MEDIUM, Conviction.HIGH)

    def test_down_side_symmetric(self):
        sig = _signal(price=99_800, ask_up=0.17, bid_up=0.15, time_remaining=60)
        # 0.2% below → DOWN; DOWN ask = 1-bid_up = 0.85
        assert sig.should_trade
        assert sig.direction == Side.DOWN
        assert sig.win_probability > 0.90

    def test_same_distance_higher_prob_with_less_time(self):
        early = _signal(price=100_200, ask_up=0.55, bid_up=0.53, time_remaining=800)
        late = _signal(price=100_200, ask_up=0.55, bid_up=0.53, time_remaining=60)
        assert late.win_probability > early.win_probability

    def test_near_strike_prob_is_half(self):
        sig = _signal(price=100_000.0, ask_up=0.52, time_remaining=450, ofi=0.3)
        # p ≈ 0.5 (+ small OFI shift); ask 0.52 → edge below floor → skip
        assert not sig.should_trade

    def test_ofi_contradiction_still_gates(self):
        sig = _signal(price=100_150, ask_up=0.55, time_remaining=300, ofi=-0.5)
        assert not sig.should_trade
        assert "ofi_contradicts" in sig.reason

    def test_regime_choppy_still_gates(self):
        sig = _signal(price=100_200, ask_up=0.60, time_remaining=120,
                      regime=_regime(RegimeType.CHOPPY, trade=False, mult=0.0))
        assert not sig.should_trade
        assert "regime_choppy" in sig.reason

    def test_kelly_size_positive_and_capped(self):
        sig = _signal(price=100_200, ask_up=0.85, bid_up=0.83, time_remaining=60)
        assert 0.02 <= sig.size_fraction <= 0.20
