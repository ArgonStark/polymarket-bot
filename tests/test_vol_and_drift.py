"""Tests for time-aware volatility estimation, drift, and fee-aware EV."""
import math
import random
from datetime import datetime, timedelta, timezone

import pytest

from src.probability import estimate_vol_15m_from_ticks, estimate_volatility
from src.strategy.edge_signal import generate_edge_signal, EdgeSignalConfig
from src.strategy.regime import RegimeState, RegimeType


def _ticks(n, dt_seconds, sigma_per_sec, start_price=100_000.0, seed=7):
    """Random-walk ticks with known per-second volatility."""
    rng = random.Random(seed)
    t = datetime(2026, 6, 12, tzinfo=timezone.utc)
    p = start_price
    out = [(t, p)]
    for _ in range(n):
        t += timedelta(seconds=dt_seconds)
        p *= math.exp(rng.gauss(0.0, sigma_per_sec * math.sqrt(dt_seconds)))
        out.append((t, p))
    return out


class TestVolEstimator:
    def test_recovers_true_15m_vol_from_2s_ticks(self):
        # True per-second vol such that 15-min vol = 0.5%
        true_15m = 0.005
        sigma_sec = true_15m / math.sqrt(900)
        history = _ticks(300, 2.0, sigma_sec)  # 10 minutes of 2s ticks
        est = estimate_vol_15m_from_ticks(history, default_vol=0.005)
        assert est == pytest.approx(true_15m, rel=0.35)  # statistical tolerance

    def test_legacy_estimator_underestimates_badly(self):
        """Documents WHY the legacy estimator was replaced for tick data."""
        true_15m = 0.005
        sigma_sec = true_15m / math.sqrt(900)
        history = _ticks(300, 2.0, sigma_sec)
        legacy = estimate_volatility([p for _, p in history], window=20,
                                     default_vol=0.005)
        # Legacy treats 2s returns as 15m returns → ~21x too small
        assert legacy < true_15m / 5

    def test_insufficient_data_returns_default(self):
        assert estimate_vol_15m_from_ticks([], default_vol=0.007) == 0.007
        short = _ticks(3, 2.0, 0.0001)
        assert estimate_vol_15m_from_ticks(short, default_vol=0.007) == 0.007

    def test_clamped_to_band_around_default(self):
        # Absurdly volatile ticks → clamped to default * max_ratio
        wild = _ticks(300, 2.0, 0.01)  # enormous per-second vol
        est = estimate_vol_15m_from_ticks(wild, default_vol=0.005, max_ratio=4.0)
        assert est <= 0.005 * 4.0 + 1e-12

    def test_large_gaps_ignored(self):
        # 2s ticks with one 10-minute gap (reconnect) — gap must not distort
        history = _ticks(100, 2.0, 0.005 / math.sqrt(900))
        last_t, last_p = history[-1]
        history.append((last_t + timedelta(seconds=600), last_p * 1.02))
        history += [
            (last_t + timedelta(seconds=600 + 2 * i), last_p * 1.02)
            for i in range(1, 80)
        ]
        est = estimate_vol_15m_from_ticks(history, default_vol=0.005)
        # The 2% jump across the gap must not be counted as a 2s move
        assert est < 0.02


def _regime():
    return RegimeState(regime=RegimeType.RANGING, efficiency_ratio=0.5,
                       atr_percentile=50.0, should_trade=True, kelly_multiplier=0.75)


def _sig(**kw):
    args = dict(
        asset="BTC", chainlink_price=100_200.0, target_price=100_000.0,
        best_bid_up=0.83, best_ask_up=0.85, best_bid_down=0.15, best_ask_down=0.17,
        ofi=0.0, ofi_acceleration=0.0, trade_rate=10.0, regime=_regime(),
        time_remaining=60.0, total_duration=900.0, config=EdgeSignalConfig(),
        vol_15m=0.005,
    )
    args.update(kw)
    return generate_edge_signal(**args)


class TestFeeAndDrift:
    def test_fee_reduces_edge(self):
        no_fee = _sig(config=EdgeSignalConfig(taker_fee_peak=0.0))
        with_fee = _sig(config=EdgeSignalConfig(taker_fee_peak=0.0156))
        assert with_fee.edge < no_fee.edge

    def test_drift_clamped(self):
        # Outrageous drift input must move the model no more than max_drift_adj
        base = _sig(drift=0.0)
        boosted = _sig(drift=0.10)  # 10% — absurd, must clamp to 0.002
        capped = _sig(drift=0.002)
        assert boosted.win_probability == pytest.approx(capped.win_probability)
        assert boosted.win_probability >= base.win_probability

    def test_drift_against_reduces_prob(self):
        base = _sig(drift=0.0)
        against = _sig(drift=-0.002)
        assert against.win_probability < base.win_probability
