"""Tests for the centralized fractional-Kelly position sizer (RiskSizer).

These lock in the post-fix behavior:
  - quarter Kelly by default
  - reduce-only (never scales UP from full Kelly or from base_size)
  - volatility can only shrink the bet, never amplify it
  - no edge / non-positive Kelly  -> 0
  - per-position cap = bankroll * max_exposure_pct
  - viable trades floored to min_trade_usd
"""

import pytest

from src.strategy.risk_sizing import RiskSizer, RiskSizingConfig


def _sizer(**overrides) -> RiskSizer:
    cfg = RiskSizingConfig(
        enabled=True,
        target_volatility=0.006,
        kelly_cap=0.20,
        max_exposure_pct=0.10,
        min_trade_usd=2.50,
        kelly_fraction=0.25,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return RiskSizer(cfg)


def test_quarter_kelly_basic():
    # prob=0.6, price=0.5 -> full Kelly = 0.2 ; quarter = 0.05 ; 1000*0.05 = 50
    size = _sizer().size_position(
        bankroll=1000, edge=0.1, prob=0.6, market_price=0.5,
        volatility=0.0, base_size=0.0,
    )
    assert size == pytest.approx(50.0)


def test_no_edge_returns_zero():
    s = _sizer()
    assert s.size_position(bankroll=1000, edge=0.0, prob=0.6,
                           market_price=0.5, volatility=0.0, base_size=0.0) == 0.0
    assert s.size_position(bankroll=1000, edge=-0.05, prob=0.6,
                           market_price=0.5, volatility=0.0, base_size=0.0) == 0.0


def test_nonpositive_kelly_returns_zero():
    # prob below price -> full Kelly <= 0 even if caller claims an edge
    size = _sizer().size_position(
        bankroll=1000, edge=0.1, prob=0.50, market_price=0.60,
        volatility=0.0, base_size=0.0,
    )
    assert size == 0.0


def test_capped_at_per_position_limit():
    # huge Kelly (prob=0.95) must be capped at bankroll * max_exposure_pct = 100
    size = _sizer().size_position(
        bankroll=1000, edge=0.4, prob=0.95, market_price=0.5,
        volatility=0.0, base_size=0.0,
    )
    assert size == pytest.approx(100.0)


def test_volatility_only_reduces():
    s = _sizer()
    low_vol = s.size_position(bankroll=1000, edge=0.1, prob=0.6,
                              market_price=0.5, volatility=0.001, base_size=0.0)
    high_vol = s.size_position(bankroll=1000, edge=0.1, prob=0.6,
                               market_price=0.5, volatility=0.012, base_size=0.0)
    # Low vol (below target) must NOT amplify above the quarter-Kelly size of 50
    assert low_vol == pytest.approx(50.0)
    # High vol (2x target) halves it
    assert high_vol == pytest.approx(25.0)
    assert high_vol < low_vol


def test_never_scales_up_to_base_size():
    # Regression: the old code did `return max(size, base_size)`, letting a
    # large requested base inflate the bet past Kelly. It must not.
    size = _sizer().size_position(
        bankroll=1000, edge=0.1, prob=0.6, market_price=0.5,
        volatility=0.0, base_size=99999.0,
    )
    assert size == pytest.approx(50.0)


def test_floored_to_min_trade_when_edge_present():
    # prob=0.52, price=0.5 -> full Kelly=0.04 ; quarter=0.01 ; 100*0.01=1.0
    # below min_trade_usd (2.50) but there is edge -> floor to 2.50
    size = _sizer().size_position(
        bankroll=100, edge=0.02, prob=0.52, market_price=0.5,
        volatility=0.0, base_size=0.0,
    )
    assert size == pytest.approx(2.50)


def test_disabled_passes_through_base_size():
    size = _sizer(enabled=False).size_position(
        bankroll=1000, edge=0.1, prob=0.6, market_price=0.5,
        volatility=0.0, base_size=42.0,
    )
    assert size == 42.0
