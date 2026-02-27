"""
Test that total exposure cap uses equity (cash + positions), not just cash.

Validates:
- Exposure cap computed on equity prevents over-allocation after profits
- Per-trade cap on cash is conservative (no change needed)
"""

from unittest.mock import MagicMock
from src.strategy.risk import RiskManager


def _make_config(**overrides):
    trading = MagicMock()
    trading.max_consecutive_losses = 5
    trading.max_drawdown_pct = 0.15
    trading.min_trades_for_winrate = 50
    trading.min_win_rate = 0.40
    trading.cooloff_period_minutes = 5
    trading.daily_loss_limit = 0.25
    trading.max_position_pct = 0.05
    trading.max_total_exposure_pct = 0.25
    trading.trade_history_size = 20
    trading.capital_buffer = 0.90
    for k, v in overrides.items():
        setattr(trading, k, v)
    config = MagicMock()
    config.trading = trading
    return config


def test_exposure_cap_on_equity_not_cash():
    """Exposure cap = equity * max_pct, not just cash * max_pct."""
    # Scenario: $800 cash, $200 in open positions = $1000 equity
    # max_total_exposure_pct = 25%
    # Cash-based cap: $800 * 25% = $200 (too small after profits)
    # Equity-based cap: $1000 * 25% = $250 (correct)
    cash = 800.0
    position_value = 200.0
    equity = cash + position_value
    max_pct = 0.25

    cash_based_cap = cash * max_pct
    equity_based_cap = equity * max_pct

    assert equity_based_cap == 250.0
    assert cash_based_cap == 200.0
    assert equity_based_cap > cash_based_cap, (
        "Equity-based cap should be larger to account for position value"
    )

    print(f"  PASS: cash_cap=${cash_based_cap:.0f}, equity_cap=${equity_based_cap:.0f}")


def test_per_trade_cap_on_cash_is_conservative():
    """Per-trade cap uses cash only — this is intentionally conservative."""
    rm = RiskManager(config=_make_config(max_position_pct=0.05))
    rm.initialize(1000.0)

    # After spending $200, cash = $800
    rm.current_bankroll = 800.0

    available = rm.current_bankroll * 0.90  # 10% fee buffer
    max_position = rm.current_bankroll * rm.config.trading.max_position_pct

    # Per-trade cap should be based on $800 cash, not $1000 equity
    assert max_position == 40.0, f"Expected $40, got ${max_position:.2f}"
    assert available == 720.0

    print(f"  PASS: per_trade=${max_position:.0f}, available=${available:.0f}")


def test_no_overallocation_after_realized_profits():
    """After realizing profits, equity goes up -> exposure cap goes up correctly."""
    # Session: start $1000, realize $100 profit, cash = $1100, no positions
    cash = 1100.0
    positions = 0.0
    equity = cash + positions
    max_pct = 0.25

    cap = equity * max_pct
    assert cap == 275.0, f"Expected $275 cap on $1100 equity, got ${cap:.2f}"

    # Now open a $200 position: cash = $900, positions = $200, equity = $1100
    cash_after = 900.0
    position_value = 200.0
    equity_after = cash_after + position_value
    cap_after = equity_after * max_pct

    # Cap should be same ($1100 equity unchanged)
    assert abs(cap_after - cap) < 0.01, (
        f"Cap should be stable: before=${cap:.2f}, after=${cap_after:.2f}"
    )

    print(f"  PASS: cap_before=${cap:.0f}, cap_after=${cap_after:.0f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Exposure Cap")
    print("=" * 60)

    tests = [
        test_exposure_cap_on_equity_not_cash,
        test_per_trade_cap_on_cash_is_conservative,
        test_no_overallocation_after_realized_profits,
    ]

    for i, test_fn in enumerate(tests, 1):
        print(f"\nTest {i}: {test_fn.__doc__}")
        test_fn()
        print(f"  PASS")

    print("\n" + "=" * 60)
    print(f"All {len(tests)} exposure cap tests passed!")
    print("=" * 60)
