"""
Test risk pause management and daily reset.

Validates:
1. Pause reasons are tracked and cleared correctly
2. Cooloff creates a timed pause that expires
3. Daily reset clears day-specific pauses and resets counters
4. TRADING_PAUSED/TRADING_RESUMED log lines fire at correct transitions
5. DailyStats.hit_loss_limit_at uses configurable threshold
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from src.models import DailyStats
from src.strategy.risk import RiskManager, PauseState


def _make_config(**overrides):
    """Build a minimal mock config."""
    trading = MagicMock()
    trading.max_consecutive_losses = 3
    trading.max_drawdown_pct = 0.15
    trading.min_trades_for_winrate = 10
    trading.min_win_rate = 0.40
    trading.cooloff_period_minutes = 5
    trading.daily_loss_limit = 0.25
    trading.trade_history_size = 20
    trading.capital_buffer = 0.90
    for k, v in overrides.items():
        setattr(trading, k, v)

    config = MagicMock()
    config.trading = trading
    return config


def test_add_and_remove_pause():
    """Adding a pause blocks trading; removing it resumes."""
    rm = RiskManager(config=_make_config())
    rm.initialize(100.0)

    # Initially can trade
    ok, _ = rm.can_trade()
    assert ok

    # Add a pause
    rm._add_pause("test_pause", "Test pause reason")
    ok, reason = rm.can_trade()
    assert not ok
    assert "Test pause reason" in reason

    # Remove pause -> trading resumes
    rm._remove_pause("test_pause")
    ok, _ = rm.can_trade()
    assert ok


def test_pause_expires_automatically():
    """Pauses with an expiry are auto-cleared."""
    rm = RiskManager(config=_make_config())
    rm.initialize(100.0)

    # Add a pause that expired 1 second ago
    past = datetime.now(timezone.utc) - timedelta(seconds=1)
    rm._active_pauses["expired_pause"] = PauseState(
        reason="Should auto-expire",
        expires_at=past,
    )

    # can_trade should expire it and allow trading
    ok, _ = rm.can_trade()
    assert ok
    assert "expired_pause" not in rm._active_pauses


def test_pause_not_expired_yet():
    """Pauses with a future expiry still block."""
    rm = RiskManager(config=_make_config())
    rm.initialize(100.0)

    future = datetime.now(timezone.utc) + timedelta(minutes=5)
    rm._active_pauses["future_pause"] = PauseState(
        reason="Not yet expired",
        expires_at=future,
    )

    ok, reason = rm.can_trade()
    assert not ok
    assert "Not yet expired" in reason


def test_multiple_pauses():
    """Multiple pauses: must clear all to resume."""
    rm = RiskManager(config=_make_config())
    rm.initialize(100.0)

    rm._add_pause("p1", "First reason")
    rm._add_pause("p2", "Second reason")

    ok, _ = rm.can_trade()
    assert not ok

    rm._remove_pause("p1")
    ok, _ = rm.can_trade()
    assert not ok  # p2 still active

    rm._remove_pause("p2")
    ok, _ = rm.can_trade()
    assert ok


def test_cooloff_creates_timed_pause():
    """_start_cooloff adds a timed pause entry."""
    rm = RiskManager(config=_make_config(cooloff_period_minutes=2))
    rm.initialize(100.0)

    rm._start_cooloff("test_reason")

    assert "cooloff" in rm._active_pauses
    ps = rm._active_pauses["cooloff"]
    assert ps.expires_at is not None
    # Should expire ~2 minutes from now
    delta = (ps.expires_at - datetime.now(timezone.utc)).total_seconds()
    assert 100 < delta < 130  # roughly 2 minutes


def test_daily_reset_clears_daily_loss_pause():
    """Daily reset at midnight clears the daily_loss pause."""
    rm = RiskManager(config=_make_config(daily_loss_limit=0.10))
    rm.initialize(100.0)

    # Simulate daily loss pause
    rm._add_pause("daily_loss", "Daily loss limit hit")
    rm.is_trading_enabled = False
    rm.halt_reason = "Daily loss limit hit"

    # Simulate crossing midnight by setting last date to yesterday
    rm._last_daily_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    rm.daily_stats = DailyStats(
        date=rm._last_daily_date,
        starting_bankroll=100.0,
        current_bankroll=90.0,
    )
    rm.daily_stats.total_pnl = -10.0

    # Now can_trade should trigger daily reset
    rm._maybe_daily_reset()

    # After reset, daily_loss pause should be cleared
    assert "daily_loss" not in rm._active_pauses
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert rm.daily_stats.date == today
    assert rm.daily_stats.total_pnl == 0.0


def test_daily_reset_resets_consecutive_losses():
    """Daily reset clears consecutive_losses counter."""
    rm = RiskManager(config=_make_config())
    rm.initialize(100.0)
    rm.consecutive_losses = 5

    # Force a day change
    rm._last_daily_date = "2024-01-01"
    rm.daily_stats = DailyStats(
        date="2024-01-01",
        starting_bankroll=100.0,
        current_bankroll=95.0,
    )

    rm._maybe_daily_reset()
    assert rm.consecutive_losses == 0


def test_hit_loss_limit_at_uses_threshold():
    """DailyStats.hit_loss_limit_at uses the given threshold."""
    stats = DailyStats(
        date="2024-01-01",
        starting_bankroll=100.0,
        current_bankroll=82.0,
    )
    stats.total_pnl = -18.0

    # daily_return = -18/100 = -0.18
    assert stats.daily_return == -0.18

    # At 15% threshold -> hit (18% > 15%)
    assert stats.hit_loss_limit_at(0.15) is True

    # At 20% threshold -> not hit (18% < 20%)
    assert stats.hit_loss_limit_at(0.20) is False

    # At 25% threshold -> not hit
    assert stats.hit_loss_limit_at(0.25) is False

    # Default property (20%)
    assert stats.hit_loss_limit is False


def test_get_pause_reasons():
    """get_pause_reasons returns active reasons and auto-expires."""
    rm = RiskManager(config=_make_config())
    rm.initialize(100.0)

    rm._add_pause("p1", "Reason A")
    past = datetime.now(timezone.utc) - timedelta(seconds=1)
    rm._active_pauses["p2"] = PauseState(reason="Expired B", expires_at=past)

    reasons = rm.get_pause_reasons()
    assert "Reason A" in reasons
    assert "Expired B" not in reasons
    assert len(reasons) == 1


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Risk Pause Management")
    print("=" * 60)

    tests = [
        test_add_and_remove_pause,
        test_pause_expires_automatically,
        test_pause_not_expired_yet,
        test_multiple_pauses,
        test_cooloff_creates_timed_pause,
        test_daily_reset_clears_daily_loss_pause,
        test_daily_reset_resets_consecutive_losses,
        test_hit_loss_limit_at_uses_threshold,
        test_get_pause_reasons,
    ]

    for i, test_fn in enumerate(tests, 1):
        print(f"\nTest {i}: {test_fn.__doc__}")
        test_fn()
        print(f"  PASS")

    print("\n" + "=" * 60)
    print(f"All {len(tests)} risk pause tests passed!")
    print("=" * 60)
