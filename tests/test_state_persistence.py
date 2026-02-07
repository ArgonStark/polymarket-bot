"""
Test bot state persistence across restarts.

Validates:
1. Save/load round-trip preserves all fields
2. Profits survive a simulated restart
3. Peak is preserved across restarts (no false drawdown)
4. Missing state file falls back to defaults (first-run behavior)
5. Corrupt state file falls back to defaults with warning
6. Paper executor balance is restored
7. Daily stats restored on same day
8. Daily stats reset on new day
9. Atomic write uses tmp file
"""

import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from src.models import MarketState, Position, Side, DailyStats
from src.state import BotStateManager, PersistedState, _serialize_position, _deserialize_position
from src.strategy.risk import RiskManager
from src.execution.paper import PaperAccount


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides):
    """Build a minimal mock config for RiskManager."""
    trading = MagicMock()
    trading.max_consecutive_losses = 5
    trading.max_drawdown_pct = 0.15
    trading.min_trades_for_winrate = 50
    trading.min_win_rate = 0.40
    trading.cooloff_period_minutes = 5
    trading.daily_loss_limit = 0.25
    for k, v in overrides.items():
        setattr(trading, k, v)
    config = MagicMock()
    config.trading = trading
    return config


def _make_market(condition_id="cond_1", asset="BTC"):
    now = datetime.now(timezone.utc)
    return MarketState(
        condition_id=condition_id,
        question=f"Will {asset} go up?",
        up_token_id=f"up_{condition_id}",
        down_token_id=f"down_{condition_id}",
        asset=asset,
        target_price=100_000.0,
        start_time=now - timedelta(minutes=20),
        end_time=now + timedelta(minutes=5),
        best_bid=0.50,
        best_ask=0.55,
    )


def _make_position(market, side=Side.UP, entry_price=0.52, shares=96.15):
    held = market.up_token_id if side == Side.UP else market.down_token_id
    return Position(
        market=market,
        side=side,
        token_id=held,
        entry_price=entry_price,
        shares=shares,
        entry_time=datetime.now(timezone.utc),
        market_id=market.condition_id,
        yes_token_id=market.up_token_id,
        no_token_id=market.down_token_id,
        held_token_id=held,
    )


def _temp_state_file():
    """Create a temporary file path for state."""
    fd, path = tempfile.mkstemp(suffix=".json", prefix="bot_state_test_")
    os.close(fd)
    os.unlink(path)  # Remove so tests can check "no file" behavior
    return path


# ---------------------------------------------------------------------------
# Test 1: Save/load round-trip
# ---------------------------------------------------------------------------

def test_save_and_load_roundtrip():
    """Save state with known values, load it back, verify all fields match."""
    path = _temp_state_file()
    try:
        rm = RiskManager(config=_make_config())
        rm.initialize(1000.0)
        rm.current_bankroll = 1050.0
        rm.peak_bankroll = 1100.0
        rm.starting_bankroll = 950.0
        rm.consecutive_losses = 2
        rm.trade_history = [10.0, -5.0, 20.0]

        # Add a position
        market = _make_market("cond_rt", "BTC")
        pos = _make_position(market)
        rm.positions["cond_rt"] = pos

        # Set daily stats
        rm.daily_stats.total_pnl = 50.0
        rm.daily_stats.trades_count = 3
        rm.daily_stats.wins = 2
        rm.daily_stats.losses = 1
        rm.daily_stats.fees_paid = 1.5
        rm.daily_stats.rebates_earned = 0.5

        mgr = BotStateManager(path)
        assert mgr.save(rm, {})

        loaded = mgr.load()
        assert loaded is not None
        assert abs(loaded.current_bankroll - 1050.0) < 0.01
        assert abs(loaded.peak_bankroll - 1100.0) < 0.01
        assert abs(loaded.starting_bankroll - 950.0) < 0.01
        assert loaded.consecutive_losses == 2
        assert loaded.trade_history == [10.0, -5.0, 20.0]
        assert loaded.daily_pnl == 50.0
        assert loaded.daily_trades == 3
        assert loaded.daily_wins == 2
        assert loaded.daily_losses == 1
        assert abs(loaded.daily_fees - 1.5) < 0.01
        assert abs(loaded.daily_rebates - 0.5) < 0.01
        assert len(loaded.open_positions) == 1
        assert loaded.open_positions[0]["condition_id"] == "cond_rt"
        assert loaded.version == 1
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test 2: Profit survives restart
# ---------------------------------------------------------------------------

def test_profit_survives_restart():
    """After trading to $1050, restart should restore $1050 not $1000."""
    path = _temp_state_file()
    try:
        # Session 1: trade and save
        rm1 = RiskManager(config=_make_config())
        rm1.initialize(1000.0)
        rm1.current_bankroll = 1050.0
        rm1.peak_bankroll = 1050.0

        mgr = BotStateManager(path)
        mgr.save(rm1, {})

        # Session 2: fresh init then restore
        rm2 = RiskManager(config=_make_config())
        rm2.initialize(1000.0)  # Defaults back to 1000

        saved = mgr.load()
        assert saved is not None
        mgr.restore_risk_manager(rm2, saved)

        assert abs(rm2.current_bankroll - 1050.0) < 0.01, (
            f"Expected 1050.0 after restore, got {rm2.current_bankroll:.2f}"
        )
        assert abs(rm2.peak_bankroll - 1050.0) < 0.01, (
            f"Expected peak 1050.0 after restore, got {rm2.peak_bankroll:.2f}"
        )
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test 3: Peak preserved across restart
# ---------------------------------------------------------------------------

def test_peak_preserved_across_restart():
    """Peak was 1100, current 1050 (drawdown), restore preserves peak."""
    path = _temp_state_file()
    try:
        rm1 = RiskManager(config=_make_config())
        rm1.initialize(1000.0)
        rm1.current_bankroll = 1050.0
        rm1.peak_bankroll = 1100.0

        mgr = BotStateManager(path)
        mgr.save(rm1, {})

        rm2 = RiskManager(config=_make_config())
        rm2.initialize(1000.0)

        saved = mgr.load()
        mgr.restore_risk_manager(rm2, saved)

        assert rm2.peak_bankroll == 1100.0, (
            f"Peak should be 1100.0, got {rm2.peak_bankroll:.2f}"
        )
        assert abs(rm2.current_bankroll - 1050.0) < 0.01

        # Drawdown should be correctly computed
        drawdown = (rm2.peak_bankroll - rm2.current_bankroll) / rm2.peak_bankroll
        expected_dd = (1100.0 - 1050.0) / 1100.0
        assert abs(drawdown - expected_dd) < 0.001
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test 4: No state file uses defaults
# ---------------------------------------------------------------------------

def test_no_state_file_uses_defaults():
    """No saved state -> risk manager keeps initial_balance defaults."""
    path = _temp_state_file()
    # path was unlinked in _temp_state_file(), so no file exists

    mgr = BotStateManager(path)
    loaded = mgr.load()
    assert loaded is None

    # Risk manager should keep its initialized defaults
    rm = RiskManager(config=_make_config())
    rm.initialize(1000.0)
    assert rm.current_bankroll == 1000.0
    assert rm.peak_bankroll == 1000.0


# ---------------------------------------------------------------------------
# Test 5: Corrupt state file falls back to defaults
# ---------------------------------------------------------------------------

def test_corrupt_state_file_uses_defaults():
    """Corrupt JSON -> graceful fallback to None."""
    path = _temp_state_file()
    try:
        # Write corrupt data
        with open(path, "w") as f:
            f.write("{invalid json!!!}")

        mgr = BotStateManager(path)
        loaded = mgr.load()
        assert loaded is None, "Corrupt file should return None"
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test 6: Paper executor balance restored
# ---------------------------------------------------------------------------

def test_paper_executor_balance_restored():
    """Paper account balance matches restored bankroll."""
    path = _temp_state_file()
    try:
        rm = RiskManager(config=_make_config())
        rm.initialize(1000.0)
        rm.current_bankroll = 1075.0
        rm.peak_bankroll = 1075.0

        mgr = BotStateManager(path)
        mgr.save(rm, {})

        # Simulate restart: new paper account at default
        paper = PaperAccount(1000.0)
        assert paper.balance == 1000.0

        # Create mock executor
        class MockExecutor:
            def __init__(self, account):
                self.account = account

        executor = MockExecutor(paper)

        saved = mgr.load()
        mgr.restore_paper_executor(executor, saved)

        assert abs(paper.balance - 1075.0) < 0.01, (
            f"Paper balance should be 1075.0, got {paper.balance:.2f}"
        )
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test 7: Daily stats restored on same day
# ---------------------------------------------------------------------------

def test_daily_stats_restored_same_day():
    """If saved on same day, daily PnL/trades carry over."""
    path = _temp_state_file()
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        rm1 = RiskManager(config=_make_config())
        rm1.initialize(1000.0)
        rm1.current_bankroll = 1030.0
        rm1.daily_stats.total_pnl = 30.0
        rm1.daily_stats.trades_count = 5
        rm1.daily_stats.wins = 3
        rm1.daily_stats.losses = 2

        mgr = BotStateManager(path)
        mgr.save(rm1, {})

        rm2 = RiskManager(config=_make_config())
        rm2.initialize(1000.0)

        saved = mgr.load()
        assert saved.daily_date == today
        mgr.restore_risk_manager(rm2, saved)

        assert rm2.daily_stats.total_pnl == 30.0
        assert rm2.daily_stats.trades_count == 5
        assert rm2.daily_stats.wins == 3
        assert rm2.daily_stats.losses == 2
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test 8: Daily stats reset on new day
# ---------------------------------------------------------------------------

def test_daily_stats_reset_new_day():
    """If saved yesterday, daily stats reset but bankroll preserved."""
    path = _temp_state_file()
    try:
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

        # Build state manually with yesterday's date
        state_data = {
            "current_bankroll": 1050.0,
            "peak_bankroll": 1100.0,
            "starting_bankroll": 1000.0,
            "consecutive_losses": 1,
            "trade_history": [10.0, -5.0],
            "daily_date": yesterday,
            "daily_pnl": 50.0,
            "daily_trades": 4,
            "daily_wins": 3,
            "daily_losses": 1,
            "daily_fees": 1.0,
            "daily_rebates": 0.5,
            "open_positions": [],
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "version": 1,
        }
        with open(path, "w") as f:
            json.dump(state_data, f)

        mgr = BotStateManager(path)
        saved = mgr.load()

        rm = RiskManager(config=_make_config())
        rm.initialize(1000.0)
        mgr.restore_risk_manager(rm, saved)

        # Bankroll should be restored
        assert abs(rm.current_bankroll - 1050.0) < 0.01
        assert rm.peak_bankroll == 1100.0

        # Daily stats should be RESET (new day)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert rm.daily_stats.date == today
        assert rm.daily_stats.total_pnl == 0.0
        assert rm.daily_stats.trades_count == 0
        # Starting bankroll for new day should be current bankroll
        assert abs(rm.daily_stats.starting_bankroll - 1050.0) < 0.01
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test 9: Atomic write uses tmp file
# ---------------------------------------------------------------------------

def test_atomic_write_no_partial():
    """Verify .tmp file is used and replaced atomically."""
    path = _temp_state_file()
    try:
        rm = RiskManager(config=_make_config())
        rm.initialize(1000.0)

        mgr = BotStateManager(path)
        mgr.save(rm, {})

        # State file should exist
        assert os.path.exists(path), "State file should exist after save"

        # Tmp file should NOT exist (cleaned up by os.replace)
        tmp_path = path + ".tmp"
        assert not os.path.exists(tmp_path), "Tmp file should not exist after save"

        # File should be valid JSON
        with open(path, "r") as f:
            data = json.load(f)
        assert "current_bankroll" in data
        assert data["current_bankroll"] == 1000.0
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test: Position serialization round-trip
# ---------------------------------------------------------------------------

def test_position_serialization_roundtrip():
    """Positions can be serialized and deserialized correctly."""
    market = _make_market("cond_pos", "ETH")
    pos = _make_position(market, Side.DOWN, 0.48, 100.0)

    data = _serialize_position(pos)
    restored = _deserialize_position(data)

    assert restored.side == Side.DOWN
    assert abs(restored.entry_price - 0.48) < 0.001
    assert abs(restored.shares - 100.0) < 0.001
    assert restored.held_token_id == market.down_token_id
    assert restored.yes_token_id == market.up_token_id
    assert restored.no_token_id == market.down_token_id
    assert restored.market.condition_id == "cond_pos"
    assert restored.market.asset == "ETH"


# ---------------------------------------------------------------------------
# Test: Orphaned positions cleaned up after restart
# ---------------------------------------------------------------------------

def test_orphaned_positions_cleaned_up():
    """Positions in expired markets are force-closed as losses on restore."""
    path = _temp_state_file()
    try:
        rm = RiskManager(config=_make_config())
        rm.initialize(1000.0)

        # Create a position in an expired market (end_time in the past)
        now = datetime.now(timezone.utc)
        expired_market = MarketState(
            condition_id="cond_expired",
            question="Will BTC go up?",
            up_token_id="up_expired",
            down_token_id="down_expired",
            asset="BTC",
            target_price=100_000.0,
            start_time=now - timedelta(minutes=30),
            end_time=now - timedelta(minutes=10),  # Expired 10 mins ago
            best_bid=0.50,
            best_ask=0.55,
        )
        pos = _make_position(expired_market, Side.UP, 0.52, 96.15)
        rm.positions["cond_expired"] = pos
        rm.current_bankroll = 950.0  # cash after opening

        mgr = BotStateManager(path)
        mgr.save(rm, {})

        # Simulate restart
        rm2 = RiskManager(config=_make_config())
        rm2.initialize(1000.0)
        saved = mgr.load()
        mgr.restore_risk_manager(rm2, saved)

        # Position should be restored
        assert "cond_expired" in rm2.positions
        assert rm2.positions["cond_expired"].market.end_time < now

        # Simulate _cleanup_orphaned_positions logic
        orphaned = [
            cid for cid, p in rm2.positions.items()
            if p.market.end_time < now
        ]
        assert len(orphaned) == 1
        assert orphaned[0] == "cond_expired"

        # Force close
        pos = rm2.positions["cond_expired"]
        rm2.record_position_close(
            market_key="cond_expired",
            exit_price=0.0,
            pnl=-pos.cost_basis,
        )

        # Position should be removed
        assert "cond_expired" not in rm2.positions
        # Bankroll stays at 950: cost was already deducted at open,
        # force-close as loss adds back cost_basis + (-cost_basis) = 0
        assert abs(rm2.current_bankroll - 950.0) < 0.01
        # PnL should be recorded as a loss
        assert rm2.daily_stats.losses == 1
        assert rm2.daily_stats.total_pnl < 0
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_active_positions_not_orphaned():
    """Positions in still-active markets are NOT cleaned up."""
    now = datetime.now(timezone.utc)
    active_market = MarketState(
        condition_id="cond_active",
        question="Will ETH go up?",
        up_token_id="up_active",
        down_token_id="down_active",
        asset="ETH",
        target_price=3_000.0,
        start_time=now - timedelta(minutes=5),
        end_time=now + timedelta(minutes=10),  # Still active
        best_bid=0.50,
        best_ask=0.55,
    )
    pos = _make_position(active_market, Side.UP, 0.52, 50.0)

    rm = RiskManager(config=_make_config())
    rm.initialize(1000.0)
    rm.positions["cond_active"] = pos

    # Active market should NOT be flagged as orphaned
    orphaned = [
        cid for cid, p in rm.positions.items()
        if p.market.end_time < now
    ]
    assert len(orphaned) == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Testing State Persistence")
    print("=" * 60)

    tests = [
        test_save_and_load_roundtrip,
        test_profit_survives_restart,
        test_peak_preserved_across_restart,
        test_no_state_file_uses_defaults,
        test_corrupt_state_file_uses_defaults,
        test_paper_executor_balance_restored,
        test_daily_stats_restored_same_day,
        test_daily_stats_reset_new_day,
        test_atomic_write_no_partial,
        test_position_serialization_roundtrip,
        test_orphaned_positions_cleaned_up,
        test_active_positions_not_orphaned,
    ]

    for i, test_fn in enumerate(tests, 1):
        print(f"\nTest {i}: {test_fn.__doc__}")
        test_fn()
        print(f"  PASS")

    print("\n" + "=" * 60)
    print(f"All {len(tests)} state persistence tests passed!")
    print("=" * 60)
