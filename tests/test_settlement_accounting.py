"""
Test settlement accounting: cash, equity, peak after settlement.

Validates the critical invariant: after settlement, BOTH the paper executor
and risk manager must reflect the settlement proceeds. The paper executor
balance must be credited so that _sync_existing_orders() doesn't overwrite
the risk manager's correct value.

Scenario A: open -> settle WIN -> no positions → cash increases, peak not reduced
Scenario B: open -> settle LOSS -> no positions → cash decreases, peak unchanged
Scenario C: two wins → cash and peak both reflect cumulative proceeds
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from src.models import MarketState, Position, Side, DailyStats
from src.strategy.risk import RiskManager, PauseState
from src.execution.paper import PaperAccount, PaperOrderExecutor


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
        end_time=now - timedelta(minutes=5),
        best_bid=0.50,
        best_ask=0.55,
    )


def _open_position(rm, market, side, entry_price, shares):
    """Simulate opening a position (deducting cost from bankroll)."""
    held = market.up_token_id if side == Side.UP else market.down_token_id
    position = Position(
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
    rm.positions[market.condition_id] = position
    cost = entry_price * shares
    rm.current_bankroll -= cost
    if rm.daily_stats:
        rm.daily_stats.current_bankroll = rm.current_bankroll
    return position


def _settle_position(rm, market, position, winning_outcome, paper_executor=None):
    """
    Simulate the settlement flow: compute payout, credit paper executor,
    then call risk_manager.record_position_close.
    """
    winning_token = (
        market.up_token_id if winning_outcome == "UP"
        else market.down_token_id
    )
    held = position.held_token_id or position.token_id
    payout_per_share = 1.0 if held == winning_token else 0.0
    pnl = (payout_per_share - position.entry_price) * position.shares
    proceeds = position.cost_basis + pnl

    # Credit paper executor (the fix)
    if paper_executor is not None:
        paper_executor.credit_settlement(proceeds, market.condition_id)

    rm.record_position_close(
        market_key=market.condition_id,
        exit_price=payout_per_share,
        pnl=pnl,
    )

    return pnl, proceeds


# ---------------------------------------------------------------------------
# Scenario A: WIN settlement → cash increases, peak not reduced
# ---------------------------------------------------------------------------

def test_win_settlement_cash_increases():
    """After settling a WIN, cash must include payout. Peak must not be reduced."""
    rm = RiskManager(config=_make_config())
    rm.initialize(1000.0)

    paper = PaperAccount(1000.0)

    market = _make_market("cond_win", "BTC")
    pos = _open_position(rm, market, Side.UP, entry_price=0.52, shares=96.15)

    # Debit paper account for the order fill
    cost = 0.52 * 96.15
    paper.debit(cost)

    # At this point:
    #   rm.current_bankroll = 1000 - cost ≈ 950.00
    #   paper.balance = 1000 - cost ≈ 950.00
    bankroll_after_open = rm.current_bankroll
    assert abs(bankroll_after_open - (1000.0 - cost)) < 0.01

    # Settle as WIN (UP wins, holding UP token)
    # Create a mock paper executor with credit_settlement
    class MockExecutor:
        def __init__(self, account):
            self.account = account
        def get_balance(self):
            return self.account.balance
        def credit_settlement(self, proceeds, market_id=""):
            if proceeds > 0:
                self.account.credit(proceeds)

    executor = MockExecutor(paper)
    pnl, proceeds = _settle_position(rm, market, pos, "UP", paper_executor=executor)

    # Verify PnL is positive
    assert pnl > 0, f"Expected positive PnL for WIN, got {pnl:.2f}"

    # Verify cash includes the payout
    expected_cash = bankroll_after_open + proceeds
    assert abs(rm.current_bankroll - expected_cash) < 0.01, (
        f"Risk manager cash should be {expected_cash:.2f}, got {rm.current_bankroll:.2f}"
    )

    # Verify paper executor also has the correct balance
    expected_paper = 1000.0 - cost + proceeds  # = 1000 + pnl
    assert abs(paper.balance - expected_paper) < 0.01, (
        f"Paper balance should be {expected_paper:.2f}, got {paper.balance:.2f}"
    )

    # Now simulate what _sync_existing_orders does in paper mode
    rm.sync_bankroll(executor.get_balance())

    # After sync, bankroll should STILL reflect the win
    assert abs(rm.current_bankroll - expected_cash) < 0.01, (
        f"After sync, bankroll should be {expected_cash:.2f}, got {rm.current_bankroll:.2f}"
    )

    # Peak should be >= starting bankroll + profits
    assert rm.peak_bankroll >= expected_cash - 0.01, (
        f"Peak should be >= {expected_cash:.2f}, got {rm.peak_bankroll:.2f}"
    )

    # No positions → drawdown check should NOT reduce peak
    ok, reason = rm.can_trade()
    assert rm.peak_bankroll >= expected_cash - 0.01, (
        f"Peak should not be reduced after can_trade(), got {rm.peak_bankroll:.2f}"
    )

    print(f"  PASS: WIN settlement. Cash=${rm.current_bankroll:.2f}, "
          f"Peak=${rm.peak_bankroll:.2f}, Paper=${paper.balance:.2f}")


# ---------------------------------------------------------------------------
# Scenario B: LOSS settlement → cash decreases, peak unchanged
# ---------------------------------------------------------------------------

def test_loss_settlement_peak_unchanged():
    """After settling a LOSS, cash decreases but peak is unchanged."""
    rm = RiskManager(config=_make_config())
    rm.initialize(1000.0)

    paper = PaperAccount(1000.0)

    market = _make_market("cond_loss", "ETH")
    pos = _open_position(rm, market, Side.UP, entry_price=0.52, shares=96.15)

    cost = 0.52 * 96.15
    paper.debit(cost)

    peak_before = rm.peak_bankroll

    # Settle as LOSS (DOWN wins, but we hold UP)
    class MockExecutor:
        def __init__(self, account):
            self.account = account
        def get_balance(self):
            return self.account.balance
        def credit_settlement(self, proceeds, market_id=""):
            if proceeds > 0:
                self.account.credit(proceeds)

    executor = MockExecutor(paper)
    pnl, proceeds = _settle_position(rm, market, pos, "DOWN", paper_executor=executor)

    # Verify PnL is negative (full loss: payout=0)
    assert pnl < 0, f"Expected negative PnL for LOSS, got {pnl:.2f}"
    assert proceeds == 0.0, f"Proceeds should be 0 for full loss, got {proceeds:.2f}"

    # Cash should be reduced by the full cost basis
    expected_cash = 1000.0 - cost + 0.0  # cost was deducted, 0 returned
    assert abs(rm.current_bankroll - expected_cash) < 0.01, (
        f"Cash should be {expected_cash:.2f}, got {rm.current_bankroll:.2f}"
    )

    # Paper balance should match (nothing credited for loss)
    assert abs(paper.balance - expected_cash) < 0.01, (
        f"Paper balance should be {expected_cash:.2f}, got {paper.balance:.2f}"
    )

    # Sync should not break anything
    rm.sync_bankroll(executor.get_balance())
    assert abs(rm.current_bankroll - expected_cash) < 0.01

    # Peak must NOT be reduced
    assert rm.peak_bankroll == peak_before, (
        f"Peak should be unchanged at {peak_before:.2f}, got {rm.peak_bankroll:.2f}"
    )

    # Drawdown should be correctly computed
    drawdown = (rm.peak_bankroll - rm.current_bankroll) / rm.peak_bankroll
    assert drawdown > 0, "Drawdown should be positive after a loss"

    print(f"  PASS: LOSS settlement. Cash=${rm.current_bankroll:.2f}, "
          f"Peak=${rm.peak_bankroll:.2f} (unchanged), Drawdown={drawdown:.1%}")


# ---------------------------------------------------------------------------
# Scenario C: Two consecutive WINs → cumulative accounting correct
# ---------------------------------------------------------------------------

def test_two_wins_cumulative_accounting():
    """Two wins: cash = starting + sum(pnl), peak = max(bankroll)."""
    rm = RiskManager(config=_make_config())
    rm.initialize(950.0)

    paper = PaperAccount(950.0)

    class MockExecutor:
        def __init__(self, account):
            self.account = account
        def get_balance(self):
            return self.account.balance
        def credit_settlement(self, proceeds, market_id=""):
            if proceeds > 0:
                self.account.credit(proceeds)

    executor = MockExecutor(paper)

    # --- Trade 1: BTC UP @ 0.52, 96.15 shares, WIN ---
    mkt1 = _make_market("cond_btc", "BTC")
    pos1 = _open_position(rm, mkt1, Side.UP, 0.52, 96.15)
    paper.debit(0.52 * 96.15)

    pnl1, proceeds1 = _settle_position(rm, mkt1, pos1, "UP", paper_executor=executor)

    # Sync after settlement 1
    rm.sync_bankroll(executor.get_balance())

    cash_after_1 = rm.current_bankroll
    peak_after_1 = rm.peak_bankroll

    assert pnl1 > 0
    assert abs(cash_after_1 - (950.0 + pnl1)) < 0.01, (
        f"After trade 1: expected {950 + pnl1:.2f}, got {cash_after_1:.2f}"
    )

    # --- Trade 2: ETH UP @ 0.48, 100 shares, WIN ---
    mkt2 = _make_market("cond_eth", "ETH")
    pos2 = _open_position(rm, mkt2, Side.UP, 0.48, 100.0)
    paper.debit(0.48 * 100.0)

    pnl2, proceeds2 = _settle_position(rm, mkt2, pos2, "UP", paper_executor=executor)

    # Sync after settlement 2
    rm.sync_bankroll(executor.get_balance())

    cash_after_2 = rm.current_bankroll
    peak_after_2 = rm.peak_bankroll

    assert pnl2 > 0
    expected_final = 950.0 + pnl1 + pnl2
    assert abs(cash_after_2 - expected_final) < 0.01, (
        f"After 2 wins: expected {expected_final:.2f}, got {cash_after_2:.2f}"
    )

    # Peak should be the highest point seen
    assert peak_after_2 >= peak_after_1, (
        f"Peak after 2 wins ({peak_after_2:.2f}) should be >= peak after 1 win ({peak_after_1:.2f})"
    )
    assert peak_after_2 >= cash_after_2 - 0.01, (
        f"Peak ({peak_after_2:.2f}) should be >= final cash ({cash_after_2:.2f})"
    )

    # can_trade should succeed (no drawdown since we're at peak)
    ok, reason = rm.can_trade()
    assert ok, f"Should be able to trade after 2 wins, blocked: {reason}"

    # Peak must not be reduced
    assert rm.peak_bankroll == peak_after_2

    print(f"  PASS: Two wins. PnL1=${pnl1:.2f} PnL2=${pnl2:.2f} "
          f"Cash=${cash_after_2:.2f} Peak=${peak_after_2:.2f}")


# ---------------------------------------------------------------------------
# Scenario D: drawdown check does NOT auto-reset peak
# ---------------------------------------------------------------------------

def test_drawdown_does_not_reduce_peak():
    """Drawdown check with no positions must NOT reduce peak_bankroll."""
    rm = RiskManager(config=_make_config(max_drawdown_pct=0.10))
    rm.initialize(1000.0)

    # Simulate: peak was 1000, now bankroll is 850 (15% drawdown > 10% limit)
    rm.current_bankroll = 850.0
    rm.peak_bankroll = 1000.0
    # No positions
    assert len(rm.positions) == 0

    ok, reason = rm.can_trade()

    # Should be BLOCKED, not auto-reset
    assert not ok, "Should be blocked by drawdown"
    assert "Drawdown" in reason

    # Peak must NOT be reduced
    assert rm.peak_bankroll == 1000.0, (
        f"Peak should still be 1000.0, got {rm.peak_bankroll:.2f}"
    )

    print(f"  PASS: Drawdown blocks trading, peak preserved at ${rm.peak_bankroll:.2f}")


# ---------------------------------------------------------------------------
# Scenario E: exact user scenario — peak=1037.81, equity=857.31, positions=0
# ---------------------------------------------------------------------------

def test_peak_never_reduced_exact_scenario():
    """Peak=1037.81, equity=857.31, 0 positions -> peak stays 1037.81, drawdown ~17.4%."""
    rm = RiskManager(config=_make_config(max_drawdown_pct=0.20))
    rm.initialize(1000.0)

    rm.peak_bankroll = 1037.81
    rm.current_bankroll = 857.31
    assert len(rm.positions) == 0

    # update_peak_equity should NOT reduce peak (equity < peak)
    rm.update_peak_equity(857.31)
    assert rm.peak_bankroll == 1037.81, (
        f"Peak must remain 1037.81 after update_peak_equity, got {rm.peak_bankroll:.2f}"
    )

    # Drawdown should be ~17.4%
    drawdown = (rm.peak_bankroll - 857.31) / rm.peak_bankroll
    expected_dd = (1037.81 - 857.31) / 1037.81
    assert abs(drawdown - expected_dd) < 0.001
    assert abs(drawdown - 0.1738) < 0.01, f"Expected ~17.4% drawdown, got {drawdown:.1%}"

    # can_trade should still allow (17.4% < 20% limit)
    ok, reason = rm.can_trade(equity=857.31)
    assert ok, f"Should allow trading (17.4% < 20%), blocked: {reason}"

    # Peak STILL unchanged after can_trade
    assert rm.peak_bankroll == 1037.81

    # With stricter limit, should block
    rm2 = RiskManager(config=_make_config(max_drawdown_pct=0.15))
    rm2.initialize(1000.0)
    rm2.peak_bankroll = 1037.81
    rm2.current_bankroll = 857.31
    ok2, reason2 = rm2.can_trade(equity=857.31)
    assert not ok2, "Should be blocked (17.4% > 15% limit)"
    assert "Drawdown" in reason2
    # Peak must STILL be unchanged
    assert rm2.peak_bankroll == 1037.81

    print(f"  PASS: Peak stays {rm.peak_bankroll:.2f}, drawdown={drawdown:.1%}")


# ---------------------------------------------------------------------------
# Scenario F: update_peak_equity only increases, never decreases
# ---------------------------------------------------------------------------

def test_update_peak_equity_only_increases():
    """update_peak_equity raises peak on new high, ignores lower values."""
    rm = RiskManager(config=_make_config())
    rm.initialize(1000.0)
    assert rm.peak_bankroll == 1000.0

    # Higher equity -> peak increases
    rm.update_peak_equity(1050.0)
    assert rm.peak_bankroll == 1050.0

    # Lower equity -> peak unchanged
    rm.update_peak_equity(1020.0)
    assert rm.peak_bankroll == 1050.0

    # Equal equity -> peak unchanged
    rm.update_peak_equity(1050.0)
    assert rm.peak_bankroll == 1050.0

    # New high -> peak increases again
    rm.update_peak_equity(1100.0)
    assert rm.peak_bankroll == 1100.0

    print(f"  PASS: Peak only increases: {rm.peak_bankroll:.2f}")


# ---------------------------------------------------------------------------
# Scenario G: paper executor credit_settlement works correctly
# ---------------------------------------------------------------------------

def test_paper_executor_credit_settlement():
    """PaperOrderExecutor.credit_settlement credits the account."""
    config = MagicMock()
    config.initial_balance = 500.0
    config.taker_fee_bps = 0
    config.maker_fee_bps = 0
    config.slippage_bps = 0

    executor = PaperOrderExecutor(config=config, clob_feed=MagicMock())
    assert executor.get_balance() == 500.0

    # Simulate order fill: buy 100 shares @ 0.50 = $50 cost
    executor.account.debit(50.0)
    assert executor.get_balance() == 450.0

    # Settle as WIN: proceeds = cost + pnl = 50 + 50 = 100
    executor.credit_settlement(proceeds=100.0, market_id="test_mkt")
    assert executor.get_balance() == 550.0, (
        f"After WIN settlement: expected 550.0, got {executor.get_balance():.2f}"
    )

    # Settle another as LOSS: proceeds = 0
    executor.account.debit(50.0)  # another order fill
    executor.credit_settlement(proceeds=0.0, market_id="test_mkt2")
    assert executor.get_balance() == 500.0, (
        f"After LOSS settlement: expected 500.0, got {executor.get_balance():.2f}"
    )

    print(f"  PASS: credit_settlement works. Balance=${executor.get_balance():.2f}")


# ---------------------------------------------------------------------------
# Scenario H: Settlement idempotency — double settlement does not double credit
# ---------------------------------------------------------------------------

def test_settlement_idempotent_no_double_credit():
    """Double-settling the same market must not credit proceeds twice."""
    rm = RiskManager(config=_make_config())
    rm.initialize(1000.0)

    paper = PaperAccount(1000.0)

    market = _make_market("cond_idem", "BTC")
    pos = _open_position(rm, market, Side.UP, 0.52, 96.15)
    cost = 0.52 * 96.15
    paper.debit(cost)

    class MockExecutor:
        def __init__(self, account):
            self.account = account
        def credit_settlement(self, proceeds, market_id=""):
            if proceeds > 0:
                self.account.credit(proceeds)

    executor = MockExecutor(paper)

    # First settlement (WIN)
    pnl, proceeds = _settle_position(rm, market, pos, "UP", paper_executor=executor)
    cash_after_first = rm.current_bankroll
    paper_after_first = paper.balance

    assert pnl > 0
    assert "cond_idem" not in rm.positions, "Position should be closed after first settle"

    # Second settlement attempt — position already gone from risk manager
    # record_position_close should be a no-op (guard: market_key not in self.positions)
    rm.record_position_close(
        market_key="cond_idem",
        exit_price=1.0,
        pnl=pnl,
    )

    # Bankroll must not change on second call
    assert abs(rm.current_bankroll - cash_after_first) < 0.01, (
        f"Double settlement changed bankroll: {cash_after_first:.2f} -> {rm.current_bankroll:.2f}"
    )

    print(f"  PASS: Settlement idempotent. Cash=${rm.current_bankroll:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Settlement Accounting")
    print("=" * 60)

    tests = [
        test_win_settlement_cash_increases,
        test_loss_settlement_peak_unchanged,
        test_two_wins_cumulative_accounting,
        test_drawdown_does_not_reduce_peak,
        test_peak_never_reduced_exact_scenario,
        test_update_peak_equity_only_increases,
        test_paper_executor_credit_settlement,
        test_settlement_idempotent_no_double_credit,
    ]

    for i, test_fn in enumerate(tests, 1):
        print(f"\nTest {i}: {test_fn.__doc__}")
        test_fn()

    print("\n" + "=" * 60)
    print(f"All {len(tests)} settlement accounting tests passed!")
    print("=" * 60)
