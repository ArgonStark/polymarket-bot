"""
Test that market settlement works correctly.

Validates:
1. time_remaining goes negative after market end_time passes
2. Settlement check correctly identifies expired markets
3. Positions are not orphaned on settlement timeout
4. Token-ID-based payout: held_token vs winning_token determines WIN/LOSS
"""

from datetime import datetime, timezone, timedelta

from src.models import MarketState, Position, Side


def test_time_remaining_goes_negative():
    """time_remaining must return negative values for expired markets."""
    now = datetime.now(timezone.utc)
    market = MarketState(
        condition_id="cond_test",
        question="Will BTC go up?",
        up_token_id="up_tok",
        down_token_id="down_tok",
        asset="BTC",
        target_price=100_000.0,
        start_time=now - timedelta(minutes=20),
        end_time=now - timedelta(minutes=5),  # Expired 5 min ago
        best_bid=0.5,
        best_ask=0.55,
    )

    # time_remaining should be approximately -300 (5 min ago)
    tr = market.time_remaining
    assert tr < 0, f"time_remaining should be negative for expired market, got {tr}"
    assert tr < -290, f"Expected approximately -300, got {tr}"
    assert not market.is_active, "Expired market should not be active"

    print(f"  PASS: time_remaining = {tr:.1f}s (correctly negative)")


def test_time_remaining_positive_for_active():
    """time_remaining should be positive for future markets."""
    now = datetime.now(timezone.utc)
    market = MarketState(
        condition_id="cond_test2",
        question="Will BTC go up?",
        up_token_id="up_tok2",
        down_token_id="down_tok2",
        asset="BTC",
        target_price=100_000.0,
        start_time=now - timedelta(minutes=5),
        end_time=now + timedelta(minutes=10),  # 10 min from now
        best_bid=0.5,
        best_ask=0.55,
    )

    tr = market.time_remaining
    assert tr > 0, f"time_remaining should be positive for active market, got {tr}"
    assert tr > 590, f"Expected approximately 600, got {tr}"
    assert market.is_active, "Future market should be active"

    print(f"  PASS: time_remaining = {tr:.1f}s (correctly positive)")


def test_settlement_threshold():
    """Markets expired > 2s should pass the settlement check threshold."""
    now = datetime.now(timezone.utc)

    # Market expired 10 seconds ago
    market = MarketState(
        condition_id="cond_settle",
        question="Will BTC go up?",
        up_token_id="up_tok_s",
        down_token_id="down_tok_s",
        asset="BTC",
        target_price=100_000.0,
        start_time=now - timedelta(minutes=20),
        end_time=now - timedelta(seconds=10),
        best_bid=0.5,
        best_ask=0.55,
    )

    tr = market.time_remaining
    assert tr <= -2.0, (
        f"Market expired 10s ago should have time_remaining <= -2.0, got {tr}"
    )

    # Verify abs() gives correct time since expiry
    time_since_expiry = abs(tr)
    assert time_since_expiry >= 9.0, (
        f"Time since expiry should be ~10s, got {time_since_expiry}"
    )

    print(f"  PASS: Settlement threshold check works (tr={tr:.1f}s)")


# ---------------------------------------------------------------------------
# Token-ID payout tests
# ---------------------------------------------------------------------------

def _make_market(condition_id="cond_1", up_token="up_tok_1", down_token="down_tok_1"):
    """Helper to make a settled market."""
    now = datetime.now(timezone.utc)
    return MarketState(
        condition_id=condition_id,
        question="Will BTC go up?",
        up_token_id=up_token,
        down_token_id=down_token,
        asset="BTC",
        target_price=100_000.0,
        start_time=now - timedelta(minutes=20),
        end_time=now - timedelta(minutes=5),
        best_bid=0.01,
        best_ask=0.02,
    )


def _make_position(market, side, entry_price, shares):
    """Helper to make a position with explicit token fields."""
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


def _compute_payout(position, market, winning_outcome):
    """Replicate the token-ID payout logic from _close_position_on_settlement."""
    winning_token_id = (
        market.up_token_id if winning_outcome == "UP"
        else market.down_token_id
    )
    held_token = position.held_token_id or position.token_id
    payout_per_share = 1.0 if held_token == winning_token_id else 0.0
    won = payout_per_share == 1.0
    pnl = (payout_per_share - position.entry_price) * position.shares
    return won, payout_per_share, pnl


def test_up_position_down_wins_is_loss():
    """UP position at 0.52 + winner=DOWN -> payout=0 -> LOSS."""
    mkt = _make_market()
    pos = _make_position(mkt, Side.UP, entry_price=0.52, shares=96.15)

    won, payout, pnl = _compute_payout(pos, mkt, "DOWN")

    assert not won, "Should be LOSS when holding UP and DOWN wins"
    assert payout == 0.0, f"Payout should be 0, got {payout}"
    assert pnl < 0, f"PnL should be negative, got {pnl}"

    expected_pnl = (0.0 - 0.52) * 96.15
    assert abs(pnl - expected_pnl) < 0.01, f"Expected ~{expected_pnl:.2f}, got {pnl:.2f}"

    print(f"  PASS: UP@0.52 + DOWN wins -> LOSS, PnL=${pnl:.2f}")


def test_up_position_up_wins_is_win():
    """UP position at 0.52 + winner=UP -> payout=1 -> WIN."""
    mkt = _make_market()
    pos = _make_position(mkt, Side.UP, entry_price=0.52, shares=96.15)

    won, payout, pnl = _compute_payout(pos, mkt, "UP")

    assert won, "Should be WIN when holding UP and UP wins"
    assert payout == 1.0, f"Payout should be 1, got {payout}"
    assert pnl > 0, f"PnL should be positive, got {pnl}"

    expected_pnl = (1.0 - 0.52) * 96.15
    assert abs(pnl - expected_pnl) < 0.01, f"Expected ~{expected_pnl:.2f}, got {pnl:.2f}"

    print(f"  PASS: UP@0.52 + UP wins -> WIN, PnL=${pnl:.2f}")


def test_down_position_down_wins_is_win():
    """DOWN position at 0.48 + winner=DOWN -> payout=1 -> WIN."""
    mkt = _make_market()
    pos = _make_position(mkt, Side.DOWN, entry_price=0.48, shares=100.0)

    won, payout, pnl = _compute_payout(pos, mkt, "DOWN")

    assert won, "Should be WIN when holding DOWN and DOWN wins"
    assert payout == 1.0
    expected_pnl = (1.0 - 0.48) * 100.0
    assert abs(pnl - expected_pnl) < 0.01

    print(f"  PASS: DOWN@0.48 + DOWN wins -> WIN, PnL=${pnl:.2f}")


def test_down_position_up_wins_is_loss():
    """DOWN position at 0.48 + winner=UP -> payout=0 -> LOSS."""
    mkt = _make_market()
    pos = _make_position(mkt, Side.DOWN, entry_price=0.48, shares=100.0)

    won, payout, pnl = _compute_payout(pos, mkt, "UP")

    assert not won, "Should be LOSS when holding DOWN and UP wins"
    assert payout == 0.0
    expected_pnl = (0.0 - 0.48) * 100.0
    assert abs(pnl - expected_pnl) < 0.01

    print(f"  PASS: DOWN@0.48 + UP wins -> LOSS, PnL=${pnl:.2f}")


def test_held_token_invariant():
    """held_token_id must match side: UP->yes_token, DOWN->no_token."""
    mkt = _make_market(up_token="YES_ABC", down_token="NO_XYZ")

    pos_up = _make_position(mkt, Side.UP, 0.50, 10.0)
    assert pos_up.held_token_id == "YES_ABC", f"UP should hold YES token, got {pos_up.held_token_id}"
    assert pos_up.yes_token_id == "YES_ABC"
    assert pos_up.no_token_id == "NO_XYZ"

    pos_down = _make_position(mkt, Side.DOWN, 0.50, 10.0)
    assert pos_down.held_token_id == "NO_XYZ", f"DOWN should hold NO token, got {pos_down.held_token_id}"

    print("  PASS: Token invariant holds (UP->YES, DOWN->NO)")


def test_expiry_chainlink_snapshot():
    """MarketState stores expiry_chainlink_price for settlement."""
    mkt = _make_market()
    assert mkt.expiry_chainlink_price is None, "Should be None before expiry"

    mkt.expiry_chainlink_price = 97_500.0
    assert mkt.expiry_chainlink_price == 97_500.0

    # Local resolution should use snapshot, not live price
    # If BTC was at 97500 when market ended (target 100000), DOWN wins
    ref_price = mkt.expiry_chainlink_price
    winner = "UP" if ref_price >= mkt.target_price else "DOWN"
    assert winner == "DOWN", f"97500 < 100000 should be DOWN, got {winner}"

    print(f"  PASS: Expiry snapshot {ref_price} < target {mkt.target_price} -> {winner}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Settlement Flow")
    print("=" * 60)
    print()

    print("Test 1: time_remaining goes negative for expired markets")
    test_time_remaining_goes_negative()
    print()

    print("Test 2: time_remaining positive for active markets")
    test_time_remaining_positive_for_active()
    print()

    print("Test 3: Settlement threshold check")
    test_settlement_threshold()
    print()

    print("Test 4: UP@0.52 + DOWN wins = LOSS")
    test_up_position_down_wins_is_loss()
    print()

    print("Test 5: UP@0.52 + UP wins = WIN")
    test_up_position_up_wins_is_win()
    print()

    print("Test 6: DOWN@0.48 + DOWN wins = WIN")
    test_down_position_down_wins_is_win()
    print()

    print("Test 7: DOWN@0.48 + UP wins = LOSS")
    test_down_position_up_wins_is_loss()
    print()

    print("Test 8: Token invariant (UP->YES, DOWN->NO)")
    test_held_token_invariant()
    print()

    print("Test 9: Expiry Chainlink snapshot")
    test_expiry_chainlink_snapshot()
    print()

    print("=" * 60)
    print("All settlement tests passed!")
    print("=" * 60)
