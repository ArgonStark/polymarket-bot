"""
Test data health gating logic.

Validates:
1. Trading is blocked when circuit breakers are open
2. Trading is blocked when feeds are disconnected
3. Trading is blocked when CLOB data is stale
4. Trading is allowed when all feeds are fresh and connected

Note: Tests the _check_data_health logic without importing the full bot class
(which has heavy dependencies). Instead we replicate the method on a stub.
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from src.models import OrderBook, OrderBookLevel, MarketState


def _check_data_health(self) -> tuple[bool, str]:
    """
    Replica of TradingBot._check_data_health for unit testing.
    Must stay in sync with the real method in src/bot.py.
    """
    max_stale_sec = 60.0

    chainlink_ok = not self.chainlink_feed._circuit_open
    clob_ok = not self.clob_feed._circuit_open

    if not chainlink_ok and not clob_ok:
        return (False, "Both Chainlink and CLOB circuit breakers open")
    if not chainlink_ok:
        return (False, "Chainlink circuit breaker open")
    if not clob_ok:
        return (False, "CLOB circuit breaker open")

    if not self.chainlink_feed.is_connected:
        return (False, "Chainlink feed disconnected")
    if not self.clob_feed.is_connected:
        return (False, "CLOB feed disconnected")

    now = datetime.now(timezone.utc)
    for market in self.markets.values():
        ob = self.clob_feed.get_orderbook(market.up_token_id)
        if ob and ob.timestamp:
            age = (now - ob.timestamp).total_seconds()
            if age > max_stale_sec:
                return (False, f"CLOB orderbook stale for {market.asset} ({age:.0f}s old, max {max_stale_sec:.0f}s)")

    return (True, "")


def _make_bot_stub():
    """Create a minimal stub with data health check dependencies."""
    stub = MagicMock()
    stub.chainlink_feed._circuit_open = False
    stub.chainlink_feed.is_connected = True
    stub.clob_feed._circuit_open = False
    stub.clob_feed.is_connected = True
    stub.markets = {}
    stub._check_data_health = lambda: _check_data_health(stub)
    return stub


def test_healthy_feeds():
    """All feeds healthy -> trading allowed."""
    bot = _make_bot_stub()
    ok, reason = bot._check_data_health()
    assert ok, f"Expected healthy, got: {reason}"


def test_both_circuit_breakers_open():
    """Both circuit breakers open -> blocked."""
    bot = _make_bot_stub()
    bot.chainlink_feed._circuit_open = True
    bot.clob_feed._circuit_open = True

    ok, reason = bot._check_data_health()
    assert not ok
    assert "Both" in reason


def test_chainlink_circuit_breaker_open():
    """Chainlink circuit breaker open -> blocked."""
    bot = _make_bot_stub()
    bot.chainlink_feed._circuit_open = True

    ok, reason = bot._check_data_health()
    assert not ok
    assert "Chainlink" in reason


def test_clob_circuit_breaker_open():
    """CLOB circuit breaker open -> blocked."""
    bot = _make_bot_stub()
    bot.clob_feed._circuit_open = True

    ok, reason = bot._check_data_health()
    assert not ok
    assert "CLOB" in reason


def test_chainlink_disconnected():
    """Chainlink disconnected -> blocked."""
    bot = _make_bot_stub()
    bot.chainlink_feed.is_connected = False

    ok, reason = bot._check_data_health()
    assert not ok
    assert "disconnected" in reason


def test_clob_disconnected():
    """CLOB disconnected -> blocked."""
    bot = _make_bot_stub()
    bot.clob_feed.is_connected = False

    ok, reason = bot._check_data_health()
    assert not ok
    assert "disconnected" in reason


def test_stale_orderbook():
    """CLOB orderbook older than 60s -> blocked."""
    bot = _make_bot_stub()

    now = datetime.now(timezone.utc)
    market = MarketState(
        condition_id="cond_test",
        question="Will BTC go up?",
        up_token_id="up_tok",
        down_token_id="down_tok",
        asset="BTC",
        target_price=100_000.0,
        start_time=now - timedelta(minutes=10),
        end_time=now + timedelta(minutes=5),
        best_bid=0.5,
        best_ask=0.55,
    )
    bot.markets = {"cond_test": market}

    stale_ob = OrderBook(
        token_id="up_tok",
        bids=[OrderBookLevel(price=0.5, size=100)],
        asks=[OrderBookLevel(price=0.55, size=100)],
        timestamp=now - timedelta(seconds=120),
    )
    bot.clob_feed.get_orderbook.return_value = stale_ob

    ok, reason = bot._check_data_health()
    assert not ok
    assert "stale" in reason.lower()


def test_fresh_orderbook_allowed():
    """CLOB orderbook updated recently -> allowed."""
    bot = _make_bot_stub()

    now = datetime.now(timezone.utc)
    market = MarketState(
        condition_id="cond_test",
        question="Will BTC go up?",
        up_token_id="up_tok",
        down_token_id="down_tok",
        asset="BTC",
        target_price=100_000.0,
        start_time=now - timedelta(minutes=10),
        end_time=now + timedelta(minutes=5),
        best_bid=0.5,
        best_ask=0.55,
    )
    bot.markets = {"cond_test": market}

    fresh_ob = OrderBook(
        token_id="up_tok",
        bids=[OrderBookLevel(price=0.5, size=100)],
        asks=[OrderBookLevel(price=0.55, size=100)],
        timestamp=now - timedelta(seconds=5),
    )
    bot.clob_feed.get_orderbook.return_value = fresh_ob

    ok, reason = bot._check_data_health()
    assert ok, f"Expected healthy with fresh data, got: {reason}"


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Data Health Gating")
    print("=" * 60)

    tests = [
        test_healthy_feeds,
        test_both_circuit_breakers_open,
        test_chainlink_circuit_breaker_open,
        test_clob_circuit_breaker_open,
        test_chainlink_disconnected,
        test_clob_disconnected,
        test_stale_orderbook,
        test_fresh_orderbook_allowed,
    ]

    for i, test_fn in enumerate(tests, 1):
        print(f"\nTest {i}: {test_fn.__doc__}")
        test_fn()
        print(f"  PASS")

    print("\n" + "=" * 60)
    print(f"All {len(tests)} data health tests passed!")
    print("=" * 60)
