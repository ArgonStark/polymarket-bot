"""Test paper executor fill price logic."""

import asyncio
from dataclasses import dataclass

# Use the actual enums from the models
from src.models import Side, OrderAction


@dataclass
class MockMarket:
    asset: str = "BTC"
    condition_id: str = "test_condition_12345678"
    up_token_id: str = "up_token_123"
    down_token_id: str = "down_token_456"
    best_bid: float = 0.5
    best_ask: float = 0.52


@dataclass
class MockSignal:
    market: MockMarket
    side: Side
    recommended_action: OrderAction
    size_usd: float = 100.0
    size_shares: float = 10.0
    recommended_price: float = 0.275  # Submitted limit price
    reasoning: str = "test signal"
    edge: float = 0.05


@dataclass
class MockOrderbook:
    best_bid: float = 0.48
    best_ask: float = 0.52


class MockClobFeed:
    def __init__(self, best_bid=0.48, best_ask=0.52):
        self._best_bid = best_bid
        self._best_ask = best_ask

    def get_orderbook(self, token_id):
        return MockOrderbook(best_bid=self._best_bid, best_ask=self._best_ask)

    def set_prices(self, best_bid, best_ask):
        """Update prices for fill simulation testing."""
        self._best_bid = best_bid
        self._best_ask = best_ask


def test_limit_order_fills_at_submitted_price():
    """LIMIT orders should fill at the submitted price when ask <= limit."""
    from src.execution.paper import PaperOrderExecutor, PaperTradingConfig
    from src.execution.orders import OrderSafetyGuard, OrderSafetyConfig

    config = PaperTradingConfig(
        enabled=True,
        initial_balance=10000.0,
        maker_fee_bps=0.0,
        taker_fee_bps=0.0,
        slippage_bps=0.0,  # No slippage for exact price test
    )

    # Set ask <= limit_price so order fills immediately
    feed = MockClobFeed(best_bid=0.25, best_ask=0.27)  # ask=0.27 <= limit=0.275
    guard = OrderSafetyGuard(OrderSafetyConfig(enabled=False))
    executor = PaperOrderExecutor(config, feed, guard)

    # Submit LIMIT order at 0.275 (like ETH DOWN @ 0.2750)
    signal = MockSignal(
        market=MockMarket(),
        side=Side.DOWN,
        recommended_action=OrderAction.LIMIT,
        size_usd=100.0,
        size_shares=0.0,  # Let it calculate from USD/price
        recommended_price=0.275,  # This is the submitted price
    )

    result = asyncio.run(executor.execute_signal_async(signal))

    assert result.success is True
    # CRITICAL: Fill price should be 0.275 (submitted), NOT 0.5
    assert abs(result.filled_price - 0.275) < 0.001, f"Expected fill at 0.275, got {result.filled_price}"
    # Size should be 100 / 0.275 = ~363.6 shares
    expected_size = 100.0 / 0.275
    assert abs(result.filled_size - expected_size) < 1.0, f"Expected size ~{expected_size}, got {result.filled_size}"
    print(f"LIMIT order: submitted=0.275, filled={result.filled_price}, size={result.filled_size}")
    print("PASSED: LIMIT fills at submitted price")


def test_limit_order_pending_then_fills():
    """LIMIT orders go to pending when ask > limit, then fill when ask drops."""
    from src.execution.paper import PaperOrderExecutor, PaperTradingConfig
    from src.execution.orders import OrderSafetyGuard, OrderSafetyConfig

    config = PaperTradingConfig(
        enabled=True,
        initial_balance=10000.0,
        maker_fee_bps=0.0,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
    )

    # Set ask > limit_price so order goes to pending
    feed = MockClobFeed(best_bid=0.48, best_ask=0.52)  # ask=0.52 > limit=0.275
    guard = OrderSafetyGuard(OrderSafetyConfig(enabled=False))
    executor = PaperOrderExecutor(config, feed, guard)

    signal = MockSignal(
        market=MockMarket(),
        side=Side.DOWN,
        recommended_action=OrderAction.LIMIT,
        size_usd=100.0,
        size_shares=0.0,
        recommended_price=0.275,
    )

    result = asyncio.run(executor.execute_signal_async(signal))

    # Order should succeed but NOT be filled yet (pending)
    assert result.success is True
    assert result.filled_size == 0.0, f"Expected pending (0 filled), got {result.filled_size}"
    assert executor.get_pending_count() == 1, "Expected 1 pending order"
    print(f"Order placed in pending state, count={executor.get_pending_count()}")

    # Now move price so ask <= limit
    feed.set_prices(0.25, 0.27)  # ask=0.27 <= limit=0.275

    # Check pending orders - should trigger fill
    filled = executor.check_pending_orders()
    assert len(filled) == 1, f"Expected 1 filled order, got {len(filled)}"
    assert executor.get_pending_count() == 0, "Expected 0 pending after fill"

    # Verify the fill price
    order_status = executor.get_order_status(result.order_id)
    assert order_status is not None
    assert order_status.status.value == "FILLED"
    assert abs(order_status.price - 0.275) < 0.001, f"Expected fill at 0.275, got {order_status.price}"
    print(f"PASSED: Pending order filled when ask dropped to {0.27}")


def test_market_order_fills_at_best_ask():
    """MARKET orders should fill at best_ask from orderbook."""
    from src.execution.paper import PaperOrderExecutor, PaperTradingConfig
    from src.execution.orders import OrderSafetyGuard, OrderSafetyConfig

    config = PaperTradingConfig(
        enabled=True,
        initial_balance=10000.0,
        maker_fee_bps=0.0,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
    )

    feed = MockClobFeed(best_bid=0.28, best_ask=0.30)  # Realistic DOWN token prices
    guard = OrderSafetyGuard(OrderSafetyConfig(enabled=False))
    executor = PaperOrderExecutor(config, feed, guard)

    signal = MockSignal(
        market=MockMarket(),
        side=Side.DOWN,
        recommended_action=OrderAction.MARKET,
        size_usd=100.0,
        size_shares=0.0,
        recommended_price=0.275,  # Ignored for MARKET orders
    )

    result = asyncio.run(executor.execute_signal_async(signal))

    assert result.success is True
    # MARKET order fills at best_ask (0.30), not 0.5 or 0.275
    assert abs(result.filled_price - 0.30) < 0.001, f"Expected fill at 0.30 (best_ask), got {result.filled_price}"
    print(f"MARKET order: best_ask=0.30, filled={result.filled_price}")
    print("PASSED: MARKET fills at best_ask")


def test_post_only_fills_at_submitted_price():
    """POST_ONLY orders should fill at submitted price when conditions met."""
    from src.execution.paper import PaperOrderExecutor, PaperTradingConfig
    from src.execution.orders import OrderSafetyGuard, OrderSafetyConfig

    config = PaperTradingConfig(
        enabled=True,
        initial_balance=10000.0,
        maker_fee_bps=0.0,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
    )

    # Set ask <= limit so order fills immediately
    feed = MockClobFeed(best_bid=0.28, best_ask=0.30)  # ask=0.30 <= limit=0.31
    guard = OrderSafetyGuard(OrderSafetyConfig(enabled=False))
    executor = PaperOrderExecutor(config, feed, guard)

    signal = MockSignal(
        market=MockMarket(),
        side=Side.DOWN,
        recommended_action=OrderAction.POST_ONLY,
        size_usd=100.0,
        size_shares=0.0,
        recommended_price=0.31,  # Maker price just above ask
    )

    result = asyncio.run(executor.execute_signal_async(signal))

    assert result.success is True
    assert abs(result.filled_price - 0.31) < 0.001, f"Expected fill at 0.31, got {result.filled_price}"
    print(f"POST_ONLY order: submitted=0.31, filled={result.filled_price}")
    print("PASSED: POST_ONLY fills at submitted price")


def test_market_order_rejected_without_orderbook():
    """MARKET orders should be rejected if orderbook has no best_ask."""
    from src.execution.paper import PaperOrderExecutor, PaperTradingConfig
    from src.execution.orders import OrderSafetyGuard, OrderSafetyConfig

    config = PaperTradingConfig(
        enabled=True,
        initial_balance=10000.0,
        maker_fee_bps=0.0,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
    )

    # None prices
    feed = MockClobFeed(best_bid=None, best_ask=None)
    guard = OrderSafetyGuard(OrderSafetyConfig(enabled=False))
    executor = PaperOrderExecutor(config, feed, guard)

    signal = MockSignal(
        market=MockMarket(),
        side=Side.DOWN,
        recommended_action=OrderAction.MARKET,
        size_usd=100.0,
        size_shares=0.0,
        recommended_price=0.275,
    )

    result = asyncio.run(executor.execute_signal_async(signal))

    # Should be rejected, not filled at 0.5
    assert result.success is False
    assert "best_ask" in result.error_message.lower() or "orderbook" in result.error_message.lower()
    print(f"MARKET with None orderbook: rejected with '{result.error_message}'")
    print("PASSED: MARKET rejected without valid orderbook")


def test_limit_order_rejected_without_submitted_price():
    """LIMIT orders should be rejected if no submitted price."""
    from src.execution.paper import PaperOrderExecutor, PaperTradingConfig
    from src.execution.orders import OrderSafetyGuard, OrderSafetyConfig

    config = PaperTradingConfig(
        enabled=True,
        initial_balance=10000.0,
        maker_fee_bps=0.0,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
    )

    feed = MockClobFeed(best_bid=0.28, best_ask=0.30)
    guard = OrderSafetyGuard(OrderSafetyConfig(enabled=False))
    executor = PaperOrderExecutor(config, feed, guard)

    signal = MockSignal(
        market=MockMarket(),
        side=Side.DOWN,
        recommended_action=OrderAction.LIMIT,
        size_usd=100.0,
        size_shares=0.0,
        recommended_price=None,  # No price!
    )

    result = asyncio.run(executor.execute_signal_async(signal))

    assert result.success is False
    print(f"LIMIT with no price: rejected with '{result.error_message}'")
    print("PASSED: LIMIT rejected without submitted price")


def test_pnl_near_zero_at_entry():
    """Entry PnL should be near zero (only fee impact) when order fills immediately."""
    from src.execution.paper import PaperOrderExecutor, PaperTradingConfig
    from src.execution.orders import OrderSafetyGuard, OrderSafetyConfig

    config = PaperTradingConfig(
        enabled=True,
        initial_balance=10000.0,
        maker_fee_bps=0.0,  # No fees for this test
        taker_fee_bps=0.0,
        slippage_bps=0.0,
    )

    # Set ask <= limit so order fills immediately
    feed = MockClobFeed(best_bid=0.25, best_ask=0.27)  # ask=0.27 <= limit=0.275
    guard = OrderSafetyGuard(OrderSafetyConfig(enabled=False))
    executor = PaperOrderExecutor(config, feed, guard)

    initial_balance = executor.account.balance

    signal = MockSignal(
        market=MockMarket(),
        side=Side.DOWN,
        recommended_action=OrderAction.LIMIT,
        size_usd=100.0,
        size_shares=0.0,
        recommended_price=0.275,
    )

    result = asyncio.run(executor.execute_signal_async(signal))

    assert result.success is True

    # Calculate expected debit: notional = fill_price * size = 0.275 * (100/0.275) = 100
    # With no fees, balance should decrease by exactly $100
    expected_balance = initial_balance - 100.0
    actual_balance = executor.account.balance

    assert abs(actual_balance - expected_balance) < 0.01, \
        f"Expected balance {expected_balance}, got {actual_balance}"

    print(f"Initial: ${initial_balance}, After: ${actual_balance}, Diff: ${initial_balance - actual_balance}")
    print("PASSED: PnL near zero at entry (balance decreased by notional only)")


class NullClobFeed:
    """CLOB feed that returns no data (simulates missing orderbook)."""
    def get_orderbook(self, token_id):
        return None


def test_no_token_fill_via_quote_cache():
    """
    NO ask == limit (0.145) should fill immediately via quote cache.

    Scenario: CLOB feed has no orderbook data for the DOWN token, but
    MarketState prices are available. The bot feeds derived NO quotes
    into the executor's quote cache. The pending order should fill.
    """
    from src.execution.paper import PaperOrderExecutor, PaperTradingConfig
    from src.execution.orders import OrderSafetyGuard, OrderSafetyConfig

    config = PaperTradingConfig(
        enabled=True,
        initial_balance=10000.0,
        maker_fee_bps=0.0,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
    )

    # CLOB feed returns nothing (simulates the bug)
    feed = NullClobFeed()
    guard = OrderSafetyGuard(OrderSafetyConfig(enabled=False))
    executor = PaperOrderExecutor(config, feed, guard)

    # Place a LIMIT BUY on DOWN token at 0.145
    market = MockMarket(
        asset="SOL",
        condition_id="cond_sol_12345678",
        up_token_id="up_sol_token",
        down_token_id="down_sol_token",
        best_bid=0.855,   # yes_bid (implies no_ask = 1 - 0.855 = 0.145)
        best_ask=0.860,   # yes_ask (implies no_bid = 1 - 0.860 = 0.140)
    )
    signal = MockSignal(
        market=market,
        side=Side.DOWN,
        recommended_action=OrderAction.LIMIT,
        size_usd=50.0,
        size_shares=0.0,
        recommended_price=0.145,  # Limit == no_ask
    )

    # Submit order — should go to pending (feed returns None)
    result = asyncio.run(executor.execute_signal_async(signal))
    assert result.success is True
    assert result.filled_size == 0.0, f"Expected pending, got filled={result.filled_size}"
    assert executor.get_pending_count() == 1
    print(f"  Order placed pending, id={result.order_id[:12]}")

    # Now feed quotes via cache (as the bot would from MarketState)
    # DOWN token: bid = 1 - yes_ask = 0.140, ask = 1 - yes_bid = 0.145
    executor.update_quotes("down_sol_token", bid=0.140, ask=0.145)

    # Evaluate fills — no_ask (0.145) == limit (0.145) → should FILL
    filled = executor.check_pending_orders()
    assert len(filled) == 1, f"Expected 1 fill, got {len(filled)}"
    assert executor.get_pending_count() == 0

    # Verify fill price is at limit (touch fill)
    order_status = executor.get_order_status(result.order_id)
    assert order_status is not None
    assert order_status.status.value == "FILLED"
    assert abs(order_status.price - 0.145) < 0.001, \
        f"Expected fill at 0.145 (touch), got {order_status.price}"

    expected_shares = 50.0 / 0.145
    assert abs(order_status.filled_size - expected_shares) < 1.0, \
        f"Expected ~{expected_shares:.1f} shares, got {order_status.filled_size}"

    print(f"  FILLED at {order_status.price:.4f} via quote cache (touch fill)")
    print("PASSED: NO ask == limit fills via quote cache")


def test_order_eval_log_emitted(capfd=None):
    """ORDER_EVAL debug log is emitted for each open order during check."""
    import logging
    from src.execution.paper import PaperOrderExecutor, PaperTradingConfig
    from src.execution.orders import OrderSafetyGuard, OrderSafetyConfig

    config = PaperTradingConfig(
        enabled=True,
        initial_balance=10000.0,
        maker_fee_bps=0.0,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
    )

    feed = NullClobFeed()
    guard = OrderSafetyGuard(OrderSafetyConfig(enabled=False))
    executor = PaperOrderExecutor(config, feed, guard)

    # Enable debug logging to capture ORDER_EVAL
    paper_logger = logging.getLogger("src.execution.paper")
    paper_logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    paper_logger.addHandler(handler)

    market = MockMarket(asset="ETH")
    signal = MockSignal(
        market=market,
        side=Side.UP,
        recommended_action=OrderAction.LIMIT,
        size_usd=100.0,
        size_shares=0.0,
        recommended_price=0.500,
    )

    # Place order (goes pending)
    result = asyncio.run(executor.execute_signal_async(signal))
    assert result.success

    # Feed quotes that DON'T meet fill condition (ask > limit)
    executor.update_quotes("up_token_123", bid=0.48, ask=0.52)

    # Check — should HOLD, and emit ORDER_EVAL log
    import io
    log_capture = io.StringIO()
    capture_handler = logging.StreamHandler(log_capture)
    capture_handler.setLevel(logging.DEBUG)
    paper_logger.addHandler(capture_handler)

    filled = executor.check_pending_orders()
    assert len(filled) == 0

    log_output = log_capture.getvalue()
    assert "ORDER_EVAL" in log_output, f"Expected ORDER_EVAL in log, got: {log_output}"
    assert "decision=HOLD" in log_output, f"Expected decision=HOLD in log, got: {log_output}"
    print(f"  ORDER_EVAL log captured: {log_output.strip()[:120]}...")
    print("PASSED: ORDER_EVAL debug log emitted")

    # Cleanup
    paper_logger.removeHandler(handler)
    paper_logger.removeHandler(capture_handler)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Paper Executor Fill Price Logic")
    print("=" * 60)
    print()

    print("Test 1: LIMIT order fills at submitted price")
    test_limit_order_fills_at_submitted_price()
    print()

    print("Test 2: LIMIT order pending then fills")
    test_limit_order_pending_then_fills()
    print()

    print("Test 3: MARKET order fills at best_ask")
    test_market_order_fills_at_best_ask()
    print()

    print("Test 4: POST_ONLY fills at submitted price")
    test_post_only_fills_at_submitted_price()
    print()

    print("Test 5: MARKET rejected without orderbook")
    test_market_order_rejected_without_orderbook()
    print()

    print("Test 6: LIMIT rejected without submitted price")
    test_limit_order_rejected_without_submitted_price()
    print()

    print("Test 7: PnL near zero at entry")
    test_pnl_near_zero_at_entry()
    print()

    print("Test 8: NO token fill via quote cache (NO ask == limit)")
    test_no_token_fill_via_quote_cache()
    print()

    print("Test 9: ORDER_EVAL log emitted")
    test_order_eval_log_emitted()
    print()

    print("=" * 60)
    print("All tests passed!")
    print("Paper executor now fills at realistic prices with pending order support.")
    print("=" * 60)
