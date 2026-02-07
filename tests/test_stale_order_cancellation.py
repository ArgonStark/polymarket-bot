"""Test stale order cancellation during market transitions."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.models import Side, OrderAction
from src.execution.orders import OrderSafetyGuard, OrderSafetyConfig


@dataclass
class MockMarket:
    asset: str = "BTC"
    condition_id: str = "test_market_001"
    up_token_id: str = "up_token_123"
    down_token_id: str = "down_token_456"
    best_bid: float = 0.5
    best_ask: float = 0.52


@dataclass
class MockSignal:
    market: MockMarket = field(default_factory=MockMarket)
    side: Side = Side.UP
    recommended_action: OrderAction = OrderAction.LIMIT
    size_usd: float = 100.0
    size_shares: float = 10.0
    recommended_price: float = 0.51
    reasoning: str = "test signal"
    edge: float = 0.05


def test_stale_market_detection():
    """Safety guard blocks orders for old market after transition."""
    guard = OrderSafetyGuard(OrderSafetyConfig(enabled=True))

    # Set initial active market
    guard.set_active_market("BTC", "market_period_1")

    # Verify current market is active
    assert guard.get_active_market("BTC") == "market_period_1"

    # Simulate market transition
    old_market = guard.set_active_market("BTC", "market_period_2")
    assert old_market == "market_period_1", "Should return old market_id"
    assert guard.get_active_market("BTC") == "market_period_2"

    # Create signal for OLD market (should be blocked)
    old_market_obj = MockMarket(condition_id="market_period_1")
    old_signal = MockSignal(market=old_market_obj)

    can_submit, reason = guard.check_can_submit(old_signal, 0.0, 10000.0)

    assert not can_submit, "Should block stale market order"
    assert "stale_market" in reason, f"Reason should be stale_market, got: {reason}"
    print(f"✅ Stale market blocked: {reason}")

    # Create signal for CURRENT market (should pass)
    current_market_obj = MockMarket(condition_id="market_period_2")
    current_signal = MockSignal(market=current_market_obj)

    can_submit, reason = guard.check_can_submit(current_signal, 0.0, 10000.0)

    assert can_submit, f"Should allow current market order, got: {reason}"
    print("✅ Current market allowed")


def test_no_active_market_allows_all():
    """When no active market is set, all orders are allowed."""
    guard = OrderSafetyGuard(OrderSafetyConfig(enabled=True))

    # Don't set any active market
    signal = MockSignal()

    can_submit, reason = guard.check_can_submit(signal, 0.0, 10000.0)

    assert can_submit, f"Should allow when no active market tracking, got: {reason}"
    print("✅ No active market set - order allowed")


def test_clear_active_market():
    """Clearing active market stops stale detection for that asset."""
    guard = OrderSafetyGuard(OrderSafetyConfig(enabled=True))

    # Set and then clear active market
    guard.set_active_market("BTC", "market_001")
    guard.clear_active_market("BTC")

    assert guard.get_active_market("BTC") is None

    # Now any market should be allowed
    signal = MockSignal(market=MockMarket(condition_id="any_market"))
    can_submit, reason = guard.check_can_submit(signal, 0.0, 10000.0)

    assert can_submit, f"Should allow after clearing, got: {reason}"
    print("✅ Active market cleared - order allowed")


def test_multiple_assets_independent():
    """Each asset has independent active market tracking."""
    guard = OrderSafetyGuard(OrderSafetyConfig(enabled=True))

    # Set different active markets for different assets
    guard.set_active_market("BTC", "btc_market_001")
    guard.set_active_market("ETH", "eth_market_001")

    # BTC signal for correct market
    btc_signal = MockSignal(market=MockMarket(asset="BTC", condition_id="btc_market_001"))
    can_submit, _ = guard.check_can_submit(btc_signal, 0.0, 10000.0)
    assert can_submit, "BTC correct market should be allowed"

    # ETH signal for correct market
    eth_signal = MockSignal(market=MockMarket(asset="ETH", condition_id="eth_market_001"))
    can_submit, _ = guard.check_can_submit(eth_signal, 0.0, 10000.0)
    assert can_submit, "ETH correct market should be allowed"

    # BTC signal for WRONG market (stale)
    btc_stale_signal = MockSignal(market=MockMarket(asset="BTC", condition_id="btc_market_old"))
    can_submit, reason = guard.check_can_submit(btc_stale_signal, 0.0, 10000.0)
    assert not can_submit, "BTC stale market should be blocked"
    assert "stale_market" in reason

    print("✅ Multiple assets tracked independently")


def test_order_tracking_with_market_transition():
    """Open orders are tracked per market_id and can be cleared."""
    guard = OrderSafetyGuard(OrderSafetyConfig(enabled=True))

    # Set active market
    guard.set_active_market("BTC", "market_001")

    # Record an order
    signal = MockSignal(market=MockMarket(condition_id="market_001"))
    guard.record_order_submitted(signal, "order_123")

    # Verify order is tracked
    assert ("BTC", "market_001", "UP") in guard._open_orders

    # Try to submit duplicate - should be blocked
    can_submit, reason = guard.check_can_submit(signal, 0.0, 10000.0)
    assert not can_submit
    assert "duplicate_order" in reason

    # Transition to new market
    guard.set_active_market("BTC", "market_002")

    # Old order should still be tracked (until explicitly cleared)
    assert ("BTC", "market_001", "UP") in guard._open_orders

    # Clear orders for old market
    guard.record_order_cancelled("BTC", "market_001", "UP", "order_123")
    assert ("BTC", "market_001", "UP") not in guard._open_orders

    print("✅ Order tracking with market transitions works")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Stale Order Cancellation")
    print("=" * 60)
    print()

    print("Test 1: Stale market detection")
    test_stale_market_detection()
    print()

    print("Test 2: No active market allows all")
    test_no_active_market_allows_all()
    print()

    print("Test 3: Clear active market")
    test_clear_active_market()
    print()

    print("Test 4: Multiple assets independent")
    test_multiple_assets_independent()
    print()

    print("Test 5: Order tracking with market transition")
    test_order_tracking_with_market_transition()
    print()

    print("=" * 60)
    print("All stale order cancellation tests passed!")
    print("=" * 60)
