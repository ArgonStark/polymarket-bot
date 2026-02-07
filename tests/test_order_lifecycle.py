#!/usr/bin/env python3
"""
Minimal test to verify ORDER_SUBMIT/ORDER_BLOCK/ORDER_RESULT/ORDER_FILL logs are emitted.

Run with:
    python tests/test_order_lifecycle.py
"""

import asyncio
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

# Setup logging to see all INFO messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)

# Import after logging setup
from src.execution.paper import PaperOrderExecutor, PaperTradingConfig
from src.execution.orders import OrderSafetyGuard, OrderSafetyConfig, reset_safety_guard
from src.models import Signal, OrderAction, Side


@dataclass
class MockMarket:
    """Minimal mock market for testing."""
    condition_id: str = "test_market_123456"
    asset: str = "BTC"
    up_token_id: str = "up_token_abc"
    down_token_id: str = "down_token_xyz"
    target_price: float = 100000.0
    question: str = "Test market"
    start_time: datetime = None
    end_time: datetime = None
    time_remaining: float = 300.0

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now(timezone.utc)
        if self.end_time is None:
            self.end_time = self.start_time + timedelta(minutes=15)


@dataclass
class MockOrderBook:
    """Mock orderbook for paper trading."""
    best_bid: float = 0.55
    best_ask: float = 0.57


def create_test_signal(market: MockMarket, side: Side = Side.UP) -> Signal:
    """Create a test signal."""
    return Signal(
        market=market,
        side=side,
        edge=0.05,
        true_prob=0.60,
        market_prob=0.55,
        recommended_action=OrderAction.LIMIT,
        recommended_price=0.56,
        size_usd=10.0,
        size_shares=17.86,
        chainlink_price=100000.0,
        time_remaining=300.0,
        reasoning="Test signal",
    )


class MockClobFeed:
    """Mock CLOB feed that returns orderbooks."""
    def get_orderbook(self, token_id: str) -> MockOrderBook:
        return MockOrderBook()


async def test_paper_order_lifecycle():
    """Test that paper executor emits ORDER_SUBMIT, ORDER_RESULT, ORDER_FILL."""
    print("\n" + "=" * 60)
    print("TEST: Paper Order Lifecycle")
    print("=" * 60)

    # Reset safety guard for clean test
    safety_config = OrderSafetyConfig(
        per_asset_cooldown_sec=1.0,  # Short cooldown for testing
        global_max_orders_per_minute=100,
        max_exposure_pct=1.0,  # Allow high exposure for test
        enabled=True,
    )
    guard = reset_safety_guard(safety_config)

    # Create paper executor
    paper_config = PaperTradingConfig(
        enabled=True,
        initial_balance=1000.0,
        maker_fee_bps=10.0,
        taker_fee_bps=25.0,
        slippage_bps=5.0,
    )
    executor = PaperOrderExecutor(paper_config, MockClobFeed(), guard)

    # Create test signal
    market = MockMarket()
    signal = create_test_signal(market, Side.UP)

    print(f"\n>>> Submitting first order for {market.asset} {signal.side.value}...")
    result = await executor.execute_signal_async(signal)
    print(f">>> Result: success={result.success}, order_id={result.order_id}")
    print(f">>> filled_size={result.filled_size:.4f}, filled_price={result.filled_price:.4f}")

    # Check safety guard stats
    stats = guard.get_stats()
    print(f"\n>>> Safety guard stats: {stats}")

    # Try to submit duplicate order (should be blocked by cooldown)
    print(f"\n>>> Submitting duplicate order (should be blocked by cooldown)...")
    result2 = await executor.execute_signal_async(signal)
    print(f">>> Result: success={result2.success}, error={result2.error_message}")

    # Wait for cooldown and try again (should be blocked by existing order for same market+side)
    print(f"\n>>> Waiting 1.5s for cooldown to expire...")
    await asyncio.sleep(1.5)

    print(f"\n>>> Submitting after cooldown (safety guard should still block duplicate)...")
    result3 = await executor.execute_signal_async(signal)
    print(f">>> Result: success={result3.success}, error={result3.error_message}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE - Check logs above for ORDER_SUBMIT/ORDER_RESULT/ORDER_FILL")
    print("=" * 60)


async def test_order_block_reasons():
    """Test various ORDER_BLOCK scenarios."""
    print("\n" + "=" * 60)
    print("TEST: ORDER_BLOCK Scenarios")
    print("=" * 60)

    # Reset with restrictive settings
    safety_config = OrderSafetyConfig(
        per_asset_cooldown_sec=10.0,
        global_max_orders_per_minute=1,  # Very restrictive
        max_exposure_pct=0.01,  # Very low
        enabled=True,
    )
    guard = reset_safety_guard(safety_config)

    paper_config = PaperTradingConfig(
        enabled=True,
        initial_balance=100.0,
        maker_fee_bps=10.0,
        taker_fee_bps=25.0,
        slippage_bps=5.0,
    )
    executor = PaperOrderExecutor(paper_config, MockClobFeed(), guard)

    market = MockMarket()

    # Test SKIP action
    print("\n>>> Testing SKIP action...")
    skip_signal = create_test_signal(market)
    skip_signal = Signal(
        market=market,
        side=Side.UP,
        edge=0.01,
        true_prob=0.51,
        market_prob=0.50,
        recommended_action=OrderAction.SKIP,
        recommended_price=0.50,
        size_usd=10.0,
        size_shares=20.0,
        chainlink_price=100000.0,
        time_remaining=300.0,
        reasoning="Skip test",
    )
    result = await executor.execute_signal_async(skip_signal)
    print(f">>> SKIP result: success={result.success}")

    # Test exposure cap
    print("\n>>> Testing exposure cap block...")
    big_signal = create_test_signal(market)
    big_signal = Signal(
        market=market,
        side=Side.UP,
        edge=0.10,
        true_prob=0.60,
        market_prob=0.50,
        recommended_action=OrderAction.LIMIT,
        recommended_price=0.50,
        size_usd=1000.0,  # Way over exposure limit
        size_shares=2000.0,
        chainlink_price=100000.0,
        time_remaining=300.0,
        reasoning="Big order test",
    )
    result = await executor.execute_signal_async(big_signal, bankroll=100.0)
    print(f">>> Exposure cap result: success={result.success}, error={result.error_message}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


async def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# ORDER LIFECYCLE TEST SUITE")
    print("#" * 60)

    # Verify imports work
    print("\n>>> Verifying module imports...")
    from src.execution import PaperOrderExecutor, OrderSafetyGuard
    print(f">>> PaperOrderExecutor module: {PaperOrderExecutor.__module__}")
    print(f">>> OrderSafetyGuard module: {OrderSafetyGuard.__module__}")

    await test_paper_order_lifecycle()
    await test_order_block_reasons()

    print("\n" + "#" * 60)
    print("# ALL TESTS COMPLETE")
    print("#" * 60)


if __name__ == "__main__":
    asyncio.run(main())
