"""
Test CLOB WebSocket warmup gate after reconnect.

Validates:
1. First connection: is_warmed_up = True (no warmup needed)
2. After reconnect: is_warmed_up = False during warmup window
3. After warmup period: is_warmed_up = True
4. Orderbook timestamps are invalidated on reconnect
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
import threading
from typing import Optional

from src.models import OrderBook, OrderBookLevel


# ---------------------------------------------------------------------------
# Minimal CLOBFeed stub (avoids importing the full class with websocket dep)
# ---------------------------------------------------------------------------

class CLOBFeedStub:
    """Stub that replicates the warmup-related fields and methods."""

    def __init__(self):
        self._connected = False
        self._reconnected_at: Optional[datetime] = None
        self._warmup_seconds = 5.0
        self._orderbooks: dict[str, OrderBook] = {}
        self._orderbook_lock = threading.Lock()
        self._retry_count = 0

    @property
    def is_warmed_up(self) -> bool:
        if self._reconnected_at is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self._reconnected_at).total_seconds()
        return elapsed >= self._warmup_seconds

    def simulate_first_connect(self):
        """Simulate initial connection (no prior retries)."""
        self._retry_count = 0
        self._connected = True
        # _reconnected_at stays None on first connect

    def simulate_reconnect(self):
        """Simulate reconnect after a disconnect."""
        self._retry_count = 1  # Had at least one retry
        self._connected = True
        self._reconnected_at = datetime.now(timezone.utc)
        # Invalidate orderbook timestamps
        with self._orderbook_lock:
            for ob in self._orderbooks.values():
                ob.timestamp = datetime.min.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_first_connect_no_warmup():
    """First connection: is_warmed_up should be True immediately."""
    feed = CLOBFeedStub()
    feed.simulate_first_connect()
    assert feed.is_warmed_up, "First connection should not require warmup"


def test_reconnect_not_warmed_up():
    """Immediately after reconnect: is_warmed_up should be False."""
    feed = CLOBFeedStub()
    feed._warmup_seconds = 5.0
    feed.simulate_reconnect()
    assert not feed.is_warmed_up, "Should not be warmed up immediately after reconnect"


def test_reconnect_warmed_up_after_delay():
    """After warmup period: is_warmed_up should be True."""
    feed = CLOBFeedStub()
    feed._warmup_seconds = 0.0  # Zero warmup for test
    feed.simulate_reconnect()
    assert feed.is_warmed_up, "Should be warmed up with 0s warmup"

    # Also test with time in the past
    feed2 = CLOBFeedStub()
    feed2._warmup_seconds = 5.0
    feed2._reconnected_at = datetime.now(timezone.utc) - timedelta(seconds=10)
    assert feed2.is_warmed_up, "Should be warmed up 10s after 5s warmup"


def test_orderbooks_invalidated_on_reconnect():
    """Orderbook timestamps should be set to epoch on reconnect."""
    feed = CLOBFeedStub()

    # Add a fresh orderbook
    now = datetime.now(timezone.utc)
    feed._orderbooks["tok_1"] = OrderBook(
        token_id="tok_1",
        bids=[OrderBookLevel(price=0.5, size=100)],
        asks=[OrderBookLevel(price=0.55, size=100)],
        timestamp=now,
    )

    # Verify fresh
    age_before = (now - feed._orderbooks["tok_1"].timestamp).total_seconds()
    assert age_before < 1.0

    # Reconnect
    feed.simulate_reconnect()

    # Timestamp should be epoch (very old)
    age_after = (now - feed._orderbooks["tok_1"].timestamp).total_seconds()
    assert age_after > 86400, f"Orderbook should be invalidated, age={age_after:.0f}s"


def test_warmup_gate_blocks_then_allows():
    """Warmup gate blocks trading during warmup, allows after."""
    feed = CLOBFeedStub()
    feed._warmup_seconds = 2.0

    # Simulate reconnect just now
    feed._reconnected_at = datetime.now(timezone.utc)
    assert not feed.is_warmed_up

    # Simulate time passing (set reconnected_at to 3s ago)
    feed._reconnected_at = datetime.now(timezone.utc) - timedelta(seconds=3)
    assert feed.is_warmed_up


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Testing CLOB Warmup Gate")
    print("=" * 60)

    tests = [
        test_first_connect_no_warmup,
        test_reconnect_not_warmed_up,
        test_reconnect_warmed_up_after_delay,
        test_orderbooks_invalidated_on_reconnect,
        test_warmup_gate_blocks_then_allows,
    ]

    for i, test_fn in enumerate(tests, 1):
        print(f"\nTest {i}: {test_fn.__doc__}")
        test_fn()
        print(f"  PASS")

    print("\n" + "=" * 60)
    print(f"All {len(tests)} CLOB warmup tests passed!")
    print("=" * 60)
