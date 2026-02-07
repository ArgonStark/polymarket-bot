"""
Test that the price cache is keyed by token_id (not by asset) so that
two markets for the same asset with different token prices never collide.

Validation checklist:
- Two BTC markets with different condition_ids and token_ids.
- Each market's UP and DOWN tokens have very different prices.
- Verify that cache returns correct prices for each token.
- Verify mark-to-market for a DOWN position uses the DOWN token_id book.
"""

from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

from src.execution.paper import PaperOrderExecutor, PaperTradingConfig, CachedQuote
from src.models import Side, Position, MarketState, OrderBook, OrderBookLevel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_market(
    condition_id: str,
    asset: str,
    up_token_id: str,
    down_token_id: str,
    up_bid: float,
    up_ask: float,
) -> MarketState:
    now = datetime.now(timezone.utc)
    from datetime import timedelta
    return MarketState(
        condition_id=condition_id,
        question=f"Will {asset} go up?",
        up_token_id=up_token_id,
        down_token_id=down_token_id,
        asset=asset,
        target_price=100_000.0,
        start_time=now - timedelta(minutes=10),
        end_time=now + timedelta(minutes=5),
        best_bid=up_bid,
        best_ask=up_ask,
    )


@dataclass
class MockClobFeed:
    """CLOB feed that returns nothing (force cache fallback)."""
    def get_orderbook(self, token_id: str) -> Optional[OrderBook]:
        return None

    @property
    def is_connected(self):
        return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cache_keyed_by_token_id():
    """Two BTC markets must not overwrite each other's cache."""
    config = PaperTradingConfig(
        enabled=True, initial_balance=1000.0,
        maker_fee_bps=0, taker_fee_bps=0, slippage_bps=0,
    )
    executor = PaperOrderExecutor(config=config, clob_feed=MockClobFeed())

    # Market A: BTC is heavily UP (UP bid/ask = 0.85/0.87)
    #   => DOWN derived bid = 1 - 0.87 = 0.13, ask = 1 - 0.85 = 0.15
    mkt_a = _make_market(
        "cond_A", "BTC",
        up_token_id="up_token_A",
        down_token_id="down_token_A",
        up_bid=0.85, up_ask=0.87,
    )

    # Market B: BTC is near 50/50 (UP bid/ask = 0.48/0.52)
    #   => DOWN derived bid = 1 - 0.52 = 0.48, ask = 1 - 0.48 = 0.52
    mkt_b = _make_market(
        "cond_B", "BTC",
        up_token_id="up_token_B",
        down_token_id="down_token_B",
        up_bid=0.48, up_ask=0.52,
    )

    # Feed quotes for both markets (same as _refresh_paper_quotes does)
    for mkt in [mkt_a, mkt_b]:
        executor.update_quotes(
            mkt.up_token_id, mkt.best_bid, mkt.best_ask,
            market_id=mkt.condition_id, source="market_state_up",
        )
        no_bid = 1.0 - mkt.best_ask
        no_ask = 1.0 - mkt.best_bid
        executor.update_quotes(
            mkt.down_token_id, no_bid, no_ask,
            market_id=mkt.condition_id, source="market_state_down_derived",
        )

    # --- Assert UP tokens ---
    q_a_up = executor.get_cached_quote("up_token_A")
    q_b_up = executor.get_cached_quote("up_token_B")

    assert q_a_up is not None, "Market A UP quote missing"
    assert q_b_up is not None, "Market B UP quote missing"
    assert abs(q_a_up.bid - 0.85) < 1e-9, f"Expected 0.85, got {q_a_up.bid}"
    assert abs(q_b_up.bid - 0.48) < 1e-9, f"Expected 0.48, got {q_b_up.bid}"
    assert q_a_up.market_id == "cond_A"
    assert q_b_up.market_id == "cond_B"

    # --- Assert DOWN tokens ---
    q_a_down = executor.get_cached_quote("down_token_A")
    q_b_down = executor.get_cached_quote("down_token_B")

    assert q_a_down is not None, "Market A DOWN quote missing"
    assert q_b_down is not None, "Market B DOWN quote missing"
    assert abs(q_a_down.bid - 0.13) < 1e-9, f"Expected 0.13, got {q_a_down.bid}"
    assert abs(q_b_down.bid - 0.48) < 1e-9, f"Expected 0.48, got {q_b_down.bid}"
    assert q_a_down.market_id == "cond_A"
    assert q_b_down.market_id == "cond_B"

    # Cross-contamination check: updating Market B must NOT change Market A
    mkt_b.best_bid = 0.10
    mkt_b.best_ask = 0.12
    executor.update_quotes(
        mkt_b.up_token_id, mkt_b.best_bid, mkt_b.best_ask,
        market_id=mkt_b.condition_id, source="market_state_up",
    )
    executor.update_quotes(
        mkt_b.down_token_id, 1.0 - mkt_b.best_ask, 1.0 - mkt_b.best_bid,
        market_id=mkt_b.condition_id, source="market_state_down_derived",
    )

    # Market A must be unchanged
    q_a_up_after = executor.get_cached_quote("up_token_A")
    q_a_down_after = executor.get_cached_quote("down_token_A")
    assert abs(q_a_up_after.bid - 0.85) < 1e-9, "Market A UP was contaminated!"
    assert abs(q_a_down_after.bid - 0.13) < 1e-9, "Market A DOWN was contaminated!"

    # Market B should reflect the new prices
    q_b_up_after = executor.get_cached_quote("up_token_B")
    q_b_down_after = executor.get_cached_quote("down_token_B")
    assert abs(q_b_up_after.bid - 0.10) < 1e-9, f"Expected 0.10, got {q_b_up_after.bid}"
    assert abs(q_b_down_after.bid - 0.88) < 1e-9, f"Expected 0.88, got {q_b_down_after.bid}"

    print("  PASS: Cache keyed by token_id - no cross-market contamination")


def test_down_position_uses_down_token():
    """A DOWN position must use the DOWN token_id for mark-to-market, not UP."""
    config = PaperTradingConfig(
        enabled=True, initial_balance=1000.0,
        maker_fee_bps=0, taker_fee_bps=0, slippage_bps=0,
    )
    executor = PaperOrderExecutor(config=config, clob_feed=MockClobFeed())

    # Market: UP bid/ask = 0.10/0.11  =>  DOWN bid/ask = 0.89/0.90
    mkt = _make_market(
        "cond_123", "BTC",
        up_token_id="up_tok_123",
        down_token_id="down_tok_123",
        up_bid=0.10, up_ask=0.11,
    )

    # Feed quotes
    executor.update_quotes(
        mkt.up_token_id, mkt.best_bid, mkt.best_ask,
        market_id=mkt.condition_id, source="market_state_up",
    )
    no_bid = 1.0 - mkt.best_ask  # 0.89
    no_ask = 1.0 - mkt.best_bid  # 0.90
    executor.update_quotes(
        mkt.down_token_id, no_bid, no_ask,
        market_id=mkt.condition_id, source="market_state_down_derived",
    )

    # Create a DOWN position (entered at 0.55, market has since moved to 0.89)
    position = Position(
        market=mkt,
        side=Side.DOWN,
        token_id=mkt.down_token_id,  # Correctly set
        entry_price=0.55,
        shares=100.0,
        entry_time=datetime.now(timezone.utc),
    )

    # The quote cache for the DOWN token should return bid=0.89
    down_quote = executor.get_cached_quote(position.token_id)
    assert down_quote is not None, "DOWN quote must exist in cache"
    assert abs(down_quote.bid - 0.89) < 1e-9, (
        f"DOWN bid should be 0.89, got {down_quote.bid} "
        f"(source={down_quote.source}, market_id={down_quote.market_id})"
    )

    # The quote cache for the UP token should return bid=0.10
    up_quote = executor.get_cached_quote(mkt.up_token_id)
    assert up_quote is not None, "UP quote must exist in cache"
    assert abs(up_quote.bid - 0.10) < 1e-9, (
        f"UP bid should be 0.10, got {up_quote.bid}"
    )

    # Verify: a mark using DOWN bid gives correct unrealized PnL
    mark_price = down_quote.bid  # 0.89
    unrealized = (mark_price - position.entry_price) * position.shares
    expected_pnl = (0.89 - 0.55) * 100.0  # +34.0
    assert abs(unrealized - expected_pnl) < 1e-6, (
        f"Unrealized PnL should be {expected_pnl:+.2f}, got {unrealized:+.2f}"
    )

    print("  PASS: DOWN position correctly uses DOWN token_id for marking")


def test_get_quotes_fallback_chain():
    """_get_quotes should prefer CLOB orderbook, then fall back to cache."""
    config = PaperTradingConfig(
        enabled=True, initial_balance=1000.0,
        maker_fee_bps=0, taker_fee_bps=0, slippage_bps=0,
    )

    # CLOB feed that returns an orderbook for one token only
    class PartialClobFeed:
        def get_orderbook(self, token_id: str) -> Optional[OrderBook]:
            if token_id == "up_tok":
                return OrderBook(
                    token_id="up_tok",
                    bids=[OrderBookLevel(price=0.50, size=100)],
                    asks=[OrderBookLevel(price=0.52, size=100)],
                )
            return None  # No data for DOWN token

    executor = PaperOrderExecutor(config=config, clob_feed=PartialClobFeed())

    # Feed cache for DOWN token
    executor.update_quotes(
        "down_tok", 0.48, 0.50,
        market_id="cond_X", source="market_state_down_derived",
    )

    # UP token: should come from CLOB orderbook
    bid, ask, source = executor._get_quotes("up_tok")
    assert source == "orderbook", f"Expected 'orderbook', got '{source}'"
    assert abs(bid - 0.50) < 1e-9

    # DOWN token: CLOB returns None, should fall back to cache
    bid, ask, source = executor._get_quotes("down_tok")
    assert source == "cache", f"Expected 'cache', got '{source}'"
    assert abs(bid - 0.48) < 1e-9

    # Unknown token: should return None
    bid, ask, source = executor._get_quotes("unknown_tok")
    assert source == "none", f"Expected 'none', got '{source}'"
    assert bid is None

    print("  PASS: _get_quotes fallback chain works correctly")


def test_cached_quote_metadata():
    """CachedQuote stores mid, timestamp, source, market_id."""
    config = PaperTradingConfig(
        enabled=True, initial_balance=1000.0,
        maker_fee_bps=0, taker_fee_bps=0, slippage_bps=0,
    )
    executor = PaperOrderExecutor(config=config, clob_feed=MockClobFeed())

    executor.update_quotes(
        "token_abc", 0.40, 0.45,
        market_id="cond_42", source="market_state_up",
    )

    q = executor.get_cached_quote("token_abc")
    assert q is not None
    assert abs(q.bid - 0.40) < 1e-9
    assert abs(q.ask - 0.45) < 1e-9
    assert abs(q.mid - 0.425) < 1e-9
    assert q.market_id == "cond_42"
    assert q.source == "market_state_up"
    assert q.timestamp is not None

    print("  PASS: CachedQuote metadata correct")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Price Cache Collision & Token Mapping")
    print("=" * 60)
    print()

    print("Test 1: Cache keyed by token_id (no collision)")
    test_cache_keyed_by_token_id()
    print()

    print("Test 2: DOWN position uses DOWN token_id")
    test_down_position_uses_down_token()
    print()

    print("Test 3: _get_quotes fallback chain")
    test_get_quotes_fallback_chain()
    print()

    print("Test 4: CachedQuote metadata")
    test_cached_quote_metadata()
    print()

    print("=" * 60)
    print("All price cache tests passed!")
    print("=" * 60)
