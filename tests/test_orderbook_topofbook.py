"""Tests for the orderbook top-of-book fix.

Regression: on fast 5m markets a stale deep level (e.g. a phantom 0.99 bid)
lingered in the accumulated level list, so best_bid = max(levels) returned 0.99
and the book looked "crossed" (bid 0.99 > ask 0.51), failing the sanity gate
and making the market untradeable. The feed reports authoritative best_bid/ask
in price_change / best_bid_ask messages — we now prefer those.
"""

import pytest

from src.models import OrderBook, OrderBookLevel


def test_feed_value_preferred_over_stale_level():
    ob = OrderBook(
        token_id="t",
        bids=[OrderBookLevel(price=0.99, size=1.0),   # stale phantom level
              OrderBookLevel(price=0.50, size=200.0)],
        asks=[OrderBookLevel(price=0.51, size=200.0)],
        feed_best_bid=0.50,   # authoritative top-of-book from the feed
        feed_best_ask=0.51,
    )
    assert ob.best_bid == 0.50          # not 0.99
    assert ob.best_ask == 0.51
    assert ob.spread == pytest.approx(0.01)   # not crossed


def test_falls_back_to_levels_when_no_feed_value():
    ob = OrderBook(
        token_id="t",
        bids=[OrderBookLevel(price=0.50, size=200.0)],
        asks=[OrderBookLevel(price=0.51, size=200.0)],
    )
    assert ob.best_bid == 0.50
    assert ob.best_ask == 0.51


def test_feed_ask_out_of_range_ignored():
    # A bogus feed ask (>=1) must not be trusted; fall back to level data.
    ob = OrderBook(
        token_id="t",
        bids=[OrderBookLevel(price=0.50, size=1.0)],
        asks=[OrderBookLevel(price=0.55, size=1.0)],
        feed_best_ask=1.0,
    )
    assert ob.best_ask == 0.55


def test_price_change_updates_feed_topofbook():
    # Exercise the handler: a stale 0.99 bid is present, then a price_change
    # arrives carrying the authoritative best_bid=0.50 → best_bid must be 0.50.
    from src.data.clob import CLOBFeed
    from src.config import BotConfig
    feed = CLOBFeed(config=BotConfig.from_env())
    feed._orderbooks["tok"] = OrderBook(
        token_id="tok",
        bids=[OrderBookLevel(price=0.99, size=1.0)],
        asks=[OrderBookLevel(price=0.51, size=1.0)],
    )
    feed._handle_price_change({
        "price_changes": [
            {"asset_id": "tok", "price": "0.50", "size": "200",
             "side": "BUY", "best_bid": "0.50", "best_ask": "0.51"},
        ]
    })
    ob = feed.get_orderbook("tok")
    assert ob.best_bid == 0.50
    assert ob.best_ask == 0.51
