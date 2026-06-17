"""Regression tests: empty book sides must not leave phantom stale levels.

Scenario from live trading (June 2026): an UP token collapsing toward zero
reports best_bid=0 / best_ask=0.01. The old handlers skipped cleanup when
best_bid was 0, so synthetic/stale bid levels (e.g. 0.32) lingered and
produced phantom crossed books (yes_bid=0.32 > yes_ask=0.01), blocking
every market with DATA_SANITY_FAIL crossed_book_yes.
"""
from src.config import BotConfig
from src.data.clob import CLOBFeed

TOKEN = "tok_test_1"


def make_feed_with_book(bids, asks):
    feed = CLOBFeed(config=BotConfig.from_env())
    feed._handle_book_snapshot({
        "asset_id": TOKEN,
        "bids": [{"price": str(p), "size": str(s)} for p, s in bids],
        "asks": [{"price": str(p), "size": str(s)} for p, s in asks],
    })
    return feed


class TestBestBidAskEmptySides:
    def test_zero_best_bid_clears_stale_bids(self):
        feed = make_feed_with_book(bids=[(0.30, 100), (0.32, 50)], asks=[(0.34, 80)])
        # Market collapses: feed reports no bids, ask at 0.01
        feed._handle_best_bid_ask({
            "event_type": "best_bid_ask", "asset_id": TOKEN,
            "best_bid": "0", "best_ask": "0.01",
        })
        ob = feed.get_orderbook(TOKEN)
        assert ob.best_bid is None          # no phantom 0.32 bid
        assert ob.best_ask == 0.01
        assert ob.bids == []

    def test_missing_fields_do_not_wipe_book(self):
        feed = make_feed_with_book(bids=[(0.50, 10)], asks=[(0.52, 10)])
        feed._handle_best_bid_ask({
            "event_type": "best_bid_ask", "asset_id": TOKEN,
            # no best_bid / best_ask fields at all
        })
        ob = feed.get_orderbook(TOKEN)
        assert ob.best_bid == 0.50
        assert ob.best_ask == 0.52

    def test_ask_out_of_range_clears_asks(self):
        feed = make_feed_with_book(bids=[(0.97, 10)], asks=[(0.99, 10)])
        feed._handle_best_bid_ask({
            "event_type": "best_bid_ask", "asset_id": TOKEN,
            "best_bid": "0.98", "best_ask": "1",
        })
        ob = feed.get_orderbook(TOKEN)
        assert ob.best_bid == 0.98
        assert ob.best_ask is None


class TestPriceChangeEmptySides:
    def test_zero_best_bid_in_price_change_clears_bids(self):
        feed = make_feed_with_book(bids=[(0.30, 100)], asks=[(0.34, 80)])
        feed._handle_price_change({
            "event_type": "price_change",
            "price_changes": [{
                "asset_id": TOKEN, "price": "0.01", "size": "500",
                "side": "SELL", "best_bid": "0", "best_ask": "0.01",
            }],
        })
        ob = feed.get_orderbook(TOKEN)
        assert ob.best_bid is None          # stale 0.30 bid gone
        assert ob.best_ask == 0.01
        # Book is degenerate but NOT crossed
        assert not (ob.best_bid is not None and ob.best_bid > ob.best_ask)

    def test_healthy_price_change_still_updates(self):
        feed = make_feed_with_book(bids=[(0.50, 10)], asks=[(0.52, 10)])
        feed._handle_price_change({
            "event_type": "price_change",
            "price_changes": [{
                "asset_id": TOKEN, "price": "0.51", "size": "20",
                "side": "BUY", "best_bid": "0.51", "best_ask": "0.52",
            }],
        })
        ob = feed.get_orderbook(TOKEN)
        assert ob.best_bid == 0.51
        assert ob.best_ask == 0.52
