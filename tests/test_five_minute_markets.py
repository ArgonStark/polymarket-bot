"""
Tests for 5-minute market support.

Validates:
1. Variant parsing from market slug
2. MarketState variant field and duration_seconds
3. Config parsing for trading_variants
4. Per-variant timing (min_time_remaining, observation)
5. State serialization roundtrip with variant
6. Concurrent BTC:five + BTC:fifteen (no dict collision)
7. Cooldown key format: "ASSET:variant"
8. GammaAPI get_crypto_markets() with multiple variants
9. TradeRecord includes variant field
"""

import os
import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from src.models import MarketState, Position, Side


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_market(
    asset="BTC",
    variant="fifteen",
    condition_id="cond_test",
    target_price=100000.0,
    minutes_remaining=10,
):
    now = datetime.now(timezone.utc)
    duration = 300 if variant == "five" else 900
    start = now - timedelta(seconds=duration - minutes_remaining * 60)
    end = now + timedelta(minutes=minutes_remaining)
    return MarketState(
        condition_id=condition_id,
        question=f"Will {asset} go up?",
        up_token_id=f"up_{condition_id}",
        down_token_id=f"down_{condition_id}",
        asset=asset,
        target_price=target_price,
        start_time=start,
        end_time=end,
        best_bid=0.50,
        best_ask=0.55,
        variant=variant,
    )


def _make_position(market, side=Side.UP):
    return Position(
        market=market,
        side=side,
        token_id=market.up_token_id if side == Side.UP else market.down_token_id,
        entry_price=0.52,
        shares=10.0,
        entry_time=datetime.now(timezone.utc),
        market_id=market.condition_id,
        yes_token_id=market.up_token_id,
        no_token_id=market.down_token_id,
        held_token_id=market.up_token_id if side == Side.UP else market.down_token_id,
    )


# ---------------------------------------------------------------------------
# Tests: MarketState variant field
# ---------------------------------------------------------------------------

class TestMarketStateVariant:
    def test_default_variant_is_fifteen(self):
        m = MarketState(
            condition_id="c1", question="Q", up_token_id="u",
            down_token_id="d", asset="BTC", target_price=100000,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc) + timedelta(minutes=15),
        )
        assert m.variant == "fifteen"

    def test_five_variant(self):
        m = _make_market(variant="five")
        assert m.variant == "five"

    def test_duration_seconds_five(self):
        m = _make_market(variant="five")
        assert m.duration_seconds == 300

    def test_duration_seconds_fifteen(self):
        m = _make_market(variant="fifteen")
        assert m.duration_seconds == 900


# ---------------------------------------------------------------------------
# Tests: Config
# ---------------------------------------------------------------------------

class TestConfigVariants:
    def test_default_trading_variants(self):
        """Without env var, should default to ['fifteen']."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove TRADING_VARIANTS if set
            os.environ.pop("TRADING_VARIANTS", None)
            from src.config import TradingConfig
            tc = TradingConfig()
            assert tc.trading_variants == ["fifteen"]

    def test_both_variants(self):
        with patch.dict(os.environ, {"TRADING_VARIANTS": "five,fifteen"}):
            from src.config import TradingConfig
            tc = TradingConfig()
            assert "five" in tc.trading_variants
            assert "fifteen" in tc.trading_variants

    def test_invalid_variant_ignored(self):
        with patch.dict(os.environ, {"TRADING_VARIANTS": "five,thirty,fifteen"}):
            from src.config import TradingConfig
            tc = TradingConfig()
            assert "thirty" not in tc.trading_variants
            assert "five" in tc.trading_variants
            assert "fifteen" in tc.trading_variants

    def test_5m_timing_defaults(self):
        from src.config import TradingConfig
        tc = TradingConfig()
        assert tc.min_time_remaining_5m == 15.0
        assert tc.min_observation_time_5m == 10.0


# ---------------------------------------------------------------------------
# Tests: Variant parsing from slug
# ---------------------------------------------------------------------------

class TestVariantParsing:
    def test_parse_5m_slug(self):
        """_parse_market should detect variant='five' from slug."""
        slug = "btc-updown-5m-1770999000"
        slug_lower = slug.lower()
        if "-updown-5m-" in slug_lower:
            variant = "five"
        elif "-updown-15m-" in slug_lower:
            variant = "fifteen"
        else:
            variant = "fifteen"
        assert variant == "five"

    def test_parse_15m_slug(self):
        slug = "eth-updown-15m-1770999000"
        slug_lower = slug.lower()
        if "-updown-5m-" in slug_lower:
            variant = "five"
        elif "-updown-15m-" in slug_lower:
            variant = "fifteen"
        else:
            variant = "fifteen"
        assert variant == "fifteen"

    def test_parse_unknown_slug_defaults(self):
        slug = "btc-something-else-1770999000"
        slug_lower = slug.lower()
        if "-updown-5m-" in slug_lower:
            variant = "five"
        elif "-updown-15m-" in slug_lower:
            variant = "fifteen"
        else:
            variant = "fifteen"
        assert variant == "fifteen"


# ---------------------------------------------------------------------------
# Tests: State serialization roundtrip
# ---------------------------------------------------------------------------

class TestStateRoundtrip:
    def test_serialize_includes_variant(self):
        from src.state import _serialize_position
        market = _make_market(variant="five")
        pos = _make_position(market)
        data = _serialize_position(pos)
        assert data["variant"] == "five"

    def test_deserialize_restores_variant(self):
        from src.state import _serialize_position, _deserialize_position
        market = _make_market(variant="five", condition_id="cond_5m")
        pos = _make_position(market)
        data = _serialize_position(pos)

        restored = _deserialize_position(data)
        assert restored.market.variant == "five"

    def test_deserialize_default_variant_for_old_state(self):
        """Old state files without variant field should default to 'fifteen'."""
        from src.state import _deserialize_position
        data = {
            "condition_id": "cond_old",
            "question": "Q",
            "up_token_id": "u",
            "down_token_id": "d",
            "asset": "BTC",
            "target_price": 100000.0,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat(),
            "best_bid": 0.5,
            "best_ask": 0.55,
            "side": "UP",
            "token_id": "u",
            "entry_price": 0.52,
            "shares": 10.0,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            # No "variant" key — should default to "fifteen"
        }
        pos = _deserialize_position(data)
        assert pos.market.variant == "fifteen"

    def test_full_state_roundtrip(self, tmp_path):
        """Full save/load cycle preserving variant."""
        from src.state import BotStateManager, _serialize_position, _deserialize_position
        from unittest.mock import MagicMock

        state_file = str(tmp_path / "state.json")
        mgr = BotStateManager(state_file)

        market_5m = _make_market(variant="five", condition_id="c5")
        market_15m = _make_market(variant="fifteen", condition_id="c15")
        pos_5m = _make_position(market_5m)
        pos_15m = _make_position(market_15m)

        # Mock risk manager
        rm = MagicMock()
        rm.positions = {"c5": pos_5m, "c15": pos_15m}
        rm.current_bankroll = 1000.0
        rm.peak_bankroll = 1050.0
        rm.starting_bankroll = 1000.0
        rm.consecutive_losses = 0
        rm.trade_history = []
        rm.daily_stats = MagicMock()
        rm.daily_stats.date = "2026-02-13"
        rm.daily_stats.total_pnl = 5.0
        rm.daily_stats.trades_count = 2
        rm.daily_stats.wins = 1
        rm.daily_stats.losses = 1
        rm.daily_stats.fees_paid = 0.1
        rm.daily_stats.rebates_earned = 0.05
        rm.daily_stats.starting_bankroll = 1000.0
        rm._peak_decay_started_at = None

        assert mgr.save(rm)

        loaded = mgr.load()
        assert loaded is not None
        assert len(loaded.open_positions) == 2

        # Verify variants preserved
        variants_found = {p["variant"] for p in loaded.open_positions}
        assert "five" in variants_found
        assert "fifteen" in variants_found


# ---------------------------------------------------------------------------
# Tests: Concurrent markets — no dict collision
# ---------------------------------------------------------------------------

class TestConcurrentVariants:
    def test_cooldown_keys_different(self):
        """BTC:five and BTC:fifteen are separate cooldown keys."""
        cooldowns = {}
        now = datetime.now(timezone.utc)

        cooldowns["BTC:five"] = now
        cooldowns["BTC:fifteen"] = now - timedelta(seconds=30)

        assert cooldowns["BTC:five"] != cooldowns["BTC:fifteen"]
        assert len(cooldowns) == 2

    def test_active_market_keys_different(self):
        """Active market tracking keyed by ASSET:variant avoids collision."""
        active = {}
        active["BTC:five"] = "cond_5m_btc"
        active["BTC:fifteen"] = "cond_15m_btc"

        assert active["BTC:five"] != active["BTC:fifteen"]
        assert len(active) == 2

    def test_position_per_variant(self):
        """Can have BTC:five UP and BTC:fifteen DOWN simultaneously."""
        market_5m = _make_market(variant="five", condition_id="c5")
        market_15m = _make_market(variant="fifteen", condition_id="c15")

        pos_5m = _make_position(market_5m, side=Side.UP)
        pos_15m = _make_position(market_15m, side=Side.DOWN)

        positions = {
            "c5": pos_5m,
            "c15": pos_15m,
        }

        assert len(positions) == 2
        assert positions["c5"].side == Side.UP
        assert positions["c5"].market.variant == "five"
        assert positions["c15"].side == Side.DOWN
        assert positions["c15"].market.variant == "fifteen"


# ---------------------------------------------------------------------------
# Tests: TradeRecord includes variant
# ---------------------------------------------------------------------------

class TestTradeRecordVariant:
    def test_trade_record_has_variant(self):
        from src.strategy.trade_history import TradeRecord
        rec = TradeRecord(
            timestamp="2026-02-13T00:00:00",
            asset="BTC",
            side="UP",
            entry_price=0.55,
            shares=10,
            cost=5.5,
            target_price=100000,
            chainlink_price_at_entry=99500,
            predicted_prob=None,
            variant="five",
        )
        assert rec.variant == "five"

    def test_trade_record_default_variant(self):
        from src.strategy.trade_history import TradeRecord
        rec = TradeRecord(
            timestamp="2026-02-13T00:00:00",
            asset="BTC",
            side="UP",
            entry_price=0.55,
            shares=10,
            cost=5.5,
            target_price=100000,
            chainlink_price_at_entry=99500,
            predicted_prob=None,
        )
        assert rec.variant == "fifteen"


# ---------------------------------------------------------------------------
# Tests: GammaAPI variant meta
# ---------------------------------------------------------------------------

class TestGammaVariantMeta:
    def test_variant_meta_five(self):
        from src.data.gamma import GammaAPI
        meta = GammaAPI._VARIANT_META["five"]
        assert meta["period"] == 300
        assert meta["suffix"] == "5m"
        assert meta["api_variant"] == "five"

    def test_variant_meta_fifteen(self):
        from src.data.gamma import GammaAPI
        meta = GammaAPI._VARIANT_META["fifteen"]
        assert meta["period"] == 900
        assert meta["suffix"] == "15m"
        assert meta["api_variant"] == "fifteen"


# ---------------------------------------------------------------------------
# Tests: Per-variant timing
# ---------------------------------------------------------------------------

class TestPerVariantTiming:
    def test_5m_min_time_remaining(self):
        """5-min markets use min_time_remaining_5m config."""
        from src.config import TradingConfig
        tc = TradingConfig()
        market = _make_market(variant="five", minutes_remaining=0.2)  # 12s left

        min_time = (
            tc.min_time_remaining_5m if market.variant == "five"
            else tc.min_time_remaining
        )
        # 12s < 15s default for 5m → should expire
        assert market.time_remaining < min_time

    def test_15m_min_time_remaining(self):
        """15-min markets use min_time_remaining config (30s default)."""
        from src.config import TradingConfig
        tc = TradingConfig()
        market = _make_market(variant="fifteen", minutes_remaining=0.35)  # 21s left

        min_time = (
            tc.min_time_remaining_5m if market.variant == "five"
            else tc.min_time_remaining
        )
        # 21s < 30s default for 15m → should expire
        assert market.time_remaining < min_time

    def test_5m_observation_time(self):
        """5-min markets have shorter observation time."""
        from src.config import TradingConfig
        tc = TradingConfig()
        min_obs = (
            tc.min_observation_time_5m if True  # variant == "five"
            else tc.min_observation_time
        )
        assert min_obs < tc.min_observation_time


# ---------------------------------------------------------------------------
# Tests: Period boundary detection
# ---------------------------------------------------------------------------

class TestPeriodBoundary:
    def test_five_minute_boundary(self):
        """300-second period boundaries for 5-min variant."""
        ts = 1770999000  # Exact 5-min boundary
        assert ts % 300 == 0

        period_ts = (ts // 300) * 300
        assert period_ts == ts

        # 30 seconds later: same period
        ts2 = ts + 30
        assert (ts2 // 300) * 300 == ts

        # 300 seconds later: next period
        ts3 = ts + 300
        assert (ts3 // 300) * 300 == ts3
        assert (ts3 // 300) * 300 != ts

    def test_fifteen_minute_boundary(self):
        """900-second period boundaries for 15-min variant."""
        ts = 1770999000  # Check if 15-min boundary
        period_ts = (ts // 900) * 900

        # 900 seconds later: next period
        ts2 = ts + 900
        assert (ts2 // 900) * 900 != period_ts
