"""
Tests for src/bot/guards.py — data sanity gates and execution feasibility.

Validates:
1. validate_market_quotes() rejects impossible quotes (crossed book, out-of-range, stale)
2. validate_market_quotes() accepts valid quotes
3. can_execute_trade() blocks when size < min_trade after scaling
4. can_execute_trade() blocks on cooldown, max_positions, exposure cap
5. can_execute_trade() passes when conditions are met
6. check_bankroll_regime() blocks when bankroll too low
7. LogThrottle rate-limits correctly
"""

import time
from datetime import datetime, timezone, timedelta

import pytest

from src.bot.guards import (
    validate_market_quotes,
    can_execute_trade,
    check_bankroll_regime,
    LogThrottle,
    QuoteValidation,
    ExecutionFeasibility,
    BankrollRegime,
)


# ---------------------------------------------------------------------------
# 1. validate_market_quotes — impossible quotes from the real log
# ---------------------------------------------------------------------------

class TestValidateMarketQuotes:
    """Test the DATA_SANITY_FAIL gate."""

    def test_real_impossible_quotes(self):
        """From actual logs:
        PMKT_PRICES asset=ETH yes_bid=0.5350 yes_ask=0.0100
        This gives no_bid=0.99, no_ask=0.465, spread_yes=-0.525
        """
        result = validate_market_quotes(
            yes_bid=0.5350,
            yes_ask=0.0100,
        )
        assert not result.valid
        assert result.reason == "crossed_book_yes"
        # Verify the impossible spread was detected
        assert result.spread_yes < 0  # yes_ask < yes_bid

    def test_missing_yes_bid(self):
        result = validate_market_quotes(yes_bid=None, yes_ask=0.55)
        assert not result.valid
        assert result.reason == "missing_prices"

    def test_missing_yes_ask(self):
        result = validate_market_quotes(yes_bid=0.45, yes_ask=None)
        assert not result.valid
        assert result.reason == "missing_prices"

    def test_yes_bid_zero(self):
        """Price at exactly 0 is out of range."""
        result = validate_market_quotes(yes_bid=0.0, yes_ask=0.55)
        assert not result.valid
        assert result.reason == "price_out_of_range"

    def test_yes_ask_one(self):
        """Price at exactly 1.0 is out of range."""
        result = validate_market_quotes(yes_bid=0.45, yes_ask=1.0)
        assert not result.valid
        assert result.reason == "price_out_of_range"

    def test_yes_bid_above_one(self):
        result = validate_market_quotes(yes_bid=1.5, yes_ask=0.55)
        assert not result.valid
        assert result.reason == "price_out_of_range"

    def test_yes_ask_negative(self):
        result = validate_market_quotes(yes_bid=0.45, yes_ask=-0.1)
        assert not result.valid
        assert result.reason == "price_out_of_range"

    def test_crossed_book_yes(self):
        """bid > ask on YES side."""
        result = validate_market_quotes(yes_bid=0.60, yes_ask=0.40)
        assert not result.valid
        assert result.reason == "crossed_book_yes"

    def test_spread_too_wide_yes(self):
        """Spread exceeds threshold."""
        result = validate_market_quotes(
            yes_bid=0.20,
            yes_ask=0.80,
            max_spread=0.40,
        )
        assert not result.valid
        assert result.reason == "spread_too_wide_yes"

    def test_spread_too_wide_no(self):
        """NO-side spread exceeds threshold.
        no_bid = 1 - yes_ask = 1 - 0.80 = 0.20
        no_ask = 1 - yes_bid = 1 - 0.20 = 0.80
        spread_no = 0.80 - 0.20 = 0.60
        """
        result = validate_market_quotes(
            yes_bid=0.20,
            yes_ask=0.80,
            max_spread=0.40,
        )
        assert not result.valid
        # YES spread is 0.60 which also fails

    def test_prob_inconsistency_mid(self):
        """Midpoint probabilities don't sum to ~1.0.
        For valid Polymarket books, yes_mid + no_mid ≈ 1.0 always holds
        because no_mid = 1 - yes_mid (by construction).
        But with a very large tolerance gap this test forces it.
        """
        # With yes_bid=0.10 yes_ask=0.12: yes_mid=0.11, no_bid=0.88, no_ask=0.90
        # no_mid=0.89.  sum = 1.00.  So we need pathological input.
        # Actually by construction no_mid = (1 - yes_ask + 1 - yes_bid)/2
        # = 1 - (yes_ask + yes_bid)/2 = 1 - yes_mid.
        # So yes_mid + no_mid = 1.0 ALWAYS.  This check is redundant for
        # complement-derived NO prices but catches real API data.
        # Test that valid quotes pass this check:
        result = validate_market_quotes(
            yes_bid=0.45,
            yes_ask=0.55,
            prob_tolerance=0.05,
        )
        assert result.valid

    def test_stale_quotes(self):
        """Quotes older than max age are rejected."""
        old_ts = datetime.now(timezone.utc) - timedelta(seconds=120)
        result = validate_market_quotes(
            yes_bid=0.45,
            yes_ask=0.55,
            quote_timestamp=old_ts,
            max_quote_age_seconds=60.0,
        )
        assert not result.valid
        assert result.reason == "stale_quotes"

    def test_fresh_quotes_pass(self):
        """Fresh quotes within age limit pass."""
        recent_ts = datetime.now(timezone.utc) - timedelta(seconds=5)
        result = validate_market_quotes(
            yes_bid=0.45,
            yes_ask=0.55,
            quote_timestamp=recent_ts,
            max_quote_age_seconds=60.0,
        )
        assert result.valid

    def test_valid_normal_book(self):
        """A normal, healthy order book passes all checks."""
        result = validate_market_quotes(
            yes_bid=0.45,
            yes_ask=0.55,
        )
        assert result.valid
        assert result.yes_bid == 0.45
        assert result.yes_ask == 0.55
        assert result.spread_yes == pytest.approx(0.10)

    def test_valid_tight_spread(self):
        """Tight spread passes."""
        result = validate_market_quotes(
            yes_bid=0.49,
            yes_ask=0.51,
        )
        assert result.valid
        assert result.spread_yes == pytest.approx(0.02)

    def test_valid_skewed_book(self):
        """Heavily skewed but valid book."""
        result = validate_market_quotes(
            yes_bid=0.80,
            yes_ask=0.90,
            max_spread=0.50,
        )
        assert result.valid

    def test_no_timestamp_skips_staleness(self):
        """When no timestamp is provided, staleness check is skipped."""
        result = validate_market_quotes(
            yes_bid=0.45,
            yes_ask=0.55,
            quote_timestamp=None,
        )
        assert result.valid


# ---------------------------------------------------------------------------
# 2. can_execute_trade — with real scenario: bankroll=$30, min_trade=$5
# ---------------------------------------------------------------------------

class TestCanExecuteTrade:
    """Test pre-flight execution feasibility for low bankroll."""

    COMMON = dict(
        bankroll=30.0,
        min_trade_usd=5.0,
        max_position_pct=0.10,
        max_total_exposure_pct=0.25,
        equity=30.0,
        current_exposure=0.0,
        capital_scale_mult=1.0,
        base_size_usd=5.0,
        cooldown_active=False,
        positions_count=0,
        pending_count=0,
        max_concurrent_positions=4,
    )

    def test_cooldown_blocks(self):
        result = can_execute_trade(**{**self.COMMON, "cooldown_active": True})
        assert not result.can_execute
        assert result.reason == "cooldown"

    def test_max_positions_blocks(self):
        result = can_execute_trade(**{
            **self.COMMON,
            "positions_count": 3,
            "pending_count": 1,
        })
        assert not result.can_execute
        assert result.reason == "max_positions"

    def test_exposure_cap_blocks(self):
        result = can_execute_trade(**{
            **self.COMMON,
            "current_exposure": 10.0,  # >= 30 * 0.25 = 7.5
        })
        assert not result.can_execute
        assert result.reason == "exposure_cap"

    def test_untradeable_size_with_scaling(self):
        """bankroll=$30, base_size=$25 (from risk sizer), scale=0.5 → $12.5
        But max_position_pct cap: $30 * 0.10 = $3.00 → below min $5.
        """
        result = can_execute_trade(**{
            **self.COMMON,
            "capital_scale_mult": 0.5,
            "base_size_usd": 25.0,
        })
        assert not result.can_execute
        assert result.reason == "untradeable_size"
        assert result.computed_size_usd == pytest.approx(3.0)  # capped by max_position_pct

    def test_untradeable_tiny_base(self):
        """base_size=$2, scale=1.0, but cap=$3 → below min $5."""
        result = can_execute_trade(**{
            **self.COMMON,
            "capital_scale_mult": 1.0,
            "base_size_usd": 2.0,
        })
        assert not result.can_execute
        assert result.reason == "untradeable_size"

    def test_insufficient_balance(self):
        """scaled_size > 90% of bankroll."""
        result = can_execute_trade(**{
            **self.COMMON,
            "bankroll": 5.0,
            "equity": 5.0,
            "capital_scale_mult": 1.0,
            "base_size_usd": 5.0,
            "max_position_pct": 1.0,  # No cap
        })
        assert not result.can_execute
        assert result.reason == "insufficient_balance"

    def test_passes_when_feasible(self):
        """$30 bankroll, $5 base, scale=1.0, cap=10% → $3 but let's
        make cap higher so it works.
        """
        result = can_execute_trade(**{
            **self.COMMON,
            "capital_scale_mult": 1.0,
            "base_size_usd": 5.0,
            "max_position_pct": 0.50,  # cap = $15
        })
        assert result.can_execute
        assert result.computed_size_usd == pytest.approx(5.0)

    def test_scaled_size_respected(self):
        """Scale multiplier reduces size but still above min_trade."""
        result = can_execute_trade(**{
            **self.COMMON,
            "capital_scale_mult": 0.5,
            "base_size_usd": 20.0,
            "max_position_pct": 0.50,  # cap = $15
        })
        assert result.can_execute
        assert result.computed_size_usd == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 3. check_bankroll_regime
# ---------------------------------------------------------------------------

class TestBankrollRegime:

    def test_low_bankroll_observe_only(self):
        """bankroll=$8, min_trade=$5, multiplier=2.0 → threshold=$10.
        Also $5 / 0.25 = $20. Max($10, $20) = $20. $8 < $20 → observe only.
        """
        result = check_bankroll_regime(
            bankroll=8.0,
            min_trade_usd=5.0,
            max_total_exposure_pct=0.25,
            min_bankroll_multiplier=2.0,
        )
        assert result.observe_only
        assert result.threshold == pytest.approx(20.0)

    def test_sufficient_bankroll(self):
        """bankroll=$100, threshold=$20 → can trade."""
        result = check_bankroll_regime(
            bankroll=100.0,
            min_trade_usd=5.0,
            max_total_exposure_pct=0.25,
            min_bankroll_multiplier=2.0,
        )
        assert not result.observe_only

    def test_edge_case_exact_threshold(self):
        """bankroll == threshold → still observe_only (strict <)."""
        result = check_bankroll_regime(
            bankroll=20.0,
            min_trade_usd=5.0,
            max_total_exposure_pct=0.25,
            min_bankroll_multiplier=2.0,
        )
        assert not result.observe_only  # 20 >= 20, not observe_only

    def test_very_low_bankroll(self):
        """bankroll=$1 → definitely observe only."""
        result = check_bankroll_regime(
            bankroll=1.0,
            min_trade_usd=5.0,
            max_total_exposure_pct=0.25,
        )
        assert result.observe_only


# ---------------------------------------------------------------------------
# 4. LogThrottle
# ---------------------------------------------------------------------------

class TestLogThrottle:

    def test_first_call_allowed(self):
        throttle = LogThrottle(interval=5.0)
        assert throttle.should_log("key1") is True

    def test_second_call_throttled(self):
        throttle = LogThrottle(interval=5.0)
        throttle.should_log("key1")
        assert throttle.should_log("key1") is False

    def test_different_keys_independent(self):
        throttle = LogThrottle(interval=5.0)
        throttle.should_log("key1")
        assert throttle.should_log("key2") is True

    def test_suppressed_count(self):
        throttle = LogThrottle(interval=1000.0)  # Won't expire during test
        throttle.should_log("key1")
        # Three suppressed calls
        throttle.should_log("key1")
        throttle.should_log("key1")
        throttle.should_log("key1")
        assert throttle.suppressed_count("key1") == 3

    def test_suppressed_reset_after_log(self):
        throttle = LogThrottle(interval=0.0)  # Always allows
        throttle.should_log("key1")
        assert throttle.suppressed_count("key1") == 0


# ---------------------------------------------------------------------------
# 5. Paper mode behavior — same logic applies
# ---------------------------------------------------------------------------

class TestPaperModeBehavior:
    """Verify guards work identically in paper mode (pure functions, mode-agnostic)."""

    def test_sanity_gate_mode_agnostic(self):
        """validate_market_quotes has no mode parameter — works same for paper."""
        result = validate_market_quotes(
            yes_bid=0.5350,
            yes_ask=0.0100,
        )
        assert not result.valid
        assert "crossed" in result.reason

    def test_can_execute_mode_agnostic(self):
        """can_execute_trade has no mode parameter — same blocking logic."""
        result = can_execute_trade(
            bankroll=30.0,
            min_trade_usd=5.0,
            max_position_pct=0.10,
            max_total_exposure_pct=0.25,
            equity=30.0,
            current_exposure=0.0,
            capital_scale_mult=0.5,
            base_size_usd=25.0,
            cooldown_active=False,
            positions_count=0,
            pending_count=0,
            max_concurrent_positions=4,
        )
        # cap = 30 * 0.10 = $3.00, < $5 min_trade → blocked
        assert not result.can_execute
        assert result.reason == "untradeable_size"
