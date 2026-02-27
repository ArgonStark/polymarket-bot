"""Pre-trade guards: data sanity validation and execution feasibility checks.

These functions are pure (no side effects) and return structured results
so callers can decide how to log/act.
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QuoteValidation:
    """Result of validate_market_quotes()."""
    valid: bool
    reason: str = ""           # machine-parseable tag (e.g. "crossed_book")
    detail: str = ""           # human-readable detail for logging
    # Raw values captured for the log line
    yes_bid: float = 0.0
    yes_ask: float = 0.0
    no_bid: float = 0.0
    no_ask: float = 0.0
    spread_yes: float = 0.0
    spread_no: float = 0.0
    prob_sum_mid: float = 0.0


@dataclass
class ExecutionFeasibility:
    """Result of can_execute_trade()."""
    can_execute: bool
    reason: str = ""           # machine-parseable tag
    detail: str = ""           # human-readable
    computed_size_usd: float = 0.0
    min_trade_usd: float = 0.0


@dataclass
class BankrollRegime:
    """Result of check_bankroll_regime()."""
    observe_only: bool
    reason: str = ""
    bankroll: float = 0.0
    threshold: float = 0.0


# ---------------------------------------------------------------------------
# 1. Market data sanity gate
# ---------------------------------------------------------------------------

def validate_market_quotes(
    *,
    yes_bid: Optional[float],
    yes_ask: Optional[float],
    max_spread: float = 0.50,
    prob_tolerance: float = 0.05,
    quote_timestamp: Optional[datetime] = None,
    max_quote_age_seconds: float = 60.0,
) -> QuoteValidation:
    """Hard sanity gate on market quotes before ANY EV/ML decision.

    Checks (in order):
    1. All prices present and within (0, 1) exclusive
    2. Book not crossed: yes_bid <= yes_ask and no_bid <= no_ask
    3. Spreads within threshold
    4. Probability consistency: mid prices and ask/bid complements ≈ 1.0
    5. Quote freshness

    Returns QuoteValidation with valid=True if all checks pass.
    """
    # --- Missing / out-of-range prices ---
    if yes_bid is None or yes_ask is None:
        return QuoteValidation(
            valid=False,
            reason="missing_prices",
            detail=f"yes_bid={yes_bid} yes_ask={yes_ask}",
        )

    if not (0 < yes_bid < 1) or not (0 < yes_ask < 1):
        return QuoteValidation(
            valid=False,
            reason="price_out_of_range",
            detail=f"yes_bid={yes_bid:.4f} yes_ask={yes_ask:.4f}",
            yes_bid=yes_bid,
            yes_ask=yes_ask,
        )

    # Derive DOWN token prices (complement)
    no_bid = 1.0 - yes_ask
    no_ask = 1.0 - yes_bid

    if not (0 < no_bid < 1) or not (0 < no_ask < 1):
        return QuoteValidation(
            valid=False,
            reason="derived_price_out_of_range",
            detail=f"no_bid={no_bid:.4f} no_ask={no_ask:.4f} (from yes_bid={yes_bid:.4f} yes_ask={yes_ask:.4f})",
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
        )

    # --- Crossed book ---
    spread_yes = yes_ask - yes_bid
    spread_no = no_ask - no_bid

    if yes_bid > yes_ask:
        return QuoteValidation(
            valid=False,
            reason="crossed_book_yes",
            detail=f"yes_bid={yes_bid:.4f} > yes_ask={yes_ask:.4f} spread={spread_yes:.4f}",
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            spread_yes=spread_yes,
            spread_no=spread_no,
        )

    if no_bid > no_ask:
        return QuoteValidation(
            valid=False,
            reason="crossed_book_no",
            detail=f"no_bid={no_bid:.4f} > no_ask={no_ask:.4f} spread={spread_no:.4f}",
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            spread_yes=spread_yes,
            spread_no=spread_no,
        )

    # --- Spread too wide ---
    if spread_yes > max_spread:
        return QuoteValidation(
            valid=False,
            reason="spread_too_wide_yes",
            detail=f"spread_yes={spread_yes:.4f} > max={max_spread:.4f}",
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            spread_yes=spread_yes,
            spread_no=spread_no,
        )

    if spread_no > max_spread:
        return QuoteValidation(
            valid=False,
            reason="spread_too_wide_no",
            detail=f"spread_no={spread_no:.4f} > max={max_spread:.4f}",
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            spread_yes=spread_yes,
            spread_no=spread_no,
        )

    # --- Probability consistency ---
    yes_mid = (yes_bid + yes_ask) / 2
    no_mid = (no_bid + no_ask) / 2
    prob_sum_mid = yes_mid + no_mid

    if abs(prob_sum_mid - 1.0) > prob_tolerance:
        return QuoteValidation(
            valid=False,
            reason="prob_inconsistency_mid",
            detail=f"yes_mid={yes_mid:.4f} + no_mid={no_mid:.4f} = {prob_sum_mid:.4f} (tol={prob_tolerance:.4f})",
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            spread_yes=spread_yes,
            spread_no=spread_no,
            prob_sum_mid=prob_sum_mid,
        )

    # Cross-check: yes_ask + no_bid should ≈ 1.0  (cost to buy yes + cost to sell yes)
    ask_bid_sum = yes_ask + no_bid
    if abs(ask_bid_sum - 1.0) > prob_tolerance:
        return QuoteValidation(
            valid=False,
            reason="prob_inconsistency_cross",
            detail=f"yes_ask={yes_ask:.4f} + no_bid={no_bid:.4f} = {ask_bid_sum:.4f} (tol={prob_tolerance:.4f})",
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            spread_yes=spread_yes,
            spread_no=spread_no,
            prob_sum_mid=prob_sum_mid,
        )

    # --- Quote staleness ---
    if quote_timestamp is not None:
        age_seconds = (datetime.now(timezone.utc) - quote_timestamp).total_seconds()
        if age_seconds > max_quote_age_seconds:
            return QuoteValidation(
                valid=False,
                reason="stale_quotes",
                detail=f"quote_age={age_seconds:.0f}s > max={max_quote_age_seconds:.0f}s",
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=no_bid,
                no_ask=no_ask,
                spread_yes=spread_yes,
                spread_no=spread_no,
                prob_sum_mid=prob_sum_mid,
            )

    # --- All checks passed ---
    return QuoteValidation(
        valid=True,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        spread_yes=spread_yes,
        spread_no=spread_no,
        prob_sum_mid=prob_sum_mid,
    )


# ---------------------------------------------------------------------------
# 2. Pre-flight execution feasibility
# ---------------------------------------------------------------------------

def can_execute_trade(
    *,
    bankroll: float,
    min_trade_usd: float,
    max_position_pct: float,
    max_total_exposure_pct: float,
    equity: float,
    current_exposure: float,
    capital_scale_mult: float,
    base_size_usd: float,
    cooldown_active: bool,
    positions_count: int,
    pending_count: int,
    max_concurrent_positions: int,
    capital_buffer: float = 0.90,
) -> ExecutionFeasibility:
    """Check whether a trade can actually execute BEFORE emitting decision=TRADE.

    This avoids logging decision=TRADE only to have it blocked downstream.
    """
    # --- Cooldown ---
    if cooldown_active:
        return ExecutionFeasibility(
            can_execute=False,
            reason="cooldown",
            detail="asset/variant cooldown active",
        )

    # --- Max positions ---
    total_positions = positions_count + pending_count
    if total_positions >= max_concurrent_positions:
        return ExecutionFeasibility(
            can_execute=False,
            reason="max_positions",
            detail=f"positions={positions_count}+{pending_count}={total_positions} >= max={max_concurrent_positions}",
        )

    # --- Exposure cap ---
    max_exposure = equity * max_total_exposure_pct
    if current_exposure >= max_exposure:
        return ExecutionFeasibility(
            can_execute=False,
            reason="exposure_cap",
            detail=f"exposure={current_exposure:.2f} >= max={max_exposure:.2f}",
        )

    # --- Size after capital scaling ---
    scaled_size = base_size_usd * capital_scale_mult

    # Apply max_position_pct cap
    cap = bankroll * max_position_pct
    if cap > 0 and scaled_size > cap:
        scaled_size = cap

    # --- Below min trade ---
    # If capital scaler (not position cap) pushed a viable trade below min,
    # floor at min_trade_usd so the bot can still trade at reduced size.
    # The scaler reduces sizing; the kill switch handles full halts.
    if scaled_size < min_trade_usd:
        size_without_scaler = min(base_size_usd, cap) if cap > 0 else base_size_usd
        if capital_scale_mult < 1.0 and size_without_scaler >= min_trade_usd:
            # Capital scaler is the specific cause — floor at min_trade_usd
            scaled_size = min_trade_usd
        else:
            return ExecutionFeasibility(
                can_execute=False,
                reason="untradeable_size",
                detail=(
                    f"size={scaled_size:.2f} < min_trade={min_trade_usd:.2f} "
                    f"(base={base_size_usd:.2f} scale={capital_scale_mult:.2f} "
                    f"cap={cap:.2f} bankroll={bankroll:.2f})"
                ),
                computed_size_usd=scaled_size,
                min_trade_usd=min_trade_usd,
            )

    # --- Insufficient balance ---
    available = bankroll * capital_buffer
    if scaled_size > available:
        return ExecutionFeasibility(
            can_execute=False,
            reason="insufficient_balance",
            detail=f"size={scaled_size:.2f} > available={available:.2f} (bankroll={bankroll:.2f})",
            computed_size_usd=scaled_size,
            min_trade_usd=min_trade_usd,
        )

    return ExecutionFeasibility(
        can_execute=True,
        computed_size_usd=scaled_size,
        min_trade_usd=min_trade_usd,
    )


# ---------------------------------------------------------------------------
# 3. Bankroll regime guard
# ---------------------------------------------------------------------------

def check_bankroll_regime(
    *,
    bankroll: float,
    min_trade_usd: float,
    max_total_exposure_pct: float,
    min_bankroll_multiplier: float = 2.0,
) -> BankrollRegime:
    """Determine if bankroll is too low to trade.

    Thresholds (whichever is higher):
      - min_trade_usd * min_bankroll_multiplier  (e.g. $5 * 2 = $10)
      - min_trade_usd / max_total_exposure_pct   (e.g. $5 / 0.25 = $20)

    If bankroll < threshold, market switches to OBSERVE_ONLY.
    """
    threshold_a = min_trade_usd * min_bankroll_multiplier
    threshold_b = min_trade_usd / max_total_exposure_pct if max_total_exposure_pct > 0 else threshold_a
    threshold = max(threshold_a, threshold_b)

    if bankroll < threshold:
        return BankrollRegime(
            observe_only=True,
            reason=f"bankroll={bankroll:.2f} < threshold={threshold:.2f} (min_trade={min_trade_usd:.2f})",
            bankroll=bankroll,
            threshold=threshold,
        )

    return BankrollRegime(
        observe_only=False,
        bankroll=bankroll,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# 4. Log throttle helper
# ---------------------------------------------------------------------------

class LogThrottle:
    """Rate-limits log messages by key. Thread-safe enough for single-loop bots."""

    def __init__(self, interval: float = 5.0, max_keys: int = 500):
        self._interval = interval
        self._max_keys = max_keys
        self._last_ts: dict[str, float] = {}
        self._suppressed: dict[str, int] = {}

    def should_log(self, key: str) -> bool:
        """Return True if this key hasn't been logged within the interval."""
        now = time.monotonic()
        last = self._last_ts.get(key, 0.0)
        if now - last < self._interval:
            self._suppressed[key] = self._suppressed.get(key, 0) + 1
            return False

        self._last_ts[key] = now
        self._suppressed.pop(key, None)

        # Evict old keys
        if len(self._last_ts) > self._max_keys:
            cutoff = now - self._interval * 10
            self._last_ts = {k: v for k, v in self._last_ts.items() if v > cutoff}
            self._suppressed = {k: v for k, v in self._suppressed.items() if k in self._last_ts}

        return True

    def suppressed_count(self, key: str) -> int:
        """Number of suppressed messages since last log for this key."""
        return self._suppressed.get(key, 0)
