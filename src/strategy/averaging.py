"""
Averaging Down Strategy Module

Implements intelligent position averaging when:
1. We have an existing position
2. The token price has improved (cheaper)
3. Chart analysis STRONGLY confirms our direction
4. Market conditions support averaging (not in reversal)

This is a MODERATE approach - only averages when very confident.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Position
    from ..data.binance_chart import ChartAnalysis

logger = logging.getLogger(__name__)


@dataclass
class AveragingConfig:
    """Configuration for averaging down strategy."""

    enabled: bool = True

    # Minimum chart confidence to average (70% = moderate, 80% = conservative)
    min_chart_confidence: float = 0.70

    # Minimum price improvement (token must be X% cheaper)
    min_price_improvement: float = 0.08  # 8% cheaper

    # Maximum times to average per position
    max_averages: int = 1  # Only average once per position

    # Minimum time remaining to average (seconds)
    min_time_remaining: int = 300  # 5 minutes

    # Minimum time between averages (seconds)
    min_time_between_averages: int = 60  # 1 minute

    # RSI bounds - don't average into extreme conditions
    rsi_min: float = 25.0  # Don't average if RSI below this (oversold, may bounce)
    rsi_max: float = 75.0  # Don't average if RSI above this (overbought, may drop)

    # Timeframe alignment requirement
    min_alignment_score: float = 0.66  # At least 2/3 timeframes agree

    # Size of averaging order relative to original position
    averaging_size_multiplier: float = 0.5  # Add 50% of original size

    # Don't average if trend is breaking
    allow_if_trend_breaking: bool = False


@dataclass
class AveragingDecision:
    """Result of averaging down analysis."""

    should_average: bool
    suggested_size_usd: float
    suggested_shares: float
    reason: str
    new_avg_price: float = 0.0  # What avg price would be after averaging
    confidence: float = 0.0  # How confident we are in this decision


def should_average_down(
    position: "Position",
    current_token_price: float,
    chart_analysis: Optional["ChartAnalysis"],
    time_remaining: float,
    config: Optional[AveragingConfig] = None,
) -> AveragingDecision:
    """
    Decide if we should average down on an existing position.

    MODERATE APPROACH:
    - Only average when chart is STRONGLY confirming our direction (70%+)
    - Token must be meaningfully cheaper (8%+)
    - RSI must not be in extreme zone (avoid reversal traps)
    - Timeframes must mostly agree (66%+)
    - Only average once per position

    Args:
        position: Existing position to potentially add to
        current_token_price: Current price of the token we'd buy
        chart_analysis: Current chart analysis
        time_remaining: Seconds until market resolution
        config: Averaging configuration

    Returns:
        AveragingDecision with recommendation
    """
    if config is None:
        config = AveragingConfig()

    if not config.enabled:
        return AveragingDecision(
            should_average=False,
            suggested_size_usd=0,
            suggested_shares=0,
            reason="Averaging disabled"
        )

    # === BASIC CHECKS ===

    # Already averaged max times?
    if position.times_averaged >= config.max_averages:
        return AveragingDecision(
            should_average=False,
            suggested_size_usd=0,
            suggested_shares=0,
            reason=f"Already averaged {position.times_averaged}x (max {config.max_averages})"
        )

    # Enough time remaining?
    if time_remaining < config.min_time_remaining:
        return AveragingDecision(
            should_average=False,
            suggested_size_usd=0,
            suggested_shares=0,
            reason=f"Only {time_remaining:.0f}s left (need {config.min_time_remaining}s)"
        )

    # Time since last average?
    if position.last_average_time:
        seconds_since = (datetime.now(timezone.utc) - position.last_average_time).total_seconds()
        if seconds_since < config.min_time_between_averages:
            return AveragingDecision(
                should_average=False,
                suggested_size_usd=0,
                suggested_shares=0,
                reason=f"Only {seconds_since:.0f}s since last average"
            )

    # === PRICE IMPROVEMENT CHECK ===

    original_price = position.original_entry_price or position.entry_price
    price_improvement = (original_price - current_token_price) / original_price

    if price_improvement < config.min_price_improvement:
        return AveragingDecision(
            should_average=False,
            suggested_size_usd=0,
            suggested_shares=0,
            reason=f"Price improvement only {price_improvement:.1%} (need {config.min_price_improvement:.0%})"
        )

    # === CHART ANALYSIS CHECKS ===

    if chart_analysis is None:
        return AveragingDecision(
            should_average=False,
            suggested_size_usd=0,
            suggested_shares=0,
            reason="No chart analysis available"
        )

    # Check chart confidence
    if chart_analysis.confidence < config.min_chart_confidence:
        return AveragingDecision(
            should_average=False,
            suggested_size_usd=0,
            suggested_shares=0,
            reason=f"Chart confidence {chart_analysis.confidence:.0%} < {config.min_chart_confidence:.0%}"
        )

    # Check chart bias matches our position
    position_is_down = position.side.value == "DOWN"
    position_is_up = position.side.value == "UP"
    chart_is_bearish = chart_analysis.bias == "bearish"
    chart_is_bullish = chart_analysis.bias == "bullish"

    if position_is_down and not chart_is_bearish:
        return AveragingDecision(
            should_average=False,
            suggested_size_usd=0,
            suggested_shares=0,
            reason=f"Position is DOWN but chart is {chart_analysis.bias}"
        )

    if position_is_up and not chart_is_bullish:
        return AveragingDecision(
            should_average=False,
            suggested_size_usd=0,
            suggested_shares=0,
            reason=f"Position is UP but chart is {chart_analysis.bias}"
        )

    # Check RSI - don't average into extremes (potential reversal)
    rsi = chart_analysis.rsi_14
    if position_is_down and rsi < config.rsi_min:
        return AveragingDecision(
            should_average=False,
            suggested_size_usd=0,
            suggested_shares=0,
            reason=f"RSI {rsi:.0f} too low for DOWN (oversold, may bounce)"
        )

    if position_is_up and rsi > config.rsi_max:
        return AveragingDecision(
            should_average=False,
            suggested_size_usd=0,
            suggested_shares=0,
            reason=f"RSI {rsi:.0f} too high for UP (overbought, may drop)"
        )

    # Check timeframe alignment
    if chart_analysis.alignment_score < config.min_alignment_score:
        return AveragingDecision(
            should_average=False,
            suggested_size_usd=0,
            suggested_shares=0,
            reason=f"Timeframe alignment {chart_analysis.alignment_score:.0%} < {config.min_alignment_score:.0%}"
        )

    # Check if trend is breaking
    if chart_analysis.trend_strength_dropping and not config.allow_if_trend_breaking:
        return AveragingDecision(
            should_average=False,
            suggested_size_usd=0,
            suggested_shares=0,
            reason="Trend is breaking - not safe to average"
        )

    # Check market uncertainty
    if chart_analysis.is_uncertain and chart_analysis.uncertainty_score > 0.5:
        return AveragingDecision(
            should_average=False,
            suggested_size_usd=0,
            suggested_shares=0,
            reason=f"Market uncertain ({chart_analysis.uncertainty_score:.0%})"
        )

    # === ALL CHECKS PASSED - CALCULATE AVERAGING ORDER ===

    # Calculate suggested size (50% of original position)
    original_cost = position.total_cost if position.total_cost > 0 else (position.entry_price * position.shares)
    suggested_size_usd = original_cost * config.averaging_size_multiplier
    suggested_shares = suggested_size_usd / current_token_price

    # Ensure minimum 5 shares (Polymarket requirement)
    if suggested_shares < 5:
        suggested_shares = 5
        suggested_size_usd = suggested_shares * current_token_price

    # Calculate new average price after averaging
    total_shares_after = position.shares + suggested_shares
    total_cost_after = original_cost + suggested_size_usd
    new_avg_price = total_cost_after / total_shares_after

    # Calculate confidence in this averaging decision
    # Higher when: better price improvement, higher chart confidence, better alignment
    confidence = (
        min(price_improvement / 0.15, 1.0) * 0.3 +  # Price improvement (max at 15%)
        chart_analysis.confidence * 0.4 +            # Chart confidence
        chart_analysis.alignment_score * 0.3         # Timeframe alignment
    )

    logger.info(
        f"📊 AVERAGING OPPORTUNITY: "
        f"{position.side.value} position | "
        f"Price improved {price_improvement:.1%} | "
        f"Chart {chart_analysis.bias} ({chart_analysis.confidence:.0%}) | "
        f"RSI={rsi:.0f} | "
        f"Alignment={chart_analysis.alignment_score:.0%} | "
        f"Avg price: ${position.entry_price:.3f} → ${new_avg_price:.3f}"
    )

    return AveragingDecision(
        should_average=True,
        suggested_size_usd=suggested_size_usd,
        suggested_shares=suggested_shares,
        reason=f"Strong {chart_analysis.bias} signal ({chart_analysis.confidence:.0%}) + {price_improvement:.0%} cheaper",
        new_avg_price=new_avg_price,
        confidence=confidence,
    )


def update_position_after_averaging(
    position: "Position",
    additional_shares: float,
    additional_cost: float,
    new_entry_price: float,
) -> None:
    """
    Update position fields after averaging down.

    Args:
        position: Position to update
        additional_shares: Shares added
        additional_cost: USD spent on additional shares
        new_entry_price: New average entry price
    """
    # Store original price if this is first average
    if position.original_entry_price is None:
        position.original_entry_price = position.entry_price

    # Update totals
    if position.total_cost == 0:
        position.total_cost = position.entry_price * position.shares
    position.total_cost += additional_cost
    position.shares += additional_shares
    position.entry_price = new_entry_price

    # Update tracking
    position.times_averaged += 1
    position.last_average_time = datetime.now(timezone.utc)

    logger.info(
        f"📈 POSITION AVERAGED: "
        f"Added {additional_shares:.1f} shares @ ${additional_cost:.2f} | "
        f"New total: {position.shares:.1f} shares | "
        f"New avg: ${position.entry_price:.3f} | "
        f"Times averaged: {position.times_averaged}"
    )
