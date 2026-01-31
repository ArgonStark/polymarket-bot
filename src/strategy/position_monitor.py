"""
Position Monitoring System

Continuously monitors open positions using 15m candlestick data
to make intelligent decisions:
- HOLD: Keep position, conditions still favorable
- CLOSE: Exit position, conditions have changed
- DCA: Average down, conditions strongly confirm our direction
- ADD: Add to winning position (pyramid)

This runs independently of new signal generation.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Position, MarketState
    from ..data.binance_chart import ChartAnalysis

logger = logging.getLogger(__name__)


class PositionAction(Enum):
    """Recommended action for an open position."""
    HOLD = "HOLD"      # Keep position as is
    CLOSE = "CLOSE"    # Exit position (sell)
    DCA = "DCA"        # Average down (buy more at lower price)
    ADD = "ADD"        # Add to winning position


@dataclass
class PositionDecision:
    """Decision about what to do with a position."""
    action: PositionAction
    reason: str
    confidence: float  # 0-1 how confident in this decision
    suggested_size: float = 0.0  # For DCA/ADD: how much to add
    urgency: str = "normal"  # "low", "normal", "high", "critical"


@dataclass
class PositionMonitorConfig:
    """Configuration for position monitoring."""

    # How often to check positions (seconds)
    check_interval: int = 15

    # Trend reversal thresholds
    reversal_confidence_threshold: float = 0.70  # Close if reversal detected with 70%+ confidence

    # DCA thresholds (from averaging.py)
    dca_min_chart_confidence: float = 0.70
    dca_min_price_improvement: float = 0.08
    dca_max_times: int = 1

    # Close thresholds
    close_on_trend_reversal: bool = True
    close_on_extreme_rsi: bool = True
    rsi_extreme_low: float = 15.0   # Close DOWN position if RSI this low
    rsi_extreme_high: float = 85.0  # Close UP position if RSI this high

    # Time-based decisions
    close_if_time_low: bool = True
    min_time_to_hold: int = 120  # Close if <2 min left and not winning


class PositionMonitor:
    """
    Monitors open positions and recommends actions based on 15m chart analysis.
    """

    def __init__(self, config: Optional[PositionMonitorConfig] = None):
        self.config = config or PositionMonitorConfig()
        self._last_check: dict[str, datetime] = {}  # asset -> last check time

    def should_check(self, asset: str) -> bool:
        """Check if enough time has passed since last check."""
        now = datetime.now(timezone.utc)
        last = self._last_check.get(asset)

        if last is None:
            return True

        elapsed = (now - last).total_seconds()
        return elapsed >= self.config.check_interval

    def analyze_position(
        self,
        position: "Position",
        chart_analysis: Optional["ChartAnalysis"],
        current_token_price: float,
        time_remaining: float,
    ) -> PositionDecision:
        """
        Analyze an open position and recommend an action.

        Args:
            position: The open position to analyze
            chart_analysis: Current 15m chart analysis
            current_token_price: Current price of the token we're holding
            time_remaining: Seconds until market resolution

        Returns:
            PositionDecision with recommended action
        """
        asset = position.market.asset
        self._last_check[asset] = datetime.now(timezone.utc)

        if chart_analysis is None:
            return PositionDecision(
                action=PositionAction.HOLD,
                reason="No chart data available",
                confidence=0.3,
            )

        position_is_down = position.side.value == "DOWN"
        position_is_up = position.side.value == "UP"

        # Get key metrics
        chart_bias = chart_analysis.bias
        chart_confidence = chart_analysis.confidence
        rsi = chart_analysis.rsi_14
        trend_change = chart_analysis.trend_change.value
        is_uncertain = chart_analysis.is_uncertain
        uncertainty_score = chart_analysis.uncertainty_score
        alignment_score = chart_analysis.alignment_score
        trend_breaking = chart_analysis.trend_strength_dropping

        # Calculate P&L
        entry_price = position.original_entry_price or position.entry_price
        pnl_pct = (entry_price - current_token_price) / entry_price if entry_price > 0 else 0
        is_profitable = pnl_pct > 0

        # === CHECK FOR CLOSE CONDITIONS ===

        # 1. TREND REVERSAL - Chart now says opposite of our position
        if self.config.close_on_trend_reversal:
            if position_is_down and chart_bias == "bullish" and chart_confidence >= self.config.reversal_confidence_threshold:
                return PositionDecision(
                    action=PositionAction.CLOSE,
                    reason=f"Trend reversal: Chart now BULLISH ({chart_confidence:.0%}) against DOWN position",
                    confidence=chart_confidence,
                    urgency="high",
                )

            if position_is_up and chart_bias == "bearish" and chart_confidence >= self.config.reversal_confidence_threshold:
                return PositionDecision(
                    action=PositionAction.CLOSE,
                    reason=f"Trend reversal: Chart now BEARISH ({chart_confidence:.0%}) against UP position",
                    confidence=chart_confidence,
                    urgency="high",
                )

        # 2. EXPLICIT REVERSAL SIGNAL
        if trend_change == "bullish_reversal" and position_is_down:
            return PositionDecision(
                action=PositionAction.CLOSE,
                reason="Bullish reversal detected - closing DOWN position",
                confidence=0.75,
                urgency="high",
            )

        if trend_change == "bearish_reversal" and position_is_up:
            return PositionDecision(
                action=PositionAction.CLOSE,
                reason="Bearish reversal detected - closing UP position",
                confidence=0.75,
                urgency="high",
            )

        # 3. EXTREME RSI - Mean reversion likely
        if self.config.close_on_extreme_rsi:
            if position_is_down and rsi < self.config.rsi_extreme_low:
                return PositionDecision(
                    action=PositionAction.CLOSE,
                    reason=f"RSI extremely oversold ({rsi:.0f}) - DOWN position at bounce risk",
                    confidence=0.65,
                    urgency="normal",
                )

            if position_is_up and rsi > self.config.rsi_extreme_high:
                return PositionDecision(
                    action=PositionAction.CLOSE,
                    reason=f"RSI extremely overbought ({rsi:.0f}) - UP position at pullback risk",
                    confidence=0.65,
                    urgency="normal",
                )

        # 4. LOW TIME + NOT WINNING - Cut losses
        if self.config.close_if_time_low and time_remaining < self.config.min_time_to_hold:
            if not is_profitable:
                return PositionDecision(
                    action=PositionAction.CLOSE,
                    reason=f"Only {time_remaining:.0f}s left and position underwater - cutting loss",
                    confidence=0.60,
                    urgency="critical",
                )

        # 5. HIGH UNCERTAINTY - Consider closing
        if is_uncertain and uncertainty_score >= 0.7:
            return PositionDecision(
                action=PositionAction.CLOSE,
                reason=f"High market uncertainty ({uncertainty_score:.0%}) - protecting position",
                confidence=uncertainty_score * 0.8,
                urgency="normal",
            )

        # === CHECK FOR DCA CONDITIONS ===

        # Only DCA if we haven't already and conditions are right
        if position.times_averaged < self.config.dca_max_times:
            # Price must have improved (token cheaper)
            price_improvement = (entry_price - current_token_price) / entry_price

            # Chart must strongly confirm our direction
            chart_confirms = (
                (position_is_down and chart_bias == "bearish") or
                (position_is_up and chart_bias == "bullish")
            )

            # RSI must not be extreme (would indicate reversal)
            rsi_safe = (
                (position_is_down and rsi > 25) or  # Not too oversold
                (position_is_up and rsi < 75)       # Not too overbought
            )

            if (price_improvement >= self.config.dca_min_price_improvement and
                chart_confirms and
                chart_confidence >= self.config.dca_min_chart_confidence and
                alignment_score >= 0.66 and
                not trend_breaking and
                rsi_safe and
                time_remaining >= 300):

                # Calculate suggested size (50% of original)
                original_cost = position.total_cost if position.total_cost > 0 else (position.entry_price * position.shares)
                suggested_size = original_cost * 0.5

                return PositionDecision(
                    action=PositionAction.DCA,
                    reason=f"Strong {chart_bias} signal ({chart_confidence:.0%}) + {price_improvement:.0%} cheaper - DCA opportunity",
                    confidence=min(chart_confidence, alignment_score),
                    suggested_size=suggested_size,
                    urgency="normal",
                )

        # === CHECK FOR ADD CONDITIONS (pyramid into winners) ===

        # Add to winning positions when very strong confirmation
        if is_profitable and pnl_pct >= 0.10:  # At least 10% in profit
            chart_confirms = (
                (position_is_down and chart_bias == "bearish") or
                (position_is_up and chart_bias == "bullish")
            )

            if (chart_confirms and
                chart_confidence >= 0.80 and
                alignment_score >= 0.80 and
                not trend_breaking and
                time_remaining >= 300):

                original_cost = position.total_cost if position.total_cost > 0 else (position.entry_price * position.shares)
                suggested_size = original_cost * 0.25  # Add 25% to winners

                return PositionDecision(
                    action=PositionAction.ADD,
                    reason=f"Winning position ({pnl_pct:.0%} profit) + very strong {chart_bias} ({chart_confidence:.0%}) - adding",
                    confidence=chart_confidence,
                    suggested_size=suggested_size,
                    urgency="low",
                )

        # === DEFAULT: HOLD ===

        # Determine hold confidence based on alignment with our position
        if (position_is_down and chart_bias == "bearish") or (position_is_up and chart_bias == "bullish"):
            hold_confidence = chart_confidence
            hold_reason = f"Chart confirms {position.side.value} ({chart_confidence:.0%})"
        elif chart_bias == "neutral":
            hold_confidence = 0.5
            hold_reason = "Chart neutral - holding"
        else:
            hold_confidence = 1 - chart_confidence
            hold_reason = f"Chart weak {chart_bias} ({chart_confidence:.0%}) - monitoring"

        return PositionDecision(
            action=PositionAction.HOLD,
            reason=hold_reason,
            confidence=hold_confidence,
        )

    def get_position_summary(
        self,
        position: "Position",
        chart_analysis: Optional["ChartAnalysis"],
        current_token_price: float,
    ) -> str:
        """
        Get a human-readable summary of position status.

        Returns:
            Summary string for logging
        """
        if chart_analysis is None:
            return f"{position.market.asset} {position.side.value}: No chart data"

        entry_price = position.original_entry_price or position.entry_price
        pnl_pct = (entry_price - current_token_price) / entry_price * 100
        pnl_sign = "+" if pnl_pct > 0 else ""

        return (
            f"{position.market.asset} {position.side.value}: "
            f"Entry ${entry_price:.3f} → ${current_token_price:.3f} ({pnl_sign}{pnl_pct:.1f}%) | "
            f"Chart: {chart_analysis.bias} ({chart_analysis.confidence:.0%}) | "
            f"RSI: {chart_analysis.rsi_14:.0f} | "
            f"Alignment: {chart_analysis.alignment_score:.0%}"
        )


# Module-level monitor instance
_position_monitor: Optional[PositionMonitor] = None


def get_position_monitor() -> PositionMonitor:
    """Get or create the position monitor singleton."""
    global _position_monitor
    if _position_monitor is None:
        _position_monitor = PositionMonitor()
    return _position_monitor


def analyze_open_position(
    position: "Position",
    chart_analysis: Optional["ChartAnalysis"],
    current_token_price: float,
    time_remaining: float,
) -> PositionDecision:
    """
    Convenience function to analyze a position.

    Args:
        position: Position to analyze
        chart_analysis: Current chart analysis
        current_token_price: Current price of position token
        time_remaining: Seconds until market resolution

    Returns:
        PositionDecision with recommended action
    """
    monitor = get_position_monitor()
    return monitor.analyze_position(position, chart_analysis, current_token_price, time_remaining)
