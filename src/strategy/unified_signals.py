"""
Unified Signal Framework for Polymarket 15-minute Crypto Trading.

This module provides a clean, organized approach to signal generation that:
1. Groups indicators by type (trend-following vs mean-reversion)
2. Uses priority-based decision making
3. Selects strategy based on market context
4. Provides unified confidence scoring

The key insight: Don't apply ALL indicators at once. Instead:
- First determine market CONTEXT (trending vs ranging)
- Then apply the APPROPRIATE strategy
- Use other indicators for CONFIRMATION only
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

# Use structured trading logger
try:
    from ..utils.trading_logger import get_trading_logger
    USE_TRADING_LOGGER = True
except ImportError:
    USE_TRADING_LOGGER = False

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & TYPES
# =============================================================================

class MarketContext(Enum):
    """Market regime determines which strategy to use."""
    STRONG_TREND = "strong_trend"      # Use trend-following
    WEAK_TREND = "weak_trend"          # Use trend-following with caution
    RANGING = "ranging"                # Use mean-reversion
    VOLATILE = "volatile"              # Reduce position size
    UNCERTAIN = "uncertain"            # Skip or minimal size


class SignalDirection(Enum):
    """Trading direction."""
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


class SignalStrength(Enum):
    """Signal strength levels."""
    STRONG = "strong"      # High confidence, full position
    MODERATE = "moderate"  # Medium confidence, reduced position
    WEAK = "weak"          # Low confidence, minimal position
    NONE = "none"          # No signal


# =============================================================================
# INDICATOR GROUPS
# =============================================================================

@dataclass
class TrendFollowingSignals:
    """
    Trend-following indicators - use when market is TRENDING.

    These indicators work best when price is moving in a direction.
    They will give FALSE signals in ranging markets.
    """
    # MACD
    macd_bullish: bool = False
    macd_bearish: bool = False
    macd_strength: float = 0.0  # 0-1

    # Heiken Ashi (smoothed trend)
    ha_bullish: bool = False
    ha_bearish: bool = False
    ha_strength: float = 0.0  # 0-1

    # Moving Average alignment
    ma_bullish: bool = False  # Price > EMA20 > EMA50
    ma_bearish: bool = False  # Price < EMA20 < EMA50

    # Multi-timeframe trend
    mtf_bullish: bool = False  # 15m, 1h, 4h all bullish
    mtf_bearish: bool = False  # 15m, 1h, 4h all bearish
    mtf_alignment: float = 0.0  # 0-1 (how aligned are timeframes)

    # BTC correlation (for altcoins)
    btc_bullish: bool = False
    btc_bearish: bool = False
    btc_velocity: float = 0.0

    # Weighted trend bias (fallback when individual signals conflict)
    trend_bias: float = 0.0  # -1 to +1, from weighted multi-timeframe analysis

    def get_direction(self) -> Tuple[SignalDirection, float]:
        """
        Calculate overall trend-following direction and strength.

        Returns:
            Tuple of (direction, confidence 0-1)
        """
        bullish_score = 0.0
        bearish_score = 0.0

        # MACD (weight: 25%)
        if self.macd_bullish:
            bullish_score += 0.25 * self.macd_strength
        elif self.macd_bearish:
            bearish_score += 0.25 * self.macd_strength

        # Heiken Ashi (weight: 20%)
        if self.ha_bullish:
            bullish_score += 0.20 * self.ha_strength
        elif self.ha_bearish:
            bearish_score += 0.20 * self.ha_strength

        # MA alignment (weight: 15%)
        if self.ma_bullish:
            bullish_score += 0.15
        elif self.ma_bearish:
            bearish_score += 0.15

        # Multi-timeframe (weight: 30% - most important)
        if self.mtf_bullish:
            bullish_score += 0.30 * self.mtf_alignment
        elif self.mtf_bearish:
            bearish_score += 0.30 * self.mtf_alignment

        # BTC correlation (weight: 10%)
        if self.btc_bullish:
            bullish_score += 0.10 * min(1.0, abs(self.btc_velocity) * 100)
        elif self.btc_bearish:
            bearish_score += 0.10 * min(1.0, abs(self.btc_velocity) * 100)

        # Trend bias fallback (weight: 15% - helps when other signals conflict)
        # This uses weighted multi-timeframe trend directly
        if self.trend_bias > 0.05:
            bullish_score += 0.15 * min(1.0, self.trend_bias * 3)  # Scale: 0.33 trend = full weight
        elif self.trend_bias < -0.05:
            bearish_score += 0.15 * min(1.0, abs(self.trend_bias) * 3)

        # Determine direction (RELAXED threshold: 0.15 for more trading)
        if bullish_score > bearish_score and bullish_score > 0.15:
            return SignalDirection.UP, bullish_score
        elif bearish_score > bullish_score and bearish_score > 0.15:
            return SignalDirection.DOWN, bearish_score
        else:
            return SignalDirection.NEUTRAL, max(bullish_score, bearish_score)


@dataclass
class MeanReversionSignals:
    """
    Mean-reversion indicators - use when market is RANGING.

    These indicators work best when price oscillates around a mean.
    They will give FALSE signals in trending markets.
    """
    # RSI
    rsi_oversold: bool = False   # RSI < 30
    rsi_overbought: bool = False  # RSI > 70
    rsi_value: float = 50.0

    # Stochastic
    stoch_oversold: bool = False   # K < 20
    stoch_overbought: bool = False  # K > 80
    stoch_k: float = 50.0

    # Bollinger Bands
    bb_below_lower: bool = False  # Price below lower band
    bb_above_upper: bool = False  # Price above upper band
    bb_squeeze: bool = False      # Low bandwidth (big move coming)

    # VWAP
    vwap_below: bool = False  # Price below VWAP (expect reversion up)
    vwap_above: bool = False  # Price above VWAP (expect reversion down)
    vwap_distance: float = 0.0  # Distance from VWAP as %

    # RSI Divergence (powerful reversal signal)
    rsi_bullish_div: bool = False
    rsi_bearish_div: bool = False
    div_strength: float = 0.0

    def get_direction(self) -> Tuple[SignalDirection, float]:
        """
        Calculate overall mean-reversion direction and strength.

        Mean reversion = expect price to move OPPOSITE to current extreme.

        Returns:
            Tuple of (direction, confidence 0-1)
        """
        bullish_score = 0.0  # Expect price to go UP
        bearish_score = 0.0  # Expect price to go DOWN

        # RSI (weight: 25%)
        if self.rsi_oversold:
            bullish_score += 0.25 * (1 - self.rsi_value / 30)  # More oversold = stronger
        elif self.rsi_overbought:
            bearish_score += 0.25 * ((self.rsi_value - 70) / 30)  # More overbought = stronger

        # Stochastic (weight: 20%)
        if self.stoch_oversold:
            bullish_score += 0.20 * (1 - self.stoch_k / 20)
        elif self.stoch_overbought:
            bearish_score += 0.20 * ((self.stoch_k - 80) / 20)

        # Bollinger Bands (weight: 20%)
        if self.bb_below_lower:
            bullish_score += 0.20
        elif self.bb_above_upper:
            bearish_score += 0.20

        # VWAP (weight: 15%)
        vwap_strength = min(1.0, abs(self.vwap_distance) * 200)  # 0.5% = full strength
        if self.vwap_below:
            bullish_score += 0.15 * vwap_strength
        elif self.vwap_above:
            bearish_score += 0.15 * vwap_strength

        # RSI Divergence (weight: 20% - powerful signal)
        if self.rsi_bullish_div:
            bullish_score += 0.20 * self.div_strength
        elif self.rsi_bearish_div:
            bearish_score += 0.20 * self.div_strength

        # Determine direction (RELAXED threshold: 0.18 instead of 0.25)
        if bullish_score > bearish_score and bullish_score > 0.18:
            return SignalDirection.UP, bullish_score
        elif bearish_score > bullish_score and bearish_score > 0.18:
            return SignalDirection.DOWN, bearish_score
        else:
            return SignalDirection.NEUTRAL, max(bullish_score, bearish_score)


@dataclass
class ConfirmationSignals:
    """
    Confirmation indicators - use to VALIDATE signals from other groups.

    These don't generate signals on their own but strengthen/weaken
    signals from trend-following or mean-reversion groups.
    """
    # Volume
    high_volume: bool = False    # Volume > 1.5x average
    volume_ratio: float = 1.0
    obv_bullish: bool = False    # OBV trending up
    obv_bearish: bool = False    # OBV trending down

    # Candlestick patterns
    bullish_pattern: bool = False
    bearish_pattern: bool = False
    pattern_name: str = ""

    # Momentum
    momentum_positive: bool = False
    momentum_negative: bool = False
    momentum_increasing: bool = False

    def get_confirmation_multiplier(self, direction: SignalDirection) -> float:
        """
        Get confirmation multiplier for a signal direction.

        Returns:
            Multiplier 0.5-1.5 (0.5 = weak confirmation, 1.5 = strong)
        """
        if direction == SignalDirection.NEUTRAL:
            return 1.0

        confirms = 0
        contradicts = 0

        is_bullish = direction == SignalDirection.UP

        # Volume confirmation
        if self.high_volume:
            confirms += 1

        # OBV
        if is_bullish and self.obv_bullish:
            confirms += 1
        elif not is_bullish and self.obv_bearish:
            confirms += 1
        elif is_bullish and self.obv_bearish:
            contradicts += 1
        elif not is_bullish and self.obv_bullish:
            contradicts += 1

        # Patterns
        if is_bullish and self.bullish_pattern:
            confirms += 1
        elif not is_bullish and self.bearish_pattern:
            confirms += 1
        elif is_bullish and self.bearish_pattern:
            contradicts += 1
        elif not is_bullish and self.bullish_pattern:
            contradicts += 1

        # Momentum
        if is_bullish and self.momentum_positive:
            confirms += 1
        elif not is_bullish and self.momentum_negative:
            confirms += 1
        elif is_bullish and self.momentum_negative:
            contradicts += 1
        elif not is_bullish and self.momentum_positive:
            contradicts += 1

        # Calculate multiplier
        net_confirms = confirms - contradicts

        if net_confirms >= 3:
            return 1.5  # Strong confirmation
        elif net_confirms >= 2:
            return 1.3
        elif net_confirms >= 1:
            return 1.15
        elif net_confirms == 0:
            return 1.0
        elif net_confirms >= -1:
            return 0.85
        elif net_confirms >= -2:
            return 0.7
        else:
            return 0.5  # Strong contradiction


@dataclass
class ContextSignals:
    """
    Context signals - determine WHAT strategy to use.

    These are evaluated FIRST to decide whether to use
    trend-following or mean-reversion approach.
    """
    # Time remaining
    time_remaining: float = 900.0  # seconds
    time_critical: bool = False    # < 2 min left

    # Volatility/Distance
    distance_to_target_pct: float = 0.0
    atr_15min: float = 0.0
    can_reach_target: bool = True  # Based on ATR vs distance

    # Market type from chart analysis
    is_trending: bool = False
    is_ranging: bool = False
    trend_strength: float = 0.0  # 0-1

    # Uncertainty
    is_uncertain: bool = False
    uncertainty_score: float = 0.0

    # Price position
    price_above_target: bool = False
    price_below_target: bool = False

    def get_market_context(self) -> MarketContext:
        """Determine market context for strategy selection."""

        # High uncertainty = skip
        if self.is_uncertain and self.uncertainty_score > 0.7:
            return MarketContext.UNCERTAIN

        # Time critical = favor current position
        if self.time_critical:
            # Not enough time for price to move much
            return MarketContext.UNCERTAIN

        # Can't reach target = favor current position
        if not self.can_reach_target:
            return MarketContext.UNCERTAIN

        # Determine trending vs ranging
        if self.is_trending and self.trend_strength > 0.6:
            return MarketContext.STRONG_TREND
        elif self.is_trending and self.trend_strength > 0.3:
            return MarketContext.WEAK_TREND
        elif self.is_ranging:
            return MarketContext.RANGING
        else:
            # Mixed signals
            if self.uncertainty_score > 0.5:
                return MarketContext.VOLATILE
            else:
                return MarketContext.WEAK_TREND


# =============================================================================
# UNIFIED SIGNAL RESULT
# =============================================================================

@dataclass
class UnifiedSignal:
    """Final unified trading signal."""

    direction: SignalDirection
    strength: SignalStrength
    confidence: float  # 0-1

    # Components
    context: MarketContext
    strategy_used: str  # "trend_following", "mean_reversion", "time_based"

    # Edge adjustments
    edge_adjustment: float  # How much to adjust edge (can be negative)
    position_multiplier: float  # 0-1, scale position size

    # Reasoning
    primary_reason: str
    supporting_reasons: list = field(default_factory=list)
    contradicting_reasons: list = field(default_factory=list)

    # Raw scores
    trend_score: float = 0.0
    reversion_score: float = 0.0
    confirmation_score: float = 0.0

    def should_trade(self, side: str) -> bool:
        """Check if we should trade in the given direction."""
        if self.strength == SignalStrength.NONE:
            return False

        if side.upper() == "UP":
            return self.direction == SignalDirection.UP
        elif side.upper() == "DOWN":
            return self.direction == SignalDirection.DOWN

        return False

    def get_edge_boost(self, base_edge: float) -> float:
        """Get the adjusted edge after applying signal adjustments."""
        adjusted = base_edge + self.edge_adjustment
        return max(0.0, adjusted)  # Don't go negative


# =============================================================================
# UNIFIED SIGNAL GENERATOR
# =============================================================================

class UnifiedSignalGenerator:
    """
    Unified signal generator that combines all indicators intelligently.

    Flow:
    1. Evaluate CONTEXT (time, volatility, market type)
    2. Select STRATEGY based on context
    3. Generate signal using appropriate indicator group
    4. Apply CONFIRMATION to adjust confidence
    5. Return unified signal
    """

    def __init__(self):
        self.last_signal: Optional[UnifiedSignal] = None

    def generate_signal(
        self,
        context: ContextSignals,
        trend_signals: TrendFollowingSignals,
        reversion_signals: MeanReversionSignals,
        confirmation: ConfirmationSignals,
    ) -> UnifiedSignal:
        """
        Generate a unified trading signal.

        Args:
            context: Market context signals
            trend_signals: Trend-following indicator values
            reversion_signals: Mean-reversion indicator values
            confirmation: Confirmation indicator values

        Returns:
            UnifiedSignal with direction, strength, and adjustments
        """

        # Step 1: Determine market context
        market_context = context.get_market_context()

        # Step 2: Handle special contexts
        if market_context == MarketContext.UNCERTAIN:
            return self._handle_uncertain_market(context)

        # Step 3: Select strategy and generate signal
        if market_context in [MarketContext.STRONG_TREND, MarketContext.WEAK_TREND]:
            direction, raw_confidence = trend_signals.get_direction()
            strategy = "trend_following"
            trend_score = raw_confidence
            reversion_score = 0.0
        elif market_context == MarketContext.RANGING:
            direction, raw_confidence = reversion_signals.get_direction()
            strategy = "mean_reversion"
            trend_score = 0.0
            reversion_score = raw_confidence
        else:  # VOLATILE
            # Use both, but reduce confidence
            trend_dir, trend_conf = trend_signals.get_direction()
            rev_dir, rev_conf = reversion_signals.get_direction()

            # If they agree, use that direction with reduced confidence
            if trend_dir == rev_dir and trend_dir != SignalDirection.NEUTRAL:
                direction = trend_dir
                raw_confidence = (trend_conf + rev_conf) / 2 * 0.7  # 30% reduction
                strategy = "mixed"
            else:
                # Disagreement - very low confidence
                direction = SignalDirection.NEUTRAL
                raw_confidence = 0.0
                strategy = "conflict"

            trend_score = trend_conf
            reversion_score = rev_conf

        # Step 4: Apply confirmation
        conf_multiplier = confirmation.get_confirmation_multiplier(direction)
        adjusted_confidence = min(1.0, raw_confidence * conf_multiplier)

        # Step 5: Determine strength
        if adjusted_confidence >= 0.7:
            strength = SignalStrength.STRONG
        elif adjusted_confidence >= 0.5:
            strength = SignalStrength.MODERATE
        elif adjusted_confidence >= 0.3:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.NONE

        # Step 6: Calculate edge adjustment
        edge_adjustment = self._calculate_edge_adjustment(
            direction, strength, market_context, context
        )

        # Step 7: Calculate position multiplier
        position_multiplier = self._calculate_position_multiplier(
            strength, market_context, context, conf_multiplier
        )

        # Step 8: Build reasoning
        primary_reason, supporting, contradicting = self._build_reasoning(
            direction, strategy, trend_signals, reversion_signals,
            confirmation, context
        )

        signal = UnifiedSignal(
            direction=direction,
            strength=strength,
            confidence=adjusted_confidence,
            context=market_context,
            strategy_used=strategy,
            edge_adjustment=edge_adjustment,
            position_multiplier=position_multiplier,
            primary_reason=primary_reason,
            supporting_reasons=supporting,
            contradicting_reasons=contradicting,
            trend_score=trend_score,
            reversion_score=reversion_score,
            confirmation_score=conf_multiplier,
        )

        self.last_signal = signal
        return signal

    def _handle_uncertain_market(self, context: ContextSignals) -> UnifiedSignal:
        """Handle uncertain/time-critical market conditions."""

        # In uncertain markets, favor the current price position
        # (if price is above target, likely stays above)
        if context.time_critical or not context.can_reach_target:
            # Time-based decision
            if context.price_above_target:
                direction = SignalDirection.UP
                reason = f"Time-based: {context.time_remaining:.0f}s left, price above target"
            elif context.price_below_target:
                direction = SignalDirection.DOWN
                reason = f"Time-based: {context.time_remaining:.0f}s left, price below target"
            else:
                direction = SignalDirection.NEUTRAL
                reason = "Price at target with little time"

            # Confidence based on distance and time
            if context.can_reach_target:
                confidence = 0.3  # Low confidence
            else:
                confidence = 0.6  # Higher confidence price stays

            return UnifiedSignal(
                direction=direction,
                strength=SignalStrength.MODERATE if confidence >= 0.5 else SignalStrength.WEAK,
                confidence=confidence,
                context=MarketContext.UNCERTAIN,
                strategy_used="time_based",
                edge_adjustment=0.05 if direction != SignalDirection.NEUTRAL else 0.0,
                position_multiplier=0.5,  # Reduced size in uncertain conditions
                primary_reason=reason,
                supporting_reasons=[],
                contradicting_reasons=["High uncertainty"],
            )

        # General uncertainty - allow small position instead of skip
        return UnifiedSignal(
            direction=SignalDirection.NEUTRAL,
            strength=SignalStrength.NONE,
            confidence=0.0,
            context=MarketContext.UNCERTAIN,
            strategy_used="none",
            edge_adjustment=-0.05,  # Smaller edge reduction
            position_multiplier=0.25,  # Was 0.0 - now allows small positions
            primary_reason=f"Market uncertain ({context.uncertainty_score:.0%})",
            supporting_reasons=[],
            contradicting_reasons=["High uncertainty score"],
        )

    def _calculate_edge_adjustment(
        self,
        direction: SignalDirection,
        strength: SignalStrength,
        context: MarketContext,
        ctx: ContextSignals,
    ) -> float:
        """Calculate how much to adjust the edge based on signal."""

        if direction == SignalDirection.NEUTRAL:
            return -0.05  # Slight negative adjustment

        # Base adjustment based on strength
        base_adj = {
            SignalStrength.STRONG: 0.08,
            SignalStrength.MODERATE: 0.04,
            SignalStrength.WEAK: 0.02,
            SignalStrength.NONE: -0.05,
        }[strength]

        # Context multiplier
        context_mult = {
            MarketContext.STRONG_TREND: 1.2,
            MarketContext.WEAK_TREND: 1.0,
            MarketContext.RANGING: 0.9,
            MarketContext.VOLATILE: 0.7,
            MarketContext.UNCERTAIN: 0.5,
        }[context]

        return base_adj * context_mult

    def _calculate_position_multiplier(
        self,
        strength: SignalStrength,
        context: MarketContext,
        ctx: ContextSignals,
        conf_multiplier: float,
    ) -> float:
        """Calculate position size multiplier."""

        # Base multiplier from strength (more aggressive - allow trading even without strong signals)
        base_mult = {
            SignalStrength.STRONG: 1.0,
            SignalStrength.MODERATE: 0.8,
            SignalStrength.WEAK: 0.5,
            SignalStrength.NONE: 0.3,  # Was 0.0 - now allows small positions
        }[strength]

        # Context adjustment (less harsh penalties)
        context_mult = {
            MarketContext.STRONG_TREND: 1.0,
            MarketContext.WEAK_TREND: 0.9,
            MarketContext.RANGING: 0.8,  # Was 0.7
            MarketContext.VOLATILE: 0.6,  # Was 0.5
            MarketContext.UNCERTAIN: 0.5,  # Was 0.3
        }[context]

        # Confirmation adjustment
        conf_adj = min(1.2, max(0.6, conf_multiplier))

        # Uncertainty penalty (reduced from 0.5 to 0.3)
        uncertainty_mult = 1.0 - (ctx.uncertainty_score * 0.3)

        return min(1.0, base_mult * context_mult * conf_adj * uncertainty_mult)

    def _build_reasoning(
        self,
        direction: SignalDirection,
        strategy: str,
        trend: TrendFollowingSignals,
        reversion: MeanReversionSignals,
        confirmation: ConfirmationSignals,
        context: ContextSignals,
    ) -> Tuple[str, list, list]:
        """Build human-readable reasoning for the signal."""

        supporting = []
        contradicting = []

        # Primary reason
        if strategy == "trend_following":
            if direction == SignalDirection.UP:
                primary = "Trend-following: Market trending bullish"
            elif direction == SignalDirection.DOWN:
                primary = "Trend-following: Market trending bearish"
            else:
                primary = "Trend-following: No clear trend direction"
        elif strategy == "mean_reversion":
            if direction == SignalDirection.UP:
                primary = "Mean-reversion: Market oversold, expect bounce"
            elif direction == SignalDirection.DOWN:
                primary = "Mean-reversion: Market overbought, expect pullback"
            else:
                primary = "Mean-reversion: Market at equilibrium"
        elif strategy == "time_based":
            primary = f"Time-based: {context.time_remaining:.0f}s remaining"
        else:
            primary = "Mixed/conflicting signals"

        # Trend-following reasons
        if trend.macd_bullish:
            supporting.append("MACD bullish crossover")
        elif trend.macd_bearish:
            if direction == SignalDirection.DOWN:
                supporting.append("MACD bearish crossover")
            else:
                contradicting.append("MACD bearish")

        if trend.mtf_bullish:
            if direction == SignalDirection.UP:
                supporting.append(f"Multi-TF aligned bullish ({trend.mtf_alignment:.0%})")
            else:
                contradicting.append("Multi-TF bullish")
        elif trend.mtf_bearish:
            if direction == SignalDirection.DOWN:
                supporting.append(f"Multi-TF aligned bearish ({trend.mtf_alignment:.0%})")
            else:
                contradicting.append("Multi-TF bearish")

        if trend.ha_bullish and trend.ha_strength > 0.5:
            if direction == SignalDirection.UP:
                supporting.append(f"Heiken Ashi bullish ({trend.ha_strength:.0%})")
            else:
                contradicting.append("Heiken Ashi bullish")
        elif trend.ha_bearish and trend.ha_strength > 0.5:
            if direction == SignalDirection.DOWN:
                supporting.append(f"Heiken Ashi bearish ({trend.ha_strength:.0%})")
            else:
                contradicting.append("Heiken Ashi bearish")

        # Mean-reversion reasons
        if reversion.rsi_oversold:
            if direction == SignalDirection.UP:
                supporting.append(f"RSI oversold ({reversion.rsi_value:.0f})")
            else:
                contradicting.append(f"RSI oversold ({reversion.rsi_value:.0f})")
        elif reversion.rsi_overbought:
            if direction == SignalDirection.DOWN:
                supporting.append(f"RSI overbought ({reversion.rsi_value:.0f})")
            else:
                contradicting.append(f"RSI overbought ({reversion.rsi_value:.0f})")

        if reversion.bb_below_lower:
            if direction == SignalDirection.UP:
                supporting.append("Below Bollinger lower band")
            else:
                contradicting.append("Below Bollinger lower band")
        elif reversion.bb_above_upper:
            if direction == SignalDirection.DOWN:
                supporting.append("Above Bollinger upper band")
            else:
                contradicting.append("Above Bollinger upper band")

        if reversion.rsi_bullish_div:
            if direction == SignalDirection.UP:
                supporting.append("RSI bullish divergence")
            else:
                contradicting.append("RSI bullish divergence")
        elif reversion.rsi_bearish_div:
            if direction == SignalDirection.DOWN:
                supporting.append("RSI bearish divergence")
            else:
                contradicting.append("RSI bearish divergence")

        # Confirmation reasons
        if confirmation.high_volume:
            supporting.append(f"High volume ({confirmation.volume_ratio:.1f}x)")

        if confirmation.bullish_pattern:
            if direction == SignalDirection.UP:
                supporting.append(f"Pattern: {confirmation.pattern_name}")
            else:
                contradicting.append(f"Bullish pattern: {confirmation.pattern_name}")
        elif confirmation.bearish_pattern:
            if direction == SignalDirection.DOWN:
                supporting.append(f"Pattern: {confirmation.pattern_name}")
            else:
                contradicting.append(f"Bearish pattern: {confirmation.pattern_name}")

        return primary, supporting[:5], contradicting[:3]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_signals_from_chart_analysis(chart_analysis) -> Tuple[
    ContextSignals, TrendFollowingSignals, MeanReversionSignals, ConfirmationSignals
]:
    """
    Extract unified signal components from ChartAnalysis object.

    This bridges the gap between the existing chart analysis and
    the new unified signal framework.
    """
    if chart_analysis is None:
        return (
            ContextSignals(),
            TrendFollowingSignals(),
            MeanReversionSignals(),
            ConfirmationSignals(),
        )

    ca = chart_analysis

    # Context signals
    context = ContextSignals(
        time_remaining=getattr(ca, 'time_remaining', 900.0),
        time_critical=getattr(ca, 'time_remaining', 900.0) < 120,
        distance_to_target_pct=getattr(ca, 'distance_to_target_pct', 0.0),
        atr_15min=ca.atr,
        can_reach_target=True,  # Will be calculated based on ATR vs distance
        is_trending=ca.market_type.value in ['uptrend', 'downtrend', 'strong_uptrend', 'strong_downtrend'],
        is_ranging=ca.market_type.value == 'ranging',
        trend_strength=ca.trend_strength,
        is_uncertain=ca.is_uncertain,
        uncertainty_score=ca.uncertainty_score,
    )

    # Calculate weighted trend score for more nuanced detection
    # Weight: 15m=40%, 1h=35%, 4h=25% (short-term matters more for 15-min markets)
    weighted_trend = (
        ca.trend_15m * 0.40 +
        ca.trend_1h * 0.35 +
        ca.trend_4h * 0.25
    )

    # Trend-following signals (RELAXED thresholds for more trading)
    trend = TrendFollowingSignals(
        # MACD
        macd_bullish=ca.macd_crossover == "bullish" or ca.macd_histogram > 0,
        macd_bearish=ca.macd_crossover == "bearish" or ca.macd_histogram < 0,
        macd_strength=min(1.0, abs(ca.macd_histogram) * 10) if ca.macd_histogram else 0.5,

        # Heiken Ashi
        ha_bullish=ca.ha_trend == "bullish",
        ha_bearish=ca.ha_trend == "bearish",
        ha_strength=ca.ha_strength,

        # MA alignment - RELAXED: use weighted trend with lower threshold
        ma_bullish=weighted_trend > 0.08,  # Was: 0.15
        ma_bearish=weighted_trend < -0.08,  # Was: -0.15

        # Multi-timeframe - RELAXED: majority rules OR any two timeframes agree
        mtf_bullish=(
            (ca.trend_15m > 0.05 and ca.trend_1h > 0.05) or  # Short+medium bullish
            (ca.trend_1h > 0.05 and ca.trend_4h > 0.0) or   # Medium+long bullish
            (ca.trend_15m > 0.1 and ca.trend_4h > 0.0) or   # Short+long bullish
            (weighted_trend > 0.12)  # Moderate weighted trend
        ),
        mtf_bearish=(
            (ca.trend_15m < -0.05 and ca.trend_1h < -0.05) or  # Short+medium bearish
            (ca.trend_1h < -0.05 and ca.trend_4h < 0.0) or    # Medium+long bearish
            (ca.trend_15m < -0.1 and ca.trend_4h < 0.0) or    # Short+long bearish
            (weighted_trend < -0.12)  # Moderate weighted trend
        ),
        mtf_alignment=ca.alignment_score,

        # Direct trend bias from weighted multi-timeframe analysis
        trend_bias=weighted_trend,
    )

    # Mean-reversion signals
    reversion = MeanReversionSignals(
        # RSI
        rsi_oversold=ca.rsi_14 < 30,
        rsi_overbought=ca.rsi_14 > 70,
        rsi_value=ca.rsi_14,

        # Stochastic
        stoch_oversold=ca.stoch_signal == "oversold",
        stoch_overbought=ca.stoch_signal == "overbought",
        stoch_k=ca.stoch_k,

        # Bollinger Bands
        bb_below_lower=ca.bb_position == "below_lower",
        bb_above_upper=ca.bb_position == "above_upper",
        bb_squeeze=ca.bb_bandwidth < 1.5,

        # VWAP
        vwap_below=ca.vwap_position == "below",
        vwap_above=ca.vwap_position == "above",
        vwap_distance=ca.vwap_distance_pct,

        # RSI Divergence
        rsi_bullish_div=ca.rsi_divergence == "bullish",
        rsi_bearish_div=ca.rsi_divergence == "bearish",
        div_strength=ca.rsi_divergence_strength,
    )

    # Confirmation signals
    confirmation = ConfirmationSignals(
        high_volume=ca.is_high_volume,
        volume_ratio=ca.volume_ratio,
        obv_bullish=ca.obv_trend > 0.3,
        obv_bearish=ca.obv_trend < -0.3,
        bullish_pattern=ca.is_bullish_pattern,
        bearish_pattern=ca.is_bearish_pattern,
        pattern_name=ca.pattern_name,
        momentum_positive=ca.momentum > 0.3,
        momentum_negative=ca.momentum < -0.3,
        momentum_increasing=ca.momentum_increasing,
    )

    return context, trend, reversion, confirmation


# Singleton instance
_unified_generator: Optional[UnifiedSignalGenerator] = None


def get_unified_signal_generator() -> UnifiedSignalGenerator:
    """Get the singleton UnifiedSignalGenerator instance."""
    global _unified_generator
    if _unified_generator is None:
        _unified_generator = UnifiedSignalGenerator()
    return _unified_generator


def generate_unified_signal(
    chart_analysis,
    time_remaining: float = 900.0,
    asset: str = "",
) -> UnifiedSignal:
    """
    Convenience function to generate a unified signal from chart analysis.

    Args:
        chart_analysis: ChartAnalysis object from binance_chart.py
        time_remaining: Seconds remaining in the market
        asset: Asset symbol for logging

    Returns:
        UnifiedSignal with direction, strength, and adjustments
    """
    context, trend, reversion, confirmation = extract_signals_from_chart_analysis(chart_analysis)

    # Update time_remaining in context
    context.time_remaining = time_remaining
    context.time_critical = time_remaining < 120

    generator = get_unified_signal_generator()
    signal = generator.generate_signal(context, trend, reversion, confirmation)

    # Log using structured trading logger
    if USE_TRADING_LOGGER and asset:
        try:
            tlog = get_trading_logger(asset)
            tlog.signal_analysis(
                direction=signal.direction.value.upper(),
                strength=signal.strength.value.upper(),
                confidence=signal.confidence,
                context=signal.context.value,
                strategy=signal.strategy_used,
                edge_adjustment=signal.edge_adjustment,
                primary_reason=signal.primary_reason,
                supporting=signal.supporting_reasons,
                contradicting=signal.contradicting_reasons,
                time_remaining=time_remaining,
            )
        except Exception:
            pass  # Don't let logging errors affect signal generation

    return signal
