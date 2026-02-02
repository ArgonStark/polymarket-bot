"""
ML Predictor Implementation

Implements MLInterface by wrapping the existing ML predictor code.
All feature extraction and model interaction happens here.
"""

import logging
from typing import Optional, Dict, List
from dataclasses import dataclass

from .interface import (
    MLInterface,
    MLInput,
    MLDecision,
    MLStats,
    TradeSide,
    MarketContext,
    TrendContext,
    ChartContext,
    IndicatorContext,
    PriceContext,
)

# Import existing ML components
from ..strategy.ml_predictor import (
    TradeFeatures,
    get_ml_predictor,
)

logger = logging.getLogger(__name__)


class MLPredictor(MLInterface):
    """
    Production ML implementation.

    Wraps the existing ML predictor with a clean interface.
    All feature extraction is centralized here.
    """

    def __init__(
        self,
        min_confidence: float = 0.52,
        min_samples: int = 30,
        model_path: str = "models/ml_model.json",
        use_hybrid: bool = True,
    ):
        """
        Initialize ML predictor.

        Args:
            min_confidence: Minimum confidence to approve a trade
            min_samples: Minimum training samples before making predictions
            model_path: Path to save/load model
            use_hybrid: Whether to use hybrid RF+NN model
        """
        self.min_confidence = min_confidence
        self.min_samples = min_samples

        # Get existing ML predictor instance
        self._predictor = get_ml_predictor(
            min_confidence=min_confidence,
            min_training_samples=min_samples,
            model_path=model_path,
            use_hybrid=use_hybrid,
        )

        logger.info(
            f"MLPredictor initialized: min_conf={min_confidence:.0%}, "
            f"min_samples={min_samples}, hybrid={use_hybrid}"
        )

    def evaluate_trade(self, input: MLInput) -> MLDecision:
        """
        Evaluate whether to take a trade.

        Converts MLInput to feature vector and gets prediction.
        """
        # Convert input to TradeFeatures
        features = self._input_to_features(input)

        # Create a minimal signal-like object for the predictor
        class SignalProxy:
            def __init__(self, side_value: str, edge: float):
                self.side = type('Side', (), {'value': side_value})()
                self.edge = edge
                self._arb_type = input.arb_type

        signal = SignalProxy(input.side.value, input.edge)

        # Create a minimal market-like object
        class MarketProxy:
            def __init__(self, asset: str, target_price: float, time_remaining: float):
                self.asset = asset
                self.target_price = target_price
                self.time_remaining = time_remaining

        market = MarketProxy(
            input.asset,
            input.price.target_price,
            input.time_remaining
        )

        # Get prediction from existing predictor
        should_trade, confidence, reason = self._predictor.should_trade(
            signal=signal,
            market=market,
            volatility=input.market.volatility,
            price_momentum=input.trend.price_momentum,
            arb_type=input.arb_type,
            spread=input.market.spread,
            bid_depth=input.market.bid_depth,
            ask_depth=input.market.ask_depth,
            price_trend=input.trend.price_trend,
            distance_from_target=input.trend.distance_from_target,
            binance_lead_pct=input.market.binance_lead_pct,
            binance_confirmation=input.market.binance_confirmation,
            trend_1h=input.trend.trend_1h,
            trend_4h=input.trend.trend_4h,
            trend_1d=input.trend.trend_1d,
            current_price=input.price.current_price,
            target_price=input.price.target_price,
            price_high=input.price.price_high,
            price_low=input.price.price_low,
            price_velocity=input.price.price_velocity,
            # Chart analysis
            chart_rsi=input.chart.rsi,
            chart_trend_strength=input.chart.trend_strength,
            chart_is_uptrend=1.0 if input.chart.is_uptrend else 0.0,
            chart_is_downtrend=1.0 if input.chart.is_downtrend else 0.0,
            chart_is_ranging=1.0 if input.chart.is_ranging else 0.0,
            chart_bullish_reversal=1.0 if input.chart.bullish_reversal else 0.0,
            chart_bearish_reversal=1.0 if input.chart.bearish_reversal else 0.0,
            chart_momentum=input.chart.momentum,
            chart_bias_bullish=1.0 if input.chart.bias_bullish else 0.0,
            chart_bias_bearish=1.0 if input.chart.bias_bearish else 0.0,
            chart_confidence=input.chart.confidence,
            chart_bullish_pattern=1.0 if input.chart.bullish_pattern else 0.0,
            chart_bearish_pattern=1.0 if input.chart.bearish_pattern else 0.0,
            chart_uncertainty_score=input.chart.uncertainty_score,
            chart_position_multiplier=input.chart.position_multiplier,
            chart_timeframe_aligned=1.0 if input.chart.timeframe_aligned else 0.0,
            chart_alignment_score=input.chart.alignment_score,
            chart_trend_breaking=1.0 if input.chart.trend_breaking else 0.0,
            chart_resume_ready=1.0 if input.chart.resume_ready else 0.0,
            chart_resume_confidence=input.chart.resume_confidence,
            # Indicators
            macd_histogram=input.indicators.macd_histogram,
            macd_crossover=input.indicators.macd_crossover,
            bb_bandwidth=input.indicators.bb_bandwidth,
            bb_position=input.indicators.bb_position,
            stoch_k=input.indicators.stoch_k,
            stoch_d=input.indicators.stoch_d,
            stoch_signal=input.indicators.stoch_signal,
            rsi_divergence=input.indicators.rsi_divergence,
            rsi_divergence_strength=input.indicators.rsi_divergence_strength,
            volume_ratio=input.indicators.volume_ratio,
            is_high_volume=input.indicators.is_high_volume,
            obv_trend=input.indicators.obv_trend,
            ha_trend=input.indicators.ha_trend,
            ha_consecutive=input.indicators.ha_consecutive,
            ha_strength=input.indicators.ha_strength,
            vwap_distance_pct=input.indicators.vwap_distance_pct,
            vwap_position=input.indicators.vwap_position,
        )

        # Determine model type from reason string
        model_type = "unknown"
        if "Hybrid" in reason:
            model_type = "hybrid"
        elif "NN" in reason:
            model_type = "neural_net"
        elif "LR" in reason:
            model_type = "logistic"

        return MLDecision(
            should_trade=should_trade,
            confidence=confidence,
            reason=reason,
            model_type=model_type,
        )

    def record_outcome(
        self,
        trade_id: str,
        input: MLInput,
        won: bool,
    ) -> None:
        """Record trade outcome for model learning."""
        # Create signal proxy
        class SignalProxy:
            def __init__(self, side_value: str, edge: float, arb_type: str):
                self.side = type('Side', (), {'value': side_value})()
                self.edge = edge
                self._arb_type = arb_type
                # Market info
                self.market = type('Market', (), {
                    'asset': input.asset,
                    'target_price': input.price.target_price,
                    'time_remaining': input.time_remaining,
                    'best_bid': input.entry_price - 0.01,
                    'best_ask': input.entry_price + 0.01,
                    'bid_depth': input.market.bid_depth,
                    'ask_depth': input.market.ask_depth,
                })()
                self.recommended_price = input.entry_price
                self.size_shares = 100  # Placeholder
                self.size_usd = 100 * input.entry_price
                self.time_remaining = input.time_remaining

        signal = SignalProxy(input.side.value, input.edge, input.arb_type)

        # Record with existing predictor
        self._predictor.record_outcome(
            signal=signal,
            volatility=input.market.volatility,
            price_momentum=input.trend.price_momentum,
            won=won,
            arb_type=input.arb_type,
            spread=input.market.spread,
            bid_depth=input.market.bid_depth,
            ask_depth=input.market.ask_depth,
            price_trend=input.trend.price_trend,
            distance_from_target=input.trend.distance_from_target,
            binance_lead_pct=input.market.binance_lead_pct,
            binance_confirmation=input.market.binance_confirmation,
            trend_1h=input.trend.trend_1h,
            trend_4h=input.trend.trend_4h,
            trend_1d=input.trend.trend_1d,
            current_price=input.price.current_price,
            target_price=input.price.target_price,
            price_high=input.price.price_high,
            price_low=input.price.price_low,
            price_velocity=input.price.price_velocity,
            # Chart analysis
            chart_rsi=input.chart.rsi,
            chart_trend_strength=input.chart.trend_strength,
            chart_is_uptrend=1.0 if input.chart.is_uptrend else 0.0,
            chart_is_downtrend=1.0 if input.chart.is_downtrend else 0.0,
            chart_is_ranging=1.0 if input.chart.is_ranging else 0.0,
            chart_bullish_reversal=1.0 if input.chart.bullish_reversal else 0.0,
            chart_bearish_reversal=1.0 if input.chart.bearish_reversal else 0.0,
            chart_momentum=input.chart.momentum,
            chart_bias_bullish=1.0 if input.chart.bias_bullish else 0.0,
            chart_bias_bearish=1.0 if input.chart.bias_bearish else 0.0,
            chart_confidence=input.chart.confidence,
            chart_bullish_pattern=1.0 if input.chart.bullish_pattern else 0.0,
            chart_bearish_pattern=1.0 if input.chart.bearish_pattern else 0.0,
            chart_uncertainty_score=input.chart.uncertainty_score,
            chart_position_multiplier=input.chart.position_multiplier,
            chart_timeframe_aligned=1.0 if input.chart.timeframe_aligned else 0.0,
            chart_alignment_score=input.chart.alignment_score,
            chart_trend_breaking=1.0 if input.chart.trend_breaking else 0.0,
            chart_resume_ready=1.0 if input.chart.resume_ready else 0.0,
            chart_resume_confidence=input.chart.resume_confidence,
            # Indicators
            macd_histogram=input.indicators.macd_histogram,
            macd_crossover=input.indicators.macd_crossover,
            bb_bandwidth=input.indicators.bb_bandwidth,
            bb_position=input.indicators.bb_position,
            stoch_k=input.indicators.stoch_k,
            stoch_d=input.indicators.stoch_d,
            stoch_signal=input.indicators.stoch_signal,
            rsi_divergence=input.indicators.rsi_divergence,
            rsi_divergence_strength=input.indicators.rsi_divergence_strength,
            volume_ratio=input.indicators.volume_ratio,
            is_high_volume=input.indicators.is_high_volume,
            obv_trend=input.indicators.obv_trend,
            ha_trend=input.indicators.ha_trend,
            ha_consecutive=input.indicators.ha_consecutive,
            ha_strength=input.indicators.ha_strength,
            vwap_distance_pct=input.indicators.vwap_distance_pct,
            vwap_position=input.indicators.vwap_position,
        )

        outcome = "WIN" if won else "LOSS"
        logger.debug(f"ML recorded outcome: {trade_id} -> {outcome}")

    def get_stats(self) -> MLStats:
        """Get ML model statistics."""
        stats = self._predictor.get_stats()

        return MLStats(
            training_samples=stats.get("training_samples", 0),
            predictions_made=stats.get("predictions_made", 0),
            correct_predictions=stats.get("correct_predictions", 0),
            accuracy=stats.get("accuracy", 0.0),
            is_active=stats.get("is_active", False),
            model_type=stats.get("model_type", "unknown"),
            min_confidence=stats.get("min_confidence", self.min_confidence),
        )

    def is_ready(self) -> bool:
        """Check if ML has trained enough to make predictions."""
        stats = self.get_stats()
        return stats.training_samples >= self.min_samples

    def save(self) -> None:
        """Save model to disk."""
        self._predictor.save()

    def load(self) -> bool:
        """Load model from disk."""
        # The predictor auto-loads on init
        return True

    def _input_to_features(self, input: MLInput) -> TradeFeatures:
        """
        Convert MLInput to TradeFeatures.

        This is the SINGLE place where input data becomes features.
        """
        return TradeFeatures(
            volatility=input.market.volatility,
            price_momentum=input.trend.price_momentum,
            side_up=1.0 if input.side == TradeSide.UP else 0.0,
            side_down=1.0 if input.side == TradeSide.DOWN else 0.0,
            arb_none=1.0 if input.arb_type == "none" else 0.0,
            arb_binary=1.0 if input.arb_type == "binary_arb" else 0.0,
            arb_synced=1.0 if input.arb_type == "synced" else 0.0,
            arb_lagging=1.0 if input.arb_type == "lagging" else 0.0,
            spread=input.market.spread,
            bid_depth=input.market.bid_depth,
            ask_depth=input.market.ask_depth,
            price_trend=input.trend.price_trend,
            distance_from_target=input.trend.distance_from_target,
            binance_lead_pct=input.market.binance_lead_pct,
            binance_confirmation_none=1.0 if input.market.binance_confirmation == "NONE" else 0.0,
            binance_confirmation_weak=1.0 if input.market.binance_confirmation == "WEAK" else 0.0,
            binance_confirmation_medium=1.0 if input.market.binance_confirmation == "MEDIUM" else 0.0,
            binance_confirmation_strong=1.0 if input.market.binance_confirmation == "STRONG" else 0.0,
            trend_1h=input.trend.trend_1h,
            trend_4h=input.trend.trend_4h,
            trend_1d=input.trend.trend_1d,
            # Price features
            price_normalized=min(1.0, input.price.current_price / 100000) if input.price.current_price else 0.0,
            price_above_target=1.0 if input.price.current_price > input.price.target_price else 0.0,
            price_range_position=0.5,
            price_velocity=input.price.price_velocity,
            # Signal features
            edge=input.edge,
            time_remaining=min(1.0, input.time_remaining / 900),
            # Chart features
            chart_rsi=input.chart.rsi / 100,
            chart_trend_strength=input.chart.trend_strength,
            chart_is_uptrend=1.0 if input.chart.is_uptrend else 0.0,
            chart_is_downtrend=1.0 if input.chart.is_downtrend else 0.0,
            chart_is_ranging=1.0 if input.chart.is_ranging else 0.0,
            chart_bullish_reversal=1.0 if input.chart.bullish_reversal else 0.0,
            chart_bearish_reversal=1.0 if input.chart.bearish_reversal else 0.0,
            chart_momentum=input.chart.momentum,
            chart_bias_bullish=1.0 if input.chart.bias_bullish else 0.0,
            chart_bias_bearish=1.0 if input.chart.bias_bearish else 0.0,
            chart_confidence=input.chart.confidence,
            chart_bullish_pattern=1.0 if input.chart.bullish_pattern else 0.0,
            chart_bearish_pattern=1.0 if input.chart.bearish_pattern else 0.0,
            chart_uncertainty_score=input.chart.uncertainty_score,
            chart_position_multiplier=input.chart.position_multiplier,
            chart_timeframe_aligned=1.0 if input.chart.timeframe_aligned else 0.0,
            chart_alignment_score=input.chart.alignment_score,
            chart_trend_breaking=1.0 if input.chart.trend_breaking else 0.0,
            chart_resume_ready=1.0 if input.chart.resume_ready else 0.0,
            chart_resume_confidence=input.chart.resume_confidence,
            # Indicator features
            macd_histogram=input.indicators.macd_histogram,
            macd_crossover_bullish=1.0 if input.indicators.macd_crossover == "bullish" else 0.0,
            macd_crossover_bearish=1.0 if input.indicators.macd_crossover == "bearish" else 0.0,
            macd_crossover_none=1.0 if input.indicators.macd_crossover == "none" else 0.0,
            bb_bandwidth=input.indicators.bb_bandwidth,
            bb_position_above=1.0 if input.indicators.bb_position == "above" else 0.0,
            bb_position_below=1.0 if input.indicators.bb_position == "below" else 0.0,
            bb_position_middle=1.0 if input.indicators.bb_position == "middle" else 0.0,
            stoch_k=input.indicators.stoch_k / 100,
            stoch_d=input.indicators.stoch_d / 100,
            stoch_overbought=1.0 if input.indicators.stoch_signal == "overbought" else 0.0,
            stoch_oversold=1.0 if input.indicators.stoch_signal == "oversold" else 0.0,
            stoch_neutral=1.0 if input.indicators.stoch_signal == "neutral" else 0.0,
            rsi_divergence_bullish=1.0 if input.indicators.rsi_divergence == "bullish" else 0.0,
            rsi_divergence_bearish=1.0 if input.indicators.rsi_divergence == "bearish" else 0.0,
            rsi_divergence_none=1.0 if input.indicators.rsi_divergence == "none" else 0.0,
            rsi_divergence_strength=input.indicators.rsi_divergence_strength,
            volume_ratio=min(1.0, input.indicators.volume_ratio / 3),
            is_high_volume=1.0 if input.indicators.is_high_volume else 0.0,
            obv_trend=input.indicators.obv_trend,
            ha_trend_bullish=1.0 if input.indicators.ha_trend == "bullish" else 0.0,
            ha_trend_bearish=1.0 if input.indicators.ha_trend == "bearish" else 0.0,
            ha_trend_neutral=1.0 if input.indicators.ha_trend == "neutral" else 0.0,
            ha_consecutive=min(1.0, input.indicators.ha_consecutive / 10),
            ha_strength=input.indicators.ha_strength,
            vwap_distance_pct=max(-1, min(1, input.indicators.vwap_distance_pct / 5)),
            vwap_position_above=1.0 if input.indicators.vwap_position == "above" else 0.0,
            vwap_position_below=1.0 if input.indicators.vwap_position == "below" else 0.0,
            vwap_position_at=1.0 if input.indicators.vwap_position == "at" else 0.0,
        )


def create_ml_input_from_signal(
    signal,
    market,
    chart_analysis=None,
    signal_generator=None,
) -> MLInput:
    """
    Factory function to create MLInput from bot signal and market data.

    This is a helper to ease migration from the old interface.
    """
    # Get volatility and momentum
    volatility = 0.0
    price_momentum = 0.0
    binance_lead_pct = 0.0
    binance_confirmation = "NONE"
    trend_1h = 0.0
    trend_4h = 0.0
    trend_1d = 0.0

    if signal_generator:
        volatility = signal_generator.get_volatility(market.asset)
        current_price = signal_generator.get_price(market.asset) or 0.0
        if current_price and market.target_price:
            price_momentum = (current_price - market.target_price) / market.target_price
            price_momentum = max(-1, min(1, price_momentum * 10))

        # Get Binance confirmation
        bn_conf = signal_generator.get_binance_confirmation(market.asset, signal.side.value)
        if bn_conf:
            binance_lead_pct = bn_conf.get("lead_pct", 0.0)
            binance_confirmation = bn_conf.get("strength", "NONE")

        # Get trends
        trends = signal_generator.get_multi_timeframe_trends(market.asset)
        if trends:
            trend_1h = trends.get("trend_1h", 0.0)
            trend_4h = trends.get("trend_4h", 0.0)
            trend_1d = trends.get("trend_1d", 0.0)

    # Calculate price trend and distance
    current_price = signal_generator.get_price(market.asset) if signal_generator else 0.0
    price_trend = 0.0
    distance_from_target = 0.0
    if current_price and market.target_price:
        price_trend = (current_price - market.target_price) / market.target_price
        distance_from_target = abs(price_trend)

    # Get price range
    price_high = 0.0
    price_low = 0.0
    price_velocity = 0.0
    if signal_generator:
        price_high, price_low = signal_generator.get_price_range(market.asset)
        price_velocity = signal_generator.get_price_velocity(market.asset)

    # Build market context
    market_ctx = MarketContext(
        spread=market.best_ask - market.best_bid if market.best_ask and market.best_bid else 0.0,
        bid_depth=market.bid_depth or 0.0,
        ask_depth=market.ask_depth or 0.0,
        volatility=volatility,
        binance_lead_pct=binance_lead_pct,
        binance_confirmation=binance_confirmation,
    )

    # Build trend context
    trend_ctx = TrendContext(
        trend_1h=trend_1h,
        trend_4h=trend_4h,
        trend_1d=trend_1d,
        price_momentum=price_momentum,
        price_trend=price_trend,
        distance_from_target=distance_from_target,
    )

    # Build chart context
    chart_ctx = ChartContext()
    if chart_analysis:
        chart_ctx = ChartContext(
            rsi=getattr(chart_analysis, 'rsi', 50.0) or 50.0,
            trend_strength=getattr(chart_analysis, 'trend_strength', 0.0) or 0.0,
            is_uptrend=getattr(chart_analysis, 'is_uptrend', False) or False,
            is_downtrend=getattr(chart_analysis, 'is_downtrend', False) or False,
            is_ranging=getattr(chart_analysis, 'is_ranging', True) or True,
            bullish_reversal=getattr(chart_analysis, 'bullish_reversal', False) or False,
            bearish_reversal=getattr(chart_analysis, 'bearish_reversal', False) or False,
            momentum=getattr(chart_analysis, 'momentum', 0.0) or 0.0,
            bias_bullish=getattr(chart_analysis, 'bias', '') == 'bullish',
            bias_bearish=getattr(chart_analysis, 'bias', '') == 'bearish',
            confidence=getattr(chart_analysis, 'confidence', 0.5) or 0.5,
            bullish_pattern=getattr(chart_analysis, 'bullish_pattern', False) or False,
            bearish_pattern=getattr(chart_analysis, 'bearish_pattern', False) or False,
            uncertainty_score=getattr(chart_analysis, 'uncertainty_score', 0.0) or 0.0,
            position_multiplier=getattr(chart_analysis, 'position_multiplier', 1.0) or 1.0,
            timeframe_aligned=getattr(chart_analysis, 'timeframes_aligned', True) or True,
            alignment_score=getattr(chart_analysis, 'alignment_score', 1.0) or 1.0,
            trend_breaking=getattr(chart_analysis, 'trend_strength_dropping', False) or False,
            resume_ready=getattr(chart_analysis, 'resume_ready', True) or True,
            resume_confidence=getattr(chart_analysis, 'resume_confidence', 1.0) or 1.0,
        )

    # Build indicator context
    indicator_ctx = IndicatorContext()
    if chart_analysis:
        indicator_ctx = IndicatorContext(
            macd_histogram=getattr(chart_analysis, 'macd_histogram', 0.0) or 0.0,
            macd_crossover=getattr(chart_analysis, 'macd_crossover', 'none') or 'none',
            bb_bandwidth=getattr(chart_analysis, 'bb_bandwidth', 0.0) or 0.0,
            bb_position=getattr(chart_analysis, 'bb_position', 'middle') or 'middle',
            stoch_k=getattr(chart_analysis, 'stoch_k', 50.0) or 50.0,
            stoch_d=getattr(chart_analysis, 'stoch_d', 50.0) or 50.0,
            stoch_signal=getattr(chart_analysis, 'stoch_signal', 'neutral') or 'neutral',
            rsi_divergence=getattr(chart_analysis, 'rsi_divergence', 'none') or 'none',
            rsi_divergence_strength=getattr(chart_analysis, 'rsi_divergence_strength', 0.0) or 0.0,
            volume_ratio=getattr(chart_analysis, 'volume_ratio', 1.0) or 1.0,
            is_high_volume=getattr(chart_analysis, 'is_high_volume', False) or False,
            obv_trend=getattr(chart_analysis, 'obv_trend', 0.0) or 0.0,
            ha_trend=getattr(chart_analysis, 'ha_trend', 'neutral') or 'neutral',
            ha_consecutive=getattr(chart_analysis, 'ha_consecutive', 0) or 0,
            ha_strength=getattr(chart_analysis, 'ha_strength', 0.0) or 0.0,
            vwap_distance_pct=getattr(chart_analysis, 'vwap_distance_pct', 0.0) or 0.0,
            vwap_position=getattr(chart_analysis, 'vwap_position', 'at') or 'at',
        )

    # Build price context
    price_ctx = PriceContext(
        current_price=current_price or 0.0,
        target_price=market.target_price or 0.0,
        price_high=price_high,
        price_low=price_low,
        price_velocity=price_velocity,
    )

    # Get arb type
    arb_type = getattr(signal, '_arb_type', 'none') or 'none'

    return MLInput(
        asset=market.asset,
        side=TradeSide.UP if signal.side.value == "UP" else TradeSide.DOWN,
        entry_price=signal.recommended_price,
        edge=signal.edge,
        time_remaining=market.time_remaining,
        arb_type=arb_type,
        market=market_ctx,
        trend=trend_ctx,
        chart=chart_ctx,
        indicators=indicator_ctx,
        price=price_ctx,
    )
