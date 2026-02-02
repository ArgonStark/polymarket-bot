"""
ML Interface Layer

Clean boundary between trading logic and ML system.
Trading code only interacts with MLInterface - implementation details are hidden.

Usage:
    from src.ml import MLInterface, MLInput, MLDecision

    # Get decision
    decision = ml.evaluate_trade(ml_input)
    if decision.should_trade:
        execute_trade()

    # Record outcome
    ml.record_outcome(trade_id, ml_input, won=True)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class TradeSide(Enum):
    """Trade direction."""
    UP = "UP"
    DOWN = "DOWN"


@dataclass
class MarketContext:
    """Market microstructure data."""
    spread: float = 0.0
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    volatility: float = 0.0

    # Binance data
    binance_lead_pct: float = 0.0
    binance_confirmation: str = "NONE"  # NONE, WEAK, MEDIUM, STRONG


@dataclass
class TrendContext:
    """Multi-timeframe trend data."""
    trend_1h: float = 0.0
    trend_4h: float = 0.0
    trend_1d: float = 0.0
    price_momentum: float = 0.0
    price_trend: float = 0.0
    distance_from_target: float = 0.0


@dataclass
class ChartContext:
    """Technical analysis from chart data."""
    # RSI
    rsi: float = 50.0

    # Trend
    trend_strength: float = 0.0
    is_uptrend: bool = False
    is_downtrend: bool = False
    is_ranging: bool = True

    # Reversals
    bullish_reversal: bool = False
    bearish_reversal: bool = False

    # Momentum & Bias
    momentum: float = 0.0
    bias_bullish: bool = False
    bias_bearish: bool = False
    confidence: float = 0.5

    # Patterns
    bullish_pattern: bool = False
    bearish_pattern: bool = False

    # Advanced
    uncertainty_score: float = 0.0
    position_multiplier: float = 1.0
    timeframe_aligned: bool = True
    alignment_score: float = 1.0
    trend_breaking: bool = False
    resume_ready: bool = True
    resume_confidence: float = 1.0


@dataclass
class IndicatorContext:
    """Technical indicator values."""
    # MACD
    macd_histogram: float = 0.0
    macd_crossover: str = "none"  # bullish, bearish, none

    # Bollinger Bands
    bb_bandwidth: float = 0.0
    bb_position: str = "middle"  # above, below, middle

    # Stochastic
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    stoch_signal: str = "neutral"  # overbought, oversold, neutral

    # RSI Divergence
    rsi_divergence: str = "none"  # bullish, bearish, none
    rsi_divergence_strength: float = 0.0

    # Volume
    volume_ratio: float = 1.0
    is_high_volume: bool = False
    obv_trend: float = 0.0

    # Heiken Ashi
    ha_trend: str = "neutral"  # bullish, bearish, neutral
    ha_consecutive: int = 0
    ha_strength: float = 0.0

    # VWAP
    vwap_distance_pct: float = 0.0
    vwap_position: str = "at"  # above, below, at


@dataclass
class PriceContext:
    """Price observation data."""
    current_price: float = 0.0
    target_price: float = 0.0
    price_high: float = 0.0
    price_low: float = 0.0
    price_velocity: float = 0.0


@dataclass
class MLInput:
    """
    All data ML needs to make a decision.

    This is the single data structure that flows from trading to ML.
    All feature extraction happens inside the ML implementation.
    """
    # Core trade info
    asset: str
    side: TradeSide
    entry_price: float
    edge: float
    time_remaining: float
    arb_type: str = "none"

    # Contexts (all optional - ML handles missing data gracefully)
    market: MarketContext = field(default_factory=MarketContext)
    trend: TrendContext = field(default_factory=TrendContext)
    chart: ChartContext = field(default_factory=ChartContext)
    indicators: IndicatorContext = field(default_factory=IndicatorContext)
    price: PriceContext = field(default_factory=PriceContext)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "asset": self.asset,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "edge": self.edge,
            "time_remaining": self.time_remaining,
            "arb_type": self.arb_type,
            "market": {
                "spread": self.market.spread,
                "bid_depth": self.market.bid_depth,
                "ask_depth": self.market.ask_depth,
                "volatility": self.market.volatility,
                "binance_lead_pct": self.market.binance_lead_pct,
                "binance_confirmation": self.market.binance_confirmation,
            },
            "trend": {
                "trend_1h": self.trend.trend_1h,
                "trend_4h": self.trend.trend_4h,
                "trend_1d": self.trend.trend_1d,
                "price_momentum": self.trend.price_momentum,
                "price_trend": self.trend.price_trend,
                "distance_from_target": self.trend.distance_from_target,
            },
            "chart": {
                "rsi": self.chart.rsi,
                "trend_strength": self.chart.trend_strength,
                "is_uptrend": self.chart.is_uptrend,
                "is_downtrend": self.chart.is_downtrend,
                "momentum": self.chart.momentum,
                "confidence": self.chart.confidence,
            },
            "indicators": {
                "macd_histogram": self.indicators.macd_histogram,
                "macd_crossover": self.indicators.macd_crossover,
                "bb_position": self.indicators.bb_position,
                "stoch_signal": self.indicators.stoch_signal,
                "volume_ratio": self.indicators.volume_ratio,
                "ha_trend": self.indicators.ha_trend,
            },
        }


@dataclass
class MLDecision:
    """
    ML's recommendation for a trade.

    This is what trading code receives back from ML.
    """
    should_trade: bool
    confidence: float  # 0.0 to 1.0
    reason: str

    # Optional details for logging/debugging
    model_type: str = "unknown"
    feature_importance: Optional[Dict[str, float]] = None

    def __str__(self) -> str:
        status = "APPROVE" if self.should_trade else "REJECT"
        return f"ML {status}: {self.confidence:.0%} - {self.reason}"


@dataclass
class MLStats:
    """ML model statistics."""
    training_samples: int = 0
    predictions_made: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    is_active: bool = False
    model_type: str = "unknown"
    min_confidence: float = 0.52

    # Feature importance (top features)
    top_features: Optional[Dict[str, float]] = None


class MLInterface(ABC):
    """
    Clean interface between trading logic and ML system.

    Trading code ONLY interacts with this interface.
    All ML implementation details are hidden behind it.

    Benefits:
    - Easy to test (mock the interface)
    - Easy to swap implementations
    - Clear separation of concerns
    - Single place for feature extraction
    """

    @abstractmethod
    def evaluate_trade(self, input: MLInput) -> MLDecision:
        """
        Evaluate whether to take a trade.

        This is the primary method trading code calls.

        Args:
            input: All market data and context for the trade

        Returns:
            MLDecision with should_trade, confidence, and reason
        """
        pass

    @abstractmethod
    def record_outcome(
        self,
        trade_id: str,
        input: MLInput,
        won: bool,
    ) -> None:
        """
        Record trade outcome for model learning.

        Called once when a trade closes (settles or exits early).

        Args:
            trade_id: Unique identifier for the trade
            input: The MLInput used when the trade was opened
            won: Whether the trade was profitable
        """
        pass

    @abstractmethod
    def get_stats(self) -> MLStats:
        """
        Get ML model statistics.

        Returns:
            MLStats with training samples, accuracy, etc.
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """
        Check if ML has trained enough to make predictions.

        Returns:
            True if model is ready to make predictions
        """
        pass

    @abstractmethod
    def save(self) -> None:
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self) -> bool:
        """
        Load model from disk.

        Returns:
            True if model was loaded successfully
        """
        pass


class NoOpML(MLInterface):
    """
    No-op ML implementation that approves all trades.

    Useful for:
    - Testing without ML
    - Running in rule-based-only mode
    - Debugging trading logic
    """

    def evaluate_trade(self, input: MLInput) -> MLDecision:
        return MLDecision(
            should_trade=True,
            confidence=0.5,
            reason="ML disabled (NoOp mode)",
            model_type="noop",
        )

    def record_outcome(self, trade_id: str, input: MLInput, won: bool) -> None:
        pass  # No-op

    def get_stats(self) -> MLStats:
        return MLStats(model_type="noop", is_active=False)

    def is_ready(self) -> bool:
        return True

    def save(self) -> None:
        pass

    def load(self) -> bool:
        return True
