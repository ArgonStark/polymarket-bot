"""
ML Module

Clean interface for machine learning predictions.

Usage:
    from src.ml import MLInterface, MLInput, MLDecision, MLPredictor

    # Create predictor
    ml = MLPredictor(min_confidence=0.52, min_samples=30)

    # Or use no-op for testing
    ml = NoOpML()

    # Evaluate trade
    decision = ml.evaluate_trade(ml_input)
    if decision.should_trade:
        execute_trade()

    # Record outcome
    ml.record_outcome(trade_id, ml_input, won=True)
"""

from .interface import (
    # Core types
    MLInterface,
    MLInput,
    MLDecision,
    MLStats,
    TradeSide,
    # Context types
    MarketContext,
    TrendContext,
    ChartContext,
    IndicatorContext,
    PriceContext,
    # No-op implementation
    NoOpML,
)

from .predictor import (
    MLPredictor,
    create_ml_input_from_signal,
)

__all__ = [
    # Interface
    "MLInterface",
    "MLInput",
    "MLDecision",
    "MLStats",
    "TradeSide",
    # Contexts
    "MarketContext",
    "TrendContext",
    "ChartContext",
    "IndicatorContext",
    "PriceContext",
    # Implementations
    "MLPredictor",
    "NoOpML",
    # Helpers
    "create_ml_input_from_signal",
]
