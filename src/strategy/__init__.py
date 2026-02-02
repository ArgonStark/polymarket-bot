"""Strategy engine for signal generation and risk management."""

from .signals import SignalGenerator
from .risk import RiskManager
from .arbitrage import ArbitrageDetector
from .ml_predictor import (
    MLSignalPredictor,
    get_ml_predictor,
    calculate_price_trend,
    extract_ml_features_from_market,
)
from .auto_retrain import (
    AutoRetrainer,
    RetrainingConfig,
    get_auto_retrainer,
    init_auto_retrainer,
)
from .unified_signals import (
    UnifiedSignalGenerator,
    UnifiedSignal,
    MarketContext,
    SignalDirection,
    SignalStrength,
    generate_unified_signal,
    get_unified_signal_generator,
)

__all__ = [
    "SignalGenerator",
    "RiskManager",
    "ArbitrageDetector",
    "MLSignalPredictor",
    "get_ml_predictor",
    "calculate_price_trend",
    "extract_ml_features_from_market",
    "AutoRetrainer",
    "RetrainingConfig",
    "get_auto_retrainer",
    "init_auto_retrainer",
    # Unified signals
    "UnifiedSignalGenerator",
    "UnifiedSignal",
    "MarketContext",
    "SignalDirection",
    "SignalStrength",
    "generate_unified_signal",
    "get_unified_signal_generator",
]
