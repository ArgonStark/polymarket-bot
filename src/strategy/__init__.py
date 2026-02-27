"""Strategy engine for signal generation and risk management."""

from .signals import SignalGenerator
from .risk import RiskManager
from .arbitrage import ArbitrageDetector
from .regime import RegimeDetector, RegimeState, RegimeType
from .edge_signal import generate_edge_signal, EdgeSignal, Conviction, EdgeSignalConfig
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
    # Unified signals
    "UnifiedSignalGenerator",
    "UnifiedSignal",
    "MarketContext",
    "SignalDirection",
    "SignalStrength",
    "generate_unified_signal",
    "get_unified_signal_generator",
    # Regime + Edge signal
    "RegimeDetector",
    "RegimeState",
    "RegimeType",
    "generate_edge_signal",
    "EdgeSignal",
    "Conviction",
    "EdgeSignalConfig",
]
