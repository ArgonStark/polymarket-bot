"""
Structured Trading Logger for Polymarket Bot.

Provides clean, organized logging that:
1. Groups related information into single log entries
2. Supports both human-readable and JSON formats
3. Uses appropriate log levels
4. Makes it easy to filter and analyze

Usage:
    from src.utils.trading_logger import TradingLogger

    tlog = TradingLogger("BTC")
    tlog.signal_analysis(direction="UP", confidence=0.75, indicators={...})
    tlog.trade_decision(action="BUY", reason="Strong signal", edge=0.08)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Any
from enum import Enum


# =============================================================================
# LOG LEVELS & FORMATS
# =============================================================================

class LogLevel(Enum):
    """Log levels for trading events."""
    TRACE = 5      # Very detailed debugging
    DEBUG = 10     # Indicator calculations
    INFO = 20      # Normal operations
    SIGNAL = 25    # Trading signals (custom level)
    WARNING = 30   # Potential issues
    ERROR = 40     # Errors
    CRITICAL = 50  # Critical failures


# Add custom SIGNAL level
logging.addLevelName(25, "SIGNAL")


def signal(self, message, *args, **kwargs):
    """Log at SIGNAL level (between INFO and WARNING)."""
    if self.isEnabledFor(25):
        self._log(25, message, *args, **kwargs)


logging.Logger.signal = signal


# =============================================================================
# STRUCTURED LOG ENTRIES
# =============================================================================

@dataclass
class SignalLog:
    """Structured log entry for trading signal analysis."""
    asset: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Market context
    price: float = 0.0
    target: float = 0.0
    time_remaining: float = 0.0

    # Signal
    direction: str = ""  # UP, DOWN, NEUTRAL
    strength: str = ""   # STRONG, MODERATE, WEAK, NONE
    confidence: float = 0.0

    # Strategy
    context: str = ""    # TRENDING, RANGING, etc.
    strategy: str = ""   # trend_following, mean_reversion

    # Edges
    edge_up: float = 0.0
    edge_down: float = 0.0
    edge_adjustment: float = 0.0

    # Key indicators (summarized)
    rsi: float = 50.0
    macd: str = ""       # bullish, bearish, neutral
    trend: str = ""      # up, down, sideways
    volume: str = ""     # high, normal, low

    # Reasons
    primary_reason: str = ""
    supporting: list = field(default_factory=list)
    contradicting: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    def to_human(self) -> str:
        """Format as human-readable single line."""
        direction_icon = {"UP": "↑", "DOWN": "↓", "NEUTRAL": "→"}.get(self.direction, "?")
        strength_stars = {"STRONG": "★★★", "MODERATE": "★★☆", "WEAK": "★☆☆", "NONE": "☆☆☆"}.get(self.strength, "")

        parts = [
            f"[{self.asset}]",
            f"{direction_icon} {self.direction}",
            f"{strength_stars}",
            f"conf={self.confidence:.0%}",
            f"ctx={self.context}",
        ]

        if self.edge_up > 0 or self.edge_down > 0:
            parts.append(f"edge=↑{self.edge_up:.1%}/↓{self.edge_down:.1%}")

        if self.primary_reason:
            parts.append(f"| {self.primary_reason}")

        return " ".join(parts)


@dataclass
class TradeLog:
    """Structured log entry for trade execution."""
    asset: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Trade details
    action: str = ""     # BUY, SELL, SKIP
    side: str = ""       # UP, DOWN
    price: float = 0.0
    size_usd: float = 0.0
    size_shares: float = 0.0

    # Signal info
    edge: float = 0.0
    confidence: float = 0.0

    # Reason
    reason: str = ""
    signal_summary: str = ""

    def to_human(self) -> str:
        action_icon = {"BUY": "💰", "SELL": "💸", "SKIP": "⏭️"}.get(self.action, "❓")

        if self.action == "SKIP":
            return f"{action_icon} [{self.asset}] SKIP | {self.reason}"

        return (
            f"{action_icon} [{self.asset}] {self.action} {self.side} "
            f"${self.size_usd:.2f} @ {self.price:.2f} | "
            f"edge={self.edge:.1%} | {self.reason}"
        )


@dataclass
class IndicatorLog:
    """Structured log for indicator values (debug level)."""
    asset: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Trends
    trend_1m: float = 0.0
    trend_5m: float = 0.0
    trend_15m: float = 0.0
    trend_1h: float = 0.0
    trend_4h: float = 0.0

    # Oscillators
    rsi: float = 50.0
    stoch_k: float = 50.0
    stoch_d: float = 50.0

    # MACD
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    macd_crossover: str = ""

    # Bollinger
    bb_position: str = ""
    bb_bandwidth: float = 0.0

    # Volume
    volume_ratio: float = 1.0
    obv_trend: float = 0.0

    # Heiken Ashi
    ha_trend: str = ""
    ha_consecutive: int = 0

    # VWAP
    vwap_position: str = ""
    vwap_distance: float = 0.0

    # Patterns
    pattern: str = ""

    def to_human(self) -> str:
        """Compact indicator summary."""
        trend_arrow = "↑" if self.trend_15m > 0.2 else "↓" if self.trend_15m < -0.2 else "→"

        parts = [
            f"[{self.asset}]",
            f"Trend:{trend_arrow}{self.trend_15m:+.2f}",
            f"RSI:{self.rsi:.0f}",
            f"Stoch:{self.stoch_k:.0f}",
        ]

        if self.macd_crossover:
            parts.append(f"MACD:{self.macd_crossover}")

        if self.bb_position and self.bb_position != "middle":
            parts.append(f"BB:{self.bb_position}")

        if self.volume_ratio > 1.5:
            parts.append(f"Vol:{self.volume_ratio:.1f}x")

        if self.pattern:
            parts.append(f"Pattern:{self.pattern}")

        return " | ".join(parts)


# =============================================================================
# TRADING LOGGER CLASS
# =============================================================================

class TradingLogger:
    """
    Structured logger for trading operations.

    Consolidates multiple log calls into single, structured entries.
    Supports both human-readable and JSON output.
    """

    def __init__(
        self,
        asset: str = "",
        logger_name: str = "trading",
        json_output: bool = False,
    ):
        self.asset = asset
        self.json_output = json_output
        self.logger = logging.getLogger(logger_name)

        # Accumulated data for batch logging
        self._signal_data: dict = {}
        self._indicator_data: dict = {}

    def set_asset(self, asset: str):
        """Set the current asset being processed."""
        self.asset = asset
        self._signal_data = {}
        self._indicator_data = {}

    # -------------------------------------------------------------------------
    # Signal Logging
    # -------------------------------------------------------------------------

    def signal_analysis(
        self,
        direction: str,
        strength: str,
        confidence: float,
        context: str = "",
        strategy: str = "",
        edge_up: float = 0.0,
        edge_down: float = 0.0,
        edge_adjustment: float = 0.0,
        primary_reason: str = "",
        supporting: list = None,
        contradicting: list = None,
        price: float = 0.0,
        target: float = 0.0,
        time_remaining: float = 0.0,
        **extra_indicators,
    ):
        """
        Log a complete signal analysis in one call.

        This replaces multiple individual log calls with a single structured entry.
        """
        log_entry = SignalLog(
            asset=self.asset,
            direction=direction,
            strength=strength,
            confidence=confidence,
            context=context,
            strategy=strategy,
            edge_up=edge_up,
            edge_down=edge_down,
            edge_adjustment=edge_adjustment,
            primary_reason=primary_reason,
            supporting=supporting or [],
            contradicting=contradicting or [],
            price=price,
            target=target,
            time_remaining=time_remaining,
            rsi=extra_indicators.get("rsi", 50.0),
            macd=extra_indicators.get("macd", ""),
            trend=extra_indicators.get("trend", ""),
            volume=extra_indicators.get("volume", ""),
        )

        if self.json_output:
            self.logger.log(25, log_entry.to_json())
        else:
            self.logger.log(25, f"📊 SIGNAL {log_entry.to_human()}")

    def signal_skip(self, reason: str, edge_up: float = 0.0, edge_down: float = 0.0):
        """Log a skipped signal with reason."""
        msg = f"⏭️ [{self.asset}] SKIP | {reason} | edges: ↑{edge_up:.1%}/↓{edge_down:.1%}"
        self.logger.info(msg)

    def signal_pause(self, reason: str, resume_conditions: str = ""):
        """Log a market pause."""
        msg = f"⏸️ [{self.asset}] PAUSE | {reason}"
        if resume_conditions:
            msg += f" | Resume when: {resume_conditions}"
        self.logger.warning(msg)

    # -------------------------------------------------------------------------
    # Trade Logging
    # -------------------------------------------------------------------------

    def trade_decision(
        self,
        action: str,
        side: str = "",
        price: float = 0.0,
        size_usd: float = 0.0,
        edge: float = 0.0,
        confidence: float = 0.0,
        reason: str = "",
    ):
        """Log a trade decision."""
        log_entry = TradeLog(
            asset=self.asset,
            action=action,
            side=side,
            price=price,
            size_usd=size_usd,
            edge=edge,
            confidence=confidence,
            reason=reason,
        )

        if self.json_output:
            self.logger.log(25, log_entry.to_json())
        else:
            self.logger.log(25, log_entry.to_human())

    def trade_executed(
        self,
        side: str,
        price: float,
        size_usd: float,
        order_id: str = "",
    ):
        """Log successful trade execution."""
        self.logger.info(
            f"✅ [{self.asset}] EXECUTED {side} ${size_usd:.2f} @ {price:.3f} | order={order_id}"
        )

    def trade_failed(self, reason: str, error: str = ""):
        """Log failed trade."""
        msg = f"❌ [{self.asset}] FAILED | {reason}"
        if error:
            msg += f" | error={error}"
        self.logger.error(msg)

    # -------------------------------------------------------------------------
    # Indicator Logging (Debug Level)
    # -------------------------------------------------------------------------

    def indicators(
        self,
        trend_15m: float = 0.0,
        trend_1h: float = 0.0,
        trend_4h: float = 0.0,
        rsi: float = 50.0,
        stoch_k: float = 50.0,
        macd_crossover: str = "",
        bb_position: str = "",
        volume_ratio: float = 1.0,
        ha_trend: str = "",
        pattern: str = "",
        **extra,
    ):
        """Log indicator values at DEBUG level."""
        log_entry = IndicatorLog(
            asset=self.asset,
            trend_15m=trend_15m,
            trend_1h=trend_1h,
            trend_4h=trend_4h,
            rsi=rsi,
            stoch_k=stoch_k,
            macd_crossover=macd_crossover,
            bb_position=bb_position,
            volume_ratio=volume_ratio,
            ha_trend=ha_trend,
            pattern=pattern,
        )

        self.logger.debug(f"📈 INDICATORS {log_entry.to_human()}")

    # -------------------------------------------------------------------------
    # Context Logging
    # -------------------------------------------------------------------------

    def market_context(
        self,
        context_type: str,
        trend_strength: float = 0.0,
        alignment: float = 0.0,
        uncertainty: float = 0.0,
    ):
        """Log market context determination."""
        self.logger.debug(
            f"🌍 [{self.asset}] CONTEXT {context_type} | "
            f"trend={trend_strength:.0%} align={alignment:.0%} uncertain={uncertainty:.0%}"
        )

    def timeframe_conflict(
        self,
        trend_15m: float,
        trend_1h: float,
        trend_4h: float,
        alignment: float,
    ):
        """Log timeframe conflict warning."""
        self.logger.warning(
            f"⚠️ [{self.asset}] TF CONFLICT | "
            f"15m={trend_15m:+.2f} 1h={trend_1h:+.2f} 4h={trend_4h:+.2f} | "
            f"alignment={alignment:.0%}"
        )

    # -------------------------------------------------------------------------
    # Strategy Logging
    # -------------------------------------------------------------------------

    def strategy_selected(self, strategy: str, reason: str = ""):
        """Log which strategy was selected."""
        self.logger.debug(f"🎯 [{self.asset}] STRATEGY {strategy} | {reason}")

    def edge_adjustment(
        self,
        direction: str,
        base_edge: float,
        adjustment: float,
        final_edge: float,
        reason: str = "",
    ):
        """Log edge adjustment."""
        self.logger.debug(
            f"📐 [{self.asset}] EDGE {direction} | "
            f"base={base_edge:.1%} adj={adjustment:+.1%} final={final_edge:.1%} | {reason}"
        )

    # -------------------------------------------------------------------------
    # Risk Logging
    # -------------------------------------------------------------------------

    def risk_check(self, passed: bool, checks: dict):
        """Log risk check results."""
        status = "✅ PASS" if passed else "❌ FAIL"
        failed = [k for k, v in checks.items() if not v]
        self.logger.info(
            f"🛡️ [{self.asset}] RISK {status}" +
            (f" | failed: {', '.join(failed)}" if failed else "")
        )

    def position_sizing(self, multiplier: float, reason: str = ""):
        """Log position sizing decision."""
        self.logger.debug(
            f"📏 [{self.asset}] SIZE {multiplier:.0%} of normal | {reason}"
        )

    # -------------------------------------------------------------------------
    # Performance Logging
    # -------------------------------------------------------------------------

    def trade_result(
        self,
        outcome: str,  # WIN, LOSS
        pnl: float,
        entry_price: float,
        exit_price: float,
        hold_time: float = 0.0,
    ):
        """Log trade result."""
        icon = "🟢" if outcome == "WIN" else "🔴"
        self.logger.info(
            f"{icon} [{self.asset}] {outcome} | "
            f"PnL=${pnl:+.2f} | entry={entry_price:.3f} exit={exit_price:.3f} | "
            f"hold={hold_time:.0f}s"
        )

    def session_summary(
        self,
        trades: int,
        wins: int,
        losses: int,
        total_pnl: float,
        win_rate: float,
    ):
        """Log session summary."""
        self.logger.info(
            f"📋 SESSION SUMMARY | "
            f"trades={trades} W={wins} L={losses} | "
            f"PnL=${total_pnl:+.2f} | WR={win_rate:.0%}"
        )


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_trading_logger: Optional[TradingLogger] = None


def get_trading_logger(asset: str = "") -> TradingLogger:
    """Get the global trading logger instance."""
    global _trading_logger
    if _trading_logger is None:
        _trading_logger = TradingLogger()
    if asset:
        _trading_logger.set_asset(asset)
    return _trading_logger


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def configure_trading_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: str = None,
):
    """
    Configure trading logging.

    Args:
        level: Minimum log level (DEBUG, INFO, SIGNAL, WARNING, ERROR)
        json_output: If True, output JSON instead of human-readable
        log_file: Optional file path for logging
    """
    logger = logging.getLogger("trading")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)

        # Simple format - the TradingLogger already formats messages nicely
        formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        console.setFormatter(formatter)
        logger.addHandler(console)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s|%(levelname)s|%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Update global logger
    global _trading_logger
    if _trading_logger:
        _trading_logger.json_output = json_output
