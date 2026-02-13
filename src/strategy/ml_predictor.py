"""
Machine Learning Signal Quality Predictor

Learns from trade history to predict which signals are likely to win.
Uses a simple but effective approach that works without heavy ML dependencies.

ENHANCED: Now includes features for:
- Arbitrage signal types (binary, asymmetric, dump, hedge)
- Market spread and order book depth
- Price trend from historical data
- Distance from target price
- Binance price lead (faster indicator)
- Binance confirmation signals (STRONG, MEDIUM, WEAK, NONE)
- Better integration with all bot components
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Arbitrage signal types for one-hot encoding
ARB_TYPES = ["none", "binary_arb", "asymmetric", "dump", "hedge"]

# Binance confirmation types for one-hot encoding
BINANCE_CONF_TYPES = ["NONE", "WEAK", "MEDIUM", "STRONG"]


@dataclass
class TradeFeatures:
    """
    Features extracted from a trade signal.

    ENHANCED with additional features:
    - arb_type: Type of arbitrage opportunity detected
    - spread: Market bid-ask spread
    - bid_depth: Order book bid depth
    - ask_depth: Order book ask depth
    - price_trend: Short-term price trend from historical data
    - distance_from_target: How far current price is from target
    - binance_lead_pct: Binance price lead over Chainlink (faster indicator)
    - binance_confirmation: Confirmation signal type (STRONG, MEDIUM, WEAK, NONE)
    - trend_1h/4h/1d: Multi-timeframe trend context for better decision making
    - price_normalized: Current asset price (normalized 0-1)
    - price_above_target: Whether price is above target (binary)
    - price_range_position: Where price sits in recent high/low range (0-1)
    - price_velocity: How fast price is moving (magnitude)

    CHART ANALYSIS FEATURES (from Binance candlestick data):
    - chart_rsi: Relative Strength Index (0-100)
    - chart_trend_strength: How strong the current trend is (0-1)
    - chart_market_type: Encoded market type (trending/ranging)
    - chart_trend_change: Encoded trend change signal
    - chart_momentum: Short-term momentum (-1 to +1)
    - chart_bias: Overall chart bias (bullish/bearish/neutral)
    - chart_pattern: Candlestick pattern detected
    """
    # Core features
    edge: float  # Expected edge/profit margin
    time_remaining: float  # Seconds until market closes
    volatility: float  # Asset volatility estimate
    price_momentum: float  # Recent price change direction (-1 to 1)
    hour_of_day: int  # 0-23 UTC
    day_of_week: int  # 0-6 (Monday=0)
    asset: str  # BTC, ETH, SOL, XRP
    side: str  # UP or DOWN

    # Enhanced features
    arb_type: str = "none"  # Type of arbitrage (none, binary_arb, asymmetric, dump, hedge)
    spread: float = 0.0  # Market bid-ask spread
    bid_depth: float = 0.0  # Order book bid depth
    ask_depth: float = 0.0  # Order book ask depth
    price_trend: float = 0.0  # Short-term price trend (-1 to 1)
    distance_from_target: float = 0.0  # Normalized distance from target price

    # Market variant (5-min vs 15-min — different dynamics)
    is_five_min: float = 0.0  # 1.0 if 5-min market, 0.0 if 15-min

    # Binance confirmation features (leading indicator)
    binance_lead_pct: float = 0.0  # (Binance - Chainlink) / Chainlink
    binance_confirmation: str = "NONE"  # STRONG, MEDIUM, WEAK, NONE

    # Multi-timeframe trends (context for ML decision making)
    # These help the model understand broader market direction
    trend_1h: float = 0.0   # 1-hour trend (-1 to +1)
    trend_4h: float = 0.0   # 4-hour trend (-1 to +1)
    trend_1d: float = 0.0   # 1-day trend (-1 to +1)

    # Price observation features (helps ML understand price context)
    price_normalized: float = 0.0  # Current price normalized (0-1 scale per asset)
    price_above_target: float = 0.0  # 1.0 if price >= target, 0.0 otherwise
    price_range_position: float = 0.5  # Position in recent range (0=low, 1=high)
    price_velocity: float = 0.0  # Speed of price movement (normalized)

    # Chart analysis features (from Binance candlestick technical analysis)
    chart_rsi: float = 50.0  # RSI (0-100)
    chart_trend_strength: float = 0.0  # Trend strength (0-1)
    chart_is_uptrend: float = 0.0  # 1 if uptrend, 0 otherwise
    chart_is_downtrend: float = 0.0  # 1 if downtrend, 0 otherwise
    chart_is_ranging: float = 0.0  # 1 if ranging market, 0 otherwise
    chart_bullish_reversal: float = 0.0  # 1 if bullish reversal detected
    chart_bearish_reversal: float = 0.0  # 1 if bearish reversal detected
    chart_momentum: float = 0.0  # Short-term momentum (-1 to +1)
    chart_bias_bullish: float = 0.0  # 1 if overall bias is bullish
    chart_bias_bearish: float = 0.0  # 1 if overall bias is bearish
    chart_confidence: float = 0.5  # Confidence in chart analysis (0-1)
    chart_bullish_pattern: float = 0.0  # 1 if bullish pattern detected
    chart_bearish_pattern: float = 0.0  # 1 if bearish pattern detected

    # Advanced chart analysis features (graduated sizing, multi-TF, trend break)
    chart_uncertainty_score: float = 0.0  # Market uncertainty (0-1)
    chart_position_multiplier: float = 1.0  # Graduated position size (0-1)
    chart_timeframe_aligned: float = 1.0  # 1 if 15m/1h/4h aligned, 0 otherwise
    chart_alignment_score: float = 1.0  # Multi-TF alignment (0-1)
    chart_trend_breaking: float = 0.0  # 1 if trend is breaking down
    chart_resume_ready: float = 1.0  # 1 if safe to trade, 0 if in pause
    chart_resume_confidence: float = 1.0  # Confidence in resuming (0-1)

    # === NEW INDICATOR FEATURES (adds 29 features, total 83 with variant) ===

    # MACD features (4)
    macd_histogram: float = 0.0  # Normalized histogram (-1 to +1)
    macd_crossover_bullish: float = 0.0  # 1 if bullish crossover
    macd_crossover_bearish: float = 0.0  # 1 if bearish crossover
    macd_crossover_none: float = 1.0  # 1 if no crossover

    # Bollinger Bands features (4)
    bb_bandwidth: float = 0.0  # Normalized bandwidth (0-1)
    bb_position_above: float = 0.0  # 1 if price above upper band
    bb_position_below: float = 0.0  # 1 if price below lower band
    bb_position_middle: float = 1.0  # 1 if price in middle

    # Stochastic features (5)
    stoch_k: float = 0.5  # Stochastic %K (0-1)
    stoch_d: float = 0.5  # Stochastic %D (0-1)
    stoch_overbought: float = 0.0  # 1 if overbought
    stoch_oversold: float = 0.0  # 1 if oversold
    stoch_neutral: float = 1.0  # 1 if neutral

    # RSI Divergence features (4)
    rsi_div_bullish: float = 0.0  # 1 if bullish divergence
    rsi_div_bearish: float = 0.0  # 1 if bearish divergence
    rsi_div_none: float = 1.0  # 1 if no divergence
    rsi_div_strength: float = 0.0  # Divergence strength (0-1)

    # Volume features (3)
    volume_ratio: float = 1.0  # Current / Average volume
    is_high_volume: float = 0.0  # 1 if high volume
    obv_trend: float = 0.0  # On-balance volume trend (-1 to +1)

    # Heiken Ashi features (5)
    ha_trend_bullish: float = 0.0  # 1 if HA bullish
    ha_trend_bearish: float = 0.0  # 1 if HA bearish
    ha_trend_neutral: float = 1.0  # 1 if HA neutral
    ha_consecutive: float = 0.0  # Consecutive same-color candles (normalized 0-1)
    ha_strength: float = 0.0  # HA trend strength (0-1)

    # VWAP features (4)
    vwap_distance_pct: float = 0.0  # Distance from VWAP (normalized)
    vwap_position_above: float = 0.0  # 1 if above VWAP
    vwap_position_below: float = 0.0  # 1 if below VWAP
    vwap_position_at: float = 1.0  # 1 if at VWAP

    def to_vector(self) -> list[float]:
        """Convert to feature vector for model (83 features total)."""
        # Normalize features to roughly 0-1 range
        # NOTE: hour_of_day and day_of_week are set to neutral (0.5) because
        # they don't predict crypto price direction - they're just noise.
        vector = [
            # Core features (6)
            max(0.0, min(self.edge * 10, 1.0)),  # edge 0.1 -> 1.0, clamp negatives to 0
            min(self.time_remaining / 900, 1.0),  # 900s -> 1.0
            min(self.volatility * 100, 1.0),  # 0.01 -> 1.0
            (self.price_momentum + 1) / 2,  # -1,1 -> 0,1
            0.5,  # hour_of_day - NEUTRALIZED (doesn't predict price direction)
            0.5,  # day_of_week - NEUTRALIZED (doesn't predict price direction)
            # Asset one-hot (4)
            1.0 if self.asset == "BTC" else 0.0,
            1.0 if self.asset == "ETH" else 0.0,
            1.0 if self.asset == "SOL" else 0.0,
            1.0 if self.asset == "XRP" else 0.0,
            # Side (1)
            1.0 if self.side == "UP" else 0.0,
            # Market variant (1) — 5-min markets have different dynamics
            self.is_five_min,
            # Arb type one-hot (5)
            1.0 if self.arb_type == "none" else 0.0,
            1.0 if self.arb_type == "binary_arb" else 0.0,
            1.0 if self.arb_type == "asymmetric" else 0.0,
            1.0 if self.arb_type == "dump" else 0.0,
            1.0 if self.arb_type == "hedge" else 0.0,
            # Market features (4)
            min(self.spread * 10, 1.0),  # spread 0.1 -> 1.0
            min(self.bid_depth / 10000, 1.0),  # depth 10k -> 1.0
            min(self.ask_depth / 10000, 1.0),
            (self.price_trend + 1) / 2,  # -1,1 -> 0,1
            # Distance from target (1)
            min(abs(self.distance_from_target) * 100, 1.0),  # 1% -> 1.0
            # Binance confirmation features (5)
            # Lead percentage normalized: -1% to +1% -> 0 to 1
            (max(-0.01, min(0.01, self.binance_lead_pct)) + 0.01) / 0.02,
            # Confirmation type one-hot (4)
            1.0 if self.binance_confirmation == "NONE" else 0.0,
            1.0 if self.binance_confirmation == "WEAK" else 0.0,
            1.0 if self.binance_confirmation == "MEDIUM" else 0.0,
            1.0 if self.binance_confirmation == "STRONG" else 0.0,
            # Multi-timeframe trends (3) - normalized from -1,1 to 0,1
            (self.trend_1h + 1) / 2,
            (self.trend_4h + 1) / 2,
            (self.trend_1d + 1) / 2,
            # Price observation features (4)
            min(max(self.price_normalized, 0.0), 1.0),  # Already 0-1
            self.price_above_target,  # Binary 0 or 1
            min(max(self.price_range_position, 0.0), 1.0),  # 0-1 range position
            min(abs(self.price_velocity) * 10, 1.0),  # Velocity normalized (0.1 = 1.0)
            # Chart analysis features (13) - from Binance candlestick analysis
            self.chart_rsi / 100.0,  # RSI normalized 0-1
            min(max(self.chart_trend_strength, 0.0), 1.0),  # Trend strength 0-1
            self.chart_is_uptrend,  # Binary
            self.chart_is_downtrend,  # Binary
            self.chart_is_ranging,  # Binary
            self.chart_bullish_reversal,  # Binary
            self.chart_bearish_reversal,  # Binary
            (self.chart_momentum + 1) / 2,  # -1,1 -> 0,1
            self.chart_bias_bullish,  # Binary
            self.chart_bias_bearish,  # Binary
            min(max(self.chart_confidence, 0.0), 1.0),  # Confidence 0-1
            self.chart_bullish_pattern,  # Binary
            self.chart_bearish_pattern,  # Binary
            # Advanced chart features (7) - graduated sizing, multi-TF, trend break
            min(max(self.chart_uncertainty_score, 0.0), 1.0),  # Uncertainty 0-1
            min(max(self.chart_position_multiplier, 0.0), 1.0),  # Position multiplier 0-1
            self.chart_timeframe_aligned,  # Binary (1 if aligned)
            min(max(self.chart_alignment_score, 0.0), 1.0),  # Alignment 0-1
            self.chart_trend_breaking,  # Binary (1 if breaking)
            self.chart_resume_ready,  # Binary (1 if ready to trade)
            min(max(self.chart_resume_confidence, 0.0), 1.0),  # Resume confidence 0-1

            # === NEW INDICATOR FEATURES (29 features) ===

            # MACD features (4)
            (self.macd_histogram + 1) / 2,  # Normalize -1,1 -> 0,1
            self.macd_crossover_bullish,  # Binary
            self.macd_crossover_bearish,  # Binary
            self.macd_crossover_none,  # Binary

            # Bollinger Bands features (4)
            min(max(self.bb_bandwidth / 10, 0.0), 1.0),  # Normalize bandwidth (10% = 1.0)
            self.bb_position_above,  # Binary
            self.bb_position_below,  # Binary
            self.bb_position_middle,  # Binary

            # Stochastic features (5)
            self.stoch_k / 100.0,  # Normalize 0-100 -> 0-1
            self.stoch_d / 100.0,  # Normalize 0-100 -> 0-1
            self.stoch_overbought,  # Binary
            self.stoch_oversold,  # Binary
            self.stoch_neutral,  # Binary

            # RSI Divergence features (4)
            self.rsi_div_bullish,  # Binary
            self.rsi_div_bearish,  # Binary
            self.rsi_div_none,  # Binary
            min(max(self.rsi_div_strength, 0.0), 1.0),  # 0-1

            # Volume features (3)
            min(self.volume_ratio / 3.0, 1.0),  # Normalize (3x volume = 1.0)
            self.is_high_volume,  # Binary
            (self.obv_trend + 1) / 2,  # Normalize -1,1 -> 0,1

            # Heiken Ashi features (5)
            self.ha_trend_bullish,  # Binary
            self.ha_trend_bearish,  # Binary
            self.ha_trend_neutral,  # Binary
            min(self.ha_consecutive / 10.0, 1.0),  # Normalize (10 candles = 1.0)
            min(max(self.ha_strength, 0.0), 1.0),  # 0-1

            # VWAP features (4)
            (min(max(self.vwap_distance_pct * 100, -1.0), 1.0) + 1) / 2,  # -1% to +1% -> 0,1
            self.vwap_position_above,  # Binary
            self.vwap_position_below,  # Binary
            self.vwap_position_at,  # Binary
        ]
        return vector

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "edge": self.edge,
            "time_remaining": self.time_remaining,
            "volatility": self.volatility,
            "price_momentum": self.price_momentum,
            "hour_of_day": self.hour_of_day,
            "day_of_week": self.day_of_week,
            "asset": self.asset,
            "side": self.side,
            "arb_type": self.arb_type,
            "spread": self.spread,
            "bid_depth": self.bid_depth,
            "ask_depth": self.ask_depth,
            "price_trend": self.price_trend,
            "distance_from_target": self.distance_from_target,
            "binance_lead_pct": self.binance_lead_pct,
            "binance_confirmation": self.binance_confirmation,
            "trend_1h": self.trend_1h,
            "trend_4h": self.trend_4h,
            "trend_1d": self.trend_1d,
            "price_normalized": self.price_normalized,
            "price_above_target": self.price_above_target,
            "price_range_position": self.price_range_position,
            "price_velocity": self.price_velocity,
            # Chart analysis features
            "chart_rsi": self.chart_rsi,
            "chart_trend_strength": self.chart_trend_strength,
            "chart_is_uptrend": self.chart_is_uptrend,
            "chart_is_downtrend": self.chart_is_downtrend,
            "chart_is_ranging": self.chart_is_ranging,
            "chart_bullish_reversal": self.chart_bullish_reversal,
            "chart_bearish_reversal": self.chart_bearish_reversal,
            "chart_momentum": self.chart_momentum,
            "chart_bias_bullish": self.chart_bias_bullish,
            "chart_bias_bearish": self.chart_bias_bearish,
            "chart_confidence": self.chart_confidence,
            "chart_bullish_pattern": self.chart_bullish_pattern,
            "chart_bearish_pattern": self.chart_bearish_pattern,
            # Advanced chart features
            "chart_uncertainty_score": self.chart_uncertainty_score,
            "chart_position_multiplier": self.chart_position_multiplier,
            "chart_timeframe_aligned": self.chart_timeframe_aligned,
            "chart_alignment_score": self.chart_alignment_score,
            "chart_trend_breaking": self.chart_trend_breaking,
            "chart_resume_ready": self.chart_resume_ready,
            "chart_resume_confidence": self.chart_resume_confidence,
            # New indicator features
            "macd_histogram": self.macd_histogram,
            "macd_crossover_bullish": self.macd_crossover_bullish,
            "macd_crossover_bearish": self.macd_crossover_bearish,
            "macd_crossover_none": self.macd_crossover_none,
            "bb_bandwidth": self.bb_bandwidth,
            "bb_position_above": self.bb_position_above,
            "bb_position_below": self.bb_position_below,
            "bb_position_middle": self.bb_position_middle,
            "stoch_k": self.stoch_k,
            "stoch_d": self.stoch_d,
            "stoch_overbought": self.stoch_overbought,
            "stoch_oversold": self.stoch_oversold,
            "stoch_neutral": self.stoch_neutral,
            "rsi_div_bullish": self.rsi_div_bullish,
            "rsi_div_bearish": self.rsi_div_bearish,
            "rsi_div_none": self.rsi_div_none,
            "rsi_div_strength": self.rsi_div_strength,
            "volume_ratio": self.volume_ratio,
            "is_high_volume": self.is_high_volume,
            "obv_trend": self.obv_trend,
            "ha_trend_bullish": self.ha_trend_bullish,
            "ha_trend_bearish": self.ha_trend_bearish,
            "ha_trend_neutral": self.ha_trend_neutral,
            "ha_consecutive": self.ha_consecutive,
            "ha_strength": self.ha_strength,
            "vwap_distance_pct": self.vwap_distance_pct,
            "vwap_position_above": self.vwap_position_above,
            "vwap_position_below": self.vwap_position_below,
            "vwap_position_at": self.vwap_position_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TradeFeatures":
        """Create from dictionary."""
        return cls(
            edge=data["edge"],
            time_remaining=data["time_remaining"],
            volatility=data["volatility"],
            price_momentum=data["price_momentum"],
            hour_of_day=data["hour_of_day"],
            day_of_week=data["day_of_week"],
            asset=data["asset"],
            side=data["side"],
            arb_type=data.get("arb_type", "none"),
            spread=data.get("spread", 0.0),
            bid_depth=data.get("bid_depth", 0.0),
            ask_depth=data.get("ask_depth", 0.0),
            price_trend=data.get("price_trend", 0.0),
            distance_from_target=data.get("distance_from_target", 0.0),
            binance_lead_pct=data.get("binance_lead_pct", 0.0),
            binance_confirmation=data.get("binance_confirmation", "NONE"),
            trend_1h=data.get("trend_1h", 0.0),
            trend_4h=data.get("trend_4h", 0.0),
            trend_1d=data.get("trend_1d", 0.0),
            price_normalized=data.get("price_normalized", 0.0),
            price_above_target=data.get("price_above_target", 0.0),
            price_range_position=data.get("price_range_position", 0.5),
            price_velocity=data.get("price_velocity", 0.0),
            # Chart analysis features
            chart_rsi=data.get("chart_rsi", 50.0),
            chart_trend_strength=data.get("chart_trend_strength", 0.0),
            chart_is_uptrend=data.get("chart_is_uptrend", 0.0),
            chart_is_downtrend=data.get("chart_is_downtrend", 0.0),
            chart_is_ranging=data.get("chart_is_ranging", 0.0),
            chart_bullish_reversal=data.get("chart_bullish_reversal", 0.0),
            chart_bearish_reversal=data.get("chart_bearish_reversal", 0.0),
            chart_momentum=data.get("chart_momentum", 0.0),
            chart_bias_bullish=data.get("chart_bias_bullish", 0.0),
            chart_bias_bearish=data.get("chart_bias_bearish", 0.0),
            chart_confidence=data.get("chart_confidence", 0.5),
            chart_bullish_pattern=data.get("chart_bullish_pattern", 0.0),
            chart_bearish_pattern=data.get("chart_bearish_pattern", 0.0),
            # Advanced chart features
            chart_uncertainty_score=data.get("chart_uncertainty_score", 0.0),
            chart_position_multiplier=data.get("chart_position_multiplier", 1.0),
            chart_timeframe_aligned=data.get("chart_timeframe_aligned", 1.0),
            chart_alignment_score=data.get("chart_alignment_score", 1.0),
            chart_trend_breaking=data.get("chart_trend_breaking", 0.0),
            chart_resume_ready=data.get("chart_resume_ready", 1.0),
            chart_resume_confidence=data.get("chart_resume_confidence", 1.0),
            # New indicator features
            macd_histogram=data.get("macd_histogram", 0.0),
            macd_crossover_bullish=data.get("macd_crossover_bullish", 0.0),
            macd_crossover_bearish=data.get("macd_crossover_bearish", 0.0),
            macd_crossover_none=data.get("macd_crossover_none", 1.0),
            bb_bandwidth=data.get("bb_bandwidth", 0.0),
            bb_position_above=data.get("bb_position_above", 0.0),
            bb_position_below=data.get("bb_position_below", 0.0),
            bb_position_middle=data.get("bb_position_middle", 1.0),
            stoch_k=data.get("stoch_k", 50.0),
            stoch_d=data.get("stoch_d", 50.0),
            stoch_overbought=data.get("stoch_overbought", 0.0),
            stoch_oversold=data.get("stoch_oversold", 0.0),
            stoch_neutral=data.get("stoch_neutral", 1.0),
            rsi_div_bullish=data.get("rsi_div_bullish", 0.0),
            rsi_div_bearish=data.get("rsi_div_bearish", 0.0),
            rsi_div_none=data.get("rsi_div_none", 1.0),
            rsi_div_strength=data.get("rsi_div_strength", 0.0),
            volume_ratio=data.get("volume_ratio", 1.0),
            is_high_volume=data.get("is_high_volume", 0.0),
            obv_trend=data.get("obv_trend", 0.0),
            ha_trend_bullish=data.get("ha_trend_bullish", 0.0),
            ha_trend_bearish=data.get("ha_trend_bearish", 0.0),
            ha_trend_neutral=data.get("ha_trend_neutral", 1.0),
            ha_consecutive=data.get("ha_consecutive", 0.0),
            ha_strength=data.get("ha_strength", 0.0),
            vwap_distance_pct=data.get("vwap_distance_pct", 0.0),
            vwap_position_above=data.get("vwap_position_above", 0.0),
            vwap_position_below=data.get("vwap_position_below", 0.0),
            vwap_position_at=data.get("vwap_position_at", 1.0),
        )


@dataclass
class SimpleLogisticRegression:
    """
    A simple logistic regression model that doesn't require sklearn.
    Uses online learning to update weights incrementally.

    UPDATED: Now supports 83 features including advanced chart analysis + new indicators + variant.
    """
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    learning_rate: float = 0.1
    n_features: int = 83  # Number of features in TradeFeatures.to_vector()

    def __post_init__(self):
        if not self.weights:
            # Initialize weights with small random values
            import random
            random.seed(42)
            self.weights = [random.uniform(-0.1, 0.1) for _ in range(self.n_features)]

    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation function."""
        # Clip to avoid overflow
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))

    def predict_proba(self, features: list[float]) -> float:
        """Predict probability of win (class 1)."""
        if len(features) != len(self.weights):
            logger.warning(f"Feature mismatch: {len(features)} vs {len(self.weights)}")
            return 0.5

        # Linear combination
        z = self.bias + sum(w * f for w, f in zip(self.weights, features))
        return self._sigmoid(z)

    def update(self, features: list[float], outcome: int):
        """
        Update weights based on a single training example.

        Args:
            features: Feature vector
            outcome: 1 for win, 0 for loss
        """
        if len(features) != len(self.weights):
            return

        # Prediction
        pred = self.predict_proba(features)

        # Error
        error = outcome - pred

        # Update weights (gradient descent)
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error * features[i]
        self.bias += self.learning_rate * error

    def to_dict(self) -> dict:
        """Serialize model."""
        return {
            "weights": self.weights,
            "bias": self.bias,
            "learning_rate": self.learning_rate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimpleLogisticRegression":
        """Deserialize model."""
        return cls(
            weights=data.get("weights", []),
            bias=data.get("bias", 0.0),
            learning_rate=data.get("learning_rate", 0.1),
        )


@dataclass
class SimpleNeuralNetwork:
    """
    Simple 2-layer neural network for online learning.
    More powerful than logistic regression, adapts to new patterns.

    Architecture: Input(83) -> Hidden(85) -> Output(1)
    Uses ReLU activation and online gradient descent.

    UPDATED: Now uses all 83 features including advanced chart analysis + new indicators + variant.
    """
    input_size: int = 83  # Full feature vector (matching TradeFeatures.to_vector())
    hidden_size: int = 85  # Larger hidden layer for more features
    learning_rate: float = 0.03  # Slightly lower for stability

    # Weights
    weights_ih: list[list[float]] = field(default_factory=list)  # Input -> Hidden
    weights_ho: list[float] = field(default_factory=list)  # Hidden -> Output
    bias_h: list[float] = field(default_factory=list)  # Hidden bias
    bias_o: float = 0.0  # Output bias

    def __post_init__(self):
        """Initialize weights if not provided."""
        import random
        random.seed(42)

        if not self.weights_ih:
            # Xavier initialization
            scale = (2.0 / (self.input_size + self.hidden_size)) ** 0.5
            self.weights_ih = [
                [random.gauss(0, scale) for _ in range(self.input_size)]
                for _ in range(self.hidden_size)
            ]

        if not self.weights_ho:
            scale = (2.0 / (self.hidden_size + 1)) ** 0.5
            self.weights_ho = [random.gauss(0, scale) for _ in range(self.hidden_size)]

        if not self.bias_h:
            self.bias_h = [0.0] * self.hidden_size

    def _relu(self, x: float) -> float:
        """ReLU activation."""
        return max(0, x)

    def _relu_derivative(self, x: float) -> float:
        """ReLU derivative."""
        return 1.0 if x > 0 else 0.0

    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation for output."""
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, features: list[float]) -> tuple[float, list[float]]:
        """Forward pass, returns (output, hidden_activations)."""
        # Pad or truncate features to match input size
        if len(features) < self.input_size:
            features = features + [0.0] * (self.input_size - len(features))
        elif len(features) > self.input_size:
            features = features[:self.input_size]

        # Hidden layer
        hidden = []
        for i in range(self.hidden_size):
            z = self.bias_h[i] + sum(
                w * f for w, f in zip(self.weights_ih[i], features)
            )
            hidden.append(self._relu(z))

        # Output layer
        z_out = self.bias_o + sum(w * h for w, h in zip(self.weights_ho, hidden))
        output = self._sigmoid(z_out)

        return output, hidden

    def predict_proba(self, features: list[float]) -> float:
        """Predict probability of win."""
        output, _ = self.forward(features)
        return output

    def update(self, features: list[float], outcome: int):
        """Update weights using backpropagation."""
        # Pad or truncate features
        if len(features) < self.input_size:
            features = features + [0.0] * (self.input_size - len(features))
        elif len(features) > self.input_size:
            features = features[:self.input_size]

        # Forward pass
        output, hidden = self.forward(features)

        # Calculate output error
        output_error = outcome - output
        output_delta = output_error * output * (1 - output)  # Sigmoid derivative

        # Update output weights
        for i in range(self.hidden_size):
            self.weights_ho[i] += self.learning_rate * output_delta * hidden[i]
        self.bias_o += self.learning_rate * output_delta

        # Calculate hidden errors and update hidden weights
        for i in range(self.hidden_size):
            # Hidden layer pre-activation for ReLU derivative
            z_h = self.bias_h[i] + sum(
                w * f for w, f in zip(self.weights_ih[i], features)
            )
            hidden_delta = output_delta * self.weights_ho[i] * self._relu_derivative(z_h)

            for j in range(self.input_size):
                self.weights_ih[i][j] += self.learning_rate * hidden_delta * features[j]
            self.bias_h[i] += self.learning_rate * hidden_delta

    def to_dict(self) -> dict:
        """Serialize model."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "weights_ih": self.weights_ih,
            "weights_ho": self.weights_ho,
            "bias_h": self.bias_h,
            "bias_o": self.bias_o,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimpleNeuralNetwork":
        """Deserialize model, migrating old feature counts if needed."""
        CURRENT_INPUT_SIZE = 83  # Must match TradeFeatures.to_vector()
        saved_input_size = data.get("input_size", 82)

        nn = cls(
            input_size=CURRENT_INPUT_SIZE,
            hidden_size=data.get("hidden_size", 85),
            learning_rate=data.get("learning_rate", 0.03),
            weights_ih=data.get("weights_ih", []),
            weights_ho=data.get("weights_ho", []),
            bias_h=data.get("bias_h", []),
            bias_o=data.get("bias_o", 0.0),
        )

        # Migrate weight rows if loaded from older model with fewer features
        # Only migrate if weights were actually in the saved data (not freshly initialized)
        has_saved_weights = bool(data.get("weights_ih"))
        if saved_input_size < CURRENT_INPUT_SIZE and has_saved_weights and nn.weights_ih:
            import random
            random.seed(42)
            delta = CURRENT_INPUT_SIZE - saved_input_size
            scale = (2.0 / (CURRENT_INPUT_SIZE + nn.hidden_size)) ** 0.5
            if saved_input_size == 82 and CURRENT_INPUT_SIZE == 83:
                # is_five_min inserted at position 11 — splice weight into each row
                for row in nn.weights_ih:
                    row.insert(11, random.gauss(0, scale))
            else:
                # Generic: append new weights at end
                for row in nn.weights_ih:
                    row.extend([random.gauss(0, scale) for _ in range(delta)])
            logger.info(
                "NN model migrated: %d -> %d input features (%d new weights per neuron)",
                saved_input_size, CURRENT_INPUT_SIZE, delta,
            )

        return nn


class TraderModelLoader:
    """
    Loads the pre-trained Random Forest model from successful trader data.
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(project_root, "trader_model.json")

        self.model_path = model_path
        self.trees = []
        self.feature_names = []
        self.is_loaded = False
        self._load()

    def _load(self):
        """Load model from disk."""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, "r") as f:
                    data = json.load(f)

                model_data = data.get("model", {})
                self.trees = model_data.get("trees", [])
                self.feature_names = model_data.get("feature_names", [])
                self.is_loaded = len(self.trees) > 0

                if self.is_loaded:
                    metrics = data.get("metrics", {}).get("test", {})
                    logger.info(
                        f"Loaded trader model: {len(self.trees)} trees | "
                        f"Accuracy: {metrics.get('accuracy', 0)*100:.1f}%"
                    )
        except Exception as e:
            logger.warning(f"Could not load trader model: {e}")
            self.is_loaded = False

    def _predict_tree(self, tree: dict, features: list[float]) -> float:
        """Predict using a single tree."""
        if tree.get("leaf"):
            return tree.get("prediction", 0.5)

        feature_idx = tree.get("feature", 0)
        threshold = tree.get("threshold", 0.5)

        if feature_idx < len(features) and features[feature_idx] <= threshold:
            return self._predict_tree(tree.get("left", {"leaf": True, "prediction": 0.5}), features)
        else:
            return self._predict_tree(tree.get("right", {"leaf": True, "prediction": 0.5}), features)

    def predict_proba(self, features: list[float]) -> float:
        """Predict probability using Random Forest."""
        if not self.is_loaded or not self.trees:
            return 0.5

        predictions = [self._predict_tree(tree, features) for tree in self.trees]
        return sum(predictions) / len(predictions)

    def extract_simple_features(self, signal, market) -> list[float]:
        """
        Extract simplified features matching the trader model format.

        The trader model uses 11 features:
        [price, size, usdc_size, hour, day, time_remaining, is_btc, is_eth, is_sol, is_xrp, is_up]
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

        # Get price (use recommended_price from signal)
        price = getattr(signal, 'recommended_price', 0.5)

        # Size (normalized)
        size = getattr(signal, 'size_shares', 10) / 100
        usdc_size = getattr(signal, 'size_usd', 10) / 200

        # Time features
        hour = now.hour / 24
        day = now.weekday() / 7

        # Time remaining
        time_remaining = getattr(signal, 'time_remaining', 450) / 900

        # Asset one-hot
        asset = market.asset if hasattr(market, 'asset') else "BTC"
        is_btc = 1.0 if asset == "BTC" else 0.0
        is_eth = 1.0 if asset == "ETH" else 0.0
        is_sol = 1.0 if asset == "SOL" else 0.0
        is_xrp = 1.0 if asset == "XRP" else 0.0

        # Side
        side = getattr(signal, 'side', None)
        is_up = 1.0 if side and side.value == "UP" else 0.0

        return [price, size, usdc_size, hour, day, time_remaining,
                is_btc, is_eth, is_sol, is_xrp, is_up]


@dataclass
class HybridPredictor:
    """
    Hybrid ML predictor combining:
    1. Random Forest (pre-trained on successful trader)
    2. Neural Network (online learning from your trades)

    IMPORTANT: The Random Forest uses features that predict "did the trader
    make money" but NOT "will the price go up or down". Features like hour,
    day, bet size don't predict price direction.

    We use ADAPTIVE WEIGHTING that reduces RF influence over time as the
    Neural Network learns from actual trade outcomes.
    """

    # Base model weight - NOW ADAPTIVE (see _get_adaptive_weight)
    # Starting at 0.5 (50/50) since RF features are questionable
    base_weight: float = 0.5

    # Models
    trader_model: TraderModelLoader = field(default_factory=TraderModelLoader)
    adaptive_model: SimpleNeuralNetwork = field(default_factory=SimpleNeuralNetwork)

    # Tracking
    predictions_made: int = 0
    correct_predictions: int = 0
    training_samples: int = 0

    def _get_adaptive_weight(self) -> float:
        """
        Get adaptive weight for base (Random Forest) model.

        The RF model uses features that don't predict price direction well
        (hour, day, bet size). As we collect more training data, we should
        rely more on the Neural Network which learns from actual outcomes.

        Returns:
            Weight for base model (0.0 to 0.5)
        """
        samples = self.training_samples

        if samples < 20:
            # Very early: 50% RF, 50% NN (equal weight, both uncertain)
            return 0.5
        elif samples < 50:
            # Early: 40% RF, 60% NN (NN starting to learn)
            return 0.4
        elif samples < 100:
            # Medium: 30% RF, 70% NN (NN has meaningful data)
            return 0.3
        else:
            # Mature: 20% RF, 80% NN (trust learned patterns)
            return 0.2

    def predict_proba(self, signal, market, full_features: list[float]) -> float:
        """
        Predict win probability using hybrid approach.

        Args:
            signal: Trading signal
            market: Market state
            full_features: Full feature vector

        Returns:
            Combined probability estimate
        """
        # Get adaptive weight based on training samples
        rf_weight = self._get_adaptive_weight()

        # Get base prediction from trader model (uses 11 simple features)
        if self.trader_model.is_loaded:
            simple_features = self.trader_model.extract_simple_features(signal, market)
            base_prob = self.trader_model.predict_proba(simple_features)
        else:
            base_prob = 0.5

        # Get adaptive prediction using full feature vector
        adaptive_prob = self.adaptive_model.predict_proba(full_features)

        # Combine predictions with adaptive weighting
        if self.trader_model.is_loaded:
            combined = rf_weight * base_prob + (1 - rf_weight) * adaptive_prob
        else:
            # If no trader model, use only adaptive
            combined = adaptive_prob

        return combined

    def update(self, features: list[float], outcome: int, signal=None, market=None):
        """Update adaptive model with new outcome."""
        # Update adaptive model with FULL 33-feature vector
        # This ensures training and prediction use the same feature space
        self.adaptive_model.update(features, outcome)
        self.training_samples += 1

    def to_dict(self) -> dict:
        """Serialize hybrid predictor."""
        return {
            "base_weight": self.base_weight,
            "adaptive_model": self.adaptive_model.to_dict(),
            "predictions_made": self.predictions_made,
            "correct_predictions": self.correct_predictions,
            "training_samples": self.training_samples,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HybridPredictor":
        """Deserialize hybrid predictor."""
        predictor = cls(
            base_weight=data.get("base_weight", 0.7),
            predictions_made=data.get("predictions_made", 0),
            correct_predictions=data.get("correct_predictions", 0),
            training_samples=data.get("training_samples", 0),
        )

        if "adaptive_model" in data:
            predictor.adaptive_model = SimpleNeuralNetwork.from_dict(data["adaptive_model"])

        return predictor


@dataclass
class MLSignalPredictor:
    """
    Machine Learning predictor for signal quality.

    Features:
    - Learns from historical trade outcomes
    - Uses online learning (updates after each trade)
    - Persists model to disk
    - Provides win probability predictions
    - HYBRID MODE: Combines pre-trained trader model with adaptive neural network
    """

    model: SimpleLogisticRegression = field(default_factory=SimpleLogisticRegression)
    min_confidence: float = 0.55  # Minimum predicted win probability to trade
    min_training_samples: int = 10  # Minimum trades before using ML filter
    training_samples: int = 0
    model_path: str = ""  # Set in __post_init__

    # Performance tracking
    predictions_made: int = 0
    correct_predictions: int = 0

    # Hybrid predictor (combines trader model + adaptive neural network)
    hybrid_predictor: Optional[HybridPredictor] = None
    use_hybrid: bool = True  # Enable hybrid mode by default

    def __post_init__(self):
        """Initialize model path and load model if exists."""
        # Use absolute path based on project root (two levels up from this file)
        if not self.model_path:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.model_path = os.path.join(project_root, "ml_model.json")
        self._load_model()

        # Create the model file if it doesn't exist
        if not os.path.exists(self.model_path):
            self._save_model()
            logger.info(f"🤖 Created new ML model at: {self.model_path}")

        # Initialize hybrid predictor
        if self.use_hybrid and self.hybrid_predictor is None:
            self.hybrid_predictor = HybridPredictor()
            if self.hybrid_predictor.trader_model.is_loaded:
                logger.info("🤖 Hybrid ML enabled: Trader model + Adaptive NN")
            else:
                logger.info("🤖 Hybrid ML enabled: Adaptive NN only (no trader model found)")

    def extract_features(
        self,
        signal,  # Signal object
        volatility: float,
        price_momentum: float = 0.0,
        arb_type: str = "none",
        spread: float = 0.0,
        bid_depth: float = 0.0,
        ask_depth: float = 0.0,
        price_trend: float = 0.0,
        distance_from_target: float = 0.0,
        binance_lead_pct: float = 0.0,
        binance_confirmation: str = "NONE",
        trend_1h: float = 0.0,
        trend_4h: float = 0.0,
        trend_1d: float = 0.0,
        current_price: float = 0.0,
        target_price: float = 0.0,
        price_high: float = 0.0,
        price_low: float = 0.0,
        price_velocity: float = 0.0,
        # Chart analysis features
        chart_rsi: float = 50.0,
        chart_trend_strength: float = 0.0,
        chart_is_uptrend: float = 0.0,
        chart_is_downtrend: float = 0.0,
        chart_is_ranging: float = 0.0,
        chart_bullish_reversal: float = 0.0,
        chart_bearish_reversal: float = 0.0,
        chart_momentum: float = 0.0,
        chart_bias_bullish: float = 0.0,
        chart_bias_bearish: float = 0.0,
        chart_confidence: float = 0.5,
        chart_bullish_pattern: float = 0.0,
        chart_bearish_pattern: float = 0.0,
        # Advanced chart features
        chart_uncertainty_score: float = 0.0,
        chart_position_multiplier: float = 1.0,
        chart_timeframe_aligned: float = 1.0,
        chart_alignment_score: float = 1.0,
        chart_trend_breaking: float = 0.0,
        chart_resume_ready: float = 1.0,
        chart_resume_confidence: float = 1.0,
        # New indicator features (29 additional features)
        macd_histogram: float = 0.0,
        macd_crossover: str = "none",  # "bullish", "bearish", "none"
        bb_bandwidth: float = 0.0,
        bb_position: str = "middle",  # "above_upper", "below_lower", "middle"
        stoch_k: float = 50.0,
        stoch_d: float = 50.0,
        stoch_signal: str = "neutral",  # "overbought", "oversold", "neutral"
        rsi_divergence: str = "none",  # "bullish", "bearish", "none"
        rsi_divergence_strength: float = 0.0,
        volume_ratio: float = 1.0,
        is_high_volume: bool = False,
        obv_trend: float = 0.0,
        ha_trend: str = "neutral",  # "bullish", "bearish", "neutral"
        ha_consecutive: int = 0,
        ha_strength: float = 0.0,
        vwap_distance_pct: float = 0.0,
        vwap_position: str = "at",  # "above", "below", "at"
    ) -> TradeFeatures:
        """
        Extract features from a trading signal.

        Args:
            signal: Trading signal
            volatility: Asset volatility estimate
            price_momentum: Recent price direction (-1 to 1)
            arb_type: Type of arbitrage signal (none, binary_arb, asymmetric, dump, hedge)
            spread: Market bid-ask spread
            bid_depth: Order book bid depth
            ask_depth: Order book ask depth
            price_trend: Short-term price trend from historical data
            distance_from_target: How far current price is from target
            binance_lead_pct: Binance price lead over Chainlink (as decimal)
            binance_confirmation: Confirmation type (STRONG, MEDIUM, WEAK, NONE)
            trend_1h: 1-hour price trend (-1 to +1)
            trend_4h: 4-hour price trend (-1 to +1)
            trend_1d: 1-day price trend (-1 to +1)
            current_price: Current asset price (e.g., BTC price in USD)
            target_price: Target price for the market
            price_high: Recent high price (for range calculation)
            price_low: Recent low price (for range calculation)
            price_velocity: Rate of price change (absolute)
            chart_rsi: RSI from Binance chart (0-100)
            chart_trend_strength: Trend strength from chart (0-1)
            chart_is_uptrend: 1 if chart shows uptrend
            chart_is_downtrend: 1 if chart shows downtrend
            chart_is_ranging: 1 if chart shows ranging market
            chart_bullish_reversal: 1 if bullish reversal detected
            chart_bearish_reversal: 1 if bearish reversal detected
            chart_momentum: Chart momentum (-1 to +1)
            chart_bias_bullish: 1 if chart bias is bullish
            chart_bias_bearish: 1 if chart bias is bearish
            chart_confidence: Chart analysis confidence (0-1)
            chart_bullish_pattern: 1 if bullish pattern detected
            chart_bearish_pattern: 1 if bearish pattern detected
            chart_uncertainty_score: Market uncertainty (0-1)
            chart_position_multiplier: Graduated position size (0-1)
            chart_timeframe_aligned: 1 if 15m/1h/4h aligned
            chart_alignment_score: Multi-TF alignment (0-1)
            chart_trend_breaking: 1 if trend is breaking down
            chart_resume_ready: 1 if safe to trade
            chart_resume_confidence: Confidence in resuming (0-1)
            macd_histogram: MACD histogram value (normalized)
            macd_crossover: MACD crossover type ("bullish", "bearish", "none")
            bb_bandwidth: Bollinger Band bandwidth
            bb_position: Price position relative to BB ("above_upper", "below_lower", "middle")
            stoch_k: Stochastic %K (0-100)
            stoch_d: Stochastic %D (0-100)
            stoch_signal: Stochastic signal ("overbought", "oversold", "neutral")
            rsi_divergence: RSI divergence type ("bullish", "bearish", "none")
            rsi_divergence_strength: RSI divergence strength (0-1)
            volume_ratio: Current volume / average volume
            is_high_volume: True if volume > 1.5x average
            obv_trend: On-balance volume trend (-1 to +1)
            ha_trend: Heiken Ashi trend ("bullish", "bearish", "neutral")
            ha_consecutive: Consecutive same-color HA candles
            ha_strength: Heiken Ashi trend strength (0-1)
            vwap_distance_pct: Distance from VWAP as percentage
            vwap_position: Position relative to VWAP ("above", "below", "at")
        """
        now = datetime.now(timezone.utc)

        # Get arb_type from signal if stored there
        if hasattr(signal, '_arb_type') and signal._arb_type:
            arb_type = signal._arb_type

        # Get prices from signal if not provided
        if current_price == 0.0 and hasattr(signal, 'market'):
            target_price = getattr(signal.market, 'target_price', 0.0)

        # Calculate normalized price (0-1 scale based on asset)
        # BTC: ~100k range, ETH: ~10k range, SOL: ~500 range, XRP: ~5 range
        asset = signal.market.asset
        price_scales = {"BTC": 150000, "ETH": 10000, "SOL": 500, "XRP": 10}
        scale = price_scales.get(asset, 100000)
        price_normalized = min(current_price / scale, 1.0) if current_price > 0 else 0.0

        # Calculate price above target (binary)
        price_above_target = 1.0 if current_price >= target_price and target_price > 0 else 0.0

        # Calculate price range position (0 = at low, 1 = at high)
        if price_high > price_low and price_low > 0:
            price_range_position = (current_price - price_low) / (price_high - price_low)
            price_range_position = max(0.0, min(1.0, price_range_position))
        else:
            price_range_position = 0.5  # Default to middle if no range data

        # Detect 5-min variant from signal's market
        variant = getattr(signal.market, 'variant', 'fifteen') if hasattr(signal, 'market') else 'fifteen'
        is_five_min = 1.0 if variant == "five" else 0.0

        return TradeFeatures(
            edge=signal.edge,
            time_remaining=signal.market.time_remaining,
            volatility=volatility,
            price_momentum=price_momentum,
            hour_of_day=now.hour,
            day_of_week=now.weekday(),
            asset=asset,
            side=signal.side.value,
            is_five_min=is_five_min,
            arb_type=arb_type,
            spread=spread,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            price_trend=price_trend,
            distance_from_target=distance_from_target,
            binance_lead_pct=binance_lead_pct,
            binance_confirmation=binance_confirmation,
            trend_1h=trend_1h,
            trend_4h=trend_4h,
            trend_1d=trend_1d,
            price_normalized=price_normalized,
            price_above_target=price_above_target,
            price_range_position=price_range_position,
            price_velocity=price_velocity,
            # Chart analysis features
            chart_rsi=chart_rsi,
            chart_trend_strength=chart_trend_strength,
            chart_is_uptrend=chart_is_uptrend,
            chart_is_downtrend=chart_is_downtrend,
            chart_is_ranging=chart_is_ranging,
            chart_bullish_reversal=chart_bullish_reversal,
            chart_bearish_reversal=chart_bearish_reversal,
            chart_momentum=chart_momentum,
            chart_bias_bullish=chart_bias_bullish,
            chart_bias_bearish=chart_bias_bearish,
            chart_confidence=chart_confidence,
            chart_bullish_pattern=chart_bullish_pattern,
            chart_bearish_pattern=chart_bearish_pattern,
            # Advanced chart features
            chart_uncertainty_score=chart_uncertainty_score,
            chart_position_multiplier=chart_position_multiplier,
            chart_timeframe_aligned=chart_timeframe_aligned,
            chart_alignment_score=chart_alignment_score,
            chart_trend_breaking=chart_trend_breaking,
            chart_resume_ready=chart_resume_ready,
            chart_resume_confidence=chart_resume_confidence,
            # New indicator features - MACD
            macd_histogram=macd_histogram,
            macd_crossover_bullish=1.0 if macd_crossover == "bullish" else 0.0,
            macd_crossover_bearish=1.0 if macd_crossover == "bearish" else 0.0,
            macd_crossover_none=1.0 if macd_crossover not in ["bullish", "bearish"] else 0.0,
            # Bollinger Bands
            bb_bandwidth=bb_bandwidth,
            bb_position_above=1.0 if bb_position == "above_upper" else 0.0,
            bb_position_below=1.0 if bb_position == "below_lower" else 0.0,
            bb_position_middle=1.0 if bb_position not in ["above_upper", "below_lower"] else 0.0,
            # Stochastic
            stoch_k=stoch_k,
            stoch_d=stoch_d,
            stoch_overbought=1.0 if stoch_signal == "overbought" else 0.0,
            stoch_oversold=1.0 if stoch_signal == "oversold" else 0.0,
            stoch_neutral=1.0 if stoch_signal not in ["overbought", "oversold"] else 0.0,
            # RSI Divergence
            rsi_div_bullish=1.0 if rsi_divergence == "bullish" else 0.0,
            rsi_div_bearish=1.0 if rsi_divergence == "bearish" else 0.0,
            rsi_div_none=1.0 if rsi_divergence not in ["bullish", "bearish"] else 0.0,
            rsi_div_strength=rsi_divergence_strength,
            # Volume
            volume_ratio=volume_ratio,
            is_high_volume=1.0 if is_high_volume else 0.0,
            obv_trend=obv_trend,
            # Heiken Ashi
            ha_trend_bullish=1.0 if ha_trend == "bullish" else 0.0,
            ha_trend_bearish=1.0 if ha_trend == "bearish" else 0.0,
            ha_trend_neutral=1.0 if ha_trend not in ["bullish", "bearish"] else 0.0,
            ha_consecutive=float(ha_consecutive),
            ha_strength=ha_strength,
            # VWAP
            vwap_distance_pct=vwap_distance_pct,
            vwap_position_above=1.0 if vwap_position == "above" else 0.0,
            vwap_position_below=1.0 if vwap_position == "below" else 0.0,
            vwap_position_at=1.0 if vwap_position not in ["above", "below"] else 0.0,
        )

    def _get_gradual_threshold(self) -> float:
        """
        Get the ML confidence threshold based on training samples.

        Uses a gradual ramp-up to collect more diverse training data
        before applying strict filtering:
        - 0-30 samples: Learning mode (allow all)
        - 30-50 samples: 50% threshold (coin flip minimum)
        - 50+ samples: Use configured min_confidence (from ML_MIN_CONFIDENCE)

        Returns:
            Current confidence threshold (0.0 to 1.0)
        """
        samples = self.training_samples

        if samples < 30:
            return 0.0  # Learning mode - allow all
        elif samples < 50:
            return 0.50  # Minimum: don't trade if model predicts loss
        else:
            # Use configured threshold (default 0.50, can be set via ML_MIN_CONFIDENCE)
            return self.min_confidence

    def should_trade(
        self,
        signal,
        volatility: float,
        price_momentum: float = 0.0,
        arb_type: str = "none",
        spread: float = 0.0,
        bid_depth: float = 0.0,
        ask_depth: float = 0.0,
        price_trend: float = 0.0,
        distance_from_target: float = 0.0,
        binance_lead_pct: float = 0.0,
        binance_confirmation: str = "NONE",
        trend_1h: float = 0.0,
        trend_4h: float = 0.0,
        trend_1d: float = 0.0,
        current_price: float = 0.0,
        target_price: float = 0.0,
        # Chart analysis features
        chart_rsi: float = 50.0,
        chart_trend_strength: float = 0.0,
        chart_is_uptrend: float = 0.0,
        chart_is_downtrend: float = 0.0,
        chart_is_ranging: float = 0.0,
        chart_bullish_reversal: float = 0.0,
        chart_bearish_reversal: float = 0.0,
        chart_momentum: float = 0.0,
        chart_bias_bullish: float = 0.0,
        chart_bias_bearish: float = 0.0,
        chart_confidence: float = 0.5,
        chart_bullish_pattern: float = 0.0,
        chart_bearish_pattern: float = 0.0,
        # Advanced chart features
        chart_uncertainty_score: float = 0.0,
        chart_position_multiplier: float = 1.0,
        chart_timeframe_aligned: float = 1.0,
        chart_alignment_score: float = 1.0,
        chart_trend_breaking: float = 0.0,
        chart_resume_ready: float = 1.0,
        chart_resume_confidence: float = 1.0,
        # New indicator features
        macd_histogram: float = 0.0,
        macd_crossover: str = "none",
        bb_bandwidth: float = 0.0,
        bb_position: str = "middle",
        stoch_k: float = 50.0,
        stoch_d: float = 50.0,
        stoch_signal: str = "neutral",
        rsi_divergence: str = "none",
        rsi_divergence_strength: float = 0.0,
        volume_ratio: float = 1.0,
        is_high_volume: bool = False,
        obv_trend: float = 0.0,
        ha_trend: str = "neutral",
        ha_consecutive: int = 0,
        ha_strength: float = 0.0,
        vwap_distance_pct: float = 0.0,
        vwap_position: str = "at",
    ) -> tuple[bool, float, str]:
        """
        Decide if we should take this trade based on ML prediction.

        Uses gradual threshold ramp-up:
        - 0-30 samples: Learning mode (allow all)
        - 30-50 samples: 50% threshold (minimum: don't predict loss)
        - 50-75 samples: 52% threshold (slight edge)
        - 75+ samples: 55% threshold (meaningful edge)

        SANITY CHECKS (applied before ML):
        - Don't bet DOWN if price is already significantly above target
        - Don't bet UP if price is already significantly below target

        Args:
            signal: Trading signal
            volatility: Asset volatility
            price_momentum: Recent price direction
            arb_type: Type of arbitrage signal
            spread: Market bid-ask spread
            bid_depth: Order book bid depth
            ask_depth: Order book ask depth
            price_trend: Short-term price trend
            distance_from_target: Distance from target price
            binance_lead_pct: Binance price lead over Chainlink
            binance_confirmation: Confirmation type (STRONG, MEDIUM, WEAK, NONE)
            trend_1h: 1-hour price trend (-1 to +1)
            trend_4h: 4-hour price trend (-1 to +1)
            trend_1d: 1-day price trend (-1 to +1)
            current_price: Current asset price (e.g., from Chainlink)
            target_price: Target price for the market

        Returns:
            Tuple of (should_trade, confidence, reason)
        """
        # Get market from signal
        market = getattr(signal, 'market', None)
        side = getattr(signal, 'side', None)
        side_value = side.value if side else "UNKNOWN"

        # Try to get prices from market if not provided
        if current_price == 0.0 and market:
            current_price = getattr(market, 'current_price', 0.0)
        if target_price == 0.0 and market:
            target_price = getattr(market, 'target_price', 0.0)

        # ============================================================
        # SANITY CHECK: Block trades where price already crossed target
        # ============================================================
        if current_price > 0 and target_price > 0:
            price_above_target_pct = (current_price - target_price) / target_price * 100

            # Block DOWN bets if price is already significantly above target
            # (price needs to DROP to win, but it's already above - bad bet)
            if side_value == "DOWN" and price_above_target_pct > 0.15:
                return (False, 0.0, f"SANITY BLOCK: Price ${current_price:,.0f} is {price_above_target_pct:.2f}% ABOVE target ${target_price:,.0f} - DOWN bet would likely lose")

            # Block UP bets if price is already significantly below target
            # (price needs to STAY ABOVE to win, but it's already below - bad bet)
            if side_value == "UP" and price_above_target_pct < -0.15:
                return (False, 0.0, f"SANITY BLOCK: Price ${current_price:,.0f} is {abs(price_above_target_pct):.2f}% BELOW target ${target_price:,.0f} - UP bet would likely lose")

        # ============================================================
        # CHART ANALYSIS SANITY CHECK: Block trades against strong chart signals
        # ============================================================
        # Determine if chart shows strong directional signal
        chart_says_down = (chart_bias_bearish > 0 and chart_confidence >= 0.6) or \
                          (chart_is_downtrend > 0 and chart_trend_strength >= 0.5)
        chart_says_up = (chart_bias_bullish > 0 and chart_confidence >= 0.6) or \
                        (chart_is_uptrend > 0 and chart_trend_strength >= 0.5)

        # Calculate signal strength
        chart_signal_strength = max(chart_confidence, chart_trend_strength)

        # Strong chart signal (>70% confidence) blocks trades going against it
        if chart_signal_strength >= 0.7:
            if chart_says_down and side_value == "UP":
                return (False, 0.0,
                    f"CHART BLOCK: UP bet against strong BEARISH chart "
                    f"(conf={chart_confidence:.0%}, trend_str={chart_trend_strength:.0%}, "
                    f"downtrend={chart_is_downtrend}, bearish={chart_bias_bearish})")

            if chart_says_up and side_value == "DOWN":
                return (False, 0.0,
                    f"CHART BLOCK: DOWN bet against strong BULLISH chart "
                    f"(conf={chart_confidence:.0%}, trend_str={chart_trend_strength:.0%}, "
                    f"uptrend={chart_is_uptrend}, bullish={chart_bias_bullish})")

        # Get current threshold based on training samples
        threshold = self._get_gradual_threshold()

        # Extract features and predict
        features = self.extract_features(
            signal, volatility, price_momentum, arb_type,
            spread, bid_depth, ask_depth, price_trend, distance_from_target,
            binance_lead_pct, binance_confirmation,
            trend_1h, trend_4h, trend_1d,
            current_price=current_price, target_price=target_price,
            # Chart analysis features
            chart_rsi=chart_rsi,
            chart_trend_strength=chart_trend_strength,
            chart_is_uptrend=chart_is_uptrend,
            chart_is_downtrend=chart_is_downtrend,
            chart_is_ranging=chart_is_ranging,
            chart_bullish_reversal=chart_bullish_reversal,
            chart_bearish_reversal=chart_bearish_reversal,
            chart_momentum=chart_momentum,
            chart_bias_bullish=chart_bias_bullish,
            chart_bias_bearish=chart_bias_bearish,
            chart_confidence=chart_confidence,
            chart_bullish_pattern=chart_bullish_pattern,
            chart_bearish_pattern=chart_bearish_pattern,
            # Advanced chart features
            chart_uncertainty_score=chart_uncertainty_score,
            chart_position_multiplier=chart_position_multiplier,
            chart_timeframe_aligned=chart_timeframe_aligned,
            chart_alignment_score=chart_alignment_score,
            chart_trend_breaking=chart_trend_breaking,
            chart_resume_ready=chart_resume_ready,
            chart_resume_confidence=chart_resume_confidence,
            # New indicator features
            macd_histogram=macd_histogram,
            macd_crossover=macd_crossover,
            bb_bandwidth=bb_bandwidth,
            bb_position=bb_position,
            stoch_k=stoch_k,
            stoch_d=stoch_d,
            stoch_signal=stoch_signal,
            rsi_divergence=rsi_divergence,
            rsi_divergence_strength=rsi_divergence_strength,
            volume_ratio=volume_ratio,
            is_high_volume=is_high_volume,
            obv_trend=obv_trend,
            ha_trend=ha_trend,
            ha_consecutive=ha_consecutive,
            ha_strength=ha_strength,
            vwap_distance_pct=vwap_distance_pct,
            vwap_position=vwap_position,
        )
        feature_vector = features.to_vector()

        # Use hybrid predictor if available, otherwise use logistic regression
        if self.use_hybrid and self.hybrid_predictor is not None:
            confidence = self.hybrid_predictor.predict_proba(signal, market, feature_vector)
            model_type = "Hybrid" if self.hybrid_predictor.trader_model.is_loaded else "NN"
        else:
            confidence = self.model.predict_proba(feature_vector)
            model_type = "LR"

        # ============================================================
        # CHART ANALYSIS CONFIDENCE ADJUSTMENT
        # Penalize confidence when chart disagrees with trade direction
        # ============================================================
        chart_penalty = 0.0
        chart_penalty_reason = ""

        # Moderate chart disagreement (not strong enough to block, but penalize)
        if chart_signal_strength >= 0.5 and chart_signal_strength < 0.7:
            if chart_says_down and side_value == "UP":
                chart_penalty = chart_signal_strength * 0.15  # Up to 10.5% penalty
                chart_penalty_reason = f" [CHART DISAGREE: bearish {chart_signal_strength:.0%} vs UP]"
            elif chart_says_up and side_value == "DOWN":
                chart_penalty = chart_signal_strength * 0.15
                chart_penalty_reason = f" [CHART DISAGREE: bullish {chart_signal_strength:.0%} vs DOWN]"

        # Apply chart penalty to confidence
        original_confidence = confidence
        confidence = max(0.0, confidence - chart_penalty)

        if chart_penalty > 0:
            logger.info(
                f"🤖 ML CHART PENALTY [{market.asset if market else 'UNK'}]: "
                f"Conf {original_confidence:.0%} → {confidence:.0%} (-{chart_penalty:.0%}){chart_penalty_reason}"
            )

        # Log feature importance info for debugging
        arb_info = f" [ARB: {arb_type}]" if arb_type != "none" else ""
        bn_info = f" [BN: {binance_confirmation}]" if binance_confirmation != "NONE" else ""

        if confidence < threshold:
            return (False, confidence, f"{model_type} rejected: {confidence:.0%} < {threshold:.0%}{arb_info}{bn_info}")
        else:
            return (True, confidence, f"{model_type} confidence: {confidence:.0%} (threshold: {threshold:.0%}){arb_info}{bn_info}")

    def record_outcome(
        self,
        signal,
        volatility: float,
        price_momentum: float,
        won: bool,
        arb_type: str = "none",
        spread: float = 0.0,
        bid_depth: float = 0.0,
        ask_depth: float = 0.0,
        price_trend: float = 0.0,
        distance_from_target: float = 0.0,
        binance_lead_pct: float = 0.0,
        binance_confirmation: str = "NONE",
        trend_1h: float = 0.0,
        trend_4h: float = 0.0,
        trend_1d: float = 0.0,
        current_price: float = 0.0,
        target_price: float = 0.0,
        price_high: float = 0.0,
        price_low: float = 0.0,
        price_velocity: float = 0.0,
        # Chart analysis features
        chart_rsi: float = 50.0,
        chart_trend_strength: float = 0.0,
        chart_is_uptrend: float = 0.0,
        chart_is_downtrend: float = 0.0,
        chart_is_ranging: float = 0.0,
        chart_bullish_reversal: float = 0.0,
        chart_bearish_reversal: float = 0.0,
        chart_momentum: float = 0.0,
        chart_bias_bullish: float = 0.0,
        chart_bias_bearish: float = 0.0,
        chart_confidence: float = 0.5,
        chart_bullish_pattern: float = 0.0,
        chart_bearish_pattern: float = 0.0,
        # Advanced chart features
        chart_uncertainty_score: float = 0.0,
        chart_position_multiplier: float = 1.0,
        chart_timeframe_aligned: float = 1.0,
        chart_alignment_score: float = 1.0,
        chart_trend_breaking: float = 0.0,
        chart_resume_ready: float = 1.0,
        chart_resume_confidence: float = 1.0,
        # New indicator features
        macd_histogram: float = 0.0,
        macd_crossover: str = "none",
        bb_bandwidth: float = 0.0,
        bb_position: str = "middle",
        stoch_k: float = 50.0,
        stoch_d: float = 50.0,
        stoch_signal: str = "neutral",
        rsi_divergence: str = "none",
        rsi_divergence_strength: float = 0.0,
        volume_ratio: float = 1.0,
        is_high_volume: bool = False,
        obv_trend: float = 0.0,
        ha_trend: str = "neutral",
        ha_consecutive: int = 0,
        ha_strength: float = 0.0,
        vwap_distance_pct: float = 0.0,
        vwap_position: str = "at",
    ):
        """
        Record a trade outcome and update the model.

        Args:
            signal: Original trading signal
            volatility: Asset volatility at trade time
            price_momentum: Price momentum at trade time
            won: Whether the trade was profitable
            arb_type: Type of arbitrage signal
            spread: Market bid-ask spread at trade time
            bid_depth: Order book bid depth at trade time
            ask_depth: Order book ask depth at trade time
            price_trend: Price trend at trade time
            distance_from_target: Distance from target at trade time
            binance_lead_pct: Binance price lead over Chainlink at trade time
            binance_confirmation: Confirmation type at trade time
            trend_1h: 1-hour price trend (-1 to +1)
            trend_4h: 4-hour price trend (-1 to +1)
            trend_1d: 1-day price trend (-1 to +1)
            current_price: Current asset price at trade time
            target_price: Target price for the market
            price_high: Recent high price
            price_low: Recent low price
            price_velocity: Rate of price change
            chart_uncertainty_score: Market uncertainty (0-1)
            chart_position_multiplier: Graduated position size (0-1)
            chart_timeframe_aligned: 1 if 15m/1h/4h aligned
            chart_alignment_score: Multi-TF alignment (0-1)
            chart_trend_breaking: 1 if trend is breaking down
            chart_resume_ready: 1 if safe to trade
            chart_resume_confidence: Confidence in resuming (0-1)
            + New indicator features: MACD, Bollinger, Stochastic, etc.
        """
        features = self.extract_features(
            signal, volatility, price_momentum, arb_type,
            spread, bid_depth, ask_depth, price_trend, distance_from_target,
            binance_lead_pct, binance_confirmation,
            trend_1h, trend_4h, trend_1d,
            current_price, target_price, price_high, price_low, price_velocity,
            # Chart analysis features
            chart_rsi=chart_rsi,
            chart_trend_strength=chart_trend_strength,
            chart_is_uptrend=chart_is_uptrend,
            chart_is_downtrend=chart_is_downtrend,
            chart_is_ranging=chart_is_ranging,
            chart_bullish_reversal=chart_bullish_reversal,
            chart_bearish_reversal=chart_bearish_reversal,
            chart_momentum=chart_momentum,
            chart_bias_bullish=chart_bias_bullish,
            chart_bias_bearish=chart_bias_bearish,
            chart_confidence=chart_confidence,
            chart_bullish_pattern=chart_bullish_pattern,
            chart_bearish_pattern=chart_bearish_pattern,
            # Advanced chart features
            chart_uncertainty_score=chart_uncertainty_score,
            chart_position_multiplier=chart_position_multiplier,
            chart_timeframe_aligned=chart_timeframe_aligned,
            chart_alignment_score=chart_alignment_score,
            chart_trend_breaking=chart_trend_breaking,
            chart_resume_ready=chart_resume_ready,
            chart_resume_confidence=chart_resume_confidence,
            # New indicator features
            macd_histogram=macd_histogram,
            macd_crossover=macd_crossover,
            bb_bandwidth=bb_bandwidth,
            bb_position=bb_position,
            stoch_k=stoch_k,
            stoch_d=stoch_d,
            stoch_signal=stoch_signal,
            rsi_divergence=rsi_divergence,
            rsi_divergence_strength=rsi_divergence_strength,
            volume_ratio=volume_ratio,
            is_high_volume=is_high_volume,
            obv_trend=obv_trend,
            ha_trend=ha_trend,
            ha_consecutive=ha_consecutive,
            ha_strength=ha_strength,
            vwap_distance_pct=vwap_distance_pct,
            vwap_position=vwap_position,
        )
        feature_vector = features.to_vector()

        # Get prediction before updating (for accuracy tracking)
        if self.training_samples >= self.min_training_samples:
            # Use hybrid predictor if available for accurate tracking
            if self.use_hybrid and self.hybrid_predictor is not None:
                market = getattr(signal, 'market', None)
                pred_prob = self.hybrid_predictor.predict_proba(signal, market, feature_vector)
            else:
                pred_prob = self.model.predict_proba(feature_vector)
            predicted_win = pred_prob >= 0.5
            if predicted_win == won:
                self.correct_predictions += 1
            self.predictions_made += 1

        # Update model with this example
        self.model.update(feature_vector, 1 if won else 0)
        self.training_samples += 1

        # Also update hybrid predictor's adaptive model
        if self.use_hybrid and self.hybrid_predictor is not None:
            market = getattr(signal, 'market', None)
            self.hybrid_predictor.update(feature_vector, 1 if won else 0, signal=signal, market=market)

        # Log outcome recording
        result_str = "WIN" if won else "LOSS"
        model_type = "Hybrid" if (self.use_hybrid and self.hybrid_predictor and self.hybrid_predictor.trader_model.is_loaded) else "ML"
        logger.info(f"🤖 {model_type} recorded: {result_str} | Sample #{self.training_samples}")

        # Log learning progress
        if self.training_samples % 5 == 0:
            accuracy = self.correct_predictions / max(1, self.predictions_made)
            logger.info(
                f"🤖 ML Model: {self.training_samples} samples | "
                f"Accuracy: {accuracy:.0%} ({self.correct_predictions}/{self.predictions_made})"
            )

        # Save model after EVERY trade (user might stop bot at any time)
        self._save_model()

    def get_model_stats(self) -> dict:
        """Get model statistics."""
        accuracy = self.correct_predictions / max(1, self.predictions_made)
        stats = {
            "training_samples": self.training_samples,
            "predictions_made": self.predictions_made,
            "correct_predictions": self.correct_predictions,
            "accuracy": accuracy,
            "min_confidence": self.min_confidence,
            "is_active": self.training_samples >= self.min_training_samples,
            "hybrid_enabled": self.use_hybrid,
        }

        # Add hybrid predictor stats
        if self.use_hybrid and self.hybrid_predictor is not None:
            stats["trader_model_loaded"] = self.hybrid_predictor.trader_model.is_loaded
            stats["hybrid_training_samples"] = self.hybrid_predictor.training_samples
            stats["model_type"] = "Hybrid RF+NN" if self.hybrid_predictor.trader_model.is_loaded else "Adaptive NN"
        else:
            stats["model_type"] = "Logistic Regression"

        return stats

    def is_prediction_accurate(self, min_accuracy: float = 0.45, min_samples: int = 10) -> tuple[bool, str]:
        """
        Check if model predictions are accurate enough to continue trading.

        Args:
            min_accuracy: Minimum required accuracy (default 45%)
            min_samples: Minimum predictions needed before checking (default 10)

        Returns:
            Tuple of (is_accurate, reason)
        """
        if self.predictions_made < min_samples:
            return (True, f"Not enough predictions yet ({self.predictions_made}/{min_samples})")

        accuracy = self.correct_predictions / self.predictions_made
        if accuracy < min_accuracy:
            return (
                False,
                f"Prediction accuracy {accuracy:.0%} below {min_accuracy:.0%} minimum "
                f"({self.correct_predictions}/{self.predictions_made} correct)"
            )

        return (True, f"Accuracy OK: {accuracy:.0%}")

    def save(self):
        """Public method to save model to disk."""
        self._save_model()

    def _save_model(self):
        """Save model to disk."""
        try:
            data = {
                "model": self.model.to_dict(),
                "training_samples": self.training_samples,
                "predictions_made": self.predictions_made,
                "correct_predictions": self.correct_predictions,
                "min_confidence": self.min_confidence,
            }

            # Save hybrid predictor state
            if self.use_hybrid and self.hybrid_predictor is not None:
                data["hybrid_predictor"] = self.hybrid_predictor.to_dict()

            with open(self.model_path, "w") as f:
                json.dump(data, f, indent=2)

            model_type = "Hybrid" if (self.use_hybrid and self.hybrid_predictor and self.hybrid_predictor.trader_model.is_loaded) else "ML"
            logger.info(f"🤖 {model_type} model saved: {self.training_samples} samples → {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save ML model: {e}")

    def _load_model(self):
        """Load model from disk if exists."""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, "r") as f:
                    data = json.load(f)

                model_data = data.get("model", {})
                old_weights = model_data.get("weights", [])

                # Check if we need to migrate feature counts
                # 11 -> 21: Added arb type and market features
                # 21 -> 26: Added Binance confirmation features
                # 26 -> 29: Added multi-timeframe trends (1h, 4h, 1d)
                # 29 -> 33: Added price observation features (price_normalized, price_above_target, price_range_position, price_velocity)
                expected_features = 33
                import random
                random.seed(42)

                if len(old_weights) == 11:
                    logger.info(
                        f"🤖 Migrating ML model from 11 to 33 features (adding arb + Binance + trends + price)..."
                    )
                    # Extend weights with small random values for new features
                    new_weights = old_weights + [random.uniform(-0.1, 0.1) for _ in range(22)]
                    model_data["weights"] = new_weights
                    # Reset training samples since feature set changed significantly
                    self.training_samples = max(0, data.get("training_samples", 0) // 2)
                    self.predictions_made = 0
                    self.correct_predictions = 0
                    logger.info(
                        f"🤖 Migration complete - model needs retraining with new features"
                    )
                elif len(old_weights) == 21:
                    logger.info(
                        f"🤖 Migrating ML model from 21 to 33 features (adding Binance + trends + price)..."
                    )
                    # Extend weights for Binance + trend + price features
                    new_weights = old_weights + [random.uniform(-0.1, 0.1) for _ in range(12)]
                    model_data["weights"] = new_weights
                    # Keep most training data, slight reset since new features added
                    self.training_samples = max(0, int(data.get("training_samples", 0) * 0.75))
                    self.predictions_made = int(data.get("predictions_made", 0) * 0.75)
                    self.correct_predictions = int(data.get("correct_predictions", 0) * 0.75)
                    logger.info(
                        f"🤖 Migration complete - model will retrain with Binance + trend + price features"
                    )
                elif len(old_weights) == 26:
                    logger.info(
                        f"🤖 Migrating ML model from 26 to 33 features (adding trends + price)..."
                    )
                    # Extend weights for trend + price features (7 new)
                    new_weights = old_weights + [random.uniform(-0.1, 0.1) for _ in range(7)]
                    model_data["weights"] = new_weights
                    # Keep most training data since this is a minor addition
                    self.training_samples = max(0, int(data.get("training_samples", 0) * 0.9))
                    self.predictions_made = int(data.get("predictions_made", 0) * 0.9)
                    self.correct_predictions = int(data.get("correct_predictions", 0) * 0.9)
                    logger.info(
                        f"🤖 Migration complete - model will retrain with trend + price features"
                    )
                elif len(old_weights) == 29:
                    logger.info(
                        f"🤖 Migrating ML model from 29 to 46 features (adding price + chart analysis)..."
                    )
                    # Extend weights for price features + chart analysis (17 new features)
                    new_weights = old_weights + [random.uniform(-0.1, 0.1) for _ in range(17)]
                    model_data["weights"] = new_weights
                    # Keep most training data since this is a significant addition
                    self.training_samples = max(0, int(data.get("training_samples", 0) * 0.8))
                    self.predictions_made = int(data.get("predictions_made", 0) * 0.8)
                    self.correct_predictions = int(data.get("correct_predictions", 0) * 0.8)
                    logger.info(
                        f"🤖 Migration complete - model will retrain with price + chart features"
                    )
                elif len(old_weights) == 33:
                    logger.info(
                        f"🤖 Migrating ML model from 33 to 53 features (adding chart analysis + advanced)..."
                    )
                    # Extend weights for chart analysis features (13 + 7 new = 20 total)
                    new_weights = old_weights + [random.uniform(-0.1, 0.1) for _ in range(20)]
                    model_data["weights"] = new_weights
                    # Keep most training data since this is a significant addition
                    self.training_samples = max(0, int(data.get("training_samples", 0) * 0.85))
                    self.predictions_made = int(data.get("predictions_made", 0) * 0.85)
                    self.correct_predictions = int(data.get("correct_predictions", 0) * 0.85)
                    logger.info(
                        f"🤖 Migration complete - model will retrain with all chart features"
                    )
                elif len(old_weights) == 46:
                    logger.info(
                        f"🤖 Migrating ML model from 46 to 53 features (adding advanced chart features)..."
                    )
                    # Extend weights for advanced chart features (7 new)
                    new_weights = old_weights + [random.uniform(-0.1, 0.1) for _ in range(7)]
                    model_data["weights"] = new_weights
                    # Keep most training data since this is a smaller addition
                    self.training_samples = max(0, int(data.get("training_samples", 0) * 0.9))
                    self.predictions_made = int(data.get("predictions_made", 0) * 0.9)
                    self.correct_predictions = int(data.get("correct_predictions", 0) * 0.9)
                    logger.info(
                        f"🤖 Migration complete - model will retrain with advanced chart features"
                    )
                elif len(old_weights) == 53:
                    logger.info(
                        f"🤖 Migrating ML model from 53 to 83 features (adding new indicators + variant)..."
                    )
                    # Extend weights for new indicator features (29 new) + variant (1 new) = 30
                    new_weights = old_weights + [random.uniform(-0.1, 0.1) for _ in range(30)]
                    model_data["weights"] = new_weights
                    self.training_samples = max(0, int(data.get("training_samples", 0) * 0.85))
                    self.predictions_made = int(data.get("predictions_made", 0) * 0.85)
                    self.correct_predictions = int(data.get("correct_predictions", 0) * 0.85)
                    logger.info(
                        f"🤖 Migration complete - model will retrain with new indicator + variant features"
                    )
                elif len(old_weights) == 82:
                    logger.info(
                        f"🤖 Migrating ML model from 82 to 83 features (adding is_five_min variant)..."
                    )
                    # Insert weight for is_five_min at position 11 (after side, before arb_type)
                    # But simpler: just append since the model will retrain quickly
                    new_weights = old_weights[:11] + [random.uniform(-0.1, 0.1)] + old_weights[11:]
                    model_data["weights"] = new_weights
                    # Keep nearly all training data — minor addition
                    self.training_samples = max(0, int(data.get("training_samples", 0) * 0.95))
                    self.predictions_made = int(data.get("predictions_made", 0) * 0.95)
                    self.correct_predictions = int(data.get("correct_predictions", 0) * 0.95)
                    logger.info(
                        f"🤖 Migration complete - model will retrain with variant feature"
                    )
                else:
                    self.training_samples = data.get("training_samples", 0)
                    self.predictions_made = data.get("predictions_made", 0)
                    self.correct_predictions = data.get("correct_predictions", 0)

                self.model = SimpleLogisticRegression.from_dict(model_data)
                self.min_confidence = data.get("min_confidence", 0.55)

                # Load hybrid predictor state if present
                if self.use_hybrid and "hybrid_predictor" in data:
                    self.hybrid_predictor = HybridPredictor.from_dict(data["hybrid_predictor"])

                logger.info(
                    f"🤖 ML model loaded: {self.training_samples} samples, "
                    f"{self.correct_predictions}/{self.predictions_made} correct, "
                    f"{len(self.model.weights)} features"
                )
        except Exception as e:
            logger.warning(f"Could not load ML model: {e}")


# Singleton instance
_ml_predictor: Optional[MLSignalPredictor] = None


def get_ml_predictor() -> MLSignalPredictor:
    """Get or create the ML predictor singleton."""
    global _ml_predictor
    if _ml_predictor is None:
        _ml_predictor = MLSignalPredictor()
    return _ml_predictor


def calculate_price_trend(prices: list, window: int = 10) -> float:
    """
    Calculate price trend from historical prices.

    Args:
        prices: List of prices - can be raw floats or (timestamp, price) tuples

    Returns:
        Trend value between -1 (strongly down) and 1 (strongly up)
    """
    if len(prices) < 2:
        return 0.0

    # Extract just the prices - handle both tuple format (timestamp, price) and raw float format
    prices_only = []
    for p in prices:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            prices_only.append(float(p[1]))
        elif isinstance(p, (int, float)):
            prices_only.append(float(p))

    if len(prices_only) < 2:
        return 0.0

    # Use the last N prices
    recent = prices_only[-min(window, len(prices_only)):]

    if len(recent) < 2:
        return 0.0

    # Calculate linear regression slope
    n = len(recent)
    x_mean = (n - 1) / 2
    y_mean = sum(recent) / n

    numerator = sum((i - x_mean) * (p - y_mean) for i, p in enumerate(recent))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return 0.0

    slope = numerator / denominator

    # Normalize by average price to get percentage change per sample
    if y_mean > 0:
        normalized_slope = slope / y_mean
        # Scale to -1, 1 range (assuming 1% per sample is extreme)
        trend = max(-1, min(1, normalized_slope * 100))
        return trend

    return 0.0


def extract_ml_features_from_market(
    signal,
    signal_generator,
    market,
    chart_analysis=None,
) -> dict:
    """
    Extract all ML features from signal, market, and signal generator.

    This is a convenience function to gather all enhanced features.

    Args:
        signal: Trading signal
        signal_generator: SignalGenerator instance
        market: MarketState instance
        chart_analysis: Optional ChartAnalysis from Binance candlestick analysis

    Returns:
        Dictionary with all ML feature parameters
    """
    # Get volatility
    volatility = signal_generator.get_volatility(market.asset)

    # Calculate price momentum
    current_price = signal_generator.get_price(market.asset)
    price_momentum = 0.0
    if current_price and market.target_price:
        price_momentum = (current_price - market.target_price) / market.target_price
        price_momentum = max(-1, min(1, price_momentum * 10))

    # Calculate price trend from historical data
    symbol = f"{market.asset.lower()}/usd"
    price_history = signal_generator.price_histories.get(symbol, [])
    price_trend = calculate_price_trend(price_history)

    # Get arb type from signal
    arb_type = getattr(signal, '_arb_type', "none") or "none"

    # Market spread and depth
    spread = market.best_ask - market.best_bid if market.best_ask and market.best_bid else 0.0
    bid_depth = market.bid_depth
    ask_depth = market.ask_depth

    # Distance from target
    distance_from_target = 0.0
    if current_price and market.target_price:
        distance_from_target = (current_price - market.target_price) / market.target_price

    # Binance confirmation features
    binance_lead_pct = 0.0
    binance_confirmation = "NONE"

    # Get Binance confirmation if available
    if hasattr(signal_generator, 'get_binance_confirmation'):
        binance_conf = signal_generator.get_binance_confirmation(
            asset=market.asset,
            target_price=market.target_price,
        )
        binance_lead_pct = binance_conf.get("lead_pct", 0.0) or 0.0
        binance_confirmation = binance_conf.get("confirmation_type", "NONE")

    # Multi-timeframe trends from Binance (1h, 4h, 1d)
    # These provide broader market context for better ML decision making
    trend_1h = 0.0
    trend_4h = 0.0
    trend_1d = 0.0

    try:
        from ..data.binance import get_multi_timeframe_trends
        trends = get_multi_timeframe_trends(market.asset)
        trend_1h = trends.get("trend_1h", 0.0)
        trend_4h = trends.get("trend_4h", 0.0)
        trend_1d = trends.get("trend_1d", 0.0)
    except Exception as e:
        logger.debug(f"Could not fetch multi-timeframe trends: {e}")

    # Chart analysis features (from Binance candlestick data)
    chart_rsi = 50.0
    chart_trend_strength = 0.0
    chart_is_uptrend = 0.0
    chart_is_downtrend = 0.0
    chart_is_ranging = 0.0
    chart_bullish_reversal = 0.0
    chart_bearish_reversal = 0.0
    chart_momentum = 0.0
    chart_bias_bullish = 0.0
    chart_bias_bearish = 0.0
    chart_confidence = 0.5
    chart_bullish_pattern = 0.0
    chart_bearish_pattern = 0.0

    if chart_analysis is not None:
        chart_rsi = chart_analysis.rsi_14
        chart_trend_strength = chart_analysis.trend_strength
        chart_momentum = chart_analysis.momentum
        chart_confidence = chart_analysis.confidence

        # Determine market type
        market_type = chart_analysis.market_type.value
        if "uptrend" in market_type:
            chart_is_uptrend = 1.0
        elif "downtrend" in market_type:
            chart_is_downtrend = 1.0
        elif market_type == "ranging":
            chart_is_ranging = 1.0

        # Trend change
        trend_change = chart_analysis.trend_change.value
        if trend_change == "bullish_reversal":
            chart_bullish_reversal = 1.0
        elif trend_change == "bearish_reversal":
            chart_bearish_reversal = 1.0

        # Bias
        if chart_analysis.bias == "bullish":
            chart_bias_bullish = 1.0
        elif chart_analysis.bias == "bearish":
            chart_bias_bearish = 1.0

        # Patterns
        if chart_analysis.is_bullish_pattern:
            chart_bullish_pattern = 1.0
        if chart_analysis.is_bearish_pattern:
            chart_bearish_pattern = 1.0

    # Advanced chart features (graduated sizing, multi-TF, trend break)
    chart_uncertainty_score = 0.0
    chart_position_multiplier = 1.0
    chart_timeframe_aligned = 1.0
    chart_alignment_score = 1.0
    chart_trend_breaking = 0.0
    chart_resume_ready = 1.0
    chart_resume_confidence = 1.0

    if chart_analysis is not None:
        chart_uncertainty_score = chart_analysis.uncertainty_score
        chart_position_multiplier = chart_analysis.position_size_multiplier
        chart_timeframe_aligned = 1.0 if chart_analysis.timeframes_aligned else 0.0
        chart_alignment_score = chart_analysis.alignment_score
        chart_trend_breaking = 1.0 if chart_analysis.trend_strength_dropping else 0.0
        chart_resume_ready = 1.0 if chart_analysis.resume_ready else 0.0
        chart_resume_confidence = chart_analysis.resume_confidence

    # =========================================================================
    # NEW INDICATOR FEATURES (29 features)
    # These were added to improve ML prediction accuracy
    # =========================================================================

    # MACD features
    macd_histogram = 0.0
    macd_crossover = "none"

    # Bollinger Bands features
    bb_bandwidth = 0.0
    bb_position = "middle"

    # Stochastic features
    stoch_k = 50.0
    stoch_d = 50.0
    stoch_signal = "neutral"

    # RSI Divergence features
    rsi_divergence = "none"
    rsi_divergence_strength = 0.0

    # Volume features
    volume_ratio = 1.0
    is_high_volume = False
    obv_trend = 0.0

    # Heiken Ashi features
    ha_trend = "neutral"
    ha_consecutive = 0
    ha_strength = 0.0

    # VWAP features
    vwap_distance_pct = 0.0
    vwap_position = "at"

    if chart_analysis is not None:
        # MACD
        macd_histogram = getattr(chart_analysis, 'macd_histogram', 0.0) or 0.0
        macd_crossover = getattr(chart_analysis, 'macd_crossover', "none") or "none"

        # Bollinger Bands
        bb_bandwidth = getattr(chart_analysis, 'bb_bandwidth', 0.0) or 0.0
        bb_position = getattr(chart_analysis, 'bb_position', "middle") or "middle"

        # Stochastic
        stoch_k = getattr(chart_analysis, 'stoch_k', 50.0) or 50.0
        stoch_d = getattr(chart_analysis, 'stoch_d', 50.0) or 50.0
        stoch_signal = getattr(chart_analysis, 'stoch_signal', "neutral") or "neutral"

        # RSI Divergence
        rsi_divergence = getattr(chart_analysis, 'rsi_divergence', "none") or "none"
        rsi_divergence_strength = getattr(chart_analysis, 'rsi_divergence_strength', 0.0) or 0.0

        # Volume
        volume_ratio = getattr(chart_analysis, 'volume_ratio', 1.0) or 1.0
        is_high_volume = getattr(chart_analysis, 'is_high_volume', False) or False
        obv_trend = getattr(chart_analysis, 'obv_trend', 0.0) or 0.0

        # Heiken Ashi
        ha_trend = getattr(chart_analysis, 'ha_trend', "neutral") or "neutral"
        ha_consecutive = getattr(chart_analysis, 'ha_consecutive', 0) or 0
        ha_strength = getattr(chart_analysis, 'ha_strength', 0.0) or 0.0

        # VWAP
        vwap_distance_pct = getattr(chart_analysis, 'vwap_distance_pct', 0.0) or 0.0
        vwap_position = getattr(chart_analysis, 'vwap_position', "at") or "at"

    return {
        "volatility": volatility,
        "price_momentum": price_momentum,
        "arb_type": arb_type,
        "spread": spread,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "price_trend": price_trend,
        "distance_from_target": distance_from_target,
        "binance_lead_pct": binance_lead_pct,
        "binance_confirmation": binance_confirmation,
        "trend_1h": trend_1h,
        "trend_4h": trend_4h,
        "trend_1d": trend_1d,
        # Pass current and target price for sanity checks
        "current_price": current_price or 0.0,
        "target_price": market.target_price if market else 0.0,
        # Chart analysis features
        "chart_rsi": chart_rsi,
        "chart_trend_strength": chart_trend_strength,
        "chart_is_uptrend": chart_is_uptrend,
        "chart_is_downtrend": chart_is_downtrend,
        "chart_is_ranging": chart_is_ranging,
        "chart_bullish_reversal": chart_bullish_reversal,
        "chart_bearish_reversal": chart_bearish_reversal,
        "chart_momentum": chart_momentum,
        "chart_bias_bullish": chart_bias_bullish,
        "chart_bias_bearish": chart_bias_bearish,
        "chart_confidence": chart_confidence,
        "chart_bullish_pattern": chart_bullish_pattern,
        "chart_bearish_pattern": chart_bearish_pattern,
        # Advanced chart features
        "chart_uncertainty_score": chart_uncertainty_score,
        "chart_position_multiplier": chart_position_multiplier,
        "chart_timeframe_aligned": chart_timeframe_aligned,
        "chart_alignment_score": chart_alignment_score,
        "chart_trend_breaking": chart_trend_breaking,
        "chart_resume_ready": chart_resume_ready,
        "chart_resume_confidence": chart_resume_confidence,
        # NEW INDICATOR FEATURES (17 parameters for 29 one-hot encoded features)
        "macd_histogram": macd_histogram,
        "macd_crossover": macd_crossover,
        "bb_bandwidth": bb_bandwidth,
        "bb_position": bb_position,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "stoch_signal": stoch_signal,
        "rsi_divergence": rsi_divergence,
        "rsi_divergence_strength": rsi_divergence_strength,
        "volume_ratio": volume_ratio,
        "is_high_volume": is_high_volume,
        "obv_trend": obv_trend,
        "ha_trend": ha_trend,
        "ha_consecutive": ha_consecutive,
        "ha_strength": ha_strength,
        "vwap_distance_pct": vwap_distance_pct,
        "vwap_position": vwap_position,
    }
