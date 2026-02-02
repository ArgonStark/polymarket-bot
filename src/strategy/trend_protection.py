"""
Trend Protection Module

Comprehensive safeguards to prevent trading against strong market trends.
This module integrates multiple protection mechanisms:

1. Trend Alignment Filter - Block trades against strong trends
2. Price Velocity Guard - Skip when price moving too fast
3. Multi-Timeframe Agreement - Require trend alignment across 1h, 4h, 1d
4. Binance Momentum Check - Use Binance's speed advantage
5. Dynamic Edge Requirement - Adjust min_edge by trend strength
6. Consecutive Move Detection - Detect sustained movements

Usage:
    protection = TrendProtection(config)
    result = protection.evaluate_trade(
        signal_side=Side.UP,
        asset="BTC",
        current_price=100500,
        target_price=100000,
        base_edge=0.05,
        price_history=[(timestamp, price), ...],
        trends={"trend_1h": 0.3, "trend_4h": 0.2, "trend_1d": 0.1},
        binance_data={...}
    )

    if result.should_trade:
        # Proceed with adjusted edge
        final_edge = result.adjusted_edge
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ProtectionResult(Enum):
    """Result of protection check."""
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"


@dataclass
class TrendProtectionResult:
    """Result of trend protection evaluation."""
    should_trade: bool
    adjusted_edge: float
    edge_adjustment: float  # Total adjustment applied
    reason: str
    details: dict = field(default_factory=dict)

    # Individual check results
    trend_aligned: bool = True
    velocity_ok: bool = True
    timeframes_agree: bool = True
    momentum_confirmed: bool = False
    consecutive_detected: bool = False


@dataclass
class TrendProtectionConfig:
    """Configuration for trend protection."""

    # === Trend Alignment Filter ===
    enabled: bool = True

    # Block trades against strong daily trend
    max_opposite_trend_1d: float = 0.25  # Block if 1d trend > this against signal
    max_opposite_trend_1h: float = 0.40  # Block if 1h trend > this against signal

    # === Price Velocity Guard ===
    velocity_guard_enabled: bool = True

    # Max velocity per asset (% per second) - skip if exceeded
    # Higher = more lenient = more trades allowed
    max_velocity: dict = field(default_factory=lambda: {
        "BTC": 0.00035,  # ~2.1% per minute (raised to trade more)
        "ETH": 0.00038,  # ~2.3% per minute (raised to trade more)
        "SOL": 0.00030,  # ~1.8% per minute
        "XRP": 0.00025,  # ~1.5% per minute (lowered to trade less)
    })

    # === Multi-Timeframe Agreement ===
    timeframe_agreement_enabled: bool = True
    timeframe_alignment_threshold: float = 0.1  # Consider trend if > this

    # Edge adjustments for alignment
    edge_boost_all_aligned: float = 0.01  # +1% when all 3 timeframes agree
    edge_penalty_mixed: float = 0.01  # -1% when signals mixed

    # === Binance Momentum ===
    binance_momentum_enabled: bool = True
    binance_velocity_threshold: float = 0.001  # 0.1% per second
    binance_lead_momentum_threshold: float = 0.003  # 0.3% per second

    # Edge boosts for Binance momentum
    binance_velocity_boost: float = 0.02  # +2% for strong velocity
    binance_momentum_boost: float = 0.01  # +1% for lead momentum

    # === Dynamic Edge Requirement ===
    dynamic_edge_enabled: bool = True

    # Multipliers for min_edge based on trend strength
    trend_multipliers: dict = field(default_factory=lambda: {
        "no_trend": 1.5,        # < 0.2 strength
        "weak": 1.2,            # 0.2 - 0.4
        "moderate": 1.0,        # 0.4 - 0.6
        "strong": 0.8,          # 0.6 - 0.8
        "very_strong": 0.6,     # > 0.8
    })

    # Bounds for dynamic edge
    dynamic_edge_min: float = 0.015  # Never go below 1.5%
    dynamic_edge_max: float = 0.05   # Never exceed 5%

    # === Consecutive Move Detection ===
    consecutive_enabled: bool = True
    consecutive_min_moves: int = 3  # Minimum moves to trigger
    consecutive_window: int = 10    # Price points to analyze

    # Edge boosts for consecutive moves
    consecutive_boost_per_move: float = 0.005  # +0.5% per move after minimum
    consecutive_boost_max: float = 0.02  # Max +2%


class TrendProtection:
    """
    Unified trend protection system.

    Evaluates multiple factors to determine if a trade should be taken
    and adjusts edge based on market conditions.
    """

    def __init__(self, config: Optional[TrendProtectionConfig] = None):
        self.config = config or TrendProtectionConfig()

    def evaluate_trade(
        self,
        signal_side: str,  # "UP" or "DOWN"
        asset: str,
        current_price: float,
        target_price: float,
        base_edge: float,
        base_min_edge: float,
        price_history: list,  # List of (timestamp, price) or just prices
        trends: dict,  # {"trend_1h": float, "trend_4h": float, "trend_1d": float}
        binance_price: Optional[float] = None,
        binance_velocity: float = 0.0,
        chainlink_velocity: float = 0.0,
    ) -> TrendProtectionResult:
        """
        Evaluate all protection checks and return trading decision.

        Args:
            signal_side: "UP" or "DOWN"
            asset: Asset symbol (BTC, ETH, SOL, XRP)
            current_price: Current Chainlink price
            target_price: Market target price
            base_edge: Calculated edge before adjustments
            base_min_edge: Base minimum edge requirement
            price_history: Recent price history
            trends: Multi-timeframe trends from Binance
            binance_price: Current Binance price
            binance_velocity: Binance price velocity
            chainlink_velocity: Chainlink price velocity

        Returns:
            TrendProtectionResult with decision and adjusted edge
        """
        result = TrendProtectionResult(
            should_trade=True,
            adjusted_edge=base_edge,
            edge_adjustment=0.0,
            reason="",
            details={}
        )

        edge_adjustments = []
        block_reasons = []
        warn_reasons = []

        # === 1. Trend Alignment Filter ===
        if self.config.enabled:
            trend_check = self._check_trend_alignment(signal_side, trends)
            result.trend_aligned = trend_check["aligned"]
            result.details["trend_alignment"] = trend_check

            if trend_check["result"] == ProtectionResult.BLOCK:
                result.should_trade = False
                block_reasons.append(trend_check["reason"])
            elif trend_check["result"] == ProtectionResult.WARN:
                warn_reasons.append(trend_check["reason"])

        # === 2. Price Velocity Guard ===
        if self.config.velocity_guard_enabled and result.should_trade:
            velocity_check = self._check_price_velocity(asset, chainlink_velocity)
            result.velocity_ok = velocity_check["ok"]
            result.details["velocity"] = velocity_check

            if velocity_check["result"] == ProtectionResult.BLOCK:
                result.should_trade = False
                block_reasons.append(velocity_check["reason"])

        # === 3. Multi-Timeframe Agreement ===
        if self.config.timeframe_agreement_enabled and result.should_trade:
            tf_check = self._check_timeframe_agreement(signal_side, trends)
            result.timeframes_agree = tf_check["agree"]
            result.details["timeframe_agreement"] = tf_check

            if tf_check["result"] == ProtectionResult.BLOCK:
                result.should_trade = False
                block_reasons.append(tf_check["reason"])
            elif tf_check["edge_adjustment"] != 0:
                edge_adjustments.append(("timeframe", tf_check["edge_adjustment"]))

        # === 4. Binance Momentum Check ===
        if self.config.binance_momentum_enabled and result.should_trade and binance_price:
            momentum_check = self._check_binance_momentum(
                signal_side, binance_velocity, chainlink_velocity,
                binance_price, current_price, target_price
            )
            result.momentum_confirmed = momentum_check["confirmed"]
            result.details["binance_momentum"] = momentum_check

            if momentum_check["edge_adjustment"] > 0:
                edge_adjustments.append(("binance_momentum", momentum_check["edge_adjustment"]))

        # === 5. Dynamic Edge Requirement ===
        if self.config.dynamic_edge_enabled and result.should_trade:
            dynamic_check = self._calculate_dynamic_edge(
                price_history, base_min_edge, trends
            )
            result.details["dynamic_edge"] = dynamic_check

            # Check if edge meets dynamic requirement
            current_edge = base_edge + sum(adj[1] for adj in edge_adjustments)
            if current_edge < dynamic_check["min_edge"]:
                result.should_trade = False
                block_reasons.append(
                    f"Edge {current_edge:.1%} < dynamic min {dynamic_check['min_edge']:.1%} "
                    f"(trend strength: {dynamic_check['trend_strength']:.0%})"
                )

        # === 6. Consecutive Move Detection ===
        if self.config.consecutive_enabled and result.should_trade:
            consecutive_check = self._detect_consecutive_moves(price_history, signal_side)
            result.consecutive_detected = consecutive_check["detected"]
            result.details["consecutive"] = consecutive_check

            if consecutive_check["edge_adjustment"] > 0:
                edge_adjustments.append(("consecutive", consecutive_check["edge_adjustment"]))

        # === Calculate Final Edge ===
        total_adjustment = sum(adj[1] for adj in edge_adjustments)
        result.adjusted_edge = base_edge + total_adjustment
        result.edge_adjustment = total_adjustment

        # === Build Reason String ===
        if block_reasons:
            result.reason = "BLOCKED: " + "; ".join(block_reasons)
        elif warn_reasons:
            result.reason = "WARN: " + "; ".join(warn_reasons)
        else:
            adjustments_str = ", ".join(
                f"{name}={adj:+.1%}" for name, adj in edge_adjustments if adj != 0
            )
            if adjustments_str:
                result.reason = f"Edge adjusted: {adjustments_str}"
            else:
                result.reason = "All checks passed"

        return result

    def _check_trend_alignment(
        self, signal_side: str, trends: dict
    ) -> dict:
        """
        Check if signal aligns with strong trends.

        Block trades against strong daily trend.
        """
        trend_1h = trends.get("trend_1h", 0.0)
        trend_4h = trends.get("trend_4h", 0.0)
        trend_1d = trends.get("trend_1d", 0.0)

        is_up_signal = signal_side.upper() == "UP"

        # Check daily trend first (strongest signal)
        if is_up_signal and trend_1d < -self.config.max_opposite_trend_1d:
            return {
                "aligned": False,
                "result": ProtectionResult.BLOCK,
                "reason": f"UP signal blocked: 1d trend {trend_1d:.0%} is strongly bearish",
                "trend_1d": trend_1d,
            }

        if not is_up_signal and trend_1d > self.config.max_opposite_trend_1d:
            return {
                "aligned": False,
                "result": ProtectionResult.BLOCK,
                "reason": f"DOWN signal blocked: 1d trend {trend_1d:.0%} is strongly bullish",
                "trend_1d": trend_1d,
            }

        # Check hourly trend
        if is_up_signal and trend_1h < -self.config.max_opposite_trend_1h:
            return {
                "aligned": False,
                "result": ProtectionResult.BLOCK,
                "reason": f"UP signal blocked: 1h trend {trend_1h:.0%} is strongly bearish",
                "trend_1h": trend_1h,
            }

        if not is_up_signal and trend_1h > self.config.max_opposite_trend_1h:
            return {
                "aligned": False,
                "result": ProtectionResult.BLOCK,
                "reason": f"DOWN signal blocked: 1h trend {trend_1h:.0%} is strongly bullish",
                "trend_1h": trend_1h,
            }

        return {
            "aligned": True,
            "result": ProtectionResult.ALLOW,
            "reason": "Trend alignment OK",
            "trend_1h": trend_1h,
            "trend_4h": trend_4h,
            "trend_1d": trend_1d,
        }

    def _check_price_velocity(self, asset: str, velocity: float) -> dict:
        """
        Check if price is moving too fast.

        Skip trading during extreme price movements.
        """
        max_velocity = self.config.max_velocity.get(asset.upper(), 0.0003)
        abs_velocity = abs(velocity)

        if abs_velocity > max_velocity:
            return {
                "ok": False,
                "result": ProtectionResult.BLOCK,
                "reason": f"Price velocity {abs_velocity:.4%}/s exceeds max {max_velocity:.4%}/s",
                "velocity": velocity,
                "max_velocity": max_velocity,
            }

        return {
            "ok": True,
            "result": ProtectionResult.ALLOW,
            "reason": "Velocity OK",
            "velocity": velocity,
            "max_velocity": max_velocity,
        }

    def _check_timeframe_agreement(
        self, signal_side: str, trends: dict
    ) -> dict:
        """
        Check multi-timeframe trend agreement.

        Returns edge adjustment based on alignment.
        """
        trend_1h = trends.get("trend_1h", 0.0)
        trend_4h = trends.get("trend_4h", 0.0)
        trend_1d = trends.get("trend_1d", 0.0)

        threshold = self.config.timeframe_alignment_threshold
        is_up_signal = signal_side.upper() == "UP"

        # Count aligned timeframes
        aligned_count = 0
        opposite_count = 0

        for trend in [trend_1h, trend_4h, trend_1d]:
            if is_up_signal:
                if trend > threshold:
                    aligned_count += 1
                elif trend < -threshold:
                    opposite_count += 1
            else:
                if trend < -threshold:
                    aligned_count += 1
                elif trend > threshold:
                    opposite_count += 1

        # Block if all timeframes strongly oppose
        if opposite_count == 3:
            return {
                "agree": False,
                "result": ProtectionResult.BLOCK,
                "reason": f"All 3 timeframes oppose {signal_side} signal",
                "aligned": aligned_count,
                "opposite": opposite_count,
                "edge_adjustment": 0.0,
            }

        # Calculate edge adjustment
        edge_adjustment = 0.0
        if aligned_count == 3:
            edge_adjustment = self.config.edge_boost_all_aligned
            reason = f"All 3 timeframes aligned (+{edge_adjustment:.1%} edge)"
        elif aligned_count >= 2:
            edge_adjustment = self.config.edge_boost_all_aligned / 2
            reason = f"{aligned_count}/3 timeframes aligned (+{edge_adjustment:.1%} edge)"
        elif opposite_count >= 2:
            edge_adjustment = -self.config.edge_penalty_mixed
            reason = f"{opposite_count}/3 timeframes oppose (-{abs(edge_adjustment):.1%} edge)"
        else:
            reason = "Mixed timeframe signals"

        return {
            "agree": aligned_count >= 2,
            "result": ProtectionResult.ALLOW,
            "reason": reason,
            "aligned": aligned_count,
            "opposite": opposite_count,
            "edge_adjustment": edge_adjustment,
        }

    def _check_binance_momentum(
        self,
        signal_side: str,
        binance_velocity: float,
        chainlink_velocity: float,
        binance_price: float,
        chainlink_price: float,
        target_price: float,
    ) -> dict:
        """
        Check Binance momentum for confirmation.

        Binance updates faster than Chainlink, giving leading signal.
        """
        is_up_signal = signal_side.upper() == "UP"

        # Calculate velocity lead (how much faster Binance is moving)
        velocity_lead = binance_velocity - chainlink_velocity

        # Calculate lead (Binance ahead of Chainlink)
        if chainlink_price > 0:
            lead_pct = (binance_price - chainlink_price) / chainlink_price
        else:
            lead_pct = 0.0

        # Calculate distance to target
        if target_price > 0:
            distance_to_target = (binance_price - target_price) / target_price
        else:
            distance_to_target = 0.0

        edge_adjustment = 0.0
        confirmations = []

        # Check velocity lead
        if is_up_signal and velocity_lead > self.config.binance_velocity_threshold:
            edge_adjustment += self.config.binance_velocity_boost
            confirmations.append(f"velocity lead +{velocity_lead:.3%}/s")
        elif not is_up_signal and velocity_lead < -self.config.binance_velocity_threshold:
            edge_adjustment += self.config.binance_velocity_boost
            confirmations.append(f"velocity lead {velocity_lead:.3%}/s")

        # Check if Binance is moving toward target faster
        if abs(binance_velocity) > self.config.binance_lead_momentum_threshold:
            if (is_up_signal and binance_velocity > 0) or (not is_up_signal and binance_velocity < 0):
                edge_adjustment += self.config.binance_momentum_boost
                confirmations.append(f"strong momentum {binance_velocity:.3%}/s")

        confirmed = edge_adjustment > 0

        return {
            "confirmed": confirmed,
            "result": ProtectionResult.ALLOW,
            "reason": ", ".join(confirmations) if confirmations else "No Binance confirmation",
            "velocity_lead": velocity_lead,
            "lead_pct": lead_pct,
            "distance_to_target": distance_to_target,
            "edge_adjustment": edge_adjustment,
        }

    def _calculate_dynamic_edge(
        self,
        price_history: list,
        base_min_edge: float,
        trends: dict,
    ) -> dict:
        """
        Calculate dynamic min_edge based on trend strength.

        Stronger trends = lower edge requirement (more confidence).
        Weaker/no trends = higher edge requirement (need more buffer).
        """
        # Calculate trend strength from price history
        trend_strength = self._calculate_trend_strength(price_history, trends)

        # Determine multiplier
        if trend_strength < 0.2:
            multiplier = self.config.trend_multipliers["no_trend"]
            category = "no_trend"
        elif trend_strength < 0.4:
            multiplier = self.config.trend_multipliers["weak"]
            category = "weak"
        elif trend_strength < 0.6:
            multiplier = self.config.trend_multipliers["moderate"]
            category = "moderate"
        elif trend_strength < 0.8:
            multiplier = self.config.trend_multipliers["strong"]
            category = "strong"
        else:
            multiplier = self.config.trend_multipliers["very_strong"]
            category = "very_strong"

        # Calculate dynamic edge with bounds
        dynamic_edge = base_min_edge * multiplier
        dynamic_edge = max(self.config.dynamic_edge_min,
                         min(dynamic_edge, self.config.dynamic_edge_max))

        return {
            "min_edge": dynamic_edge,
            "base_min_edge": base_min_edge,
            "multiplier": multiplier,
            "trend_strength": trend_strength,
            "category": category,
        }

    def _calculate_trend_strength(self, price_history: list, trends: dict) -> float:
        """
        Calculate overall trend strength (0-1).

        Combines short-term price movement with multi-timeframe trends.
        """
        if not price_history or len(price_history) < 5:
            # Use multi-timeframe trends if no price history
            trend_1h = abs(trends.get("trend_1h", 0.0))
            trend_4h = abs(trends.get("trend_4h", 0.0))
            trend_1d = abs(trends.get("trend_1d", 0.0))
            return max(trend_1h, trend_4h, trend_1d)

        # Extract prices from history
        if isinstance(price_history[0], (list, tuple)):
            prices = [p[1] for p in price_history[-20:]]
        else:
            prices = price_history[-20:]

        if len(prices) < 3:
            return 0.0

        # Calculate price trend using simple linear regression
        n = len(prices)
        x_mean = (n - 1) / 2
        y_mean = sum(prices) / n

        numerator = sum((i - x_mean) * (p - y_mean) for i, p in enumerate(prices))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator

        # Normalize slope by average price
        if y_mean > 0:
            normalized_slope = abs(slope / y_mean) * 100  # Percentage change per sample
        else:
            normalized_slope = 0.0

        # Combine with multi-timeframe trends (weighted average)
        price_strength = min(normalized_slope, 1.0)
        tf_strength = max(
            abs(trends.get("trend_1h", 0.0)),
            abs(trends.get("trend_4h", 0.0)),
            abs(trends.get("trend_1d", 0.0))
        )

        # 60% weight to multi-timeframe, 40% to price history
        combined_strength = 0.4 * price_strength + 0.6 * tf_strength

        return min(combined_strength, 1.0)

    def _detect_consecutive_moves(
        self, price_history: list, signal_side: str
    ) -> dict:
        """
        Detect consecutive price moves in same direction.

        3+ consecutive moves in signal direction = momentum confirmation.
        """
        if not price_history or len(price_history) < 4:
            return {
                "detected": False,
                "count": 0,
                "direction": None,
                "magnitude": 0.0,
                "edge_adjustment": 0.0,
            }

        # Extract prices
        window = self.config.consecutive_window
        if isinstance(price_history[0], (list, tuple)):
            prices = [p[1] for p in price_history[-window:]]
        else:
            prices = price_history[-window:]

        if len(prices) < 4:
            return {
                "detected": False,
                "count": 0,
                "direction": None,
                "magnitude": 0.0,
                "edge_adjustment": 0.0,
            }

        # Detect consecutive moves
        is_up_signal = signal_side.upper() == "UP"

        # Count consecutive moves in signal direction
        consecutive_count = 0
        for i in range(len(prices) - 1, 0, -1):
            if is_up_signal and prices[i] > prices[i-1]:
                consecutive_count += 1
            elif not is_up_signal and prices[i] < prices[i-1]:
                consecutive_count += 1
            else:
                break

        # Calculate magnitude
        if consecutive_count >= 1 and prices[0] > 0:
            start_idx = len(prices) - consecutive_count - 1
            start_price = prices[max(0, start_idx)]
            end_price = prices[-1]
            magnitude = (end_price - start_price) / start_price
        else:
            magnitude = 0.0

        # Calculate edge boost if enough consecutive moves
        min_moves = self.config.consecutive_min_moves
        if consecutive_count >= min_moves:
            extra_moves = consecutive_count - min_moves
            edge_adjustment = (extra_moves + 1) * self.config.consecutive_boost_per_move
            edge_adjustment = min(edge_adjustment, self.config.consecutive_boost_max)

            return {
                "detected": True,
                "count": consecutive_count,
                "direction": "UP" if is_up_signal else "DOWN",
                "magnitude": magnitude,
                "edge_adjustment": edge_adjustment,
            }

        return {
            "detected": False,
            "count": consecutive_count,
            "direction": None,
            "magnitude": magnitude,
            "edge_adjustment": 0.0,
        }


def calculate_price_velocity(
    price_history: list,
    window: int = 10
) -> float:
    """
    Calculate price velocity from history.

    Args:
        price_history: List of (timestamp, price) tuples or just prices
        window: Number of recent prices to use

    Returns:
        Velocity as percentage change per second
    """
    if not price_history or len(price_history) < 2:
        return 0.0

    recent = price_history[-window:]
    if len(recent) < 2:
        return 0.0

    # Check if timestamps are included
    if isinstance(recent[0], (list, tuple)) and len(recent[0]) >= 2:
        # Has timestamps
        first_time, first_price = recent[0]
        last_time, last_price = recent[-1]

        # Calculate time difference
        if hasattr(first_time, 'timestamp') and hasattr(last_time, 'timestamp'):
            time_diff = last_time.timestamp() - first_time.timestamp()
        else:
            time_diff = float(last_time - first_time)

        if time_diff <= 0 or first_price <= 0:
            return 0.0

        price_change = (last_price - first_price) / first_price
        return price_change / time_diff
    else:
        # No timestamps - assume ~0.5 second intervals (Chainlink update rate)
        prices = recent if not isinstance(recent[0], (list, tuple)) else [p[1] for p in recent]
        if len(prices) < 2 or prices[0] <= 0:
            return 0.0

        price_change = (prices[-1] - prices[0]) / prices[0]
        estimated_time = (len(prices) - 1) * 0.5  # ~0.5 sec per update

        return price_change / estimated_time if estimated_time > 0 else 0.0
