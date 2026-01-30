"""
Kelly Criterion Position Sizing

Implements optimal bet sizing based on edge and win probability.
Uses fractional Kelly for safety (typically 0.25-0.5 of full Kelly).

The Kelly Criterion maximizes long-term growth rate while managing risk.
Full Kelly can be volatile, so fractional Kelly is recommended.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class KellyCalculator:
    """
    Kelly Criterion calculator for optimal position sizing.

    The Kelly formula: f* = (bp - q) / b
    Where:
        f* = fraction of bankroll to bet
        b = odds received on the bet (net odds)
        p = probability of winning
        q = probability of losing (1 - p)

    For binary outcomes with even payoff:
        f* = 2p - 1 (simplified)

    For prediction markets:
        f* = (edge) / (odds - 1) when edge = p - market_price
    """

    # Fractional Kelly multiplier (0.25 = quarter Kelly, safer)
    kelly_fraction: float = 0.25

    # Minimum confidence to apply Kelly (below this, use minimum size)
    min_confidence: float = 0.52

    # Maximum Kelly fraction (cap to prevent over-betting)
    max_kelly_fraction: float = 0.15  # Never bet more than 15% of bankroll

    # Minimum position size as fraction of bankroll
    min_position_fraction: float = 0.02  # At least 2% of bankroll

    # Track Kelly performance
    kelly_bets: int = 0
    kelly_wins: int = 0
    total_kelly_edge: float = 0.0

    def calculate_kelly_fraction(
        self,
        win_probability: float,
        market_price: float,
        edge: Optional[float] = None,
    ) -> float:
        """
        Calculate the optimal Kelly fraction for a bet.

        Args:
            win_probability: ML predicted probability of winning (0-1)
            market_price: Current market price / cost to enter (0-1)
            edge: Optional pre-calculated edge (win_prob - market_price)

        Returns:
            Optimal fraction of bankroll to bet (0-1)
        """
        # Calculate edge if not provided
        if edge is None:
            edge = win_probability - market_price

        # No bet if negative or zero edge
        if edge <= 0:
            return 0.0

        # No bet if confidence too low
        if win_probability < self.min_confidence:
            return self.min_position_fraction

        # For binary prediction markets:
        # Payoff is 1/price - 1 if we win (e.g., buy at 0.4, win 1.0, profit = 1.5x)
        # Loss is 1.0 (lose entire stake)

        # Net odds (b) = (1 / market_price) - 1
        # For market_price = 0.4: odds = 2.5 - 1 = 1.5 (150% profit if win)

        if market_price <= 0 or market_price >= 1:
            return self.min_position_fraction

        odds = (1 / market_price) - 1

        # Kelly formula: f* = (b*p - q) / b
        # Where p = win_probability, q = 1 - p, b = odds
        p = win_probability
        q = 1 - p

        if odds <= 0:
            return self.min_position_fraction

        kelly = (odds * p - q) / odds

        # Apply fractional Kelly for safety
        fractional_kelly = kelly * self.kelly_fraction

        # Clamp to min/max bounds
        fractional_kelly = max(self.min_position_fraction, fractional_kelly)
        fractional_kelly = min(self.max_kelly_fraction, fractional_kelly)

        return fractional_kelly

    def calculate_position_size(
        self,
        bankroll: float,
        win_probability: float,
        market_price: float,
        edge: Optional[float] = None,
        max_position_usd: Optional[float] = None,
    ) -> tuple[float, float, dict]:
        """
        Calculate the optimal position size in USD.

        Args:
            bankroll: Current bankroll in USD
            win_probability: ML predicted win probability
            market_price: Current market price
            edge: Optional pre-calculated edge
            max_position_usd: Optional maximum position size cap

        Returns:
            Tuple of (position_size_usd, kelly_fraction, details_dict)
        """
        # Calculate Kelly fraction
        kelly_frac = self.calculate_kelly_fraction(
            win_probability=win_probability,
            market_price=market_price,
            edge=edge,
        )

        # Calculate position size
        position_size = bankroll * kelly_frac

        # Apply max cap if provided
        if max_position_usd is not None:
            position_size = min(position_size, max_position_usd)

        # Minimum viable trade size
        if position_size < 3.0:
            position_size = 3.0
            kelly_frac = position_size / bankroll if bankroll > 0 else 0

        details = {
            "kelly_fraction": kelly_frac,
            "full_kelly": kelly_frac / self.kelly_fraction if self.kelly_fraction > 0 else 0,
            "win_probability": win_probability,
            "market_price": market_price,
            "edge": edge if edge is not None else win_probability - market_price,
            "bankroll": bankroll,
            "position_size": position_size,
        }

        return position_size, kelly_frac, details

    def record_outcome(self, won: bool, edge: float):
        """Record bet outcome for Kelly performance tracking."""
        self.kelly_bets += 1
        if won:
            self.kelly_wins += 1
        self.total_kelly_edge += edge

    def get_stats(self) -> dict:
        """Get Kelly performance statistics."""
        win_rate = self.kelly_wins / max(1, self.kelly_bets)
        avg_edge = self.total_kelly_edge / max(1, self.kelly_bets)

        return {
            "total_bets": self.kelly_bets,
            "wins": self.kelly_wins,
            "win_rate": win_rate,
            "average_edge": avg_edge,
            "kelly_fraction": self.kelly_fraction,
            "expected_growth": win_rate * avg_edge if avg_edge > 0 else 0,
        }


@dataclass
class DynamicKelly:
    """
    Dynamic Kelly that adjusts fraction based on recent performance.

    Increases Kelly fraction when winning, decreases when losing.
    This provides automatic risk adjustment based on strategy performance.
    """

    base_kelly: KellyCalculator = field(default_factory=KellyCalculator)

    # Performance tracking window
    recent_outcomes: list = field(default_factory=list)
    window_size: int = 20

    # Dynamic adjustment bounds
    min_kelly_multiplier: float = 0.5   # Can go down to 50% of base
    max_kelly_multiplier: float = 1.5   # Can go up to 150% of base

    # Current multiplier
    current_multiplier: float = 1.0

    def update_multiplier(self):
        """Update Kelly multiplier based on recent performance."""
        if len(self.recent_outcomes) < 5:
            self.current_multiplier = 1.0
            return

        # Calculate recent win rate
        recent_wins = sum(1 for o in self.recent_outcomes if o > 0)
        recent_win_rate = recent_wins / len(self.recent_outcomes)

        # Adjust multiplier based on win rate
        # Win rate > 60%: increase Kelly
        # Win rate < 40%: decrease Kelly
        if recent_win_rate > 0.60:
            # Winning streak - increase gradually
            self.current_multiplier = min(
                self.max_kelly_multiplier,
                self.current_multiplier * 1.05
            )
        elif recent_win_rate < 0.40:
            # Losing streak - decrease more aggressively
            self.current_multiplier = max(
                self.min_kelly_multiplier,
                self.current_multiplier * 0.90
            )
        else:
            # Neutral - drift toward 1.0
            self.current_multiplier = 0.95 * self.current_multiplier + 0.05 * 1.0

    def record_outcome(self, pnl: float, edge: float):
        """Record trade outcome and update multiplier."""
        self.recent_outcomes.append(pnl)
        if len(self.recent_outcomes) > self.window_size:
            self.recent_outcomes.pop(0)

        self.base_kelly.record_outcome(won=pnl > 0, edge=edge)
        self.update_multiplier()

    def calculate_position_size(
        self,
        bankroll: float,
        win_probability: float,
        market_price: float,
        edge: Optional[float] = None,
        max_position_usd: Optional[float] = None,
    ) -> tuple[float, float, dict]:
        """Calculate position size with dynamic Kelly adjustment."""
        # Get base Kelly calculation
        position_size, kelly_frac, details = self.base_kelly.calculate_position_size(
            bankroll=bankroll,
            win_probability=win_probability,
            market_price=market_price,
            edge=edge,
            max_position_usd=max_position_usd,
        )

        # Apply dynamic multiplier
        adjusted_size = position_size * self.current_multiplier
        adjusted_kelly = kelly_frac * self.current_multiplier

        # Apply max cap again after adjustment
        if max_position_usd is not None:
            adjusted_size = min(adjusted_size, max_position_usd)

        # Update details
        details["dynamic_multiplier"] = self.current_multiplier
        details["adjusted_position_size"] = adjusted_size
        details["adjusted_kelly_fraction"] = adjusted_kelly
        details["recent_win_rate"] = (
            sum(1 for o in self.recent_outcomes if o > 0) / len(self.recent_outcomes)
            if self.recent_outcomes else 0.5
        )

        return adjusted_size, adjusted_kelly, details

    def get_stats(self) -> dict:
        """Get dynamic Kelly statistics."""
        base_stats = self.base_kelly.get_stats()
        base_stats["dynamic_multiplier"] = self.current_multiplier
        base_stats["recent_outcomes"] = len(self.recent_outcomes)
        if self.recent_outcomes:
            base_stats["recent_win_rate"] = (
                sum(1 for o in self.recent_outcomes if o > 0) / len(self.recent_outcomes)
            )
        return base_stats


# Singleton instance
_kelly_calculator: Optional[DynamicKelly] = None


def get_kelly_calculator() -> DynamicKelly:
    """Get or create the Kelly calculator singleton."""
    global _kelly_calculator
    if _kelly_calculator is None:
        _kelly_calculator = DynamicKelly()
    return _kelly_calculator


def calculate_kelly_position(
    bankroll: float,
    win_probability: float,
    market_price: float,
    edge: Optional[float] = None,
    max_position_usd: Optional[float] = None,
) -> tuple[float, dict]:
    """
    Convenience function to calculate Kelly position size.

    Args:
        bankroll: Current bankroll in USD
        win_probability: ML predicted win probability
        market_price: Current market price
        edge: Optional pre-calculated edge
        max_position_usd: Optional maximum position size cap

    Returns:
        Tuple of (position_size_usd, details_dict)
    """
    kelly = get_kelly_calculator()
    position_size, _, details = kelly.calculate_position_size(
        bankroll=bankroll,
        win_probability=win_probability,
        market_price=market_price,
        edge=edge,
        max_position_usd=max_position_usd,
    )
    return position_size, details
