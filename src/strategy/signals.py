"""
Signal generation module for temporal arbitrage strategy.

Generates trading signals by comparing calculated true probabilities
(based on Chainlink prices) with market odds.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from ..models import MarketState, Signal, Side, OrderAction
from ..probability import (
    calculate_true_probability,
    calculate_edge,
    estimate_volatility,
)
from ..config import BotConfig


logger = logging.getLogger(__name__)


@dataclass
class SignalGenerator:
    """
    Generates trading signals for 15-minute crypto markets.

    Core strategy: Exploit temporal arbitrage between real-time
    Chainlink prices and lagging market odds.
    """

    config: BotConfig

    # Price data (updated by data feeds)
    chainlink_prices: dict[str, float] = None
    price_histories: dict[str, list[float]] = None

    # Volatility estimates
    volatilities: dict[str, float] = None

    def __post_init__(self):
        """Initialize data structures."""
        if self.chainlink_prices is None:
            self.chainlink_prices = {}
        if self.price_histories is None:
            self.price_histories = {}
        if self.volatilities is None:
            self.volatilities = {}
            # Initialize with default volatilities
            for asset in self.config.supported_assets:
                self.volatilities[asset.lower()] = self.config.volatility.get(asset)

    def update_price(self, symbol: str, price: float):
        """
        Update Chainlink price for a symbol.

        Args:
            symbol: Symbol like "btc/usd" or "BTC"
        """
        symbol_lower = symbol.lower()
        if "/" not in symbol_lower:
            symbol_lower = f"{symbol_lower}/usd"

        self.chainlink_prices[symbol_lower] = price

        # Update price history
        if symbol_lower not in self.price_histories:
            self.price_histories[symbol_lower] = []
        self.price_histories[symbol_lower].append(price)

        # Keep only last 100 prices
        if len(self.price_histories[symbol_lower]) > 100:
            self.price_histories[symbol_lower] = self.price_histories[symbol_lower][-100:]

        # Update volatility estimate periodically
        if len(self.price_histories[symbol_lower]) >= 5:
            asset = symbol_lower.split("/")[0]
            self.volatilities[asset] = estimate_volatility(
                self.price_histories[symbol_lower],
                window=20,
                default_vol=self.config.volatility.get(asset.upper()),
            )

    def get_price(self, asset: str) -> Optional[float]:
        """Get current Chainlink price for an asset."""
        symbol = f"{asset.lower()}/usd"
        return self.chainlink_prices.get(symbol)

    def get_volatility(self, asset: str) -> float:
        """Get volatility estimate for an asset."""
        return self.volatilities.get(
            asset.lower(),
            self.config.volatility.get(asset.upper()),
        )

    def generate_signal(self, market: MarketState) -> Optional[Signal]:
        """
        Generate a trading signal for a market.

        Calculates true probability based on Chainlink price
        and compares with market odds to find edge.

        Args:
            market: Market state to analyze

        Returns:
            Signal object or None if no valid signal
        """
        # Get current Chainlink price
        current_price = self.get_price(market.asset)
        if current_price is None:
            logger.debug(f"No price data for {market.asset}")
            return None

        # Check time remaining
        time_remaining = market.time_remaining
        if time_remaining < self.config.trading.min_time_remaining:
            logger.debug(
                f"Market {market.asset} has only {time_remaining:.0f}s remaining"
            )
            return None

        # Get volatility
        volatility = self.get_volatility(market.asset)

        # Calculate true probability for UP outcome
        true_prob_up = calculate_true_probability(
            current_price=current_price,
            target_price=market.target_price,
            time_remaining_sec=time_remaining,
            volatility_15min=volatility,
        )

        # Get market implied probabilities
        # For UP: cost to buy = best_ask
        # For DOWN: cost to buy = 1 - best_bid (for UP token)
        market_prob_up = market.best_ask
        market_prob_down = 1 - market.best_bid

        # Calculate edges
        edge_up = calculate_edge(true_prob_up, market_prob_up)
        edge_down = calculate_edge(1 - true_prob_up, market_prob_down)

        # Choose best side if edge exceeds minimum
        min_edge = self.config.trading.min_edge

        if edge_up > edge_down and edge_up >= min_edge:
            side = Side.UP
            edge = edge_up
            true_prob = true_prob_up
            market_prob = market_prob_up
            entry_price = market.best_ask
        elif edge_down >= min_edge:
            side = Side.DOWN
            edge = edge_down
            true_prob = 1 - true_prob_up
            market_prob = market_prob_down
            entry_price = 1 - market.best_bid
        else:
            # No sufficient edge - return skip signal
            return Signal(
                market=market,
                side=Side.NONE,
                edge=max(edge_up, edge_down),
                true_prob=true_prob_up,
                market_prob=market_prob_up,
                recommended_action=OrderAction.SKIP,
                recommended_price=0.0,
                size_usd=0.0,
                size_shares=0.0,
                chainlink_price=current_price,
                time_remaining=time_remaining,
                reasoning=(
                    f"Edge too low: UP={edge_up:.1%}, DOWN={edge_down:.1%}. "
                    f"Current ${current_price:,.2f} vs target ${market.target_price:,.2f}"
                ),
            )

        # Determine order action based on edge and time
        action = self._determine_action(edge, time_remaining)

        # Determine recommended price for order
        recommended_price = self._calculate_entry_price(
            side, market, edge, action
        )

        # Calculate position size
        size_usd, size_shares = self._calculate_position_size(
            edge, recommended_price
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            current_price=current_price,
            target_price=market.target_price,
            true_prob=true_prob,
            market_prob=market_prob,
            edge=edge,
            time_remaining=time_remaining,
            volatility=volatility,
        )

        return Signal(
            market=market,
            side=side,
            edge=edge,
            true_prob=true_prob,
            market_prob=market_prob,
            recommended_action=action,
            recommended_price=recommended_price,
            size_usd=size_usd,
            size_shares=size_shares,
            chainlink_price=current_price,
            time_remaining=time_remaining,
            reasoning=reasoning,
        )

    def _determine_action(
        self,
        edge: float,
        time_remaining: float,
    ) -> OrderAction:
        """
        Determine order action based on edge and urgency.

        Decision matrix:
        - edge >= 50% AND time < 60s -> MARKET (guaranteed fill)
        - edge >= 40% AND time < 120s -> LIMIT (may take)
        - edge >= 30% AND time > 120s -> POST_ONLY (earn rebates)
        - Otherwise -> SKIP
        """
        trading = self.config.trading

        if edge >= trading.edge_for_market and time_remaining < trading.time_for_market:
            return OrderAction.MARKET

        elif edge >= trading.edge_for_limit and time_remaining < trading.time_for_limit:
            return OrderAction.LIMIT

        elif edge >= trading.edge_for_post_only and time_remaining > trading.time_for_limit:
            return OrderAction.POST_ONLY

        else:
            return OrderAction.SKIP

    def _calculate_entry_price(
        self,
        side: Side,
        market: MarketState,
        edge: float,
        action: OrderAction,
    ) -> float:
        """Calculate recommended entry price for order."""
        if action == OrderAction.MARKET:
            # Aggressive price for immediate fill
            return 0.99 if side == Side.UP else 0.01

        elif action == OrderAction.LIMIT:
            # Slightly aggressive price
            if side == Side.UP:
                return min(0.99, market.best_ask + 0.02)
            else:
                return max(0.01, (1 - market.best_bid) + 0.02)

        elif action == OrderAction.POST_ONLY:
            # Price at or better than current bid to sit on book
            if side == Side.UP:
                return min(0.99, market.best_bid + 0.01)
            else:
                return max(0.01, (1 - market.best_ask) + 0.01)

        return market.best_ask if side == Side.UP else (1 - market.best_bid)

    def _calculate_position_size(
        self,
        edge: float,
        price: float,
    ) -> tuple[float, float]:
        """
        Calculate position size in USD and shares.

        Uses base position size from config, potentially
        adjusted by edge strength.

        Returns:
            Tuple of (size_usd, size_shares)
        """
        trading = self.config.trading

        # Start with base position size
        size_usd = trading.base_position_size

        # Could add Kelly sizing or edge-based scaling here
        # For now, use fixed size

        # Convert to shares
        price = max(0.01, min(0.99, price))
        size_shares = size_usd / price

        return (size_usd, size_shares)

    def _generate_reasoning(
        self,
        current_price: float,
        target_price: float,
        true_prob: float,
        market_prob: float,
        edge: float,
        time_remaining: float,
        volatility: float,
    ) -> str:
        """Generate human-readable reasoning for the signal."""
        direction = "above" if current_price >= target_price else "below"
        distance_pct = abs(current_price - target_price) / target_price * 100

        return (
            f"Chainlink ${current_price:,.2f} is {distance_pct:.2f}% {direction} "
            f"target ${target_price:,.2f}. "
            f"True prob {true_prob:.1%} vs market {market_prob:.1%}. "
            f"Edge {edge:.1%}, time {time_remaining:.0f}s, vol {volatility:.3%}"
        )
