"""
Arbitrage Strategy Module - Based on Successful Bot Patterns

Key insights from research:
1. Don't predict direction - exploit mispricing
2. Buy whichever side is temporarily cheap
3. Enter when YES + NO < 1.0 (binary arbitrage)
4. Enter when price drops 15%+ rapidly (dump detection)
5. Hedge when possible to lock in profits
6. Focus on first 2 minutes of the 15-minute period

Sources:
- Bots earning $5-10k daily on 15-minute BTC/ETH markets
- 98% win rate achieved by exploiting latency between Chainlink and Polymarket
- $313 -> $414k in one month using this strategy
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ..models import MarketState, Signal, Side, OrderAction
from ..config import BotConfig

logger = logging.getLogger(__name__)


@dataclass
class PriceSnapshot:
    """Snapshot of market prices at a point in time."""
    timestamp: datetime
    up_ask: float
    up_bid: float
    down_ask: float  # 1 - up_bid
    down_bid: float  # 1 - up_ask
    chainlink_price: float


@dataclass
class ArbitrageDetector:
    """
    Detects arbitrage opportunities in 15-minute crypto markets.

    Key strategies:
    1. Binary mispricing: YES + NO < 1.0 (buy both for guaranteed profit)
    2. Asymmetric pricing: One side temporarily too cheap
    3. Dump detection: Rapid 15%+ price drop creates opportunity
    4. Hedge execution: Lock in profit when leg1 + opposite <= threshold
    """

    config: BotConfig

    # Price history for each market (for dump detection)
    price_history: dict[str, list[PriceSnapshot]] = field(default_factory=dict)

    # Active arbitrage legs (for hedging)
    active_legs: dict[str, dict] = field(default_factory=dict)

    @property
    def dump_threshold(self) -> float:
        """Price drop % that triggers entry."""
        return self.config.trading.arb_dump_threshold

    @property
    def hedge_sum_threshold(self) -> float:
        """Hedge when leg1 + opposite <= this."""
        return self.config.trading.arb_hedge_threshold

    @property
    def entry_window_minutes(self) -> float:
        """Focus on first N minutes of period."""
        return self.config.trading.arb_entry_window_minutes

    @property
    def min_spread_profit(self) -> float:
        """Minimum profit after fees."""
        return self.config.trading.arb_min_spread

    def record_prices(
        self,
        market: MarketState,
        chainlink_price: float,
    ):
        """Record price snapshot for dump detection."""
        market_id = market.condition_id

        if market_id not in self.price_history:
            self.price_history[market_id] = []

        snapshot = PriceSnapshot(
            timestamp=datetime.now(timezone.utc),
            up_ask=market.best_ask,
            up_bid=market.best_bid,
            down_ask=1 - market.best_bid,
            down_bid=1 - market.best_ask,
            chainlink_price=chainlink_price,
        )

        self.price_history[market_id].append(snapshot)

        # Keep only last 60 snapshots (~30 seconds at 0.5s intervals)
        if len(self.price_history[market_id]) > 60:
            self.price_history[market_id] = self.price_history[market_id][-60:]

    def detect_opportunities(
        self,
        market: MarketState,
        chainlink_price: float,
    ) -> list[dict]:
        """
        Detect all arbitrage opportunities in a market.

        Returns list of opportunity dicts with:
        - type: "binary_arb", "asymmetric", "dump", "hedge"
        - side: Side.UP or Side.DOWN
        - edge: Expected profit percentage
        - price: Entry price
        - reasoning: Human-readable explanation
        """
        opportunities = []

        # Record current prices
        self.record_prices(market, chainlink_price)

        # Check timing window (first 2 minutes preferred)
        period_elapsed = 900 - market.time_remaining  # seconds into period
        in_entry_window = period_elapsed <= (self.entry_window_minutes * 60)

        # 1. Binary arbitrage: YES + NO < 1.0
        binary_opp = self._check_binary_arbitrage(market)
        if binary_opp:
            opportunities.append(binary_opp)

        # 2. Asymmetric pricing: One side too cheap
        asym_opp = self._check_asymmetric_pricing(market, chainlink_price)
        if asym_opp:
            opportunities.append(asym_opp)

        # 3. Dump detection: Rapid price drop
        if in_entry_window:
            dump_opp = self._check_dump_opportunity(market)
            if dump_opp:
                opportunities.append(dump_opp)

        # 4. Hedge check: Can we lock in profit on existing leg?
        hedge_opp = self._check_hedge_opportunity(market)
        if hedge_opp:
            opportunities.append(hedge_opp)

        return opportunities

    def _check_binary_arbitrage(self, market: MarketState) -> Optional[dict]:
        """
        Check for binary arbitrage where YES + NO < 1.0

        If we can buy both sides for less than $1, we're guaranteed profit.
        Example: YES @ $0.48, NO @ $0.49 = $0.97 total -> $0.03 profit
        """
        yes_cost = market.best_ask  # Cost to buy YES
        no_cost = 1 - market.best_bid  # Cost to buy NO
        total_cost = yes_cost + no_cost

        # Account for 2% fee on winning outcome
        fee_adjusted_profit = 1.0 - total_cost - 0.02

        if fee_adjusted_profit >= self.min_spread_profit:
            # Buy both sides!
            return {
                "type": "binary_arb",
                "side": Side.UP,  # Buy both, but signal UP as primary
                "buy_both": True,
                "edge": fee_adjusted_profit,
                "yes_price": yes_cost,
                "no_price": no_cost,
                "price": yes_cost,  # Primary entry
                "reasoning": (
                    f"BINARY ARB: YES@{yes_cost:.2f} + NO@{no_cost:.2f} = "
                    f"{total_cost:.2f} < $1.00 | Profit: ${fee_adjusted_profit:.3f}/share"
                ),
            }

        return None

    def _check_asymmetric_pricing(
        self,
        market: MarketState,
        chainlink_price: float,
    ) -> Optional[dict]:
        """
        Check for asymmetric pricing where one side is too cheap.

        Key insight: Don't predict direction - buy whichever side is cheap.
        If YES + NO should be ~1.0 but one side is temporarily cheap,
        that's an opportunity.

        UPDATED: More conservative fair value calculation that accounts for
        time remaining and uses realistic probability estimates.
        """
        yes_price = market.best_ask
        no_price = 1 - market.best_bid

        # Calculate distance from target as percentage
        distance_pct = (chainlink_price - market.target_price) / market.target_price

        # More conservative fair value calculation:
        # - For small distances (<1%), fair value should be close to 0.50
        # - Only significant price moves justify higher fair values
        # - Use square root scaling to be less aggressive
        # - Cap at 0.65 instead of 0.70 to be more conservative
        abs_distance = abs(distance_pct)

        # Minimum distance to consider asymmetric (0.2% = 20 basis points)
        if abs_distance < 0.002:
            return None

        # Fair value scales with square root of distance for more conservative estimate
        # At 0.5% distance: 0.50 + sqrt(0.005) * 2 = 0.50 + 0.14 = 0.64
        # At 1% distance: 0.50 + sqrt(0.01) * 2 = 0.50 + 0.20 = 0.70 (capped at 0.65)
        import math
        fair_value_adjustment = min(0.15, math.sqrt(abs_distance) * 2)
        fair_value = min(0.65, 0.50 + fair_value_adjustment)

        # If price is above target, YES should be expensive
        if distance_pct > 0.002:  # Price above target by at least 0.2%
            fair_yes = fair_value
            # Check if YES is underpriced (market hasn't caught up)
            mispricing = fair_yes - yes_price
            if mispricing > 0.05:  # At least 5% mispricing
                edge = mispricing - 0.02  # minus 2% fee
                if edge >= self.min_spread_profit:
                    return {
                        "type": "asymmetric",
                        "side": Side.UP,
                        "edge": edge,
                        "price": yes_price,
                        "reasoning": (
                            f"ASYMMETRIC: Price ${chainlink_price:,.0f} > target "
                            f"${market.target_price:,.0f}, YES@{yes_price:.2f} "
                            f"below fair {fair_yes:.2f}"
                        ),
                    }

        elif distance_pct < -0.002:  # Price below target by at least 0.2%
            fair_no = fair_value
            # Check if NO is underpriced (market hasn't caught up)
            mispricing = fair_no - no_price
            if mispricing > 0.05:  # At least 5% mispricing
                edge = mispricing - 0.02
                if edge >= self.min_spread_profit:
                    return {
                        "type": "asymmetric",
                        "side": Side.DOWN,
                        "edge": edge,
                        "price": no_price,
                        "reasoning": (
                            f"ASYMMETRIC: Price ${chainlink_price:,.0f} < target "
                            f"${market.target_price:,.0f}, NO@{no_price:.2f} "
                            f"below fair {fair_no:.2f}"
                        ),
                    }

        return None

    def _check_dump_opportunity(self, market: MarketState) -> Optional[dict]:
        """
        Check for rapid price dump creating opportunity.

        If a side drops 15%+ within ~3 seconds, it's likely a temporary
        dislocation that will revert.
        """
        market_id = market.condition_id
        history = self.price_history.get(market_id, [])

        if len(history) < 6:  # Need ~3 seconds of data
            return None

        # Get prices from ~3 seconds ago
        recent = history[-6:]
        old_snapshot = recent[0]
        new_snapshot = recent[-1]

        # Check YES dump
        if old_snapshot.up_ask > 0.20:  # Avoid very low prices
            yes_drop = (old_snapshot.up_ask - new_snapshot.up_ask) / old_snapshot.up_ask
            if yes_drop >= self.dump_threshold:
                edge = yes_drop - 0.02  # Expect partial recovery
                return {
                    "type": "dump",
                    "side": Side.UP,
                    "edge": edge,
                    "price": new_snapshot.up_ask,
                    "reasoning": (
                        f"DUMP DETECTED: YES dropped {yes_drop:.0%} in 3s "
                        f"({old_snapshot.up_ask:.2f} -> {new_snapshot.up_ask:.2f})"
                    ),
                }

        # Check NO dump
        if old_snapshot.down_ask > 0.20:
            no_drop = (old_snapshot.down_ask - new_snapshot.down_ask) / old_snapshot.down_ask
            if no_drop >= self.dump_threshold:
                edge = no_drop - 0.02
                return {
                    "type": "dump",
                    "side": Side.DOWN,
                    "edge": edge,
                    "price": new_snapshot.down_ask,
                    "reasoning": (
                        f"DUMP DETECTED: NO dropped {no_drop:.0%} in 3s "
                        f"({old_snapshot.down_ask:.2f} -> {new_snapshot.down_ask:.2f})"
                    ),
                }

        return None

    def record_leg1_entry(
        self,
        market: MarketState,
        side: Side,
        entry_price: float,
    ):
        """Record that we entered leg 1 of an arb trade."""
        self.active_legs[market.condition_id] = {
            "side": side,
            "entry_price": entry_price,
            "entry_time": datetime.now(timezone.utc),
        }
        logger.info(f"LEG1 recorded: {market.asset} {side.value} @ {entry_price:.2f}")

    def _check_hedge_opportunity(self, market: MarketState) -> Optional[dict]:
        """
        Check if we can hedge an existing leg to lock in profit.

        If leg1_entry + opposite_ask <= threshold (0.95), execute leg2.
        This guarantees profit regardless of outcome.
        """
        market_id = market.condition_id
        leg1 = self.active_legs.get(market_id)

        if not leg1:
            return None

        leg1_price = leg1["entry_price"]
        leg1_side = leg1["side"]

        # Get opposite side price
        if leg1_side == Side.UP:
            opposite_price = 1 - market.best_bid  # NO price
        else:
            opposite_price = market.best_ask  # YES price

        combined = leg1_price + opposite_price

        if combined <= self.hedge_sum_threshold:
            profit = 1.0 - combined - 0.02  # Guaranteed profit minus fee

            return {
                "type": "hedge",
                "side": Side.DOWN if leg1_side == Side.UP else Side.UP,
                "edge": profit,
                "price": opposite_price,
                "is_leg2": True,
                "reasoning": (
                    f"HEDGE: Leg1 {leg1_side.value}@{leg1_price:.2f} + "
                    f"Leg2@{opposite_price:.2f} = {combined:.2f} | "
                    f"Locked profit: ${profit:.3f}/share"
                ),
            }

        return None

    def clear_leg(self, market_id: str):
        """Clear an active leg after hedge or market settlement."""
        if market_id in self.active_legs:
            del self.active_legs[market_id]

    def clear_market_history(self, market_id: str):
        """Clear price history for a market (on new period)."""
        if market_id in self.price_history:
            del self.price_history[market_id]


def select_best_opportunity(opportunities: list[dict]) -> Optional[dict]:
    """
    Select the best opportunity from a list.

    Priority:
    1. Hedge (lock in guaranteed profit)
    2. Binary arbitrage (guaranteed profit)
    3. Highest edge opportunity
    """
    if not opportunities:
        return None

    # Hedge always wins (locks in profit)
    hedges = [o for o in opportunities if o["type"] == "hedge"]
    if hedges:
        return max(hedges, key=lambda x: x["edge"])

    # Binary arbitrage is also guaranteed
    binary = [o for o in opportunities if o["type"] == "binary_arb"]
    if binary:
        return max(binary, key=lambda x: x["edge"])

    # Otherwise pick highest edge
    return max(opportunities, key=lambda x: x["edge"])
