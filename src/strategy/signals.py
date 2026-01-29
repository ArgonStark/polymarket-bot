"""
Signal generation module for temporal arbitrage strategy.

Generates trading signals by comparing calculated true probabilities
(based on Chainlink prices) with market odds.

ENHANCED: Now includes arbitrage detection based on successful bot patterns:
- Binary mispricing (YES + NO < 1.0)
- Asymmetric pricing (buy the cheap side)
- Dump detection (15%+ drops in 3 seconds)
- Hedge execution (lock in profits)

BINANCE CONFIRMATION: Uses Binance prices as a leading indicator
- Binance moves ~50-500ms faster than Chainlink
- When Binance crosses target before Chainlink → early signal
- Used as CONFIRMATION, not primary signal:
  * Chainlink crossed target → STRONG
  * Chainlink near target + Binance crossed → MEDIUM (boost)
  * Only Binance crossed → WEAK (don't trade, watch)
"""

import logging
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timezone

from ..models import MarketState, Signal, Side, OrderAction
from ..probability import (
    calculate_true_probability,
    calculate_edge,
    estimate_volatility,
)
from ..config import BotConfig
from .arbitrage import ArbitrageDetector, select_best_opportunity


logger = logging.getLogger(__name__)


# Binance confirmation thresholds
# "Near target" means within this percentage of target price
NEAR_TARGET_THRESHOLD = 0.002  # 0.2% - considered "near" target

# Edge boost when Binance confirms direction
BINANCE_CONFIRMATION_BOOST = 0.02  # Add 2% to edge when Binance confirms


@dataclass
class SignalGenerator:
    """
    Generates trading signals for 15-minute crypto markets.

    Core strategy: Exploit temporal arbitrage between real-time
    Chainlink prices and lagging market odds.

    ENHANCED: Now uses ArbitrageDetector for pattern-based opportunities:
    - Binary mispricing (YES + NO < 1.0)
    - Asymmetric pricing (buy the cheap side)
    - Dump detection (15%+ drops in 3 seconds)
    - Hedge execution (lock in profits)

    BINANCE CONFIRMATION: Uses faster Binance prices as leading indicator
    - Binance updates ~50ms, Chainlink ~500ms+
    - Provides early warning when target will be crossed
    """

    config: BotConfig

    # Price data (updated by data feeds)
    chainlink_prices: dict[str, float] = None
    price_histories: dict[str, list[float]] = None

    # Binance prices (faster, leading indicator)
    binance_prices: dict[str, float] = None
    binance_last_update: dict[str, datetime] = None

    # Volatility estimates
    volatilities: dict[str, float] = None

    # Arbitrage detector for pattern-based opportunities
    arb_detector: ArbitrageDetector = None

    def __post_init__(self):
        """Initialize data structures."""
        if self.chainlink_prices is None:
            self.chainlink_prices = {}
        if self.price_histories is None:
            self.price_histories = {}
        if self.binance_prices is None:
            self.binance_prices = {}
        if self.binance_last_update is None:
            self.binance_last_update = {}
        if self.volatilities is None:
            self.volatilities = {}
            # Initialize with default volatilities
            for asset in self.config.supported_assets:
                self.volatilities[asset.lower()] = self.config.volatility.get(asset)
        if self.arb_detector is None:
            self.arb_detector = ArbitrageDetector(config=self.config)

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

    def update_binance_price(self, asset: str, price: float):
        """
        Update Binance price for an asset.

        Binance prices are faster (~50ms updates) and used as
        a leading indicator to predict Chainlink movements.

        Args:
            asset: Asset symbol like "BTC", "ETH", etc.
        """
        asset_lower = asset.lower()
        self.binance_prices[asset_lower] = price
        self.binance_last_update[asset_lower] = datetime.now(timezone.utc)

    def get_binance_price(self, asset: str) -> Optional[float]:
        """Get current Binance price for an asset."""
        return self.binance_prices.get(asset.lower())

    def get_binance_lead(self, asset: str) -> Optional[float]:
        """
        Calculate Binance "lead" over Chainlink.

        Lead = Binance price - Chainlink price

        Positive lead: Binance is ahead (price rising faster)
        Negative lead: Binance is behind (price falling faster)

        Returns:
            Lead in absolute price, or None if prices unavailable
        """
        chainlink = self.get_price(asset)
        binance = self.get_binance_price(asset)

        if chainlink is None or binance is None:
            return None

        return binance - chainlink

    def get_binance_lead_pct(self, asset: str) -> Optional[float]:
        """
        Calculate Binance lead as percentage of price.

        Returns:
            Lead as decimal (e.g., 0.001 = 0.1%), or None
        """
        chainlink = self.get_price(asset)
        lead = self.get_binance_lead(asset)

        if chainlink is None or lead is None or chainlink == 0:
            return None

        return lead / chainlink

    def get_binance_confirmation(
        self,
        asset: str,
        target_price: float,
    ) -> dict:
        """
        Get Binance confirmation signal for a target price.

        Analyzes whether Binance is confirming the direction relative to target.

        Returns dict with:
            - binance_price: Current Binance price
            - chainlink_price: Current Chainlink price
            - lead: Binance - Chainlink difference
            - lead_pct: Lead as percentage
            - binance_crossed: True if Binance crossed target
            - chainlink_crossed: True if Chainlink crossed target
            - binance_direction: "above", "below", or "at" target
            - chainlink_direction: "above", "below", or "at" target
            - confirmation_type: "STRONG", "MEDIUM", "WEAK", or "NONE"
            - edge_boost: Extra edge to add (0 if no confirmation)
        """
        chainlink = self.get_price(asset)
        binance = self.get_binance_price(asset)

        result = {
            "binance_price": binance,
            "chainlink_price": chainlink,
            "lead": None,
            "lead_pct": None,
            "binance_crossed": False,
            "chainlink_crossed": False,
            "binance_direction": "unknown",
            "chainlink_direction": "unknown",
            "confirmation_type": "NONE",
            "edge_boost": 0.0,
        }

        if chainlink is None:
            return result

        # Calculate Chainlink direction relative to target
        chainlink_distance = (chainlink - target_price) / target_price
        if chainlink >= target_price:
            result["chainlink_direction"] = "above"
            result["chainlink_crossed"] = True
        else:
            result["chainlink_direction"] = "below"
            result["chainlink_crossed"] = False

        # If no Binance data, just return Chainlink-only result
        if binance is None:
            if result["chainlink_crossed"]:
                result["confirmation_type"] = "STRONG"
            return result

        # Calculate lead
        result["lead"] = binance - chainlink
        result["lead_pct"] = result["lead"] / chainlink if chainlink != 0 else 0

        # Calculate Binance direction relative to target
        binance_distance = (binance - target_price) / target_price
        if binance >= target_price:
            result["binance_direction"] = "above"
            result["binance_crossed"] = True
        else:
            result["binance_direction"] = "below"
            result["binance_crossed"] = False

        # Determine confirmation type based on signal strength:
        # STRONG: Chainlink crossed target (existing behavior)
        # MEDIUM: Chainlink near target AND Binance crossed (confirmation boost)
        # WEAK: Only Binance crossed (early signal, don't trade yet)
        # NONE: Neither crossed

        chainlink_near_target = abs(chainlink_distance) < NEAR_TARGET_THRESHOLD

        if result["chainlink_crossed"]:
            # Chainlink already crossed - this is the strong signal
            result["confirmation_type"] = "STRONG"
            # Binance confirmation adds a small boost
            if result["binance_crossed"] and result["binance_direction"] == result["chainlink_direction"]:
                result["edge_boost"] = BINANCE_CONFIRMATION_BOOST / 2  # 1% extra

        elif chainlink_near_target and result["binance_crossed"]:
            # Chainlink is close but hasn't crossed, Binance has crossed
            # This is a MEDIUM signal - Chainlink likely to follow soon
            result["confirmation_type"] = "MEDIUM"
            result["edge_boost"] = BINANCE_CONFIRMATION_BOOST  # 2% boost

        elif result["binance_crossed"]:
            # Only Binance crossed, Chainlink is still far
            # This is a WEAK signal - don't trade, but log for analysis
            result["confirmation_type"] = "WEAK"
            # No edge boost - too risky to trade on Binance alone

        return result

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

        ENHANCED: Now uses two-stage approach:
        1. Check for arbitrage opportunities (pattern-based)
        2. Fall back to probability-based edge calculation

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

        # STAGE 1: Check for arbitrage opportunities (pattern-based)
        arb_signal = self._check_arbitrage_opportunities(market, current_price, time_remaining)
        if arb_signal:
            return arb_signal

        # STAGE 2: Fall back to probability-based edge calculation
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

        # Get Binance confirmation signal
        binance_conf = self.get_binance_confirmation(
            asset=market.asset,
            target_price=market.target_price,
        )

        # Apply Binance edge boost if confirmed
        edge_boost = binance_conf.get("edge_boost", 0.0)
        if edge_boost > 0:
            # Apply boost to the side that matches Binance direction
            if binance_conf["binance_direction"] == "above":
                edge_up += edge_boost
            else:
                edge_down += edge_boost

        # Log edge calculation with Binance info
        binance_price = binance_conf.get("binance_price")
        binance_info = ""
        if binance_price is not None:
            lead_pct = binance_conf.get("lead_pct", 0) or 0
            conf_type = binance_conf.get("confirmation_type", "NONE")
            binance_info = (
                f", Binance=${binance_price:,.2f} (lead={lead_pct:+.3%}), "
                f"conf={conf_type}"
            )
            if edge_boost > 0:
                binance_info += f", boost=+{edge_boost:.1%}"

        logger.debug(
            f"EDGE CALC {market.asset}: "
            f"Chainlink=${current_price:,.2f}, Target=${market.target_price:,.2f}, "
            f"true_prob_up={true_prob_up:.1%}, "
            f"market_bid={market.best_bid:.2f}, market_ask={market.best_ask:.2f}, "
            f"edge_up={edge_up:.1%}, edge_down={edge_down:.1%}, min_edge={min_edge:.1%}"
            f"{binance_info}"
        )

        # Log early Binance signals (WEAK confirmation - Binance crossed but Chainlink hasn't)
        if binance_conf["confirmation_type"] == "WEAK":
            logger.info(
                f"⚡ BINANCE EARLY SIGNAL [{market.asset}]: "
                f"Binance ${binance_price:,.2f} crossed target ${market.target_price:,.2f} "
                f"({binance_conf['binance_direction']}) | "
                f"Chainlink ${current_price:,.2f} still {binance_conf['chainlink_direction']} | "
                f"Lead: {binance_conf.get('lead_pct', 0) or 0:+.3%} | "
                f"Watching for Chainlink to follow..."
            )

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
            best_edge = max(edge_up, edge_down)
            if best_edge > 0.01:  # Only log if edge is somewhat close
                logger.debug(
                    f"SKIP {market.asset}: best edge {best_edge:.1%} < min {min_edge:.1%}"
                )
            return Signal(
                market=market,
                side=Side.NONE,
                edge=best_edge,
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

        # Generate reasoning with Binance info
        reasoning = self._generate_reasoning(
            current_price=current_price,
            target_price=market.target_price,
            true_prob=true_prob,
            market_prob=market_prob,
            edge=edge,
            time_remaining=time_remaining,
            volatility=volatility,
            binance_confirmation=binance_conf,
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

        UPDATED: Conservative near expiry to avoid losses.
        Getting stuck in a position with no time to recover is costly.

        Decision matrix:
        - time < 60s: SKIP (too risky, not enough time for order to fill/profit)
        - time < 90s AND edge < 10%: SKIP (need strong edge for short time)
        - time < 90s AND edge >= 10%: MARKET (urgent, take the opportunity)
        - time >= 90s AND edge >= 3%: LIMIT (normal trading)
        - Otherwise -> SKIP
        """
        trading = self.config.trading

        # Very short time: Skip entirely (too risky)
        if time_remaining < 60:
            logger.debug(f"Skipping trade: only {time_remaining:.0f}s remaining (< 60s minimum)")
            return OrderAction.SKIP

        # Short time (60-90s): Require strong edge, use MARKET orders
        if time_remaining < 90:
            if edge >= 0.10:  # 10% edge minimum for short time trades
                logger.debug(f"Short time trade: {time_remaining:.0f}s, edge={edge:.1%} - using MARKET")
                return OrderAction.MARKET
            else:
                logger.debug(f"Skipping: {time_remaining:.0f}s remaining, edge {edge:.1%} < 10% required")
                return OrderAction.SKIP

        # Normal time (90s+): Standard logic
        # MARKET orders: Use when we have high edge AND limited time
        if edge >= trading.edge_for_market and time_remaining < 120:
            return OrderAction.MARKET

        # LIMIT orders: Default choice for reliable execution
        elif edge >= trading.min_edge:
            return OrderAction.LIMIT

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
            # Very aggressive price for guaranteed immediate fill
            # For UP: pay up to 0.99 for UP token
            # For DOWN: pay up to 0.99 for DOWN token
            return 0.99

        elif action == OrderAction.LIMIT:
            # Buy at best ask to fill immediately
            # For UP: buy UP token at its best_ask
            # For DOWN: buy DOWN token at (1 - UP's best_bid)
            if side == Side.UP:
                return min(0.99, market.best_ask)
            else:
                return min(0.99, 1 - market.best_bid)

        # Fallback
        return market.best_ask if side == Side.UP else (1 - market.best_bid)

    def _calculate_position_size(
        self,
        edge: float,
        price: float,
    ) -> tuple[float, float]:
        """
        Calculate position size in USD and shares.

        Uses Kelly-inspired sizing: bet more when edge is higher.
        Position size scales from base_position_size up to 2x
        based on edge strength.

        Returns:
            Tuple of (size_usd, size_shares)
        """
        trading = self.config.trading

        # Base position size
        base_size = trading.base_position_size

        # Kelly-inspired scaling: higher edge = larger position
        # Clamp edge between 0 and 0.5 for scaling
        edge_factor = min(edge, 0.5) / 0.5  # 0 to 1 based on edge

        # Scale from 1x to 2x base size based on edge
        # 10% edge = 1x, 50% edge = 2x
        kelly_multiplier = 1.0 + edge_factor

        size_usd = base_size * kelly_multiplier

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
        binance_confirmation: Optional[dict] = None,
    ) -> str:
        """Generate human-readable reasoning for the signal."""
        direction = "above" if current_price >= target_price else "below"
        distance_pct = abs(current_price - target_price) / target_price * 100

        base_reasoning = (
            f"Chainlink ${current_price:,.2f} is {distance_pct:.2f}% {direction} "
            f"target ${target_price:,.2f}. "
            f"True prob {true_prob:.1%} vs market {market_prob:.1%}. "
            f"Edge {edge:.1%}, time {time_remaining:.0f}s, vol {volatility:.3%}"
        )

        # Add Binance confirmation info if available
        if binance_confirmation:
            binance_price = binance_confirmation.get("binance_price")
            conf_type = binance_confirmation.get("confirmation_type", "NONE")
            edge_boost = binance_confirmation.get("edge_boost", 0)

            if binance_price is not None:
                lead_pct = binance_confirmation.get("lead_pct", 0) or 0
                binance_info = (
                    f" | Binance ${binance_price:,.2f} ({lead_pct:+.3%} lead), "
                    f"conf={conf_type}"
                )
                if edge_boost > 0:
                    binance_info += f" +{edge_boost:.1%} boost"
                base_reasoning += binance_info

        return base_reasoning

    def _check_arbitrage_opportunities(
        self,
        market: MarketState,
        current_price: float,
        time_remaining: float,
    ) -> Optional[Signal]:
        """
        Check for arbitrage opportunities using pattern detection.

        This implements strategies from successful bots:
        - Binary mispricing (YES + NO < 1.0)
        - Asymmetric pricing (buy the cheap side)
        - Dump detection (15%+ drops in 3 seconds)
        - Hedge execution (lock in profits)

        Returns:
            Signal if opportunity found, None otherwise
        """
        # Detect all opportunities
        opportunities = self.arb_detector.detect_opportunities(market, current_price)

        if not opportunities:
            return None

        # Select the best opportunity
        best = select_best_opportunity(opportunities)

        if not best:
            return None

        # Log the opportunity
        logger.info(
            f"🎯 ARB [{market.asset}]: {best['type'].upper()} | "
            f"Edge: {best['edge']:.1%} | {best['reasoning']}"
        )

        # Create signal from opportunity
        side = best["side"]
        edge = best["edge"]
        price = best["price"]

        # Determine action (usually LIMIT for arb)
        action = OrderAction.LIMIT
        if best["type"] == "hedge":
            action = OrderAction.LIMIT  # Always LIMIT for hedges

        # Calculate position size
        size_usd, size_shares = self._calculate_position_size(edge, price)

        # NOTE: LEG1 is recorded in bot.py AFTER successful trade execution
        # This prevents recording legs for trades that don't execute due to
        # cooldown, insufficient balance, or execution failures.

        # Clear leg if this is a hedge
        if best.get("is_leg2"):
            self.arb_detector.clear_leg(market.condition_id)

        # Mark signal as arbitrage-based
        signal = Signal(
            market=market,
            side=side,
            edge=edge,
            true_prob=0.5,  # Not probability-based
            market_prob=price,
            recommended_action=action,
            recommended_price=price,
            size_usd=size_usd,
            size_shares=size_shares,
            chainlink_price=current_price,
            time_remaining=time_remaining,
            reasoning=f"[ARB] {best['reasoning']}",
        )

        # Store arb type for tracking
        signal._arb_type = best["type"]
        signal._is_hedge = best.get("is_leg2", False)

        return signal

    def record_trade_result(self, market_id: str, won: bool):
        """
        Record trade result for arbitrage learning.

        Called when a position settles.
        """
        # Clear any active leg for this market
        self.arb_detector.clear_leg(market_id)
        self.arb_detector.clear_market_history(market_id)
