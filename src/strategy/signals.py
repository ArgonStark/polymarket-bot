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
from .trend_protection import TrendProtection, TrendProtectionConfig
from .technical_analysis import get_smart_trading_decision

# Import with fallback for get_multi_timeframe_trends
try:
    from ..data.binance import get_multi_timeframe_trends
except ImportError:
    def get_multi_timeframe_trends(asset: str) -> dict:
        return {"trend_1h": 0.0, "trend_4h": 0.0, "trend_1d": 0.0}

# Import Binance chart analyzer for comprehensive technical analysis
try:
    from ..data.binance_chart import (
        analyze_chart,
        get_chart_bias,
        should_pause_trading,
        get_position_size_multiplier,
        record_trade_for_cooldown,
        get_multi_timeframe_decision,
        MarketType,
        TrendChange,
    )
    CHART_ANALYSIS_AVAILABLE = True
except ImportError:
    CHART_ANALYSIS_AVAILABLE = False
    def analyze_chart(asset: str):
        return None
    def get_chart_bias(asset: str):
        return "neutral", 0.5
    def should_pause_trading(asset: str, min_uncertainty: float = 0.5):
        return False, ""
    def get_position_size_multiplier(asset: str):
        return 1.0, "no chart data"
    def record_trade_for_cooldown(asset: str, uncertainty_was_high: bool = False):
        pass
    def get_multi_timeframe_decision(asset: str):
        return "NONE", 0.5, True


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

    # Trend protection to avoid trading against strong trends
    trend_protection: TrendProtection = None

    # Reference to BinanceFeed for velocity calculations
    binance_feed: object = None  # Will be set externally

    # Position size multiplier from chart analysis (graduated sizing)
    _position_multiplier: float = 1.0
    _position_multiplier_reason: str = ""

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
        if self.trend_protection is None:
            self._init_trend_protection()

    def _init_trend_protection(self):
        """Initialize trend protection from config."""
        tp_config = self.config.trend_protection
        trend_config = TrendProtectionConfig(
            enabled=tp_config.enabled,
            max_opposite_trend_1d=tp_config.max_opposite_trend_1d,
            max_opposite_trend_1h=tp_config.max_opposite_trend_1h,
            velocity_guard_enabled=tp_config.velocity_guard_enabled,
            max_velocity={
                "BTC": tp_config.velocity_btc,
                "ETH": tp_config.velocity_eth,
                "SOL": tp_config.velocity_sol,
                "XRP": tp_config.velocity_xrp,
            },
            timeframe_agreement_enabled=tp_config.timeframe_agreement_enabled,
            edge_boost_all_aligned=tp_config.edge_boost_all_aligned,
            edge_penalty_mixed=tp_config.edge_penalty_mixed,
            binance_momentum_enabled=tp_config.binance_momentum_enabled,
            binance_velocity_threshold=tp_config.binance_velocity_threshold,
            binance_velocity_boost=tp_config.binance_momentum_boost,
            binance_momentum_boost=tp_config.binance_momentum_boost,
            dynamic_edge_enabled=tp_config.dynamic_edge_enabled,
            dynamic_edge_min=tp_config.dynamic_edge_min,
            dynamic_edge_max=tp_config.dynamic_edge_max,
            consecutive_enabled=tp_config.consecutive_enabled,
            consecutive_min_moves=tp_config.consecutive_min_moves,
            consecutive_boost_max=tp_config.consecutive_boost_max,
        )
        self.trend_protection = TrendProtection(trend_config)

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

        # Update price history with timestamp
        now = datetime.now(timezone.utc)
        if symbol_lower not in self.price_histories:
            self.price_histories[symbol_lower] = []
        self.price_histories[symbol_lower].append((now, price))

        # Keep only last 100 prices
        if len(self.price_histories[symbol_lower]) > 100:
            self.price_histories[symbol_lower] = self.price_histories[symbol_lower][-100:]

        # Update volatility estimate periodically
        if len(self.price_histories[symbol_lower]) >= 5:
            asset = symbol_lower.split("/")[0]
            # Extract just the prices for volatility calculation
            # Handle both tuple format (timestamp, price) and raw float format
            prices_only = []
            for p in self.price_histories[symbol_lower]:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    prices_only.append(p[1])
                elif isinstance(p, (int, float)):
                    prices_only.append(float(p))
            self.volatilities[asset] = estimate_volatility(
                prices_only,
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

        # Validate target_price to prevent division by zero
        if target_price <= 0:
            logger.warning(f"Invalid target_price: {target_price}")
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

    def get_price_range(self, asset: str) -> tuple[float, float]:
        """
        Get recent high/low price range for an asset.

        Returns:
            Tuple of (high, low) prices from recent price history.
            Returns (0.0, 0.0) if no data available.
        """
        symbol = f"{asset.lower()}/usd"
        history = self.price_histories.get(symbol, [])
        if not history:
            return (0.0, 0.0)

        # Get prices from history (handles both tuple and raw float formats)
        prices = []
        for p in history:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                prices.append(p[1])
            elif isinstance(p, (int, float)):
                prices.append(float(p))

        if not prices:
            return (0.0, 0.0)

        return (max(prices), min(prices))

    def get_price_velocity(self, asset: str) -> float:
        """
        Get price velocity (rate of change) for an asset.

        Returns:
            Price change per second, normalized by price.
            Positive = price increasing, negative = decreasing.
            Returns 0.0 if insufficient data.
        """
        symbol = f"{asset.lower()}/usd"
        history = self.price_histories.get(symbol, [])
        if len(history) < 2:
            return 0.0

        # Get recent prices with timestamps
        recent = history[-10:]  # Last 10 data points
        if len(recent) < 2:
            return 0.0

        # Handle both tuple format (timestamp, price) and raw float format
        first_entry = recent[0]
        last_entry = recent[-1]

        if isinstance(first_entry, (list, tuple)) and len(first_entry) >= 2:
            # Tuple format: (timestamp, price)
            first_time, first_price = first_entry[0], first_entry[1]
            last_time, last_price = last_entry[0], last_entry[1]

            # Calculate time difference
            if hasattr(first_time, 'timestamp') and hasattr(last_time, 'timestamp'):
                time_diff = last_time.timestamp() - first_time.timestamp()
            elif hasattr(first_time, 'total_seconds'):
                time_diff = (last_time - first_time).total_seconds()
            else:
                time_diff = float(last_time - first_time)
        else:
            # Raw float format - assume ~0.5 second intervals
            first_price = float(first_entry)
            last_price = float(last_entry)
            time_diff = (len(recent) - 1) * 0.5

        if time_diff <= 0 or first_price <= 0:
            return 0.0

        # Velocity as percentage change per second
        price_change = (last_price - first_price) / first_price
        velocity = price_change / time_diff

        return velocity

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

        # === COMPREHENSIVE CHART ANALYSIS ===
        # Use Binance candlestick data for advanced trend analysis
        # CHART ANALYSIS IS THE PRIMARY DECISION MAKER
        chart_analysis = None
        chart_decision = None  # Will hold the chart-based trading decision

        if CHART_ANALYSIS_AVAILABLE:
            try:
                chart_analysis = analyze_chart(market.asset)
                if chart_analysis:
                    # Log comprehensive chart analysis at INFO level for visibility
                    uncertainty_info = ""
                    if chart_analysis.is_uncertain:
                        uncertainty_info = f" | ⚠️ UNCERTAIN ({chart_analysis.uncertainty_score:.0%}): {chart_analysis.uncertainty_reason}"

                    logger.info(
                        f"📊 CHART ANALYSIS [{market.asset}]: "
                        f"Type={chart_analysis.market_type.value} | "
                        f"Bias={chart_analysis.bias} ({chart_analysis.confidence:.0%}) | "
                        f"RSI={chart_analysis.rsi_14:.0f} | "
                        f"Trends: 15m={chart_analysis.trend_15m:+.2f}, "
                        f"1h={chart_analysis.trend_1h:+.2f}, "
                        f"4h={chart_analysis.trend_4h:+.2f} | "
                        f"Change={chart_analysis.trend_change.value}"
                        + (f" | Pattern={chart_analysis.pattern_name}" if chart_analysis.pattern_name else "")
                        + uncertainty_info
                    )

                    # === GRADUATED POSITION SIZING ===
                    # Instead of binary pause, we scale position size based on conditions
                    position_multiplier = chart_analysis.position_size_multiplier
                    position_reason = ""

                    # Log position sizing factors
                    sizing_factors = []
                    if chart_analysis.is_uncertain:
                        sizing_factors.append(f"uncertain={chart_analysis.uncertainty_score:.0%}")
                    if not chart_analysis.timeframes_aligned:
                        sizing_factors.append(f"alignment={chart_analysis.alignment_score:.0%}")
                    if chart_analysis.trend_strength_dropping:
                        sizing_factors.append("trend_breaking")
                    if not chart_analysis.resume_ready and chart_analysis.is_uncertain:
                        sizing_factors.append(f"resume_conf={chart_analysis.resume_confidence:.0%}")

                    if sizing_factors:
                        position_reason = ", ".join(sizing_factors)
                        logger.info(
                            f"📏 POSITION SIZING [{market.asset}]: {position_multiplier:.0%} of normal | "
                            f"Factors: {position_reason}"
                        )

                    # === HARD PAUSE: Only pause if position multiplier is 0 AND not ready to resume ===
                    if position_multiplier == 0.0 and not chart_analysis.resume_ready:
                        logger.info(
                            f"⏸️ MARKET PAUSE [{market.asset}]: Full pause - "
                            f"waiting for resume conditions | "
                            f"Uncertainty: {chart_analysis.uncertainty_score:.0%} | "
                            f"Resume confidence: {chart_analysis.resume_confidence:.0%} | "
                            f"Consecutive candles: {chart_analysis.consecutive_candles_same_dir} | "
                            f"Reason: {chart_analysis.uncertainty_reason}"
                        )
                        return Signal(
                            market=market,
                            side=Side.NONE,
                            edge=0.0,
                            true_prob=0.5,
                            market_prob=0.5,
                            recommended_action=OrderAction.SKIP,
                            recommended_price=0.0,
                            size_usd=0.0,
                            size_shares=0.0,
                            chainlink_price=current_price,
                            time_remaining=time_remaining,
                            reasoning=f"[MARKET PAUSE] Waiting for resume: need 3+ candles same dir, RSI decisive. Current: {chart_analysis.consecutive_candles_same_dir} candles, RSI={chart_analysis.rsi_14:.0f}",
                        )

                    # === MULTI-TIMEFRAME ALIGNMENT CHECK ===
                    if not chart_analysis.timeframes_aligned and chart_analysis.alignment_score < 0.5:
                        logger.info(
                            f"⚠️ TIMEFRAME CONFLICT [{market.asset}]: "
                            f"15m={chart_analysis.trend_15m:+.2f}, "
                            f"1h={chart_analysis.trend_1h:+.2f}, "
                            f"4h={chart_analysis.trend_4h:+.2f} | "
                            f"Alignment: {chart_analysis.alignment_score:.0%} | "
                            f"Using reduced position"
                        )

                    # Store position multiplier for later use
                    self._position_multiplier = position_multiplier
                    self._position_multiplier_reason = position_reason

            except Exception as e:
                logger.debug(f"Chart analysis failed for {market.asset}: {e}")

        # === CHART-BASED TRADING DECISION ===
        # The chart analysis determines our trading direction
        # We only trade when chart gives a clear signal
        try:
            if chart_analysis:
                trend_1h = chart_analysis.trend_1h
                trend_4h = chart_analysis.trend_4h
                trend_15m = chart_analysis.trend_15m
                avg_trend = (trend_15m + trend_1h + trend_4h) / 3
                chart_bias = chart_analysis.bias
                chart_confidence = chart_analysis.confidence
                market_type = chart_analysis.market_type.value
                rsi = chart_analysis.rsi_14
                trend_change = chart_analysis.trend_change.value
                is_bullish_pattern = chart_analysis.is_bullish_pattern
                is_bearish_pattern = chart_analysis.is_bearish_pattern
            else:
                # Fallback to simple trends
                trends = get_multi_timeframe_trends(market.asset)
                trend_1h = trends.get("trend_1h", 0.0)
                trend_4h = trends.get("trend_4h", 0.0)
                trend_15m = 0.0
                avg_trend = (trend_1h + trend_4h) / 2
                chart_bias = "neutral"
                chart_confidence = 0.5
                market_type = "unknown"
                rsi = 50.0
                is_bullish_pattern = False
                is_bearish_pattern = False
                trend_change = "no_change"

            # Determine price position relative to target
            price_below_target = current_price < market.target_price
            price_above_target = current_price > market.target_price
            distance_from_target = abs(current_price - market.target_price) / market.target_price

            # === CHART-BASED DECISION LOGIC ===
            # Determine what the chart says we should do
            chart_says_up = False
            chart_says_down = False
            chart_signal_strength = 0.0  # 0 to 1

            # Strong bearish signals
            if chart_bias == "bearish" and chart_confidence >= 0.6:
                chart_says_down = True
                chart_signal_strength = chart_confidence

                # Extra strength for downtrends
                if "downtrend" in market_type:
                    chart_signal_strength = min(1.0, chart_signal_strength + 0.1)
                if "strong_downtrend" in market_type:
                    chart_signal_strength = min(1.0, chart_signal_strength + 0.2)

            # Strong bullish signals
            elif chart_bias == "bullish" and chart_confidence >= 0.6:
                chart_says_up = True
                chart_signal_strength = chart_confidence

                if "uptrend" in market_type:
                    chart_signal_strength = min(1.0, chart_signal_strength + 0.1)
                if "strong_uptrend" in market_type:
                    chart_signal_strength = min(1.0, chart_signal_strength + 0.2)

            # Trend-based signals (even if bias is neutral)
            if not chart_says_up and not chart_says_down:
                if avg_trend < -0.5:  # Strong downtrend
                    chart_says_down = True
                    chart_signal_strength = min(1.0, abs(avg_trend))
                elif avg_trend > 0.5:  # Strong uptrend
                    chart_says_up = True
                    chart_signal_strength = min(1.0, avg_trend)

            # === APPLY CHART DECISION TO EDGES ===
            # When chart gives a clear signal, it OVERRIDES probability-based edges

            if chart_says_down and chart_signal_strength >= 0.6:
                # Chart says DOWN - this should be our primary signal

                if price_above_target:
                    # IDEAL: Chart bearish + price above target = strong DOWN signal
                    # Price is likely to fall back below target
                    chart_edge_boost = chart_signal_strength * 0.15  # Up to 15% boost
                    edge_down += chart_edge_boost

                    # HEAVILY penalize UP - chart says price will fall
                    chart_edge_penalty = chart_signal_strength * 0.40  # Up to 40% penalty
                    edge_up -= chart_edge_penalty

                    logger.info(
                        f"🔴 CHART DECISION [{market.asset}]: BEARISH ({chart_signal_strength:.0%}) + "
                        f"above target → DOWN +{chart_edge_boost:.1%}, UP -{chart_edge_penalty:.1%}"
                    )

                elif price_below_target and distance_from_target < 0.01:  # Within 1% of target
                    # Price barely below target in downtrend - might still go DOWN
                    # But don't take UP position, price could fall further
                    chart_edge_penalty = chart_signal_strength * 0.25
                    edge_up -= chart_edge_penalty

                    logger.info(
                        f"🔴 CHART CAUTION [{market.asset}]: BEARISH ({chart_signal_strength:.0%}) + "
                        f"barely below target → UP -{chart_edge_penalty:.1%}"
                    )

            elif chart_says_up and chart_signal_strength >= 0.6:
                # Chart says UP - this should be our primary signal

                if price_below_target:
                    # IDEAL: Chart bullish + price below target = strong UP signal
                    chart_edge_boost = chart_signal_strength * 0.15
                    edge_up += chart_edge_boost

                    # HEAVILY penalize DOWN - chart says price will rise
                    chart_edge_penalty = chart_signal_strength * 0.40
                    edge_down -= chart_edge_penalty

                    logger.info(
                        f"🟢 CHART DECISION [{market.asset}]: BULLISH ({chart_signal_strength:.0%}) + "
                        f"below target → UP +{chart_edge_boost:.1%}, DOWN -{chart_edge_penalty:.1%}"
                    )

                elif price_above_target and distance_from_target < 0.01:
                    # Price barely above target in uptrend
                    chart_edge_penalty = chart_signal_strength * 0.25
                    edge_down -= chart_edge_penalty

                    logger.info(
                        f"🟢 CHART CAUTION [{market.asset}]: BULLISH ({chart_signal_strength:.0%}) + "
                        f"barely above target → DOWN -{chart_edge_penalty:.1%}"
                    )

            # === RSI EXTREMES ===
            # RSI overrides trend in extreme cases (mean reversion)
            if chart_analysis:
                if rsi < 20:  # Extremely oversold
                    # Expect bounce - reduce bearish conviction
                    if chart_says_down:
                        logger.info(f"⚠️ RSI WARNING [{market.asset}]: Extremely oversold ({rsi:.0f}) - bearish signal may reverse")
                        # Don't fully trust the bearish signal
                        edge_down -= 0.05

                elif rsi > 80:  # Extremely overbought
                    # Expect pullback - reduce bullish conviction
                    if chart_says_up:
                        logger.info(f"⚠️ RSI WARNING [{market.asset}]: Extremely overbought ({rsi:.0f}) - bullish signal may reverse")
                        edge_up -= 0.05

            # === PATTERN RECOGNITION BONUS ===
            if is_bullish_pattern and chart_says_up:
                edge_up += 0.03
                logger.info(f"🕯️ PATTERN CONFIRMS [{market.asset}]: Bullish pattern confirms UP signal +3%")
            elif is_bearish_pattern and chart_says_down:
                edge_down += 0.03
                logger.info(f"🕯️ PATTERN CONFIRMS [{market.asset}]: Bearish pattern confirms DOWN signal +3%")

            # === TREND REVERSAL SIGNALS ===
            if trend_change == "bullish_reversal":
                edge_up += 0.05
                edge_down -= 0.05
                logger.info(f"🔄 TREND REVERSAL [{market.asset}]: Bullish reversal detected → UP +5%, DOWN -5%")
            elif trend_change == "bearish_reversal":
                edge_down += 0.05
                edge_up -= 0.05
                logger.info(f"🔄 TREND REVERSAL [{market.asset}]: Bearish reversal detected → DOWN +5%, UP -5%")

            # === MULTI-TIMEFRAME EDGE ADJUSTMENT ===
            # When 1h and 15m trends align, boost edge
            # When they disagree, penalize the signal going against 1h
            if chart_analysis:
                higher_tf_bias = chart_analysis.higher_timeframe_bias
                alignment = chart_analysis.alignment_score

                if alignment >= 0.8:
                    # Strong alignment - boost the aligned direction
                    if higher_tf_bias == "bullish":
                        edge_up += 0.03
                        logger.info(f"🎯 TIMEFRAME ALIGNMENT [{market.asset}]: Strong bullish alignment ({alignment:.0%}) → UP +3%")
                    elif higher_tf_bias == "bearish":
                        edge_down += 0.03
                        logger.info(f"🎯 TIMEFRAME ALIGNMENT [{market.asset}]: Strong bearish alignment ({alignment:.0%}) → DOWN +3%")
                elif alignment < 0.5:
                    # Poor alignment - penalize trades against 1h trend
                    if higher_tf_bias == "bullish" and chart_says_down:
                        edge_down -= 0.05
                        logger.info(f"⚠️ TIMEFRAME CONFLICT [{market.asset}]: DOWN signal against 1h bullish → DOWN -5%")
                    elif higher_tf_bias == "bearish" and chart_says_up:
                        edge_up -= 0.05
                        logger.info(f"⚠️ TIMEFRAME CONFLICT [{market.asset}]: UP signal against 1h bearish → UP -5%")

            # Log final decision
            logger.info(
                f"📈 FINAL EDGES [{market.asset}]: UP={edge_up:.1%}, DOWN={edge_down:.1%} | "
                f"Chart: {'DOWN' if chart_says_down else 'UP' if chart_says_up else 'NEUTRAL'} ({chart_signal_strength:.0%})"
            )

        except Exception as e:
            logger.debug(f"Chart-based analysis failed for {market.asset}: {e}")

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

        # === TREND PROTECTION CHECK ===
        # Evaluate trade against trend protection rules
        if self.config.trend_protection.enabled:
            trend_result = self._evaluate_trend_protection(
                signal_side=side,
                asset=market.asset,
                current_price=current_price,
                target_price=market.target_price,
                edge=edge,
                binance_conf=binance_conf,
            )

            if not trend_result.should_trade:
                logger.info(
                    f"🛡️ TREND PROTECTION BLOCKED [{market.asset}]: "
                    f"{side.value} signal blocked | {trend_result.reason}"
                )
                return Signal(
                    market=market,
                    side=Side.NONE,
                    edge=edge,
                    true_prob=true_prob,
                    market_prob=market_prob,
                    recommended_action=OrderAction.SKIP,
                    recommended_price=0.0,
                    size_usd=0.0,
                    size_shares=0.0,
                    chainlink_price=current_price,
                    time_remaining=time_remaining,
                    reasoning=f"[TREND PROTECTION] {trend_result.reason}",
                )

            # Apply edge adjustments from trend protection
            if trend_result.edge_adjustment != 0:
                logger.debug(
                    f"TREND PROTECTION [{market.asset}]: Edge adjusted "
                    f"{edge:.1%} -> {trend_result.adjusted_edge:.1%} "
                    f"({trend_result.reason})"
                )
                edge = trend_result.adjusted_edge

        # === SMART TREND-FOLLOWING CHECK ===
        # Use technical analysis to boost edge (not block trades)
        try:
            smart_decision = get_smart_trading_decision(
                asset=market.asset,
                current_price=current_price,
                target_price=market.target_price,
                time_remaining=time_remaining,
                original_side=side.value,
                original_edge=edge,
            )

            # NOTE: Smart trend only provides BOOST, does NOT block trades
            # Blocking was too aggressive and prevented profitable opportunities
            if not smart_decision.should_trade:
                logger.debug(
                    f"📊 SMART TREND INFO [{market.asset}]: "
                    f"{side.value} - {smart_decision.reason} (not blocking)"
                )

            # Apply edge boost from smart analysis
            if smart_decision.edge_boost > 0:
                logger.info(
                    f"📈 SMART TREND BOOST [{market.asset}]: "
                    f"Edge {edge:.1%} -> {edge + smart_decision.edge_boost:.1%} | "
                    f"{smart_decision.reason}"
                )
                edge += smart_decision.edge_boost

        except Exception as e:
            logger.debug(f"Smart trend analysis failed for {market.asset}: {e}")
            # Continue with original signal if analysis fails

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

        Decision matrix (uses config min_time_remaining):
        - time < min_time_remaining: SKIP (handled by caller, shouldn't reach here)
        - time < 60s AND edge < 5%: SKIP (need some edge for short time)
        - time < 60s AND edge >= 5%: MARKET (urgent, take the opportunity)
        - time >= 60s: Use LIMIT orders for reliable execution
        - Otherwise -> SKIP
        """
        trading = self.config.trading

        # Short time (30-60s): Require moderate edge, use MARKET orders
        if time_remaining < 60:
            if edge >= 0.05:  # 5% edge minimum for short time trades
                logger.debug(f"Short time trade: {time_remaining:.0f}s, edge={edge:.1%} - using MARKET")
                return OrderAction.MARKET
            else:
                logger.debug(f"Skipping: {time_remaining:.0f}s remaining, edge {edge:.1%} < 5% required")
                return OrderAction.SKIP

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

        GRADUATED SIZING: Applies position multiplier from chart analysis
        to reduce position during uncertain/transitional markets.

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

        # === GRADUATED POSITION SIZING ===
        # Apply position multiplier from chart analysis
        # This reduces position size during uncertain/transitional markets
        position_multiplier = getattr(self, '_position_multiplier', 1.0)
        if position_multiplier < 1.0:
            original_size = size_usd
            size_usd *= position_multiplier
            logger.info(
                f"📏 GRADUATED SIZING: ${original_size:.2f} × {position_multiplier:.0%} = ${size_usd:.2f} | "
                f"Reason: {getattr(self, '_position_multiplier_reason', 'unknown')}"
            )

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
        # Prevent division by zero
        if target_price > 0:
            distance_pct = abs(current_price - target_price) / target_price * 100
        else:
            distance_pct = 0.0

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

    def record_trade_execution(self, asset: str, uncertainty_was_high: bool = False):
        """
        Record a trade execution for cooldown tracking.

        Call this after each successful trade to track cooldown progress.

        Args:
            asset: Asset symbol (BTC, ETH, etc.)
            uncertainty_was_high: True if market uncertainty was >= 60%
        """
        if CHART_ANALYSIS_AVAILABLE:
            record_trade_for_cooldown(asset, uncertainty_was_high)

    def _evaluate_trend_protection(
        self,
        signal_side: Side,
        asset: str,
        current_price: float,
        target_price: float,
        edge: float,
        binance_conf: dict,
    ):
        """
        Evaluate trade against trend protection rules.

        Args:
            signal_side: The side we want to trade (UP or DOWN)
            asset: Asset symbol (BTC, ETH, etc.)
            current_price: Current Chainlink price
            target_price: Market target price
            edge: Calculated edge
            binance_conf: Binance confirmation data

        Returns:
            TrendProtectionResult with decision and adjusted edge
        """
        from .trend_protection import TrendProtectionResult

        # Get price history for the asset
        symbol = f"{asset.lower()}/usd"
        price_history = self.price_histories.get(symbol, [])

        # Get multi-timeframe trends from Binance
        try:
            trends = get_multi_timeframe_trends(asset)
        except Exception as e:
            logger.debug(f"Failed to get trends for {asset}: {e}")
            trends = {"trend_1h": 0.0, "trend_4h": 0.0, "trend_1d": 0.0}

        # Get Binance velocity
        binance_velocity = 0.0
        if self.binance_feed is not None:
            try:
                binance_velocity = self.binance_feed.get_velocity(asset)
            except Exception:
                pass

        # Get Chainlink velocity
        chainlink_velocity = self.get_price_velocity(asset)

        # Get Binance price
        binance_price = binance_conf.get("binance_price")

        # Evaluate using trend protection
        result = self.trend_protection.evaluate_trade(
            signal_side=signal_side.value,
            asset=asset.upper(),
            current_price=current_price,
            target_price=target_price,
            base_edge=edge,
            base_min_edge=self.config.trading.min_edge,
            price_history=price_history,
            trends=trends,
            binance_price=binance_price,
            binance_velocity=binance_velocity,
            chainlink_velocity=chainlink_velocity,
        )

        return result
