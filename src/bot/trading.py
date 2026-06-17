"""Trading signal processing and order execution mixin."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from src.models import MarketState, Signal, OrderAction, Side
from src.strategy.trade_history import get_trade_history
from src.strategy.strategies.context import StrategyContext
from src.strategy.edge_signal import generate_edge_signal, EdgeSignalConfig, Conviction
from src.strategy.regime import RegimeState, RegimeType
from src.utils import Colors
from src.bot.guards import (
    validate_market_quotes,
    can_execute_trade,
    check_bankroll_regime,
    LogThrottle,
)

logger = logging.getLogger(__name__)


def _round_to_tick(price: float, tick: float) -> float:
    """Round a price to the market's tick size so the order isn't rejected.

    Polymarket rejects orders whose price doesn't conform to the market's
    ``orderPriceMinTickSize`` (e.g. 0.001). Rounds to the nearest tick and
    trims float noise to the tick's decimal precision.
    """
    if tick <= 0:
        return price
    steps = round(price / tick)
    # Decimal places implied by the tick (0.001 -> 3, 0.01 -> 2)
    decimals = max(0, len(f"{tick:.10f}".rstrip("0").split(".")[-1]))
    return round(steps * tick, decimals)


# Import Binance chart analyzer for ML features
try:
    from src.data.binance_chart import analyze_chart
    CHART_ANALYSIS_AVAILABLE = True
except ImportError:
    CHART_ANALYSIS_AVAILABLE = False
    def analyze_chart(asset: str):
        return None


class TradingMixin:
    """Mixin for market processing, signal generation, and order execution. Mixed into TradingBot."""

    def _record_block(self, reason: str):
        """Record a signal-processing block reason for tick diagnostics."""
        d = getattr(self, '_tick_block_reasons', None)
        if d is not None:
            d[reason] = d.get(reason, 0) + 1

    def _generate_edge_signal(self, market: MarketState, chainlink_price: Optional[float]) -> Optional[Signal]:
        """
        Generate a trading signal using the OFI + Regime + Distance pipeline.

        Returns a Signal compatible with the existing execution path, or None.
        """
        if not chainlink_price or not market.target_price:
            return None

        asset = market.asset

        # Query OFI from aggTrade feed
        ofi = 0.0
        ofi_accel = 0.0
        trade_rate = 0.0
        trade_feed = getattr(self, 'binance_trade_feed', None)
        if trade_feed is not None and trade_feed.is_connected:
            ofi_window = self.config.ofi.default_window if hasattr(self.config, 'ofi') else 30.0
            ofi = trade_feed.get_ofi(asset, ofi_window)
            ofi_accel = trade_feed.get_ofi_acceleration(asset)
            trade_rate = trade_feed.get_trade_rate(asset)

        # Query regime
        regime_detector = getattr(self, 'regime_detector', None)
        if regime_detector is not None:
            regime = regime_detector.get_regime(asset)
        else:
            # Default: assume ranging, allow trading
            regime = RegimeState(
                regime=RegimeType.RANGING,
                efficiency_ratio=0.5,
                atr_percentile=50.0,
                should_trade=True,
                kelly_multiplier=0.75,
            )

        best_bid_up = market.best_bid or 0.0
        best_ask_up = market.best_ask or 1.0

        # DOWN token prices: prefer the REAL DOWN orderbook (subscribed
        # alongside UP). The complement (1 - UP price) is only an
        # approximation — real DOWN asks can be cheaper, which is exactly
        # where edge and complete-set gaps live. Fall back to complement.
        best_bid_down = 1.0 - best_ask_up
        best_ask_down = 1.0 - best_bid_up
        down_book = self.clob_feed.get_orderbook(market.down_token_id) if self.clob_feed else None
        if down_book:
            if down_book.best_bid is not None and down_book.best_bid > 0:
                best_bid_down = down_book.best_bid
            if down_book.best_ask is not None and 0 < down_book.best_ask < 1:
                best_ask_down = down_book.best_ask

        # Total market duration
        total_duration = 900.0  # Default 15 min
        if market.variant == "five":
            total_duration = 300.0

        # Build edge signal config from bot config
        cfg = self.config.edge_signal
        es_config = EdgeSignalConfig(
            enabled=True,
            high_distance_threshold=cfg.high_distance_threshold,
            medium_distance_threshold=cfg.medium_distance_threshold,
            complete_set_min_gap=cfg.complete_set_min_gap,
            kelly_high=cfg.kelly_high,
            kelly_medium=cfg.kelly_medium,
            kelly_low=cfg.kelly_low,
            max_edge=cfg.max_edge,
            min_edge_ev=cfg.min_edge_ev,
            medium_edge_ev=cfg.medium_edge_ev,
            high_edge_ev=cfg.high_edge_ev,
            taker_fee_peak=cfg.taker_fee_peak,
            max_drift_adj=cfg.max_drift_adj,
            min_ofi_threshold=self.config.ofi.min_ofi_threshold if hasattr(self.config, 'ofi') else 0.15,
            strong_ofi_threshold=self.config.ofi.strong_ofi_threshold if hasattr(self.config, 'ofi') else 0.40,
            ofi_confirmation_threshold=self.config.ofi.confirmation_threshold if hasattr(self.config, 'ofi') else 0.10,
        )

        # Per-asset 15-minute volatility for the probability model
        vol_15m = self.signal_generator.get_volatility(asset)

        # Binance-lead drift: where Chainlink is about to print (0 if stale)
        drift = self.signal_generator.get_binance_drift(asset)

        edge_signal = generate_edge_signal(
            asset=asset,
            chainlink_price=chainlink_price,
            target_price=market.target_price,
            best_bid_up=best_bid_up,
            best_ask_up=best_ask_up,
            best_bid_down=best_bid_down,
            best_ask_down=best_ask_down,
            ofi=ofi,
            ofi_acceleration=ofi_accel,
            trade_rate=trade_rate,
            regime=regime,
            time_remaining=market.time_remaining,
            total_duration=total_duration,
            config=es_config,
            vol_15m=vol_15m,
            drift=drift,
        )

        if not edge_signal.should_trade:
            logger.debug(
                "EDGE_SIGNAL asset=%s result=SKIP reason=%s",
                asset, edge_signal.reason,
            )
            return None

        # Convert EdgeSignal → Signal (existing model)
        direction = edge_signal.direction
        if direction == Side.UP:
            bid, ask = best_bid_up, best_ask_up
            market_prob = best_bid_up  # Market implied probability
        else:
            bid, ask = best_bid_down, best_ask_down
            market_prob = best_bid_down

        # Order routing: resting maker bids suffer adverse selection in these
        # short markets — they fill fully when the signal is wrong and barely
        # at all when it's right. Cross the spread (taker) when the signal is
        # strong or time is short; only rest at the bid for weaker signals.
        spread = (ask - bid) if (0 < bid < ask < 1) else 1.0
        timing = self.config.trading.get_timing(market.variant)
        take_liquidity = (
            0 < ask < 0.95
            and spread <= self.config.execution.max_spread
            and (
                edge_signal.conviction == Conviction.HIGH
                or edge_signal.edge >= self.config.execution.taker_edge_threshold
                or market.time_remaining < timing["time_for_market"]
            )
        )
        if take_liquidity:
            price = ask
            action = OrderAction.LIMIT  # crosses the book, fills immediately
        else:
            price = bid
            action = OrderAction.POST_ONLY

        if price <= 0 or price >= 1:
            price = 0.50
            action = OrderAction.POST_ONLY

        # Conform the limit price to the market's tick size (Gamma orderPriceMinTickSize).
        price = _round_to_tick(price, market.tick_size)

        # Don't chase nearly-resolved markets: paying >=0.92 risks ~12x the
        # remaining payoff, and a resting bid up there only fills on an
        # adverse flash reversal. Kelly would mostly reject these anyway —
        # skip early to avoid noise and accidental fills.
        if price >= 0.92:
            logger.debug(
                "EDGE_SIGNAL asset=%s result=SKIP reason=price_too_high price=%.3f dir=%s",
                asset, price, direction.value,
            )
            return None

        # Size from Kelly fraction * bankroll
        bankroll = self.risk_manager.current_bankroll
        size_usd = bankroll * edge_signal.size_fraction
        # Floor at the market's real minimum order size (shares), cap at 20% of bankroll
        min_size_usd = max(1.0, market.min_order_size * price) if price > 0 else 5.0
        size_usd = max(min_size_usd, min(size_usd, bankroll * 0.20))
        size_shares = size_usd / price if price > 0 else 0.0

        # Throttle EDGE_SIGNAL logs to once per 5s per asset to prevent spam
        _es_log_ts = getattr(self, '_edge_signal_log_ts', {})
        if not hasattr(self, '_edge_signal_log_ts'):
            self._edge_signal_log_ts = {}
            _es_log_ts = self._edge_signal_log_ts
        _now_ts = time.time()
        _last_es_log = _es_log_ts.get(asset, 0.0)
        _es_log_fn = logger.info if _now_ts - _last_es_log >= 5.0 else logger.debug
        if _es_log_fn == logger.info:
            self._edge_signal_log_ts[asset] = _now_ts
        _es_log_fn(
            "EDGE_SIGNAL asset=%s conviction=%s dir=%s edge=%.4f win_prob=%.2f "
            "size=$%.2f ofi=%.2f regime=%s route=%s price=%.3f reason=%s",
            asset, edge_signal.conviction.value, direction.value,
            edge_signal.edge, edge_signal.win_probability,
            size_usd, ofi, regime.regime.value,
            "taker" if take_liquidity else "maker", price,
            edge_signal.reason[:60],
        )

        return Signal(
            market=market,
            side=direction,
            edge=edge_signal.edge,
            true_prob=edge_signal.win_probability,
            market_prob=market_prob,
            recommended_action=action,
            recommended_price=price,
            size_usd=size_usd,
            size_shares=size_shares,
            chainlink_price=chainlink_price,
            time_remaining=market.time_remaining,
            reasoning=edge_signal.reason,
        )

    async def _process_market(self, market: MarketState):
        """
        Process a single market for trading opportunities.

        Args:
            market: Market to process
        """
        # Skip if market is about to expire (should be in expiring_markets)
        min_time = (
            self.config.trading.min_time_remaining_5m
            if market.variant == "five"
            else self.config.trading.min_time_remaining
        )
        if market.time_remaining < min_time:
            self._record_block("expiring")
            return

        # === SHORT-CIRCUIT: Skip if we already have position for this (asset, variant) ===
        # This avoids generating signals and all downstream processing
        for pos in self.risk_manager.positions.values():
            if (hasattr(pos, 'market')
                    and pos.market.asset == market.asset
                    and pos.market.variant == market.variant):
                # Already have position - skip all processing (no log, very common)
                self._record_block("has_position")
                return

        # === SHORT-CIRCUIT: Skip if max positions reached ===
        if len(self.risk_manager.positions) >= self.config.trading.max_concurrent_positions:
            # Max positions reached - skip (logged via throttled block later if signal generated)
            self._record_block("max_positions")
            return

        # === SHORT-CIRCUIT: Skip if this variant's slots are full ===
        # Reserves capacity per market duration so long-lived 15m positions
        # can't crowd out 5m trading.
        variant_cap = self.config.trading.variant_position_cap()
        variant_count = sum(
            1 for pos in self.risk_manager.positions.values()
            if hasattr(pos, 'market')
            and getattr(pos.market, 'variant', 'fifteen') == market.variant
        )
        if variant_count >= variant_cap:
            self._record_block("variant_cap")
            return

        # Track when we first saw this market
        market_id = market.condition_id
        now = datetime.now(timezone.utc)
        if market_id not in self._market_first_seen:
            self._market_first_seen[market_id] = now
            logger.debug(f"[{market.asset}] First observation - collecting data...")

        # Check minimum observation time for this market (per-variant)
        observation_time = (now - self._market_first_seen[market_id]).total_seconds()
        min_observation = (
            self.config.trading.min_observation_time_5m
            if market.variant == "five"
            else self.config.trading.min_observation_time
        )
        if observation_time < min_observation:
            logger.debug(
                f"[{market.asset}] Observing: {observation_time:.0f}s / {min_observation:.0f}s"
            )
            # Still update orderbook data, just don't trade yet
            self._update_market_from_orderbook(market)
            self._record_block("observing")
            return

        # Check minimum price samples for this asset (Chainlink)
        asset_symbol = f"{market.asset.lower()}/usd"
        price_history = self.signal_generator.price_histories.get(asset_symbol, [])
        min_samples = self.config.trading.min_price_samples
        if len(price_history) < min_samples:
            logger.debug(
                f"[{market.asset}] Need more Chainlink data: {len(price_history)}/{min_samples} samples"
            )
            self._update_market_from_orderbook(market)
            self._record_block("low_samples")
            return

        # Check Binance price availability (used for confirmation signal - optional)
        binance_price = self.signal_generator.get_binance_price(market.asset)
        chainlink_price = self.signal_generator.get_price(market.asset)

        # CRITICAL: Validate target price before trading
        # After period transitions, the API may return stale target prices
        if chainlink_price and market.target_price:
            target_deviation = abs(chainlink_price - market.target_price) / market.target_price
            # For a 15-min market, target price should be within ~5% of current price
            # (crypto can move, but >5% in 15 mins is very unusual and suggests stale data)
            max_deviation = 0.05  # 5% max deviation
            if target_deviation > max_deviation:
                stale_key = f"{market.condition_id}:stale_target"
                if stale_key not in self._logged_rejections:
                    self._logged_rejections.add(stale_key)
                    logger.warning(
                        f"\u26a0\ufe0f [{market.asset}] STALE TARGET PRICE DETECTED! "
                        f"Target ${market.target_price:,.2f} deviates {target_deviation:.1%} "
                        f"from Chainlink ${chainlink_price:,.2f} (max {max_deviation:.0%}). "
                        f"Skipping trades until target price updates."
                    )
                self._record_block("stale_target")
                return

        # Binance is optional - used for confirmation signal boost, not required
        # The signal generator handles missing Binance data gracefully

        # Log price comparison when both prices are available
        if chainlink_price and binance_price:
            lead = binance_price - chainlink_price
            lead_pct = (lead / chainlink_price) * 100 if chainlink_price else 0
            logger.debug(
                f"[{market.asset}] Price check: Chainlink=${chainlink_price:,.2f}, "
                f"Binance=${binance_price:,.2f}, Lead={lead_pct:+.3f}%"
            )

        # Update market state from order book
        self._update_market_from_orderbook(market)

        # Track that we processed this market past pre-filters
        self._tick_markets_processed = getattr(self, '_tick_markets_processed', 0) + 1

        # === DATA SANITY GATE ===
        # Hard filter before ANY EV/ML computation — reject impossible quotes
        up_book = self.clob_feed.get_orderbook(market.up_token_id)
        quote_ts = up_book.timestamp if up_book else None
        qv = validate_market_quotes(
            yes_bid=market.best_bid if market.best_bid > 0 else None,
            yes_ask=market.best_ask if market.best_ask < 1.0 else None,
            max_spread=self.config.trading.max_spread,
            prob_tolerance=self.config.trading.prob_tolerance,
            quote_timestamp=quote_ts,
            max_quote_age_seconds=self.config.trading.quote_staleness_seconds,
        )
        if not qv.valid:
            # Throttle: once per 10s per market+reason
            if not hasattr(self, '_sanity_throttle'):
                self._sanity_throttle = LogThrottle(interval=10.0)
            throttle_key = f"{market.condition_id}:{qv.reason}"
            if self._sanity_throttle.should_log(throttle_key):
                # A side collapsing to empty near resolution is EXPECTED, not
                # a fault — log those quietly at DEBUG. Genuinely anomalous
                # states (crossed book, probability inconsistency) stay at WARN.
                _expected = {
                    "missing_prices", "price_out_of_range",
                    "derived_price_out_of_range", "spread_too_wide_yes",
                    "spread_too_wide_no",
                }
                _log = logger.debug if qv.reason in _expected else logger.warning
                _log(
                    "DATA_SANITY_FAIL asset=%s variant=%s reason=%s "
                    "yes_bid=%.4f yes_ask=%.4f no_bid=%.4f no_ask=%.4f "
                    "spread_yes=%.4f detail=%s",
                    market.asset, market.variant, qv.reason,
                    qv.yes_bid, qv.yes_ask, qv.no_bid, qv.no_ask,
                    qv.spread_yes, qv.detail,
                )
            self._record_block("data_sanity")
            return

        # === BANKROLL REGIME GUARD ===
        # If bankroll is too low to trade, switch to observe-only
        regime = check_bankroll_regime(
            bankroll=self.risk_manager.current_bankroll,
            min_trade_usd=self.config.risk_sizing.min_trade_usd,
            max_total_exposure_pct=getattr(self.config.trading, 'max_total_exposure_pct', 0.25),
            min_bankroll_multiplier=self.config.trading.min_bankroll_multiplier,
        )
        if regime.observe_only:
            if not hasattr(self, '_regime_throttle'):
                self._regime_throttle = LogThrottle(interval=60.0)
            throttle_key = f"{market.asset}:observe_only"
            if self._regime_throttle.should_log(throttle_key):
                logger.warning(
                    "OBSERVE_ONLY asset=%s %s",
                    market.asset, regime.reason,
                )
            self._record_block("observe_only")
            return

        # Generate trading signal
        signal = None

        # New edge signal pipeline (OFI + Regime + Distance)
        if getattr(self.config, 'edge_signal', None) and self.config.edge_signal.enabled:
            signal = self._generate_edge_signal(market, chainlink_price)
        elif self.meta_strategy:
            price_history = self.signal_generator.price_histories.get(market.asset, [])
            mid_price = (market.best_bid + market.best_ask) / 2 if market.best_bid and market.best_ask else 0.0
            orderbook_imbalance = 0.0
            if market.bid_depth + market.ask_depth > 0:
                orderbook_imbalance = (market.bid_depth - market.ask_depth) / (market.bid_depth + market.ask_depth)

            price_velocity = 0.0
            if len(price_history) >= 2 and price_history[-2] > 0:
                price_velocity = (price_history[-1] - price_history[-2]) / price_history[-2]

            volatility = self.signal_generator.get_volatility(market.asset)
            ctx = StrategyContext(
                asset=market.asset,
                mid_price=mid_price,
                reference_price=chainlink_price,
                orderbook_imbalance=orderbook_imbalance,
                price_velocity=price_velocity,
                volatility=volatility,
                time_remaining=market.time_remaining,
            )
            meta_signal = self.meta_strategy.generate(ctx)
            if not meta_signal:
                logger.debug("ORDER_BLOCK asset=%s reason=meta_strategy_no_signal", market.asset)
                self._record_block("no_signal")
                return
            self.metrics.record_strategy_performance("meta", meta_signal.edge, meta_signal.confidence)
            signal = self._signal_from_meta(market, meta_signal, chainlink_price or 0.0)
        else:
            signal = self.signal_generator.generate_signal(market)
        if not signal:
            logger.debug("ORDER_BLOCK asset=%s reason=no_signal_generated", market.asset)
            self._record_block("no_signal")
            return

        # Track that a signal was generated (past all pre-signal gates)
        self._tick_signals_generated = getattr(self, '_tick_signals_generated', 0) + 1

        # Dynamic risk sizing
        if self.config.risk_sizing.enabled:
            volatility = self.signal_generator.get_volatility(market.asset)
            market_price = max(0.01, signal.recommended_price)
            # Debug log to trace edge/prob/price values entering sizing
            logger.debug(
                "SIZING_INPUT asset=%s side=%s edge=%.4f true_prob=%.4f market_price=%.4f "
                "recommended_price=%.4f best_bid=%.4f best_ask=%.4f vol=%.4f bankroll=%.2f",
                market.asset, signal.side.value, signal.edge, signal.true_prob, market_price,
                signal.recommended_price, market.best_bid, market.best_ask, volatility,
                self.risk_manager.current_bankroll
            )
            desired_size = self.risk_sizer.size_position(
                bankroll=self.risk_manager.current_bankroll,
                edge=signal.edge,
                prob=signal.true_prob,
                market_price=market_price,
                volatility=volatility,
                base_size=signal.size_usd,
            )
            if desired_size <= 0:
                # Throttle this log to once per 30s per asset to avoid spam
                block_key = f"{market.asset}:risk_sizing_zero"
                now_ts = time.time()
                last_log = getattr(self, '_risk_sizing_log_ts', {}).get(block_key, 0.0)
                if now_ts - last_log >= 30.0:
                    if not hasattr(self, '_risk_sizing_log_ts'):
                        self._risk_sizing_log_ts = {}
                    self._risk_sizing_log_ts[block_key] = now_ts
                    logger.info(
                        "ORDER_BLOCK asset=%s side=%s reason=risk_sizing_zero edge=%.4f vol=%.4f "
                        "bankroll=%.2f prob=%.4f price=%.4f",
                        market.asset, signal.side.value, signal.edge, volatility,
                        self.risk_manager.current_bankroll, signal.true_prob, market_price
                    )
                self.dashboard.record_veto("risk_sizing_zero")
                self._record_block("sizing_zero")
                return
            signal.size_usd = desired_size
            signal.size_shares = desired_size / market_price
            logger.debug(
                "SIGNAL_MUTATED stage=risk_sizing asset=%s size_usd=%.2f size_shares=%.2f",
                market.asset, signal.size_usd, signal.size_shares,
            )

        # Log current Polymarket best prices and computed edge (throttled per asset)
        now_ts = time.time()
        last_price_log = self._price_log_ts.get(market.asset, 0.0)
        if now_ts - last_price_log >= 5.0:
            yes_bid = market.best_bid or 0.0
            yes_ask = market.best_ask or 0.0
            if yes_bid > 0 and yes_ask > 0:
                no_bid = 1.0 - yes_ask
                no_ask = 1.0 - yes_bid
                spread_yes = yes_ask - yes_bid
                edge_val = float(getattr(signal, "edge", 0.0) or 0.0)
                logger.debug(
                    "PMKT_PRICES asset=%s yes_bid=%.4f yes_ask=%.4f no_bid=%.4f no_ask=%.4f "
                    "spread_yes=%.4f edge=%.4f",
                    market.asset,
                    yes_bid,
                    yes_ask,
                    no_bid,
                    no_ask,
                    spread_yes,
                    edge_val,
                )
                self._price_log_ts[market.asset] = now_ts

        # Skip signals without action
        if signal.recommended_action == OrderAction.SKIP:
            logger.debug(
                "ORDER_DECISION asset=%s variant=%s decision=SKIP edge=%.4f min_edge=%.4f "
                "time_remaining=%.0f reason=%s",
                market.asset, market.variant,
                float(getattr(signal, "edge", 0.0) or 0.0),
                float(self.config.trading.min_edge),
                float(getattr(signal, "time_remaining", 0.0) or 0.0),
                signal.reasoning[:60] if signal.reasoning else "none",
            )
            self._record_block("low_edge")
            return

        # NOTE: ORDER_DECISION=PLACE_ORDER is logged AFTER all gating checks pass,
        # right before _execute_signal() is called. See line ~4000.

        # Validate signal against risk limits (no verbose logging - only log final decision)
        is_valid, reason = self.risk_manager.validate_signal(signal)
        if not is_valid:
            # Throttle common blocks (max_positions, already_have_position)
            block_reason = f"risk_manager:{reason.replace(' ', '_')[:30]}"
            self._log_order_block(
                market.asset, block_reason,
                market=market.condition_id[:8], side=signal.side.value
            )
            self._record_block("risk_limit")
            return

        # Trade history filter - check past performance for this asset/side
        trade_history = get_trade_history()
        should_proceed, history_reason = trade_history.evaluate_trade(
            asset=market.asset,
            side=signal.side.value,
            min_trades=self.config.trading.history_min_trades,
            min_win_rate=self.config.trading.history_min_win_rate,
            check_prediction_accuracy=True,
            min_prediction_accuracy=self.config.trading.history_min_prediction_accuracy,
        )
        if not should_proceed:
            block_reason = f"trade_history:{history_reason.replace(' ', '_')[:30]}"
            self._log_order_block(
                market.asset, block_reason,
                market=market.condition_id[:8], side=signal.side.value
            )
            self._record_block("trade_history")
            return

        # === CHART-BASED FILTER ===
        # Skip when edge signal pipeline is active — regime detector replaces chart filter
        # Block trades that go against strong chart signals
        chart_analysis = None
        _edge_signal_active = getattr(self.config, 'edge_signal', None) and self.config.edge_signal.enabled
        if CHART_ANALYSIS_AVAILABLE and not _edge_signal_active:
            try:
                chart_analysis = analyze_chart(market.asset)
                if chart_analysis:
                    chart_bias = chart_analysis.bias
                    chart_confidence = chart_analysis.confidence
                    market_type = chart_analysis.market_type.value
                    avg_trend = (chart_analysis.trend_15m + chart_analysis.trend_1h + chart_analysis.trend_4h) / 3

                    # Determine chart direction
                    chart_says_down = (chart_bias == "bearish" and chart_confidence >= 0.6) or avg_trend < -0.5
                    chart_says_up = (chart_bias == "bullish" and chart_confidence >= 0.6) or avg_trend > 0.5

                    # Calculate signal strength
                    chart_signal_strength = chart_confidence if chart_bias != "neutral" else abs(avg_trend)
                    if "strong_downtrend" in market_type or "strong_uptrend" in market_type:
                        chart_signal_strength = min(1.0, chart_signal_strength + 0.2)

                    # Block trades that go AGAINST strong chart signals
                    # EXCEPTION: Allow REVERSAL trades (these intentionally go against the trend!)
                    # Can be disabled with CHART_FILTER_ENABLED=false for testing
                    is_reversal_trade = "REVERSAL" in signal.reasoning.upper()
                    chart_filter_enabled = getattr(self.config.trading, 'chart_filter_enabled', True)

                    if chart_filter_enabled and chart_signal_strength >= 0.7 and not is_reversal_trade:
                        # Throttle chart filter logs to once per 30s per asset+side
                        block_key = f"{market.asset}:{signal.side.value}:chart_filter"
                        now_ts = time.time()
                        if not hasattr(self, '_chart_block_log_ts'):
                            self._chart_block_log_ts = {}
                        last_log = self._chart_block_log_ts.get(block_key, 0.0)

                        if chart_says_down and signal.side == Side.UP:
                            if now_ts - last_log >= 30.0:
                                self._chart_block_log_ts[block_key] = now_ts
                                logger.info(
                                    "ORDER_BLOCK asset=%s market=%s side=%s reason=chart_bearish_vs_up "
                                    "strength=%.2f edge=%.4f",
                                    market.asset, market.condition_id[:8], signal.side.value,
                                    chart_signal_strength, signal.edge
                                )
                            self.dashboard.record_veto("chart_bearish_vs_up")
                            self._record_block("chart_filter")
                            return
                        elif chart_says_up and signal.side == Side.DOWN:
                            if now_ts - last_log >= 30.0:
                                self._chart_block_log_ts[block_key] = now_ts
                                logger.info(
                                    "ORDER_BLOCK asset=%s market=%s side=%s reason=chart_bullish_vs_down "
                                    "strength=%.2f edge=%.4f",
                                    market.asset, market.condition_id[:8], signal.side.value,
                                    chart_signal_strength, signal.edge
                                )
                            self.dashboard.record_veto("chart_bullish_vs_down")
                            self._record_block("chart_filter")
                            return

            except Exception as e:
                logger.debug(f"Chart analysis failed: {e}")

        # Adjust size to fit within risk limits (caps only, never increases)
        signal = self.risk_manager.adjust_signal_size(signal)

        # Check if signal was rejected due to size
        if signal.size_usd <= 0 or signal.size_shares <= 0:
            # Throttle to once per 30s per asset
            block_key = f"{market.asset}:size_too_small"
            now_ts = time.time()
            last_log = getattr(self, '_size_block_log_ts', {}).get(block_key, 0.0)
            if now_ts - last_log >= 30.0:
                if not hasattr(self, '_size_block_log_ts'):
                    self._size_block_log_ts = {}
                self._size_block_log_ts[block_key] = now_ts
                logger.info(
                    "ORDER_BLOCK asset=%s market=%s side=%s reason=size_too_small usd=%.2f shares=%.4f",
                    market.asset, market.condition_id[:8], signal.side.value,
                    signal.size_usd, signal.size_shares
                )
            self._record_block("size_small")
            return

        # === CONSOLIDATED SIGNAL LOG (single line, for signals passing initial checks) ===
        # NOTE: ORDER_DECISION is logged in _execute_signal after ALL blocking checks pass
        # Throttle to once per 30s per market to avoid spam
        signal_msg = (
            f"\U0001f3af {market.asset} {signal.side.value} ${signal.size_usd:.0f} | "
            f"edge={signal.edge:.0%} | "
            f"{signal.reasoning[:40]}"
        )
        now_ts = time.time()
        last_signal_log = self._signal_log_ts.get(market.asset, 0.0)
        if now_ts - last_signal_log >= 30.0:
            self._signal_log_ts[market.asset] = now_ts
            logger.info(signal_msg)
        else:
            logger.debug(signal_msg)

        # === PRE-FLIGHT EXECUTION FEASIBILITY ===
        # Check if the trade CAN actually execute before entering _execute_signal
        av_key = f"{market.asset}:{market.variant}"
        cooldown_active = False
        if av_key in self._last_order_time:
            elapsed = (datetime.now(timezone.utc) - self._last_order_time[av_key]).total_seconds()
            cooldown_active = elapsed < self._order_cooldown_seconds

        # Compute capital scaling multiplier cheaply
        recent_pnls = list(self.risk_manager.trade_history)
        equity = self._calculate_equity()
        scale_mult, _ = self.capital_scaler.compute_multiplier(
            peak_bankroll=self.risk_manager.peak_bankroll,
            current_equity=equity,
            recent_pnls=recent_pnls,
            max_position_pct=self.config.trading.max_position_pct,
            bankroll=self.risk_manager.current_bankroll,
            current_size_usd=signal.size_usd,
        )

        # Compute current exposure
        _exposure = sum(
            p.shares * p.entry_price
            for p in self.risk_manager.positions.values()
        )
        async with self._pending_orders_lock:
            _pending_count = len(self._pending_orders)

        feasibility = can_execute_trade(
            bankroll=self.risk_manager.current_bankroll,
            min_trade_usd=self.config.risk_sizing.min_trade_usd,
            max_position_pct=self.config.trading.max_position_pct,
            max_total_exposure_pct=getattr(self.config.trading, 'max_total_exposure_pct', 0.25),
            equity=equity,
            current_exposure=_exposure,
            capital_scale_mult=scale_mult,
            base_size_usd=signal.size_usd,
            cooldown_active=cooldown_active,
            positions_count=len(self.risk_manager.positions),
            pending_count=_pending_count,
            max_concurrent_positions=self.config.trading.max_concurrent_positions,
            capital_buffer=self.config.trading.capital_buffer,
        )
        if not feasibility.can_execute:
            if not hasattr(self, '_preflight_throttle'):
                self._preflight_throttle = LogThrottle(interval=30.0)
            throttle_key = f"{market.asset}:{feasibility.reason}"
            if self._preflight_throttle.should_log(throttle_key):
                # Structural blocks (untradeable_size) are debug — nothing actionable per-tick
                log_fn = logger.debug if feasibility.reason == "untradeable_size" else logger.info
                log_fn(
                    "PREFLIGHT_SKIP asset=%s variant=%s side=%s reason=%s detail=%s",
                    market.asset, market.variant, signal.side.value,
                    feasibility.reason, feasibility.detail,
                )
            self._record_block("preflight")
            return

        # Execute the signal
        await self._execute_signal(signal)

    def _update_market_from_orderbook(self, market: MarketState):
        """Update market state from CLOB order book."""
        # Get UP token order book
        up_book = self.clob_feed.get_orderbook(market.up_token_id)
        if up_book:
            # Propagate emptiness, don't keep stale prices. When a side
            # collapses (near-resolved market: bid empty, ask 0.01) the feed
            # returns None for that side. Keeping the previous value would make
            # the validator see a phantom crossed book (stale bid > fresh ask)
            # and spam DATA_SANITY_FAIL forever. Reset to a sentinel so the
            # sanity gate reports a quiet "missing_prices" instead.
            market.best_bid = up_book.best_bid if up_book.best_bid is not None else 0.0
            market.best_ask = up_book.best_ask if up_book.best_ask is not None else 1.0

            # Calculate depth - make thread-safe copies to avoid race conditions
            # with WebSocket thread that may be modifying the orderbook
            try:
                bids_snapshot = list(up_book.bids)
                asks_snapshot = list(up_book.asks)
                # Validate sizes are numeric before summing (handle malformed data)
                market.bid_depth = sum(
                    level.size for level in bids_snapshot
                    if hasattr(level, 'size') and isinstance(level.size, (int, float))
                )
                market.ask_depth = sum(
                    level.size for level in asks_snapshot
                    if hasattr(level, 'size') and isinstance(level.size, (int, float))
                )
            except Exception as e:
                logger.debug(f"[{market.asset}] Orderbook depth calc error: {e}")
                # Keep previous depth values on error

        market.last_updated = datetime.now(timezone.utc)

    def _signal_from_meta(self, market: MarketState, meta_signal, chainlink_price: float) -> Optional[Signal]:
        market_prob = market.implied_prob_up
        side = Side.UP if meta_signal.side == "YES" else Side.DOWN

        if side == Side.UP:
            true_prob = min(0.99, max(0.01, market_prob + meta_signal.edge))
        else:
            true_prob = min(0.99, max(0.01, market_prob - meta_signal.edge))

        edge = abs(true_prob - market_prob)
        time_remaining = market.time_remaining
        timing = self.config.trading.get_timing(market.variant)

        action = OrderAction.SKIP
        if edge >= self.config.trading.edge_for_market or time_remaining < timing["time_for_market"]:
            action = OrderAction.MARKET
        elif edge >= self.config.trading.edge_for_limit or time_remaining < timing["time_for_limit"]:
            action = OrderAction.LIMIT
        elif edge >= self.config.trading.edge_for_post_only:
            action = OrderAction.POST_ONLY

        if side == Side.UP:
            price = market.best_ask
        else:
            price = 1.0 - market.best_bid if market.best_bid else 0.5

        size_usd = self.config.trading.base_position_size
        volatility = self.signal_generator.get_volatility(market.asset)
        size_usd = self.risk_sizer.size_position(
            bankroll=self.risk_manager.current_bankroll,
            edge=edge,
            prob=true_prob if side == Side.UP else 1 - true_prob,
            market_price=price if price > 0 else 0.5,
            volatility=volatility,
            base_size=size_usd,
        )
        if size_usd <= 0:
            return None

        size_shares = size_usd / price if price > 0 else 0.0

        return Signal(
            market=market,
            side=side,
            edge=edge,
            true_prob=true_prob,
            market_prob=market_prob,
            recommended_action=action,
            recommended_price=price,
            size_usd=size_usd,
            size_shares=size_shares,
            chainlink_price=chainlink_price or 0.0,
            time_remaining=time_remaining,
            reasoning=f"META {meta_signal.side} edge={edge:.3f} conf={meta_signal.confidence:.2f}",
        )

    def _check_averaging_opportunity(self, position, signal, market):
        """
        Check if we should average down on an existing position.

        Args:
            position: Existing position
            signal: New signal that triggered this check
            market: Current market state

        Returns:
            AveragingDecision or None
        """
        try:
            from .strategy.averaging import should_average_down, AveragingConfig
            from .data.binance_chart import analyze_chart
        except ImportError:
            return None

        # Only consider averaging if signal matches position direction
        if signal.side != position.side:
            return None

        # Get current token price
        if position.side.value == "UP":
            current_token_price = market.best_ask
        else:
            current_token_price = 1 - market.best_bid

        # Get chart analysis
        chart_analysis = analyze_chart(market.asset)

        # Check averaging conditions with moderate config
        config = AveragingConfig(
            enabled=True,
            min_chart_confidence=0.70,  # 70% chart confidence
            min_price_improvement=0.08,  # 8% cheaper
            max_averages=1,  # Only average once
            min_time_remaining=300,  # 5 minutes
            rsi_min=25.0,  # Don't average if too oversold
            rsi_max=75.0,  # Don't average if too overbought
            min_alignment_score=0.66,  # 2/3 timeframes agree
            averaging_size_multiplier=0.5,  # Add 50% of original
        )

        return should_average_down(
            position=position,
            current_token_price=current_token_price,
            chart_analysis=chart_analysis,
            time_remaining=market.time_remaining,
            config=config,
        )

    async def _execute_averaging_order(self, position, signal, averaging_decision):
        """
        Execute an averaging down order.

        Args:
            position: Position to add to
            signal: Original signal (for order execution)
            averaging_decision: AveragingDecision with suggested size
        """
        try:
            from .strategy.averaging import update_position_after_averaging
        except ImportError:
            logger.error("Could not import averaging module")
            return

        market = signal.market
        asset = market.asset

        # Create a modified signal for the averaging order
        averaging_signal = Signal(
            market=market,
            side=position.side,
            edge=signal.edge,
            true_prob=signal.true_prob,
            market_prob=signal.market_prob,
            recommended_action=OrderAction.LIMIT,
            recommended_price=signal.recommended_price,
            size_usd=averaging_decision.suggested_size_usd,
            size_shares=averaging_decision.suggested_shares,
            chainlink_price=signal.chainlink_price,
            time_remaining=signal.time_remaining,
            reasoning=f"[AVERAGING] {averaging_decision.reason}",
        )

        # Execute the order
        try:
            result = await self.executor.execute_signal_async(averaging_signal)
        except Exception as e:
            logger.error(
                "ORDER_RESULT asset=%s variant=%s side=%s status=exception error=%s mode=averaging",
                asset, getattr(position.market, 'variant', 'fifteen'), position.side.value, str(e)[:200]
            )
            logger.debug("Averaging execution exception traceback:", exc_info=True)
            return  # Exit gracefully

        if result.success:
            # Update position with new average
            update_position_after_averaging(
                position=position,
                additional_shares=averaging_decision.suggested_shares,
                additional_cost=averaging_decision.suggested_size_usd,
                new_entry_price=averaging_decision.new_avg_price,
            )

            # Set cooldown
            now = datetime.now(timezone.utc)
            self._last_order_time[f"{asset}:{market.variant}"] = now

            logger.info(
                f"[{asset}/{market.variant}] AVERAGED DOWN: "
                f"Added {averaging_decision.suggested_shares:.1f} shares @ ${averaging_decision.suggested_size_usd:.2f} | "
                f"New avg price: ${averaging_decision.new_avg_price:.3f} | "
                f"Total shares: {position.shares:.1f}"
            )
        else:
            logger.warning(
                f"[{asset}] \u274c AVERAGING FAILED: {result.error_message}"
            )

    async def _execute_signal(self, signal: Signal):
        """Execute a trading signal."""
        # Guard: reject if shutdown has started (no in-flight orders after state save)
        if not self._running:
            return

        # Kill switch gate — blocks all new orders
        blocked, ks_reason = self.kill_switch.should_block_order()
        if blocked:
            self.dashboard.record_veto("kill_switch")
            self.attribution.record_veto(
                asset=signal.market.asset,
                market_id=signal.market.condition_id,
                side=signal.side.value,
                source="RULES",
                veto_reason=ks_reason,
            )
            return

        asset = signal.market.asset
        variant = signal.market.variant
        av_key = f"{asset}:{variant}"
        now = datetime.now(timezone.utc)

        # Check cooldown - prevent rapid duplicate orders per (asset, variant)
        if av_key in self._last_order_time:
            elapsed = (now - self._last_order_time[av_key]).total_seconds()
            if elapsed < self._order_cooldown_seconds:
                remaining = self._order_cooldown_seconds - elapsed
                # Log once per cooldown period at INFO level so user knows why trades aren't executing
                cooldown_key = f"{signal.market.condition_id}:cooldown"
                if cooldown_key not in self._logged_rejections:
                    self._logged_rejections.add(cooldown_key)
                    logger.info(
                        "ORDER_SKIP_DUP asset=%s variant=%s reason=cooldown secs_since_last=%.0f",
                        asset, variant,
                        elapsed,
                    )
                return

        # Check for existing FILLED positions for this market+side (prevents duplicate positions)
        market_id = signal.market.condition_id
        for pos_key, pos in self.risk_manager.positions.items():
            pos_market_id = getattr(pos.market, 'condition_id', None) if hasattr(pos, 'market') else None
            pos_side = pos.side.value if hasattr(pos.side, 'value') else str(pos.side)
            if pos_market_id == market_id and pos_side == signal.side.value:
                # Throttled - this can spam when position is open
                self._log_order_block(
                    asset, "position_exists",
                    market=market_id[:8], side=signal.side.value
                )
                return

        # Check for opposing positions in same asset, different variant
        for pos in self.risk_manager.positions.values():
            if (hasattr(pos, 'market')
                    and pos.market.asset == asset
                    and getattr(pos.market, 'variant', 'fifteen') != variant
                    and pos.side.value != signal.side.value):
                logger.warning(
                    "%sVARIANT_CONFLICT%s asset=%s new=%s/%s existing=%s/%s — opposing sides!",
                    Colors.BG_YELLOW + Colors.BRIGHT_RED,
                    Colors.RESET,
                    asset,
                    variant, signal.side.value,
                    getattr(pos.market, 'variant', '?'), pos.side.value,
                )
                break  # warn once, allow trade to proceed

        # Check for existing PENDING orders for this asset (prevents duplicate unfilled orders)
        async with self._pending_orders_lock:
            pending_items = list(self._pending_orders.items())
            pending_count = len(self._pending_orders)
        for order_id, order_data in pending_items:
            if order_data.get("asset") == asset:
                placed_time = order_data.get("placed_time")
                if placed_time:
                    in_window = signal.market.start_time <= placed_time <= signal.market.end_time
                    same_side = order_data.get("side") == signal.side.value
                    if in_window and same_side:
                        # Throttled
                        self._log_order_block(
                            asset, "pending_order_exists",
                            market=market_id[:8], side=signal.side.value
                        )
                        return
                # Already have any pending order for this asset
                self._log_order_block(
                    asset, "pending_order_any",
                    market=market_id[:8], order_id=order_id[:8]
                )
                return

        # Check total positions (filled + pending) against max concurrent limit
        filled_positions = len(self.risk_manager.positions)
        pending_orders = pending_count
        total_positions = filled_positions + pending_orders
        max_positions = self.config.trading.max_concurrent_positions

        if total_positions >= max_positions:
            self._log_order_block(
                asset, "position_limit",
                total=total_positions, filled=filled_positions,
                pending=pending_orders, max=max_positions,
            )
            return

        # Per-variant cap (filled + pending) — reserves slots per market
        # duration so 15m positions can't occupy all capacity
        variant_cap = self.config.trading.variant_position_cap()
        variant_filled = sum(
            1 for pos in self.risk_manager.positions.values()
            if hasattr(pos, 'market')
            and getattr(pos.market, 'variant', 'fifteen') == variant
        )
        variant_pending = sum(
            1 for _, od in pending_items
            if od.get("variant", "fifteen") == variant
        )
        if variant_filled + variant_pending >= variant_cap:
            self._log_order_block(
                asset, "variant_position_limit",
                variant=variant, filled=variant_filled,
                pending=variant_pending, cap=variant_cap,
            )
            return

        # Compute exposure breakdown (positions + pending orders) — reused for all exposure checks
        _per_asset_exposure: dict[str, float] = {}
        for pos in self.risk_manager.positions.values():
            _pa = pos.market.asset if hasattr(pos, "market") else asset
            _pv = pos.shares * pos.entry_price
            _per_asset_exposure[_pa] = _per_asset_exposure.get(_pa, 0.0) + _pv
        for _, order_data in pending_items:
            _oa = order_data.get("asset")
            if _oa:
                _ov = float(order_data.get("size", 0)) * float(order_data.get("price", 0))
                _per_asset_exposure[_oa] = _per_asset_exposure.get(_oa, 0.0) + _ov
        current_exposure = sum(_per_asset_exposure.values())

        # Check 1: Equity-based percentage cap
        new_order_value = signal.size_usd
        total_exposure = current_exposure + new_order_value
        equity = self._calculate_equity()
        max_exposure_pct = getattr(self.config.trading, 'max_total_exposure_pct', 0.25)
        max_exposure = equity * max_exposure_pct

        if total_exposure > max_exposure:
            self._log_order_block(
                asset, "exposure_limit",
                total_exposure=f"{total_exposure:.2f}",
                current=f"{current_exposure:.2f}",
                new=f"{new_order_value:.2f}",
                max_exposure=f"{max_exposure:.2f}",
            )
            return

        # CRITICAL: Check API positions FIRST (prevents duplicate trades after restart)
        if hasattr(self, '_api_positions') and asset in self._api_positions:
            api_pos = self._api_positions[asset]
            api_shares = api_pos.get('size', api_pos.get('shares', 0)) if isinstance(api_pos, dict) else getattr(api_pos, 'size', getattr(api_pos, 'shares', 0))
            if api_shares > 0:
                position_key = f"{asset}:api_position"
                if position_key not in self._logged_rejections:
                    self._logged_rejections.add(position_key)
                    logger.warning(
                        f"[{asset}] \u26d4 BLOCKING ORDER: Found {api_shares:.2f} shares in API! "
                        f"Position not in internal tracking - possible restart. Skipping to prevent duplicate."
                    )
                return

        # Check for existing positions from internal tracking (match variant too)
        market_variant = getattr(signal.market, "variant", "fifteen")
        existing_position = self._get_active_position(asset, variant=market_variant)
        if existing_position:
            # Check if we should average down instead of skipping
            averaging_decision = self._check_averaging_opportunity(
                position=existing_position,
                signal=signal,
                market=signal.market,
            )

            if averaging_decision and averaging_decision.should_average:
                # Execute averaging order
                logger.info(
                    f"[{asset}] \U0001f4c8 AVERAGING DOWN: {averaging_decision.reason} | "
                    f"Adding ${averaging_decision.suggested_size_usd:.2f} ({averaging_decision.suggested_shares:.1f} shares)"
                )
                await self._execute_averaging_order(existing_position, signal, averaging_decision)
                return

            # Not averaging - skip as usual
            position_key = f"{signal.market.condition_id}:active_position"
            if position_key not in self._logged_rejections:
                self._logged_rejections.add(position_key)
                reason = averaging_decision.reason if averaging_decision else "already has position"
                logger.info(f"[{asset}] \u274c ACTIVE POSITION: {reason} - skipping signal")
            return

        # Check if we have enough balance before attempting
        available = self.risk_manager.current_bankroll * 0.90  # 10% buffer
        if signal.size_usd > available:
            # Log at INFO level - insufficient balance is critical info
            balance_key = f"{signal.market.condition_id}:balance"
            if balance_key not in self._logged_rejections:
                self._logged_rejections.add(balance_key)
                logger.info(
                    f"[{asset}] \u274c INSUFFICIENT BALANCE: Need ${signal.size_usd:.2f}, "
                    f"have ${available:.2f} (bankroll ${self.risk_manager.current_bankroll:.2f})"
                )
            # Set cooldown to prevent spam (never move timestamps into the future)
            self._last_order_time[av_key] = now
            return

        # Get current price for logging
        current_price = self.signal_generator.get_price(signal.market.asset)
        target = signal.market.target_price

        # FINAL SAFETY CHECK: Validate target price at execution time
        # This catches any stale target prices that slipped through earlier checks
        if current_price and target:
            deviation = abs(current_price - target) / target
            # Calculate how long into the current period we are
            current_ts = int(now.timestamp())
            period_start = (current_ts // 900) * 900
            seconds_into_period = current_ts - period_start

            # In the first 60 seconds of a period, be extra strict about target price
            # The target should be VERY close to current price at period start
            if seconds_into_period < 60:
                max_deviation = 0.02  # Only 2% deviation allowed in first 60 seconds
                if deviation > max_deviation:
                    logger.warning(
                        f"\u26a0\ufe0f [{asset}] BLOCKING TRADE - Target price ${target:,.2f} "
                        f"deviates {deviation:.1%} from Chainlink ${current_price:,.2f} "
                        f"({seconds_into_period}s into period). Likely stale price from previous period."
                    )
                    return
            else:
                # After 60 seconds, use 5% threshold
                max_deviation = 0.05
                if deviation > max_deviation:
                    logger.warning(
                        f"\u26a0\ufe0f [{asset}] BLOCKING TRADE - Target price ${target:,.2f} "
                        f"deviates {deviation:.1%} from Chainlink ${current_price:,.2f}. "
                        f"Target price may be stale."
                    )
                    return

        # Log positions BEFORE trade (debug level - detailed analysis moved to after safety checks)
        api_pos_count = len(self._api_positions) if hasattr(self, '_api_positions') else 0
        risk_pos_count = len(self.risk_manager.positions)
        logger.debug(
            f"PRE-TRADE CHECK [{asset}]: API positions={api_pos_count} | "
            f"Tracked positions={risk_pos_count} | Bankroll=${self.risk_manager.current_bankroll:.2f}"
        )

        # Hard safety brakes: dedup, cooldown, rate limit, exposure caps
        market_id = signal.market.condition_id
        for _, order_data in pending_items:
            if (
                order_data.get("market_id") == market_id
                and order_data.get("asset") == asset
                and order_data.get("side") == signal.side.value
            ):
                self._log_order_block(asset, "duplicate_open_order")
                return

        now_ts = time.time()
        key = (market_id, asset)
        last_ts = self._last_order_time_by_market.get(key)
        if last_ts is not None:
            elapsed = now_ts - last_ts
            if elapsed < self._order_cooldown_seconds:
                remaining = self._order_cooldown_seconds - elapsed
                self._log_order_block(asset, "cooldown", secs_remaining=f"{remaining:.0f}")
                return

        # Global order rate limit (rolling 60s)
        while self._order_submit_ts and now_ts - self._order_submit_ts[0] > 60.0:
            self._order_submit_ts.popleft()
        if len(self._order_submit_ts) >= self.config.trading.max_orders_per_min:
            self._log_order_block(asset, "global_rate_limit")
            return

        # Check 2+3: Absolute USD exposure caps (reuses _per_asset_exposure from above,
        # but signal.size_usd may have been reduced by capital_scaling since then)
        total_exposure_after = current_exposure + signal.size_usd
        per_asset_after = _per_asset_exposure.get(asset, 0.0) + signal.size_usd
        if total_exposure_after > self.config.trading.max_total_exposure_usd:
            self._log_order_block(asset, "exposure_cap_total")
            return
        if per_asset_after > self.config.trading.max_exposure_per_asset_usd:
            self._log_order_block(asset, "exposure_cap_per_asset")
            return

        self._last_order_time_by_market[key] = now_ts
        self._order_submit_ts.append(now_ts)

        # === ALL SAFETY CHECKS PASSED ===

        # Capital scaling — adjust size based on drawdown tier / recent EV
        recent_pnls = list(self.risk_manager.trade_history)
        scaled_size, scale_mult, scale_reason = self.capital_scaler.apply(
            size_usd=signal.size_usd,
            peak_bankroll=self.risk_manager.peak_bankroll,
            current_equity=self._calculate_equity(),
            recent_pnls=recent_pnls,
            max_position_pct=self.config.trading.max_position_pct,
            bankroll=self.risk_manager.current_bankroll,
        )
        pre_scale_size = signal.size_usd  # Save before capital scaling
        if abs(scale_mult - 1.0) > 0.001:
            signal.size_usd = scaled_size
            if signal.recommended_price > 0:
                signal.size_shares = signal.size_usd / signal.recommended_price
            logger.info(
                "CAPITAL_SCALING asset=%s multiplier=%.2f reason=%s size=%.2f->%.2f",
                asset, scale_mult, scale_reason, pre_scale_size, signal.size_usd,
            )

        # Floor: ensure size meets the market's real minimum order size (shares).
        # Capital scaling + position cap can both push below the exchange minimum.
        # The kill switch handles full halts; sizing should reduce, not block.
        min_shares = signal.market.min_order_size
        min_floor_usd = max(1.0, min_shares * signal.recommended_price) if signal.recommended_price > 0 else 5.0
        if signal.size_usd < min_floor_usd:
            signal.size_usd = min_floor_usd
            signal.size_shares = min_shares
            logger.debug(
                "ORDER_SIZE_FLOOR asset=%s floored=$%.2f (%.1f shares) pre_scale=%.2f mult=%.2f",
                asset, min_floor_usd, min_shares, pre_scale_size, scale_mult,
            )

        # ORDER_DECISION is logged here AFTER all blocking gates (cooldown/dup/rate-limit/exposure)
        logger.info(
            "ORDER_DECISION asset=%s variant=%s decision=PLACE_ORDER side=%s edge=%.4f min_edge=%.4f "
            "true_prob=%.4f price=%.4f size_usd=%.2f time_remaining=%.0f order_type=%s",
            asset, signal.market.variant,
            signal.side.value,
            float(signal.edge or 0.0),
            float(self.config.trading.min_edge),
            float(signal.true_prob or 0.0),
            float(signal.recommended_price or 0.0),
            float(signal.size_usd or 0.0),
            float(signal.time_remaining or 0.0),
            str(signal.recommended_action),
        )

        # Log detailed trade analysis
        arb_type = getattr(signal, '_arb_type', 'unknown')

        # Price analysis
        price_vs_target = "ABOVE" if current_price and target and current_price >= target else "BELOW"
        distance_pct = abs(current_price - target) / target * 100 if current_price and target else 0

        logger.info(f"{Colors.BRIGHT_CYAN}\u2554{'═' * 62}\u2557{Colors.RESET}")
        logger.info(f"{Colors.BRIGHT_CYAN}\u2551  \U0001f4ca TRADE ANALYSIS: {asset} {signal.side.value:4}{Colors.RESET}")
        logger.info(f"{Colors.BRIGHT_CYAN}\u2560{'═' * 62}\u2563{Colors.RESET}")

        # Why this trade?
        logger.info(f"{Colors.BRIGHT_CYAN}\u2551{Colors.RESET}  \U0001f3af Signal Type: {arb_type.upper()}")
        logger.info(f"{Colors.BRIGHT_CYAN}\u2551{Colors.RESET}  \U0001f4dd Reasoning: {signal.reasoning}")

        # Price info
        logger.info(f"{Colors.BRIGHT_CYAN}\u2551{Colors.RESET}  \U0001f4b0 Chainlink: ${current_price:,.2f} ({price_vs_target} target by {distance_pct:.2f}%)")
        logger.info(f"{Colors.BRIGHT_CYAN}\u2551{Colors.RESET}  \U0001f3af Target: ${target:,.2f}")

        # Market prices
        market = signal.market
        logger.info(f"{Colors.BRIGHT_CYAN}\u2551{Colors.RESET}  \U0001f4c8 UP price: bid={market.best_bid:.3f} / ask={market.best_ask:.3f}")
        logger.info(f"{Colors.BRIGHT_CYAN}\u2551{Colors.RESET}  \U0001f4c9 DOWN price: bid={1-market.best_ask:.3f} / ask={1-market.best_bid:.3f}")

        # Edge calculation
        logger.info(f"{Colors.BRIGHT_CYAN}\u2551{Colors.RESET}  \u2728 Edge: {signal.edge:.1%} (min required: {self.config.trading.min_edge:.1%})")

        # Time remaining
        logger.info(f"{Colors.BRIGHT_CYAN}\u2551{Colors.RESET}  \u23f1\ufe0f  Time left: {signal.time_remaining:.0f}s")

        # Position size
        logger.info(f"{Colors.BRIGHT_CYAN}\u2551{Colors.RESET}  \U0001f4b5 Size: ${signal.size_usd:.2f} ({signal.size_shares:.2f} shares @ {signal.recommended_price:.3f})")

        logger.info(f"{Colors.BRIGHT_CYAN}\u255a{'═' * 62}\u255d{Colors.RESET}")

        # Log the trade attempt with colors
        direction = "\u25b2" if signal.side == Side.UP else "\u25bc"
        side_color = Colors.BRIGHT_GREEN if signal.side == Side.UP else Colors.BRIGHT_RED
        mode = self._execution_mode.lower()

        # Log ORDER_SUBMIT BEFORE execution attempt
        logger.info(
            "ORDER_SUBMIT asset=%s variant=%s market=%s side=%s price=%.4f size_usd=%.2f size_shares=%.4f type=%s mode=%s",
            asset, signal.market.variant,
            market_id[:8],
            signal.side.value,
            float(signal.recommended_price or 0.0),
            float(signal.size_usd or 0.0),
            float(signal.size_shares or 0.0),
            str(signal.recommended_action),
            mode,
        )
        # Record to metrics file
        self.metrics.record_order_event(
            event_type="ORDER_SUBMIT",
            asset=asset,
            side=signal.side.value,
            price=float(signal.recommended_price or 0.0),
            size_usd=float(signal.size_usd or 0.0),
            mode=mode,
        )

        logger.info(
            f"{side_color}>>> EXECUTING: {asset} {signal.side.value} {direction}{Colors.RESET} \u2502 "
            f"Edge: {Colors.BRIGHT_YELLOW}{signal.edge:.0%}{Colors.RESET} \u2502 "
            f"${current_price:,.0f} vs ${target:,.0f} \u2502 "
            f"${signal.size_usd:.2f}"
        )

        # Execute through order executor
        try:
            result = await self.executor.execute_signal_async(signal)
        except Exception as e:
            # Log but DO NOT re-raise - keep market loop alive
            logger.error(
                "ORDER_RESULT asset=%s market=%s side=%s status=exception error=%s mode=%s",
                asset,
                market_id[:8],
                signal.side.value,
                str(e)[:200],
                mode,
            )
            logger.debug("Executor exception traceback:", exc_info=True)
            # Record failed order to metrics
            self.metrics.record_order_event(
                event_type="ORDER_RESULT",
                asset=asset,
                side=signal.side.value,
                reason=f"exception: {str(e)[:50]}",
                mode=mode,
            )
            return  # Exit gracefully, don't kill the market loop

        # Log ORDER_RESULT AFTER execution
        if result.success:
            logger.info(
                "ORDER_RESULT asset=%s variant=%s market=%s side=%s status=success order_id=%s "
                "filled_size=%.4f filled_price=%.4f mode=%s",
                asset, signal.market.variant,
                market_id[:8],
                signal.side.value,
                result.order_id[:16] if result.order_id else "none",
                result.filled_size,
                result.filled_price,
                mode,
            )
            # Record ORDER_FILL to metrics
            self.metrics.record_order_event(
                event_type="ORDER_FILL",
                asset=asset,
                side=signal.side.value,
                price=result.filled_price,
                size_usd=result.filled_size * result.filled_price,
                order_id=result.order_id,
                mode=mode,
            )
        else:
            logger.info(
                "ORDER_RESULT asset=%s variant=%s market=%s side=%s status=failed error=%s mode=%s",
                asset, signal.market.variant,
                market_id[:8],
                signal.side.value,
                result.error_message[:50] if result.error_message else "unknown",
                mode,
            )
            # Record failed order to metrics
            self.metrics.record_order_event(
                event_type="ORDER_RESULT",
                asset=asset,
                side=signal.side.value,
                price=float(signal.recommended_price or 0.0),
                size_usd=float(signal.size_usd or 0.0),
                reason=result.error_message[:50] if result.error_message else "unknown",
                mode=mode,
            )

        if result.success:
            # Track successful trade for diagnostics
            self._tick_trades_executed = getattr(self, '_tick_trades_executed', 0) + 1
            self._last_trade_ts = time.monotonic()

            # Set full cooldown for successful orders
            self._last_order_time[av_key] = now

            # Check if order was immediately filled (market orders or crossing limit orders)
            if result.filled_size and result.filled_size > 0:
                # Order filled immediately - record position
                self.risk_manager.record_position_open(
                    signal=signal,
                    entry_price=result.filled_price or signal.recommended_price,
                    shares=result.filled_size,
                )

                # Record in trade history
                trade_history = get_trade_history()
                trade_history.record_open(
                    asset=asset,
                    side=signal.side.value,
                    entry_price=result.filled_price or signal.recommended_price,
                    shares=result.filled_size,
                    target_price=signal.market.target_price,
                    chainlink_price=current_price,
                    market_id=signal.market.condition_id,
                    predicted_prob=None,
                    arb_type="none",
                    edge=signal.edge,
                    variant=signal.market.variant,
                )

                # Trade attribution + dashboard
                trade_source = "RULES"
                self.attribution.record_execution(
                    asset=asset,
                    market_id=signal.market.condition_id,
                    side=signal.side.value,
                    source=trade_source,
                    decision_context={
                        "min_edge": float(self.config.trading.min_edge),
                        "edge": float(signal.edge or 0),
                        "true_prob": float(signal.true_prob or 0),
                        "price": float(signal.recommended_price or 0),
                        "time_remaining": float(signal.time_remaining or 0),
                    },
                )
                self.dashboard.record_open_position(
                    market_key=signal.market.condition_id,
                    predicted_edge=float(signal.edge or 0),
                )

                logger.info(f"    {Colors.BRIGHT_GREEN}\u2713 FILLED @ {result.filled_price:.2f}{Colors.RESET}")

                # Log positions AFTER trade
                logger.info(
                    f"\U0001f4cb POST-TRADE [{asset}]: Tracked positions={len(self.risk_manager.positions)} | "
                    f"Bankroll=${self.risk_manager.current_bankroll:.2f}"
                )
            else:
                # Order placed but not filled yet - track as pending
                logger.info(
                    f"    {Colors.BRIGHT_YELLOW}\u23f3 ORDER PLACED @ {signal.recommended_price:.2f} "
                    f"(waiting for fill){Colors.RESET}"
                )

                # Store order details for later fill checking
                async with self._pending_orders_lock:
                    self._pending_orders[result.order_id] = {
                        "signal": signal,
                        "asset": asset,
                        "variant": variant,
                        "side": signal.side.value,  # "UP" or "DOWN"
                        "market_id": signal.market.condition_id,
                        "placed_time": now,
                        "price": signal.recommended_price,
                        "size": signal.size_shares,  # For display in status logs
                    }

                async with self._pending_orders_lock:
                    pending_count = len(self._pending_orders)
                logger.info(
                    "PENDING_ORDER_ADDED id=%s asset=%s market=%s side=%s "
                    "limit=%.4f shares=%.4f usd=%.2f pending_total=%d",
                    result.order_id[:16], asset,
                    signal.market.condition_id[:8], signal.side.value,
                    float(signal.recommended_price or 0),
                    float(signal.size_shares or 0),
                    float(signal.size_usd or 0),
                    pending_count,
                )
        else:
            # Set shorter cooldown (30s) on failures to prevent spam
            self._last_order_time[av_key] = now - timedelta(seconds=self._order_cooldown_seconds - 30)
            logger.info(
                "ORDER_REJECT asset=%s reason=execution_error error=%s",
                asset,
                result.error_message,
            )
            logger.warning(f"    {Colors.BRIGHT_RED}\u2717 {result.error_message}{Colors.RESET}")
