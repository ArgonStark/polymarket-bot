"""Early exit (take-profit, stop-loss, chart-based) mixin."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from src.models import MarketState, Signal, OrderAction, Side
from src.strategy.trade_history import get_trade_history
from src.utils import Colors

logger = logging.getLogger(__name__)


class EarlyExitMixin:
    """Mixin for early exit logic (take-profit, stop-loss, chart, time). Mixed into TradingBot."""

    async def _check_early_exits(self):
        """
        Check open positions for take-profit, stop-loss, or chart-based exit conditions.

        For each position:
        1. Get current market price from order book
        2. Calculate unrealized P&L percentage
        3. If >= take_profit_pct: sell to lock in profit
        4. If <= -stop_loss_pct: sell to limit loss
        5. Check chart analysis for trend reversal or DCA opportunity
        """
        if not self.risk_manager.positions:
            return

        now = datetime.now(timezone.utc)
        trading = self.config.trading

        # Import position monitor
        try:
            from .strategy.position_monitor import analyze_open_position, PositionAction
            from .data.binance_chart import analyze_chart
            POSITION_MONITOR_AVAILABLE = True
        except ImportError:
            POSITION_MONITOR_AVAILABLE = False

        for market_key, position in list(self.risk_manager.positions.items()):
            try:
                # Find the market (in active or expiring) — needed for variant timing
                market = self.markets.get(market_key) or self.expiring_markets.get(market_key)
                if not market:
                    continue

                # Use variant-aware timing
                timing = trading.get_timing(getattr(market, "variant", "fifteen"))

                # Check minimum hold time
                hold_time = (now - position.entry_time).total_seconds()
                if hold_time < timing["early_exit_min_hold"]:
                    continue

                # Get current market price for our position side
                if position.side == Side.UP:
                    current_price = market.best_bid  # Selling UP, get bid
                else:
                    # For DOWN, price is 1 - UP_ask (we sell DOWN = buy UP at ask)
                    current_price = 1.0 - market.best_ask if market.best_ask else None

                if current_price is None or current_price <= 0:
                    continue

                # Calculate unrealized P&L percentage
                entry_price = position.entry_price
                pnl_pct = (current_price - entry_price) / entry_price

                # Check take-profit
                if pnl_pct >= trading.take_profit_pct:
                    logger.info(
                        f"{Colors.BRIGHT_GREEN}💰 TAKE PROFIT: {market.asset} {position.side.value} | "
                        f"P&L: {pnl_pct:+.0%} | Entry: {entry_price:.2f} → Exit: {current_price:.2f}{Colors.RESET}"
                    )
                    await self._execute_early_exit(market, position, current_price, "take_profit")
                    continue

                # Check stop-loss
                if pnl_pct <= -trading.stop_loss_pct:
                    logger.info(
                        f"{Colors.BRIGHT_RED}🛑 STOP LOSS: {market.asset} {position.side.value} | "
                        f"P&L: {pnl_pct:+.0%} | Entry: {entry_price:.2f} → Exit: {current_price:.2f}{Colors.RESET}"
                    )
                    await self._execute_early_exit(market, position, current_price, "stop_loss")
                    continue

                # Time-based exit if edge decays near settlement
                if trading.time_exit_enabled and market.time_remaining <= timing["time_exit_seconds"]:
                    decay_signal = self.signal_generator.generate_signal(market)
                    decay_edge = decay_signal.edge if decay_signal else 0.0
                    if decay_edge < trading.exit_edge_threshold:
                        logger.info(
                            f"{Colors.BRIGHT_YELLOW}⏱️ TIME EXIT: {market.asset} {position.side.value} | "
                            f"Edge {decay_edge:.2%} < {trading.exit_edge_threshold:.2%} | "
                            f"{market.time_remaining:.0f}s remaining{Colors.RESET}"
                        )
                        await self._execute_early_exit(market, position, current_price, "time_exit")
                        continue

                # === CHART-BASED POSITION MONITORING ===
                # Check 15m chart to decide: hold, close, or DCA
                if POSITION_MONITOR_AVAILABLE:
                    await self._check_chart_based_exit(market, position, current_price)

            except Exception as e:
                logger.error(f"Error checking early exit for {market_key}: {e}")

    async def _check_chart_based_exit(self, market: MarketState, position, current_token_price: float):
        """
        Check if chart analysis suggests exiting or modifying a position.

        Uses 15m candlestick data to detect:
        - Trend reversals (close position)
        - Strong confirmation (DCA opportunity)
        - Extreme RSI (potential reversal)
        """
        try:
            from .strategy.position_monitor import analyze_open_position, PositionAction, get_position_monitor
            from .data.binance_chart import analyze_chart
        except ImportError:
            return

        asset = market.asset
        monitor = get_position_monitor()

        # Check if we should analyze (rate limited)
        if not monitor.should_check(asset):
            return

        # Get chart analysis
        chart_analysis = analyze_chart(asset)
        if not chart_analysis:
            return

        # Analyze position
        decision = analyze_open_position(
            position=position,
            chart_analysis=chart_analysis,
            current_token_price=current_token_price,
            time_remaining=market.time_remaining,
        )

        # Log position status periodically
        summary = monitor.get_position_summary(position, chart_analysis, current_token_price)
        logger.info(f"📊 POSITION MONITOR: {summary} | Action: {decision.action.value}")

        # Act on decision
        if decision.action == PositionAction.CLOSE:
            logger.info(
                f"{Colors.BRIGHT_YELLOW}📉 CHART EXIT: {asset} {position.side.value} | "
                f"Reason: {decision.reason} | Confidence: {decision.confidence:.0%}{Colors.RESET}"
            )
            await self._execute_early_exit(market, position, current_token_price, f"chart_exit_{decision.urgency}")

        elif decision.action == PositionAction.DCA:
            logger.info(
                f"{Colors.BRIGHT_CYAN}📈 CHART DCA: {asset} {position.side.value} | "
                f"Reason: {decision.reason} | Size: ${decision.suggested_size:.2f}{Colors.RESET}"
            )
            # Execute DCA through averaging system
            await self._execute_chart_dca(market, position, decision)

        elif decision.action == PositionAction.ADD:
            logger.info(
                f"{Colors.BRIGHT_GREEN}📈 CHART ADD: {asset} {position.side.value} | "
                f"Reason: {decision.reason} | Size: ${decision.suggested_size:.2f}{Colors.RESET}"
            )
            # Execute ADD through averaging system
            await self._execute_chart_dca(market, position, decision)

        # HOLD - just log at debug level
        else:
            logger.debug(f"[{asset}] Position HOLD: {decision.reason}")

    async def _execute_chart_dca(self, market: MarketState, position, decision):
        """
        Execute a DCA or ADD order based on chart analysis.

        Args:
            market: Market state
            position: Position to add to
            decision: PositionDecision with suggested size
        """
        try:
            from .strategy.averaging import update_position_after_averaging
        except ImportError:
            logger.error("Could not import averaging module")
            return

        asset = market.asset

        # Risk guard: check balance before DCA
        available = self.risk_manager.current_bankroll * 0.90
        if decision.suggested_size > available:
            logger.info(
                "[%s] CHART_DCA_BLOCKED reason=insufficient_balance need=%.2f have=%.2f",
                asset, decision.suggested_size, available,
            )
            return

        # Risk guard: check total exposure cap
        current_exposure = sum(
            pos.shares * pos.entry_price for pos in self.risk_manager.positions.values()
        )
        equity = self._calculate_equity()
        max_exposure_pct = getattr(self.config.trading, 'max_total_exposure_pct', 0.25)
        if current_exposure + decision.suggested_size > equity * max_exposure_pct:
            logger.info(
                "[%s] CHART_DCA_BLOCKED reason=exposure_cap current=%.2f add=%.2f cap=%.2f",
                asset, current_exposure, decision.suggested_size, equity * max_exposure_pct,
            )
            return

        # Calculate shares and price
        if position.side == Side.UP:
            token_price = market.best_ask
        else:
            token_price = 1 - market.best_bid

        shares = decision.suggested_size / token_price
        if shares < 5:
            shares = 5
            decision.suggested_size = shares * token_price

        # Create signal for execution
        signal = Signal(
            market=market,
            side=position.side,
            edge=0.10,  # Placeholder
            true_prob=0.5,
            market_prob=token_price,
            recommended_action=OrderAction.LIMIT,
            recommended_price=token_price,
            size_usd=decision.suggested_size,
            size_shares=shares,
            chainlink_price=self.signal_generator.get_price(asset) or 0,
            time_remaining=market.time_remaining,
            reasoning=f"[CHART {decision.action.value}] {decision.reason}",
        )

        # Execute the order
        start_exec = time.perf_counter()
        try:
            if self.smart_router:
                result = await self.smart_router.execute(signal)
            else:
                result = await self.executor.execute_signal_async(signal)
        except Exception as e:
            logger.error(
                "ORDER_RESULT asset=%s variant=%s side=%s status=exception error=%s mode=%s",
                asset, signal.market.variant, signal.side.value, str(e)[:200],
                "paper" if self.config.paper_trading.enabled else "live"
            )
            logger.debug("Chart execution exception traceback:", exc_info=True)
            return  # Exit gracefully
        exec_latency_ms = (time.perf_counter() - start_exec) * 1000
        self.metrics.record_latency("execute_signal", exec_latency_ms)

        if result.success:
            # Calculate new average price
            old_cost = position.total_cost if position.total_cost > 0 else (position.entry_price * position.shares)
            new_total_cost = old_cost + decision.suggested_size
            new_total_shares = position.shares + shares
            new_avg_price = new_total_cost / new_total_shares

            # Update position
            update_position_after_averaging(
                position=position,
                additional_shares=shares,
                additional_cost=decision.suggested_size,
                new_entry_price=new_avg_price,
            )

            # Set cooldown
            now = datetime.now(timezone.utc)
            self._last_order_time[f"{asset}:{market.variant}"] = now

            logger.info(
                f"[{asset}/{market.variant}] CHART {decision.action.value}: "
                f"Added {shares:.1f} shares @ ${decision.suggested_size:.2f} | "
                f"New avg: ${new_avg_price:.3f}"
            )
        else:
            logger.warning(f"[{asset}] CHART {decision.action.value} FAILED: {result.error_message}")

    async def _execute_early_exit(
        self,
        market: MarketState,
        position,
        exit_price: float,
        exit_reason: str,
    ):
        """
        Execute early exit (sell position before settlement).

        Args:
            market: Market state
            position: Position to exit
            exit_price: Price to sell at
            exit_reason: "take_profit" or "stop_loss"
        """
        # In dry run mode (but NOT paper trading), just log and process tracking
        if self.config.dry_run and not self.config.paper_trading.enabled:
            logger.info(f"[DRY RUN] Would sell {position.shares:.2f} shares at {exit_price:.4f}")
            # Still process the exit for tracking
            pnl = (exit_price - position.entry_price) * position.shares
            await self._process_early_exit_result(market, position, exit_price, pnl, exit_reason)
            return

        if not self.executor:
            logger.warning("No executor available for early exit")
            return

        try:
            # Create a sell signal
            from .models import Signal, OrderAction
            sell_signal = Signal(
                market=market,
                side=position.side,
                recommended_action=OrderAction.MARKET,  # Market order for quick exit
                recommended_price=exit_price,
                size_usd=position.shares * exit_price,
                size_shares=position.shares,
                edge=0.0,  # Not relevant for exit
                reasoning=f"Early exit: {exit_reason}",
            )

            # Execute sell order
            result = self.executor.sell_position(
                token_id=position.token_id,
                shares=position.shares,
                min_price=exit_price * 0.98,  # Allow 2% slippage
            )

            if result and result.success:
                actual_price = result.filled_price or exit_price
                pnl = (actual_price - position.entry_price) * position.shares
                logger.info(
                    f"✅ Early exit executed: {market.asset} {position.side.value} | "
                    f"Sold {result.filled_size:.2f} @ {actual_price:.4f} | P&L: ${pnl:+.2f}"
                )
                await self._process_early_exit_result(market, position, actual_price, pnl, exit_reason)
            else:
                error_msg = result.error_message if result else "Unknown error"
                logger.warning(f"Early exit failed: {error_msg}")

        except Exception as e:
            logger.error(f"Error executing early exit: {e}")

    async def _process_early_exit_result(
        self,
        market: MarketState,
        position,
        exit_price: float,
        pnl: float,
        exit_reason: str,
    ):
        """
        Process the result of an early exit - update tracking, ML, etc.

        Args:
            market: Market state
            position: Closed position
            exit_price: Actual exit price
            pnl: Realized P&L
            exit_reason: "take_profit" or "stop_loss"
        """
        won = pnl > 0

        # Record with risk manager
        self.risk_manager.record_position_close(
            market_key=market.condition_id,
            exit_price=exit_price,
            pnl=pnl,
        )
        self.metrics.record_trade(market.asset, pnl)

        # Record with trade history (mark as early exit)
        trade_history = get_trade_history()
        trade_history.record_close(
            market_id=market.condition_id,
            won=won,
            pnl=pnl,
            exit_price=exit_price,
        )

        # Clear (asset, variant) cooldown so we can trade again
        asset = market.asset
        _av_key = f"{asset}:{market.variant}"
        if _av_key in self._last_order_time:
            del self._last_order_time[_av_key]

        # Log summary
        result_emoji = "💰" if exit_reason == "take_profit" else "🛑"
        logger.info(
            f"{result_emoji} EARLY EXIT COMPLETE: {market.asset} | "
            f"Reason: {exit_reason} | P&L: ${pnl:+.2f} | "
            f"Bankroll: ${self.risk_manager.current_bankroll:.2f}"
        )
