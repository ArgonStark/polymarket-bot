"""
Risk management module for the trading bot.

Handles position limits, daily loss limits, and trade validation
to ensure the bot operates within defined risk parameters.
"""

import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

from ..models import Signal, Position, DailyStats, Side, OrderAction
from ..config import BotConfig


logger = logging.getLogger(__name__)


@dataclass
class RiskManager:
    """
    Manages risk parameters and enforces trading limits.

    Key responsibilities:
    - Track open positions
    - Enforce position limits
    - Monitor daily P&L
    - Stop trading if limits hit
    - Track consecutive losses
    - Monitor drawdown from peak
    - Check win rate
    """

    config: BotConfig

    # State tracking
    positions: dict[str, Position] = field(default_factory=dict)
    daily_stats: Optional[DailyStats] = None
    starting_bankroll: float = 0.0
    current_bankroll: float = 0.0

    # Trading state
    is_trading_enabled: bool = True
    halt_reason: Optional[str] = None

    # Advanced protection state
    consecutive_losses: int = 0
    peak_bankroll: float = 0.0  # Highest bankroll achieved (for drawdown)
    cooloff_until: Optional[datetime] = None  # When cooloff ends
    trade_history: list = field(default_factory=list)  # Recent trade results

    def initialize(self, bankroll: float):
        """
        Initialize risk manager with starting bankroll.

        Args:
            bankroll: Starting USDC balance
        """
        self.starting_bankroll = bankroll
        self.current_bankroll = bankroll
        self.peak_bankroll = bankroll  # Track peak for drawdown
        self.consecutive_losses = 0
        self.trade_history = []
        self.cooloff_until = None

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.daily_stats = DailyStats(
            date=today,
            starting_bankroll=bankroll,
            current_bankroll=bankroll,
        )

        logger.info(f"Risk manager initialized with ${bankroll:.2f} bankroll")

    def can_trade(self, equity: Optional[float] = None) -> tuple[bool, str]:
        """
        Check if trading is currently allowed.

        Args:
            equity: Current equity (cash + unrealized position value).
                   If not provided, uses current_bankroll (conservative).

        Returns:
            Tuple of (can_trade, reason_if_not)
        """
        if not self.is_trading_enabled:
            return (False, self.halt_reason or "Trading halted")

        # Check cooloff period
        if self.cooloff_until:
            now = datetime.now(timezone.utc)
            if now < self.cooloff_until:
                remaining = (self.cooloff_until - now).total_seconds() / 60
                return (False, f"Cooling off ({remaining:.0f}m remaining)")
            else:
                # Cooloff expired, reset
                self.cooloff_until = None
                self._reset_after_cooloff()

        if self.daily_stats and self.daily_stats.hit_loss_limit:
            return (False, "Daily loss limit hit")

        # Check consecutive losses
        trading = self.config.trading
        if self.consecutive_losses >= trading.max_consecutive_losses:
            return (False, f"Hit {self.consecutive_losses} consecutive losses")

        # Check drawdown from peak using EQUITY (not just cash)
        # Equity = cash + unrealized position value
        # This prevents false drawdown triggers when positions are open and winning
        current_equity = equity if equity is not None else self.current_bankroll
        if self.peak_bankroll > 0:
            drawdown = (self.peak_bankroll - current_equity) / self.peak_bankroll
            if drawdown >= trading.max_drawdown_pct:
                # AUTO-RESET: If no positions, losses are realized - reset peak to continue trading
                # This prevents the bot from being stuck forever after a losing streak
                if len(self.positions) == 0:
                    old_peak = self.peak_bankroll
                    self.peak_bankroll = current_equity
                    self.consecutive_losses = 0
                    logger.warning(
                        f"🔄 AUTO-RESET DRAWDOWN: No positions, acknowledging realized losses. "
                        f"Peak ${old_peak:.2f} → ${current_equity:.2f}"
                    )
                    # Don't return False - allow trading to continue
                else:
                    return (False, f"Drawdown {drawdown:.1%} exceeds {trading.max_drawdown_pct:.0%} limit")

        # Check win rate (only after minimum trades)
        if len(self.trade_history) >= trading.min_trades_for_winrate:
            wins = sum(1 for t in self.trade_history if t > 0)
            win_rate = wins / len(self.trade_history)
            if win_rate < trading.min_win_rate:
                return (False, f"Win rate {win_rate:.0%} below {trading.min_win_rate:.0%} minimum")

        return (True, "")

    def validate_signal(self, signal: Signal) -> tuple[bool, str]:
        """
        Validate a trading signal against risk limits.

        Checks:
        - Trading is enabled
        - Position limit not exceeded
        - No duplicate positions
        - Have sufficient capital (basic check)

        Note: Position size is NOT validated here - it will be adjusted
        by adjust_signal_size() to fit within limits.

        Args:
            signal: Signal to validate

        Returns:
            Tuple of (is_valid, reason_if_not)
        """
        can, reason = self.can_trade()
        if not can:
            return (False, reason)

        # Check if we already have a position in this market
        market_key = signal.market.condition_id
        if market_key in self.positions:
            return (False, "Already have position in this market")

        # Check if we already have a position in this ASSET (any market)
        # This prevents opening positions in both current and future markets
        for pos in self.positions.values():
            if pos.market.asset == signal.market.asset:
                return (False, f"Already have {signal.market.asset} position")

        # Check position count limit
        trading = self.config.trading
        if len(self.positions) >= trading.max_concurrent_positions:
            return (
                False,
                f"Max positions ({trading.max_concurrent_positions}) reached",
            )

        # Basic capital check - need at least $3 available to trade
        min_trade_size = 3.0
        available = self.current_bankroll * 0.90  # Keep 10% buffer
        if available < min_trade_size:
            return (
                False,
                f"Insufficient capital: ${available:.2f} available (need ${min_trade_size})",
            )

        return (True, "")

    def adjust_signal_size(
        self,
        signal: Signal,
        ml_confidence: Optional[float] = None,
        use_kelly: bool = True,
    ) -> Signal:
        """
        Adjust signal size to fit within risk limits.

        Supports two modes:
        1. Kelly Criterion (when ml_confidence provided and use_kelly=True)
        2. Fixed percentage (fallback)

        Args:
            signal: Trading signal to adjust
            ml_confidence: ML predicted win probability (enables Kelly sizing)
            use_kelly: Whether to use Kelly criterion when confidence available
        """
        trading = self.config.trading

        # Keep 10% buffer for fees
        available = self.current_bankroll * 0.90
        max_position = self.current_bankroll * trading.max_position_pct

        # Try Kelly criterion if ML confidence available
        if use_kelly and ml_confidence is not None and ml_confidence > 0.52:
            try:
                from .kelly import calculate_kelly_position
                kelly_size, kelly_details = calculate_kelly_position(
                    bankroll=self.current_bankroll,
                    win_probability=ml_confidence,
                    market_price=signal.recommended_price,
                    edge=signal.edge,
                    max_position_usd=max_position,
                )
                position_size = min(kelly_size, available)

                # Log Kelly sizing
                kelly_frac = kelly_details.get("adjusted_kelly_fraction", 0)
                logger.debug(
                    f"[{signal.market.asset}] Kelly size: ${position_size:.2f} "
                    f"({kelly_frac:.1%} of bankroll) | ML conf: {ml_confidence:.1%}"
                )
            except Exception as e:
                logger.debug(f"Kelly calculation failed, using fixed %: {e}")
                position_size = min(max_position, available)
        else:
            # Fallback: fixed percentage of bankroll
            position_size = min(max_position, available)

        # Ensure minimum viable trade size ($3)
        if position_size < 3.0:
            logger.debug(f"[{signal.market.asset}] Size ${position_size:.2f} below $3 min, skipping")
            signal.size_usd = 0
            signal.size_shares = 0
            return signal

        # Scale signal to target size
        if signal.size_usd != position_size:
            if signal.size_usd > 0:
                ratio = position_size / signal.size_usd
                signal.size_shares = signal.size_shares * ratio
            signal.size_usd = position_size

        return signal

    def record_position_open(
        self,
        signal: Signal,
        entry_price: float,
        shares: float,
        ml_volatility: Optional[float] = None,
        ml_momentum: Optional[float] = None,
        ml_confidence: Optional[float] = None,
        ml_arb_type: Optional[str] = None,
        ml_spread: Optional[float] = None,
        ml_bid_depth: Optional[float] = None,
        ml_ask_depth: Optional[float] = None,
        ml_price_trend: Optional[float] = None,
        ml_distance_from_target: Optional[float] = None,
        ml_binance_lead_pct: Optional[float] = None,
        ml_binance_confirmation: Optional[str] = None,
        ml_trend_1h: Optional[float] = None,
        ml_trend_4h: Optional[float] = None,
        ml_trend_1d: Optional[float] = None,
        # Chart analysis features
        ml_chart_rsi: Optional[float] = None,
        ml_chart_trend_strength: Optional[float] = None,
        ml_chart_is_uptrend: Optional[float] = None,
        ml_chart_is_downtrend: Optional[float] = None,
        ml_chart_is_ranging: Optional[float] = None,
        ml_chart_bullish_reversal: Optional[float] = None,
        ml_chart_bearish_reversal: Optional[float] = None,
        ml_chart_momentum: Optional[float] = None,
        ml_chart_bias_bullish: Optional[float] = None,
        ml_chart_bias_bearish: Optional[float] = None,
        ml_chart_confidence: Optional[float] = None,
        ml_chart_bullish_pattern: Optional[float] = None,
        ml_chart_bearish_pattern: Optional[float] = None,
        # NEW INDICATOR FEATURES
        ml_macd_histogram: Optional[float] = None,
        ml_macd_crossover: Optional[str] = None,
        ml_bb_bandwidth: Optional[float] = None,
        ml_bb_position: Optional[str] = None,
        ml_stoch_k: Optional[float] = None,
        ml_stoch_d: Optional[float] = None,
        ml_stoch_signal: Optional[str] = None,
        ml_rsi_divergence: Optional[str] = None,
        ml_rsi_divergence_strength: Optional[float] = None,
        ml_volume_ratio: Optional[float] = None,
        ml_is_high_volume: Optional[bool] = None,
        ml_obv_trend: Optional[float] = None,
        ml_ha_trend: Optional[str] = None,
        ml_ha_consecutive: Optional[int] = None,
        ml_ha_strength: Optional[float] = None,
        ml_vwap_distance_pct: Optional[float] = None,
        ml_vwap_position: Optional[str] = None,
    ):
        """
        Record a new position being opened.

        Args:
            signal: Signal that generated the position
            entry_price: Actual entry price
            shares: Number of shares acquired
            ml_volatility: ML feature - asset volatility at entry
            ml_momentum: ML feature - price momentum at entry
            ml_confidence: ML predicted win probability
            ml_arb_type: ML feature - arbitrage type (none, binary_arb, etc.)
            ml_spread: ML feature - market spread at entry
            ml_bid_depth: ML feature - bid depth at entry
            ml_ask_depth: ML feature - ask depth at entry
            ml_price_trend: ML feature - price trend at entry
            ml_distance_from_target: ML feature - distance from target at entry
            ml_binance_lead_pct: ML feature - Binance price lead percentage
            ml_binance_confirmation: ML feature - Binance confirmation type
            ml_trend_1h: ML feature - 1-hour trend
            ml_trend_4h: ML feature - 4-hour trend
            ml_trend_1d: ML feature - 1-day trend
            ml_chart_*: Chart analysis features from Binance candlestick data
            ml_macd_*: MACD indicator features
            ml_bb_*: Bollinger Bands features
            ml_stoch_*: Stochastic oscillator features
            ml_rsi_divergence*: RSI divergence features
            ml_volume_*: Volume analysis features
            ml_ha_*: Heiken Ashi features
            ml_vwap_*: VWAP features
        """
        market_key = signal.market.condition_id

        position = Position(
            market=signal.market,
            side=signal.side,
            token_id=(
                signal.market.up_token_id
                if signal.side == Side.UP
                else signal.market.down_token_id
            ),
            entry_price=entry_price,
            shares=shares,
            entry_time=datetime.now(timezone.utc),
            ml_volatility=ml_volatility,
            ml_momentum=ml_momentum,
            ml_confidence=ml_confidence,
            ml_arb_type=ml_arb_type,
            ml_spread=ml_spread,
            ml_bid_depth=ml_bid_depth,
            ml_ask_depth=ml_ask_depth,
            ml_price_trend=ml_price_trend,
            ml_distance_from_target=ml_distance_from_target,
            ml_binance_lead_pct=ml_binance_lead_pct,
            ml_binance_confirmation=ml_binance_confirmation,
            ml_trend_1h=ml_trend_1h,
            ml_trend_4h=ml_trend_4h,
            ml_trend_1d=ml_trend_1d,
            # Chart analysis features
            ml_chart_rsi=ml_chart_rsi,
            ml_chart_trend_strength=ml_chart_trend_strength,
            ml_chart_is_uptrend=ml_chart_is_uptrend,
            ml_chart_is_downtrend=ml_chart_is_downtrend,
            ml_chart_is_ranging=ml_chart_is_ranging,
            ml_chart_bullish_reversal=ml_chart_bullish_reversal,
            ml_chart_bearish_reversal=ml_chart_bearish_reversal,
            ml_chart_momentum=ml_chart_momentum,
            ml_chart_bias_bullish=ml_chart_bias_bullish,
            ml_chart_bias_bearish=ml_chart_bias_bearish,
            ml_chart_confidence=ml_chart_confidence,
            ml_chart_bullish_pattern=ml_chart_bullish_pattern,
            ml_chart_bearish_pattern=ml_chart_bearish_pattern,
            # NEW INDICATOR FEATURES
            ml_macd_histogram=ml_macd_histogram,
            ml_macd_crossover=ml_macd_crossover,
            ml_bb_bandwidth=ml_bb_bandwidth,
            ml_bb_position=ml_bb_position,
            ml_stoch_k=ml_stoch_k,
            ml_stoch_d=ml_stoch_d,
            ml_stoch_signal=ml_stoch_signal,
            ml_rsi_divergence=ml_rsi_divergence,
            ml_rsi_divergence_strength=ml_rsi_divergence_strength,
            ml_volume_ratio=ml_volume_ratio,
            ml_is_high_volume=ml_is_high_volume,
            ml_obv_trend=ml_obv_trend,
            ml_ha_trend=ml_ha_trend,
            ml_ha_consecutive=ml_ha_consecutive,
            ml_ha_strength=ml_ha_strength,
            ml_vwap_distance_pct=ml_vwap_distance_pct,
            ml_vwap_position=ml_vwap_position,
        )

        self.positions[market_key] = position

        # Deduct cost from available bankroll immediately
        cost = shares * entry_price
        self.current_bankroll -= cost

        confidence_str = f" | ML: {ml_confidence:.0%}" if ml_confidence else ""
        arb_str = f" | ARB: {ml_arb_type}" if ml_arb_type and ml_arb_type != "none" else ""
        logger.info(
            f"Position opened: {signal.side.value} {signal.market.asset} "
            f"{shares:.2f} shares @ {entry_price:.4f} (${cost:.2f}){confidence_str}{arb_str}"
        )

    def record_position_close(
        self,
        market_key: str,
        exit_price: float,
        pnl: float,
    ):
        """
        Record a position being closed.

        Args:
            market_key: Market condition ID
            exit_price: Exit price
            pnl: Realized P&L
        """
        if market_key not in self.positions:
            logger.warning(f"Position not found: {market_key}")
            return

        position = self.positions.pop(market_key)

        # Update stats
        if self.daily_stats:
            self.daily_stats.trades_count += 1
            self.daily_stats.total_pnl += pnl
            if pnl > 0:
                self.daily_stats.wins += 1
            else:
                self.daily_stats.losses += 1

        # Add back the proceeds (original cost + pnl)
        proceeds = position.cost_basis + pnl
        self.current_bankroll += proceeds
        if self.daily_stats:
            self.daily_stats.current_bankroll = self.current_bankroll

        # Track consecutive losses and wins
        self.trade_history.append(pnl)
        if len(self.trade_history) > 20:  # Keep last 20 trades
            self.trade_history.pop(0)

        if pnl > 0:
            # Win - reset consecutive losses, update peak
            self.consecutive_losses = 0
            if self.current_bankroll > self.peak_bankroll:
                self.peak_bankroll = self.current_bankroll
                logger.info(f"New peak bankroll: ${self.peak_bankroll:.2f}")
        else:
            # Loss - increment counter
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.config.trading.max_consecutive_losses:
                logger.warning(
                    f"⚠️ Hit {self.consecutive_losses} consecutive losses - "
                    f"entering {self.config.trading.cooloff_period_minutes}m cooloff"
                )
                self._start_cooloff("consecutive losses")

        logger.info(
            f"Position closed: {position.side.value} {position.market.asset} "
            f"P&L: ${pnl:+.2f} | Streak: {self.consecutive_losses} losses"
        )

        # Check if we've hit any limits
        self._check_all_limits()

    def record_fee(self, fee: float):
        """Record a trading fee."""
        if self.daily_stats:
            self.daily_stats.fees_paid += fee
        self.current_bankroll -= fee

    def record_rebate(self, rebate: float):
        """Record a maker rebate."""
        if self.daily_stats:
            self.daily_stats.rebates_earned += rebate
        self.current_bankroll += rebate

    def _check_loss_limit(self):
        """Check if daily loss limit has been exceeded."""
        if not self.daily_stats:
            return

        daily_return = self.daily_stats.daily_return
        loss_limit = -self.config.trading.daily_loss_limit

        if daily_return <= loss_limit:
            self.is_trading_enabled = False
            self.halt_reason = (
                f"Daily loss limit hit: {daily_return:.1%} "
                f"(limit: {loss_limit:.1%})"
            )
            logger.warning(self.halt_reason)

    def _check_all_limits(self):
        """Check all protection limits after a trade closes."""
        trading = self.config.trading

        # Check daily loss limit
        self._check_loss_limit()

        # Check drawdown
        if self.peak_bankroll > 0:
            drawdown = (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll
            if drawdown >= trading.max_drawdown_pct:
                logger.warning(
                    f"⚠️ Drawdown {drawdown:.1%} exceeds {trading.max_drawdown_pct:.0%} limit - "
                    f"entering cooloff"
                )
                self._start_cooloff("max drawdown")

        # Check win rate
        if len(self.trade_history) >= trading.min_trades_for_winrate:
            wins = sum(1 for t in self.trade_history if t > 0)
            win_rate = wins / len(self.trade_history)
            if win_rate < trading.min_win_rate:
                logger.warning(
                    f"⚠️ Win rate {win_rate:.0%} below {trading.min_win_rate:.0%} - "
                    f"entering cooloff"
                )
                self._start_cooloff("low win rate")

    def _start_cooloff(self, reason: str):
        """Start a cooling off period."""
        from datetime import timedelta
        cooloff_minutes = self.config.trading.cooloff_period_minutes
        self.cooloff_until = datetime.now(timezone.utc) + timedelta(minutes=cooloff_minutes)
        logger.warning(
            f"🛑 COOLOFF STARTED: {reason} | "
            f"Trading paused for {cooloff_minutes} minutes until {self.cooloff_until.strftime('%H:%M:%S')} UTC"
        )

    def _reset_after_cooloff(self):
        """Reset state after cooloff period ends."""
        logger.info("✅ Cooloff period ended - resetting protection counters")
        self.consecutive_losses = 0
        # Don't reset trade_history - win rate should still be monitored
        # Don't reset peak_bankroll - drawdown is still relevant

    def halt_trading(self, reason: str):
        """
        Manually halt trading.

        Args:
            reason: Reason for halt
        """
        self.is_trading_enabled = False
        self.halt_reason = reason
        logger.warning(f"Trading halted: {reason}")

    def resume_trading(self):
        """Resume trading after a halt."""
        if self.daily_stats and self.daily_stats.hit_loss_limit:
            logger.warning("Cannot resume: daily loss limit still in effect")
            return

        self.is_trading_enabled = True
        self.halt_reason = None
        logger.info("Trading resumed")

    def reset_drawdown(self):
        """
        Reset the drawdown tracking by setting peak to current bankroll.

        Use this when you want to restart fresh after a losing streak,
        or when manually adding/removing funds.
        """
        old_peak = self.peak_bankroll
        self.peak_bankroll = self.current_bankroll
        self.consecutive_losses = 0
        logger.info(
            f"🔄 DRAWDOWN RESET: Peak ${old_peak:.2f} → ${self.current_bankroll:.2f} | "
            f"Trading can resume"
        )

    def get_open_positions(self) -> list[Position]:
        """Get list of open positions."""
        return list(self.positions.values())

    def get_position_count(self) -> int:
        """Get number of open positions."""
        return len(self.positions)

    def get_total_exposure(self) -> float:
        """Get total USD exposure across all positions."""
        return sum(p.cost_basis for p in self.positions.values())

    def get_daily_stats(self) -> Optional[DailyStats]:
        """Get current daily statistics."""
        return self.daily_stats

    def reset_daily_stats(self):
        """Reset daily stats for new trading day."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if self.daily_stats and self.daily_stats.date == today:
            logger.debug("Daily stats already current")
            return

        self.daily_stats = DailyStats(
            date=today,
            starting_bankroll=self.current_bankroll,
            current_bankroll=self.current_bankroll,
        )

        # Reset trading state for new day
        if not self.is_trading_enabled and "loss limit" in (self.halt_reason or ""):
            self.is_trading_enabled = True
            self.halt_reason = None

        logger.info(f"Daily stats reset for {today}")

    def sync_bankroll(self, actual_balance: float):
        """
        Sync bankroll with actual balance from exchange.

        Args:
            actual_balance: Current USDC balance from Polymarket
        """
        old_bankroll = self.current_bankroll
        self.current_bankroll = actual_balance
        if self.daily_stats:
            self.daily_stats.current_bankroll = actual_balance

        if abs(old_bankroll - actual_balance) > 0.01:
            logger.debug(f"Bankroll synced: ${old_bankroll:.2f} -> ${actual_balance:.2f}")

    def get_status_summary(self) -> dict:
        """Get summary of current risk status."""
        # Calculate current drawdown
        drawdown = 0.0
        if self.peak_bankroll > 0:
            drawdown = (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll

        # Calculate recent win rate
        recent_win_rate = 0.0
        if self.trade_history:
            wins = sum(1 for t in self.trade_history if t > 0)
            recent_win_rate = wins / len(self.trade_history)

        return {
            "trading_enabled": self.is_trading_enabled,
            "halt_reason": self.halt_reason,
            "starting_bankroll": self.starting_bankroll,
            "current_bankroll": self.current_bankroll,
            "peak_bankroll": self.peak_bankroll,
            "open_positions": self.get_position_count(),
            "total_exposure": self.get_total_exposure(),
            "protection": {
                "consecutive_losses": self.consecutive_losses,
                "max_consecutive_losses": self.config.trading.max_consecutive_losses,
                "drawdown": drawdown,
                "max_drawdown": self.config.trading.max_drawdown_pct,
                "recent_win_rate": recent_win_rate,
                "min_win_rate": self.config.trading.min_win_rate,
                "cooloff_until": self.cooloff_until.isoformat() if self.cooloff_until else None,
            },
            "daily_stats": {
                "trades": self.daily_stats.trades_count if self.daily_stats else 0,
                "win_rate": self.daily_stats.win_rate if self.daily_stats else 0,
                "total_pnl": self.daily_stats.total_pnl if self.daily_stats else 0,
                "net_pnl": self.daily_stats.net_pnl if self.daily_stats else 0,
                "daily_return": self.daily_stats.daily_return if self.daily_stats else 0,
            },
        }

    def get_protection_status(self) -> str:
        """Get a human-readable protection status."""
        trading = self.config.trading
        lines = []

        # Consecutive losses
        lines.append(f"Consecutive losses: {self.consecutive_losses}/{trading.max_consecutive_losses}")

        # Drawdown
        if self.peak_bankroll > 0:
            drawdown = (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll
            lines.append(f"Drawdown: {drawdown:.1%}/{trading.max_drawdown_pct:.0%}")

        # Win rate
        if self.trade_history:
            wins = sum(1 for t in self.trade_history if t > 0)
            win_rate = wins / len(self.trade_history)
            lines.append(f"Win rate: {win_rate:.0%}/{trading.min_win_rate:.0%} ({len(self.trade_history)} trades)")

        # Cooloff
        if self.cooloff_until:
            now = datetime.now(timezone.utc)
            if now < self.cooloff_until:
                remaining = (self.cooloff_until - now).total_seconds() / 60
                lines.append(f"COOLOFF: {remaining:.0f}m remaining")

        return " | ".join(lines)
