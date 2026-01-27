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

    def initialize(self, bankroll: float):
        """
        Initialize risk manager with starting bankroll.

        Args:
            bankroll: Starting USDC balance
        """
        self.starting_bankroll = bankroll
        self.current_bankroll = bankroll

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.daily_stats = DailyStats(
            date=today,
            starting_bankroll=bankroll,
            current_bankroll=bankroll,
        )

        logger.info(f"Risk manager initialized with ${bankroll:.2f} bankroll")

    def can_trade(self) -> tuple[bool, str]:
        """
        Check if trading is currently allowed.

        Returns:
            Tuple of (can_trade, reason_if_not)
        """
        if not self.is_trading_enabled:
            return (False, self.halt_reason or "Trading halted")

        if self.daily_stats and self.daily_stats.hit_loss_limit:
            return (False, "Daily loss limit hit")

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

    def adjust_signal_size(self, signal: Signal) -> Signal:
        """
        Adjust signal size to fit within risk limits.

        Uses ONLY bankroll percentage - no fixed base size.
        Position size = bankroll × max_position_pct
        """
        trading = self.config.trading

        # Simple: position size is percentage of current bankroll
        # Keep 10% buffer for fees
        available = self.current_bankroll * 0.90
        position_size = min(
            self.current_bankroll * trading.max_position_pct,
            available
        )

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
    ):
        """
        Record a new position being opened.

        Args:
            signal: Signal that generated the position
            entry_price: Actual entry price
            shares: Number of shares acquired
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
        )

        self.positions[market_key] = position

        # Deduct cost from available bankroll immediately
        cost = shares * entry_price
        self.current_bankroll -= cost

        logger.info(
            f"Position opened: {signal.side.value} {signal.market.asset} "
            f"{shares:.2f} shares @ {entry_price:.4f} (${cost:.2f})"
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

        logger.info(
            f"Position closed: {position.side.value} {position.market.asset} "
            f"P&L: ${pnl:+.2f}"
        )

        # Check if we've hit loss limit
        self._check_loss_limit()

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
        return {
            "trading_enabled": self.is_trading_enabled,
            "halt_reason": self.halt_reason,
            "starting_bankroll": self.starting_bankroll,
            "current_bankroll": self.current_bankroll,
            "open_positions": self.get_position_count(),
            "total_exposure": self.get_total_exposure(),
            "daily_stats": {
                "trades": self.daily_stats.trades_count if self.daily_stats else 0,
                "win_rate": self.daily_stats.win_rate if self.daily_stats else 0,
                "total_pnl": self.daily_stats.total_pnl if self.daily_stats else 0,
                "net_pnl": self.daily_stats.net_pnl if self.daily_stats else 0,
                "daily_return": self.daily_stats.daily_return if self.daily_stats else 0,
            },
        }
