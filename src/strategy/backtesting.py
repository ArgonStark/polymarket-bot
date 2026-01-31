"""
Backtesting Framework

Replay historical trades through ML models to evaluate performance.
Supports walk-forward testing, Monte Carlo simulation, and performance metrics.
"""

import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Tuple, Callable

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a historical trade."""
    timestamp: str
    asset: str
    side: str  # UP or DOWN
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    won: bool

    # Features at entry (if available)
    features: List[float] = field(default_factory=list)

    # Metadata
    edge: float = 0.0
    arb_type: str = "none"
    ml_confidence: float = 0.5

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "asset": self.asset,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "shares": self.shares,
            "pnl": self.pnl,
            "won": self.won,
            "features": self.features,
            "edge": self.edge,
            "arb_type": self.arb_type,
            "ml_confidence": self.ml_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trade":
        return cls(
            timestamp=data.get("timestamp", ""),
            asset=data.get("asset", "BTC"),
            side=data.get("side", "UP"),
            entry_price=data.get("entry_price", 0.5),
            exit_price=data.get("exit_price", 0.5),
            shares=data.get("shares", 10),
            pnl=data.get("pnl", 0.0),
            won=data.get("won", False),
            features=data.get("features", []),
            edge=data.get("edge", 0.0),
            arb_type=data.get("arb_type", "none"),
            ml_confidence=data.get("ml_confidence", 0.5),
        )


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    start_date: str
    end_date: str
    initial_bankroll: float
    final_bankroll: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int

    # Performance metrics
    total_pnl: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    calmar_ratio: float

    # Per-trade statistics
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_pnl: float

    # Streaks
    max_consecutive_wins: int
    max_consecutive_losses: int

    # Equity curve
    equity_curve: List[float] = field(default_factory=list)

    # Trade details
    trades: List[Trade] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"BacktestResult(\n"
            f"  Total PnL: ${self.total_pnl:.2f}\n"
            f"  Win Rate: {self.win_rate:.1%}\n"
            f"  Trades: {self.total_trades}\n"
            f"  Sharpe: {self.sharpe_ratio:.2f}\n"
            f"  Max DD: {self.max_drawdown_pct:.1%}\n"
            f")"
        )

    def to_dict(self) -> dict:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_bankroll": self.initial_bankroll,
            "final_bankroll": self.final_bankroll,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": self.total_pnl,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "calmar_ratio": self.calmar_ratio,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "avg_trade_pnl": self.avg_trade_pnl,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
        }


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio from returns."""
    if not returns or len(returns) < 2:
        return 0.0

    mean_return = sum(returns) / len(returns)
    excess_returns = [r - risk_free_rate for r in returns]
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    std_dev = math.sqrt(variance)

    if std_dev == 0:
        return 0.0

    # Annualize (assuming daily returns)
    return (mean_return - risk_free_rate) / std_dev * math.sqrt(252)


def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio (only considers downside volatility)."""
    if not returns or len(returns) < 2:
        return 0.0

    mean_return = sum(returns) / len(returns)
    negative_returns = [r for r in returns if r < 0]

    if not negative_returns:
        return float('inf') if mean_return > 0 else 0.0

    downside_variance = sum(r ** 2 for r in negative_returns) / len(returns)
    downside_std = math.sqrt(downside_variance)

    if downside_std == 0:
        return 0.0

    return (mean_return - risk_free_rate) / downside_std * math.sqrt(252)


def calculate_max_drawdown(equity_curve: List[float]) -> Tuple[float, float]:
    """Calculate maximum drawdown in absolute and percentage terms."""
    if not equity_curve:
        return 0.0, 0.0

    peak = equity_curve[0]
    max_dd = 0.0
    max_dd_pct = 0.0

    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = peak - equity
        dd_pct = dd / peak if peak > 0 else 0

        if dd > max_dd:
            max_dd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    return max_dd, max_dd_pct


class Backtester:
    """
    Backtesting engine for ML trading strategies.

    Features:
    - Replay historical trades
    - Walk-forward testing (train on past, test on future)
    - Monte Carlo simulation
    - Comprehensive performance metrics
    """

    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        position_size_pct: float = 0.10,
        fee_rate: float = 0.002,  # 0.2% fee
    ):
        self.initial_bankroll = initial_bankroll
        self.position_size_pct = position_size_pct
        self.fee_rate = fee_rate

        # Historical trades
        self.trades: List[Trade] = []
        self._load_trades()

    def _load_trades(self):
        """Load historical trades from trade history."""
        try:
            from .trade_history import get_trade_history
            history = get_trade_history()
            completed = history.get_completed_trades()

            for trade_data in completed:
                trade = Trade(
                    timestamp=trade_data.get("entry_time", ""),
                    asset=trade_data.get("asset", "BTC"),
                    side=trade_data.get("side", "UP"),
                    entry_price=trade_data.get("entry_price", 0.5),
                    exit_price=trade_data.get("exit_price", 0.5),
                    shares=trade_data.get("shares", 10),
                    pnl=trade_data.get("pnl", 0.0),
                    won=trade_data.get("pnl", 0.0) > 0,
                    edge=trade_data.get("edge", 0.0),
                    arb_type=trade_data.get("arb_type", "none"),
                    ml_confidence=trade_data.get("ml_confidence", 0.5),
                )
                self.trades.append(trade)

            logger.info(f"Loaded {len(self.trades)} historical trades for backtesting")
        except Exception as e:
            logger.warning(f"Could not load trade history: {e}")

    def add_trade(self, trade: Trade):
        """Add a trade to the history."""
        self.trades.append(trade)

    def run_backtest(
        self,
        trades: Optional[List[Trade]] = None,
        ml_filter: Optional[Callable[[List[float]], bool]] = None,
        use_kelly: bool = False,
    ) -> BacktestResult:
        """
        Run a backtest on historical trades.

        Args:
            trades: List of trades to backtest (uses self.trades if None)
            ml_filter: Optional function that takes features and returns True if trade should be taken
            use_kelly: Use Kelly criterion for position sizing

        Returns:
            BacktestResult with comprehensive metrics
        """
        trades = trades or self.trades
        if not trades:
            return self._empty_result()

        # Initialize
        bankroll = self.initial_bankroll
        equity_curve = [bankroll]
        executed_trades = []

        # Tracking
        wins = []
        losses = []
        returns = []
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0

        for trade in trades:
            # Apply ML filter if provided
            if ml_filter is not None and trade.features:
                if not ml_filter(trade.features):
                    continue  # Skip this trade

            # Calculate position size
            if use_kelly and trade.ml_confidence > 0.5:
                from .kelly import calculate_kelly_position
                position_size, _ = calculate_kelly_position(
                    bankroll=bankroll,
                    win_probability=trade.ml_confidence,
                    market_price=trade.entry_price,
                    edge=trade.edge,
                )
            else:
                position_size = bankroll * self.position_size_pct

            # Cap at available bankroll
            position_size = min(position_size, bankroll * 0.9)

            # Calculate shares and fees
            shares = position_size / trade.entry_price if trade.entry_price > 0 else 0
            fee = position_size * self.fee_rate

            # Calculate PnL
            if trade.won:
                pnl = shares * (trade.exit_price - trade.entry_price) - fee
            else:
                pnl = -position_size * (1 - trade.exit_price) - fee

            # Update bankroll
            bankroll += pnl
            equity_curve.append(bankroll)

            # Track wins/losses
            if pnl > 0:
                wins.append(pnl)
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                losses.append(pnl)
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

            # Track returns
            if equity_curve[-2] > 0:
                ret = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
                returns.append(ret)

            # Record executed trade
            executed_trade = Trade(
                timestamp=trade.timestamp,
                asset=trade.asset,
                side=trade.side,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                shares=shares,
                pnl=pnl,
                won=pnl > 0,
                features=trade.features,
                edge=trade.edge,
                arb_type=trade.arb_type,
                ml_confidence=trade.ml_confidence,
            )
            executed_trades.append(executed_trade)

        # Calculate metrics
        total_pnl = bankroll - self.initial_bankroll
        total_trades = len(executed_trades)
        winning_trades = len(wins)
        losing_trades = len(losses)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')

        sharpe = calculate_sharpe_ratio(returns)
        sortino = calculate_sortino_ratio(returns)
        max_dd, max_dd_pct = calculate_max_drawdown(equity_curve)

        # Calmar ratio = annualized return / max drawdown
        days = len(trades)  # Approximate
        annual_return = (bankroll / self.initial_bankroll) ** (365 / max(1, days)) - 1
        calmar = annual_return / max_dd_pct if max_dd_pct > 0 else 0

        return BacktestResult(
            start_date=trades[0].timestamp if trades else "",
            end_date=trades[-1].timestamp if trades else "",
            initial_bankroll=self.initial_bankroll,
            final_bankroll=bankroll,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            calmar_ratio=calmar,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=max(wins) if wins else 0,
            largest_loss=min(losses) if losses else 0,
            avg_trade_pnl=total_pnl / total_trades if total_trades > 0 else 0,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            equity_curve=equity_curve,
            trades=executed_trades,
        )

    def walk_forward_test(
        self,
        train_ratio: float = 0.7,
        ml_model=None,
    ) -> Tuple[BacktestResult, BacktestResult]:
        """
        Walk-forward testing: train on past data, test on future data.

        Args:
            train_ratio: Fraction of data to use for training
            ml_model: ML model with predict_proba and update methods

        Returns:
            Tuple of (train_result, test_result)
        """
        if not self.trades:
            return self._empty_result(), self._empty_result()

        # Split data
        split_idx = int(len(self.trades) * train_ratio)
        train_trades = self.trades[:split_idx]
        test_trades = self.trades[split_idx:]

        # Train on historical data
        if ml_model is not None:
            for trade in train_trades:
                if trade.features:
                    ml_model.update(trade.features, 1 if trade.won else 0)

        # Create ML filter function
        def ml_filter(features: List[float]) -> bool:
            if ml_model is None or not features:
                return True
            prob = ml_model.predict_proba(features)
            return prob >= 0.5

        # Run backtest on training data (without ML filter)
        train_result = self.run_backtest(trades=train_trades)

        # Run backtest on test data (with ML filter)
        test_result = self.run_backtest(trades=test_trades, ml_filter=ml_filter)

        return train_result, test_result

    def monte_carlo_simulation(
        self,
        n_simulations: int = 1000,
        trades: Optional[List[Trade]] = None,
    ) -> Dict:
        """
        Monte Carlo simulation to estimate distribution of outcomes.

        Randomly resamples trades with replacement to estimate:
        - Distribution of final bankroll
        - Distribution of max drawdown
        - Confidence intervals

        Args:
            n_simulations: Number of simulations to run
            trades: Trades to sample from (uses self.trades if None)

        Returns:
            Dictionary with simulation results
        """
        trades = trades or self.trades
        if not trades:
            return {"error": "No trades available"}

        final_bankrolls = []
        max_drawdowns = []
        win_rates = []

        for _ in range(n_simulations):
            # Resample trades with replacement
            sampled_trades = random.choices(trades, k=len(trades))

            # Run backtest
            result = self.run_backtest(trades=sampled_trades)

            final_bankrolls.append(result.final_bankroll)
            max_drawdowns.append(result.max_drawdown_pct)
            win_rates.append(result.win_rate)

        # Calculate statistics
        final_bankrolls.sort()
        max_drawdowns.sort()

        return {
            "n_simulations": n_simulations,
            "n_trades": len(trades),
            "final_bankroll": {
                "mean": sum(final_bankrolls) / len(final_bankrolls),
                "median": final_bankrolls[len(final_bankrolls) // 2],
                "std": math.sqrt(sum((x - sum(final_bankrolls)/len(final_bankrolls))**2 for x in final_bankrolls) / len(final_bankrolls)),
                "percentile_5": final_bankrolls[int(0.05 * len(final_bankrolls))],
                "percentile_25": final_bankrolls[int(0.25 * len(final_bankrolls))],
                "percentile_75": final_bankrolls[int(0.75 * len(final_bankrolls))],
                "percentile_95": final_bankrolls[int(0.95 * len(final_bankrolls))],
            },
            "max_drawdown": {
                "mean": sum(max_drawdowns) / len(max_drawdowns),
                "median": max_drawdowns[len(max_drawdowns) // 2],
                "percentile_95": max_drawdowns[int(0.95 * len(max_drawdowns))],
            },
            "win_rate": {
                "mean": sum(win_rates) / len(win_rates),
                "std": math.sqrt(sum((x - sum(win_rates)/len(win_rates))**2 for x in win_rates) / len(win_rates)),
            },
        }

    def _empty_result(self) -> BacktestResult:
        """Return an empty backtest result."""
        return BacktestResult(
            start_date="",
            end_date="",
            initial_bankroll=self.initial_bankroll,
            final_bankroll=self.initial_bankroll,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_pnl=0,
            win_rate=0,
            profit_factor=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown=0,
            max_drawdown_pct=0,
            calmar_ratio=0,
            avg_win=0,
            avg_loss=0,
            largest_win=0,
            largest_loss=0,
            avg_trade_pnl=0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
        )


# Singleton instance
_backtester: Optional[Backtester] = None


def get_backtester(initial_bankroll: float = 1000.0) -> Backtester:
    """Get or create the backtester singleton."""
    global _backtester
    if _backtester is None:
        _backtester = Backtester(initial_bankroll=initial_bankroll)
    return _backtester


def run_quick_backtest() -> BacktestResult:
    """Run a quick backtest on historical trades."""
    backtester = get_backtester()
    return backtester.run_backtest()


def run_monte_carlo(n_simulations: int = 1000) -> Dict:
    """Run Monte Carlo simulation."""
    backtester = get_backtester()
    return backtester.monte_carlo_simulation(n_simulations=n_simulations)
