"""
Trade History Tracker

Persists all trades to disk for analysis and review.
Tracks prediction accuracy, win rates by asset, and identifies patterns.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Single trade record."""
    timestamp: str
    asset: str
    side: str  # "UP" or "DOWN"
    entry_price: float
    shares: float
    cost: float
    target_price: float
    chainlink_price_at_entry: float
    predicted_prob: Optional[float]  # ML predicted win probability
    actual_outcome: Optional[str] = None  # "WIN" or "LOSS"
    pnl: Optional[float] = None
    exit_price: Optional[float] = None
    market_id: str = ""
    arb_type: str = "none"
    edge: float = 0.0


@dataclass
class TradeHistory:
    """
    Tracks all trades and provides analysis.

    Persists to trade_history.json for review between sessions.
    """

    history_path: str = ""
    trades: list = field(default_factory=list)

    def __post_init__(self):
        """Initialize path and load history."""
        if not self.history_path:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.history_path = os.path.join(project_root, "trade_history.json")
        self._load_history()

    def record_open(
        self,
        asset: str,
        side: str,
        entry_price: float,
        shares: float,
        target_price: float,
        chainlink_price: float,
        market_id: str,
        predicted_prob: Optional[float] = None,
        arb_type: str = "none",
        edge: float = 0.0,
    ) -> str:
        """
        Record a new trade being opened.

        Returns:
            Trade ID (index in list)
        """
        trade = TradeRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            asset=asset,
            side=side,
            entry_price=entry_price,
            shares=shares,
            cost=entry_price * shares,
            target_price=target_price,
            chainlink_price_at_entry=chainlink_price,
            predicted_prob=predicted_prob,
            market_id=market_id,
            arb_type=arb_type,
            edge=edge,
        )

        self.trades.append(asdict(trade))
        self._save_history()

        logger.debug(f"Trade recorded: {asset} {side} @ {entry_price:.4f}")
        return str(len(self.trades) - 1)

    def record_close(
        self,
        market_id: str,
        won: bool,
        pnl: float,
        exit_price: float,
    ):
        """
        Record a trade being closed.

        Args:
            market_id: Market condition ID
            won: Whether trade won
            pnl: Profit/loss amount
            exit_price: Exit price (1.0 for win, 0.0 for loss)
        """
        # Find the trade by market_id (most recent matching)
        for trade in reversed(self.trades):
            if trade.get("market_id") == market_id and trade.get("actual_outcome") is None:
                trade["actual_outcome"] = "WIN" if won else "LOSS"
                trade["pnl"] = pnl
                trade["exit_price"] = exit_price
                self._save_history()
                logger.debug(f"Trade closed: {trade['asset']} {trade['side']} - {'WIN' if won else 'LOSS'}")
                return

        logger.warning(f"Could not find open trade for market {market_id}")

    def get_stats(self) -> dict:
        """Get overall trading statistics."""
        completed = [t for t in self.trades if t.get("actual_outcome") is not None]

        if not completed:
            return {
                "total_trades": len(self.trades),
                "completed_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "prediction_accuracy": 0.0,
            }

        wins = sum(1 for t in completed if t["actual_outcome"] == "WIN")
        losses = len(completed) - wins
        total_pnl = sum(t.get("pnl", 0) or 0 for t in completed)

        # Calculate prediction accuracy (how often ML prediction matched outcome)
        predictions_correct = 0
        predictions_made = 0
        for t in completed:
            if t.get("predicted_prob") is not None:
                predictions_made += 1
                predicted_win = t["predicted_prob"] >= 0.5
                actual_win = t["actual_outcome"] == "WIN"
                if predicted_win == actual_win:
                    predictions_correct += 1

        prediction_accuracy = predictions_correct / predictions_made if predictions_made > 0 else 0.0

        return {
            "total_trades": len(self.trades),
            "completed_trades": len(completed),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(completed) if completed else 0.0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(completed) if completed else 0.0,
            "prediction_accuracy": prediction_accuracy,
            "predictions_made": predictions_made,
        }

    def get_stats_by_asset(self) -> dict:
        """Get statistics broken down by asset."""
        completed = [t for t in self.trades if t.get("actual_outcome") is not None]

        stats = {}
        for asset in ["BTC", "ETH", "SOL", "XRP"]:
            asset_trades = [t for t in completed if t["asset"] == asset]
            if not asset_trades:
                continue

            wins = sum(1 for t in asset_trades if t["actual_outcome"] == "WIN")
            total_pnl = sum(t.get("pnl", 0) or 0 for t in asset_trades)

            stats[asset] = {
                "trades": len(asset_trades),
                "wins": wins,
                "losses": len(asset_trades) - wins,
                "win_rate": wins / len(asset_trades),
                "total_pnl": total_pnl,
            }

        return stats

    def get_stats_by_side(self) -> dict:
        """Get statistics broken down by side (UP vs DOWN)."""
        completed = [t for t in self.trades if t.get("actual_outcome") is not None]

        stats = {}
        for side in ["UP", "DOWN"]:
            side_trades = [t for t in completed if t["side"] == side]
            if not side_trades:
                continue

            wins = sum(1 for t in side_trades if t["actual_outcome"] == "WIN")
            total_pnl = sum(t.get("pnl", 0) or 0 for t in side_trades)

            stats[side] = {
                "trades": len(side_trades),
                "wins": wins,
                "losses": len(side_trades) - wins,
                "win_rate": wins / len(side_trades),
                "total_pnl": total_pnl,
            }

        return stats

    def get_recent_trades(self, count: int = 10) -> list:
        """Get most recent trades."""
        return self.trades[-count:] if self.trades else []

    def _save_history(self):
        """Save history to disk."""
        try:
            with open(self.history_path, "w") as f:
                json.dump({
                    "trades": self.trades,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save trade history: {e}")

    def _load_history(self):
        """Load history from disk."""
        try:
            if os.path.exists(self.history_path):
                with open(self.history_path, "r") as f:
                    data = json.load(f)
                    self.trades = data.get("trades", [])
                    logger.info(f"Loaded {len(self.trades)} trades from history")
        except Exception as e:
            logger.warning(f"Could not load trade history: {e}")
            self.trades = []


# Singleton instance
_trade_history: Optional[TradeHistory] = None


def get_trade_history() -> TradeHistory:
    """Get the singleton trade history instance."""
    global _trade_history
    if _trade_history is None:
        _trade_history = TradeHistory()
    return _trade_history
