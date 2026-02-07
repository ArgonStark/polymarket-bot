"""
Bot state persistence — save/restore bankroll, peak, positions across restarts.

Uses atomic writes (tmp file + os.replace) to prevent corruption on crash.
State file defaults to bot_state.json in project root.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PersistedState:
    """Serializable snapshot of bot state."""

    # Accounting
    current_bankroll: float = 0.0
    peak_bankroll: float = 0.0
    starting_bankroll: float = 0.0

    # Risk tracking
    consecutive_losses: int = 0
    trade_history: list = field(default_factory=list)  # last 20 PnLs

    # Daily stats snapshot
    daily_date: str = ""
    daily_pnl: float = 0.0
    daily_trades: int = 0
    daily_wins: int = 0
    daily_losses: int = 0
    daily_fees: float = 0.0
    daily_rebates: float = 0.0

    # Open positions (minimal dicts for reconstruction)
    open_positions: list = field(default_factory=list)

    # Metadata
    saved_at: str = ""
    version: int = 1


def _serialize_position(position) -> dict:
    """Serialize a Position to a minimal dict for persistence."""
    from .models import Side

    market = position.market
    return {
        "condition_id": market.condition_id,
        "question": market.question,
        "asset": market.asset,
        "target_price": market.target_price,
        "up_token_id": market.up_token_id,
        "down_token_id": market.down_token_id,
        "start_time": market.start_time.isoformat(),
        "end_time": market.end_time.isoformat(),
        "best_bid": market.best_bid,
        "best_ask": market.best_ask,
        "side": position.side.value,
        "token_id": position.token_id,
        "entry_price": position.entry_price,
        "shares": position.shares,
        "entry_time": position.entry_time.isoformat(),
        "market_id": position.market_id,
        "yes_token_id": position.yes_token_id,
        "no_token_id": position.no_token_id,
        "held_token_id": position.held_token_id,
        "times_averaged": position.times_averaged,
        "total_cost": position.total_cost,
        "original_entry_price": position.original_entry_price,
    }


def _deserialize_position(data: dict):
    """Reconstruct a Position from a persisted dict."""
    from .models import MarketState, Position, Side

    market = MarketState(
        condition_id=data["condition_id"],
        question=data.get("question", ""),
        up_token_id=data["up_token_id"],
        down_token_id=data["down_token_id"],
        asset=data.get("asset", ""),
        target_price=data.get("target_price", 0.0),
        start_time=datetime.fromisoformat(data["start_time"]),
        end_time=datetime.fromisoformat(data["end_time"]),
        best_bid=data.get("best_bid", 0.0),
        best_ask=data.get("best_ask", 1.0),
    )

    side = Side(data["side"])
    return Position(
        market=market,
        side=side,
        token_id=data["token_id"],
        entry_price=data["entry_price"],
        shares=data["shares"],
        entry_time=datetime.fromisoformat(data["entry_time"]),
        market_id=data.get("market_id", ""),
        yes_token_id=data.get("yes_token_id", ""),
        no_token_id=data.get("no_token_id", ""),
        held_token_id=data.get("held_token_id", ""),
        times_averaged=data.get("times_averaged", 0),
        total_cost=data.get("total_cost", 0.0),
        original_entry_price=data.get("original_entry_price"),
    )


class BotStateManager:
    """Save and restore bot state to/from a JSON file."""

    def __init__(self, state_file: str = ""):
        if not state_file:
            project_root = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
            state_file = os.path.join(project_root, "bot_state.json")
        self.state_file = state_file

    def save(self, risk_manager, markets: Optional[dict] = None) -> bool:
        """
        Snapshot risk manager state and write atomically to disk.

        Returns True on success, False on error.
        """
        try:
            # Build position list from risk manager
            positions = []
            for key, pos in risk_manager.positions.items():
                try:
                    positions.append(_serialize_position(pos))
                except Exception as e:
                    logger.warning("STATE_SAVE skip position %s: %s", key, e)

            # Daily stats
            ds = risk_manager.daily_stats
            state = PersistedState(
                current_bankroll=risk_manager.current_bankroll,
                peak_bankroll=risk_manager.peak_bankroll,
                starting_bankroll=risk_manager.starting_bankroll,
                consecutive_losses=risk_manager.consecutive_losses,
                trade_history=list(risk_manager.trade_history),
                daily_date=ds.date if ds else "",
                daily_pnl=ds.total_pnl if ds else 0.0,
                daily_trades=ds.trades_count if ds else 0,
                daily_wins=ds.wins if ds else 0,
                daily_losses=ds.losses if ds else 0,
                daily_fees=ds.fees_paid if ds else 0.0,
                daily_rebates=ds.rebates_earned if ds else 0.0,
                open_positions=positions,
                saved_at=datetime.now(timezone.utc).isoformat(),
            )

            data = asdict(state)

            # Atomic write: tmp file -> fsync -> rename
            tmp_path = self.state_file + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.state_file)

            logger.info(
                "STATE_SAVED bankroll=%.2f peak=%.2f positions=%d file=%s",
                state.current_bankroll,
                state.peak_bankroll,
                len(positions),
                self.state_file,
            )
            return True

        except Exception as e:
            logger.error("STATE_SAVE_FAILED: %s", e)
            return False

    def load(self) -> Optional[PersistedState]:
        """
        Load persisted state from disk.

        Returns PersistedState on success, None if file missing or corrupt.
        """
        if not os.path.exists(self.state_file):
            logger.info("STATE_LOAD no saved state at %s (first run)", self.state_file)
            return None

        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)

            state = PersistedState(
                current_bankroll=data["current_bankroll"],
                peak_bankroll=data["peak_bankroll"],
                starting_bankroll=data.get("starting_bankroll", data["current_bankroll"]),
                consecutive_losses=data.get("consecutive_losses", 0),
                trade_history=data.get("trade_history", []),
                daily_date=data.get("daily_date", ""),
                daily_pnl=data.get("daily_pnl", 0.0),
                daily_trades=data.get("daily_trades", 0),
                daily_wins=data.get("daily_wins", 0),
                daily_losses=data.get("daily_losses", 0),
                daily_fees=data.get("daily_fees", 0.0),
                daily_rebates=data.get("daily_rebates", 0.0),
                open_positions=data.get("open_positions", []),
                saved_at=data.get("saved_at", ""),
                version=data.get("version", 1),
            )

            logger.info(
                "STATE_LOADED bankroll=%.2f peak=%.2f positions=%d saved_at=%s",
                state.current_bankroll,
                state.peak_bankroll,
                len(state.open_positions),
                state.saved_at,
            )
            return state

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("STATE_LOAD_CORRUPT file=%s error=%s (using defaults)", self.state_file, e)
            return None
        except Exception as e:
            logger.error("STATE_LOAD_FAILED file=%s error=%s", self.state_file, e)
            return None

    def restore_risk_manager(self, rm, state: PersistedState):
        """
        Apply saved state to an already-initialized risk manager.

        Called AFTER rm.initialize() so defaults are set, then overwritten.
        Peak is set to max(saved_peak, current_bankroll) to prevent false drawdown.
        """
        from .models import DailyStats

        rm.current_bankroll = state.current_bankroll
        rm.peak_bankroll = max(state.peak_bankroll, state.current_bankroll)
        rm.starting_bankroll = state.starting_bankroll
        rm.consecutive_losses = state.consecutive_losses
        rm.trade_history = list(state.trade_history)

        # Restore daily stats if same day, otherwise let daily reset handle it
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if state.daily_date == today:
            rm.daily_stats = DailyStats(
                date=today,
                starting_bankroll=state.starting_bankroll,
                current_bankroll=state.current_bankroll,
            )
            rm.daily_stats.total_pnl = state.daily_pnl
            rm.daily_stats.trades_count = state.daily_trades
            rm.daily_stats.wins = state.daily_wins
            rm.daily_stats.losses = state.daily_losses
            rm.daily_stats.fees_paid = state.daily_fees
            rm.daily_stats.rebates_earned = state.daily_rebates
        else:
            # New day — start fresh daily stats but keep bankroll
            rm.daily_stats = DailyStats(
                date=today,
                starting_bankroll=state.current_bankroll,
                current_bankroll=state.current_bankroll,
            )
            rm._last_daily_date = today

        # Restore positions
        for pos_data in state.open_positions:
            try:
                position = _deserialize_position(pos_data)
                cid = pos_data["condition_id"]
                rm.positions[cid] = position
                # Deduct cost from bankroll (it was included in current_bankroll
                # but positions are tracked separately)
                # Actually: current_bankroll already reflects the cash AFTER
                # opening these positions, so no adjustment needed.
            except Exception as e:
                logger.warning("STATE_RESTORE skip position: %s", e)

        logger.info(
            "STATE_RESTORE_RM bankroll=%.2f peak=%.2f positions=%d consecutive_losses=%d",
            rm.current_bankroll,
            rm.peak_bankroll,
            len(rm.positions),
            rm.consecutive_losses,
        )

    def restore_paper_executor(self, executor, state: PersistedState):
        """Set paper executor balance to match restored bankroll."""
        executor.account.balance = state.current_bankroll
        logger.info(
            "STATE_RESTORE_PAPER balance=%.2f",
            executor.account.balance,
        )
