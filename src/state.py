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
    peak_decay_started_at: str = ""  # ISO timestamp if peak decay is active, empty if not

    # Daily stats snapshot
    daily_date: str = ""
    daily_pnl: float = 0.0
    daily_trades: int = 0
    daily_wins: int = 0
    daily_losses: int = 0
    daily_fees: float = 0.0
    daily_rebates: float = 0.0
    daily_starting_bankroll: float = 0.0  # bankroll at start of day (for loss limit)

    # Open positions (minimal dicts for reconstruction)
    open_positions: list = field(default_factory=list)

    # Kill switch state
    kill_switch: dict = field(default_factory=dict)

    # Settled market IDs (prevent re-processing after restart)
    settled_markets: list = field(default_factory=list)

    # Metadata
    execution_mode: str = ""  # "PAPER", "DRY", or "LIVE" — detects mode switches
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
        "variant": getattr(market, "variant", "fifteen"),
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
        variant=data.get("variant", "fifteen"),
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

    def save(self, risk_manager, markets: Optional[dict] = None, kill_switch=None, execution_mode: str = "", settled_markets: Optional[set] = None) -> bool:
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
            ks_dict = kill_switch.to_dict() if kill_switch else {}

            # Persist peak decay timestamp if active
            decay_ts = ""
            if risk_manager._peak_decay_started_at is not None:
                decay_ts = risk_manager._peak_decay_started_at.isoformat()

            state = PersistedState(
                current_bankroll=risk_manager.current_bankroll,
                peak_bankroll=risk_manager.peak_bankroll,
                starting_bankroll=risk_manager.starting_bankroll,
                consecutive_losses=risk_manager.consecutive_losses,
                trade_history=list(risk_manager.trade_history),
                peak_decay_started_at=decay_ts,
                daily_date=ds.date if ds else "",
                daily_pnl=ds.total_pnl if ds else 0.0,
                daily_trades=ds.trades_count if ds else 0,
                daily_wins=ds.wins if ds else 0,
                daily_losses=ds.losses if ds else 0,
                daily_fees=ds.fees_paid if ds else 0.0,
                daily_rebates=ds.rebates_earned if ds else 0.0,
                daily_starting_bankroll=ds.starting_bankroll if ds else risk_manager.current_bankroll,
                open_positions=positions,
                kill_switch=ks_dict,
                settled_markets=list(settled_markets)[-100:] if settled_markets else [],
                execution_mode=execution_mode,
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
                peak_decay_started_at=data.get("peak_decay_started_at", ""),
                daily_date=data.get("daily_date", ""),
                daily_pnl=data.get("daily_pnl", 0.0),
                daily_trades=data.get("daily_trades", 0),
                daily_wins=data.get("daily_wins", 0),
                daily_losses=data.get("daily_losses", 0),
                daily_fees=data.get("daily_fees", 0.0),
                daily_rebates=data.get("daily_rebates", 0.0),
                daily_starting_bankroll=data.get("daily_starting_bankroll", data.get("current_bankroll", 0.0)),
                open_positions=data.get("open_positions", []),
                kill_switch=data.get("kill_switch", {}),
                settled_markets=data.get("settled_markets", []),
                execution_mode=data.get("execution_mode", ""),
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

    def restore_risk_manager(self, rm, state: PersistedState, current_mode: str = ""):
        """
        Apply saved state to an already-initialized risk manager.

        Called AFTER rm.initialize() so defaults are set, then overwritten.
        Peak is set to max(saved_peak, current_bankroll) to prevent false drawdown.

        If execution mode has changed (e.g. PAPER → LIVE), skip restoring
        bankroll/peak to avoid stale paper values triggering false drawdown
        on a live account with a different balance.
        """
        from .models import DailyStats

        mode_changed = (
            state.execution_mode
            and current_mode
            and state.execution_mode != current_mode
        )

        # Heuristic: if saved state has no execution_mode (old format), detect
        # likely mode switch by comparing fresh balance vs saved peak.
        # A 10x+ divergence between fresh balance and saved peak strongly
        # suggests a paper→live or live→paper switch.
        if not mode_changed and not state.execution_mode and current_mode:
            fresh_balance = rm.current_bankroll  # set by initialize() before this call
            if fresh_balance > 0 and state.peak_bankroll > 0:
                ratio = max(fresh_balance, state.peak_bankroll) / min(fresh_balance, state.peak_bankroll)
                if ratio >= 10.0:
                    logger.warning(
                        "STATE_BALANCE_MISMATCH fresh_balance=%.2f saved_peak=%.2f ratio=%.1fx — "
                        "possible mode switch (saved state has no execution_mode). "
                        "Treating as mode switch to prevent false drawdown.",
                        fresh_balance, state.peak_bankroll, ratio,
                    )
                    mode_changed = True

        if mode_changed:
            logger.warning(
                "STATE_MODE_SWITCH saved_mode=%s current_mode=%s — "
                "skipping bankroll/peak restore (using fresh balance %.2f)",
                state.execution_mode,
                current_mode,
                rm.current_bankroll,
            )
            # Keep rm.current_bankroll and rm.peak_bankroll as set by initialize()
            # Only restore non-financial state that's still useful
            rm.consecutive_losses = 0
            rm.trade_history = []
        else:
            rm.current_bankroll = state.current_bankroll
            rm.peak_bankroll = max(state.peak_bankroll, state.current_bankroll)
            rm.starting_bankroll = state.starting_bankroll
            rm.consecutive_losses = state.consecutive_losses
            rm.trade_history = list(state.trade_history)

        # Restore daily stats if same day and same mode, otherwise start fresh
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if state.daily_date == today and not mode_changed:
            # Use the day's actual starting bankroll, not the original starting_bankroll
            day_start = state.daily_starting_bankroll or state.current_bankroll
            rm.daily_stats = DailyStats(
                date=today,
                starting_bankroll=day_start,
                current_bankroll=state.current_bankroll,
            )
            rm.daily_stats.total_pnl = state.daily_pnl
            rm.daily_stats.trades_count = state.daily_trades
            rm.daily_stats.wins = state.daily_wins
            rm.daily_stats.losses = state.daily_losses
            rm.daily_stats.fees_paid = state.daily_fees
            rm.daily_stats.rebates_earned = state.daily_rebates
        else:
            # New day or mode switch — start fresh daily stats
            rm.daily_stats = DailyStats(
                date=today,
                starting_bankroll=rm.current_bankroll,
                current_bankroll=rm.current_bankroll,
            )
            rm._last_daily_date = today

        # Restore peak decay state (skip on mode switch — not relevant)
        if state.peak_decay_started_at and not mode_changed:
            try:
                rm._peak_decay_started_at = datetime.fromisoformat(state.peak_decay_started_at)
                logger.info("STATE_RESTORE peak_decay active since %s", state.peak_decay_started_at)
            except (ValueError, TypeError):
                rm._peak_decay_started_at = None

        # Restore positions (skip on mode switch — paper positions don't exist on-chain)
        if not mode_changed:
            for pos_data in state.open_positions:
                try:
                    position = _deserialize_position(pos_data)
                    cid = pos_data["condition_id"]
                    rm.positions[cid] = position
                except Exception as e:
                    logger.warning("STATE_RESTORE skip position: %s", e)
        elif state.open_positions:
            logger.warning(
                "STATE_MODE_SWITCH dropping %d positions from %s mode",
                len(state.open_positions),
                state.execution_mode,
            )

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
