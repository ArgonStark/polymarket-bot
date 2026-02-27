"""
Kill switch — hard safety gate that disables new order placement.

Triggers:
  1. Max drawdown from peak_bankroll exceeds KILL_DD_PCT (default: MAX_DRAWDOWN_PCT)
  2. Daily loss exceeds config.trading.daily_loss_limit
  3. N consecutive losses exceeds KILL_CONSEC_LOSSES (default 5)
  4. Manual file flag: presence of ``kill.switch`` file in project root
  5. Env var KILL_SWITCH=1 (checked on every evaluation)

When tripped the bot continues market subscriptions, settlement, and state
saves — only new order submission is blocked.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_FLAG_PATH = os.path.join(_PROJECT_ROOT, "kill.switch")


@dataclass
class KillSwitchState:
    """Serialisable kill-switch snapshot for bot_state.json."""
    active: bool = False
    reason: str = ""
    triggered_at: str = ""  # ISO timestamp


@dataclass
class KillSwitch:
    """Hard safety gate — blocks all new order placement when tripped."""

    # Configurable thresholds (env var overrides constructor arg)
    dd_pct: float = 0.0  # Set by __post_init__; 0.0 means "use config default"
    consec_losses: int = field(default_factory=lambda: int(os.getenv("KILL_CONSEC_LOSSES", "5")))
    flag_path: str = field(default_factory=lambda: os.getenv("KILL_SWITCH_FILE", _DEFAULT_FLAG_PATH))

    # Internal state
    active: bool = field(default=False, init=False)
    reason: str = field(default="", init=False)
    triggered_at: Optional[datetime] = field(default=None, init=False)
    _last_log_ts: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        # Precedence: KILL_DD_PCT env var > constructor arg > MAX_DRAWDOWN_PCT config
        env_val = os.getenv("KILL_DD_PCT")
        if env_val is not None:
            self.dd_pct = float(env_val)
        elif self.dd_pct <= 0:
            # No explicit value — fall back to config's MAX_DRAWDOWN_PCT
            self.dd_pct = float(os.getenv("MAX_DRAWDOWN_PCT", "0.40"))

    # ── public API ──────────────────────────────────────────────

    def evaluate(
        self,
        peak_bankroll: float,
        current_equity: float,
        daily_loss_hit: bool,
        consecutive_losses: int,
    ) -> bool:
        """
        Check all triggers and activate kill switch if any fire.

        Returns True when the kill switch is (or becomes) active.
        """
        if self.active:
            # Auto-recover from drawdown kills when drawdown falls below limit
            if self.reason.startswith("drawdown") and peak_bankroll > 0:
                dd = (peak_bankroll - current_equity) / peak_bankroll
                if dd < self.dd_pct:
                    self.reset(reason=f"drawdown recovered to {dd:.1%}")
                    return False

            # Auto-recover from daily loss kills when daily loss clears
            # (e.g. new day started, or starting_bankroll was corrected)
            if self.reason == "daily loss limit exceeded" and not daily_loss_hit:
                self.reset(reason="daily loss recovered")
                return False

            # Auto-recover from consecutive losses when streak resets
            if "consecutive losses" in self.reason and consecutive_losses < self.consec_losses:
                self.reset(reason=f"consecutive losses recovered to {consecutive_losses}")
                return False

            return True

        # 1. Drawdown
        if peak_bankroll > 0:
            dd = (peak_bankroll - current_equity) / peak_bankroll
            if dd >= self.dd_pct:
                self._trip(f"drawdown {dd:.1%} >= {self.dd_pct:.0%} limit")
                return True

        # 2. Daily loss
        if daily_loss_hit:
            self._trip("daily loss limit exceeded")
            return True

        # 3. Consecutive losses
        if consecutive_losses >= self.consec_losses:
            self._trip(f"{consecutive_losses} consecutive losses >= {self.consec_losses} limit")
            return True

        # 4. File flag
        if os.path.exists(self.flag_path):
            self._trip(f"manual file flag ({self.flag_path})")
            return True

        # 5. Env var
        if os.getenv("KILL_SWITCH", "0") == "1":
            self._trip("KILL_SWITCH=1 env var")
            return True

        return False

    def is_active(self) -> bool:
        """Return whether the kill switch is currently engaged."""
        return self.active

    def should_block_order(self) -> tuple[bool, str]:
        """
        Gate for order submission paths.

        Returns (blocked, reason).  Callers should log the reason
        and skip the order.
        """
        if not self.active:
            return False, ""
        # Throttle repeated KILL_SWITCH_ACTIVE log to once per 15 s
        now = time.time()
        if now - self._last_log_ts >= 15.0:
            self._last_log_ts = now
            logger.warning("KILL_SWITCH_ACTIVE reason=%s", self.reason)
        return True, f"kill_switch:{self.reason}"

    def reset(self, reason: str = "manual"):
        """Deactivate the kill switch (e.g. after manual review)."""
        if not self.active:
            return
        logger.info(
            "KILL_SWITCH_RESET previous_reason=%s reset_reason=%s",
            self.reason, reason,
        )
        self.active = False
        self.reason = ""
        self.triggered_at = None

    # ── serialisation helpers (for BotStateManager) ─────────────

    def to_dict(self) -> dict:
        return {
            "active": self.active,
            "reason": self.reason,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else "",
        }

    @classmethod
    def from_dict(cls, data: dict, **kwargs) -> "KillSwitch":
        ks = cls(**kwargs)
        if data.get("active"):
            ks.active = True
            ks.reason = data.get("reason", "restored from state")
            ts = data.get("triggered_at", "")
            if ts:
                try:
                    ks.triggered_at = datetime.fromisoformat(ts)
                except (ValueError, TypeError):
                    ks.triggered_at = datetime.now(timezone.utc)
            else:
                ks.triggered_at = datetime.now(timezone.utc)
            logger.warning(
                "KILL_SWITCH_RESTORED reason=%s triggered_at=%s",
                ks.reason, ks.triggered_at.isoformat(),
            )
        return ks

    # ── private ─────────────────────────────────────────────────

    def _trip(self, reason: str):
        self.active = True
        self.reason = reason
        self.triggered_at = datetime.now(timezone.utc)
        logger.critical(
            "KILL_SWITCH_TRIPPED reason=%s timestamp=%s",
            reason, self.triggered_at.isoformat(),
        )
