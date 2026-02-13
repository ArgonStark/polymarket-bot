"""
Trade attribution — track decision source, veto reasons, and context.

Each trade record gets:
  * source: ML_EV | RULES | HYBRID
  * veto_reason: why it was blocked (if applicable)
  * decision_context: min_edge, p_up, prices, time_remaining, etc.

Also tracks blocked-decision records for veto analysis.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DecisionRecord:
    """Immutable record of a trading decision (executed or vetoed)."""

    timestamp: str  # ISO 8601
    asset: str
    market_id: str
    side: str
    source: str  # ML_EV, RULES, HYBRID
    executed: bool
    veto_reason: str = ""
    decision_context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "asset": self.asset,
            "market_id": self.market_id,
            "side": self.side,
            "source": self.source,
            "executed": self.executed,
            "veto_reason": self.veto_reason,
            "decision_context": self.decision_context,
        }


class AttributionTracker:
    """Tracks trade attribution and veto history for analysis."""

    def __init__(self, max_history: int = 200):
        self._decisions: list[DecisionRecord] = []
        self._max_history = max_history

    def record_execution(
        self,
        asset: str,
        market_id: str,
        side: str,
        source: str,
        decision_context: Optional[dict] = None,
    ) -> DecisionRecord:
        """Record a trade that was executed."""
        rec = DecisionRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            asset=asset,
            market_id=market_id,
            side=side,
            source=source,
            executed=True,
            decision_context=decision_context or {},
        )
        self._append(rec)
        logger.info(
            "TRADE_ATTRIBUTION asset=%s side=%s source=%s market=%s",
            asset, side, source, market_id[:8],
        )
        return rec

    def record_veto(
        self,
        asset: str,
        market_id: str,
        side: str,
        source: str,
        veto_reason: str,
        decision_context: Optional[dict] = None,
    ) -> DecisionRecord:
        """Record a trade that was vetoed/blocked."""
        rec = DecisionRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            asset=asset,
            market_id=market_id,
            side=side,
            source=source,
            executed=False,
            veto_reason=veto_reason,
            decision_context=decision_context or {},
        )
        self._append(rec)
        logger.debug(
            "TRADE_VETO asset=%s side=%s source=%s reason=%s",
            asset, side, source, veto_reason,
        )
        return rec

    def get_summary(self) -> dict:
        """
        Return breakdown counts for Trade History Review.

        Returns:
            {
              "executed_by_source": {"ML_EV": 10, "RULES": 5, ...},
              "blocked_by_reason": {"chart_bearish_vs_up": 3, ...},  # top 5
              "total_executed": 15,
              "total_blocked": 20,
            }
        """
        executed_by_source: dict[str, int] = {}
        blocked_by_reason: dict[str, int] = {}

        for rec in self._decisions:
            if rec.executed:
                executed_by_source[rec.source] = executed_by_source.get(rec.source, 0) + 1
            else:
                reason = rec.veto_reason or "unknown"
                blocked_by_reason[reason] = blocked_by_reason.get(reason, 0) + 1

        # Top 5 veto reasons
        top_vetoes = dict(
            sorted(blocked_by_reason.items(), key=lambda x: x[1], reverse=True)[:5]
        )

        return {
            "executed_by_source": executed_by_source,
            "blocked_by_reason": top_vetoes,
            "total_executed": sum(executed_by_source.values()),
            "total_blocked": sum(blocked_by_reason.values()),
        }

    def get_recent_decisions(self, n: int = 20) -> list[dict]:
        """Return last *n* decisions as dicts."""
        return [d.to_dict() for d in self._decisions[-n:]]

    def _append(self, rec: DecisionRecord):
        self._decisions.append(rec)
        if len(self._decisions) > self._max_history:
            self._decisions = self._decisions[-self._max_history:]


def determine_source(ml_engine_active: bool, ml_confidence: Optional[float]) -> str:
    """
    Determine trade source label.

    - ML_EV:  ML engine is active and provided confidence
    - RULES:  No ML, pure signal-generator rules
    - HYBRID: ML active but no confidence for this signal (fallback path)
    """
    if not ml_engine_active:
        return "RULES"
    if ml_confidence is not None:
        return "ML_EV"
    return "HYBRID"
