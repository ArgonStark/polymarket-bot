"""Base strategy interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from .context import StrategyContext


@dataclass
class StrategySignal:
    """Signal emitted by a strategy."""

    side: str  # "YES" or "NO"
    edge: float
    confidence: float


class Strategy(Protocol):
    """Strategy protocol."""

    def generate(self, ctx: StrategyContext) -> Optional[StrategySignal]:
        ...
