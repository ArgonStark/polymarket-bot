"""Bot package wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_module_path = Path(__file__).resolve().parent.parent / "bot.py"
_spec = importlib.util.spec_from_file_location("src.bot_legacy", _module_path)
_module = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_module)

TradingBot = _module.TradingBot
main = _module.main

__all__ = ["TradingBot", "main"]
