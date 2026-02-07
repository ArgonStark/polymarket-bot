"""Execution layer for order placement and trading client."""

from .client import create_trading_client
from .orders import (
    OrderExecutor,
    OrderSafetyGuard,
    OrderSafetyConfig,
    get_safety_guard,
    reset_safety_guard,
)
from .paper import PaperOrderExecutor

__all__ = [
    "create_trading_client",
    "OrderExecutor",
    "OrderSafetyGuard",
    "OrderSafetyConfig",
    "get_safety_guard",
    "reset_safety_guard",
    "PaperOrderExecutor",
]
