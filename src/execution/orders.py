"""
Order execution module for Polymarket CLOB.

Handles order creation, placement, and management with
support for maker (rebate) and taker orders.

Safety Features:
- One open order per asset+market+side
- Per-asset cooldown (default 10s)
- Global max orders per minute (default 10)
- Exposure cap enforced BEFORE submission
"""

import logging
import traceback
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, Set, Tuple, Deque
import asyncio
import threading

from py_clob_client_v2.client import ClobClient
from py_clob_client_v2.clob_types import OrderArgs, MarketOrderArgs, OrderPayload, OrderType
from py_clob_client_v2.order_builder.constants import BUY, SELL

from ..models import Order, OrderStatus, TradeResult, Signal, OrderAction
from ..config import BotConfig


logger = logging.getLogger(__name__)


# ==============================================================================
# ORDER SAFETY GUARD - Prevents order spam and enforces safety limits
# ==============================================================================

@dataclass
class OrderSafetyConfig:
    """Configuration for order safety limits."""
    per_asset_cooldown_sec: float = 10.0  # Minimum seconds between orders per asset
    global_max_orders_per_minute: int = 10  # Maximum orders across all assets per minute
    max_exposure_pct: float = 0.25  # Maximum exposure as fraction of bankroll
    enabled: bool = True  # Master switch to enable/disable safety checks


class OrderSafetyGuard:
    """
    Enforces safety limits on order submission to prevent spam and runaway losses.

    Tracks:
    - Open orders by (asset, market_id, side) - only one allowed
    - Per-asset cooldowns
    - Global order rate (rolling 60-second window)
    - Total exposure vs bankroll
    """

    def __init__(self, config: Optional[OrderSafetyConfig] = None):
        self.config = config or OrderSafetyConfig()
        self._lock = threading.Lock()

        # Track open orders: key = (asset, market_id, side) -> order_id
        self._open_orders: Dict[Tuple[str, str, str], str] = {}

        # Track last order time per asset
        self._last_order_time: Dict[str, float] = {}

        # Rolling window of order timestamps (last 60 seconds)
        self._order_timestamps: Deque[float] = deque()

        # Pending order count for paper trading
        self._pending_count: int = 0

        # Active market per asset:variant - for stale market detection
        # Key: "asset:variant" (e.g. "BTC:five", "BTC:fifteen"), Value: condition_id
        self._active_market_per_asset: Dict[str, str] = {}

    def check_can_submit(
        self,
        signal: Signal,
        current_exposure: float,
        bankroll: float,
    ) -> Tuple[bool, str]:
        """
        Check if an order can be submitted based on safety limits.

        Args:
            signal: The trading signal to check
            current_exposure: Current total exposure in USD
            bankroll: Current bankroll in USD

        Returns:
            Tuple of (can_submit, block_reason)
        """
        if not self.config.enabled:
            return True, ""

        # Defensive extraction with defaults for None values
        asset = getattr(signal.market, 'asset', 'UNKNOWN') if signal and signal.market else 'UNKNOWN'
        market_id = getattr(signal.market, 'condition_id', '????????') if signal and signal.market else '????????'
        side = signal.side.value if signal and signal.side else 'UNKNOWN'
        signal_size_usd = float(signal.size_usd) if signal and signal.size_usd is not None else 0.0

        order_key = (asset, market_id, side)
        now = time.time()

        # Ensure current_exposure and bankroll are valid floats
        current_exposure = float(current_exposure) if current_exposure is not None else 0.0
        bankroll = float(bankroll) if bankroll is not None else 0.0

        with self._lock:
            # 0. FIRST: Check for stale market (order for old market_id after transition)
            # Key by asset:variant so 5m and 15m markets don't conflict
            variant = getattr(signal.market, 'variant', 'fifteen') if signal and signal.market else 'fifteen'
            av_key = f"{asset}:{variant}"
            active_market = self._active_market_per_asset.get(av_key)
            if active_market and market_id != active_market:
                return False, f"stale_market:{market_id[:8]}!=active:{active_market[:8]}"

            # 1. Check for existing open order on same asset+market+side
            if order_key in self._open_orders:
                existing_order_id = self._open_orders[order_key]
                return False, f"duplicate_order:{existing_order_id[:8]}"

            # 2. Check per-asset cooldown
            last_time = self._last_order_time.get(asset, 0.0)
            if last_time is None:
                last_time = 0.0
            elapsed = now - last_time
            if elapsed < self.config.per_asset_cooldown_sec:
                remaining = self.config.per_asset_cooldown_sec - elapsed
                return False, f"cooldown:{remaining:.1f}s_remaining"

            # 3. Check global rate limit (orders per minute)
            self._prune_old_timestamps(now)
            if len(self._order_timestamps) >= self.config.global_max_orders_per_minute:
                return False, f"rate_limit:{len(self._order_timestamps)}/min"

            # 4. Check exposure cap
            new_exposure = current_exposure + signal_size_usd
            max_exposure = bankroll * self.config.max_exposure_pct
            if new_exposure > max_exposure and bankroll > 0:
                return False, f"exposure_cap:${new_exposure:.2f}>${max_exposure:.2f}"

        return True, ""

    def record_order_submitted(
        self,
        signal: Signal,
        order_id: str,
    ) -> None:
        """Record that an order was successfully submitted."""
        asset = signal.market.asset
        market_id = signal.market.condition_id
        side = signal.side.value
        order_key = (asset, market_id, side)
        now = time.time()

        with self._lock:
            self._open_orders[order_key] = order_id
            self._last_order_time[asset] = now
            self._order_timestamps.append(now)
            self._pending_count += 1

    def record_order_filled(
        self,
        asset: str,
        market_id: str,
        side: str,
        order_id: str,
    ) -> None:
        """Record that an order was filled (remove from open orders)."""
        order_key = (asset, market_id, side)
        with self._lock:
            if order_key in self._open_orders:
                if self._open_orders[order_key] == order_id:
                    del self._open_orders[order_key]
            if self._pending_count > 0:
                self._pending_count -= 1

    def record_order_cancelled(
        self,
        asset: str,
        market_id: str,
        side: str,
        order_id: str,
    ) -> None:
        """Record that an order was cancelled (remove from open orders)."""
        # Same logic as filled
        self.record_order_filled(asset, market_id, side, order_id)

    def clear_orders_for_asset(self, asset: str) -> None:
        """Clear all tracked orders for an asset (e.g., on market settlement)."""
        with self._lock:
            keys_to_remove = [k for k in self._open_orders if k[0] == asset]
            for key in keys_to_remove:
                del self._open_orders[key]
                if self._pending_count > 0:
                    self._pending_count -= 1

    def set_active_market(self, asset: str, market_id: str, variant: str = "fifteen") -> Optional[str]:
        """
        Set the active market for an asset:variant pair.

        Returns the old market_id if there was one (for stale order cancellation).

        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP)
            market_id: condition_id of the new active market
            variant: Market variant ("five" or "fifteen")

        Returns:
            Old market_id if there was a transition, None otherwise
        """
        av_key = f"{asset}:{variant}"
        with self._lock:
            old_market_id = self._active_market_per_asset.get(av_key)
            self._active_market_per_asset[av_key] = market_id
            if old_market_id and old_market_id != market_id:
                return old_market_id
            return None

    def get_active_market(self, asset: str, variant: str = "fifteen") -> Optional[str]:
        """Get the active market_id for an asset:variant pair."""
        av_key = f"{asset}:{variant}"
        with self._lock:
            return self._active_market_per_asset.get(av_key)

    def clear_active_market(self, asset: str, variant: str = "fifteen") -> None:
        """Clear the active market for an asset:variant pair (e.g., on settlement)."""
        av_key = f"{asset}:{variant}"
        with self._lock:
            self._active_market_per_asset.pop(av_key, None)

    def get_pending_count(self) -> int:
        """Get current pending order count."""
        with self._lock:
            return self._pending_count

    def reset_pending_count(self, count: int = 0) -> None:
        """Reset pending count (for sync with external state)."""
        with self._lock:
            self._pending_count = count

    def _prune_old_timestamps(self, now: float) -> None:
        """Remove timestamps older than 60 seconds."""
        cutoff = now - 60.0
        while self._order_timestamps and self._order_timestamps[0] < cutoff:
            self._order_timestamps.popleft()

    def get_stats(self) -> Dict:
        """Get current safety guard statistics."""
        now = time.time()
        with self._lock:
            self._prune_old_timestamps(now)
            return {
                "open_orders": len(self._open_orders),
                "pending_count": self._pending_count,
                "orders_last_minute": len(self._order_timestamps),
                "assets_on_cooldown": len(self._last_order_time),
                "active_markets": dict(self._active_market_per_asset),
            }


# Global safety guard instance (can be shared across executors)
_global_safety_guard: Optional[OrderSafetyGuard] = None


def get_safety_guard(config: Optional[OrderSafetyConfig] = None) -> OrderSafetyGuard:
    """Get or create the global safety guard instance."""
    global _global_safety_guard
    if _global_safety_guard is None:
        _global_safety_guard = OrderSafetyGuard(config)
    return _global_safety_guard


def reset_safety_guard(config: Optional[OrderSafetyConfig] = None) -> OrderSafetyGuard:
    """Reset the global safety guard (for testing or reconfiguration)."""
    global _global_safety_guard
    _global_safety_guard = OrderSafetyGuard(config)
    return _global_safety_guard


def _extract_error_details(e: Exception) -> str:
    """Extract detailed error information from an exception."""
    details = [str(e)]

    # Try to get more info from common exception attributes
    if hasattr(e, 'response'):
        resp = e.response
        if hasattr(resp, 'status_code'):
            details.append(f"status={resp.status_code}")
        if hasattr(resp, 'text'):
            try:
                details.append(f"body={resp.text[:200]}")
            except:
                pass

    if hasattr(e, 'status_code') and e.status_code:
        details.append(f"status={e.status_code}")

    if hasattr(e, 'error_message') and e.error_message:
        details.append(f"msg={e.error_message}")

    # Get the underlying cause if it's a chained exception
    if e.__cause__:
        details.append(f"cause={e.__cause__}")

    return " | ".join(details)


def _validate_order_response(response: dict) -> tuple[bool, str]:
    """
    Validate order response from the API.

    Args:
        response: API response dict

    Returns:
        Tuple of (is_valid, error_message)
    """
    if response is None:
        return False, "Empty response from API"

    if not isinstance(response, dict):
        return False, f"Invalid response type: {type(response)}"

    # Check for success first - if success=True, order went through
    if response.get("success") is True:
        order_id = response.get("orderID") or response.get("order_id")
        if order_id:
            return True, ""

    # Check for error indicators - handle empty strings and nested errors
    if "error" in response:
        error = response["error"]
        if isinstance(error, dict):
            error_msg = error.get("message") or error.get("error") or str(error)
        elif error:  # Non-empty error string
            error_msg = str(error)
        else:
            # Empty error string with no success flag - check for order ID
            order_id = response.get("orderID") or response.get("order_id")
            if order_id:
                return True, ""
            return False, f"API error (response: {response})"
        return False, error_msg

    if "errorMsg" in response:
        error_msg = response["errorMsg"]
        if error_msg:  # Only fail if errorMsg is non-empty
            return False, error_msg
        # Empty errorMsg - check if order was placed successfully
        order_id = response.get("orderID") or response.get("order_id")
        if order_id:
            return True, ""

    if response.get("status") == "error":
        return False, response.get("message", "Unknown error")

    # Check for required fields
    order_id = response.get("orderID") or response.get("order_id")
    if not order_id:
        logger.debug(f"Order response without ID: {response}")
        return False, "No order ID in response"

    return True, ""


@dataclass
class OrderExecutor:
    """
    Handles order execution for the trading bot.

    Supports:
    - POST_ONLY: Maker orders that earn rebates
    - GTC: Good-till-cancelled limit orders
    - FOK: Fill-or-kill market orders
    """

    client: ClobClient
    config: BotConfig

    def place_maker_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size_shares: float,
        post_only: bool = True,
    ) -> TradeResult:
        """
        Place a maker order to earn rebates.

        Maker orders are placed at or below the best bid (for buys)
        or at or above the best ask (for sells) to ensure they
        rest on the book and earn maker rebates.

        Args:
            token_id: Token ID to trade
            side: "BUY" or "SELL"
            price: Limit price (0.01 - 0.99)
            size_shares: Number of shares
            post_only: If True, order will be rejected if it would take

        Returns:
            TradeResult with execution details
        """
        if self.config.dry_run:
            logger.info(
                f"[DRY RUN] MAKER {side} {size_shares:.2f} shares "
                f"@ {price:.4f} for {token_id[:16]}..."
            )
            return TradeResult(
                success=True,
                order_id="dry_run_order",
                filled_size=0.0,
                filled_price=price,
            )

        try:
            # Validate inputs
            price = max(0.01, min(0.99, price))
            size_shares = max(0.01, size_shares)

            # Create order arguments
            order_args = OrderArgs(
                price=price,
                size=size_shares,
                side=BUY if side.upper() == "BUY" else SELL,
                token_id=token_id,
            )

            # Sign the order
            signed_order = self.client.create_order(order_args)

            # Post with post_only flag (pass directly to API, not as options dict)
            response = self.client.post_order(
                signed_order,
                OrderType.GTC,
                post_only=post_only,
            )

            # Validate response
            is_valid, error_msg = _validate_order_response(response)
            if not is_valid:
                return TradeResult(success=False, error_message=error_msg)

            order_id = response.get("orderID") or response.get("order_id", "")

            return TradeResult(
                success=True,
                order_id=order_id,
                filled_size=0.0,  # Maker orders don't fill immediately
                filled_price=price,
            )

        except Exception as e:
            logger.error(f"Failed to place maker order: {e}")
            return TradeResult(
                success=False,
                error_message=str(e),
            )

    def place_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size_shares: float,
    ) -> TradeResult:
        """
        Place a GTC limit order (may take or make).

        Standard limit order that may match immediately if
        price crosses the spread.

        Args:
            token_id: Token ID to trade
            side: "BUY" or "SELL"
            price: Limit price (0.01 - 0.99)
            size_shares: Number of shares

        Returns:
            TradeResult with execution details
        """
        if self.config.dry_run:
            logger.info(
                f"[DRY RUN] LIMIT {side} {size_shares:.2f} shares "
                f"@ {price:.4f} for {token_id[:16]}..."
            )
            return TradeResult(
                success=True,
                order_id="dry_run_order",
                filled_size=0.0,
                filled_price=price,
            )

        price = max(0.01, min(0.99, price))
        size_shares = max(0.01, size_shares)

        # Retry logic for network errors
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                order_args = OrderArgs(
                    price=price,
                    size=size_shares,
                    side=BUY if side.upper() == "BUY" else SELL,
                    token_id=token_id,
                )

                signed_order = self.client.create_order(order_args)
                response = self.client.post_order(signed_order, OrderType.GTC)

                # Log raw response for debugging
                logger.debug(f"Order API response: {response}")

                # Validate response
                is_valid, error_msg = _validate_order_response(response)
                if not is_valid:
                    logger.warning(f"Order rejected. Response: {response}")
                    return TradeResult(success=False, error_message=error_msg)

                order_id = response.get("orderID") or response.get("order_id", "")
                return TradeResult(
                    success=True,
                    order_id=order_id,
                    filled_size=0.0,
                    filled_price=price,
                )

            except Exception as e:
                last_error = e
                error_details = _extract_error_details(e)

                # Check if it's a network/request error worth retrying
                error_str = str(e).lower()
                is_network_error = any(x in error_str for x in [
                    "request exception", "timeout", "connection",
                    "network", "ssl", "socket"
                ])

                if is_network_error and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 2s, 4s backoff
                    logger.warning(
                        f"Order failed (attempt {attempt + 1}/{max_retries}): {error_details} | "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    # Log full details on final failure
                    logger.error(
                        f"Failed to place limit order: {error_details} | "
                        f"Token: {token_id[:16]}... | Price: {price} | Size: {size_shares}"
                    )
                    logger.debug(f"Full traceback: {traceback.format_exc()}")
                    break

        return TradeResult(
            success=False,
            error_message=_extract_error_details(last_error) if last_error else "Unknown error",
        )

    def place_market_order(
        self,
        token_id: str,
        side: str,
        size_usd: float,
    ) -> TradeResult:
        """
        Place a FOK market order for guaranteed fill.

        Fill-or-kill order that either fills completely
        at the current market price or fails entirely.

        Args:
            token_id: Token ID to trade
            side: "BUY" or "SELL"
            size_usd: Dollar amount to spend

        Returns:
            TradeResult with execution details
        """
        if self.config.dry_run:
            logger.info(
                f"[DRY RUN] MARKET {side} ${size_usd:.2f} "
                f"for {token_id[:16]}..."
            )
            return TradeResult(
                success=True,
                order_id="dry_run_order",
                filled_size=size_usd,  # Assume full fill in dry run
                filled_price=0.50,  # Placeholder
            )

        try:
            # Use MarketOrderArgs + create_market_order for proper amount rounding.
            # The SDK rounds maker_amount to 2 decimals (API requirement for market orders)
            # and taker_amount to 4 decimals automatically.
            price = 0.99 if side.upper() == "BUY" else 0.01

            market_order_args = MarketOrderArgs(
                token_id=token_id,
                amount=round(size_usd, 2),
                side=BUY if side.upper() == "BUY" else SELL,
                price=price,
                order_type=OrderType.FOK,
            )

            signed_order = self.client.create_market_order(market_order_args)
            response = self.client.post_order(signed_order, OrderType.FOK)

            # Validate response
            is_valid, error_msg = _validate_order_response(response)
            if not is_valid:
                logger.error(f"Market order rejected: {error_msg}")
                return TradeResult(success=False, error_message=error_msg)

            order_id = response.get("orderID") or response.get("order_id", "")

            # Extract fill information with safe parsing
            filled_size_raw = response.get("filledSize") or response.get("size_matched", 0)
            filled_price_raw = response.get("avgPrice") or response.get("price", price)
            try:
                filled_size = float(filled_size_raw) if filled_size_raw else 0.0
                filled_price = float(filled_price_raw) if filled_price_raw else price
            except (ValueError, TypeError):
                logger.warning(f"Failed to parse fill data: size={filled_size_raw}, price={filled_price_raw}")
                filled_size = 0.0
                filled_price = price

            # FOK orders are Fill-or-Kill: if the API accepted it, it filled.
            # Some responses omit filledSize — use intended amount as fallback.
            if filled_size == 0.0 and order_id:
                logger.warning(
                    f"FOK_FILL_FALLBACK: API returned success but no fill data. "
                    f"Using intended size=${size_usd:.2f} as filled_size."
                )
                # size_usd is dollars; shares = dollars / price
                filled_size = round(size_usd / filled_price, 4) if filled_price > 0 else 0.0
                filled_price = filled_price or price

            logger.info(
                f"MARKET ORDER executed: {side} ${size_usd:.2f} "
                f"filled {filled_size:.2f} @ {filled_price:.4f} "
                f"- Order ID: {order_id[:16]}..."
            )

            return TradeResult(
                success=filled_size > 0,
                order_id=order_id,
                filled_size=filled_size,
                filled_price=filled_price,
            )

        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            return TradeResult(
                success=False,
                error_message=str(e),
            )

    def execute_signal(
        self,
        signal: Signal,
        current_exposure: float = 0.0,
        bankroll: float = 0.0,
        safety_guard: Optional[OrderSafetyGuard] = None,
    ) -> TradeResult:
        """
        Execute a trading signal with safety checks and structured logging.

        Routes to appropriate order type based on signal's recommended action.
        Enforces safety limits via OrderSafetyGuard.

        Args:
            signal: Trading signal to execute
            current_exposure: Current total exposure in USD (for safety checks)
            bankroll: Current bankroll in USD (for safety checks)
            safety_guard: Optional safety guard instance (uses global if not provided)

        Returns:
            TradeResult with execution details
        """
        asset = signal.market.asset
        market_id = signal.market.condition_id
        side = signal.side.value
        order_type = signal.recommended_action.value if signal.recommended_action else "UNKNOWN"

        # Use provided safety guard or get global instance
        guard = safety_guard or get_safety_guard()

        if signal.recommended_action == OrderAction.SKIP:
            logger.info(
                "ORDER_BLOCK asset=%s market=%s side=%s reason=skip_action reasoning=%s",
                asset, market_id[:8], side, signal.reasoning[:50] if signal.reasoning else "none"
            )
            return TradeResult(success=False, error_message="Signal skipped")

        # Check for valid size
        if signal.size_shares <= 0 or signal.size_usd <= 0:
            logger.info(
                "ORDER_BLOCK asset=%s market=%s side=%s reason=invalid_size "
                "size_usd=%.2f size_shares=%.4f",
                asset, market_id[:8], side, signal.size_usd, signal.size_shares
            )
            return TradeResult(success=False, error_message="Invalid signal size")

        # Safety guard checks (exposure, cooldown, rate limit, duplicate)
        can_submit, block_reason = guard.check_can_submit(signal, current_exposure, bankroll)
        if not can_submit:
            logger.info(
                "ORDER_BLOCK asset=%s market=%s side=%s type=%s reason=%s "
                "size_usd=%.2f price=%.4f",
                asset, market_id[:8], side, order_type, block_reason,
                signal.size_usd, signal.recommended_price
            )
            return TradeResult(success=False, error_message=f"Safety block: {block_reason}")

        # Determine token to trade
        if signal.side.value == "UP":
            token_id = signal.market.up_token_id
        else:
            token_id = signal.market.down_token_id

        # Check that client exists for live trading
        if not self.config.dry_run and self.client is None:
            logger.info(
                "ORDER_BLOCK asset=%s market=%s side=%s reason=client_not_initialized",
                asset, market_id[:8], side
            )
            return TradeResult(success=False, error_message="Client not initialized")

        # Log ORDER_SUBMIT before attempting
        logger.info(
            "ORDER_SUBMIT asset=%s market=%s side=%s type=%s price=%.4f "
            "size_usd=%.2f size_shares=%.4f mode=%s",
            asset, market_id[:8], side, order_type, signal.recommended_price,
            signal.size_usd, signal.size_shares, "dry_run" if self.config.dry_run else "live"
        )

        # Execute the order
        result: TradeResult
        if signal.recommended_action == OrderAction.POST_ONLY:
            result = self.place_maker_order(
                token_id=token_id,
                side="BUY",
                price=signal.recommended_price,
                size_shares=signal.size_shares,
                post_only=True,
            )

        elif signal.recommended_action == OrderAction.LIMIT:
            result = self.place_limit_order(
                token_id=token_id,
                side="BUY",
                price=signal.recommended_price,
                size_shares=signal.size_shares,
            )

        elif signal.recommended_action == OrderAction.MARKET:
            result = self.place_market_order(
                token_id=token_id,
                side="BUY",
                size_usd=signal.size_usd,
            )

        else:
            logger.info(
                "ORDER_BLOCK asset=%s market=%s side=%s reason=unknown_action action=%s",
                asset, market_id[:8], side, signal.recommended_action
            )
            return TradeResult(
                success=False,
                error_message=f"Unknown action: {signal.recommended_action}",
            )

        # Log ORDER_RESULT after attempt
        if result.success:
            logger.info(
                "ORDER_RESULT asset=%s market=%s side=%s status=success order_id=%s "
                "filled_size=%.4f filled_price=%.4f",
                asset, market_id[:8], side, result.order_id[:16] if result.order_id else "none",
                result.filled_size, result.filled_price
            )
            # Record successful submission in safety guard
            if result.order_id and result.order_id != "dry_run_order":
                guard.record_order_submitted(signal, result.order_id)
        else:
            logger.info(
                "ORDER_RESULT asset=%s market=%s side=%s status=failed error=%s",
                asset, market_id[:8], side, result.error_message[:50] if result.error_message else "unknown"
            )

        return result

    async def execute_signal_async(
        self,
        signal: Signal,
        current_exposure: float = 0.0,
        bankroll: float = 0.0,
        safety_guard: Optional[OrderSafetyGuard] = None,
    ) -> TradeResult:
        """
        Async wrapper for execute_signal to avoid blocking the event loop.
        """
        return await asyncio.to_thread(
            self.execute_signal, signal, current_exposure, bankroll, safety_guard
        )

    def sell_position(
        self,
        token_id: str,
        shares: float,
        min_price: float = 0.01,
    ) -> TradeResult:
        """
        Sell an existing position (early exit).

        Used for take-profit and stop-loss exits.

        Args:
            token_id: Token ID to sell
            shares: Number of shares to sell
            min_price: Minimum acceptable price (for slippage control)

        Returns:
            TradeResult with execution details
        """
        if self.config.dry_run:
            logger.info(
                f"[DRY RUN] SELL {shares:.2f} shares "
                f"for {token_id[:16]}... (min price: {min_price:.4f})"
            )
            return TradeResult(
                success=True,
                order_id="dry_run_sell",
                filled_size=shares,
                filled_price=min_price,
            )

        try:
            # Use aggressive price to ensure fill (market sell)
            # We set a minimum but aim to fill quickly
            price = max(0.01, min_price)
            shares = max(0.01, shares)

            order_args = OrderArgs(
                price=price,
                size=shares,
                side=SELL,
                token_id=token_id,
            )

            signed_order = self.client.create_order(order_args)

            # Use FOK for immediate fill or GTC for resting order
            # FOK ensures we either exit now or not at all
            response = self.client.post_order(signed_order, OrderType.FOK)

            # Log raw response for debugging
            logger.debug(f"Sell order response: {response}")

            # Validate response
            is_valid, error_msg = _validate_order_response(response)
            if not is_valid:
                # Try GTC if FOK fails (allow partial fills)
                logger.warning(f"FOK sell failed ({error_msg}), trying GTC...")
                signed_order = self.client.create_order(order_args)
                response = self.client.post_order(signed_order, OrderType.GTC)
                is_valid, error_msg = _validate_order_response(response)

                if not is_valid:
                    logger.error(f"Sell order rejected: {error_msg}")
                    return TradeResult(success=False, error_message=error_msg)

            order_id = response.get("orderID") or response.get("order_id", "")

            # Extract fill information
            filled_size_raw = response.get("filledSize") or response.get("size_matched", 0)
            filled_price_raw = response.get("avgPrice") or response.get("price", price)
            try:
                filled_size = float(filled_size_raw) if filled_size_raw else shares
                filled_price = float(filled_price_raw) if filled_price_raw else price
            except (ValueError, TypeError):
                filled_size = shares
                filled_price = price

            logger.info(
                f"SELL ORDER executed: {shares:.2f} shares "
                f"filled {filled_size:.2f} @ {filled_price:.4f} "
                f"- Order ID: {order_id[:16]}..."
            )

            return TradeResult(
                success=True,
                order_id=order_id,
                filled_size=filled_size,
                filled_price=filled_price,
            )

        except Exception as e:
            logger.error(f"Failed to sell position: {e}")
            return TradeResult(
                success=False,
                error_message=str(e),
            )

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a specific order with retry logic.

        Args:
            order_id: ID of order to cancel

        Returns:
            True if successful
        """
        if self.config.dry_run:
            logger.info(f"[DRY RUN] Cancel order {order_id[:16]}...")
            return True

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.client.cancel_order(OrderPayload(orderID=order_id))
                logger.info(f"Cancelled order: {order_id[:16]}...")
                return True
            except Exception as e:
                error_details = _extract_error_details(e)
                error_str = str(e).lower()

                # Check if order doesn't exist (already filled/cancelled)
                if "not found" in error_str or "does not exist" in error_str:
                    logger.info(f"Order {order_id[:16]}... already cancelled or filled")
                    return True

                # Check if it's a network error worth retrying
                is_network_error = any(x in error_str for x in [
                    "request exception", "timeout", "connection",
                    "network", "ssl", "socket"
                ])

                if is_network_error and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logger.warning(
                        f"Cancel failed (attempt {attempt + 1}/{max_retries}): {error_details} | "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Failed to cancel order {order_id[:16]}...: {error_details}")
                    logger.debug(f"Full traceback: {traceback.format_exc()}")
                    return False

        return False

    def cancel_all_orders(self) -> bool:
        """
        Cancel all open orders with retry logic.

        Returns:
            True if successful
        """
        if self.config.dry_run:
            logger.info("[DRY RUN] Cancel all orders")
            return True

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.client.cancel_all()
                logger.info("Cancelled all orders")
                return True
            except Exception as e:
                error_details = _extract_error_details(e)
                error_str = str(e).lower()

                is_network_error = any(x in error_str for x in [
                    "request exception", "timeout", "connection",
                    "network", "ssl", "socket"
                ])

                if is_network_error and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logger.warning(
                        f"Cancel all failed (attempt {attempt + 1}/{max_retries}): {error_details} | "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Failed to cancel all orders: {error_details}")
                    logger.debug(f"Full traceback: {traceback.format_exc()}")
                    return False

        return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Get current status of an order.

        Args:
            order_id: ID of order to check

        Returns:
            Order object or None if not found
        """
        try:
            order_info = self.client.get_order(order_id)
            if not order_info:
                return None

            # Log the raw status for debugging
            raw_status = order_info.get("status", "")
            logger.debug(f"Order {order_id[:8]}... raw status: {raw_status}")

            # Polymarket uses different status values than expected:
            # MATCHED = Trade matched, being executed
            # MINED = Transaction mined into blockchain
            # CONFIRMED = Trade successful (finalized) - treat as FILLED
            # RETRYING = Transaction failed, being retried
            # FAILED = Trade failed permanently - treat as CANCELLED
            # LIVE = Order is live on the book (not yet matched)
            status_map = {
                # Standard statuses
                "OPEN": OrderStatus.OPEN,
                "LIVE": OrderStatus.OPEN,  # Polymarket uses LIVE for open orders
                "FILLED": OrderStatus.FILLED,
                "CANCELLED": OrderStatus.CANCELLED,
                "EXPIRED": OrderStatus.EXPIRED,
                "PARTIAL": OrderStatus.PARTIAL,
                # Polymarket-specific statuses
                "MATCHED": OrderStatus.PARTIAL,  # Being executed, not yet confirmed
                "MINED": OrderStatus.PARTIAL,    # Mined but not confirmed
                "CONFIRMED": OrderStatus.FILLED,  # Successfully filled
                "RETRYING": OrderStatus.OPEN,     # Still trying
                "FAILED": OrderStatus.CANCELLED,  # Failed permanently
            }

            return Order(
                order_id=order_id,
                token_id=order_info.get("asset_id", ""),
                side=order_info.get("side", "BUY"),
                price=float(order_info.get("price", 0)),
                size=float(order_info.get("original_size", 0)),
                order_type=order_info.get("order_type", "GTC"),
                status=status_map.get(
                    order_info.get("status", ""),
                    OrderStatus.PENDING
                ),
                filled_size=float(order_info.get("size_matched", 0)),
            )

        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return None
