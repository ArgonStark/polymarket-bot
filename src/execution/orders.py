"""
Order execution module for Polymarket CLOB.

Handles order creation, placement, and management with
support for maker (rebate) and taker orders.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

from ..models import Order, OrderStatus, TradeResult, Signal, OrderAction
from ..config import BotConfig


logger = logging.getLogger(__name__)


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

    # Check for error indicators
    if "error" in response:
        return False, f"API error: {response['error']}"

    if "errorMsg" in response:
        return False, f"API error: {response['errorMsg']}"

    if response.get("status") == "error":
        return False, f"Order rejected: {response.get('message', 'Unknown error')}"

    # Check for required fields
    order_id = response.get("orderID") or response.get("order_id")
    if not order_id:
        return False, "Response missing order ID"

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

            # Post with post_only flag
            options = {"post_only": post_only} if post_only else {}
            response = self.client.post_order(
                signed_order,
                OrderType.GTC,
                options=options,
            )

            # Validate response
            is_valid, error_msg = _validate_order_response(response)
            if not is_valid:
                logger.error(f"Maker order rejected: {error_msg}")
                return TradeResult(success=False, error_message=error_msg)

            order_id = response.get("orderID") or response.get("order_id", "")
            logger.info(
                f"MAKER ORDER placed: {side} {size_shares:.2f} shares "
                f"@ {price:.4f} - Order ID: {order_id[:16]}..."
            )

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

        try:
            price = max(0.01, min(0.99, price))
            size_shares = max(0.01, size_shares)

            order_args = OrderArgs(
                price=price,
                size=size_shares,
                side=BUY if side.upper() == "BUY" else SELL,
                token_id=token_id,
            )

            signed_order = self.client.create_order(order_args)
            response = self.client.post_order(signed_order, OrderType.GTC)

            # Validate response
            is_valid, error_msg = _validate_order_response(response)
            if not is_valid:
                logger.error(f"Limit order rejected: {error_msg}")
                return TradeResult(success=False, error_message=error_msg)

            order_id = response.get("orderID") or response.get("order_id", "")
            logger.info(
                f"LIMIT ORDER placed: {side} {size_shares:.2f} shares "
                f"@ {price:.4f} - Order ID: {order_id[:16]}..."
            )

            return TradeResult(
                success=True,
                order_id=order_id,
                filled_size=0.0,
                filled_price=price,
            )

        except Exception as e:
            logger.error(f"Failed to place limit order: {e}")
            return TradeResult(
                success=False,
                error_message=str(e),
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
            # For market buy, use aggressive price
            # For market sell, use low price
            price = 0.99 if side.upper() == "BUY" else 0.01

            # Calculate approximate shares from USD
            # This is approximate - actual fill may vary
            size_shares = size_usd / price

            order_args = OrderArgs(
                price=price,
                size=size_shares,
                side=BUY if side.upper() == "BUY" else SELL,
                token_id=token_id,
            )

            signed_order = self.client.create_order(order_args)
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

    def execute_signal(self, signal: Signal) -> TradeResult:
        """
        Execute a trading signal.

        Routes to appropriate order type based on signal's
        recommended action.

        Args:
            signal: Trading signal to execute

        Returns:
            TradeResult with execution details
        """
        if signal.recommended_action == OrderAction.SKIP:
            logger.debug(f"SKIP: {signal.reasoning}")
            return TradeResult(success=False, error_message="Signal skipped")

        # Check for valid size
        if signal.size_shares <= 0 or signal.size_usd <= 0:
            logger.warning(
                f"Invalid signal size for {signal.market.asset}: "
                f"size_usd=${signal.size_usd:.2f}, size_shares={signal.size_shares:.2f}"
            )
            return TradeResult(success=False, error_message="Invalid signal size")

        # Determine token to trade
        if signal.side.value == "UP":
            token_id = signal.market.up_token_id
        else:
            token_id = signal.market.down_token_id

        logger.info(
            f"Executing {signal.recommended_action.value} signal: "
            f"{signal.side.value} {signal.market.asset} | "
            f"Price: {signal.recommended_price:.4f} | "
            f"Size: {signal.size_shares:.2f} shares (${signal.size_usd:.2f}) | "
            f"Token: {token_id[:16]}..."
        )

        # Check that client exists for live trading
        if not self.config.dry_run and self.client is None:
            logger.error("Cannot execute trade: client not initialized")
            return TradeResult(success=False, error_message="Client not initialized")

        if signal.recommended_action == OrderAction.POST_ONLY:
            logger.info(f"Placing POST_ONLY order: BUY {signal.size_shares:.2f} @ {signal.recommended_price:.4f}")
            return self.place_maker_order(
                token_id=token_id,
                side="BUY",
                price=signal.recommended_price,
                size_shares=signal.size_shares,
                post_only=True,
            )

        elif signal.recommended_action == OrderAction.LIMIT:
            logger.info(f"Placing LIMIT order: BUY {signal.size_shares:.2f} @ {signal.recommended_price:.4f}")
            return self.place_limit_order(
                token_id=token_id,
                side="BUY",
                price=signal.recommended_price,
                size_shares=signal.size_shares,
            )

        elif signal.recommended_action == OrderAction.MARKET:
            logger.info(f"Placing MARKET order: BUY ${signal.size_usd:.2f}")
            return self.place_market_order(
                token_id=token_id,
                side="BUY",
                size_usd=signal.size_usd,
            )

        else:
            logger.warning(f"Unknown action: {signal.recommended_action}")
            return TradeResult(
                success=False,
                error_message=f"Unknown action: {signal.recommended_action}",
            )

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a specific order.

        Args:
            order_id: ID of order to cancel

        Returns:
            True if successful
        """
        if self.config.dry_run:
            logger.info(f"[DRY RUN] Cancel order {order_id[:16]}...")
            return True

        try:
            self.client.cancel(order_id=order_id)
            logger.info(f"Cancelled order: {order_id[:16]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def cancel_all_orders(self) -> bool:
        """
        Cancel all open orders.

        Returns:
            True if successful
        """
        if self.config.dry_run:
            logger.info("[DRY RUN] Cancel all orders")
            return True

        try:
            self.client.cancel_all()
            logger.info("Cancelled all orders")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
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

            status_map = {
                "OPEN": OrderStatus.OPEN,
                "FILLED": OrderStatus.FILLED,
                "CANCELLED": OrderStatus.CANCELLED,
                "EXPIRED": OrderStatus.EXPIRED,
                "PARTIAL": OrderStatus.PARTIAL,
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
