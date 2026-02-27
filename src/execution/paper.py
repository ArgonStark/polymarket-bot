"""Paper trading execution engine with realistic fill simulation."""

from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict

from ..models import TradeResult, Signal, OrderAction, Order, OrderStatus
from .orders import OrderSafetyGuard, OrderSafetyConfig, get_safety_guard


logger = logging.getLogger(__name__)


@dataclass
class CachedQuote:
    """Token-level cached quote with metadata for debugging."""
    token_id: str
    market_id: str
    bid: float
    ask: float
    mid: float
    timestamp: datetime
    source: str  # "market_state_up", "market_state_down_derived"


@dataclass
class PaperTradingConfig:
    enabled: bool
    initial_balance: float
    maker_fee_bps: float
    taker_fee_bps: float
    slippage_bps: float


@dataclass
class PaperOrder:
    """Pending paper order waiting to be filled."""
    order_id: str
    token_id: str
    asset: str
    market_id: str
    side: str           # "UP" or "DOWN" (bot side)
    action: str         # "BUY" (always BUY for opening positions)
    limit_price: float
    size: float         # in shares
    size_usd: float     # original USD amount
    order_type: str     # "LIMIT", "POST_ONLY"
    status: OrderStatus = OrderStatus.OPEN
    filled_size: float = 0.0
    filled_price: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    signal: Optional[Signal] = None


class PaperAccount:
    """Virtual account for paper trading."""

    def __init__(self, balance: float):
        self.balance = balance

    def debit(self, amount: float):
        self.balance -= amount

    def credit(self, amount: float):
        self.balance += amount


class PaperOrderExecutor:
    """Simulated order execution with realistic fill logic based on live quotes."""

    def __init__(
        self,
        config: PaperTradingConfig,
        clob_feed,
        safety_guard: Optional[OrderSafetyGuard] = None,
    ):
        self.config = config
        self.clob_feed = clob_feed
        self.account = PaperAccount(config.initial_balance)
        self._safety_guard = safety_guard

        # Pending orders awaiting fill
        self._pending_orders: Dict[str, PaperOrder] = {}
        self._orders_lock = threading.Lock()

        # Filled orders history (for reference)
        self._filled_orders: Dict[str, PaperOrder] = {}

        # Quote cache: token_id -> CachedQuote (LRU-bounded)
        # Fallback when clob_feed.get_orderbook() returns None
        self._quote_cache: OrderedDict[str, CachedQuote] = OrderedDict()
        self._quote_cache_max_size: int = 200

    @property
    def safety_guard(self) -> OrderSafetyGuard:
        """Get safety guard (uses provided or global instance)."""
        if self._safety_guard is None:
            self._safety_guard = get_safety_guard()
        return self._safety_guard

    def update_quotes(
        self, token_id: str, bid: float, ask: float,
        market_id: str = "", source: str = "market_state",
    ):
        """Cache a bid/ask quote for a token (fed from MarketState each tick)."""
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
        # Move to end if already exists (LRU behavior)
        if token_id in self._quote_cache:
            self._quote_cache.move_to_end(token_id)
        self._quote_cache[token_id] = CachedQuote(
            token_id=token_id,
            market_id=market_id,
            bid=bid,
            ask=ask,
            mid=mid,
            timestamp=datetime.now(timezone.utc),
            source=source,
        )
        # Evict oldest entries if over limit
        while len(self._quote_cache) > self._quote_cache_max_size:
            self._quote_cache.popitem(last=False)

    def get_cached_quote(self, token_id: str) -> Optional[CachedQuote]:
        """Get cached quote for a token (used by mark-to-market)."""
        return self._quote_cache.get(token_id)

    def _get_quotes(self, token_id: str) -> tuple[Optional[float], Optional[float], str]:
        """
        Get best bid/ask for a token.

        Tries clob_feed orderbook first, then falls back to cached quotes.
        Returns (bid, ask, source) where source is 'orderbook' or 'cache'.
        """
        orderbook = self.clob_feed.get_orderbook(token_id) if self.clob_feed else None
        if orderbook and (orderbook.best_bid is not None or orderbook.best_ask is not None):
            return orderbook.best_bid, orderbook.best_ask, "orderbook"

        # Fallback to cached quotes from MarketState
        cached = self._quote_cache.get(token_id)
        if cached:
            return cached.bid, cached.ask, "cache"

        return None, None, "none"

    async def execute_signal_async(
        self,
        signal: Signal,
        current_exposure: float = 0.0,
        bankroll: float = 0.0,
        safety_guard: Optional[OrderSafetyGuard] = None,
    ) -> TradeResult:
        """
        Execute a trading signal in paper mode with full exception handling.

        LIMIT/POST_ONLY orders go into pending state and fill when price conditions are met.
        MARKET orders fill immediately at best_ask.
        """
        # Extract basic info for error logging
        asset = getattr(signal.market, 'asset', 'UNKNOWN') if signal and signal.market else 'UNKNOWN'
        market_id = getattr(signal.market, 'condition_id', '????????') if signal and signal.market else '????????'
        side = signal.side.value if signal and signal.side else 'UNKNOWN'

        try:
            return await self._execute_signal_impl(
                signal, current_exposure, bankroll, safety_guard
            )
        except Exception as e:
            logger.error(
                "ORDER_RESULT asset=%s market=%s side=%s status=exception "
                "error=%s mode=paper",
                asset, market_id[:8], side, str(e)[:200]
            )
            logger.debug("Paper executor exception traceback:", exc_info=True)
            return TradeResult(
                success=False,
                error_message=f"Paper execution error: {str(e)[:100]}"
            )

    async def _execute_signal_impl(
        self,
        signal: Signal,
        current_exposure: float,
        bankroll: float,
        safety_guard: Optional[OrderSafetyGuard],
    ) -> TradeResult:
        """Internal implementation with realistic fill simulation."""
        asset = signal.market.asset
        market_id = signal.market.condition_id
        side = signal.side.value
        order_type = signal.recommended_action.value if signal.recommended_action else "UNKNOWN"

        guard = safety_guard or self.safety_guard

        if signal.recommended_action == OrderAction.SKIP:
            logger.info(
                "ORDER_BLOCK asset=%s market=%s side=%s reason=skip_action mode=paper",
                asset, market_id[:8], side
            )
            return TradeResult(success=False, error_message="Signal skipped")

        # Safety guard checks
        can_submit, block_reason = guard.check_can_submit(
            signal, current_exposure, bankroll or self.account.balance
        )
        if not can_submit:
            logger.info(
                "ORDER_BLOCK asset=%s market=%s side=%s type=%s reason=%s "
                "size_usd=%.2f price=%.4f mode=paper",
                asset, market_id[:8], side, order_type, block_reason,
                float(signal.size_usd or 0), float(signal.recommended_price or 0)
            )
            return TradeResult(success=False, error_message=f"Safety block: {block_reason}")

        # Get token ID for the side we're trading
        # UP -> up_token_id (YES token), DOWN -> down_token_id (NO token)
        token_id = signal.market.up_token_id if side == "UP" else signal.market.down_token_id
        q_bid, q_ask, q_source = self._get_quotes(token_id)

        # Extract signal values
        signal_rec_price = float(signal.recommended_price) if signal.recommended_price is not None else 0.0
        signal_size_usd = float(signal.size_usd) if signal.size_usd is not None else 0.0
        signal_size_shares = float(signal.size_shares) if signal.size_shares is not None else 0.0

        if signal_rec_price <= 0:
            logger.info(
                "ORDER_BLOCK asset=%s market=%s side=%s reason=no_submitted_price "
                "recommended_price=%s mode=paper",
                asset, market_id[:8], side, signal.recommended_price
            )
            return TradeResult(success=False, error_message="No valid submitted price")

        paper_order_id = f"paper_{uuid.uuid4().hex[:12]}"

        # Calculate size
        if signal_size_shares > 0:
            size = signal_size_shares
        elif signal_size_usd > 0 and signal_rec_price > 0:
            size = signal_size_usd / signal_rec_price
        else:
            logger.info(
                "ORDER_RESULT asset=%s market=%s side=%s status=rejected "
                "reason=invalid_size mode=paper",
                asset, market_id[:8], side
            )
            return TradeResult(success=False, error_message="Invalid order size")

        # Log ORDER_SUBMIT
        logger.info(
            "ORDER_SUBMIT asset=%s market=%s side=%s type=%s price=%.4f "
            "size_usd=%.2f size_shares=%.4f mode=paper",
            asset, market_id[:8], side, order_type, signal_rec_price,
            signal_size_usd, size
        )

        # === MARKET ORDER - fills immediately at best_ask ===
        if signal.recommended_action == OrderAction.MARKET:
            if q_ask is None or q_ask <= 0:
                logger.info(
                    "ORDER_RESULT asset=%s market=%s side=%s status=rejected "
                    "reason=no_best_ask best_ask=%s source=%s mode=paper",
                    asset, market_id[:8], side, q_ask, q_source
                )
                return TradeResult(success=False, error_message="No valid best_ask for MARKET order")

            fill_price = _apply_slippage(q_ask, self.config.slippage_bps)
            notional = fill_price * size
            fee = self._fee(notional, taker=True)
            self.account.debit(notional + fee)

            guard.record_order_submitted(signal, paper_order_id)

            logger.info(
                "ORDER_RESULT asset=%s market=%s side=%s status=success "
                "order_id=%s filled_size=%.4f filled_price=%.4f mode=paper",
                asset, market_id[:8], side, paper_order_id, size, fill_price
            )
            logger.info(
                "ORDER_FILL asset=%s market=%s side=%s order_id=%s "
                "fill_price=%.4f size=%.4f fee=%.4f notional=%.4f mode=paper",
                asset, market_id[:8], side, paper_order_id,
                fill_price, size, fee, notional
            )

            guard.record_order_filled(asset, market_id, side, paper_order_id)

            return TradeResult(
                success=True, order_id=paper_order_id, filled_size=size, filled_price=fill_price
            )

        # === LIMIT / POST_ONLY - goes to pending, fills when price condition met ===
        if signal.recommended_action in (OrderAction.LIMIT, OrderAction.POST_ONLY):
            # Create pending order
            paper_order = PaperOrder(
                order_id=paper_order_id,
                token_id=token_id,
                asset=asset,
                market_id=market_id,
                side=side,
                action="BUY",  # Always buying tokens when opening position
                limit_price=signal_rec_price,
                size=size,
                size_usd=signal_size_usd,
                order_type=order_type,
                status=OrderStatus.OPEN,
                signal=signal,
            )

            with self._orders_lock:
                self._pending_orders[paper_order_id] = paper_order

            guard.record_order_submitted(signal, paper_order_id)

            # Log PAPER_ORDER_OPEN with quote source
            logger.info(
                "PAPER_ORDER_OPEN order_id=%s asset=%s market=%s side=%s token=%s "
                "action=BUY limit=%.4f size=%.4f current_bid=%s current_ask=%s source=%s",
                paper_order_id[:12], asset, market_id[:8], side, token_id[:12],
                signal_rec_price, size,
                f"{q_bid:.4f}" if q_bid else "None",
                f"{q_ask:.4f}" if q_ask else "None",
                q_source,
            )

            # Check if we can fill immediately (ask <= limit for BUY)
            if q_ask is not None and q_ask <= signal_rec_price:
                # Immediate fill at limit price (maker fill)
                return self._fill_paper_order(paper_order, signal_rec_price, is_taker=False)

            # Return success with OPEN status (order placed, awaiting fill)
            logger.info(
                "ORDER_RESULT asset=%s market=%s side=%s status=open "
                "order_id=%s limit_price=%.4f size=%.4f mode=paper",
                asset, market_id[:8], side, paper_order_id, signal_rec_price, size
            )

            return TradeResult(
                success=True,
                order_id=paper_order_id,
                filled_size=0.0,  # Not filled yet
                filled_price=0.0,
            )

        logger.info(
            "ORDER_BLOCK asset=%s market=%s side=%s reason=unsupported_action "
            "action=%s mode=paper",
            asset, market_id[:8], side, signal.recommended_action
        )
        return TradeResult(success=False, error_message="Unsupported paper action")

    def _fill_paper_order(self, order: PaperOrder, fill_price: float, is_taker: bool = True) -> TradeResult:
        """Fill a paper order and update account."""
        # Only apply slippage for taker fills. Maker orders rest on the book
        # and fill at their exact limit price — no slippage.
        if is_taker:
            fill_price = _apply_slippage(fill_price, self.config.slippage_bps)
        notional = fill_price * order.size
        fee = self._fee(notional, taker=is_taker)
        self.account.debit(notional + fee)

        # Update order state
        order.status = OrderStatus.FILLED
        order.filled_size = order.size
        order.filled_price = fill_price

        # Move from pending to filled
        with self._orders_lock:
            self._pending_orders.pop(order.order_id, None)
            self._filled_orders[order.order_id] = order

        # Log PAPER_ORDER_FILL
        logger.info(
            "PAPER_ORDER_FILL order_id=%s asset=%s market=%s side=%s "
            "fill_price=%.4f size=%.4f fee=%.4f notional=%.4f taker=%s",
            order.order_id[:12], order.asset, order.market_id[:8], order.side,
            fill_price, order.size, fee, notional, is_taker
        )

        # Also log ORDER_FILL for consistency
        logger.info(
            "ORDER_FILL asset=%s market=%s side=%s order_id=%s "
            "fill_price=%.4f size=%.4f fee=%.4f notional=%.4f mode=paper",
            order.asset, order.market_id[:8], order.side, order.order_id,
            fill_price, order.size, fee, notional
        )

        # Update safety guard
        self.safety_guard.record_order_filled(
            order.asset, order.market_id, order.side, order.order_id
        )

        return TradeResult(
            success=True,
            order_id=order.order_id,
            filled_size=order.size,
            filled_price=fill_price,
        )

    def check_pending_orders(self) -> list[tuple[str, PaperOrder]]:
        """
        Check all pending orders against current quotes and fill if conditions met.

        Fill conditions:
        - BUY order fills when ask <= limit_price
        - SELL order fills when bid >= limit_price

        Returns list of (order_id, filled_order) for orders that were filled.
        """
        filled_orders = []

        with self._orders_lock:
            pending_copy = list(self._pending_orders.items())

        for order_id, order in pending_copy:
            # Get quotes (orderbook first, then cached MarketState quotes)
            best_bid, best_ask, q_source = self._get_quotes(order.token_id)

            if best_bid is None and best_ask is None:
                logger.debug(
                    "ORDER_EVAL order_id=%s token=%s side=%s limit=%.4f "
                    "quote_used=none decision=HOLD reason=no_quote",
                    order_id[:12], order.token_id[:12], order.side,
                    order.limit_price,
                )
                continue

            # Determine fill condition based on action (touch fill)
            fill_condition_met = False
            fill_price = 0.0
            reason = ""

            if order.action == "BUY":
                # BUY fills when ask <= limit
                if best_ask is not None and best_ask <= order.limit_price:
                    fill_condition_met = True
                    fill_price = order.limit_price  # Fill at limit (maker)
                    reason = f"ask({best_ask:.4f})<=limit({order.limit_price:.4f})"
                else:
                    reason = f"ask({best_ask})>limit({order.limit_price:.4f})" if best_ask else "no_ask"
            elif order.action == "SELL":
                # SELL fills when bid >= limit
                if best_bid is not None and best_bid >= order.limit_price:
                    fill_condition_met = True
                    fill_price = order.limit_price  # Fill at limit (maker)
                    reason = f"bid({best_bid:.4f})>=limit({order.limit_price:.4f})"
                else:
                    reason = f"bid({best_bid})<limit({order.limit_price:.4f})" if best_bid else "no_bid"

            decision = "FILL" if fill_condition_met else "HOLD"

            # ORDER_EVAL debug log per open order
            logger.debug(
                "ORDER_EVAL order_id=%s token=%s side=%s limit=%.4f "
                "quote_used=%s(bid=%s,ask=%s) decision=%s reason=%s",
                order_id[:12], order.token_id[:12], order.side,
                order.limit_price, q_source,
                f"{best_bid:.4f}" if best_bid is not None else "None",
                f"{best_ask:.4f}" if best_ask is not None else "None",
                decision, reason,
            )

            if fill_condition_met:
                result = self._fill_paper_order(order, fill_price, is_taker=False)
                if result.success:
                    filled_orders.append((order_id, order))

        return filled_orders

    def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Get current status of a paper order.

        This is called by bot._check_pending_orders to determine if orders filled.
        """
        # First check pending orders
        with self._orders_lock:
            if order_id in self._pending_orders:
                paper_order = self._pending_orders[order_id]
                return Order(
                    order_id=order_id,
                    token_id=paper_order.token_id,
                    side=paper_order.action,
                    price=paper_order.limit_price,
                    size=paper_order.size,
                    order_type="GTC",
                    status=paper_order.status,
                    filled_size=paper_order.filled_size,
                    created_at=paper_order.created_at,
                )

            # Check filled orders
            if order_id in self._filled_orders:
                paper_order = self._filled_orders[order_id]
                return Order(
                    order_id=order_id,
                    token_id=paper_order.token_id,
                    side=paper_order.action,
                    price=paper_order.filled_price,
                    size=paper_order.size,
                    order_type="GTC",
                    status=OrderStatus.FILLED,
                    filled_size=paper_order.filled_size,
                    created_at=paper_order.created_at,
                )

        return None

    def get_pending_orders(self) -> Dict[str, PaperOrder]:
        """Get all pending orders."""
        with self._orders_lock:
            return dict(self._pending_orders)

    def get_pending_count(self) -> int:
        """Get count of pending orders."""
        with self._orders_lock:
            return len(self._pending_orders)

    def cancel_order(self, order_id: str) -> tuple[bool, str]:
        """
        Cancel a pending paper order.

        Returns:
            Tuple of (success, reason) where reason is one of:
            - "cancelled": Order was pending and is now cancelled
            - "already_filled": Order already filled (not an error)
            - "already_cancelled": Order was already cancelled
            - "not_found": Order not found in any registry
        """
        with self._orders_lock:
            # Check pending orders first
            if order_id in self._pending_orders:
                order = self._pending_orders.pop(order_id)
                order.status = OrderStatus.CANCELLED
                logger.info(
                    "PAPER_ORDER_CANCEL order_id=%s asset=%s market=%s side=%s reason=cancelled",
                    order_id[:12], order.asset, order.market_id[:8], order.side
                )
                return True, "cancelled"

            # Check if already filled
            if order_id in self._filled_orders:
                logger.debug(
                    "PAPER_ORDER_CANCEL order_id=%s reason=already_filled",
                    order_id[:12]
                )
                return True, "already_filled"  # Not an error - order completed

        # Not found anywhere
        logger.debug(
            "PAPER_ORDER_CANCEL order_id=%s reason=not_found",
            order_id[:12]
        )
        return True, "not_found"  # Idempotent - treat as success

    def cancel_all_orders(self) -> None:
        """Cancel all pending orders."""
        with self._orders_lock:
            for order_id, order in list(self._pending_orders.items()):
                order.status = OrderStatus.CANCELLED
                logger.info(
                    "PAPER_ORDER_CANCEL order_id=%s asset=%s market=%s side=%s",
                    order_id[:12], order.asset, order.market_id[:8], order.side
                )
            self._pending_orders.clear()

    def sell_position(self, token_id: str, shares: float, min_price: float = 0.01) -> TradeResult:
        """Sell a position (for exits)."""
        orderbook = self.clob_feed.get_orderbook(token_id) if self.clob_feed else None
        if not orderbook:
            return TradeResult(success=False, error_message="No orderbook for paper sell")

        fill_price = max(min_price, orderbook.best_bid or min_price)
        fill_price = _apply_slippage(fill_price, self.config.slippage_bps)
        proceeds = fill_price * shares
        fee = self._fee(proceeds, taker=True)
        self.account.credit(proceeds - fee)

        logger.info(
            "PAPER_SELL token=%s shares=%.4f fill_price=%.4f proceeds=%.4f fee=%.4f",
            token_id[:12], shares, fill_price, proceeds, fee
        )

        return TradeResult(
            success=True,
            order_id=f"paper_sell_{uuid.uuid4().hex[:8]}",
            filled_size=shares,
            filled_price=fill_price
        )

    def get_balance(self) -> float:
        return self.account.balance

    def credit_settlement(self, proceeds: float, market_id: str = ""):
        """
        Credit settlement proceeds to the paper account.

        Called when a position settles (binary payout).
        Proceeds = payout_per_share * shares (1.0*shares for WIN, 0 for LOSS).
        The original cost was already debited at order fill time, so this
        credit restores cost + realised PnL.

        Args:
            proceeds: Cash to credit (cost_basis + pnl). May be 0 for full loss.
            market_id: For audit logging.
        """
        if proceeds > 0:
            self.account.credit(proceeds)
            logger.info(
                "PAPER_SETTLE_CREDIT amount=%.2f balance=%.2f market=%s",
                proceeds, self.account.balance, market_id[:16] if market_id else "",
            )
        else:
            logger.debug(
                "PAPER_SETTLE_CREDIT amount=0.00 (full loss) balance=%.2f market=%s",
                self.account.balance, market_id[:16] if market_id else "",
            )

    def _fee(self, notional: float, taker: bool) -> float:
        bps = self.config.taker_fee_bps if taker else self.config.maker_fee_bps
        return notional * (bps / 10000.0)


def _apply_slippage(price: float, bps: float) -> float:
    return price * (1 + bps / 10000.0) if price else price
