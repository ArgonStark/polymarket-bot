"""Position tracking, marking, and equity calculation mixin."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from src.models import Side, PositionState

logger = logging.getLogger(__name__)


class PositionMixin:
    """Mixin for position tracking and mark-to-market. Mixed into TradingBot."""

    def _has_active_position(self, asset: str) -> bool:
        """
        Check if we already have an active position or pending order for this asset.

        Checks:
        1. Internal position tracking (risk_manager.positions)
        2. Cached API positions (updated by _sync_existing_orders)
        3. Pending orders (orders placed but not yet filled)

        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP)

        Returns:
            True if position or pending order exists, False otherwise
        """
        # Check internal tracking
        for pos in self.risk_manager.positions.values():
            if pos.market.asset == asset:
                return True

        # Check cached API positions (stored during sync)
        if hasattr(self, '_api_positions') and asset in self._api_positions:
            return True

        # Check pending orders
        if hasattr(self, '_pending_orders'):
            for order_data in self._pending_orders.values():
                if order_data.get("asset") == asset:
                    return True

        return False

    def _get_active_position(self, asset: str, variant: str = None):
        """
        Get the active position for an asset (optionally filtered by variant).

        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP)
            variant: Optional market variant ("five" or "fifteen").
                     If provided, only matches positions in that variant.

        Returns:
            Position object or None
        """
        # Check internal tracking
        for pos in self.risk_manager.positions.values():
            if pos.market.asset == asset:
                if variant and getattr(pos.market, "variant", "fifteen") != variant:
                    continue
                return pos

        return None

    def _derive_position_state(
        self, position, market, resolved_outcome: str = None
    ) -> PositionState:
        """
        Derive the current state of a position.

        Args:
            position: The position object
            market: The market state (may be None)
            resolved_outcome: If known, "WIN" or "LOSS" from settlement verification

        Returns:
            PositionState enum value

        State machine:
        - OPEN: market.time_remaining > 0
        - ENDED_UNRESOLVED: time_remaining <= 0 and no resolved_outcome
        - RESOLVED_WIN: resolved_outcome == "WIN"
        - RESOLVED_LOSS: resolved_outcome == "LOSS"
        - SETTLING: (future use - redemption in progress)
        - CLOSED: (future use - finalized)
        """
        if resolved_outcome == "WIN":
            return PositionState.RESOLVED_WIN
        if resolved_outcome == "LOSS":
            return PositionState.RESOLVED_LOSS

        if not market:
            return PositionState.ENDED_UNRESOLVED

        time_remaining = market.time_remaining if hasattr(market, 'time_remaining') else 0

        if time_remaining > 0:
            return PositionState.OPEN
        else:
            return PositionState.ENDED_UNRESOLVED

    def _get_position_mark_price(
        self, position, market, state: PositionState = None
    ) -> tuple[float, str, dict]:
        """
        Get mark price for a position based on its state.

        Marking rules by state:
        - RESOLVED_WIN: mark_price = 1.0
        - RESOLVED_LOSS: mark_price = 0.0
        - OPEN / ENDED_UNRESOLVED: mark-to-market using best available quote

        For long positions, mark = bid of the SAME outcome token:
          UP position  -> YES token bid
          DOWN position -> NO token bid

        Quote fallback chain:
        1. CLOB orderbook bid for this token
        2. Paper executor quote cache bid for this token (keyed by token_id)
        3. MarketState prices (derived from UP token)
        4. None (unmarked)

        Returns:
            Tuple of (mark_price, source_description, debug_info)
            mark_price is None if no valid price available
            debug_info contains token_id_used, bid, ask, mid for logging
        """
        debug_info = {
            "token_id_used": position.token_id if position else "",
            "bid": None, "ask": None, "mid": None,
        }

        # Derive state if not provided
        if state is None:
            state = self._derive_position_state(position, market)

        # RESOLVED_WIN: mark = 1.0
        if state == PositionState.RESOLVED_WIN:
            return 1.0, "resolved_win", debug_info

        # RESOLVED_LOSS: mark = 0.0
        if state == PositionState.RESOLVED_LOSS:
            return 0.0, "resolved_loss", debug_info

        # OPEN or ENDED_UNRESOLVED: use best available quote
        if not market:
            return None, "no_market", debug_info

        # Determine the correct token_id for this position's side
        if position.side == Side.UP:
            token_id = market.up_token_id
        else:
            token_id = market.down_token_id
        debug_info["token_id_used"] = token_id

        # Verify position.token_id matches expected token for side
        if position.token_id and position.token_id != token_id:
            logger.warning(
                "TOKEN_MISMATCH position.token_id=%s expected_%s_token=%s market=%s asset=%s",
                position.token_id[:16], position.side.value.lower(),
                token_id[:16], market.condition_id[:8], market.asset,
            )

        # 1) CLOB orderbook — mark at bid (what we could sell at)
        orderbook = self.clob_feed.get_orderbook(token_id) if self.clob_feed else None
        if orderbook and orderbook.best_bid is not None and orderbook.best_bid > 0:
            debug_info["bid"] = orderbook.best_bid
            debug_info["ask"] = orderbook.best_ask
            debug_info["mid"] = orderbook.mid_price
            return orderbook.best_bid, "clob_bid", debug_info

        # 2) Paper executor quote cache (fed from MarketState each tick)
        if self.executor and hasattr(self.executor, 'get_cached_quote'):
            cached = self.executor.get_cached_quote(token_id)
            if cached and cached.bid > 0:
                debug_info["bid"] = cached.bid
                debug_info["ask"] = cached.ask
                debug_info["mid"] = cached.mid
                return cached.bid, "cache_bid", debug_info

        # 3) MarketState prices (derived from UP token orderbook)
        if position.side == Side.UP:
            if market.best_bid and market.best_bid > 0:
                debug_info["bid"] = market.best_bid
                debug_info["ask"] = market.best_ask
                debug_info["mid"] = (market.best_bid + market.best_ask) / 2
                return market.best_bid, "market_yes_bid", debug_info
        else:
            if market.best_ask and market.best_ask < 1.0:
                no_bid = 1.0 - market.best_ask
                no_ask = 1.0 - market.best_bid if market.best_bid else None
                debug_info["bid"] = no_bid
                debug_info["ask"] = no_ask
                debug_info["mid"] = (no_bid + no_ask) / 2 if no_ask else no_bid
                return no_bid, "market_no_bid", debug_info

        return None, "no_price", debug_info

    def _calculate_equity(self, debug_log: bool = False) -> float:
        """
        Calculate current equity: cash + sum(mark_value or cost_basis).

        Uses state-aware marking:
        - RESOLVED_WIN: mark = 1.0
        - RESOLVED_LOSS: mark = 0.0
        - OPEN/ENDED_UNRESOLVED: mark = best available quote
        - No price: fallback to cost basis (equity neutral, 0 unrealized)

        Returns:
            Total equity (cash + position value)
        """
        cash = self.risk_manager.current_bankroll
        sum_mark_value = 0.0
        sum_cost_basis = 0.0
        unrealized_pnl = 0.0

        for market_key, position in self.risk_manager.positions.items():
            market = self.markets.get(market_key) or self.expiring_markets.get(market_key)

            state = self._derive_position_state(position, market)
            mark_price, source, _dbg = self._get_position_mark_price(position, market, state)

            cost = position.cost_basis
            sum_cost_basis += cost

            if mark_price is not None and mark_price >= 0:
                mark_val = position.shares * mark_price
                sum_mark_value += mark_val
                unrealized_pnl += mark_val - cost
            else:
                # Unmarked: use cost_basis for equity, contribute 0 to unrealized
                sum_mark_value += cost

        # Realized P&L from daily stats
        daily = self.risk_manager.get_daily_stats()
        realized_pnl = daily.total_pnl if daily else 0.0

        equity = cash + sum_mark_value
        total_pnl = realized_pnl + unrealized_pnl

        if debug_log:
            logger.debug(
                "EQUITY_CALC cash=%.2f sum_cost_basis=%.2f sum_mark_value=%.2f "
                "realized=%.2f unrealized=%.2f equity=%.2f pnl=%.2f",
                cash, sum_cost_basis, sum_mark_value,
                realized_pnl, unrealized_pnl, equity, total_pnl,
            )

        # Stash for header use (avoids recomputing)
        self._last_equity_detail = {
            "cash": cash,
            "sum_cost_basis": sum_cost_basis,
            "sum_mark_value": sum_mark_value,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "equity": equity,
            "total_pnl": total_pnl,
        }

        return equity

    def _update_cached_positions(self, positions: dict):
        """Update cached API positions."""
        self._api_positions = positions
