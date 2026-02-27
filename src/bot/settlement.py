"""Settlement processing mixin."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from src.models import MarketState, Side
from src.strategy.trade_history import get_trade_history
from src.utils import Colors

_VARIANT_LABELS = {"five": "5m", "fifteen": "15m"}

logger = logging.getLogger(__name__)


class SettlementMixin:
    """Mixin for market settlement and position closing. Mixed into TradingBot."""

    def _cleanup_orphaned_positions(self):
        """
        Force-close positions in markets that expired while the bot was down.

        Called after state restore. Positions in expired markets cannot be
        settled normally (no live feed data), so close them as losses to
        prevent orphaned entries that block trading.
        """
        now = datetime.now(timezone.utc)
        orphaned = []
        for cid, position in list(self.risk_manager.positions.items()):
            market = position.market
            if market.end_time < now:
                orphaned.append(cid)

        if not orphaned:
            return

        for cid in orphaned:
            position = self.risk_manager.positions.get(cid)
            if not position:
                continue

            age_sec = (now - position.market.end_time).total_seconds()
            logger.warning(
                "ORPHAN_CLEANUP market=%s asset=%s variant=%s side=%s cost=%.2f "
                "expired_ago=%.0fs — force-closing as LOSS",
                cid[:16], position.market.asset,
                getattr(position.market, 'variant', 'fifteen'),
                position.side.value, position.cost_basis, age_sec,
            )

            # Credit paper executor with zero proceeds (full loss)
            if self.config.paper_trading.enabled and hasattr(self.executor, 'credit_settlement'):
                self.executor.credit_settlement(0.0, cid)

            pnl = -position.cost_basis
            self.risk_manager.record_position_close(
                market_key=cid,
                exit_price=0.0,
                pnl=pnl,
            )

            # Record in trade history so stats reflect the loss
            trade_history = get_trade_history()
            trade_history.record_close(
                market_id=cid,
                won=False,
                pnl=pnl,
                exit_price=0.0,
            )

            # Mark as settled so we don't try to settle again
            self.settled_markets.add(cid)

        logger.info(
            "ORPHAN_CLEANUP_DONE closed=%d bankroll=%.2f",
            len(orphaned), self.risk_manager.current_bankroll,
        )

    async def _check_settlements(self):
        """
        Check expiring markets for settlement and close positions.

        Uses per-market exponential backoff to avoid log spam when
        API resolution is slow (5s -> 10s -> 20s -> 40s, capped at 60s).
        """
        now_mono = time.monotonic()

        async with self._markets_lock:
            markets_to_settle = []
            for market_id, market in self.expiring_markets.items():
                # Check if market has passed end time (with small buffer)
                if market.time_remaining <= -self.config.settlement.trigger_buffer:
                    # Exponential backoff: skip if not yet time for next check
                    next_at = self._settle_next_check.get(market_id, 0.0)
                    if now_mono < next_at:
                        continue
                    markets_to_settle.append(market_id)

            for market_id in markets_to_settle:
                market = self.expiring_markets[market_id]
                settled = await self._settle_market(market)

                # Only remove from expiring if actually settled
                if settled:
                    # Clean up backoff state
                    self._settle_next_check.pop(market_id, None)
                    self._settle_attempts.pop(market_id, None)

                    # Move to settled set and cleanup
                    self.expiring_markets.pop(market_id, None)
                    self.settled_markets.add(market_id)

                    # Clean up token-to-market mapping
                    self._token_to_market.pop(market.up_token_id, None)
                    self._token_to_market.pop(market.down_token_id, None)

                    # Unsubscribe from order book
                    self.clob_feed.unsubscribe(market.up_token_id)
                    self.clob_feed.unsubscribe(market.down_token_id)

                    # IMPORTANT: Clear all tracking for this asset so new market can trade
                    asset = market.asset

                    # Clear cached API positions
                    if hasattr(self, '_api_positions') and asset in self._api_positions:
                        del self._api_positions[asset]
                        logger.info(f"✅ Cleared cached position for {asset} after settlement")

                    # Clear cooldown for this (asset, variant)
                    _av_key = f"{asset}:{market.variant}"
                    if _av_key in self._last_order_time:
                        del self._last_order_time[_av_key]
                        logger.debug("Cleared cooldown for %s after settlement", _av_key)

                    # Cancel any pending orders for this asset
                    await self._cancel_pending_orders_for_asset(asset)

                    # Cleanup old settled markets (keep last 100)
                    if len(self.settled_markets) > 100:
                        self.settled_markets = set(list(self.settled_markets)[-100:])
                else:
                    # Not yet resolved — apply exponential backoff
                    # 5s -> 10s -> 20s -> 40s -> 60s (capped)
                    attempts = self._settle_attempts.get(market_id, 0) + 1
                    self._settle_attempts[market_id] = attempts
                    delay = min(5.0 * (2 ** (attempts - 1)), 60.0)
                    self._settle_next_check[market_id] = now_mono + delay

    async def _settle_market(self, market: MarketState) -> bool:
        """
        Process settlement for a single market.

        Args:
            market: Market to settle

        Returns:
            True if settlement was successful, False if should retry
        """
        # Idempotency guard: skip if already settled
        if market.condition_id in self.settled_markets:
            logger.debug("SETTLE_SKIP already settled market=%s", market.condition_id[:16])
            return True

        # Check if we have a position in this market
        position = self.risk_manager.positions.get(market.condition_id)
        has_position = position is not None

        # Log that we're attempting settlement
        pos_str = f" (HAVE POSITION: {position.side.value})" if has_position else ""
        logger.info(f"SETTLING: {market.asset}{pos_str} | {market.question[:50]}...")

        time_since_expiry = abs(market.time_remaining)

        # Fetch resolution from API
        resolution = self.gamma_api.get_market_resolution(market.condition_id)

        winning_outcome = None
        resolution_price = None

        if resolution and resolution.get("resolved"):
            # API has resolution
            winning_outcome = resolution.get("winning_outcome")
            resolution_price = resolution.get("resolution_price")
            price_str = f"${resolution_price:,.2f}" if resolution_price else "N/A"
            logger.info(
                f"✅ RESOLVED (API): {market.asset} | "
                f"Winner: {winning_outcome} | "
                f"Settlement Price: {price_str}"
            )

            # Cross-verify with on-chain data (The Graph)
            if self.settlement_verifier.is_enabled() and winning_outcome:
                verified, verify_msg = self.settlement_verifier.cross_verify_settlement(
                    condition_id=market.condition_id,
                    api_outcome=winning_outcome,
                    api_price=resolution_price
                )
                if verified:
                    logger.debug(f"  └─ On-chain: {verify_msg}")
                else:
                    # On-chain disagrees with API — prefer on-chain
                    onchain_result = self.settlement_verifier.verify_settlement(market.condition_id)
                    if onchain_result and onchain_result.on_chain_resolved and onchain_result.winning_outcome:
                        old_outcome = winning_outcome
                        winning_outcome = onchain_result.winning_outcome.upper()
                        logger.warning(
                            "SETTLE_OVERRIDE market=%s api=%s onchain=%s — using on-chain",
                            market.condition_id[:16], old_outcome, winning_outcome,
                        )
                    else:
                        logger.warning(
                            "SETTLE_MISMATCH_UNRESOLVED market=%s api=%s onchain_msg=%s — using API",
                            market.condition_id[:16], winning_outcome, verify_msg,
                        )
        else:
            # API doesn't have resolution yet.
            # Method A: Check CLOB orderbook — winning token bid near 1.0,
            # losing token bid near 0.0. Much more reliable than local Chainlink.
            clob_outcome = self._resolve_from_clob(market)
            if clob_outcome:
                winning_outcome = clob_outcome
                resolution_price = None
                logger.info(
                    "✅ RESOLVED (CLOB): %s | Winner: %s | "
                    "UP bid confirms direction",
                    market.asset, winning_outcome,
                )

            # Method B: Local Chainlink determination (fallback)
            # Only used when CLOB is indeterminate AND margin is wide enough to trust.
            if not winning_outcome:
                local_outcome, local_price = self._determine_outcome_locally(market)

                if local_outcome:
                    # Wait at least 30 seconds after expiry before using local resolution
                    # This gives the API time to update
                    if time_since_expiry >= self.config.settlement.local_delay_seconds:
                        winning_outcome = local_outcome
                        resolution_price = local_price
                        logger.info(
                            f"✅ RESOLVED (LOCAL): {market.asset} | "
                            f"Winner: {winning_outcome} | "
                            f"Price: ${resolution_price:,.2f} vs Target: ${market.target_price:,.2f}"
                        )

                if not winning_outcome:
                    # Still no resolution — wait and retry
                    attempts = self._settle_attempts.get(market.condition_id, 0)
                    log_fn = logger.info if attempts <= 1 else logger.debug
                    log_fn(
                        "SETTLE_WAIT market=%s asset=%s local=%s waited=%.0fs attempt=%d",
                        market.condition_id[:16], market.asset,
                        local_outcome, time_since_expiry, attempts,
                    )
                    # If we've waited more than 5 minutes, give up
                    if time_since_expiry > self.config.settlement.timeout_seconds:
                        logger.error(
                            f"SETTLEMENT TIMEOUT: {market.asset} - no resolution after 5 minutes"
                        )
                        # Force-close any open position as a loss to avoid orphaning
                        if position:
                            logger.warning(
                                f"Force-closing position in {market.asset} as LOSS (timeout)"
                            )
                            pnl = -position.cost_basis
                            if self.config.paper_trading.enabled and hasattr(self.executor, 'credit_settlement'):
                                self.executor.credit_settlement(0.0, market.condition_id)
                            self.risk_manager.record_position_close(
                                market_key=market.condition_id,
                                exit_price=0.0,
                                pnl=pnl,
                            )
                            trade_history = get_trade_history()
                            trade_history.record_close(
                                market_id=market.condition_id,
                                won=False,
                                pnl=pnl,
                                exit_price=0.0,
                            )
                            variant_label = _VARIANT_LABELS.get(market.variant, market.variant)
                            if hasattr(self, '_recent_settlements'):
                                self._recent_settlements.appendleft({
                                    "outcome": "LOSS",
                                    "asset": market.asset,
                                    "variant": variant_label,
                                    "side": position.side.value,
                                    "pnl": round(pnl, 2),
                                    "roi": -100.0,
                                    "cost": round(position.cost_basis, 2),
                                    "timestamp": time.time(),
                                })
                        return True  # Force remove to prevent infinite retry
                    return False  # Retry later

        # Process position if we had one
        if position and winning_outcome:
            await self._close_position_on_settlement(
                market=market,
                position=position,
                winning_outcome=winning_outcome,
            )
        elif position:
            logger.warning(f"Have position in {market.asset} but no winning outcome - force-closing as LOSS")
            pnl = -position.cost_basis
            if self.config.paper_trading.enabled and hasattr(self.executor, 'credit_settlement'):
                self.executor.credit_settlement(0.0, market.condition_id)
            self.risk_manager.record_position_close(
                market_key=market.condition_id,
                exit_price=0.0,
                pnl=pnl,
            )
            trade_history = get_trade_history()
            trade_history.record_close(
                market_id=market.condition_id,
                won=False,
                pnl=pnl,
                exit_price=0.0,
            )
            # Record to recent settlements
            variant_label = _VARIANT_LABELS.get(market.variant, market.variant)
            if hasattr(self, '_recent_settlements'):
                self._recent_settlements.appendleft({
                    "outcome": "LOSS",
                    "asset": market.asset,
                    "variant": variant_label,
                    "side": position.side.value,
                    "pnl": round(pnl, 2),
                    "roi": -100.0,
                    "cost": round(position.cost_basis, 2),
                    "timestamp": time.time(),
                })
        else:
            logger.debug(f"No position in {market.asset}, nothing to settle")

        return True  # Settlement successful

    def _resolve_from_clob(self, market: MarketState) -> Optional[str]:
        """
        Determine market outcome from CLOB orderbook.

        After resolution, winning token bid → ~1.0, losing token bid → ~0.0.
        This is more reliable than Chainlink snapshot because it reflects
        what Polymarket actually resolved, not our local price reading.

        Returns:
            "UP" or "DOWN" if clear winner, None if indeterminate.
        """
        try:
            clob = getattr(self, 'clob_feed', None)
            if not clob:
                return None

            up_token = market.up_token_id
            down_token = market.down_token_id
            if not up_token or not down_token:
                return None

            up_book = clob.get_orderbook(up_token)
            down_book = clob.get_orderbook(down_token)

            up_bid = 0.0
            down_bid = 0.0

            if up_book and up_book.bids:
                up_bid = up_book.bids[0].price
            if down_book and down_book.bids:
                down_bid = down_book.bids[0].price

            # Clear resolution: one side near 1.0, other near 0.0
            winner_thresh = self.config.settlement.clob_winner_threshold
            loser_thresh = self.config.settlement.clob_loser_threshold
            if up_bid >= winner_thresh and down_bid <= loser_thresh:
                logger.info(
                    "CLOB_RESOLVE market=%s asset=%s up_bid=%.2f down_bid=%.2f winner=UP",
                    market.condition_id[:8], market.asset, up_bid, down_bid,
                )
                return "UP"
            elif down_bid >= winner_thresh and up_bid <= loser_thresh:
                logger.info(
                    "CLOB_RESOLVE market=%s asset=%s up_bid=%.2f down_bid=%.2f winner=DOWN",
                    market.condition_id[:8], market.asset, up_bid, down_bid,
                )
                return "DOWN"

            # Also check using market.best_bid/best_ask (UP token prices from market state)
            if market.best_bid >= winner_thresh:
                logger.info(
                    "CLOB_RESOLVE market=%s asset=%s market_bid=%.2f winner=UP",
                    market.condition_id[:8], market.asset, market.best_bid,
                )
                return "UP"
            elif market.best_ask <= loser_thresh:
                logger.info(
                    "CLOB_RESOLVE market=%s asset=%s market_ask=%.2f winner=DOWN",
                    market.condition_id[:8], market.asset, market.best_ask,
                )
                return "DOWN"

            logger.debug(
                "CLOB_RESOLVE_INDETERMINATE market=%s asset=%s up_bid=%.2f down_bid=%.2f "
                "market_bid=%.2f market_ask=%.2f",
                market.condition_id[:8], market.asset,
                up_bid, down_bid, market.best_bid, market.best_ask,
            )
            return None

        except Exception as e:
            logger.debug("CLOB_RESOLVE_ERROR market=%s: %s", market.condition_id[:8], e)
            return None

    def _determine_outcome_locally(self, market: MarketState) -> tuple[Optional[str], Optional[float]]:
        """
        Determine market outcome locally using Chainlink price snapshot.

        Uses the expiry_chainlink_price (captured when market moved to expiring
        queue) to avoid using stale/moved live prices after market ends.

        For 15-minute crypto markets:
        - If ref_price >= target price: UP wins
        - If ref_price < target price: DOWN wins

        Args:
            market: Market to determine outcome for

        Returns:
            Tuple of (winning_outcome, ref_price) or (None, None) if can't determine
        """
        # Prefer snapshot price captured at expiry; fall back to live with warning
        ref_price = market.expiry_chainlink_price
        source = "snapshot"

        if ref_price is None or ref_price <= 0:
            ref_price = self.signal_generator.get_price(market.asset)
            source = "live_fallback"
            if ref_price is not None and ref_price > 0:
                logger.warning(
                    "LOCAL_RESOLVE_FALLBACK market=%s asset=%s using live price %.2f "
                    "(no expiry snapshot available)",
                    market.condition_id[:8], market.asset, ref_price,
                )

        if ref_price is None or ref_price <= 0:
            return (None, None)

        target_price = market.target_price
        if target_price is None or target_price <= 0:
            return (None, None)

        # Determine winner based on ref price vs target
        winner = "UP" if ref_price >= target_price else "DOWN"

        # Margin-of-safety check: if the margin is too thin relative to the
        # price, don't trust the snapshot — it was taken seconds before expiry
        # and tiny price movements can flip the outcome.
        margin_pct = abs(ref_price - target_price) / target_price if target_price > 0 else 0
        if margin_pct < self.config.settlement.local_margin_min:
            logger.warning(
                "LOCAL_RESOLVE_LOW_MARGIN market=%s asset=%s ref=%.2f target=%.2f "
                "margin=%.4f%% source=%s — refusing to resolve locally",
                market.condition_id[:8], market.asset, ref_price,
                target_price, margin_pct * 100, source,
            )
            return (None, None)

        logger.info(
            "LOCAL_RESOLVE market=%s asset=%s ref_price=%.2f target=%.2f "
            "winner=%s source=%s margin=%.3f%%",
            market.condition_id[:8], market.asset, ref_price,
            target_price, winner, source, margin_pct * 100,
        )

        return (winner, ref_price)

    async def _close_position_on_settlement(
        self,
        market: MarketState,
        position,
        winning_outcome: str,
    ):
        """
        Close a position based on market settlement using token-ID payout.

        Payout logic (binary market):
          winning_token = yes_token if winner==UP else no_token
          payout_per_share = 1.0 if held_token == winning_token else 0.0
          realized = (payout_per_share - entry_price) * shares

        Args:
            market: Settled market
            position: Our position in the market
            winning_outcome: "UP" or "DOWN"
        """
        # Idempotency guard: verify position still exists in risk manager
        if market.condition_id not in self.risk_manager.positions:
            logger.warning(
                "SETTLE_SKIP_CLOSED position already closed market=%s",
                market.condition_id[:16],
            )
            return

        position_side = position.side.value  # "UP" or "DOWN"

        # --- Token-ID based payout (authoritative) ---
        # Prefer position's own token IDs (captured at open time, immutable)
        # over market's current token IDs which could theoretically drift.
        pos_yes = position.yes_token_id
        pos_no = position.no_token_id

        if pos_yes and pos_no:
            winning_token_id = pos_yes if winning_outcome == "UP" else pos_no
        else:
            winning_token_id = (
                market.up_token_id if winning_outcome == "UP"
                else market.down_token_id
            )

        # Use explicit held_token_id if available, fall back to token_id
        held_token = position.held_token_id or position.token_id
        payout_per_share = 1.0 if held_token == winning_token_id else 0.0
        won = payout_per_share == 1.0

        # Cross-check: token-ID result should match side-string comparison
        side_match = (position_side == winning_outcome)
        if won != side_match:
            logger.error(
                "SETTLE_MISMATCH token-based won=%s but side-match=%s | "
                "side=%s winner=%s held=%s winning_token=%s "
                "pos_yes=%s pos_no=%s mkt_up=%s mkt_down=%s",
                won, side_match, position_side, winning_outcome,
                held_token[:16], winning_token_id[:16],
                (pos_yes or "")[:16], (pos_no or "")[:16],
                market.up_token_id[:16], market.down_token_id[:16],
            )

        # Calculate P&L
        pnl = (payout_per_share - position.entry_price) * position.shares

        # Audit log
        logger.info(
            "SETTLE_APPLY market_id=%s asset=%s variant=%s side=%s held_token=%s "
            "winning_token=%s payout=%.1f entry=%.4f shares=%.2f realized=%+.2f",
            market.condition_id[:8], market.asset, market.variant, position_side,
            held_token[:16], winning_token_id[:16],
            payout_per_share, position.entry_price, position.shares, pnl,
        )

        # Color and format the result
        result_color = Colors.BRIGHT_GREEN if won else Colors.BRIGHT_RED
        result_emoji = "🎉" if won else "💔"
        result_text = "WIN" if won else "LOSS"
        side_arrow = "▲" if position_side == "UP" else "▼"

        # Calculate ROI
        roi = (pnl / position.cost_basis * 100) if position.cost_basis > 0 else 0

        # Box format for position close
        logger.info(f"{result_color}╔══════════════════════════════════════════════════════════════════╗{Colors.RESET}")
        logger.info(f"{result_color}║  {result_emoji} POSITION CLOSED: {result_text:4}                                        ║{Colors.RESET}")
        logger.info(f"{result_color}╠══════════════════════════════════════════════════════════════════╣{Colors.RESET}")
        logger.info(f"{result_color}║{Colors.RESET}  Asset: {market.asset:4} {side_arrow}  │  Shares: {position.shares:>8.2f}  │  Entry: {position.entry_price:.4f}     {result_color}║{Colors.RESET}")
        logger.info(f"{result_color}║{Colors.RESET}  P&L: {result_color}${pnl:>+10.2f}{Colors.RESET}  │  ROI: {result_color}{roi:>+7.1f}%{Colors.RESET}  │  Cost: ${position.cost_basis:.2f}     {result_color}║{Colors.RESET}")
        logger.info(f"{result_color}╚══════════════════════════════════════════════════════════════════╝{Colors.RESET}")

        # Structured machine-parseable settlement log
        variant_label = _VARIANT_LABELS.get(market.variant, market.variant)
        logger.info(
            "SETTLEMENT_RESULT asset=%s variant=%s side=%s outcome=%s pnl=%+.2f "
            "roi=%+.1f%% cost=%.2f bankroll_after=%.2f",
            market.asset, variant_label, position_side, result_text,
            pnl, roi, position.cost_basis,
            self.risk_manager.current_bankroll + pnl,
        )

        # Populate recent settlements deque for compact table + dashboard
        if hasattr(self, '_recent_settlements'):
            self._recent_settlements.appendleft({
                "outcome": result_text,
                "asset": market.asset,
                "variant": variant_label,
                "side": position_side,
                "pnl": round(pnl, 2),
                "roi": round(roi, 1),
                "cost": round(position.cost_basis, 2),
                "timestamp": time.time(),
            })

        # Credit paper executor BEFORE updating risk manager so that
        # _sync_existing_orders() will read the correct balance.
        # proceeds = cost_basis + pnl  (e.g. WIN: $50 cost + $49 pnl = $99)
        settlement_proceeds = position.cost_basis + pnl
        if self.config.paper_trading.enabled and hasattr(self.executor, 'credit_settlement'):
            self.executor.credit_settlement(
                proceeds=settlement_proceeds,
                market_id=market.condition_id,
            )

        # Record with risk manager
        self.risk_manager.record_position_close(
            market_key=market.condition_id,
            exit_price=payout_per_share,
            pnl=pnl,
        )
        self.metrics.record_trade(market.asset, pnl)

        # Record with trade history
        trade_history = get_trade_history()
        trade_history.record_close(
            market_id=market.condition_id,
            won=won,
            pnl=pnl,
            exit_price=payout_per_share,
        )

        # (Legacy ML recording removed — EV engine handles learning separately)
