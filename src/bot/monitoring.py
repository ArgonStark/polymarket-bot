"""Monitoring, status logging, and dashboard snapshot mixin."""
from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from src.models import PositionState, Side
from src.utils import Colors

logger = logging.getLogger(__name__)


def _fmt_duration(seconds: float) -> str:
    """Format seconds into compact 'XmYs' or 'EXPIRED' string."""
    if seconds <= 0:
        return "EXPIRED"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m > 0 else f"{int(s)}s"


def _fmt_ago(seconds: float) -> str:
    """Format elapsed seconds as 'Xm ago' or 'Xs ago'."""
    s = int(seconds)
    if s >= 60:
        return f"{s // 60}m ago"
    return f"{s}s ago"


class MonitoringMixin:
    """Mixin for status logging, monitoring snapshots, and order block logging. Mixed into TradingBot."""

    def _log_compact_table(self):
        """Log a compact fixed-width position table every 10s.

        Shows asset prices with target distance, open positions with
        mark-to-market, cash/equity summary, and recent settlements.
        Placed AFTER signal processing in the tick loop so it's always
        the last visible block — never buried under per-market noise.
        """
        now_mono = time.monotonic()
        if now_mono - self._last_compact_table_ts < self._compact_table_interval:
            return
        self._last_compact_table_ts = now_mono

        positions = list(self.risk_manager.positions.items())
        bankroll = self.risk_manager.current_bankroll
        equity = self._calculate_equity()
        realized = self.risk_manager.daily_stats.total_pnl if self.risk_manager.daily_stats else 0.0
        unrealized = equity - bankroll
        total_pnl = realized + unrealized
        peak = self.risk_manager.peak_bankroll
        dd_pct = (1 - equity / peak) * 100 if peak > 0 else 0.0
        pos_count = len(positions)

        W = 68  # inner box width

        # ── Asset prices header ──
        chainlink_prices = self.chainlink_feed.get_all_prices()
        price_parts = []
        for asset in self.config.supported_assets:
            asset_key = f"{asset.lower()}/usd"
            cl_price = chainlink_prices.get(asset_key)
            if not cl_price:
                continue
            target = None
            for m in self.markets.values():
                if m.asset == asset:
                    target = m.target_price
                    break
            if target and target > 0:
                dist_pct = ((cl_price - target) / target) * 100
                arrow = "\u2191" if dist_pct > 0 else "\u2193"
                price_parts.append(f"{asset}=${cl_price:,.0f} {arrow}{abs(dist_pct):.2f}%")
            else:
                price_parts.append(f"{asset}=${cl_price:,.0f}")

        mkt_count = len(self.markets)
        five_count = sum(1 for m in self.markets.values() if m.variant == "five")
        fifteen_count = mkt_count - five_count
        mkt_str = f"{fifteen_count}\u00d715m"
        if five_count > 0:
            mkt_str += f"+{five_count}\u00d75m"

        logger.info("")  # blank line separator
        logger.info(
            "\u2500\u2500 %s | %s | %d pos \u2500\u2500",
            " ".join(price_parts) if price_parts else "no prices",
            mkt_str,
            pos_count,
        )

        # ── Positions ──
        if positions:
            logger.info("\u250c\u2500 POSITIONS %s\u2510", "\u2500" * (W - 12))
            for mk, pos in positions:
                market = self.markets.get(mk) or self.expiring_markets.get(mk)
                state = self._derive_position_state(pos, market)
                mark_price, source, _ = self._get_position_mark_price(pos, market, state)

                asset = pos.market.asset if pos.market else "?"
                variant = "5m" if getattr(pos.market, 'variant', 'fifteen') == "five" else "15m"
                side = pos.side.value  # UP / DOWN
                entry = pos.entry_price

                if mark_price is not None and mark_price >= 0:
                    upnl = (mark_price - entry) * pos.shares
                    roi = ((mark_price - entry) / entry * 100) if entry > 0 else 0.0
                    pnl_sign = "+" if upnl >= 0 else ""
                    roi_sign = "+" if roi >= 0 else ""
                    mark_str = f"{mark_price:.3f}"
                    pnl_str = f"{pnl_sign}${upnl:.2f} ({roi_sign}{roi:.1f}%)"
                else:
                    mark_str = " -- "
                    pnl_str = "(pending)"

                time_left = _fmt_duration(market.time_remaining) if market else "--"
                src_tag = source[:4] if source else "?"

                line = (
                    f"\u2502 {asset:4} {variant:3} {side:5} "
                    f"@{entry:.3f} \u2192 {mark_str:>5}  "
                    f"{pnl_str:>22}  {time_left:>6}  {src_tag:4} \u2502"
                )
                logger.info(line)

            logger.info("\u251c%s\u2524", "\u2500" * W)
        else:
            logger.info("\u250c%s\u2510", "\u2500" * W)

        # ── Signal processing diagnostics (when no positions, show WHY) ──
        blocks = getattr(self, '_tick_block_reasons', {})
        n_processed = getattr(self, '_tick_markets_processed', 0)
        n_signals = getattr(self, '_tick_signals_generated', 0)
        n_trades = getattr(self, '_tick_trades_executed', 0)
        last_trade_ts = getattr(self, '_last_trade_ts', None)

        if not positions and (blocks or n_processed > 0):
            # Format top blocking reasons (sorted by count, top 3)
            sorted_blocks = sorted(blocks.items(), key=lambda x: -x[1])[:3]
            block_str = " ".join(f"{r}:{c}" for r, c in sorted_blocks) if sorted_blocks else "none"

            # Time since last trade
            if last_trade_ts is not None:
                secs_ago = time.monotonic() - last_trade_ts
                if secs_ago < 60:
                    trade_ago = f"{int(secs_ago)}s ago"
                elif secs_ago < 3600:
                    trade_ago = f"{int(secs_ago // 60)}m ago"
                else:
                    trade_ago = f"{int(secs_ago // 3600)}h ago"
            else:
                trade_ago = "never"

            diag_line = (
                f"\u2502 proc={n_processed} sig={n_signals} trades={n_trades} "
                f"blocks=[{block_str}] last_trade={trade_ago}"
            )
            padding = max(0, W - len(diag_line) + 1)
            logger.info(f"{diag_line}{' ' * padding}\u2502")
            logger.info("\u251c%s\u2524", "\u2500" * W)

        # ── Account summary (always shown) ──
        summary = (
            f"\u2502 Cash: ${bankroll:.2f}  Equity: ${equity:.2f}  "
            f"P&L: {'+' if total_pnl >= 0 else ''}${total_pnl:.2f} "
            f"(R:{'+' if realized >= 0 else ''}{realized:.2f} "
            f"U:{'+' if unrealized >= 0 else ''}{unrealized:.2f})  "
            f"DD: {dd_pct:.1f}%"
        )
        padding = max(0, W - len(summary) + 1)
        logger.info(f"{summary}{' ' * padding}\u2502")
        logger.info("\u2514%s\u2518", "\u2500" * W)

        # ── Recent settlements ──
        settlements = list(getattr(self, '_recent_settlements', []))
        if settlements:
            logger.info("\u250c\u2500 SETTLEMENTS (last %d) %s\u2510", len(settlements), "\u2500" * (W - 22))
            now_ts = time.time()
            for s in settlements:
                outcome = s.get("outcome", "?")
                s_asset = s.get("asset", "?")
                variant = s.get("variant", "?")
                side = s.get("side", "?")
                pnl = s.get("pnl", 0.0)
                roi = s.get("roi", 0.0)
                ts = s.get("timestamp", now_ts)
                ago = _fmt_ago(now_ts - ts)

                tag = "WIN " if outcome == "WIN" else "LOSS"
                pnl_sign = "+" if pnl >= 0 else ""
                roi_sign = "+" if roi >= 0 else ""
                line = (
                    f"\u2502 {tag:4}  {s_asset:4}/{variant} {side:5}  "
                    f"{pnl_sign}${pnl:.2f} ({roi_sign}{roi:.0f}%)   {ago:>8}"
                )
                padding = max(0, W - len(line) + 1)
                logger.info(f"{line}{' ' * padding}\u2502")
            logger.info("\u2514%s\u2518", "\u2500" * W)

    def _log_status_line(self):
        """Log a clean status line with key info."""
        now = datetime.now(timezone.utc)
        bankroll = self.risk_manager.current_bankroll
        pos_count = len(self.risk_manager.positions)
        pending_count = len(self._pending_orders) if hasattr(self, '_pending_orders') else 0
        market_count = len(self.markets)

        # Get prices
        chainlink_prices = self.chainlink_feed.get_all_prices()
        binance_prices = self._get_binance_prices()

        # Build compact price string with distance info
        price_parts = []
        for asset in self.config.supported_assets:
            asset_key = f"{asset.lower()}/usd"
            cl_price = chainlink_prices.get(asset_key)
            bn_price = binance_prices.get(asset)

            # Find market for this asset to get target
            target = None
            for m in self.markets.values():
                if m.asset == asset:
                    target = m.target_price
                    break

            if cl_price:
                part = f"{asset}=${cl_price:,.0f}"
                if target and target > 0:
                    dist_pct = ((cl_price - target) / target) * 100
                    if dist_pct > 0:
                        part += f"↑{dist_pct:.1f}%"
                    else:
                        part += f"↓{abs(dist_pct):.1f}%"
                price_parts.append(part)

        # Count markets by variant
        five_count = sum(1 for m in self.markets.values() if m.variant == "five")
        fifteen_count = market_count - five_count
        if five_count > 0:
            mkt_str = f"{fifteen_count}×15m+{five_count}×5m"
        else:
            mkt_str = f"{market_count} mkts"

        # Single clean status line
        status = f"💰${bankroll:.0f} | {mkt_str} | {pos_count} pos | {pending_count} pnd"
        if price_parts:
            status += f" | {' '.join(price_parts)}"

        logger.info(status)

    def _build_monitor_snapshot(
        self,
        equity: float,
        realized_today: float,
        unrealized: float,
    ) -> dict:
        """Build a data dict for the real-time monitoring dashboard.

        Wrapped in try/except so a single attribute error never crashes the
        trading loop.
        """
        try:
            return self._build_monitor_snapshot_inner(equity, realized_today, unrealized)
        except Exception as e:
            logger.warning("MONITOR_SNAPSHOT_FAILED: %s", e)
            return {
                "bot_status": "ERROR",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }

    def _build_monitor_snapshot_inner(
        self,
        equity: float,
        realized_today: float,
        unrealized: float,
    ) -> dict:
        now = datetime.now(timezone.utc)
        bankroll = self.risk_manager.current_bankroll
        peak = self.risk_manager.peak_bankroll

        # Chainlink prices
        chainlink_prices = self.chainlink_feed.get_all_prices()
        binance_prices = self._get_binance_prices()

        # Pick the primary asset's price for the header
        primary_asset = self.config.supported_assets[0] if self.config.supported_assets else "BTC"
        asset_key = f"{primary_asset.lower()}/usd"
        price = chainlink_prices.get(asset_key, 0.0)

        # Build per-asset price info with signal pipeline diagnostics
        assets_info = []
        for asset in self.config.supported_assets:
            akey = f"{asset.lower()}/usd"
            cl = chainlink_prices.get(akey)
            bn = binance_prices.get(asset)
            if cl:
                target = None
                variant = None
                time_remaining = None
                for m in self.markets.values():
                    if m.asset == asset:
                        target = m.target_price
                        variant = m.variant
                        time_remaining = m.time_remaining
                        break
                dist_pct = ((cl - target) / target * 100) if target and target > 0 else 0.0

                # OFI data from aggTrade feed
                ofi_val = 0.0
                ofi_accel = 0.0
                trade_rate_val = 0.0
                ofi_age = None
                trade_feed = getattr(self, 'binance_trade_feed', None)
                if trade_feed is not None and trade_feed.is_connected:
                    ofi_window = self.config.ofi.default_window if hasattr(self.config, 'ofi') else 30.0
                    ofi_val = trade_feed.get_ofi(asset, ofi_window)
                    ofi_accel = trade_feed.get_ofi_acceleration(asset)
                    trade_rate_val = trade_feed.get_trade_rate(asset)
                    ofi_age = trade_feed.get_data_age(asset)

                # Regime data
                regime_type = "UNKNOWN"
                regime_er = 0.0
                regime_atr = 0.0
                regime_should_trade = True
                regime_kelly = 1.0
                regime_detector = getattr(self, 'regime_detector', None)
                if regime_detector is not None:
                    regime_state = regime_detector.get_regime(asset)
                    regime_type = regime_state.regime.value
                    regime_er = regime_state.efficiency_ratio
                    regime_atr = regime_state.atr_percentile
                    regime_should_trade = regime_state.should_trade
                    regime_kelly = regime_state.kelly_multiplier

                # Last edge signal info (from throttle log timestamps)
                last_signal_ts = getattr(self, '_edge_signal_log_ts', {}).get(asset)

                assets_info.append({
                    "asset": asset,
                    "chainlink": round(cl, 2),
                    "binance": round(bn, 2) if bn else None,
                    "target": round(target, 2) if target else None,
                    "distance_pct": round(dist_pct, 2),
                    "variant": variant or "fifteen",
                    "time_remaining": round(time_remaining, 0) if time_remaining else None,
                    # OFI
                    "ofi": round(ofi_val, 3),
                    "ofi_accel": round(ofi_accel, 3),
                    "trade_rate": round(trade_rate_val, 1),
                    "ofi_age": round(ofi_age, 1) if ofi_age is not None else None,
                    # Regime
                    "regime": regime_type,
                    "regime_er": round(regime_er, 3),
                    "regime_atr": round(regime_atr, 0),
                    "regime_trade": regime_should_trade,
                    "regime_kelly": round(regime_kelly, 2),
                    # Last signal
                    "last_signal_ts": round(last_signal_ts, 1) if last_signal_ts else None,
                })

        # Open positions — snapshot the dict to avoid RuntimeError during iteration
        open_positions = []
        for mk, pos in list(self.risk_manager.positions.items()):
            market = self.markets.get(mk) or self.expiring_markets.get(mk)
            mark_price, source, _ = self._get_position_mark_price(
                pos, market, self._derive_position_state(pos, market)
            )
            mark_val = pos.shares * mark_price if mark_price else pos.cost_basis
            pos_pnl = mark_val - pos.cost_basis
            pnl_pct = (pos_pnl / pos.cost_basis * 100) if pos.cost_basis > 0 else 0.0
            open_positions.append({
                "symbol": pos.market.asset if pos.market else "?",
                "side": pos.side.value if hasattr(pos.side, 'value') else str(pos.side),
                "entry_price": round(pos.entry_price, 4),
                "size": round(pos.shares, 2),
                "cost": round(pos.cost_basis, 2),
                "mark": round(mark_price, 4) if mark_price else None,
                "mark_source": source,
                "pnl": round(pos_pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "variant": getattr(pos.market, 'variant', 'fifteen') if pos.market else 'fifteen',
                "time_remaining": round(market.time_remaining, 0) if market else None,
            })

        # Recent closed trades (risk_manager.trade_history is a list of PnL floats)
        # Today's daily stats (session-level)
        daily = self.risk_manager.daily_stats

        # Pull cumulative stats from persistent trade history (survives restarts)
        from src.strategy.trade_history import get_trade_history
        persistent = get_trade_history()
        persistent_stats = persistent.get_stats()

        total_trades = persistent_stats.get("completed_trades", 0)
        wins = persistent_stats.get("wins", 0)
        losses = persistent_stats.get("losses", 0)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

        # Detailed trade stats from persistent history
        completed = [t for t in persistent.trades if t.get("actual_outcome") is not None]
        w_pnls = [t.get("pnl", 0) or 0 for t in completed if (t.get("pnl") or 0) > 0]
        l_pnls = [t.get("pnl", 0) or 0 for t in completed if (t.get("pnl") or 0) <= 0]
        trade_stats = {
            "avg_win": round(sum(w_pnls) / len(w_pnls), 2) if w_pnls else 0.0,
            "avg_loss": round(sum(l_pnls) / len(l_pnls), 2) if l_pnls else 0.0,
            "largest_win": round(max(w_pnls), 2) if w_pnls else 0.0,
            "largest_loss": round(min(l_pnls), 2) if l_pnls else 0.0,
        }

        # Recent trades from persistent history (last 10 completed)
        recent_trades = []
        for t in reversed(completed[-10:]):
            pnl_val = t.get("pnl", 0) or 0
            recent_trades.append({
                "side": t.get("actual_outcome", "LOSS"),
                "symbol": t.get("asset", primary_asset),
                "pnl": round(pnl_val, 2),
                "price": t.get("exit_price", 0),
                "entry_price": t.get("entry_price", 0),
                "variant": t.get("variant", "fifteen"),
            })

        # Sample equity for chart
        if hasattr(self, 'dashboard'):
            self.dashboard.record_equity_sample(equity, bankroll)

        # Bot status
        pauses = getattr(self, '_active_pauses', {})
        kill_active = self.kill_switch.is_active()
        if kill_active:
            bot_status = "KILLED"
        elif pauses:
            bot_status = "PAUSED"
        elif not self._running:
            bot_status = "STOPPED"
        else:
            bot_status = "RUNNING"

        # Data feed health (detailed)
        clob_ok = self.clob_feed.is_connected and self.clob_feed.is_warmed_up
        chainlink_ok = not self.chainlink_feed._circuit_open
        trade_feed = getattr(self, 'binance_trade_feed', None)
        binance_trade_ok = trade_feed is not None and trade_feed.is_connected
        binance_price_ok = not getattr(self.binance_feed, '_circuit_open', True)

        feed_health = {
            "clob": {"ok": clob_ok, "connected": self.clob_feed.is_connected, "warmed_up": self.clob_feed.is_warmed_up},
            "chainlink": {"ok": chainlink_ok, "connected": chainlink_ok},
            "binance_price": {"ok": binance_price_ok, "connected": binance_price_ok},
            "binance_trades": {"ok": binance_trade_ok, "connected": binance_trade_ok},
        }

        # Capital scaling info
        cap_scale_mult = 1.0
        cap_scale_reason = "normal"
        cap_scale_dd = 0.0
        try:
            cap_scale_mult, cap_scale_reason = self.capital_scaler.compute_multiplier(
                peak_bankroll=peak,
                current_equity=equity,
                recent_pnls=list(self.risk_manager.trade_history)[-20:] if hasattr(self.risk_manager, 'trade_history') else [],
            )
            cap_scale_dd = (1 - equity / peak) if peak > 0 else 0.0
        except Exception:
            pass

        # Kill switch / pause state
        pauses_detail = []
        for reason, ps in getattr(self, '_active_pauses', {}).items():
            expiry_s = None
            if hasattr(ps, 'expiry') and ps.expiry:
                expiry_s = max(0, (ps.expiry - now).total_seconds())
            pauses_detail.append({"reason": reason, "expiry_s": round(expiry_s, 0) if expiry_s else None})

        return {
            "symbol": primary_asset,
            "price": round(price, 2),
            "assets": assets_info,
            "bot_status": bot_status,
            "timestamp": now.isoformat(),
            "mode": self._execution_mode,

            # Account
            "bankroll": round(bankroll, 2),
            "equity": round(equity, 2),
            "peak_bankroll": round(peak, 2),
            "drawdown_pct": round((1 - equity / peak) * 100, 2) if peak > 0 else 0.0,
            "total_exposure": round(self.risk_manager.get_total_exposure(), 2),

            # PnL
            "total_pnl": round(persistent_stats.get("total_pnl", 0.0), 2),
            "realized_pnl": round(realized_today, 2),
            "unrealized_pnl": round(unrealized, 2),
            "win_rate": round(win_rate, 1),
            "total_trades": total_trades,
            "total_wins": wins,
            "total_losses": losses,

            # Today's stats (from daily_stats)
            "today_trades": daily.trades_count if daily else 0,
            "today_wins": daily.wins if daily else 0,
            "today_losses": daily.losses if daily else 0,
            "today_pnl": round(realized_today, 2),

            # Detailed trade stats (cumulative from persistent history)
            "avg_win": trade_stats.get("avg_win", 0.0),
            "avg_loss": trade_stats.get("avg_loss", 0.0),
            "largest_win": trade_stats.get("largest_win", 0.0),
            "largest_loss": trade_stats.get("largest_loss", 0.0),

            # Equity history for chart
            "equity_history": self.dashboard.equity_history if hasattr(self, 'dashboard') else [],

            # Positions & trades
            "open_positions": open_positions[:10],
            "recent_trades": recent_trades[:10],
            "recent_settlements": list(getattr(self, '_recent_settlements', [])),

            # Markets
            "active_markets": len(self.markets),
            "expiring_markets": len(self.expiring_markets),
            "pending_orders": len(self._pending_orders) if hasattr(self, '_pending_orders') else 0,

            # Feed health
            "clob_connected": clob_ok,
            "chainlink_connected": chainlink_ok,
            "feed_health": feed_health,

            # Capital scaling
            "capital_scale_mult": round(cap_scale_mult, 2),
            "capital_scale_reason": cap_scale_reason,
            "capital_scale_dd": round(cap_scale_dd * 100, 2),

            # Kill switch / pauses
            "kill_switch_active": kill_active,
            "kill_switch_reason": self.kill_switch.reason if kill_active else None,
            "pauses": pauses_detail,

            # Signal processing diagnostics
            "tick_blocks": dict(getattr(self, '_tick_block_reasons', {})),
            "tick_markets_processed": getattr(self, '_tick_markets_processed', 0),
            "tick_signals_generated": getattr(self, '_tick_signals_generated', 0),
            "tick_trades_executed": getattr(self, '_tick_trades_executed', 0),

            # Cooldowns
            "active_cooldowns": self._get_active_cooldowns_info(now),
        }

    def _get_active_cooldowns_info(self, now: datetime) -> list:
        """Return active cooldowns for dashboard display."""
        cooldowns = []
        for av_key, last_time in getattr(self, '_last_order_time', {}).items():
            elapsed = (now - last_time).total_seconds()
            remaining = self._order_cooldown_seconds - elapsed
            if remaining > 0:
                cooldowns.append({"key": av_key, "remaining_s": round(remaining, 0)})
        return cooldowns

    def _log_order_block(self, asset: str, reason: str, **kwargs) -> bool:
        """
        Log ORDER_BLOCK with throttling to prevent spam.

        Rate-limits identical messages per market to once per 5s while
        tracking suppressed count internally.

        Args:
            asset: Asset symbol
            reason: Block reason
            **kwargs: Additional log fields

        Returns:
            True if logged, False if throttled
        """
        block_key = f"{asset}:{reason}"
        now_ts = time.time()
        last_log = self._order_block_log_ts.get(block_key, 0.0)

        if now_ts - last_log < self._order_block_log_interval:
            # Track suppressed count
            if not hasattr(self, '_order_block_suppressed'):
                self._order_block_suppressed = {}
            self._order_block_suppressed[block_key] = (
                self._order_block_suppressed.get(block_key, 0) + 1
            )
            return False  # Throttled

        self._order_block_log_ts[block_key] = now_ts

        # Prevent unbounded growth — evict entries older than 10 minutes
        if len(self._order_block_log_ts) > 200:
            cutoff = now_ts - 600.0
            self._order_block_log_ts = {
                k: v for k, v in self._order_block_log_ts.items() if v > cutoff
            }

        # Include suppressed count if any
        suppressed = 0
        if hasattr(self, '_order_block_suppressed'):
            suppressed = self._order_block_suppressed.pop(block_key, 0)

        # Build log message with extra fields
        extra_parts = " ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
        suppressed_str = f" suppressed={suppressed}" if suppressed > 0 else ""
        if extra_parts:
            logger.info(f"ORDER_BLOCK asset={asset} reason={reason} {extra_parts}{suppressed_str}")
        else:
            logger.info(f"ORDER_BLOCK asset={asset} reason={reason}{suppressed_str}")
        return True

    def _log_position_status(self):
        """
        Log detailed position status including all tracking state.

        Called periodically (every 30s) to give visibility into:
        - Tracked positions (risk_manager.positions)
        - API-discovered positions (_api_positions)
        - Pending orders (_pending_orders)
        - Active cooldowns (_last_order_time)
        """
        now = datetime.now(timezone.utc)

        # Check if it's time to log
        if self.last_position_log:
            elapsed = (now - self.last_position_log).total_seconds()
            if elapsed < self.position_log_interval:
                return
        self.last_position_log = now

        # Snapshot dicts to avoid RuntimeError during concurrent modification
        positions = dict(self.risk_manager.positions)
        api_positions = dict(getattr(self, '_api_positions', {}))
        pending_orders = dict(getattr(self, '_pending_orders', {}))

        # Calculate equity with debug logging
        equity = self._calculate_equity(debug_log=True)
        detail = getattr(self, '_last_equity_detail', {})
        cash = detail.get("cash", self.risk_manager.current_bankroll)
        total_pnl = detail.get("total_pnl", 0.0)
        self.metrics.record_position(
            positions=len(self.risk_manager.positions),
            exposure=self.risk_manager.get_total_exposure(),
            equity=equity,
        )

        # Format P&L (realized + unrealized)
        if total_pnl > 0:
            pnl_color = Colors.BRIGHT_GREEN
            pnl_icon = "📈"
        elif total_pnl < 0:
            pnl_color = Colors.BRIGHT_RED
            pnl_icon = "📉"
        else:
            pnl_color = Colors.DIM
            pnl_icon = "➖"

        # Box header
        box_width = 64
        logger.info(f"{Colors.BRIGHT_CYAN}┌{'─' * box_width}┐{Colors.RESET}")
        logger.info(f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  📊 POSITION STATUS                                              {Colors.BRIGHT_CYAN}│{Colors.RESET}")
        logger.info(f"{Colors.BRIGHT_CYAN}├{'─' * box_width}┤{Colors.RESET}")

        # Account summary line
        logger.info(
            f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  💰 Cash: {Colors.BRIGHT_WHITE}${cash:>10.2f}{Colors.RESET}  │  "
            f"💎 Equity: {Colors.BRIGHT_WHITE}${equity:>10.2f}{Colors.RESET}  │  "
            f"{pnl_icon} P&L: {pnl_color}${total_pnl:>+9.2f}{Colors.RESET}  {Colors.BRIGHT_CYAN}│{Colors.RESET}"
        )

        # Counts summary
        pos_count = len(positions)
        api_count = len(api_positions)
        pending_count = len(pending_orders)
        logger.info(
            f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  📦 Positions: {Colors.BRIGHT_YELLOW}{pos_count}{Colors.RESET}  │  "
            f"🔗 API: {Colors.BRIGHT_YELLOW}{api_count}{Colors.RESET}  │  "
            f"⏳ Pending: {Colors.BRIGHT_YELLOW}{pending_count}{Colors.RESET}                       {Colors.BRIGHT_CYAN}│{Colors.RESET}"
        )
        logger.info(f"{Colors.BRIGHT_CYAN}├{'─' * box_width}┤{Colors.RESET}")

        # Log tracked positions with P&L and win probability
        if positions:
            for market_key, position in positions.items():
                market = self.markets.get(market_key) or self.expiring_markets.get(market_key)
                asset = position.market.asset if hasattr(position, 'market') else "???"
                side = position.side.value
                time_remaining = market.time_remaining if market else 0

                # === DERIVE POSITION STATE ===
                # Check if we have a resolved outcome from settlement verification
                resolved_outcome = None  # TODO: Get from settlement_verifier when available
                state = self._derive_position_state(position, market, resolved_outcome)

                # === GET MARK PRICE BASED ON STATE ===
                mark_price, mark_source, mark_debug = self._get_position_mark_price(position, market, state)

                # === COMPUTE UNREALIZED PNL ===
                unrealized_pnl = None
                pnl_pct = None
                if mark_price is not None and mark_price >= 0:
                    unrealized_pnl = (mark_price - position.entry_price) * position.shares
                    if position.entry_price > 0:
                        pnl_pct = ((mark_price - position.entry_price) / position.entry_price * 100)

                # === CALCULATE WIN PROBABILITY ===
                sg = getattr(self, 'signal_generator', None)
                chainlink_price = sg.get_price(asset) if sg else None
                target_price = market.target_price if market else None
                our_win_prob = 0.5  # Default

                if chainlink_price and target_price and market and time_remaining > 0 and sg:
                    from src.probability import calculate_true_probability
                    volatility = sg.get_volatility(asset)
                    true_prob_up = calculate_true_probability(
                        current_price=chainlink_price,
                        target_price=target_price,
                        time_remaining_sec=time_remaining,
                        volatility_15min=volatility,
                    )
                    our_win_prob = true_prob_up if side == "UP" else (1 - true_prob_up)
                elif state == PositionState.RESOLVED_WIN:
                    our_win_prob = 1.0
                elif state == PositionState.RESOLVED_LOSS:
                    our_win_prob = 0.0

                # Color and icon based on probability
                if our_win_prob >= 0.7:
                    prob_color = Colors.BRIGHT_GREEN
                    prob_icon = "🟢"
                elif our_win_prob >= 0.5:
                    prob_color = Colors.BRIGHT_YELLOW
                    prob_icon = "🟡"
                else:
                    prob_color = Colors.BRIGHT_RED
                    prob_icon = "🔴"

                # === STATUS STRING BASED ON STATE ===
                if state == PositionState.RESOLVED_WIN:
                    status_str = "✅ WON"
                elif state == PositionState.RESOLVED_LOSS:
                    status_str = "❌ LOST"
                elif state == PositionState.ENDED_UNRESOLVED:
                    status_str = "⏰ PENDING"
                elif time_remaining > 60:
                    status_str = f"⏱️  {int(time_remaining // 60)}m {int(time_remaining % 60)}s"
                else:
                    status_str = f"⏱️  {int(time_remaining)}s"

                # === EXPLICIT DEBUG LOG FOR EACH POSITION ROW ===
                token_id_used = mark_debug.get("token_id_used", "")
                dbg_bid = mark_debug.get("bid")
                dbg_ask = mark_debug.get("ask")
                dbg_mid = mark_debug.get("mid")
                is_pending = state in (
                    PositionState.ENDED_UNRESOLVED,
                    PositionState.RESOLVED_WIN,
                    PositionState.RESOLVED_LOSS,
                )
                logger.info(
                    "POSITION_VALUATION market=%s asset=%s variant=%s side=%s "
                    "token_id_used=%s bid=%s ask=%s mid=%s "
                    "mark_price=%s mark_source=%s "
                    "entry=%.4f shares=%.2f unrealized_pnl=%s realized_pnl=0.00 "
                    "state=%s pending_settlement=%s",
                    market_key[:8] if market_key else "????????",
                    asset,
                    getattr(position.market, 'variant', 'fifteen') if position.market else 'fifteen',
                    side,
                    token_id_used[:16] if token_id_used else "None",
                    f"{dbg_bid:.4f}" if dbg_bid is not None else "None",
                    f"{dbg_ask:.4f}" if dbg_ask is not None else "None",
                    f"{dbg_mid:.4f}" if dbg_mid is not None else "None",
                    f"{mark_price:.4f}" if mark_price is not None else "None",
                    mark_source,
                    position.entry_price,
                    position.shares,
                    f"{unrealized_pnl:+.2f}" if unrealized_pnl is not None else "pending",
                    state.value,
                    "true" if is_pending else "false",
                )

                # Side arrow
                side_arrow = "▲" if side == "UP" else "▼"
                side_color = Colors.BRIGHT_GREEN if side == "UP" else Colors.BRIGHT_RED

                if mark_price is not None and unrealized_pnl is not None:
                    pnl_color = Colors.BRIGHT_GREEN if unrealized_pnl > 0 else Colors.BRIGHT_RED

                    # Format mark price display
                    if state == PositionState.RESOLVED_WIN:
                        mark_display = "1.000"
                    elif state == PositionState.RESOLVED_LOSS:
                        mark_display = "0.000"
                    else:
                        mark_display = f"{mark_price:.3f}"

                    variant_tag = "5m" if getattr(position.market, 'variant', 'fifteen') == "five" else "15"
                    logger.info(
                        f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  {side_color}{side_arrow} {asset:4}{Colors.RESET}{Colors.DIM}/{variant_tag}{Colors.RESET} │ "
                        f"{position.shares:>6.1f} @ {position.entry_price:.3f} → {mark_display} │ "
                        f"{pnl_color}{unrealized_pnl:>+7.2f} ({pnl_pct:>+5.0f}%){Colors.RESET} │ "
                        f"{prob_color}{prob_icon} {our_win_prob:>3.0%}{Colors.RESET} │ {status_str}  {Colors.BRIGHT_CYAN}│{Colors.RESET}"
                    )
                else:
                    # No valid mark - show pending
                    variant_tag = "5m" if getattr(position.market, 'variant', 'fifteen') == "five" else "15"
                    logger.info(
                        f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  {side_color}{side_arrow} {asset:4}{Colors.RESET}{Colors.DIM}/{variant_tag}{Colors.RESET} │ "
                        f"{position.shares:>6.1f} @ {position.entry_price:.3f} → {'--':>5} │ "
                        f"{'(pending)':^16} │ "
                        f"{prob_color}{prob_icon} {our_win_prob:>3.0%}{Colors.RESET} │ {status_str}  {Colors.BRIGHT_CYAN}│{Colors.RESET}"
                    )
        else:
            logger.info(f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  {Colors.DIM}No active positions{Colors.RESET}                                            {Colors.BRIGHT_CYAN}│{Colors.RESET}")

        # Log API-discovered positions (if any not already tracked)
        tracked_assets = [p.market.asset for p in positions.values() if hasattr(p, 'market')]
        untracked_api = {k: v for k, v in api_positions.items() if k not in tracked_assets}
        if untracked_api:
            logger.info(f"{Colors.BRIGHT_CYAN}├{'─' * 64}┤{Colors.RESET}")
            logger.info(f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  🔗 API Positions (untracked):                                   {Colors.BRIGHT_CYAN}│{Colors.RESET}")
            for asset, pos_info in untracked_api.items():
                logger.info(
                    f"{Colors.BRIGHT_CYAN}│{Colors.RESET}     {asset}: {pos_info.get('size', 0):.1f} shares @ "
                    f"${pos_info.get('price', 0):.2f}                                {Colors.BRIGHT_CYAN}│{Colors.RESET}"
                )

        # Log pending orders
        if pending_orders:
            logger.info(f"{Colors.BRIGHT_CYAN}├{'─' * 64}┤{Colors.RESET}")
            logger.info(f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  ⏳ Pending Orders:                                               {Colors.BRIGHT_CYAN}│{Colors.RESET}")
            for order_id, order_data in pending_orders.items():
                asset = order_data.get("asset", "???")
                side = order_data.get("side", "???")
                size = order_data.get("size", 0)
                limit_price = order_data.get("price", 0)
                placed_time = order_data.get("placed_time")
                age_s = int((now - placed_time).total_seconds()) if placed_time else 0

                # Try to get live status from executor
                order_status = "OPEN"
                filled = 0.0
                if self.executor and hasattr(self.executor, 'get_order_status'):
                    oi = self.executor.get_order_status(order_id)
                    if oi:
                        order_status = oi.status.value
                        filled = oi.filled_size

                logger.info(
                    f"{Colors.BRIGHT_CYAN}│{Colors.RESET}     {asset} {side}: {size:.1f} @ {limit_price:.3f} │ "
                    f"{order_status} │ filled={filled:.2f} │ age={age_s}s │ "
                    f"id={order_id[:12]}  {Colors.BRIGHT_CYAN}│{Colors.RESET}"
                )

        # Log active cooldowns
        active_cooldowns = []
        for av_key, last_time in self._last_order_time.items():
            elapsed = (now - last_time).total_seconds()
            remaining = self._order_cooldown_seconds - elapsed
            if remaining > 0:
                active_cooldowns.append(f"{av_key}:{remaining:.0f}s")

        if active_cooldowns:
            logger.info(f"{Colors.BRIGHT_CYAN}├{'─' * 64}┤{Colors.RESET}")
            cooldown_str = ", ".join(active_cooldowns)
            logger.info(f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  ⏱️  Cooldowns: {cooldown_str:<48}{Colors.BRIGHT_CYAN}│{Colors.RESET}")

        # On-chain stats from The Graph
        if self.settlement_verifier.is_enabled():
            graph_stats = self._get_graph_stats()
            if graph_stats:
                logger.info(f"{Colors.BRIGHT_CYAN}├{'─' * 64}┤{Colors.RESET}")
                logger.info(f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  📈 On-Chain Stats (The Graph):                                  {Colors.BRIGHT_CYAN}│{Colors.RESET}")
                wins = graph_stats.get("winning_redemptions", 0)
                losses = graph_stats.get("losing_redemptions", 0)
                total_payouts = graph_stats.get("total_redemption_payout_usdc", 0)
                win_rate = graph_stats.get("redemption_win_rate", 0) * 100
                splits = graph_stats.get("total_splits", 0)

                # Win rate color
                if win_rate >= 60:
                    wr_color = Colors.BRIGHT_GREEN
                elif win_rate >= 50:
                    wr_color = Colors.BRIGHT_YELLOW
                else:
                    wr_color = Colors.BRIGHT_RED

                logger.info(
                    f"{Colors.BRIGHT_CYAN}│{Colors.RESET}     Trades: {Colors.BRIGHT_WHITE}{splits}{Colors.RESET}  │  "
                    f"Wins: {Colors.BRIGHT_GREEN}{wins}{Colors.RESET}  │  "
                    f"Losses: {Colors.BRIGHT_RED}{losses}{Colors.RESET}  │  "
                    f"Win Rate: {wr_color}{win_rate:.0f}%{Colors.RESET}          {Colors.BRIGHT_CYAN}│{Colors.RESET}"
                )
                logger.info(
                    f"{Colors.BRIGHT_CYAN}│{Colors.RESET}     Total Payouts: {Colors.BRIGHT_GREEN}${total_payouts:,.2f}{Colors.RESET}                                     {Colors.BRIGHT_CYAN}│{Colors.RESET}"
                )

        # Box footer
        logger.info(f"{Colors.BRIGHT_CYAN}└{'─' * 64}┘{Colors.RESET}")

    def _get_graph_stats(self) -> Optional[dict]:
        """
        Get user stats from The Graph with caching.

        Returns cached stats or fetches new ones if cache expired.
        """
        now = datetime.now(timezone.utc)

        # Check if we have cached stats that are still fresh
        if (self._graph_stats_cache is not None and
            self._graph_stats_last_fetch is not None):
            elapsed = (now - self._graph_stats_last_fetch).total_seconds()
            if elapsed < self._graph_stats_refresh_interval:
                return self._graph_stats_cache

        # Fetch new stats
        try:
            if not self.client:
                return None
            wallet_address = self.client.get_address()
            if not wallet_address:
                return None

            graph = self.settlement_verifier.graph
            if not graph:
                return None

            stats = graph.calculate_user_stats(wallet_address)
            self._graph_stats_cache = stats
            self._graph_stats_last_fetch = now
            return stats

        except Exception as e:
            logger.debug(f"Failed to fetch Graph stats: {e}")
            return self._graph_stats_cache  # Return stale cache on error

    def get_status(self) -> dict:
        """Get current bot status."""
        return {
            "running": self._running,
            "chainlink_connected": self.chainlink_feed.is_connected,
            "clob_connected": self.clob_feed.is_connected,
            "active_markets": len(self.markets),
            "expiring_markets": len(self.expiring_markets),
            "settled_count": len(self.settled_markets),
            "chainlink_prices": self.chainlink_feed.get_all_prices(),
            "risk_status": self.risk_manager.get_status_summary(),
            "config": self.config.to_dict(),
        }
