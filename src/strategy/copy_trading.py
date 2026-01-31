"""
Copy Trading Module

Monitor and copy trades from successful Polymarket traders.

Features:
- Track multiple wallet addresses
- Analyze trader performance (win rate, P&L, etc.)
- Copy trades with configurable delay and position sizing
- Filter for 15-min crypto markets only
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Set
from enum import Enum

from ..data.polymarket_data import get_data_api

logger = logging.getLogger(__name__)


# 15-min crypto market patterns
CRYPTO_15M_PATTERNS = ["updown-15m", "15m", "15-min"]
CRYPTO_ASSETS = ["btc", "eth", "sol", "xrp"]


@dataclass
class TraderStats:
    """Statistics for a trader we're tracking."""
    wallet: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    last_updated: Optional[datetime] = None

    # 15-min crypto specific stats
    crypto_15m_trades: int = 0
    crypto_15m_wins: int = 0
    crypto_15m_pnl: float = 0.0

    @property
    def win_rate(self) -> float:
        """Overall win rate."""
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades

    @property
    def crypto_win_rate(self) -> float:
        """Win rate for 15-min crypto markets."""
        if self.crypto_15m_trades == 0:
            return 0.0
        return self.crypto_15m_wins / self.crypto_15m_trades

    @property
    def avg_pnl_per_trade(self) -> float:
        """Average P&L per trade."""
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades


@dataclass
class TrackedTrade:
    """A trade we've seen from a tracked wallet."""
    wallet: str
    timestamp: datetime
    condition_id: str
    asset: str
    side: str  # "BUY" or "SELL"
    outcome: str  # "Yes" or "No"
    size: float
    price: float
    usdc_size: float
    slug: str
    tx_hash: str

    # Copy status
    copied: bool = False
    copy_timestamp: Optional[datetime] = None
    our_order_id: Optional[str] = None


@dataclass
class CopyTradingConfig:
    """Configuration for copy trading."""
    # Wallets to copy from
    wallets: List[str] = field(default_factory=list)

    # Minimum stats required to copy
    min_win_rate: float = 0.55  # 55% win rate
    min_trades: int = 10  # At least 10 trades history
    min_crypto_trades: int = 5  # At least 5 crypto trades

    # Position sizing
    copy_size_mode: str = "fixed"  # "fixed", "proportional", "kelly"
    fixed_size_usd: float = 10.0  # Fixed USD size per copy
    proportional_multiplier: float = 0.5  # Copy at 50% of their size

    # Timing - FAST MODE
    max_copy_delay_seconds: int = 10  # Only copy trades within 10s (was 30)
    poll_interval_seconds: float = 0.5  # Check every 500ms for speed

    # Filters
    only_crypto_15m: bool = True  # Only copy 15-min crypto markets
    only_buys: bool = True  # Only copy BUY trades (not sells/exits)

    # Safety
    max_copies_per_period: int = 3  # Max 3 copy trades per 15-min period
    enabled: bool = False  # Disabled by default

    # Fast mode settings
    fast_mode: bool = True  # Use aggressive polling
    skip_trader_validation: bool = False  # Skip win rate check for speed


class CopyTradingManager:
    """
    Manages copy trading from tracked wallets.

    Monitors specified wallets for trades and copies them
    when they meet our criteria.
    """

    def __init__(self, config: CopyTradingConfig):
        self.config = config
        self.data_api = get_data_api()

        # Trader stats cache
        self.trader_stats: Dict[str, TraderStats] = {}

        # Seen trades (to avoid duplicates)
        self.seen_trades: Set[str] = set()  # tx_hash set

        # Recent trades to potentially copy
        self.pending_copies: List[TrackedTrade] = []

        # Copy count per period
        self.copies_this_period: int = 0
        self.period_start: Optional[int] = None

        # Last poll time per wallet
        self.last_poll: Dict[str, float] = {}

    def add_wallet(self, wallet: str):
        """Add a wallet to track."""
        wallet = wallet.lower()
        if wallet not in self.config.wallets:
            self.config.wallets.append(wallet)
            logger.info(f"Added wallet to copy trading: {wallet[:10]}...")

    def remove_wallet(self, wallet: str):
        """Remove a wallet from tracking."""
        wallet = wallet.lower()
        if wallet in self.config.wallets:
            self.config.wallets.remove(wallet)
            logger.info(f"Removed wallet from copy trading: {wallet[:10]}...")

    def analyze_trader(self, wallet: str) -> TraderStats:
        """
        Analyze a trader's historical performance.

        Fetches closed positions and calculates win rate, P&L, etc.
        """
        wallet = wallet.lower()

        # Check cache
        if wallet in self.trader_stats:
            stats = self.trader_stats[wallet]
            if stats.last_updated:
                age = (datetime.now(timezone.utc) - stats.last_updated).total_seconds()
                if age < 300:  # Cache for 5 minutes
                    return stats

        logger.info(f"Analyzing trader: {wallet[:10]}...")

        stats = TraderStats(wallet=wallet)

        try:
            # Get closed positions (historical performance)
            closed_positions = self.data_api.get_all_closed_positions(wallet, max_positions=200)

            for pos in closed_positions:
                slug = (pos.get("slug", "") or "").lower()
                title = (pos.get("title", "") or "").lower()
                pnl = float(pos.get("realizedPnl", 0) or 0)

                stats.total_trades += 1
                stats.total_pnl += pnl

                if pnl > 0:
                    stats.wins += 1
                else:
                    stats.losses += 1

                # Check if it's a 15-min crypto market
                is_crypto_15m = any(p in slug for p in CRYPTO_15M_PATTERNS)
                is_crypto = any(a in slug or a in title for a in CRYPTO_ASSETS)

                if is_crypto_15m and is_crypto:
                    stats.crypto_15m_trades += 1
                    stats.crypto_15m_pnl += pnl
                    if pnl > 0:
                        stats.crypto_15m_wins += 1

            stats.last_updated = datetime.now(timezone.utc)
            self.trader_stats[wallet] = stats

            logger.info(
                f"Trader {wallet[:10]}...: "
                f"{stats.total_trades} trades, {stats.win_rate:.0%} win rate, "
                f"${stats.total_pnl:+.2f} P&L | "
                f"15m crypto: {stats.crypto_15m_trades} trades, {stats.crypto_win_rate:.0%} win rate"
            )

        except Exception as e:
            logger.error(f"Failed to analyze trader {wallet[:10]}...: {e}")

        return stats

    def should_copy_trader(self, wallet: str) -> tuple[bool, str]:
        """
        Check if we should copy trades from this trader.

        Returns:
            Tuple of (should_copy, reason)
        """
        stats = self.analyze_trader(wallet)

        if stats.total_trades < self.config.min_trades:
            return False, f"Not enough trades ({stats.total_trades} < {self.config.min_trades})"

        if stats.win_rate < self.config.min_win_rate:
            return False, f"Win rate too low ({stats.win_rate:.0%} < {self.config.min_win_rate:.0%})"

        if self.config.only_crypto_15m:
            if stats.crypto_15m_trades < self.config.min_crypto_trades:
                return False, f"Not enough crypto trades ({stats.crypto_15m_trades} < {self.config.min_crypto_trades})"

            if stats.crypto_win_rate < self.config.min_win_rate:
                return False, f"Crypto win rate too low ({stats.crypto_win_rate:.0%})"

        return True, f"Good trader: {stats.crypto_win_rate:.0%} crypto win rate, ${stats.crypto_15m_pnl:+.2f} P&L"

    def poll_wallet_fast(self, wallet: str) -> List[TrackedTrade]:
        """
        Fast poll a wallet for new trades - optimized for speed.

        Uses minimal processing for fastest detection.
        """
        wallet = wallet.lower()
        now = time.time()

        # No rate limiting in fast mode - poll as fast as possible
        if not self.config.fast_mode:
            last = self.last_poll.get(wallet, 0)
            if now - last < self.config.poll_interval_seconds:
                return []

        self.last_poll[wallet] = now

        new_trades = []

        try:
            # Get only last 5 seconds of activity for speed
            start_ts = int(now - 5)
            activity = self.data_api.get_activity(
                user=wallet,
                limit=5,  # Only check last 5 trades
                activity_type="TRADE",
                side="BUY" if self.config.only_buys else None,
                start=start_ts,
            )

            for trade in activity:
                tx_hash = trade.get("transactionHash", "")
                if not tx_hash or tx_hash in self.seen_trades:
                    continue

                self.seen_trades.add(tx_hash)

                # Quick parse - minimal processing
                slug = (trade.get("slug", "") or "").lower()
                if self.config.only_crypto_15m:
                    if not any(p in slug for p in CRYPTO_15M_PATTERNS):
                        continue

                # Quick asset detection
                asset = None
                for a in CRYPTO_ASSETS:
                    if a in slug:
                        asset = a.upper()
                        break
                if not asset:
                    continue

                # Build trade object
                tracked = TrackedTrade(
                    wallet=wallet,
                    timestamp=datetime.now(timezone.utc),  # Use now for speed
                    condition_id=trade.get("conditionId", ""),
                    asset=asset,
                    side=trade.get("side", ""),
                    outcome=trade.get("outcome", ""),
                    size=float(trade.get("size", 0) or 0),
                    price=float(trade.get("price", 0) or 0),
                    usdc_size=float(trade.get("usdcSize", 0) or 0),
                    slug=slug,
                    tx_hash=tx_hash,
                )

                new_trades.append(tracked)
                detect_time = int((time.time() - now) * 1000)
                logger.info(
                    f"⚡ FAST COPY [{asset}]: {wallet[:6]}... {trade.get('side')} | "
                    f"Detected in {detect_time}ms"
                )

        except Exception as e:
            logger.debug(f"Fast poll error: {e}")

        return new_trades

    def poll_wallet(self, wallet: str) -> List[TrackedTrade]:
        """
        Poll a wallet for new trades.

        Returns list of new trades since last poll.
        """
        # Use fast mode if enabled
        if self.config.fast_mode:
            return self.poll_wallet_fast(wallet)

        wallet = wallet.lower()
        now = time.time()

        # Rate limit polling
        last = self.last_poll.get(wallet, 0)
        if now - last < self.config.poll_interval_seconds:
            return []

        self.last_poll[wallet] = now

        new_trades = []

        try:
            # Get recent activity (last 60 seconds)
            start_ts = int(now - 60)
            activity = self.data_api.get_activity(
                user=wallet,
                limit=20,
                activity_type="TRADE",
                side="BUY" if self.config.only_buys else None,
                start=start_ts,
            )

            for trade in activity:
                tx_hash = trade.get("transactionHash", "")
                if not tx_hash or tx_hash in self.seen_trades:
                    continue

                self.seen_trades.add(tx_hash)

                # Parse trade
                slug = (trade.get("slug", "") or "").lower()
                title = (trade.get("title", "") or "").lower()
                side = trade.get("side", "")
                size = float(trade.get("size", 0) or 0)
                price = float(trade.get("price", 0) or 0)
                usdc_size = float(trade.get("usdcSize", 0) or 0)
                outcome = trade.get("outcome", "")
                condition_id = trade.get("conditionId", "")
                timestamp = trade.get("timestamp", "")

                # Filter for 15-min crypto
                if self.config.only_crypto_15m:
                    is_crypto_15m = any(p in slug for p in CRYPTO_15M_PATTERNS)
                    if not is_crypto_15m:
                        continue

                # Determine asset
                asset = None
                for a in CRYPTO_ASSETS:
                    if a in slug or a in title:
                        asset = a.upper()
                        break

                if not asset:
                    continue

                # Parse timestamp
                try:
                    if isinstance(timestamp, str):
                        ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    else:
                        ts = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                except:
                    ts = datetime.now(timezone.utc)

                tracked = TrackedTrade(
                    wallet=wallet,
                    timestamp=ts,
                    condition_id=condition_id,
                    asset=asset,
                    side=side,
                    outcome=outcome,
                    size=size,
                    price=price,
                    usdc_size=usdc_size,
                    slug=slug,
                    tx_hash=tx_hash,
                )

                new_trades.append(tracked)

                logger.info(
                    f"📡 COPY SIGNAL [{asset}]: {wallet[:8]}... {side} {outcome} | "
                    f"{size:.2f} shares @ ${price:.3f} = ${usdc_size:.2f}"
                )

        except Exception as e:
            logger.error(f"Failed to poll wallet {wallet[:10]}...: {e}")

        return new_trades

    def poll_all_wallets(self) -> List[TrackedTrade]:
        """Poll all tracked wallets for new trades."""
        all_trades = []
        for wallet in self.config.wallets:
            trades = self.poll_wallet(wallet)
            all_trades.extend(trades)
        return all_trades

    def get_copy_signal(self, trade: TrackedTrade) -> Optional[dict]:
        """
        Generate a copy signal from a tracked trade.

        Returns signal dict if we should copy, None otherwise.
        """
        # Check if trade is too old
        age = (datetime.now(timezone.utc) - trade.timestamp).total_seconds()
        if age > self.config.max_copy_delay_seconds:
            logger.debug(f"Trade too old to copy: {age:.0f}s > {self.config.max_copy_delay_seconds}s")
            return None

        # Check period limits
        current_period = int(time.time()) // 900 * 900
        if self.period_start != current_period:
            self.period_start = current_period
            self.copies_this_period = 0

        if self.copies_this_period >= self.config.max_copies_per_period:
            logger.debug(f"Hit copy limit for this period: {self.copies_this_period}")
            return None

        # Check trader quality
        should_copy, reason = self.should_copy_trader(trade.wallet)
        if not should_copy:
            logger.debug(f"Not copying from {trade.wallet[:8]}...: {reason}")
            return None

        # Calculate copy size
        if self.config.copy_size_mode == "fixed":
            copy_size_usd = self.config.fixed_size_usd
        elif self.config.copy_size_mode == "proportional":
            copy_size_usd = trade.usdc_size * self.config.proportional_multiplier
        else:
            copy_size_usd = self.config.fixed_size_usd

        # Determine our side based on their outcome
        # If they bought "Yes" on UP market, we go UP
        # If they bought "No" on UP market, we go DOWN
        our_side = "UP" if trade.outcome.lower() == "yes" else "DOWN"

        signal = {
            "asset": trade.asset,
            "side": our_side,
            "size_usd": copy_size_usd,
            "price": trade.price,
            "condition_id": trade.condition_id,
            "slug": trade.slug,
            "source_wallet": trade.wallet,
            "source_size": trade.usdc_size,
            "reason": f"Copying {trade.wallet[:8]}... ({reason})",
        }

        self.copies_this_period += 1
        trade.copied = True
        trade.copy_timestamp = datetime.now(timezone.utc)

        logger.info(
            f"📋 COPY TRADE [{trade.asset}]: {our_side} ${copy_size_usd:.2f} | "
            f"From: {trade.wallet[:8]}... | Source: ${trade.usdc_size:.2f}"
        )

        return signal

    def get_tracked_wallets_summary(self) -> str:
        """Get summary of tracked wallets and their stats."""
        lines = ["Copy Trading Summary:"]

        if not self.config.wallets:
            lines.append("  No wallets being tracked")
            return "\n".join(lines)

        for wallet in self.config.wallets:
            stats = self.trader_stats.get(wallet)
            if stats:
                lines.append(
                    f"  {wallet[:10]}...: {stats.crypto_win_rate:.0%} win rate, "
                    f"{stats.crypto_15m_trades} crypto trades, ${stats.crypto_15m_pnl:+.2f}"
                )
            else:
                lines.append(f"  {wallet[:10]}...: (not analyzed yet)")

        return "\n".join(lines)


# Global instance
_copy_manager: Optional[CopyTradingManager] = None


def get_copy_trading_manager(config: Optional[CopyTradingConfig] = None) -> CopyTradingManager:
    """Get or create copy trading manager singleton."""
    global _copy_manager
    if _copy_manager is None:
        if config is None:
            config = CopyTradingConfig()
        _copy_manager = CopyTradingManager(config)
    return _copy_manager


def analyze_wallet(wallet: str) -> TraderStats:
    """Convenience function to analyze a wallet."""
    manager = get_copy_trading_manager()
    return manager.analyze_trader(wallet)


def add_wallet_to_copy(wallet: str):
    """Convenience function to add a wallet to copy."""
    manager = get_copy_trading_manager()
    manager.add_wallet(wallet)


def remove_wallet_from_copy(wallet: str):
    """Convenience function to remove a wallet from copy."""
    manager = get_copy_trading_manager()
    manager.remove_wallet(wallet)
