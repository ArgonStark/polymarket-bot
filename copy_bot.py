#!/usr/bin/env python3
"""
Standalone Copy Trading Bot

A fast, independent copy trading system that:
- Monitors target wallets for trades
- Auto-sizes based on your balance vs their trade size
- Executes trades immediately with market orders

Usage:
    python copy_bot.py --wallet 0x1234... --wallet 0x5678...
    python copy_bot.py --wallet 0x1234... --size-pct 0.05

Environment:
    PK: Your private key
    FUNDER: Your deposit address
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Dict, Set

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import BotConfig
from src.execution import create_trading_client
from src.execution.client import get_account_balance
from src.data.polymarket_data import get_data_api
from src.data.gamma_api import GammaAPI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Constants
CRYPTO_15M_PATTERNS = ["updown-15m", "15m", "15-min"]
CRYPTO_ASSETS = {"btc": "BTC", "eth": "ETH", "sol": "SOL", "xrp": "XRP"}


@dataclass
class CopyConfig:
    """Copy trading configuration."""
    wallets: List[str]  # Wallets to copy

    # Sizing mode: "fixed", "percent", "proportional", "kelly"
    size_mode: str = "percent"

    # For "fixed" mode - always trade this amount
    fixed_size_usd: float = 5.0

    # For "percent" mode - use this % of YOUR balance
    size_pct: float = 0.05  # 5% of balance

    # For "proportional" mode - match their size scaled to your balance
    # If they trade 1% of their portfolio, you trade 1% of yours
    their_estimated_balance: float = 100000.0  # Estimate their bankroll

    # Limits (apply to all modes)
    min_size_usd: float = 5.0  # Minimum $5 (Polymarket minimum)
    max_size_usd: float = 50.0  # Maximum $50 per copy
    max_risk_pct: float = 0.10  # Never risk more than 10% of balance per trade

    # Timing
    poll_interval: float = 0.5  # Poll every 500ms
    max_delay_seconds: int = 15  # Copy within 15 seconds

    # Safety
    dry_run: bool = False
    stop_loss_pct: float = 0.50  # Stop if balance drops 50% from start


class CopyBot:
    """
    Standalone copy trading bot.

    Fast and focused - only does copy trading.
    """

    def __init__(self, config: CopyConfig):
        self.config = config
        self.data_api = get_data_api()
        self.gamma_api = GammaAPI()

        # Trading client
        self.client = None
        self.balance = 0.0

        # Track seen trades to avoid duplicates
        self.seen_trades: Set[str] = set()

        # Active markets cache
        self.markets: Dict[str, dict] = {}  # condition_id -> market info
        self.last_market_refresh = 0

        # Stats
        self.copies_executed = 0
        self.copies_failed = 0
        self.total_spent = 0.0

        self._running = False

    async def start(self):
        """Start the copy bot."""
        logger.info("=" * 60)
        logger.info("  COPY TRADING BOT - STANDALONE")
        logger.info("=" * 60)
        logger.info(f"Tracking {len(self.config.wallets)} wallets:")
        for w in self.config.wallets:
            logger.info(f"  • {w[:10]}...{w[-6:]}")
        # Show sizing configuration based on mode
        if self.config.size_mode == "fixed":
            logger.info(f"Sizing: FIXED ${self.config.fixed_size_usd:.2f} per trade")
        elif self.config.size_mode == "proportional":
            logger.info(f"Sizing: PROPORTIONAL (their est. balance: ${self.config.their_estimated_balance:,.0f})")
        elif self.config.size_mode == "kelly":
            logger.info(f"Sizing: KELLY (1/4 fractional)")
        else:
            logger.info(f"Sizing: {self.config.size_pct:.0%} of balance")
        logger.info(f"Limits: ${self.config.min_size_usd}-${self.config.max_size_usd}, max {self.config.max_risk_pct:.0%} risk per trade")
        logger.info(f"Poll interval: {self.config.poll_interval}s")
        logger.info("=" * 60)

        # Initialize trading client
        if not self.config.dry_run:
            bot_config = BotConfig.from_env()
            self.client = create_trading_client(bot_config)
            if not self.client:
                logger.error("Failed to create trading client - check PK and FUNDER")
                return

            # Get initial balance
            self.balance = get_account_balance(self.client)
            logger.info(f"💰 Starting balance: ${self.balance:.2f}")
        else:
            logger.info("🔸 DRY RUN MODE - no real trades")
            self.balance = 100.0  # Fake balance for testing

        self._running = True

        try:
            await asyncio.gather(
                self._run_copy_loop(),
                self._run_market_refresh_loop(),
                self._run_balance_sync_loop(),
            )
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            self._print_stats()

    async def _run_copy_loop(self):
        """Main copy trading loop - runs as fast as possible."""
        logger.info("🚀 Copy loop started")

        while self._running:
            try:
                start = time.time()

                # Poll all wallets
                for wallet in self.config.wallets:
                    trades = await self._poll_wallet(wallet)

                    for trade in trades:
                        # Execute copy immediately
                        await self._execute_copy(trade)

                # Calculate sleep time to maintain poll interval
                elapsed = time.time() - start
                sleep_time = max(0.1, self.config.poll_interval - elapsed)
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Copy loop error: {e}")
                await asyncio.sleep(1.0)

        logger.info("Copy loop stopped")

    async def _poll_wallet(self, wallet: str) -> List[dict]:
        """Poll a wallet for new trades."""
        wallet = wallet.lower()
        new_trades = []

        try:
            # Get last 10 seconds of activity
            now = int(time.time())
            activity = self.data_api.get_activity(
                user=wallet,
                limit=10,
                activity_type="TRADE",
                side="BUY",
                start=now - 10,
            )

            for trade in activity:
                tx_hash = trade.get("transactionHash", "")
                if not tx_hash or tx_hash in self.seen_trades:
                    continue

                self.seen_trades.add(tx_hash)

                # Check if 15-min crypto market
                slug = (trade.get("slug", "") or "").lower()
                if not any(p in slug for p in CRYPTO_15M_PATTERNS):
                    continue

                # Identify asset
                asset = None
                for pattern, name in CRYPTO_ASSETS.items():
                    if pattern in slug:
                        asset = name
                        break

                if not asset:
                    continue

                # Build trade info
                trade_info = {
                    "wallet": wallet,
                    "asset": asset,
                    "condition_id": trade.get("conditionId", ""),
                    "outcome": trade.get("outcome", ""),
                    "side": trade.get("side", ""),
                    "size": float(trade.get("size", 0) or 0),
                    "price": float(trade.get("price", 0) or 0),
                    "usdc_size": float(trade.get("usdcSize", 0) or 0),
                    "slug": slug,
                    "tx_hash": tx_hash,
                    "timestamp": time.time(),
                }

                new_trades.append(trade_info)

                logger.info(
                    f"⚡ DETECTED [{asset}]: {wallet[:8]}... bought {trade_info['outcome']} | "
                    f"{trade_info['size']:.1f} shares @ ${trade_info['price']:.3f} = ${trade_info['usdc_size']:.2f}"
                )

        except Exception as e:
            logger.debug(f"Poll error for {wallet[:8]}...: {e}")

        return new_trades

    async def _execute_copy(self, trade: dict):
        """Execute a copy trade."""
        asset = trade["asset"]
        condition_id = trade["condition_id"]
        outcome = trade["outcome"]
        their_size = trade["usdc_size"]

        start_time = time.time()

        try:
            # Calculate our size based on balance
            our_size = self._calculate_copy_size(their_size)

            if our_size < self.config.min_size_usd:
                logger.warning(f"Skip [{asset}]: size ${our_size:.2f} below minimum ${self.config.min_size_usd}")
                return

            # Find market info
            market = self.markets.get(condition_id)
            if not market:
                # Try to find by asset
                for cid, m in self.markets.items():
                    if m.get("asset") == asset:
                        market = m
                        condition_id = cid
                        break

            if not market:
                logger.warning(f"Skip [{asset}]: market not found")
                return

            # Determine token and price
            # outcome is "Yes" or "No" - we copy their exact position
            if outcome.lower() == "yes":
                token_id = market.get("up_token_id")
                price = market.get("best_ask", 0.5)
                our_side = "UP"
            else:
                token_id = market.get("down_token_id")
                price = market.get("best_ask", 0.5)
                our_side = "DOWN"

            if not token_id or price <= 0:
                logger.warning(f"Skip [{asset}]: invalid token/price")
                return

            # Calculate shares
            shares = our_size / price
            if shares < 5:  # Polymarket minimum
                shares = 5
                our_size = shares * price

            # Execute trade
            if self.client and not self.config.dry_run:
                from py_clob_client.clob_types import OrderArgs
                from py_clob_client.order_builder.constants import BUY

                order_args = OrderArgs(
                    token_id=token_id,
                    price=price,
                    size=shares,
                    side=BUY,
                )

                # Use create_and_post_order for market order
                try:
                    signed_order = self.client.create_order(order_args)
                    result = self.client.post_order(signed_order, orderType="FOK")  # Fill or Kill

                    exec_time = int((time.time() - start_time) * 1000)

                    if result and result.get("success"):
                        self.copies_executed += 1
                        self.total_spent += our_size

                        logger.info(
                            f"✅ COPIED [{asset}]: {our_side} {shares:.1f} shares @ ${price:.3f} | "
                            f"Cost: ${our_size:.2f} | Time: {exec_time}ms | "
                            f"From: {trade['wallet'][:8]}... (${their_size:.2f})"
                        )
                    else:
                        self.copies_failed += 1
                        error = result.get("errorMsg", "Unknown error") if result else "No response"
                        logger.warning(f"❌ COPY FAILED [{asset}]: {error} | Time: {exec_time}ms")

                except Exception as e:
                    self.copies_failed += 1
                    exec_time = int((time.time() - start_time) * 1000)
                    logger.error(f"❌ COPY ERROR [{asset}]: {e} | Time: {exec_time}ms")

            else:
                # Dry run
                exec_time = int((time.time() - start_time) * 1000)
                self.copies_executed += 1
                self.total_spent += our_size

                logger.info(
                    f"🔸 DRY COPY [{asset}]: {our_side} {shares:.1f} shares @ ${price:.3f} | "
                    f"Cost: ${our_size:.2f} | Time: {exec_time}ms | "
                    f"From: {trade['wallet'][:8]}... (${their_size:.2f})"
                )

        except Exception as e:
            self.copies_failed += 1
            logger.error(f"Copy execution error: {e}")

    def _calculate_copy_size(self, their_size: float) -> float:
        """
        Calculate our copy size based on configured sizing mode.

        Modes:
        - fixed: Always trade fixed_size_usd
        - percent: Use size_pct of YOUR balance
        - proportional: Match their trade size scaled to your balance
        - kelly: Use Kelly criterion based on their win rate (advanced)
        """
        if self.config.size_mode == "fixed":
            # Fixed dollar amount per trade
            our_size = self.config.fixed_size_usd

        elif self.config.size_mode == "proportional":
            # Match their sizing proportionally
            # If they trade $100 of estimated $10,000 portfolio (1%),
            # we trade 1% of our portfolio
            their_pct = their_size / self.config.their_estimated_balance
            our_size = self.balance * their_pct

        elif self.config.size_mode == "kelly":
            # Simplified Kelly: f = (bp - q) / b
            # For 50% win rate at even odds: f = 0 (don't bet)
            # For 55% win rate: f = 0.10 (10% of bankroll)
            # We use a fractional Kelly (1/4) for safety
            assumed_win_rate = 0.55  # Target trader's estimated win rate
            assumed_odds = 1.0  # Even money approximation
            kelly_fraction = (assumed_win_rate * assumed_odds - (1 - assumed_win_rate)) / assumed_odds
            kelly_fraction = max(0, kelly_fraction) * 0.25  # 1/4 Kelly for safety
            our_size = self.balance * kelly_fraction

        else:  # "percent" mode (default)
            # Simple percentage of our balance
            our_size = self.balance * self.config.size_pct

        # Apply limits (all modes)
        our_size = max(self.config.min_size_usd, our_size)
        our_size = min(self.config.max_size_usd, our_size)

        # Never risk more than max_risk_pct of balance per trade
        max_risk = self.balance * self.config.max_risk_pct
        our_size = min(our_size, max_risk)

        # Keep a small reserve (don't spend last 10%)
        our_size = min(our_size, self.balance * 0.9)

        return our_size

    async def _run_market_refresh_loop(self):
        """Refresh active markets periodically."""
        while self._running:
            try:
                await self._refresh_markets()
                await asyncio.sleep(30)  # Refresh every 30s
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Market refresh error: {e}")
                await asyncio.sleep(5)

    async def _refresh_markets(self):
        """Fetch active 15-min crypto markets."""
        try:
            # Get markets from Gamma API
            markets_data = self.gamma_api.get_active_markets()

            new_markets = {}
            for market in markets_data:
                asset = market.get("asset")
                if asset not in ["BTC", "ETH", "SOL", "XRP"]:
                    continue

                condition_id = market.get("condition_id", "")
                if not condition_id:
                    continue

                new_markets[condition_id] = {
                    "asset": asset,
                    "condition_id": condition_id,
                    "up_token_id": market.get("up_token_id"),
                    "down_token_id": market.get("down_token_id"),
                    "target_price": market.get("target_price"),
                    "best_ask": market.get("best_ask", 0.5),
                    "best_bid": market.get("best_bid", 0.5),
                    "end_time": market.get("end_time"),
                }

            if new_markets:
                self.markets = new_markets
                logger.debug(f"Refreshed {len(self.markets)} active markets")

        except Exception as e:
            logger.error(f"Failed to refresh markets: {e}")

    async def _run_balance_sync_loop(self):
        """Sync balance periodically."""
        while self._running:
            try:
                if self.client and not self.config.dry_run:
                    self.balance = get_account_balance(self.client)
                    logger.debug(f"Balance synced: ${self.balance:.2f}")
                await asyncio.sleep(60)  # Sync every 60s
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Balance sync error: {e}")
                await asyncio.sleep(10)

    def _print_stats(self):
        """Print final statistics."""
        logger.info("=" * 60)
        logger.info("  COPY BOT STATS")
        logger.info("=" * 60)
        logger.info(f"Copies executed: {self.copies_executed}")
        logger.info(f"Copies failed: {self.copies_failed}")
        logger.info(f"Total spent: ${self.total_spent:.2f}")
        logger.info(f"Final balance: ${self.balance:.2f}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Copy Trading Bot")

    parser.add_argument(
        "--wallet", "-w",
        action="append",
        required=True,
        help="Wallet address to copy (can specify multiple)",
    )

    parser.add_argument(
        "--size-mode",
        type=str,
        choices=["fixed", "percent", "proportional", "kelly"],
        default="percent",
        help="Sizing mode: fixed, percent, proportional, kelly (default: percent)",
    )

    parser.add_argument(
        "--size-pct",
        type=float,
        default=0.05,
        help="Percentage of balance per trade for 'percent' mode (default: 0.05 = 5%%)",
    )

    parser.add_argument(
        "--fixed-size",
        type=float,
        default=10.0,
        help="Fixed trade size in USD for 'fixed' mode (default: 10)",
    )

    parser.add_argument(
        "--their-balance",
        type=float,
        default=100000.0,
        help="Estimated target's balance for 'proportional' mode (default: 100000)",
    )

    parser.add_argument(
        "--min-size",
        type=float,
        default=5.0,
        help="Minimum trade size in USD (default: 5)",
    )

    parser.add_argument(
        "--max-size",
        type=float,
        default=50.0,
        help="Maximum trade size in USD (default: 50)",
    )

    parser.add_argument(
        "--max-risk-pct",
        type=float,
        default=0.10,
        help="Max risk per trade as %% of balance (default: 0.10 = 10%%)",
    )

    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Poll interval in seconds (default: 0.5)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without executing real trades",
    )

    args = parser.parse_args()

    config = CopyConfig(
        wallets=args.wallet,
        size_mode=args.size_mode,
        fixed_size_usd=args.fixed_size,
        size_pct=args.size_pct,
        their_estimated_balance=args.their_balance,
        min_size_usd=args.min_size,
        max_size_usd=args.max_size,
        max_risk_pct=args.max_risk_pct,
        poll_interval=args.poll_interval,
        dry_run=args.dry_run,
    )

    bot = CopyBot(config)

    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    main()
