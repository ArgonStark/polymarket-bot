#!/usr/bin/env python3
"""
Standalone Copy Trading Bot for Polymarket

Monitors a target wallet and copies their 15-min crypto trades
with proportional position sizing.

Usage:
    python copy_trader.py --target 0x1234... [--check-interval 10]

Environment:
    PK - Your private key
    CLOB_API_KEY, CLOB_SECRET, CLOB_PASS_PHRASE - API credentials
"""

import os
import sys
import time
import asyncio
import argparse
import logging
from datetime import datetime, timezone
from typing import Optional, Set

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import BotConfig
from src.execution import create_trading_client
from src.execution.client import get_account_balance
from src.data.polymarket_data import get_data_api
from src.data.gamma import GammaAPI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class CopyTrader:
    """
    Copy trading bot that mirrors trades from a target wallet.

    Features:
    - Monitors target wallet activity in real-time
    - Copies 15-min crypto trades only (BTC/ETH/SOL/XRP)
    - Proportional sizing based on balance percentage
    - Tracks copied trades to avoid duplicates
    """

    def __init__(
        self,
        target_wallet: str,
        check_interval: float = 10.0,
        dry_run: bool = False,
    ):
        """
        Initialize copy trader.

        Args:
            target_wallet: Wallet address to copy trades from
            check_interval: Seconds between activity checks
            dry_run: If True, simulate trades without executing
        """
        self.target_wallet = target_wallet.lower()
        self.check_interval = check_interval
        self.dry_run = dry_run

        # Initialize components
        self.config = BotConfig()
        self.client = None
        self.data_api = get_data_api()
        self.gamma = GammaAPI()

        # Track state
        self.copied_trades: Set[str] = set()  # Transaction hashes we've copied
        self.last_check_time: Optional[datetime] = None
        self.my_wallet: Optional[str] = None
        self.running = False

        # Stats
        self.trades_copied = 0
        self.total_volume = 0.0

    def _is_15min_crypto(self, slug: str, title: str) -> bool:
        """Check if trade is for a 15-min crypto market."""
        slug_lower = slug.lower()
        title_lower = title.lower()
        return "updown-15m" in slug_lower or "15m" in title_lower

    def _get_asset(self, slug: str) -> Optional[str]:
        """Extract asset from market slug."""
        slug_lower = slug.lower()
        for asset in ["btc", "eth", "sol", "xrp"]:
            if slug_lower.startswith(asset) or f"-{asset}-" in slug_lower:
                return asset.upper()
        return None

    async def _get_target_balance(self) -> float:
        """Estimate target wallet's trading balance from positions."""
        try:
            positions = self.data_api.get_positions(self.target_wallet, limit=100)
            total_value = sum(
                float(p.get("currentValue", 0) or 0)
                for p in positions
            )
            # Rough estimate: positions are ~50% of trading capital
            return max(total_value * 2, 100.0)
        except Exception as e:
            logger.warning(f"Could not estimate target balance: {e}")
            return 1000.0  # Default assumption

    async def _copy_trade(self, activity: dict) -> bool:
        """
        Copy a single trade from target wallet.

        Args:
            activity: Trade activity object from Data API

        Returns:
            True if trade was copied successfully
        """
        try:
            # Extract trade details
            condition_id = activity.get("conditionId", "")
            side = activity.get("side", "")  # BUY or SELL
            outcome = activity.get("outcome", "")  # Up or Down
            price = float(activity.get("price", 0) or 0)
            size = float(activity.get("size", 0) or 0)
            usdc_size = float(activity.get("usdcSize", 0) or 0)
            slug = activity.get("slug", "") or activity.get("eventSlug", "")
            title = activity.get("title", "")
            tx_hash = activity.get("transactionHash", "")

            # Get asset
            asset = self._get_asset(slug)
            if not asset:
                return False

            # Calculate proportional size
            my_balance = get_account_balance(self.client)
            target_balance = await self._get_target_balance()

            if target_balance <= 0:
                target_balance = 1000.0

            # Same percentage of balance
            size_ratio = usdc_size / target_balance
            my_size_usdc = my_balance * size_ratio

            # Apply min/max limits
            min_size = 1.0
            max_size = my_balance * 0.25  # Max 25% per trade
            my_size_usdc = max(min_size, min(my_size_usdc, max_size))

            # Calculate shares
            if price > 0:
                my_shares = my_size_usdc / price
            else:
                my_shares = my_size_usdc

            # Log the copy
            side_color = "\033[92m" if side == "BUY" else "\033[91m"
            reset = "\033[0m"

            logger.info(
                f"{side_color}COPY{reset} | {asset} {outcome} | "
                f"Target: ${usdc_size:.2f} -> You: ${my_size_usdc:.2f} | "
                f"Price: {price:.4f}"
            )

            if self.dry_run:
                logger.info("  [DRY RUN - Trade not executed]")
                return True

            # Find the market to get token_id
            market = self.gamma.get_market_by_slug(slug)
            if not market:
                logger.warning(f"  Could not find market: {slug}")
                return False

            # Get token ID based on outcome
            tokens = market.get("tokens", [])
            token_id = None
            for token in tokens:
                if token.get("outcome", "").lower() == outcome.lower():
                    token_id = token.get("token_id")
                    break

            if not token_id:
                logger.warning(f"  Could not find token for outcome: {outcome}")
                return False

            # Execute the trade
            from py_clob_client.order_builder.constants import BUY, SELL

            order_side = BUY if side == "BUY" else SELL

            order = self.client.create_and_post_order(
                token_id=token_id,
                price=price,
                size=my_shares,
                side=order_side,
            )

            if order:
                logger.info(f"  Order placed: {order.get('orderID', 'unknown')}")
                self.trades_copied += 1
                self.total_volume += my_size_usdc
                return True
            else:
                logger.warning("  Order failed")
                return False

        except Exception as e:
            logger.error(f"Error copying trade: {e}")
            return False

    async def _check_and_copy(self):
        """Check target wallet for new trades and copy them."""
        try:
            # Fetch recent activity
            activity = self.data_api.get_activity(
                self.target_wallet,
                activity_type="TRADE",
                limit=20,
            )

            if not activity:
                return

            for trade in activity:
                tx_hash = trade.get("transactionHash", "")

                # Skip if already copied
                if tx_hash in self.copied_trades:
                    continue

                # Check if 15-min crypto
                slug = trade.get("slug", "") or trade.get("eventSlug", "")
                title = trade.get("title", "")

                if not self._is_15min_crypto(slug, title):
                    self.copied_trades.add(tx_hash)  # Mark as seen
                    continue

                # Copy the trade
                success = await self._copy_trade(trade)
                self.copied_trades.add(tx_hash)

                if success:
                    logger.info(f"  Trade copied successfully")

        except Exception as e:
            logger.error(f"Error checking activity: {e}")

    async def start(self):
        """Start the copy trading bot."""
        print("\n" + "=" * 60)
        print("  POLYMARKET COPY TRADER")
        print("=" * 60)

        # Initialize client
        self.client = create_trading_client(self.config)
        if not self.client:
            print("\nERROR: Failed to create trading client")
            print("Make sure your environment variables are set")
            return

        self.my_wallet = self.client.get_address()
        my_balance = get_account_balance(self.client)

        print(f"\n  Your wallet:    {self.my_wallet}")
        print(f"  Your balance:   ${my_balance:.2f}")
        print(f"  Target wallet:  {self.target_wallet}")
        print(f"  Check interval: {self.check_interval}s")
        print(f"  Dry run:        {self.dry_run}")
        print(f"\n  Copying: 15-min crypto trades only")
        print(f"  Sizing:  Same percentage of balance")
        print("\n" + "=" * 60)

        # Load existing activity to avoid copying old trades
        print("\nLoading target's recent activity...")
        existing = self.data_api.get_activity(
            self.target_wallet,
            activity_type="TRADE",
            limit=100,
        )
        for trade in existing:
            self.copied_trades.add(trade.get("transactionHash", ""))
        print(f"Marked {len(self.copied_trades)} existing trades as seen")

        print("\nMonitoring for new trades... (Ctrl+C to stop)\n")

        self.running = True
        try:
            while self.running:
                await self._check_and_copy()
                await asyncio.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("\n\nStopping copy trader...")

        # Print summary
        print("\n" + "=" * 60)
        print("  SESSION SUMMARY")
        print("=" * 60)
        print(f"  Trades copied:  {self.trades_copied}")
        print(f"  Total volume:   ${self.total_volume:.2f}")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Copy trades from a target Polymarket wallet"
    )
    parser.add_argument(
        "--target", "-t", required=True,
        help="Target wallet address to copy from (0x...)"
    )
    parser.add_argument(
        "--interval", "-i", type=float, default=10.0,
        help="Seconds between activity checks (default: 10)"
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Simulate trades without executing"
    )

    args = parser.parse_args()

    # Validate target address
    if not args.target.startswith("0x") or len(args.target) != 42:
        print("ERROR: Invalid wallet address. Must be 0x followed by 40 hex characters")
        sys.exit(1)

    trader = CopyTrader(
        target_wallet=args.target,
        check_interval=args.interval,
        dry_run=args.dry_run,
    )

    asyncio.run(trader.start())


if __name__ == "__main__":
    main()
