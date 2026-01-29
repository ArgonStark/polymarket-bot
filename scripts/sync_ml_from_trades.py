#!/usr/bin/env python3
"""
Manual ML Sync from Polymarket Trade History

This script fetches your actual trades from Polymarket and syncs them
with the ML model. Useful for:
- Backfilling the ML model with historical trades
- Debugging ML learning issues
- Manual sync if automatic sync fails

Usage:
    python scripts/sync_ml_from_trades.py [--limit 100]
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import BotConfig
from src.execution import create_trading_client
from src.execution.client import get_trades
from src.data.gamma import GammaAPI
from src.strategy.ml_predictor import get_ml_predictor, MLSignalPredictor
from src.models import Side

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)


def load_config() -> BotConfig:
    """Load bot configuration."""
    return BotConfig()


def sync_trades_to_ml(limit: int = 100, dry_run: bool = False):
    """
    Sync trades from Polymarket to ML model.

    Args:
        limit: Maximum number of trades to fetch
        dry_run: If True, don't actually save to ML model
    """
    print("\n" + "=" * 60)
    print("  POLYMARKET -> ML SYNC")
    print("=" * 60 + "\n")

    # Load config and create clients
    config = load_config()
    client = create_trading_client(config)

    if not client:
        print("ERROR: Failed to create trading client")
        print("Make sure your POLYMARKET_API_KEY and POLYMARKET_PRIVATE_KEY are set")
        return

    # Initialize ML predictor
    ml_predictor = get_ml_predictor()
    print(f"ML Model loaded: {ml_predictor.training_samples} existing samples")

    # Initialize Gamma API for market resolution checks
    gamma_api = GammaAPI(config=config)

    # Fetch trades from Polymarket
    print(f"\nFetching last {limit} trades from Polymarket API...")
    trades = get_trades(client, limit=limit)

    if not trades:
        print("No trades found")
        return

    print(f"Found {len(trades)} trades\n")

    # Track synced trades
    synced_count = 0
    skipped_count = 0
    pending_count = 0
    non_crypto_count = 0
    current_ts = int(time.time())

    # Process each trade
    for i, trade in enumerate(trades):
        try:
            trade_id = trade.get("id") or trade.get("trade_id") or str(trade.get("matchTime", ""))
            market_slug = trade.get("marketSlug", "") or trade.get("market_slug", "")

            # Check if this is a 15-min crypto market
            if "updown-15m" not in market_slug.lower():
                non_crypto_count += 1
                continue

            # Extract market timestamp from slug
            market_ts = None
            try:
                parts = market_slug.split("-")
                if len(parts) >= 4:
                    market_ts = int(parts[-1])
            except (ValueError, IndexError):
                continue

            if not market_ts:
                continue

            # Check if market has settled
            settle_time = market_ts + 900
            if current_ts < settle_time + 30:
                pending_count += 1
                continue

            # Identify asset
            asset = None
            for a in ["btc", "eth", "sol", "xrp"]:
                if market_slug.lower().startswith(a):
                    asset = a.upper()
                    break

            if not asset:
                continue

            # Determine trade side
            outcome = trade.get("outcome", "") or trade.get("side", "")
            if isinstance(outcome, str):
                outcome_lower = outcome.lower()
                if "up" in outcome_lower or "yes" in outcome_lower:
                    side = Side.UP
                elif "down" in outcome_lower or "no" in outcome_lower:
                    side = Side.DOWN
                else:
                    continue
            else:
                continue

            # Get market resolution
            condition_id = trade.get("conditionId", "") or trade.get("condition_id", "")
            resolution = None
            if condition_id:
                resolution = gamma_api.get_market_resolution(condition_id)

            winning_outcome = None
            if resolution and resolution.get("resolved"):
                winning_outcome = resolution.get("winning_outcome")
            else:
                # Try to determine from price
                target_price = gamma_api.fetch_price_to_beat(asset, market_ts)
                if target_price:
                    # We need to know the settlement price
                    # For now, assume API will eventually have it
                    pass

            if not winning_outcome:
                pending_count += 1
                continue

            # Determine if trade won
            won = side.value == winning_outcome

            # Get trade details
            trade_price = float(trade.get("price", 0) or 0)
            trade_size = float(trade.get("size", 0) or 0)

            if trade_price <= 0 or trade_size <= 0:
                continue

            # Display trade info
            result = "WIN" if won else "LOSS"
            result_color = "\033[92m" if won else "\033[91m"  # Green/Red
            reset = "\033[0m"

            print(
                f"[{i+1:3d}] {asset:4s} {side.value:4s} @ {trade_price:.4f} | "
                f"Size: {trade_size:8.2f} | {result_color}{result:4s}{reset} | "
                f"Slug: {market_slug}"
            )

            if not dry_run:
                # Build minimal signal for ML
                from types import SimpleNamespace

                fake_signal = SimpleNamespace(
                    edge=0.0,
                    market=SimpleNamespace(
                        asset=asset,
                        target_price=0,
                        time_remaining=0,
                        best_bid=trade_price,
                        best_ask=trade_price,
                        bid_depth=0,
                        ask_depth=0,
                    ),
                    side=side,
                    _arb_type="none",
                    recommended_price=trade_price,
                    size_shares=trade_size,
                    size_usd=trade_size * trade_price,
                    time_remaining=0,
                )

                # Record outcome
                ml_predictor.record_outcome(
                    signal=fake_signal,
                    volatility=0.003,  # Default volatility
                    price_momentum=0.0,
                    won=won,
                )

            synced_count += 1

        except Exception as e:
            logger.debug(f"Error processing trade: {e}")
            continue

    # Summary
    print("\n" + "-" * 60)
    print("SYNC SUMMARY")
    print("-" * 60)
    print(f"  Total trades fetched: {len(trades)}")
    print(f"  Synced to ML:         {synced_count}")
    print(f"  Pending settlement:   {pending_count}")
    print(f"  Non-15min crypto:     {non_crypto_count}")
    print(f"  Skipped/errors:       {len(trades) - synced_count - pending_count - non_crypto_count}")

    if not dry_run:
        print(f"\n  ML Model now has: {ml_predictor.training_samples} samples")
        print(f"  Model saved to: {ml_predictor.model_path}")

    # Check if model file was created
    if os.path.exists(ml_predictor.model_path):
        size = os.path.getsize(ml_predictor.model_path)
        print(f"  Model file size: {size:,} bytes")
    else:
        print("  WARNING: Model file was not created!")

    print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Sync Polymarket trades to ML model"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=100,
        help="Maximum number of trades to fetch (default: 100)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Don't actually save to ML model"
    )

    args = parser.parse_args()
    sync_trades_to_ml(limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
