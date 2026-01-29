#!/usr/bin/env python3
"""
Manual ML Sync from Polymarket Trade History

This script fetches your closed positions from Polymarket Data API
and syncs them with the ML model.

Usage:
    python scripts/sync_ml_from_trades.py [--limit 100]
"""

import os
import sys
import argparse
import logging
import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import BotConfig
from src.execution import create_trading_client
from src.strategy.ml_predictor import get_ml_predictor
from src.models import Side

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def get_closed_positions(wallet_address: str, limit: int = 100) -> list:
    """
    Fetch closed positions from Polymarket Data API.

    Args:
        wallet_address: User's wallet address
        limit: Maximum positions to fetch

    Returns:
        List of closed position objects
    """
    url = "https://data-api.polymarket.com/closed-positions"
    params = {
        "user": wallet_address,
        "limit": limit,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch closed positions: {e}")
        return []


def sync_trades_to_ml(limit: int = 100, dry_run: bool = False):
    """
    Sync closed positions from Polymarket to ML model.

    Args:
        limit: Maximum number of positions to fetch
        dry_run: If True, don't actually save to ML model
    """
    print("\n" + "=" * 60)
    print("  POLYMARKET -> ML SYNC (Closed Positions)")
    print("=" * 60 + "\n")

    # Load config and create client to get wallet address
    config = BotConfig()
    client = create_trading_client(config)

    if not client:
        print("ERROR: Failed to create trading client")
        print("Make sure your POLYMARKET_API_KEY and POLYMARKET_PRIVATE_KEY are set")
        return

    # Get wallet address
    wallet_address = client.get_address()
    if not wallet_address:
        print("ERROR: Could not get wallet address")
        return

    print(f"Wallet: {wallet_address}")

    # Initialize ML predictor
    ml_predictor = get_ml_predictor()
    print(f"ML Model loaded: {ml_predictor.training_samples} existing samples\n")

    # Fetch closed positions
    print(f"Fetching closed positions from Polymarket Data API...")
    positions = get_closed_positions(wallet_address, limit=limit)

    if not positions:
        print("No closed positions found")
        return

    print(f"Found {len(positions)} closed positions\n")

    # Debug: show first few
    print("Sample position structure:")
    print("-" * 60)
    if positions:
        sample = positions[0]
        for key in ["title", "slug", "outcome", "avgPrice", "totalBought", "realizedPnl"]:
            if key in sample:
                print(f"  {key}: {sample[key]}")
    print("-" * 60 + "\n")

    # Track stats
    synced_count = 0
    non_crypto_count = 0
    skipped_count = 0

    print("Processing positions...\n")

    for i, pos in enumerate(positions):
        try:
            slug = pos.get("slug", "") or pos.get("eventSlug", "")
            title = pos.get("title", "")

            # Check if this is a 15-min crypto market
            if "updown-15m" not in slug.lower() and "15m" not in title.lower():
                non_crypto_count += 1
                continue

            # Identify asset from slug
            asset = None
            for a in ["btc", "eth", "sol", "xrp"]:
                if slug.lower().startswith(a) or a in slug.lower():
                    asset = a.upper()
                    break

            if not asset:
                skipped_count += 1
                continue

            # Get outcome (Up/Down)
            outcome = pos.get("outcome", "")
            if isinstance(outcome, str):
                outcome_lower = outcome.lower()
                if "up" in outcome_lower:
                    side = Side.UP
                elif "down" in outcome_lower:
                    side = Side.DOWN
                else:
                    skipped_count += 1
                    continue
            else:
                skipped_count += 1
                continue

            # Determine if we won based on realizedPnl
            realized_pnl = float(pos.get("realizedPnl", 0) or 0)
            won = realized_pnl > 0

            # Get trade details
            avg_price = float(pos.get("avgPrice", 0) or 0)
            total_bought = float(pos.get("totalBought", 0) or 0)

            if avg_price <= 0 or total_bought <= 0:
                skipped_count += 1
                continue

            # Display position info
            result = "WIN" if won else "LOSS"
            result_color = "\033[92m" if won else "\033[91m"
            reset = "\033[0m"
            pnl_str = f"${realized_pnl:+.2f}"

            print(
                f"[{i+1:3d}] {asset:4s} {side.value:4s} @ {avg_price:.4f} | "
                f"Size: {total_bought:8.2f} | {result_color}{result:4s}{reset} | "
                f"P&L: {pnl_str} | {slug[:40]}"
            )

            if not dry_run:
                from types import SimpleNamespace

                fake_signal = SimpleNamespace(
                    edge=0.0,
                    market=SimpleNamespace(
                        asset=asset,
                        target_price=0,
                        time_remaining=0,
                        best_bid=avg_price,
                        best_ask=avg_price,
                        bid_depth=0,
                        ask_depth=0,
                    ),
                    side=side,
                    _arb_type="none",
                    recommended_price=avg_price,
                    size_shares=total_bought,
                    size_usd=total_bought * avg_price,
                    time_remaining=0,
                )

                ml_predictor.record_outcome(
                    signal=fake_signal,
                    volatility=0.003,
                    price_momentum=0.0,
                    won=won,
                )

            synced_count += 1

        except Exception as e:
            logger.debug(f"Error processing position: {e}")
            skipped_count += 1
            continue

    # Summary
    print("\n" + "-" * 60)
    print("SYNC SUMMARY")
    print("-" * 60)
    print(f"  Total positions fetched: {len(positions)}")
    print(f"  Synced to ML:            {synced_count}")
    print(f"  Non-15min crypto:        {non_crypto_count}")
    print(f"  Skipped/errors:          {skipped_count}")

    if not dry_run and synced_count > 0:
        print(f"\n  ML Model now has: {ml_predictor.training_samples} samples")
        print(f"  Model saved to: {ml_predictor.model_path}")

    if os.path.exists(ml_predictor.model_path):
        size = os.path.getsize(ml_predictor.model_path)
        print(f"  Model file size: {size:,} bytes")

    print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Sync Polymarket closed positions to ML model")
    parser.add_argument("--limit", "-l", type=int, default=500, help="Max positions to fetch")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Don't save to ML")

    args = parser.parse_args()
    sync_trades_to_ml(limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
