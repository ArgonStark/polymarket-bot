#!/usr/bin/env python3
"""
Manual ML Sync from Polymarket Trade History

This script fetches your trade history from Polymarket Data API and syncs
it with the ML model. It uses all three endpoints:
- /positions - current open positions
- /activity - all trade activity (buys/sells)
- /closed-positions - completed trades with P&L

The ML learns from closed positions where win/loss is determined by realizedPnl.

Usage:
    python scripts/sync_ml_from_trades.py [--limit 500] [--dry-run]
"""

import os
import sys
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import BotConfig
from src.execution import create_trading_client
from src.strategy.ml_predictor import get_ml_predictor
from src.data.polymarket_data import get_data_api
from src.models import Side

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def sync_trades_to_ml(limit: int = 500, dry_run: bool = False, show_activity: bool = False):
    """
    Sync trades from Polymarket to ML model using all three APIs.

    Args:
        limit: Maximum number of items to fetch per endpoint
        dry_run: If True, don't actually save to ML model
        show_activity: If True, also show recent trade activity
    """
    print("\n" + "=" * 70)
    print("  POLYMARKET -> ML SYNC (Full Trade History)")
    print("=" * 70 + "\n")

    # Load config and create client to get wallet address
    config = BotConfig()
    client = create_trading_client(config)

    if not client:
        print("ERROR: Failed to create trading client")
        print("Make sure your environment variables are set (PK, etc.)")
        return

    # Get wallet address
    wallet_address = client.get_address()
    if not wallet_address:
        print("ERROR: Could not get wallet address")
        return

    print(f"Wallet: {wallet_address}")

    # Initialize ML predictor and Data API
    ml_predictor = get_ml_predictor()
    data_api = get_data_api()
    print(f"ML Model loaded: {ml_predictor.training_samples} existing samples\n")

    # ==================== CURRENT POSITIONS ====================
    print("-" * 70)
    print("1. CURRENT POSITIONS (Open)")
    print("-" * 70)

    positions = data_api.get_positions(wallet_address, limit=limit)
    crypto_positions = []

    for pos in positions:
        slug = pos.get("slug", "") or pos.get("eventSlug", "")
        if "updown-15m" in slug.lower() or "15m" in pos.get("title", "").lower():
            crypto_positions.append(pos)

    if crypto_positions:
        print(f"Found {len(crypto_positions)} open 15-min crypto positions:\n")
        for pos in crypto_positions[:10]:  # Show max 10
            asset = pos.get("asset", "")[:8]
            outcome = pos.get("outcome", "")
            size = float(pos.get("size", 0) or 0)
            avg_price = float(pos.get("avgPrice", 0) or 0)
            cur_price = float(pos.get("curPrice", 0) or 0)
            cash_pnl = float(pos.get("cashPnl", 0) or 0)
            pnl_color = "\033[92m" if cash_pnl >= 0 else "\033[91m"
            reset = "\033[0m"
            print(
                f"  {outcome:5s} | Size: {size:8.2f} | Avg: {avg_price:.4f} | "
                f"Cur: {cur_price:.4f} | {pnl_color}P&L: ${cash_pnl:+.2f}{reset}"
            )
        if len(crypto_positions) > 10:
            print(f"  ... and {len(crypto_positions) - 10} more")
    else:
        print("No open 15-min crypto positions")
    print()

    # ==================== TRADE ACTIVITY ====================
    if show_activity:
        print("-" * 70)
        print("2. RECENT TRADE ACTIVITY")
        print("-" * 70)

        activity = data_api.get_all_activity(wallet_address, activity_type="TRADE", max_items=limit)
        crypto_activity = []

        for act in activity:
            slug = act.get("slug", "") or act.get("eventSlug", "")
            if "updown-15m" in slug.lower() or "15m" in act.get("title", "").lower():
                crypto_activity.append(act)

        if crypto_activity:
            print(f"Found {len(crypto_activity)} 15-min crypto trades:\n")
            for act in crypto_activity[:15]:  # Show max 15
                side = act.get("side", "")
                outcome = act.get("outcome", "")
                size = float(act.get("size", 0) or 0)
                price = float(act.get("price", 0) or 0)
                usdc = float(act.get("usdcSize", 0) or 0)
                timestamp = act.get("timestamp", "")[:19]
                side_color = "\033[92m" if side == "BUY" else "\033[91m"
                reset = "\033[0m"
                print(
                    f"  {timestamp} | {side_color}{side:4s}{reset} {outcome:5s} | "
                    f"Size: {size:8.2f} @ {price:.4f} | ${usdc:.2f}"
                )
            if len(crypto_activity) > 15:
                print(f"  ... and {len(crypto_activity) - 15} more")
        else:
            print("No recent 15-min crypto trade activity")
        print()

    # ==================== CLOSED POSITIONS (ML SYNC) ====================
    print("-" * 70)
    print("3. CLOSED POSITIONS (Completed Trades for ML)")
    print("-" * 70)

    closed = data_api.get_all_closed_positions(wallet_address, max_positions=limit)

    if not closed:
        print("No closed positions found")
        return

    print(f"Found {len(closed)} total closed positions\n")

    # Show sample slugs for debugging
    print("Sample slugs from API (first 5):")
    for pos in closed[:5]:
        slug = pos.get("slug", "") or pos.get("eventSlug", "")
        title = pos.get("title", "")
        print(f"  slug={slug!r}  title={title!r}")
    print()

    # Track stats
    synced_count = 0
    non_crypto_count = 0
    skipped_count = 0
    wins = 0
    losses = 0

    print("Processing 15-min crypto positions...\n")

    for i, pos in enumerate(closed):
        try:
            slug = pos.get("slug", "") or pos.get("eventSlug", "")
            title = pos.get("title", "")
            slug_lower = slug.lower()
            title_lower = title.lower()

            # Check if this is a 15-min crypto market
            # Patterns: "updown-15m", "15m", "15min", "15-min", "up-down-15m"
            is_15min = any(p in slug_lower or p in title_lower for p in [
                "updown-15m", "-15m-", "15min", "15-min", "15m"
            ])

            if not is_15min:
                non_crypto_count += 1
                continue

            # Identify asset from slug or title
            asset = None
            for a in ["btc", "eth", "sol", "xrp", "bitcoin", "ethereum", "solana"]:
                asset_name = a[:3].upper() if len(a) > 3 else a.upper()
                if a in slug_lower or a in title_lower:
                    asset = asset_name
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
                f"[{synced_count+1:3d}] {asset:4s} {side.value:4s} @ {avg_price:.4f} | "
                f"Size: {total_bought:8.2f} | {result_color}{result:4s}{reset} | "
                f"P&L: {pnl_str}"
            )

            if won:
                wins += 1
            else:
                losses += 1

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
    print("\n" + "=" * 70)
    print("SYNC SUMMARY")
    print("=" * 70)
    print(f"  Total closed positions:  {len(closed)}")
    print(f"  15-min crypto synced:    {synced_count}")
    print(f"  Non-crypto positions:    {non_crypto_count}")
    print(f"  Skipped/errors:          {skipped_count}")
    print()
    if synced_count > 0:
        win_rate = wins / synced_count * 100
        print(f"  Wins:  {wins:3d} ({win_rate:.1f}%)")
        print(f"  Losses: {losses:3d} ({100-win_rate:.1f}%)")

    if not dry_run:
        # Always save the model (even if 0 synced) to create the file
        if synced_count > 0:
            print(f"\n  ML Model now has: {ml_predictor.training_samples} samples")
        else:
            # Force save to create the file if it doesn't exist
            ml_predictor.save()
            print(f"\n  No 15-min crypto positions found to sync")
            print(f"  Created empty ML model at: {ml_predictor.model_path}")
        print(f"  Model saved to: {ml_predictor.model_path}")

    if os.path.exists(ml_predictor.model_path):
        size = os.path.getsize(ml_predictor.model_path)
        print(f"  Model file size: {size:,} bytes")

    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Sync Polymarket trade history to ML model"
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=500,
        help="Max items to fetch per endpoint (default: 500)"
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Preview without saving to ML model"
    )
    parser.add_argument(
        "--activity", "-a", action="store_true",
        help="Also show recent trade activity"
    )

    args = parser.parse_args()
    sync_trades_to_ml(limit=args.limit, dry_run=args.dry_run, show_activity=args.activity)


if __name__ == "__main__":
    main()
