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
import requests
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


def get_market_info(condition_id: str, gamma_api: GammaAPI, cache: dict) -> dict:
    """
    Get market info from condition ID, with caching.

    Returns dict with: slug, asset, resolved, winning_outcome
    """
    if condition_id in cache:
        return cache[condition_id]

    try:
        # Use query parameter filtering (gamma_api.get_market_by_id now uses this)
        market = gamma_api.get_market_by_id(condition_id)
        if market:
            slug = market.get("slug", "")

            # Extract asset from slug (e.g., btc-updown-15m-1234567890)
            asset = None
            for a in ["btc", "eth", "sol", "xrp"]:
                if slug.lower().startswith(a):
                    asset = a.upper()
                    break

            # Check if resolved
            resolved = market.get("resolved", False)
            winning_outcome = None

            if resolved:
                tokens = market.get("tokens", [])
                for token in tokens:
                    if token.get("winner", False):
                        outcome_str = token.get("outcome", "").lower()
                        if "up" in outcome_str or "yes" in outcome_str:
                            winning_outcome = "UP"
                        elif "down" in outcome_str or "no" in outcome_str:
                            winning_outcome = "DOWN"
                        break

            info = {
                "slug": slug,
                "asset": asset,
                "resolved": resolved,
                "winning_outcome": winning_outcome,
            }
            cache[condition_id] = info
            return info
        else:
            # API returned empty - cache as None to avoid repeated calls
            logger.debug(f"No market found for condition_id: {condition_id[:16]}...")
    except Exception as e:
        logger.debug(f"Error fetching market {condition_id[:16]}...: {e}")

    cache[condition_id] = None
    return None


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

    # Initialize Gamma API for market lookups
    gamma_api = GammaAPI(config=config)

    # Fetch trades from Polymarket
    print(f"\nFetching last {limit} trades from Polymarket API...")
    trades = get_trades(client, limit=limit)

    if not trades:
        print("No trades found")
        return

    print(f"Found {len(trades)} trades\n")

    # Debug: show first few trades structure
    print("DEBUG: Sample trade structure (first 3 trades):")
    print("-" * 60)
    for i, trade in enumerate(trades[:3]):
        print(f"\nTrade {i+1} keys: {list(trade.keys())}")
        # Print key fields
        for key in ["id", "market", "asset_id", "outcome", "side", "price", "size", "status", "match_time"]:
            if key in trade:
                print(f"  {key}: {trade[key]}")
        print()
    print("-" * 60 + "\n")

    # Track synced trades
    synced_count = 0
    skipped_count = 0
    pending_count = 0
    non_crypto_count = 0
    current_ts = int(time.time())

    # Cache for market info lookups
    market_cache = {}

    print("Processing trades (looking up markets)...\n")

    # Process each trade
    for i, trade in enumerate(trades):
        try:
            trade_id = trade.get("id", "")
            condition_id = trade.get("market", "")  # This is the condition ID

            if not condition_id:
                skipped_count += 1
                continue

            # Look up market info
            market_info = get_market_info(condition_id, gamma_api, market_cache)

            if not market_info:
                skipped_count += 1
                continue

            slug = market_info.get("slug", "")
            asset = market_info.get("asset")

            # Check if this is a 15-min crypto market
            if "updown-15m" not in slug.lower():
                non_crypto_count += 1
                continue

            if not asset:
                skipped_count += 1
                continue

            # Extract market timestamp from slug (e.g., btc-updown-15m-1234567890)
            market_ts = None
            try:
                parts = slug.split("-")
                if len(parts) >= 4:
                    market_ts = int(parts[-1])
            except (ValueError, IndexError):
                skipped_count += 1
                continue

            if not market_ts:
                skipped_count += 1
                continue

            # Check if market has settled (15 min = 900 seconds after start)
            settle_time = market_ts + 900
            if current_ts < settle_time + 60:  # Wait 60s after settle
                pending_count += 1
                continue

            # Determine trade side from outcome field
            outcome = trade.get("outcome", "")
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

            # Get winning outcome
            winning_outcome = market_info.get("winning_outcome")

            if not winning_outcome:
                # Market not resolved yet
                pending_count += 1
                continue

            # Determine if trade won
            won = side.value == winning_outcome

            # Get trade details
            trade_price = float(trade.get("price", 0) or 0)
            trade_size = float(trade.get("size", 0) or 0)

            if trade_price <= 0 or trade_size <= 0:
                skipped_count += 1
                continue

            # Display trade info
            result = "WIN" if won else "LOSS"
            result_color = "\033[92m" if won else "\033[91m"  # Green/Red
            reset = "\033[0m"

            print(
                f"[{i+1:3d}] {asset:4s} {side.value:4s} @ {trade_price:.4f} | "
                f"Size: {trade_size:8.2f} | {result_color}{result:4s}{reset} | "
                f"Slug: {slug}"
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
            skipped_count += 1
            continue

    # Summary
    print("\n" + "-" * 60)
    print("SYNC SUMMARY")
    print("-" * 60)
    print(f"  Total trades fetched: {len(trades)}")
    print(f"  Synced to ML:         {synced_count}")
    print(f"  Pending settlement:   {pending_count}")
    print(f"  Non-15min crypto:     {non_crypto_count}")
    print(f"  Skipped/errors:       {skipped_count}")

    if not dry_run and synced_count > 0:
        print(f"\n  ML Model now has: {ml_predictor.training_samples} samples")
        print(f"  Model saved to: {ml_predictor.model_path}")

    # Check if model file was created
    if os.path.exists(ml_predictor.model_path):
        size = os.path.getsize(ml_predictor.model_path)
        print(f"  Model file size: {size:,} bytes")
    elif synced_count > 0:
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
