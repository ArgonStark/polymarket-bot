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
from src.strategy.ml_predictor import get_ml_predictor
from src.models import Side

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_config() -> BotConfig:
    """Load bot configuration."""
    return BotConfig()


def build_market_lookup(gamma_api: GammaAPI, hours_back: int = 24) -> dict:
    """
    Build a lookup table of condition_id -> market info for recent 15-min crypto markets.

    This fetches markets by constructing slugs for recent time periods.
    """
    lookup = {}
    now = int(time.time())

    # Round down to current 15-min period
    current_period = (now // 900) * 900

    # Go back N hours (4 periods per hour)
    periods_to_check = hours_back * 4

    assets = ["btc", "eth", "sol", "xrp"]

    print(f"Building market lookup (last {hours_back} hours)...")

    for i in range(periods_to_check):
        ts = current_period - (i * 900)

        for asset in assets:
            slug = f"{asset}-updown-15m-{ts}"

            try:
                # Use the existing _fetch_market_by_slug method pattern
                import requests
                url = f"{gamma_api.base_url}/markets"
                params = {"slug": slug}
                response = requests.get(url, params=params, timeout=5)

                if response.status_code == 200:
                    markets = response.json()
                    if markets and len(markets) > 0:
                        market = markets[0]
                        condition_id = market.get("conditionId", "")
                        if condition_id:
                            # Check if resolved
                            resolved = market.get("closed", False)
                            winning_outcome = None

                            if resolved:
                                # Try to get winner from tokens
                                tokens = market.get("tokens", [])
                                for token in tokens:
                                    if token.get("winner", False):
                                        outcome_str = token.get("outcome", "").lower()
                                        if "up" in outcome_str:
                                            winning_outcome = "UP"
                                        elif "down" in outcome_str:
                                            winning_outcome = "DOWN"
                                        break

                            lookup[condition_id] = {
                                "slug": slug,
                                "asset": asset.upper(),
                                "resolved": resolved,
                                "winning_outcome": winning_outcome,
                                "market_ts": ts,
                            }
            except Exception:
                pass

        # Progress indicator
        if i % 20 == 0 and i > 0:
            print(f"  Checked {i}/{periods_to_check} periods, found {len(lookup)} markets...")

    print(f"  Found {len(lookup)} 15-min crypto markets\n")
    return lookup


def sync_trades_to_ml(limit: int = 100, dry_run: bool = False, hours_back: int = 24):
    """
    Sync trades from Polymarket to ML model.

    Args:
        limit: Maximum number of trades to fetch
        dry_run: If True, don't actually save to ML model
        hours_back: Hours of market history to fetch for lookup
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

    # Initialize Gamma API
    gamma_api = GammaAPI(config=config)

    # Build lookup table of recent 15-min crypto markets
    market_lookup = build_market_lookup(gamma_api, hours_back=hours_back)

    # Fetch trades from Polymarket
    print(f"Fetching last {limit} trades from Polymarket API...")
    trades = get_trades(client, limit=limit)

    if not trades:
        print("No trades found")
        return

    print(f"Found {len(trades)} trades\n")

    # Track stats
    synced_count = 0
    pending_count = 0
    non_crypto_count = 0
    skipped_count = 0
    current_ts = int(time.time())

    print("Processing trades...\n")

    for i, trade in enumerate(trades):
        try:
            trade_id = trade.get("id", "")
            condition_id = trade.get("market", "")

            if not condition_id:
                skipped_count += 1
                continue

            # Look up in our pre-built table
            market_info = market_lookup.get(condition_id)

            if not market_info:
                non_crypto_count += 1
                continue

            slug = market_info["slug"]
            asset = market_info["asset"]
            market_ts = market_info["market_ts"]

            # Check if market has settled (15 min + buffer)
            settle_time = market_ts + 900
            if current_ts < settle_time + 60:
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
            result_color = "\033[92m" if won else "\033[91m"
            reset = "\033[0m"

            print(
                f"[{i+1:3d}] {asset:4s} {side.value:4s} @ {trade_price:.4f} | "
                f"Size: {trade_size:8.2f} | {result_color}{result:4s}{reset} | "
                f"Slug: {slug}"
            )

            if not dry_run:
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

                ml_predictor.record_outcome(
                    signal=fake_signal,
                    volatility=0.003,
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

    if os.path.exists(ml_predictor.model_path):
        size = os.path.getsize(ml_predictor.model_path)
        print(f"  Model file size: {size:,} bytes")

    print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Sync Polymarket trades to ML model")
    parser.add_argument("--limit", "-l", type=int, default=100, help="Max trades to fetch")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Don't save to ML")
    parser.add_argument("--hours", "-H", type=int, default=48, help="Hours of market history")

    args = parser.parse_args()
    sync_trades_to_ml(limit=args.limit, dry_run=args.dry_run, hours_back=args.hours)


if __name__ == "__main__":
    main()
