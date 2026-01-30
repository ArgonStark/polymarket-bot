#!/usr/bin/env python3
"""
Learn from a successful Polymarket trader.

Fetches their 15-min crypto trades and feeds WIN/LOSS outcomes to the ML model.
This lets your ML learn from successful traders instead of your own trades.

Usage:
    python scripts/learn_from_trader.py <wallet_address> [--limit 5000] [--dry-run]

Example:
    python scripts/learn_from_trader.py 0x1234...abcd --limit 5000
"""

import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace
from src.data.polymarket_data import get_data_api
from src.strategy.ml_predictor import get_ml_predictor
from src.models import Side


def learn_from_trader(wallet: str, limit: int = 5000, dry_run: bool = False):
    """
    Fetch a trader's 15-min crypto trades and train ML on their outcomes.

    Args:
        wallet: Trader's wallet address (0x...)
        limit: Max trades to fetch (default 5000 for more training data)
        dry_run: If True, show what would be learned without saving
    """
    print("\n" + "=" * 70)
    print("  LEARN FROM SUCCESSFUL TRADER")
    print("  Training ML on their winning patterns")
    print("=" * 70 + "\n")

    print(f"Wallet: {wallet}")
    print(f"Limit: {limit} trades")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will update ML)'}\n")

    data_api = get_data_api()
    ml_predictor = get_ml_predictor()

    initial_samples = ml_predictor.training_samples
    print(f"ML Model: {initial_samples} existing samples\n")

    # Fetch closed positions with progress
    print(f"Fetching trader's closed positions (up to {limit})...")
    print("This may take a while for large datasets...\n")
    closed = data_api.get_all_closed_positions(wallet, max_positions=limit)
    print(f"✓ Fetched {len(closed)} positions\n")

    if not closed:
        print("❌ No closed positions found for this wallet")
        return

    print(f"Found {len(closed)} total closed positions\n")

    # Process 15-min crypto trades
    synced_count = 0
    wins = 0
    losses = 0
    skipped = 0

    print("-" * 70)
    print("Processing 15-min crypto trades...")
    print("-" * 70 + "\n")

    for pos in closed:
        try:
            slug = (pos.get("slug", "") or "").lower()
            title = (pos.get("title", "") or "").lower()

            # Check if 15-min crypto
            if not any(p in slug or p in title for p in ["15m", "15-min", "updown"]):
                continue

            # Get outcome (Up/Down)
            outcome = pos.get("outcome", "").lower()
            if outcome == "up":
                side = Side.UP
            elif outcome == "down":
                side = Side.DOWN
            else:
                skipped += 1
                continue

            # Identify asset
            asset = None
            for pattern, name in [("btc", "BTC"), ("eth", "ETH"), ("sol", "SOL"), ("xrp", "XRP")]:
                if pattern in slug or pattern in title:
                    asset = name
                    break

            if not asset:
                skipped += 1
                continue

            # Determine win/loss from P&L
            pnl = float(pos.get("realizedPnl", 0) or 0)
            won = pnl > 0

            # Get trade details
            avg_price = float(pos.get("avgPrice", 0) or 0)
            size = float(pos.get("totalBought", 0) or pos.get("size", 0) or 0)

            if avg_price <= 0:
                skipped += 1
                continue

            # Extract meaningful features from available data
            # 1. Entry price deviation from 0.50 (market mid-point)
            #    - Price < 0.50 means betting on underdog
            #    - Price > 0.50 means betting on favorite
            price_deviation = avg_price - 0.50  # -0.5 to +0.5

            # 2. Estimate edge based on entry price
            #    - If won at low price, had good edge
            #    - If lost at high price, had bad edge
            estimated_edge = (1.0 - avg_price) if won else (avg_price - 1.0)
            estimated_edge = max(0.01, min(0.50, abs(estimated_edge)))

            # 3. Position sizing (normalize by typical size)
            position_size_normalized = min(size / 100, 1.0)

            # 4. Price momentum inference
            #    - UP winners likely had upward momentum
            #    - DOWN winners likely had downward momentum
            inferred_momentum = 0.2 if (side == Side.UP and won) or (side == Side.DOWN and not won) else -0.2

            # 5. Infer volatility from entry price
            #    - Prices near 0.50 suggest high uncertainty/volatility
            #    - Prices near 0 or 1 suggest low volatility
            inferred_volatility = 0.5 - abs(price_deviation)  # 0 to 0.5
            inferred_volatility = max(0.01, inferred_volatility * 0.1)

            # 6. Distance from fair value (0.50)
            distance_from_fair = abs(price_deviation)

            # Create a fake signal/market for ML recording
            fake_market = SimpleNamespace(
                asset=asset,
                condition_id=pos.get("conditionId", ""),
                time_remaining=450,  # Mid-market assumption
                target_price=1.0,  # Normalized
                best_bid=avg_price - 0.01,
                best_ask=avg_price + 0.01,
                bid_depth=1000 * position_size_normalized,
                ask_depth=1000 * position_size_normalized,
            )

            fake_signal = SimpleNamespace(
                market=fake_market,
                side=side,
                edge=estimated_edge,
                recommended_price=avg_price,
                size_shares=size,
                size_usd=size * avg_price,
                time_remaining=450,
                _arb_type="learned",
            )

            # Display (show first 20 and every 100th for large datasets)
            result = "WIN" if won else "LOSS"
            color = "\033[92m" if won else "\033[91m"
            reset = "\033[0m"
            emoji = "▲" if side == Side.UP else "▼"

            if synced_count < 20 or synced_count % 100 == 0:
                print(f"  [{synced_count+1:4d}] {asset} {emoji} {side.value:4} @ {avg_price:.4f} | {color}{result:4}{reset} | ${pnl:+.2f} | edge:{estimated_edge:.0%}")

            if won:
                wins += 1
            else:
                losses += 1

            # Record to ML model with enhanced features
            if not dry_run:
                ml_predictor.record_outcome(
                    signal=fake_signal,
                    volatility=inferred_volatility,
                    price_momentum=inferred_momentum,
                    won=won,
                    arb_type="learned",
                    spread=0.02,
                    bid_depth=1000 * position_size_normalized,
                    ask_depth=1000 * position_size_normalized,
                    price_trend=inferred_momentum * 0.5,
                    distance_from_target=distance_from_fair,
                    binance_lead_pct=inferred_momentum * 0.005,
                    binance_confirmation="MEDIUM" if abs(inferred_momentum) > 0.1 else "NONE",
                    trend_1h=inferred_momentum * 0.3,
                    trend_4h=inferred_momentum * 0.2,
                    trend_1d=inferred_momentum * 0.1,
                    current_price=avg_price,
                    target_price=0.50,  # Fair value
                    price_high=avg_price + inferred_volatility,
                    price_low=avg_price - inferred_volatility,
                    price_velocity=abs(inferred_momentum) * 0.01,
                )

            synced_count += 1

        except Exception as e:
            print(f"  Error processing trade: {e}")
            skipped += 1
            continue

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    if synced_count > 0:
        win_rate = wins / synced_count * 100
        print(f"\n  Trades processed: {synced_count}")
        print(f"  Wins: {wins} | Losses: {losses}")
        print(f"  Win rate: {win_rate:.1f}%")
        print(f"  Skipped: {skipped}")

        if not dry_run:
            final_samples = ml_predictor.training_samples
            print(f"\n  ML samples: {initial_samples} → {final_samples} (+{final_samples - initial_samples})")
            print(f"\n  ✅ ML model updated with {synced_count} trades from successful trader!")
        else:
            print(f"\n  🔍 DRY RUN: Would have added {synced_count} samples to ML")
            print(f"     Run without --dry-run to actually update the model")
    else:
        print(f"\n  ❌ No 15-min crypto trades found for this trader")
        print(f"     Skipped: {skipped} non-crypto trades")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn from a successful Polymarket trader")
    parser.add_argument("wallet", help="Trader's wallet address (0x...)")
    parser.add_argument("--limit", type=int, default=5000, help="Max trades to fetch (default: 5000)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be learned without saving")

    args = parser.parse_args()

    if not args.wallet.startswith("0x"):
        print("ERROR: Wallet address must start with 0x")
        sys.exit(1)

    learn_from_trader(args.wallet, args.limit, args.dry_run)
