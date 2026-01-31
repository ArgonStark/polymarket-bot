#!/usr/bin/env python3
"""
Wallet Analyzer

Analyze a Polymarket wallet's trading history before copying them.
Shows win rate, P&L, trade sizes, and activity patterns.

Usage:
    python analyze_wallet.py 0x1234...
    python analyze_wallet.py 0x1234... --detailed
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.polymarket_data import get_data_api

# Constants
CRYPTO_15M_PATTERNS = ["updown-15m", "15m", "15-min"]
CRYPTO_ASSETS = ["btc", "eth", "sol", "xrp"]


def analyze_wallet(wallet: str, detailed: bool = False):
    """Analyze a wallet's trading activity."""
    wallet = wallet.lower()
    data_api = get_data_api()

    print("=" * 70)
    print(f"  WALLET ANALYSIS: {wallet[:10]}...{wallet[-6:]}")
    print("=" * 70)

    # Get closed positions (completed trades with P&L)
    print("\n📊 Fetching closed positions...")
    closed = data_api.get_all_closed_positions(wallet, max_positions=500)
    print(f"   Found {len(closed)} closed positions")

    # Get recent activity
    print("📊 Fetching recent activity...")
    activity = data_api.get_all_activity(wallet, activity_type="TRADE", max_items=500)
    print(f"   Found {len(activity)} trades")

    # Analyze closed positions
    total_trades = 0
    total_wins = 0
    total_pnl = 0.0
    total_volume = 0.0

    # 15-min crypto specific
    crypto_trades = 0
    crypto_wins = 0
    crypto_pnl = 0.0
    crypto_volume = 0.0

    # Per-asset stats
    asset_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "volume": 0.0})

    # Trade size distribution
    trade_sizes = []

    for pos in closed:
        slug = (pos.get("slug", "") or "").lower()
        title = (pos.get("title", "") or "").lower()
        pnl = float(pos.get("realizedPnl", 0) or 0)
        total_bought = float(pos.get("totalBought", 0) or 0)

        total_trades += 1
        total_pnl += pnl
        total_volume += total_bought

        if pnl > 0:
            total_wins += 1

        trade_sizes.append(total_bought)

        # Check if crypto 15m
        is_crypto_15m = any(p in slug for p in CRYPTO_15M_PATTERNS)
        asset = None
        for a in CRYPTO_ASSETS:
            if a in slug or a in title:
                asset = a.upper()
                break

        if is_crypto_15m and asset:
            crypto_trades += 1
            crypto_pnl += pnl
            crypto_volume += total_bought
            if pnl > 0:
                crypto_wins += 1

            asset_stats[asset]["trades"] += 1
            asset_stats[asset]["pnl"] += pnl
            asset_stats[asset]["volume"] += total_bought
            if pnl > 0:
                asset_stats[asset]["wins"] += 1

    # Calculate stats
    win_rate = total_wins / total_trades if total_trades > 0 else 0
    crypto_win_rate = crypto_wins / crypto_trades if crypto_trades > 0 else 0
    avg_trade_size = sum(trade_sizes) / len(trade_sizes) if trade_sizes else 0
    median_trade_size = sorted(trade_sizes)[len(trade_sizes) // 2] if trade_sizes else 0

    # Print overall stats
    print("\n" + "=" * 70)
    print("  OVERALL STATS")
    print("=" * 70)
    print(f"  Total trades:     {total_trades}")
    print(f"  Win rate:         {win_rate:.1%} ({total_wins}W / {total_trades - total_wins}L)")
    print(f"  Total P&L:        ${total_pnl:+,.2f}")
    print(f"  Total volume:     ${total_volume:,.2f}")
    print(f"  Avg trade size:   ${avg_trade_size:.2f}")
    print(f"  Median size:      ${median_trade_size:.2f}")

    # Print crypto 15m stats
    print("\n" + "=" * 70)
    print("  15-MIN CRYPTO STATS (What we copy)")
    print("=" * 70)
    print(f"  Crypto trades:    {crypto_trades}")
    print(f"  Crypto win rate:  {crypto_win_rate:.1%} ({crypto_wins}W / {crypto_trades - crypto_wins}L)")
    print(f"  Crypto P&L:       ${crypto_pnl:+,.2f}")
    print(f"  Crypto volume:    ${crypto_volume:,.2f}")

    # Print per-asset stats
    if asset_stats:
        print("\n" + "-" * 70)
        print("  Per-Asset Breakdown:")
        print("-" * 70)
        for asset in ["BTC", "ETH", "SOL", "XRP"]:
            if asset in asset_stats:
                stats = asset_stats[asset]
                asset_wr = stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0
                print(
                    f"  {asset}: {stats['trades']} trades, {asset_wr:.0%} win rate, "
                    f"${stats['pnl']:+,.2f} P&L"
                )

    # Recommendation
    print("\n" + "=" * 70)
    print("  RECOMMENDATION")
    print("=" * 70)

    issues = []
    if crypto_trades < 10:
        issues.append(f"Low sample size ({crypto_trades} crypto trades)")
    if crypto_win_rate < 0.50:
        issues.append(f"Win rate below 50% ({crypto_win_rate:.0%})")
    if crypto_pnl < 0:
        issues.append(f"Negative P&L (${crypto_pnl:+,.2f})")

    if not issues:
        print("  ✅ GOOD TO COPY")
        print(f"     {crypto_win_rate:.0%} win rate with ${crypto_pnl:+,.2f} profit")
    else:
        print("  ⚠️  CAUTION - Issues found:")
        for issue in issues:
            print(f"     • {issue}")

    # Funding check
    print("\n" + "=" * 70)
    print("  FUNDING CHECK")
    print("=" * 70)
    print(f"  Their avg trade:  ${avg_trade_size:.2f}")
    print(f"  Their median:     ${median_trade_size:.2f}")
    print("\n  Suggested settings for YOUR balance:")
    print("  ┌─────────────────┬─────────────────┬─────────────────┐")
    print("  │ Your Balance    │ Size (5%)       │ Recommended     │")
    print("  ├─────────────────┼─────────────────┼─────────────────┤")
    for bal in [50, 100, 200, 500, 1000]:
        size = bal * 0.05
        size = max(5, min(50, size))  # Clamp to $5-$50
        print(f"  │ ${bal:>12,.0f}   │ ${size:>12.2f}   │ {'OK' if size >= 5 else 'Too low':^15} │")
    print("  └─────────────────┴─────────────────┴─────────────────┘")

    # Recent activity (if detailed)
    if detailed and activity:
        print("\n" + "=" * 70)
        print("  RECENT TRADES (Last 20)")
        print("=" * 70)

        crypto_activity = []
        for trade in activity:
            slug = (trade.get("slug", "") or "").lower()
            if any(p in slug for p in CRYPTO_15M_PATTERNS):
                crypto_activity.append(trade)

        for trade in crypto_activity[:20]:
            slug = (trade.get("slug", "") or "").lower()
            side = trade.get("side", "")
            outcome = trade.get("outcome", "")
            size = float(trade.get("size", 0) or 0)
            price = float(trade.get("price", 0) or 0)
            usdc = float(trade.get("usdcSize", 0) or 0)
            ts = trade.get("timestamp", "")

            # Parse asset
            asset = "???"
            for a in CRYPTO_ASSETS:
                if a in slug:
                    asset = a.upper()
                    break

            print(f"  {ts[:19]} | {asset} | {side} {outcome} | {size:.1f} @ ${price:.3f} = ${usdc:.2f}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze a wallet's trading activity")
    parser.add_argument("wallet", help="Wallet address to analyze")
    parser.add_argument("--detailed", "-d", action="store_true", help="Show recent trades")

    args = parser.parse_args()

    if not args.wallet.startswith("0x"):
        print("Error: Wallet must start with 0x")
        sys.exit(1)

    analyze_wallet(args.wallet, detailed=args.detailed)


if __name__ == "__main__":
    main()
