#!/usr/bin/env python3
"""
Analyze a successful Polymarket trader's activity to learn winning patterns.

Usage:
    python scripts/analyze_successful_trader.py [wallet_address_or_username]
"""

import os
import sys
import requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.polymarket_data import get_data_api

# Known successful traders
KNOWN_TRADERS = {
    "0x8dxd": "0x8d8d9d8f9d8d9d8f9d8d9d8f9d8d9d8f9d8d9d8f",  # Placeholder - need real address
}


def get_wallet_from_username(username: str) -> str:
    """Try to get wallet address from Polymarket username."""
    # The profile page has the wallet address
    # We can try the Polymarket API or scrape it

    # Try the profile API
    try:
        url = f"https://polymarket.com/api/profile/{username.lower()}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            wallet = data.get("address") or data.get("wallet")
            if wallet:
                return wallet
    except Exception as e:
        print(f"Could not fetch profile: {e}")

    # Try gamma API
    try:
        url = f"https://gamma-api.polymarket.com/users/{username}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            wallet = data.get("address") or data.get("walletAddress")
            if wallet:
                return wallet
    except:
        pass

    return None


def analyze_trader(wallet: str, limit: int = 500):
    """Analyze a trader's activity to find winning patterns."""

    print("\n" + "=" * 70)
    print("  SUCCESSFUL TRADER ANALYSIS")
    print("=" * 70 + "\n")

    print(f"Wallet: {wallet}\n")

    data_api = get_data_api()

    # Get closed positions (completed trades with P&L)
    print("Fetching closed positions...")
    closed = data_api.get_all_closed_positions(wallet, max_positions=limit)

    if not closed:
        print("No closed positions found")
        return

    print(f"Found {len(closed)} total closed positions\n")

    # Filter to 15-min crypto
    crypto_trades = []
    for pos in closed:
        slug = (pos.get("slug", "") or "").lower()
        title = (pos.get("title", "") or "").lower()

        if not any(p in slug or p in title for p in ["15m", "15-min", "updown"]):
            continue

        outcome = pos.get("outcome", "").lower()
        if outcome not in ["up", "down"]:
            continue

        # Identify asset
        asset = None
        for pattern, name in [("btc", "BTC"), ("eth", "ETH"), ("sol", "SOL"), ("xrp", "XRP")]:
            if pattern in slug or pattern in title:
                asset = name
                break

        if not asset:
            continue

        pnl = float(pos.get("realizedPnl", 0) or 0)
        avg_price = float(pos.get("avgPrice", 0) or 0)
        size = float(pos.get("totalBought", 0) or pos.get("size", 0) or 0)

        crypto_trades.append({
            "asset": asset,
            "side": outcome.upper(),
            "won": pnl > 0,
            "pnl": pnl,
            "entry_price": avg_price,
            "size": size,
            "slug": slug,
        })

    if not crypto_trades:
        print("No 15-min crypto trades found for this trader")
        print("\nShowing sample slugs from their trades:")
        for pos in closed[:10]:
            slug = pos.get("slug", "")
            title = pos.get("title", "")
            pnl = float(pos.get("realizedPnl", 0) or 0)
            print(f"  [{'+' if pnl > 0 else '-'}] {slug[:50]} | ${pnl:+.2f}")
        return

    print(f"Found {len(crypto_trades)} 15-min crypto trades\n")

    # Calculate overall stats
    total_wins = sum(1 for t in crypto_trades if t["won"])
    total_pnl = sum(t["pnl"] for t in crypto_trades)
    win_rate = total_wins / len(crypto_trades) * 100

    print("-" * 70)
    print(f"OVERALL: {len(crypto_trades)} trades | {win_rate:.1f}% win rate | ${total_pnl:+.2f}")
    print("-" * 70)

    # By asset
    print("\nBY ASSET:")
    for asset in ["BTC", "ETH", "SOL", "XRP"]:
        trades = [t for t in crypto_trades if t["asset"] == asset]
        if not trades:
            continue
        wins = sum(1 for t in trades if t["won"])
        pnl = sum(t["pnl"] for t in trades)
        wr = wins / len(trades) * 100

        bar = "█" * int(wr / 5)
        color = "\033[92m" if wr >= 50 else "\033[91m"
        reset = "\033[0m"
        print(f"  {asset:4} | {len(trades):4} trades | {color}{wr:5.1f}%{reset} | ${pnl:+10.2f} | {bar}")

    # By side
    print("\nBY SIDE:")
    for side in ["UP", "DOWN"]:
        trades = [t for t in crypto_trades if t["side"] == side]
        if not trades:
            continue
        wins = sum(1 for t in trades if t["won"])
        pnl = sum(t["pnl"] for t in trades)
        wr = wins / len(trades) * 100

        emoji = "▲" if side == "UP" else "▼"
        bar = "█" * int(wr / 5)
        color = "\033[92m" if wr >= 50 else "\033[91m"
        reset = "\033[0m"
        print(f"  {emoji} {side:5} | {len(trades):4} trades | {color}{wr:5.1f}%{reset} | ${pnl:+10.2f} | {bar}")

    # By entry price
    print("\nBY ENTRY PRICE:")
    ranges = [
        ("Very cheap (<0.25)", lambda p: p < 0.25),
        ("Cheap (0.25-0.40)", lambda p: 0.25 <= p < 0.40),
        ("Mid (0.40-0.60)", lambda p: 0.40 <= p <= 0.60),
        ("Expensive (0.60-0.75)", lambda p: 0.60 < p <= 0.75),
        ("Very expensive (>0.75)", lambda p: p > 0.75),
    ]

    for name, filter_fn in ranges:
        trades = [t for t in crypto_trades if filter_fn(t["entry_price"])]
        if not trades:
            continue
        wins = sum(1 for t in trades if t["won"])
        pnl = sum(t["pnl"] for t in trades)
        wr = wins / len(trades) * 100

        bar = "█" * int(wr / 5)
        color = "\033[92m" if wr >= 50 else "\033[91m"
        reset = "\033[0m"
        print(f"  {name:22} | {len(trades):4} trades | {color}{wr:5.1f}%{reset} | ${pnl:+10.2f} | {bar}")

    # Best combinations
    print("\nBEST COMBINATIONS (sorted by win rate):")
    combos = []
    for asset in ["BTC", "ETH", "SOL", "XRP"]:
        for side in ["UP", "DOWN"]:
            trades = [t for t in crypto_trades if t["asset"] == asset and t["side"] == side]
            if len(trades) < 3:  # Need at least 3 trades
                continue
            wins = sum(1 for t in trades if t["won"])
            pnl = sum(t["pnl"] for t in trades)
            wr = wins / len(trades) * 100
            combos.append((asset, side, len(trades), wr, pnl))

    combos.sort(key=lambda x: x[3], reverse=True)

    for asset, side, count, wr, pnl in combos[:8]:
        emoji = "▲" if side == "UP" else "▼"
        bar = "█" * int(wr / 5)
        color = "\033[92m" if wr >= 50 else "\033[91m"
        reset = "\033[0m"
        print(f"  {asset} {emoji} {side:5} | {count:4} trades | {color}{wr:5.1f}%{reset} | ${pnl:+10.2f} | {bar}")

    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS FOR YOUR BOT:")
    print("=" * 70)

    if combos:
        best = combos[0]
        worst = combos[-1]

        if best[3] >= 55:
            print(f"\n✅ FOCUS ON: {best[0]} {best[1]} trades ({best[3]:.1f}% win rate)")

        if worst[3] < 45:
            print(f"❌ AVOID: {worst[0]} {worst[1]} trades ({worst[3]:.1f}% win rate)")

    # Average winning entry price vs losing
    winning_entries = [t["entry_price"] for t in crypto_trades if t["won"]]
    losing_entries = [t["entry_price"] for t in crypto_trades if not t["won"]]

    if winning_entries and losing_entries:
        avg_win_entry = sum(winning_entries) / len(winning_entries)
        avg_lose_entry = sum(losing_entries) / len(losing_entries)

        print(f"\n📊 Average entry price:")
        print(f"   Winning trades: {avg_win_entry:.4f}")
        print(f"   Losing trades:  {avg_lose_entry:.4f}")

        if avg_win_entry < avg_lose_entry:
            print(f"   → Winners tend to enter at LOWER prices (better value)")
        else:
            print(f"   → Winners tend to enter at HIGHER prices (momentum?)")


if __name__ == "__main__":
    # Default to the successful trader
    wallet = sys.argv[1] if len(sys.argv) > 1 else None

    if not wallet:
        print("Trying to find wallet for @0x8dxd...")
        wallet = get_wallet_from_username("0x8dxd")

    if not wallet:
        # Try directly as a wallet if it looks like one
        print("\nUsage: python scripts/analyze_successful_trader.py <wallet_address>")
        print("\nTo find a trader's wallet address:")
        print("1. Go to their Polymarket profile")
        print("2. Click on any transaction")
        print("3. Copy the wallet address from the transaction")
        sys.exit(1)

    analyze_trader(wallet)
