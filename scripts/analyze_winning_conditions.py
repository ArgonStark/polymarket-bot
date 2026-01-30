#!/usr/bin/env python3
"""
Analyze what conditions actually predict winning outcomes.

Instead of learning from "did my trade win", we analyze:
- What conditions existed when UP won vs DOWN won
- What features actually correlate with the correct outcome
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import BotConfig
from src.execution import create_trading_client
from src.data.polymarket_data import get_data_api
from src.execution.client import get_trades

def analyze_trades():
    """Analyze actual trade outcomes to find what predicts success."""

    print("\n" + "=" * 70)
    print("  WINNING CONDITIONS ANALYSIS")
    print("  What actually predicts market outcomes?")
    print("=" * 70 + "\n")

    config = BotConfig()
    client = create_trading_client(config)

    if not client:
        print("ERROR: Need credentials to fetch trades")
        return

    wallet = client.get_address()
    print(f"Wallet: {wallet[:12]}...\n")

    data_api = get_data_api()

    # Get closed positions
    closed = data_api.get_all_closed_positions(wallet, max_positions=500)

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

        crypto_trades.append({
            "asset": asset,
            "side": outcome.upper(),
            "won": pnl > 0,
            "pnl": pnl,
            "entry_price": avg_price,
        })

    if not crypto_trades:
        print("No 15-min crypto trades found")
        return

    print(f"Found {len(crypto_trades)} 15-min crypto trades\n")

    # Analyze by asset
    print("-" * 70)
    print("BY ASSET: Which assets are most profitable?")
    print("-" * 70)

    for asset in ["BTC", "ETH", "SOL", "XRP"]:
        trades = [t for t in crypto_trades if t["asset"] == asset]
        if not trades:
            continue
        wins = sum(1 for t in trades if t["won"])
        total_pnl = sum(t["pnl"] for t in trades)
        win_rate = wins / len(trades) * 100

        bar = "█" * int(win_rate / 5)
        color = "\033[92m" if win_rate >= 50 else "\033[91m"
        reset = "\033[0m"

        print(f"  {asset:4} | {len(trades):3} trades | {color}{win_rate:5.1f}%{reset} | ${total_pnl:+8.2f} | {bar}")

    # Analyze by side
    print("\n" + "-" * 70)
    print("BY SIDE: UP vs DOWN - which is more successful?")
    print("-" * 70)

    for side in ["UP", "DOWN"]:
        trades = [t for t in crypto_trades if t["side"] == side]
        if not trades:
            continue
        wins = sum(1 for t in trades if t["won"])
        total_pnl = sum(t["pnl"] for t in trades)
        win_rate = wins / len(trades) * 100

        emoji = "▲" if side == "UP" else "▼"
        bar = "█" * int(win_rate / 5)
        color = "\033[92m" if win_rate >= 50 else "\033[91m"
        reset = "\033[0m"

        print(f"  {emoji} {side:5} | {len(trades):3} trades | {color}{win_rate:5.1f}%{reset} | ${total_pnl:+8.2f} | {bar}")

    # Analyze by entry price ranges
    print("\n" + "-" * 70)
    print("BY ENTRY PRICE: Cheap vs Expensive entries")
    print("-" * 70)

    cheap = [t for t in crypto_trades if t["entry_price"] < 0.40]
    mid = [t for t in crypto_trades if 0.40 <= t["entry_price"] <= 0.60]
    expensive = [t for t in crypto_trades if t["entry_price"] > 0.60]

    for name, trades in [("Cheap (<0.40)", cheap), ("Mid (0.40-0.60)", mid), ("Expensive (>0.60)", expensive)]:
        if not trades:
            continue
        wins = sum(1 for t in trades if t["won"])
        total_pnl = sum(t["pnl"] for t in trades)
        win_rate = wins / len(trades) * 100

        bar = "█" * int(win_rate / 5)
        color = "\033[92m" if win_rate >= 50 else "\033[91m"
        reset = "\033[0m"

        print(f"  {name:18} | {len(trades):3} trades | {color}{win_rate:5.1f}%{reset} | ${total_pnl:+8.2f} | {bar}")

    # Cross-analysis: Asset + Side
    print("\n" + "-" * 70)
    print("BY ASSET + SIDE: Best combinations")
    print("-" * 70)

    combos = []
    for asset in ["BTC", "ETH", "SOL", "XRP"]:
        for side in ["UP", "DOWN"]:
            trades = [t for t in crypto_trades if t["asset"] == asset and t["side"] == side]
            if not trades:
                continue
            wins = sum(1 for t in trades if t["won"])
            total_pnl = sum(t["pnl"] for t in trades)
            win_rate = wins / len(trades) * 100
            combos.append((asset, side, len(trades), win_rate, total_pnl))

    # Sort by win rate
    combos.sort(key=lambda x: x[3], reverse=True)

    for asset, side, count, win_rate, total_pnl in combos:
        emoji = "▲" if side == "UP" else "▼"
        bar = "█" * int(win_rate / 5)
        color = "\033[92m" if win_rate >= 50 else "\033[91m"
        reset = "\033[0m"

        print(f"  {asset} {emoji} {side:5} | {count:3} trades | {color}{win_rate:5.1f}%{reset} | ${total_pnl:+8.2f} | {bar}")

    # Summary
    print("\n" + "=" * 70)
    print("  RECOMMENDATIONS")
    print("=" * 70)

    total_wins = sum(1 for t in crypto_trades if t["won"])
    total_pnl = sum(t["pnl"] for t in crypto_trades)
    overall_win_rate = total_wins / len(crypto_trades) * 100

    print(f"\n  Overall: {len(crypto_trades)} trades, {overall_win_rate:.1f}% win rate, ${total_pnl:+.2f} P&L")

    if combos:
        best = combos[0]
        worst = combos[-1]
        print(f"\n  BEST combo:  {best[0]} {best[1]} ({best[3]:.1f}% win rate)")
        print(f"  WORST combo: {worst[0]} {worst[1]} ({worst[3]:.1f}% win rate)")

        if best[3] >= 55:
            print(f"\n  💡 Consider focusing on {best[0]} {best[1]} trades")
        if worst[3] < 45:
            print(f"  ⚠️  Consider avoiding {worst[0]} {worst[1]} trades")


if __name__ == "__main__":
    analyze_trades()
