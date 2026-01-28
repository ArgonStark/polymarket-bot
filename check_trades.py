#!/usr/bin/env python3
"""Check trade history and performance statistics."""

import json
import os
from datetime import datetime

HISTORY_PATH = "trade_history.json"


def main():
    if not os.path.exists(HISTORY_PATH):
        print("No trade history found yet.")
        print("Trade history is created after your first trade.")
        return

    with open(HISTORY_PATH) as f:
        data = json.load(f)

    trades = data.get("trades", [])
    last_updated = data.get("last_updated", "unknown")

    print("=" * 60)
    print("TRADE HISTORY")
    print("=" * 60)
    print(f"Total trades:    {len(trades)}")
    print(f"Last updated:    {last_updated}")
    print()

    # Calculate stats
    completed = [t for t in trades if t.get("actual_outcome") is not None]
    open_trades = [t for t in trades if t.get("actual_outcome") is None]

    wins = sum(1 for t in completed if t["actual_outcome"] == "WIN")
    losses = len(completed) - wins
    total_pnl = sum(t.get("pnl", 0) or 0 for t in completed)

    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Completed trades: {len(completed)}")
    print(f"Open trades:      {len(open_trades)}")
    print(f"Wins:             {wins}")
    print(f"Losses:           {losses}")
    if completed:
        print(f"Win rate:         {wins / len(completed):.1%}")
    print(f"Total P&L:        ${total_pnl:+.2f}")
    if completed:
        print(f"Avg P&L/trade:    ${total_pnl / len(completed):+.2f}")
    print()

    # Stats by asset
    print("=" * 60)
    print("PERFORMANCE BY ASSET")
    print("=" * 60)
    for asset in ["BTC", "ETH", "SOL", "XRP"]:
        asset_trades = [t for t in completed if t["asset"] == asset]
        if not asset_trades:
            continue
        asset_wins = sum(1 for t in asset_trades if t["actual_outcome"] == "WIN")
        asset_pnl = sum(t.get("pnl", 0) or 0 for t in asset_trades)
        win_rate = asset_wins / len(asset_trades) if asset_trades else 0
        print(f"  {asset:4} │ {len(asset_trades):3} trades │ {win_rate:5.1%} win │ ${asset_pnl:+8.2f}")
    print()

    # Stats by side
    print("=" * 60)
    print("PERFORMANCE BY SIDE")
    print("=" * 60)
    for side in ["UP", "DOWN"]:
        side_trades = [t for t in completed if t["side"] == side]
        if not side_trades:
            continue
        side_wins = sum(1 for t in side_trades if t["actual_outcome"] == "WIN")
        side_pnl = sum(t.get("pnl", 0) or 0 for t in side_trades)
        win_rate = side_wins / len(side_trades) if side_trades else 0
        emoji = "▲" if side == "UP" else "▼"
        print(f"  {emoji} {side:5} │ {len(side_trades):3} trades │ {win_rate:5.1%} win │ ${side_pnl:+8.2f}")
    print()

    # Prediction accuracy
    print("=" * 60)
    print("ML PREDICTION ACCURACY")
    print("=" * 60)
    predictions_correct = 0
    predictions_made = 0
    for t in completed:
        if t.get("predicted_prob") is not None:
            predictions_made += 1
            predicted_win = t["predicted_prob"] >= 0.5
            actual_win = t["actual_outcome"] == "WIN"
            if predicted_win == actual_win:
                predictions_correct += 1

    if predictions_made > 0:
        accuracy = predictions_correct / predictions_made
        print(f"Predictions made:    {predictions_made}")
        print(f"Correct:             {predictions_correct}")
        print(f"Accuracy:            {accuracy:.1%}")
    else:
        print("No ML predictions recorded yet")
    print()

    # Arbitrage stats
    print("=" * 60)
    print("ARBITRAGE TYPE PERFORMANCE")
    print("=" * 60)
    arb_types = {}
    for t in completed:
        arb = t.get("arb_type", "none")
        if arb not in arb_types:
            arb_types[arb] = {"count": 0, "wins": 0, "pnl": 0}
        arb_types[arb]["count"] += 1
        if t["actual_outcome"] == "WIN":
            arb_types[arb]["wins"] += 1
        arb_types[arb]["pnl"] += t.get("pnl", 0) or 0

    for arb, stats in arb_types.items():
        win_rate = stats["wins"] / stats["count"] if stats["count"] > 0 else 0
        print(f"  {arb:10} │ {stats['count']:3} trades │ {win_rate:5.1%} win │ ${stats['pnl']:+8.2f}")
    print()

    # Recent trades
    print("=" * 60)
    print("RECENT TRADES (last 10)")
    print("=" * 60)
    recent = trades[-10:] if trades else []
    for t in reversed(recent):
        timestamp = t.get("timestamp", "")[:16]  # Truncate to YYYY-MM-DDTHH:MM
        asset = t.get("asset", "???")
        side = t.get("side", "?")
        outcome = t.get("actual_outcome", "OPEN")
        pnl = t.get("pnl")
        entry = t.get("entry_price", 0)

        outcome_str = outcome if outcome else "OPEN"
        pnl_str = f"${pnl:+.2f}" if pnl is not None else "---"

        emoji = "▲" if side == "UP" else "▼"
        result_emoji = "✓" if outcome == "WIN" else "✗" if outcome == "LOSS" else "⏳"

        print(f"  {timestamp} │ {asset:4} {emoji} │ @ {entry:.2f} │ {result_emoji} {outcome_str:4} │ {pnl_str}")
    print()


if __name__ == "__main__":
    main()
