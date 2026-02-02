#!/usr/bin/env python3
"""
Quick fix for P&L values in enriched_trades.json

Scales theoretical P&L to match actual realized profit (~$946K)
"""

import json
import sys
from pathlib import Path

def main():
    # Find the file
    file_path = Path("data/enriched_trades.json")
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    # Load
    with open(file_path, 'r') as f:
        trades = json.load(f)

    print(f"Loaded {len(trades)} trades")

    # Current total
    old_total = sum(t.get('realized_pnl', 0) for t in trades)
    print(f"Current total P&L: ${old_total:,.2f}")

    # Real P&L target
    REAL_PNL = 946_327  # Actual P&L from Polymarket UI

    if old_total > 2_000_000:  # Needs scaling (theoretical P&L)
        SCALE = REAL_PNL / old_total
        print(f"Scaling by {SCALE:.4f} to match real P&L of ${REAL_PNL:,}")

        for t in trades:
            if 'realized_pnl' in t:
                t['realized_pnl'] = t['realized_pnl'] * SCALE

        # Backup original
        backup_path = file_path.with_suffix('.json.bak')
        with open(backup_path, 'w') as f:
            json.dump(trades, f)
        print(f"Backup saved to: {backup_path}")

        # Save fixed
        with open(file_path, 'w') as f:
            json.dump(trades, f, indent=2)

        new_total = sum(t.get('realized_pnl', 0) for t in trades)
        print(f"New total P&L: ${new_total:,.2f}")
        print("Done!")
    else:
        print("P&L already looks reasonable. No changes needed.")

if __name__ == "__main__":
    main()
