#!/usr/bin/env python3
"""Debug script to check why positions aren't being detected."""

import os
import sys
import time
import requests
from dotenv import load_dotenv

load_dotenv()

def main():
    # Get wallet from private key
    pk = os.getenv('PK')
    if not pk:
        print("ERROR: No PK found in .env file")
        return

    from eth_account import Account
    acct = Account.from_key(pk)
    wallet = acct.address
    print(f"Wallet: {wallet}")
    print()

    # Fetch positions from Polymarket Data API
    url = f"https://data-api.polymarket.com/positions?user={wallet}"
    print(f"Fetching: {url}")

    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        print(f"ERROR: API returned {response.status_code}")
        return

    positions = response.json()
    print(f"\n{'='*70}")
    print(f"FOUND {len(positions)} TOTAL POSITIONS FROM API")
    print(f"{'='*70}\n")

    if not positions:
        print("No positions found on Polymarket!")
        return

    current_time = int(time.time())

    # Analyze each position
    detected_15m = []
    for i, p in enumerate(positions):
        slug = p.get('slug', '') or p.get('eventSlug', '') or ''
        title = p.get('title', '') or ''
        size = float(p.get('size', 0))
        value = float(p.get('currentValue', 0))
        cond_id = p.get('conditionId', 'N/A')

        if size <= 0:
            continue

        print(f"[{i+1}] {title[:60]}")
        print(f"    Slug: {slug}")
        print(f"    Size: {size:.2f} shares | Value: ${value:.2f}")
        print(f"    ConditionID: {cond_id}")

        # Check 1: Is this a 15-min market?
        is_15m = 'updown-15m' in slug.lower() or '15m' in title.lower()
        print(f"    Is 15-min market: {'✅ YES' if is_15m else '❌ NO'}")

        if not is_15m:
            print(f"    ⚠️  FILTERED: Not a 15-min crypto market")
            print()
            continue

        # Check 2: Can we parse the timestamp?
        market_ts = None
        try:
            parts = slug.split('-')
            if len(parts) >= 4:
                market_ts = int(parts[-1])
        except (ValueError, IndexError):
            pass

        if market_ts is None:
            print(f"    ⚠️  FILTERED: Cannot parse timestamp from slug")
            print()
            continue

        print(f"    Market timestamp: {market_ts}")

        # Check 3: Is the market settled?
        settle_time = market_ts + 900
        if current_time > settle_time:
            elapsed = current_time - settle_time
            print(f"    ⚠️  FILTERED: Market settled {elapsed}s ago")
            print()
            continue

        remaining = settle_time - current_time
        print(f"    Time remaining: {remaining}s (settles in {remaining//60}m {remaining%60}s)")

        # Check 4: Is this a supported asset?
        asset = None
        for pattern in ['btc', 'eth', 'sol', 'xrp']:
            if pattern in slug.lower():
                asset = pattern.upper()
                break

        if not asset:
            print(f"    ⚠️  FILTERED: Unknown asset (not BTC/ETH/SOL/XRP)")
            print()
            continue

        print(f"    Asset: {asset}")
        print(f"    ✅ SHOULD BE TRACKED!")
        detected_15m.append({
            'asset': asset,
            'size': size,
            'value': value,
            'slug': slug,
            'remaining': remaining,
        })
        print()

    print(f"{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total positions on Polymarket: {len(positions)}")
    print(f"Valid 15-min crypto positions: {len(detected_15m)}")

    if detected_15m:
        print(f"\nPositions that SHOULD be tracked:")
        for p in detected_15m:
            print(f"  - {p['asset']}: {p['size']:.2f} shares (${p['value']:.2f}) - {p['remaining']}s remaining")
    else:
        print(f"\nNo valid 15-min crypto positions found.")
        print("Possible reasons:")
        print("  1. Your positions have already settled (15 minutes elapsed)")
        print("  2. Your positions are not in 15-minute crypto markets")
        print("  3. The market slug format has changed")

if __name__ == "__main__":
    main()
