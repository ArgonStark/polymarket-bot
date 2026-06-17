#!/usr/bin/env python3
"""
Read-only Polymarket auth check.

Verifies the live CLOB connection and L0/L1/L2 authentication WITHOUT placing,
cancelling, or modifying any orders. Safe to run against a funded account.

Exercises:
  - client creation (L1: private key)
  - API credential derivation / use (L2)
  - balance fetch (L2 read)
  - open orders + active positions fetch (L2 read)

Usage:
    python scripts/check_auth.py
"""
import os
import sys

# Ensure project root is importable when run as `python scripts/check_auth.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importing config triggers load_dotenv() so the env file is read automatically.
from src.config import BotConfig
from src.utils.logging import resolve_execution_mode


def _redact(v: str) -> str:
    if not v:
        return "(NOT SET)"
    return f"SET (len={len(v)}, prefix={v[:4]}…)"


def main() -> int:
    cfg = BotConfig.from_env()

    print("=" * 60)
    print("POLYMARKET AUTH CHECK (read-only)")
    print("=" * 60)

    mode = resolve_execution_mode(cfg)
    print(f"Execution mode      : {mode}")
    print(f"CLOB host           : {cfg.endpoints.clob_api_url}")
    print(f"PK (private key)     : {_redact(cfg.wallet.private_key)}")
    print(f"FUNDER (proxy)       : {_redact(cfg.wallet.funder_address)}")
    print(f"Wallet type          : {'proxy (sig_type=1)' if cfg.wallet.is_proxy_wallet else 'EOA (sig_type=0)'}")
    print(f"API creds configured : {cfg.api.is_configured}")
    print("-" * 60)

    if cfg.paper_trading.enabled:
        print("⚠️  PAPER_TRADING_ENABLED=true — this is NOT a live config.")
        print("    Set PAPER_TRADING_ENABLED=false to test live auth.")
        return 1

    if not cfg.wallet.validate():
        print("❌ No private key (PK) configured — cannot authenticate live.")
        return 1

    # Create the live client (L1, derives L2 creds if not provided).
    from src.execution.client import (
        create_trading_client,
        get_account_balance,
        get_open_orders,
        get_active_positions,
    )

    client = create_trading_client(cfg)
    if client is None:
        print("❌ Failed to create trading client (see logs above).")
        return 1

    mode_level = getattr(client, "mode", "?")
    print(f"✅ Client created. Auth level: L{mode_level}")
    if isinstance(mode_level, int) and mode_level < 2:
        print("❌ Not L2 — balance/orders won't work. Check API cred derivation.")
        return 1

    # Read-only L2 calls.
    ok = True
    try:
        bal = get_account_balance(client)
        if bal is None:
            print("❌ Balance fetch returned None.")
            ok = False
        else:
            print(f"✅ Balance (collateral): ${bal:,.2f}")
    except Exception as e:
        print(f"❌ Balance fetch raised: {e}")
        ok = False

    try:
        orders = get_open_orders(client)
        print(f"✅ Open orders fetch OK: {len(orders)} open")
    except Exception as e:
        print(f"❌ Open orders fetch raised: {e}")
        ok = False

    try:
        positions = get_active_positions(client)
        print(f"✅ Active positions fetch OK: {len(positions)} positions")
    except Exception as e:
        print(f"❌ Active positions fetch raised: {e}")
        ok = False

    print("=" * 60)
    print("RESULT:", "✅ AUTH OK — ready for live" if ok else "❌ AUTH PROBLEM — see above")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
