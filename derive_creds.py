#!/usr/bin/env python3
"""
Derive Polymarket API credentials from your private key.
Run this script and copy the output to your .env file.

Supports both EOA wallets and proxy wallets (Magic/browser wallets).
"""

import os
from dotenv import load_dotenv
from py_clob_client.client import ClobClient

load_dotenv()


def main():
    private_key = os.getenv("PK")
    funder = os.getenv("FUNDER", "").strip()

    if not private_key:
        print("ERROR: PK not found in .env file")
        print("Make sure your .env file contains: PK=0x...")
        return

    # Ensure private key has 0x prefix
    if not private_key.startswith("0x"):
        private_key = "0x" + private_key

    # Determine wallet type
    is_proxy_wallet = bool(funder)
    wallet_type = "Proxy Wallet" if is_proxy_wallet else "EOA Wallet"
    signature_type = 1 if is_proxy_wallet else 0

    print("=" * 60)
    print("  POLYMARKET API CREDENTIAL DERIVATION")
    print("=" * 60)
    print(f"  Wallet Type: {wallet_type}")
    print(f"  Signature Type: {signature_type}")
    if is_proxy_wallet:
        print(f"  Funder Address: {funder}")
    print("=" * 60)
    print()
    print("Deriving API credentials...")

    try:
        # Build client kwargs
        client_kwargs = {
            "host": "https://clob.polymarket.com",
            "key": private_key,
            "chain_id": 137,  # Polygon
            "signature_type": signature_type,
        }

        # Add funder for proxy wallet
        if is_proxy_wallet:
            client_kwargs["funder"] = funder

        # Create client
        client = ClobClient(**client_kwargs)

        # Derive or create API credentials
        creds = client.create_or_derive_api_creds()

        print("\n" + "=" * 60)
        print("  SUCCESS! Add these to your .env file:")
        print("=" * 60)
        print()
        print(f"CLOB_API_KEY={creds.api_key}")
        print(f"CLOB_SECRET={creds.api_secret}")
        print(f"CLOB_PASS_PHRASE={creds.api_passphrase}")
        print()
        print("=" * 60)
        print("Note: These credentials are derived from your private key")
        print("and wallet type. They will be the same each time.")
        print("=" * 60)

        # Also try to fetch balance to verify everything works
        print()
        print("Verifying credentials by fetching balance...")
        try:
            from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams, AssetType

            # Set credentials and try to get balance
            client.set_api_creds(creds)

            params = BalanceAllowanceParams(
                asset_type=AssetType.COLLATERAL,
                signature_type=signature_type,
            )
            balance_info = client.get_balance_allowance(params)

            if balance_info:
                balance = balance_info.get("balance", 0)
                if isinstance(balance, str):
                    balance = float(balance)
                if balance > 1_000_000:
                    balance = balance / 1_000_000
                print(f"  Balance: ${balance:,.2f} USDC")
                print("  Credentials verified successfully!")
            else:
                print("  Warning: Could not fetch balance (response was empty)")
                print("  The credentials may still be valid.")

        except Exception as e:
            print(f"  Warning: Could not verify balance: {e}")
            print("  The credentials may still be valid for trading.")

    except Exception as e:
        print(f"\nERROR: Failed to derive credentials: {e}")
        print()
        print("Troubleshooting tips:")
        print("1. Make sure PK contains your private key (with 0x prefix)")
        print("2. For proxy wallets (Magic/browser), set FUNDER to your deposit address")
        print("   Find it at: https://polymarket.com/wallet")
        print("3. Make sure you have internet connectivity")
        print("4. The py-clob-client package must be installed")

        import traceback
        print()
        print("Full error:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
