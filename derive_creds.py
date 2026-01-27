#!/usr/bin/env python3
"""
Derive Polymarket API credentials from your private key.
Run this script and copy the output to your .env file.
"""

import os
from dotenv import load_dotenv
from py_clob_client.client import ClobClient

load_dotenv()

def main():
    private_key = os.getenv("PK")

    if not private_key:
        print("ERROR: PK not found in .env file")
        print("Make sure your .env file contains: PK=0x...")
        return

    # Ensure private key has 0x prefix
    if not private_key.startswith("0x"):
        private_key = "0x" + private_key

    print("Deriving API credentials from your private key...")
    print("=" * 60)

    try:
        # Create client (Polygon mainnet)
        client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=137,  # Polygon
        )

        # Derive or create API credentials
        creds = client.create_or_derive_api_creds()

        print("\nSuccess! Add these to your .env file:\n")
        print(f"POLY_API_KEY={creds.api_key}")
        print(f"POLY_API_SECRET={creds.api_secret}")
        print(f"POLY_PASSPHRASE={creds.api_passphrase}")
        print("\n" + "=" * 60)
        print("Note: These credentials are derived from your private key.")
        print("They will be the same each time you run this script.")

    except Exception as e:
        print(f"ERROR: Failed to derive credentials: {e}")
        print("\nMake sure:")
        print("1. Your PRIVATE_KEY is correct")
        print("2. You have internet connectivity")
        print("3. The py-clob-client package is installed")

if __name__ == "__main__":
    main()
