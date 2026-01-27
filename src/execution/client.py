"""
Trading client setup for Polymarket CLOB.

Handles authentication and client creation using py-clob-client.
"""

import logging
from typing import Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams, AssetType
from py_clob_client.constants import POLYGON

from ..config import BotConfig


logger = logging.getLogger(__name__)


def create_trading_client(config: BotConfig) -> Optional[ClobClient]:
    """
    Create and configure a trading client for Polymarket CLOB.

    Supports two wallet types:
    - EOA wallet (signature_type=0): Standard externally owned account
    - Proxy wallet (signature_type=1): Magic/browser wallet with funder address

    Authentication levels:
    - L0: No auth (public endpoints only)
    - L1: Private key (can derive API keys)
    - L2: Private key + API creds (full access)

    Args:
        config: Bot configuration with wallet and API settings

    Returns:
        Configured ClobClient or None if setup fails
    """
    # Validate wallet configuration
    if not config.wallet.validate():
        logger.warning(
            "Wallet not configured. Set PK environment variable."
        )
        return None

    try:
        # Determine wallet type
        signature_type = config.wallet.signature_type
        wallet_mode = "proxy wallet" if config.wallet.is_proxy_wallet else "EOA wallet"
        logger.info(f"Creating client with {wallet_mode} (signature_type={signature_type})")

        # Build client kwargs based on wallet type
        client_kwargs = {
            "host": config.endpoints.clob_api_url,
            "key": config.wallet.private_key,
            "chain_id": POLYGON,
            "signature_type": signature_type,
        }

        # Add funder for proxy wallet mode
        if config.wallet.is_proxy_wallet:
            client_kwargs["funder"] = config.wallet.funder_address

        # Add credentials if available (enables L2 mode)
        if config.api.is_configured:
            client_kwargs["creds"] = ApiCreds(
                api_key=config.api.api_key,
                api_secret=config.api.api_secret,
                api_passphrase=config.api.api_passphrase,
            )
            logger.info("Using existing API credentials (L2 mode)")

        # Create client
        client = ClobClient(**client_kwargs)

        # If no creds provided, try to derive them
        if not config.api.is_configured:
            logger.info("No API credentials configured, attempting to derive...")
            try:
                derived_creds = client.create_or_derive_api_creds()
                client.set_api_creds(derived_creds)
                logger.info(
                    f"Derived API credentials successfully. "
                    f"API Key: {derived_creds.api_key[:8]}..."
                )
                # Log the credentials so user can save them
                logger.info(
                    "Save these credentials to your .env file:\n"
                    f"  CLOB_API_KEY={derived_creds.api_key}\n"
                    f"  CLOB_SECRET={derived_creds.api_secret}\n"
                    f"  CLOB_PASS_PHRASE={derived_creds.api_passphrase}"
                )
            except Exception as e:
                logger.warning(f"Could not derive API credentials: {e}")
                logger.warning("L2 operations (balance, orders) will not work")

        logger.info(f"Trading client created successfully (mode: L{client.mode})")
        return client

    except Exception as e:
        logger.error(f"Failed to create trading client: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def get_account_balance(client: Optional[ClobClient]) -> Optional[float]:
    """
    Get USDC balance for the trading account.

    Requires L2 authentication.

    Args:
        client: Configured ClobClient (can be None)

    Returns:
        USDC balance or None if fetch fails
    """
    if client is None:
        return None

    # Check if client has L2 auth
    if client.mode < 2:  # L2 = 2
        logger.warning("Client not in L2 mode, cannot fetch balance")
        return None

    try:
        # Use the client's configured signature type
        # 0 = EOA wallet, 1 = proxy/browser wallet
        sig_type = getattr(client, 'signature_type', 0)
        # AssetType.COLLATERAL = USDC collateral balance
        params = BalanceAllowanceParams(
            asset_type=AssetType.COLLATERAL,
            signature_type=sig_type,
        )
        balance_info = client.get_balance_allowance(params)

        if balance_info:
            # Balance is typically returned as string in micro-units
            balance = balance_info.get("balance", 0)
            if isinstance(balance, str):
                balance = float(balance)
            # USDC has 6 decimals, but check if already in correct units
            if balance > 1_000_000:
                balance = balance / 1_000_000
            return float(balance)
        return None
    except Exception as e:
        logger.error(f"Failed to get balance: {e}")
        return None


def get_open_orders(client: Optional[ClobClient]) -> list[dict]:
    """
    Get all open orders for the account.

    Requires L2 authentication.

    Args:
        client: Configured ClobClient (can be None)

    Returns:
        List of open order dictionaries
    """
    if client is None:
        return []

    # Check if client has L2 auth
    if client.mode < 2:
        logger.warning("Client not in L2 mode, cannot fetch orders")
        return []

    try:
        orders = client.get_orders()
        return orders if orders else []
    except Exception as e:
        logger.error(f"Failed to get open orders: {e}")
        return []


def get_trades(client: Optional[ClobClient], limit: int = 100) -> list[dict]:
    """
    Get recent trades for the account.

    Requires L2 authentication.

    Args:
        client: Configured ClobClient (can be None)
        limit: Maximum number of trades to return

    Returns:
        List of trade dictionaries
    """
    if client is None:
        return []

    # Check if client has L2 auth
    if client.mode < 2:
        logger.warning("Client not in L2 mode, cannot fetch trades")
        return []

    try:
        trades = client.get_trades()
        return trades[:limit] if trades else []
    except Exception as e:
        logger.error(f"Failed to get trades: {e}")
        return []


def cancel_all_orders(client: Optional[ClobClient]) -> bool:
    """
    Cancel all open orders.

    Requires L2 authentication.

    Args:
        client: Configured ClobClient (can be None)

    Returns:
        True if successful, False otherwise
    """
    if client is None:
        return False

    if client.mode < 2:
        logger.warning("Client not in L2 mode, cannot cancel orders")
        return False

    try:
        result = client.cancel_all()
        logger.info(f"Cancelled all orders: {result}")
        return True
    except Exception as e:
        logger.error(f"Failed to cancel all orders: {e}")
        return False


def cancel_order(client: Optional[ClobClient], order_id: str) -> bool:
    """
    Cancel a specific order.

    Requires L2 authentication.

    Args:
        client: Configured ClobClient (can be None)
        order_id: ID of order to cancel

    Returns:
        True if successful, False otherwise
    """
    if client is None:
        return False

    if client.mode < 2:
        logger.warning("Client not in L2 mode, cannot cancel order")
        return False

    try:
        result = client.cancel(order_id=order_id)
        logger.info(f"Cancelled order {order_id}: {result}")
        return True
    except Exception as e:
        logger.error(f"Failed to cancel order {order_id}: {e}")
        return False
