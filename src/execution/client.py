"""
Trading client setup for Polymarket CLOB.

Handles authentication and client creation using py-clob-client.
"""

import logging
from typing import Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams
from py_clob_client.constants import POLYGON

from ..config import BotConfig


logger = logging.getLogger(__name__)


def create_trading_client(config: BotConfig) -> Optional[ClobClient]:
    """
    Create and configure a trading client for Polymarket CLOB.

    Handles authentication using either:
    1. Existing API credentials from environment
    2. Deriving new credentials from private key

    Args:
        config: Bot configuration with wallet and API settings

    Returns:
        Configured ClobClient or None if setup fails
    """
    # Validate wallet configuration
    if not config.wallet.validate():
        logger.warning(
            "Wallet not configured. Set PK and FUNDER environment variables."
        )
        return None

    try:
        # Create base client
        client = ClobClient(
            host=config.endpoints.clob_api_url,
            key=config.wallet.private_key,
            chain_id=POLYGON,
            funder=config.wallet.funder_address,
            signature_type=1,  # Magic/email wallet signature type
        )

        # Set up API credentials
        if config.api.is_configured:
            # Use existing credentials
            creds = ApiCreds(
                api_key=config.api.api_key,
                api_secret=config.api.api_secret,
                api_passphrase=config.api.api_passphrase,
            )
            client.set_api_creds(creds)
            logger.info("Using existing API credentials")
        else:
            # Derive new credentials from private key
            logger.info("Deriving API credentials from private key...")
            try:
                derived_creds = client.create_or_derive_api_creds()
                client.set_api_creds(derived_creds)
                logger.info(
                    f"Derived API credentials. "
                    f"API Key: {derived_creds.api_key[:8]}... "
                    f"Save these to your .env file!"
                )
            except Exception as e:
                logger.warning(f"Could not derive API credentials: {e}")
                # Client can still work for some operations without L2 auth

        logger.info("Trading client created successfully")
        return client

    except Exception as e:
        logger.error(f"Failed to create trading client: {e}")
        return None


def get_account_balance(client: Optional[ClobClient]) -> Optional[float]:
    """
    Get USDC balance for the trading account.

    Args:
        client: Configured ClobClient (can be None)

    Returns:
        USDC balance or None if fetch fails
    """
    if client is None:
        return None

    try:
        # Use SDK method with proper params
        balance_info = client.get_balance_allowance()
        if balance_info:
            # The SDK returns balance in wei for USDC (6 decimals)
            balance = balance_info.get("balance", 0)
            if isinstance(balance, str):
                balance = float(balance)
            # Convert from micro-units if needed (USDC has 6 decimals)
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

    Args:
        client: Configured ClobClient (can be None)

    Returns:
        List of open order dictionaries
    """
    if client is None:
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

    Args:
        client: Configured ClobClient (can be None)
        limit: Maximum number of trades to return

    Returns:
        List of trade dictionaries
    """
    if client is None:
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

    Args:
        client: Configured ClobClient (can be None)

    Returns:
        True if successful, False otherwise
    """
    if client is None:
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

    Args:
        client: Configured ClobClient (can be None)
        order_id: ID of order to cancel

    Returns:
        True if successful, False otherwise
    """
    if client is None:
        return False

    try:
        result = client.cancel(order_id=order_id)
        logger.info(f"Cancelled order {order_id}: {result}")
        return True
    except Exception as e:
        logger.error(f"Failed to cancel order {order_id}: {e}")
        return False
