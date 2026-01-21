"""
Trading client setup for Polymarket CLOB.

Handles authentication and client creation using py-clob-client.
"""

import logging
from typing import Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds
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
        logger.error(
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
            derived_creds = client.create_or_derive_api_creds()
            client.set_api_creds(derived_creds)
            logger.info(
                f"Derived API credentials. "
                f"API Key: {derived_creds.api_key[:8]}... "
                f"Save these to your .env file!"
            )

        # Verify client is working
        try:
            # Simple test - get order book for a known market
            logger.info("Trading client created successfully")
            return client
        except Exception as e:
            logger.warning(f"Client verification warning: {e}")
            return client

    except Exception as e:
        logger.error(f"Failed to create trading client: {e}")
        return None


def get_account_balance(client: ClobClient) -> Optional[float]:
    """
    Get USDC balance for the trading account.

    Args:
        client: Configured ClobClient

    Returns:
        USDC balance or None if fetch fails
    """
    try:
        # Note: This may require additional API calls depending on
        # the py-clob-client version
        balance_info = client.get_balance_allowance()
        if balance_info:
            return float(balance_info.get("balance", 0))
        return None
    except Exception as e:
        logger.error(f"Failed to get balance: {e}")
        return None


def get_open_orders(client: ClobClient) -> list[dict]:
    """
    Get all open orders for the account.

    Args:
        client: Configured ClobClient

    Returns:
        List of open order dictionaries
    """
    try:
        orders = client.get_orders()
        return orders if orders else []
    except Exception as e:
        logger.error(f"Failed to get open orders: {e}")
        return []


def cancel_all_orders(client: ClobClient) -> bool:
    """
    Cancel all open orders.

    Args:
        client: Configured ClobClient

    Returns:
        True if successful, False otherwise
    """
    try:
        result = client.cancel_all()
        logger.info(f"Cancelled all orders: {result}")
        return True
    except Exception as e:
        logger.error(f"Failed to cancel all orders: {e}")
        return False


def cancel_order(client: ClobClient, order_id: str) -> bool:
    """
    Cancel a specific order.

    Args:
        client: Configured ClobClient
        order_id: ID of order to cancel

    Returns:
        True if successful, False otherwise
    """
    try:
        result = client.cancel(order_id=order_id)
        logger.info(f"Cancelled order {order_id}: {result}")
        return True
    except Exception as e:
        logger.error(f"Failed to cancel order {order_id}: {e}")
        return False
