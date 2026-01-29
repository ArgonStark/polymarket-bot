"""
Polymarket Data API client.

Provides access to user positions, activity, and closed positions.
API Documentation: https://docs.polymarket.com/api-reference/core
"""

import logging
from typing import Optional
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://data-api.polymarket.com"


class PolymarketDataAPI:
    """Client for Polymarket Data API."""

    def __init__(self, timeout: int = 30):
        self.base_url = BASE_URL
        self.timeout = timeout
        self.session = requests.Session()

    def get_positions(
        self,
        user: str,
        limit: int = 100,
        offset: int = 0,
        size_threshold: float = 0,
    ) -> list:
        """
        Get current/open positions for a user.

        Args:
            user: Wallet address (0x...)
            limit: Max results (default 100, max 500)
            offset: Pagination offset
            size_threshold: Minimum position size

        Returns:
            List of position objects with fields:
            - conditionId, asset, size, avgPrice, curPrice
            - currentValue, cashPnl, percentPnl, realizedPnl
            - title, slug, outcome, outcomeIndex
        """
        url = f"{self.base_url}/positions"
        params = {
            "user": user,
            "limit": min(limit, 500),
            "offset": offset,
            "sizeThreshold": size_threshold,
        }

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    def get_activity(
        self,
        user: str,
        limit: int = 100,
        offset: int = 0,
        activity_type: Optional[str] = None,
        side: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> list:
        """
        Get user activity (trades, splits, merges, redeems).

        Args:
            user: Wallet address (0x...)
            limit: Max results (default 100, max 500)
            offset: Pagination offset
            activity_type: Filter by type (TRADE, SPLIT, MERGE, REDEEM, etc.)
            side: Filter by side (BUY or SELL)
            start: Unix timestamp minimum
            end: Unix timestamp maximum

        Returns:
            List of activity objects with fields:
            - timestamp, conditionId, type, size, usdcSize, price
            - side, outcome, outcomeIndex, title, slug, transactionHash
        """
        url = f"{self.base_url}/activity"
        params = {
            "user": user,
            "limit": min(limit, 500),
            "offset": offset,
            "sortBy": "TIMESTAMP",
            "sortDirection": "DESC",
        }

        if activity_type:
            params["type"] = activity_type
        if side:
            params["side"] = side
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch activity: {e}")
            return []

    def get_closed_positions(
        self,
        user: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list:
        """
        Get closed positions for a user (completed trades with P&L).

        Args:
            user: Wallet address (0x...)
            limit: Max results (default 50, max 50)
            offset: Pagination offset

        Returns:
            List of closed position objects with fields:
            - conditionId, asset, avgPrice, totalBought, realizedPnl
            - title, slug, outcome, outcomeIndex, timestamp
        """
        url = f"{self.base_url}/closed-positions"
        params = {
            "user": user,
            "limit": min(limit, 50),
            "offset": offset,
            "sortBy": "TIMESTAMP",
            "sortDirection": "DESC",
        }

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch closed positions: {e}")
            return []

    def get_all_closed_positions(self, user: str, max_positions: int = 500) -> list:
        """
        Fetch all closed positions with pagination.

        Args:
            user: Wallet address
            max_positions: Maximum total positions to fetch

        Returns:
            List of all closed positions
        """
        all_positions = []
        offset = 0
        batch_size = 50  # API max

        while len(all_positions) < max_positions:
            batch = self.get_closed_positions(user, limit=batch_size, offset=offset)
            if not batch:
                break

            all_positions.extend(batch)
            offset += len(batch)

            if len(batch) < batch_size:
                break

        return all_positions[:max_positions]

    def get_all_activity(
        self,
        user: str,
        activity_type: Optional[str] = "TRADE",
        max_items: int = 500,
    ) -> list:
        """
        Fetch all activity with pagination.

        Args:
            user: Wallet address
            activity_type: Filter type (default: TRADE)
            max_items: Maximum total items to fetch

        Returns:
            List of all activity items
        """
        all_activity = []
        offset = 0
        batch_size = 500  # API max

        while len(all_activity) < max_items:
            batch = self.get_activity(
                user, limit=batch_size, offset=offset, activity_type=activity_type
            )
            if not batch:
                break

            all_activity.extend(batch)
            offset += len(batch)

            if len(batch) < batch_size:
                break

        return all_activity[:max_items]


# Singleton instance
_data_api: Optional[PolymarketDataAPI] = None


def get_data_api() -> PolymarketDataAPI:
    """Get or create Polymarket Data API client singleton."""
    global _data_api
    if _data_api is None:
        _data_api = PolymarketDataAPI()
    return _data_api
