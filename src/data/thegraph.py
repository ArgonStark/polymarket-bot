"""
The Graph Subgraph Integration for Polymarket

Provides real-time on-chain data for:
- Redemptions (settlement payouts)
- Splits (position opens)
- Merges (position combines)
- Conditions (market outcomes)

Usage:
    from src.data.thegraph import PolymarketGraph

    graph = PolymarketGraph(api_key="your_api_key")
    redemptions = graph.get_user_redemptions("0x63ce...")
    splits = graph.get_user_splits("0x63ce...")
"""

import logging
import requests
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Subgraph endpoints
ACTIVITY_SUBGRAPH_ID = "Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp"
GRAPH_GATEWAY = "https://gateway.thegraph.com/api"


@dataclass
class Redemption:
    """Settlement payout event."""
    id: str
    timestamp: int
    redeemer: str
    condition: str
    payout: float  # In USDC

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)


@dataclass
class Split:
    """Position open event (collateral split into outcome tokens)."""
    id: str
    timestamp: int
    stakeholder: str
    condition: str
    amount: float  # In USDC

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)


@dataclass
class Merge:
    """Position close event (outcome tokens merged back to collateral)."""
    id: str
    timestamp: int
    stakeholder: str
    condition: str
    amount: float  # In USDC

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)


@dataclass
class Condition:
    """Market condition with payout info."""
    id: str
    position_ids: List[int]
    payout_numerators: List[int]
    payout_denominator: int


class PolymarketGraph:
    """
    The Graph client for Polymarket on-chain data.

    Provides real-time access to:
    - Redemptions: Actual settlement payouts
    - Splits: When positions are opened
    - Merges: When positions are closed/combined
    - Conditions: Market outcome data
    """

    def __init__(self, api_key: str, subgraph_id: str = ACTIVITY_SUBGRAPH_ID):
        """
        Initialize Graph client.

        Args:
            api_key: The Graph API key (free tier: 100k queries/month)
            subgraph_id: Subgraph ID (default: Polymarket Activity)
        """
        self.api_key = api_key
        self.subgraph_id = subgraph_id
        self.endpoint = f"{GRAPH_GATEWAY}/{api_key}/subgraphs/id/{subgraph_id}"
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def _query(self, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute GraphQL query."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        try:
            response = self._session.post(self.endpoint, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                logger.error(f"GraphQL errors: {data['errors']}")
                return {}

            return data.get("data", {})
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return {}

    def get_user_redemptions(
        self,
        user_address: str,
        first: int = 100,
        skip: int = 0,
        order_by: str = "timestamp",
        order_direction: str = "desc"
    ) -> List[Redemption]:
        """
        Get redemptions (settlement payouts) for a user.

        Args:
            user_address: Wallet address (lowercase)
            first: Max results to return
            skip: Offset for pagination
            order_by: Field to sort by
            order_direction: asc or desc

        Returns:
            List of Redemption objects
        """
        query = """
        query GetRedemptions($user: String!, $first: Int!, $skip: Int!, $orderBy: String!, $orderDirection: String!) {
            redemptions(
                first: $first,
                skip: $skip,
                where: {redeemer: $user},
                orderBy: $orderBy,
                orderDirection: $orderDirection
            ) {
                id
                timestamp
                redeemer
                condition
                payout
            }
        }
        """

        variables = {
            "user": user_address.lower(),
            "first": first,
            "skip": skip,
            "orderBy": order_by,
            "orderDirection": order_direction
        }

        data = self._query(query, variables)
        redemptions = []

        for r in data.get("redemptions", []):
            redemptions.append(Redemption(
                id=r["id"],
                timestamp=int(r["timestamp"]),
                redeemer=r["redeemer"],
                condition=r["condition"],
                payout=int(r["payout"]) / 1e6  # Convert from raw to USDC
            ))

        return redemptions

    def get_user_splits(
        self,
        user_address: str,
        first: int = 100,
        skip: int = 0,
        order_by: str = "timestamp",
        order_direction: str = "desc"
    ) -> List[Split]:
        """
        Get splits (position opens) for a user.

        Args:
            user_address: Wallet address (lowercase)
            first: Max results to return
            skip: Offset for pagination

        Returns:
            List of Split objects
        """
        query = """
        query GetSplits($user: String!, $first: Int!, $skip: Int!, $orderBy: String!, $orderDirection: String!) {
            splits(
                first: $first,
                skip: $skip,
                where: {stakeholder: $user},
                orderBy: $orderBy,
                orderDirection: $orderDirection
            ) {
                id
                timestamp
                stakeholder
                condition
                amount
            }
        }
        """

        variables = {
            "user": user_address.lower(),
            "first": first,
            "skip": skip,
            "orderBy": order_by,
            "orderDirection": order_direction
        }

        data = self._query(query, variables)
        splits = []

        for s in data.get("splits", []):
            splits.append(Split(
                id=s["id"],
                timestamp=int(s["timestamp"]),
                stakeholder=s["stakeholder"],
                condition=s["condition"],
                amount=int(s["amount"]) / 1e6  # Convert from raw to USDC
            ))

        return splits

    def get_user_merges(
        self,
        user_address: str,
        first: int = 100,
        skip: int = 0,
        order_by: str = "timestamp",
        order_direction: str = "desc"
    ) -> List[Merge]:
        """
        Get merges (position closes) for a user.

        Args:
            user_address: Wallet address (lowercase)
            first: Max results to return
            skip: Offset for pagination

        Returns:
            List of Merge objects
        """
        query = """
        query GetMerges($user: String!, $first: Int!, $skip: Int!, $orderBy: String!, $orderDirection: String!) {
            merges(
                first: $first,
                skip: $skip,
                where: {stakeholder: $user},
                orderBy: $orderBy,
                orderDirection: $orderDirection
            ) {
                id
                timestamp
                stakeholder
                condition
                amount
            }
        }
        """

        variables = {
            "user": user_address.lower(),
            "first": first,
            "skip": skip,
            "orderBy": order_by,
            "orderDirection": order_direction
        }

        data = self._query(query, variables)
        merges = []

        for m in data.get("merges", []):
            merges.append(Merge(
                id=m["id"],
                timestamp=int(m["timestamp"]),
                stakeholder=m["stakeholder"],
                condition=m["condition"],
                amount=int(m["amount"]) / 1e6  # Convert from raw to USDC
            ))

        return merges

    def get_condition(self, condition_id: str) -> Optional[Condition]:
        """
        Get condition (market) details including payouts.

        Args:
            condition_id: The condition ID (market identifier)

        Returns:
            Condition object or None
        """
        query = """
        query GetCondition($id: ID!) {
            condition(id: $id) {
                id
                positionIds
                payoutNumerators
                payoutDenominator
            }
        }
        """

        data = self._query(query, {"id": condition_id})
        c = data.get("condition")

        if not c:
            return None

        return Condition(
            id=c["id"],
            position_ids=[int(p) for p in c.get("positionIds", [])],
            payout_numerators=[int(p) for p in c.get("payoutNumerators", [])],
            payout_denominator=int(c.get("payoutDenominator", 1))
        )

    def get_recent_redemptions(
        self,
        first: int = 100,
        min_payout: float = 0
    ) -> List[Redemption]:
        """
        Get recent redemptions across all users.

        Args:
            first: Max results
            min_payout: Minimum payout in USDC to filter

        Returns:
            List of recent Redemption objects
        """
        # Convert USDC to raw (6 decimals)
        min_payout_raw = int(min_payout * 1e6)

        query = """
        query GetRecentRedemptions($first: Int!, $minPayout: BigInt!) {
            redemptions(
                first: $first,
                orderBy: timestamp,
                orderDirection: desc,
                where: {payout_gte: $minPayout}
            ) {
                id
                timestamp
                redeemer
                condition
                payout
            }
        }
        """

        data = self._query(query, {"first": first, "minPayout": str(min_payout_raw)})
        redemptions = []

        for r in data.get("redemptions", []):
            redemptions.append(Redemption(
                id=r["id"],
                timestamp=int(r["timestamp"]),
                redeemer=r["redeemer"],
                condition=r["condition"],
                payout=int(r["payout"]) / 1e6
            ))

        return redemptions

    def calculate_user_stats(self, user_address: str) -> Dict[str, Any]:
        """
        Calculate user statistics from on-chain data.

        Args:
            user_address: Wallet address

        Returns:
            Dict with stats: total_splits, total_merges, total_redemptions, etc.
        """
        # Get all user activity
        splits = self.get_user_splits(user_address, first=1000)
        merges = self.get_user_merges(user_address, first=1000)
        redemptions = self.get_user_redemptions(user_address, first=1000)

        total_split_amount = sum(s.amount for s in splits)
        total_merge_amount = sum(m.amount for m in merges)
        total_redemption_payout = sum(r.payout for r in redemptions)

        # Non-zero redemptions are actual settlement wins
        winning_redemptions = [r for r in redemptions if r.payout > 0]
        losing_redemptions = [r for r in redemptions if r.payout == 0]

        return {
            "user": user_address,
            "total_splits": len(splits),
            "total_merges": len(merges),
            "total_redemptions": len(redemptions),
            "total_split_amount_usdc": total_split_amount,
            "total_merge_amount_usdc": total_merge_amount,
            "total_redemption_payout_usdc": total_redemption_payout,
            "winning_redemptions": len(winning_redemptions),
            "losing_redemptions": len(losing_redemptions),
            "redemption_win_rate": len(winning_redemptions) / len(redemptions) if redemptions else 0,
        }

    def is_healthy(self) -> bool:
        """Check if the subgraph is accessible."""
        query = "{ _meta { block { number } } }"
        data = self._query(query)
        return "_meta" in data


# Convenience function
def get_graph_client(api_key: Optional[str] = None) -> Optional[PolymarketGraph]:
    """
    Get a configured Graph client.

    Args:
        api_key: Graph API key. If None, checks GRAPH_API_KEY env var.

    Returns:
        PolymarketGraph client or None if no API key
    """
    import os

    key = api_key or os.environ.get("GRAPH_API_KEY")
    if not key:
        logger.warning("No Graph API key provided. Set GRAPH_API_KEY env var.")
        return None

    return PolymarketGraph(api_key=key)
