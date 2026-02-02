"""
Settlement Verifier - On-chain verification using The Graph

Provides on-chain verification of:
- Market settlements via condition payouts
- Position redemptions (actual P&L)
- Trade confirmations via splits/merges

This adds a layer of trust by verifying API responses against on-chain data.
"""

import os
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

from .thegraph import PolymarketGraph, get_graph_client, Condition, Redemption

logger = logging.getLogger(__name__)


@dataclass
class SettlementVerification:
    """Result of verifying a settlement."""
    condition_id: str
    verified: bool
    on_chain_resolved: bool
    payout_numerators: List[int]
    payout_denominator: int
    winning_outcome: Optional[str]  # "UP", "DOWN", or None
    redemption_amount: float  # USDC redeemed (0 if none)
    message: str


@dataclass
class PositionVerification:
    """Result of verifying a position."""
    user_address: str
    total_splits: int  # Position opens
    total_merges: int  # Position closes
    total_redemptions: int  # Settlements
    total_redeemed_usdc: float
    recent_activity: List[Dict]


class SettlementVerifier:
    """
    Verifies settlements and positions using The Graph.

    Uses the Polymarket Activity subgraph to:
    1. Check if markets are resolved on-chain
    2. Verify redemption amounts match expectations
    3. Track actual P&L from on-chain data
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the settlement verifier.

        Args:
            api_key: The Graph API key (or uses GRAPH_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("GRAPH_API_KEY")
        self._graph: Optional[PolymarketGraph] = None
        self._enabled = bool(self.api_key)

        if self._enabled:
            logger.info("Settlement verifier enabled (The Graph)")
        else:
            logger.debug("Settlement verifier disabled (no GRAPH_API_KEY)")

    @property
    def graph(self) -> Optional[PolymarketGraph]:
        """Get or create Graph client."""
        if not self._enabled:
            return None
        if self._graph is None:
            self._graph = get_graph_client(self.api_key)
        return self._graph

    def is_enabled(self) -> bool:
        """Check if verifier is enabled."""
        return self._enabled

    def verify_settlement(self, condition_id: str) -> Optional[SettlementVerification]:
        """
        Verify a market settlement on-chain.

        Args:
            condition_id: The market condition ID

        Returns:
            SettlementVerification with on-chain data, or None if disabled/error
        """
        if not self._enabled or not self.graph:
            return None

        try:
            # Get condition data from The Graph
            condition = self.graph.get_condition(condition_id)

            if not condition:
                return SettlementVerification(
                    condition_id=condition_id,
                    verified=False,
                    on_chain_resolved=False,
                    payout_numerators=[],
                    payout_denominator=0,
                    winning_outcome=None,
                    redemption_amount=0,
                    message="Condition not found on-chain"
                )

            # Check if resolved (payout denominator > 0 means resolved)
            is_resolved = condition.payout_denominator > 0

            # Determine winning outcome from payouts
            # For binary markets: [UP_payout, DOWN_payout]
            # Winner has non-zero payout
            winning_outcome = None
            if is_resolved and len(condition.payout_numerators) >= 2:
                if condition.payout_numerators[0] > 0:
                    winning_outcome = "UP"  # First outcome (Yes/Up) wins
                elif condition.payout_numerators[1] > 0:
                    winning_outcome = "DOWN"  # Second outcome (No/Down) wins

            return SettlementVerification(
                condition_id=condition_id,
                verified=True,
                on_chain_resolved=is_resolved,
                payout_numerators=condition.payout_numerators,
                payout_denominator=condition.payout_denominator,
                winning_outcome=winning_outcome,
                redemption_amount=0,  # Will be filled by verify_redemption
                message="Resolved" if is_resolved else "Pending"
            )

        except Exception as e:
            logger.warning(f"Failed to verify settlement on-chain: {e}")
            return None

    def verify_user_redemption(
        self,
        user_address: str,
        condition_id: str,
        expected_payout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Verify a user's redemption for a specific market.

        Args:
            user_address: User's wallet address
            condition_id: Market condition ID
            expected_payout: Expected payout in USDC (for comparison)

        Returns:
            Dict with redemption details, or None if not found
        """
        if not self._enabled or not self.graph:
            return None

        try:
            # Get user's redemptions
            redemptions = self.graph.get_user_redemptions(
                user_address=user_address.lower(),
                first=100
            )

            # Find redemption for this condition
            for r in redemptions:
                if r.condition_id.lower() == condition_id.lower():
                    payout_usdc = r.payout / 1e6  # Convert from 6 decimals

                    result = {
                        "found": True,
                        "payout_usdc": payout_usdc,
                        "timestamp": r.timestamp,
                        "tx_hash": r.tx_hash,
                        "collateral_amount": r.collateral_amount / 1e6,
                    }

                    # Compare with expected if provided
                    if expected_payout is not None:
                        diff = abs(payout_usdc - expected_payout)
                        diff_pct = diff / expected_payout if expected_payout > 0 else 0
                        result["matches_expected"] = diff_pct < 0.01  # Within 1%
                        result["difference"] = diff

                    return result

            return {"found": False, "message": "No redemption found for this condition"}

        except Exception as e:
            logger.warning(f"Failed to verify redemption: {e}")
            return None

    def get_user_position_activity(
        self,
        user_address: str,
        hours_back: int = 24
    ) -> Optional[PositionVerification]:
        """
        Get a user's recent position activity.

        Args:
            user_address: User's wallet address
            hours_back: How far back to look

        Returns:
            PositionVerification with activity summary
        """
        if not self._enabled or not self.graph:
            return None

        try:
            stats = self.graph.calculate_user_stats(user_address.lower())

            # Get recent activity
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)

            splits = self.graph.get_user_splits(user_address.lower(), first=50)
            merges = self.graph.get_user_merges(user_address.lower(), first=50)
            redemptions = self.graph.get_user_redemptions(user_address.lower(), first=50)

            recent_activity = []

            for s in splits:
                if s.timestamp > cutoff:
                    recent_activity.append({
                        "type": "SPLIT",
                        "amount": s.amount / 1e6,
                        "timestamp": s.timestamp,
                        "condition": s.condition_id[:10] + "..."
                    })

            for m in merges:
                if m.timestamp > cutoff:
                    recent_activity.append({
                        "type": "MERGE",
                        "amount": m.amount / 1e6,
                        "timestamp": m.timestamp,
                        "condition": m.condition_id[:10] + "..."
                    })

            for r in redemptions:
                if r.timestamp > cutoff:
                    recent_activity.append({
                        "type": "REDEEM",
                        "amount": r.payout / 1e6,
                        "timestamp": r.timestamp,
                        "condition": r.condition_id[:10] + "..."
                    })

            # Sort by timestamp
            recent_activity.sort(key=lambda x: x["timestamp"], reverse=True)

            return PositionVerification(
                user_address=user_address,
                total_splits=stats["total_splits"],
                total_merges=stats["total_merges"],
                total_redemptions=stats["total_redemptions"],
                total_redeemed_usdc=stats["total_redeemed_usdc"],
                recent_activity=recent_activity[:20]  # Last 20 activities
            )

        except Exception as e:
            logger.warning(f"Failed to get position activity: {e}")
            return None

    def cross_verify_settlement(
        self,
        condition_id: str,
        api_outcome: str,
        api_price: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Cross-verify a settlement between API and on-chain.

        Args:
            condition_id: Market condition ID
            api_outcome: Outcome from API ("UP" or "DOWN")
            api_price: Resolution price from API

        Returns:
            Tuple of (matches, message)
        """
        if not self._enabled:
            return True, "Verification skipped (not enabled)"

        verification = self.verify_settlement(condition_id)

        if not verification:
            return True, "Could not verify on-chain (will trust API)"

        if not verification.on_chain_resolved:
            return True, "Not yet resolved on-chain (trusting API)"

        if not verification.winning_outcome:
            return True, "Could not determine on-chain winner (trusting API)"

        # Compare outcomes
        if verification.winning_outcome.upper() == api_outcome.upper():
            return True, f"Verified: API and on-chain agree ({api_outcome})"
        else:
            return False, (
                f"MISMATCH: API says {api_outcome}, "
                f"on-chain says {verification.winning_outcome}"
            )


# Singleton instance
_verifier: Optional[SettlementVerifier] = None


def get_settlement_verifier() -> SettlementVerifier:
    """Get or create the singleton settlement verifier."""
    global _verifier
    if _verifier is None:
        _verifier = SettlementVerifier()
    return _verifier
