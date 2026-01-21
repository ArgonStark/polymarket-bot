"""
Gamma API client for market discovery.

Uses Polymarket's Gamma API to discover and fetch information
about active 15-minute cryptocurrency prediction markets.
"""

import re
import json
import logging
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass

import requests

from ..models import MarketState
from ..config import BotConfig


logger = logging.getLogger(__name__)


@dataclass
class GammaAPI:
    """
    Client for Polymarket Gamma API.

    Used for discovering active 15-minute crypto markets
    and fetching market metadata.
    """

    config: BotConfig
    _session: Optional[requests.Session] = None

    def __post_init__(self):
        """Initialize HTTP session."""
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PolymarketArbitrageBot/1.0",
        })

    @property
    def base_url(self) -> str:
        """Get Gamma API base URL."""
        return self.config.endpoints.gamma_api_url

    def get_active_markets(
        self,
        tag: str = "crypto",
        limit: int = 100,
    ) -> list[dict]:
        """
        Fetch all active markets with given tag.

        Args:
            tag: Market tag to filter by (default: "crypto")
            limit: Maximum number of markets to return

        Returns:
            List of market dictionaries from the API
        """
        url = f"{self.base_url}/markets"
        params = {
            "active": True,
            "closed": False,
            "tag": tag,
            "limit": limit,
        }

        try:
            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            markets = response.json()
            logger.debug(f"Fetched {len(markets)} active {tag} markets")
            return markets
        except requests.RequestException as e:
            logger.error(f"Failed to fetch markets: {e}")
            return []

    def get_15min_crypto_markets(self) -> list[MarketState]:
        """
        Fetch currently active 15-minute crypto markets.

        Filters markets by:
        - Contains "15" in the question (15-minute markets)
        - Contains a supported asset (BTC, ETH, SOL, XRP)

        Returns:
            List of MarketState objects for active 15-min markets
        """
        markets = self.get_active_markets(tag="crypto")
        filtered = []

        for market in markets:
            question = market.get("question", "").lower()

            # Check for 15-minute market
            if "15" not in question:
                continue

            # Check for supported asset
            asset = None
            for supported_asset in self.config.supported_assets:
                if supported_asset.lower() in question:
                    asset = supported_asset
                    break

            if not asset:
                continue

            # Parse market into MarketState
            market_state = self._parse_market(market, asset)
            if market_state:
                filtered.append(market_state)

        logger.info(f"Found {len(filtered)} active 15-minute crypto markets")
        return filtered

    def _parse_market(
        self,
        market: dict,
        asset: str,
    ) -> Optional[MarketState]:
        """
        Parse API market response into MarketState.

        Handles both formats:
        - tokens array with token objects
        - clobTokenIds/outcomePrices as JSON strings

        Args:
            market: Raw market dict from API
            asset: Asset symbol (BTC, ETH, etc.)

        Returns:
            MarketState or None if parsing fails
        """
        try:
            condition_id = market.get("conditionId", "")
            question = market.get("question", "")

            # Parse clobTokenIds - may be JSON string or list
            clob_token_ids = market.get("clobTokenIds")
            if isinstance(clob_token_ids, str):
                try:
                    clob_token_ids = json.loads(clob_token_ids)
                except json.JSONDecodeError:
                    clob_token_ids = []

            # Parse outcomePrices - may be JSON string or list
            outcome_prices = market.get("outcomePrices")
            if isinstance(outcome_prices, str):
                try:
                    outcome_prices = json.loads(outcome_prices)
                except json.JSONDecodeError:
                    outcome_prices = []

            # Get outcomes list
            outcomes = market.get("outcomes", [])
            if isinstance(outcomes, str):
                try:
                    outcomes = json.loads(outcomes)
                except json.JSONDecodeError:
                    outcomes = []

            # Also try tokens array (alternative format)
            tokens = market.get("tokens", [])

            # Determine UP and DOWN token IDs
            up_token_id = None
            down_token_id = None
            best_bid = 0.0
            best_ask = 1.0

            # Method 1: Use outcomes + clobTokenIds
            if outcomes and clob_token_ids and len(outcomes) >= 2 and len(clob_token_ids) >= 2:
                up_index = None
                down_index = None

                for i, outcome in enumerate(outcomes):
                    outcome_lower = str(outcome).lower()
                    if any(kw in outcome_lower for kw in ["up", "yes", "higher", ">="]):
                        up_index = i
                    elif any(kw in outcome_lower for kw in ["down", "no", "lower", "<"]):
                        down_index = i

                # Default to index 0=UP, 1=DOWN if not found
                if up_index is None:
                    up_index = 0
                if down_index is None:
                    down_index = 1

                if len(clob_token_ids) > max(up_index, down_index):
                    up_token_id = clob_token_ids[up_index]
                    down_token_id = clob_token_ids[down_index]

                    # Get prices from outcomePrices
                    if outcome_prices and len(outcome_prices) > up_index:
                        up_price = float(outcome_prices[up_index])
                        best_bid = max(0.01, up_price - 0.01)
                        best_ask = min(0.99, up_price + 0.01)

            # Method 2: Use tokens array (fallback)
            if not up_token_id or not down_token_id:
                if len(tokens) >= 2:
                    for token in tokens:
                        outcome = str(token.get("outcome", "")).lower()
                        token_id = token.get("token_id", "")

                        if any(kw in outcome for kw in ["up", "yes", "higher", ">="]):
                            up_token_id = token_id
                            best_bid = float(token.get("bestBid", 0.0) or 0.0)
                            best_ask = float(token.get("bestAsk", 1.0) or 1.0)
                        elif any(kw in outcome for kw in ["down", "no", "lower", "<"]):
                            down_token_id = token_id

                    # Fallback to index assignment
                    if not up_token_id and len(tokens) >= 1:
                        up_token_id = tokens[0].get("token_id", "")
                        best_bid = float(tokens[0].get("bestBid", 0.0) or 0.0)
                        best_ask = float(tokens[0].get("bestAsk", 1.0) or 1.0)
                    if not down_token_id and len(tokens) >= 2:
                        down_token_id = tokens[1].get("token_id", "")

            if not up_token_id or not down_token_id:
                logger.warning(f"Cannot identify UP/DOWN tokens: {condition_id}")
                return None

            # Parse target price from question
            target_price = self._extract_price_from_question(question)
            if target_price is None or target_price == 0:
                # Try to get from market metadata
                target_price = market.get("startPrice") or market.get("targetPrice")

            # Validate target price - CRITICAL: cannot be 0 or None (causes division by zero)
            if not target_price or target_price <= 0:
                logger.warning(
                    f"Market missing valid target price: {condition_id} "
                    f"(extracted: {target_price})"
                )
                return None

            # Parse timestamps - try multiple field names
            end_time_str = (
                market.get("endDateIso")
                or market.get("endDate")
                or market.get("end_date_iso")
            )
            start_time_str = (
                market.get("startDateIso")
                or market.get("startDate")
                or market.get("start_date_iso")
            )

            if not end_time_str:
                logger.warning(f"Market missing end time: {condition_id}")
                return None

            # Parse end time
            end_time = self._parse_datetime(end_time_str)
            if not end_time:
                logger.warning(f"Could not parse end time: {end_time_str}")
                return None

            # Parse start time
            start_time = self._parse_datetime(start_time_str) if start_time_str else end_time

            return MarketState(
                condition_id=condition_id,
                question=question,
                up_token_id=up_token_id,
                down_token_id=down_token_id,
                asset=asset,
                target_price=target_price,
                start_time=start_time,
                end_time=end_time,
                best_bid=best_bid,
                best_ask=best_ask,
            )

        except Exception as e:
            logger.error(f"Failed to parse market {market.get('conditionId', 'unknown')}: {e}")
            return None

    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse datetime string in various formats."""
        if not dt_str:
            return None

        try:
            # Handle ISO format with Z suffix
            if dt_str.endswith("Z"):
                dt_str = dt_str[:-1] + "+00:00"
            return datetime.fromisoformat(dt_str)
        except ValueError:
            pass

        # Try other common formats
        formats = [
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(dt_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue

        return None

    def _extract_price_from_question(self, question: str) -> Optional[float]:
        """
        Extract target price from market question.

        Examples:
        - "Will BTC be above $97,500 at 3:15 PM?" -> 97500.0
        - "ETH 15min: >= $3,245.50?" -> 3245.50

        Args:
            question: Market question text

        Returns:
            Extracted price or None
        """
        # Pattern: $XX,XXX.XX or $XX,XXX or $XXXXX
        patterns = [
            r"\$([0-9,]+\.?[0-9]*)",
            r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, question)
            if match:
                price_str = match.group(1).replace(",", "")
                try:
                    return float(price_str)
                except ValueError:
                    continue

        return None

    def get_market_by_id(self, condition_id: str) -> Optional[dict]:
        """
        Fetch a specific market by condition ID.

        Args:
            condition_id: Market condition ID

        Returns:
            Market dict or None if not found
        """
        url = f"{self.base_url}/markets/{condition_id}"

        try:
            response = self._session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch market {condition_id}: {e}")
            return None

    def get_market_prices(self, condition_id: str) -> Optional[dict]:
        """
        Fetch current prices for a market.

        Args:
            condition_id: Market condition ID

        Returns:
            Dict with price information or None
        """
        url = f"{self.base_url}/markets/{condition_id}/prices"

        try:
            response = self._session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch prices for {condition_id}: {e}")
            return None

    def get_market_resolution(self, condition_id: str) -> Optional[dict]:
        """
        Fetch resolution/settlement status for a market.

        Args:
            condition_id: Market condition ID

        Returns:
            Dict with resolution info:
            - resolved: bool
            - winning_outcome: "UP" or "DOWN" or None
            - resolution_price: float (Chainlink settlement price)
        """
        market = self.get_market_by_id(condition_id)
        if not market:
            return None

        try:
            resolved = market.get("resolved", False)
            if not resolved:
                return {"resolved": False, "winning_outcome": None, "resolution_price": None}

            # Determine winning outcome from token payouts
            tokens = market.get("tokens", [])
            winning_outcome = None
            resolution_price = market.get("resolutionPrice")

            for token in tokens:
                winner = token.get("winner", False)
                if winner:
                    outcome = token.get("outcome", "").lower()
                    if "up" in outcome or "yes" in outcome or ">=" in outcome:
                        winning_outcome = "UP"
                    elif "down" in outcome or "no" in outcome or "<" in outcome:
                        winning_outcome = "DOWN"
                    break

            return {
                "resolved": True,
                "winning_outcome": winning_outcome,
                "resolution_price": resolution_price,
            }

        except Exception as e:
            logger.error(f"Failed to parse resolution for {condition_id}: {e}")
            return None

    def get_upcoming_markets(
        self,
        lookahead_minutes: int = 30,
    ) -> list[MarketState]:
        """
        Fetch 15-minute markets starting soon.

        Looks for markets that haven't started yet but will
        start within the lookahead window.

        Args:
            lookahead_minutes: How far ahead to look (default 30 min)

        Returns:
            List of upcoming MarketState objects
        """
        # Fetch markets including those not yet active
        url = f"{self.base_url}/markets"
        params = {
            "closed": False,
            "tag": "crypto",
            "limit": 100,
        }

        try:
            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            markets = response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch upcoming markets: {e}")
            return []

        now = datetime.now(timezone.utc)
        upcoming = []

        for market in markets:
            question = market.get("question", "").lower()

            # Check for 15-minute market
            if "15" not in question:
                continue

            # Check for supported asset
            asset = None
            for supported_asset in self.config.supported_assets:
                if supported_asset.lower() in question:
                    asset = supported_asset
                    break

            if not asset:
                continue

            # Check timing
            start_time_str = market.get("startDateIso")
            if not start_time_str:
                continue

            start_time = datetime.fromisoformat(
                start_time_str.replace("Z", "+00:00")
            )

            # Only include markets starting within lookahead window
            time_until_start = (start_time - now).total_seconds() / 60
            if 0 < time_until_start <= lookahead_minutes:
                market_state = self._parse_market(market, asset)
                if market_state:
                    upcoming.append(market_state)

        logger.debug(f"Found {len(upcoming)} upcoming 15-minute markets")
        return upcoming

    def close(self):
        """Close HTTP session."""
        if self._session:
            self._session.close()
            logger.debug("Gamma API session closed")
