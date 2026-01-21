"""
Gamma API client for market discovery.

Uses Polymarket's Gamma API to discover and fetch information
about active 15-minute cryptocurrency prediction markets.
"""

import re
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

        Args:
            market: Raw market dict from API
            asset: Asset symbol (BTC, ETH, etc.)

        Returns:
            MarketState or None if parsing fails
        """
        try:
            condition_id = market.get("conditionId", "")
            question = market.get("question", "")

            # Extract tokens (outcomes)
            tokens = market.get("tokens", [])
            if len(tokens) < 2:
                logger.warning(f"Market missing tokens: {condition_id}")
                return None

            # Find UP and DOWN tokens
            up_token_id = None
            down_token_id = None

            for token in tokens:
                outcome = token.get("outcome", "").lower()
                token_id = token.get("token_id", "")

                if "up" in outcome or "yes" in outcome or ">=" in outcome:
                    up_token_id = token_id
                elif "down" in outcome or "no" in outcome or "<" in outcome:
                    down_token_id = token_id

            if not up_token_id or not down_token_id:
                # Try to assign by index
                if len(tokens) >= 2:
                    up_token_id = tokens[0].get("token_id", "")
                    down_token_id = tokens[1].get("token_id", "")
                else:
                    logger.warning(f"Cannot identify UP/DOWN tokens: {condition_id}")
                    return None

            # Parse target price from question
            target_price = self._extract_price_from_question(question)
            if target_price is None:
                # Try to get from market metadata
                target_price = market.get("startPrice", 0.0)

            # Parse timestamps
            end_time_str = market.get("endDateIso")
            start_time_str = market.get("startDateIso")

            if end_time_str:
                end_time = datetime.fromisoformat(
                    end_time_str.replace("Z", "+00:00")
                )
            else:
                logger.warning(f"Market missing end time: {condition_id}")
                return None

            if start_time_str:
                start_time = datetime.fromisoformat(
                    start_time_str.replace("Z", "+00:00")
                )
            else:
                start_time = end_time  # Fallback

            # Get current best prices
            best_bid = 0.0
            best_ask = 1.0

            for token in tokens:
                if token.get("token_id") == up_token_id:
                    best_bid = float(token.get("bestBid", 0.0) or 0.0)
                    best_ask = float(token.get("bestAsk", 1.0) or 1.0)
                    break

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
            logger.error(f"Failed to parse market: {e}")
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

    def close(self):
        """Close HTTP session."""
        if self._session:
            self._session.close()
            logger.debug("Gamma API session closed")
