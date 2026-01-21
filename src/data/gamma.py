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
        tag: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Fetch all active markets, optionally filtered by tag.

        Args:
            tag: Market tag to filter by (None for no tag filter)
            limit: Maximum number of markets to return

        Returns:
            List of market dictionaries from the API
        """
        url = f"{self.base_url}/markets"
        params = {
            "active": True,
            "closed": False,
            "limit": limit,
        }

        # Only add tag filter if specified
        if tag:
            params["tag"] = tag

        try:
            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            markets = response.json()
            tag_desc = tag if tag else "all"
            logger.debug(f"Fetched {len(markets)} active {tag_desc} markets")
            return markets
        except requests.RequestException as e:
            logger.error(f"Failed to fetch markets: {e}")
            return []

    def get_15min_crypto_markets(self) -> list[MarketState]:
        """
        Fetch currently active 15-minute crypto markets.

        These markets must be fetched by constructing specific slugs:
        - Format: {asset}-updown-15m-{unix_timestamp}
        - Timestamp is the market start time (every 15 min: :00, :15, :30, :45)

        Returns:
            List of MarketState objects for active 15-min markets
        """
        filtered = []
        now = datetime.now(timezone.utc)
        current_ts = int(now.timestamp())

        # Round down to nearest 15 minutes (900 seconds)
        base_ts = (current_ts // 900) * 900

        # Generate timestamps for current and next few periods
        # Check current, previous (may still be active), and next period
        timestamps = [
            base_ts - 900,   # Previous period (may still be in final minutes)
            base_ts,         # Current period
            base_ts + 900,   # Next period (for upcoming)
        ]

        # Supported assets with their slug prefix
        asset_slugs = {
            "BTC": "btc-updown-15m-",
            "ETH": "eth-updown-15m-",
            "SOL": "sol-updown-15m-",
            "XRP": "xrp-updown-15m-",
        }

        for asset, slug_prefix in asset_slugs.items():
            # Only fetch supported assets
            if asset not in self.config.supported_assets:
                continue

            for ts in timestamps:
                slug = f"{slug_prefix}{ts}"
                market = self._fetch_market_by_slug(slug)

                if market:
                    # Check if market is active and not closed
                    if not market.get("active", False) or market.get("closed", True):
                        continue

                    # Check if market is still accepting orders
                    if not market.get("acceptingOrders", False):
                        continue

                    # Parse market into MarketState
                    market_state = self._parse_market(market, asset)
                    if market_state:
                        # Only include if not already in list and has time remaining
                        time_remaining = (market_state.end_time - now).total_seconds()
                        if time_remaining > 0:
                            # Avoid duplicates
                            if not any(m.condition_id == market_state.condition_id for m in filtered):
                                filtered.append(market_state)

        # Sort by end time (soonest first)
        filtered.sort(key=lambda m: m.end_time)

        logger.info(f"Found {len(filtered)} active 15-minute crypto markets")
        return filtered

    def _fetch_market_by_slug(self, slug: str) -> Optional[dict]:
        """
        Fetch a specific market by its slug.

        Args:
            slug: Market slug (e.g., btc-updown-15m-1769027400)

        Returns:
            Market dict or None if not found
        """
        url = f"{self.base_url}/markets"
        params = {"slug": slug}

        try:
            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            markets = response.json()

            if markets and len(markets) > 0:
                return markets[0]
            return None

        except requests.RequestException as e:
            logger.debug(f"Failed to fetch market {slug}: {e}")
            return None

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

        For 15-minute markets:
        - Slug format: {asset}-updown-15m-{unix_timestamp}
        - Outcomes are always ["Up", "Down"]
        - Start time can be extracted from slug timestamp

        Args:
            market: Raw market dict from API
            asset: Asset symbol (BTC, ETH, etc.)

        Returns:
            MarketState or None if parsing fails
        """
        try:
            condition_id = market.get("conditionId", "")
            question = market.get("question", "")
            slug = market.get("slug", "")

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
                    # 15-min markets use exactly "Up" and "Down"
                    if outcome_lower == "up" or any(kw in outcome_lower for kw in ["yes", "higher", ">="]):
                        up_index = i
                    elif outcome_lower == "down" or any(kw in outcome_lower for kw in ["no", "lower", "<"]):
                        down_index = i

                # Default to index 0=UP, 1=DOWN if not found (standard for 15-min markets)
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

                        if outcome == "up" or any(kw in outcome for kw in ["yes", "higher", ">="]):
                            up_token_id = token_id
                            best_bid = float(token.get("bestBid", 0.0) or 0.0)
                            best_ask = float(token.get("bestAsk", 1.0) or 1.0)
                        elif outcome == "down" or any(kw in outcome for kw in ["no", "lower", "<"]):
                            down_token_id = token_id

                    # Fallback to index assignment (15-min markets: index 0=Up, 1=Down)
                    if not up_token_id and len(tokens) >= 1:
                        up_token_id = tokens[0].get("token_id", "")
                        best_bid = float(tokens[0].get("bestBid", 0.0) or 0.0)
                        best_ask = float(tokens[0].get("bestAsk", 1.0) or 1.0)
                    if not down_token_id and len(tokens) >= 2:
                        down_token_id = tokens[1].get("token_id", "")

            if not up_token_id or not down_token_id:
                logger.warning(f"Cannot identify UP/DOWN tokens: {condition_id}")
                return None

            # Parse target price from question or metadata
            target_price = self._extract_price_from_question(question)
            if target_price is None or target_price == 0:
                # Try to get from market metadata
                target_price = market.get("startPrice") or market.get("targetPrice")

            # For 15-min markets, target price may not be in API response
            # Use a placeholder that will be updated from Chainlink at runtime
            if not target_price or target_price <= 0:
                # Check if this is a 15-minute market (has updown-15m in slug)
                if "-updown-15m-" in slug.lower():
                    # Use a temporary placeholder - will be updated from Chainlink
                    # The bot should fetch the actual price from Chainlink stream
                    target_price = self._get_placeholder_price(asset)
                    if target_price and target_price > 0:
                        logger.debug(
                            f"Using placeholder target price for {asset}: {target_price}"
                        )
                    else:
                        logger.warning(
                            f"Market missing valid target price: {condition_id} "
                            f"(15-min market, will need Chainlink price)"
                        )
                        return None
                else:
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

            # Parse start time - for 15-min markets, try to extract from slug timestamp
            start_time = None
            if start_time_str:
                start_time = self._parse_datetime(start_time_str)

            if not start_time:
                # Try to extract start time from slug (format: asset-updown-15m-{timestamp})
                start_time = self._extract_start_time_from_slug(slug)

            if not start_time:
                # Fallback: start time is 15 minutes before end time
                from datetime import timedelta
                start_time = end_time - timedelta(minutes=15)

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

    def _extract_start_time_from_slug(self, slug: str) -> Optional[datetime]:
        """
        Extract start time from 15-minute market slug.

        Slug format: {asset}-updown-15m-{unix_timestamp}
        Example: btc-updown-15m-1769027400

        Args:
            slug: Market slug

        Returns:
            Start time as datetime or None
        """
        match = re.search(r"-(\d{10})$", slug)
        if match:
            try:
                timestamp = int(match.group(1))
                return datetime.fromtimestamp(timestamp, tz=timezone.utc)
            except (ValueError, OSError):
                pass
        return None

    def _get_placeholder_price(self, asset: str) -> Optional[float]:
        """
        Get a placeholder price for an asset when target price is not available.

        These are approximate values that should be updated from Chainlink.
        Used only to allow market parsing to succeed initially.

        Args:
            asset: Asset symbol (BTC, ETH, etc.)

        Returns:
            Placeholder price or None
        """
        # Approximate prices as of 2025 - these are just placeholders
        # The actual target price should come from Chainlink at market start
        placeholders = {
            "BTC": 100000.0,
            "ETH": 3500.0,
            "SOL": 200.0,
            "XRP": 2.5,
        }
        return placeholders.get(asset.upper())

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

        Looks for markets that will start within the lookahead window.

        Args:
            lookahead_minutes: How far ahead to look (default 30 min)

        Returns:
            List of upcoming MarketState objects
        """
        now = datetime.now(timezone.utc)
        current_ts = int(now.timestamp())
        upcoming = []

        # Round up to next 15-minute boundary
        base_ts = ((current_ts // 900) + 1) * 900

        # Generate timestamps for upcoming periods within lookahead window
        num_periods = (lookahead_minutes // 15) + 1
        timestamps = [base_ts + (i * 900) for i in range(num_periods)]

        # Supported assets with their slug prefix
        asset_slugs = {
            "BTC": "btc-updown-15m-",
            "ETH": "eth-updown-15m-",
            "SOL": "sol-updown-15m-",
            "XRP": "xrp-updown-15m-",
        }

        for asset, slug_prefix in asset_slugs.items():
            # Only fetch supported assets
            if asset not in self.config.supported_assets:
                continue

            for ts in timestamps:
                start_time = datetime.fromtimestamp(ts, tz=timezone.utc)
                time_until_start = (start_time - now).total_seconds() / 60

                # Only include markets starting within lookahead window
                if time_until_start <= 0 or time_until_start > lookahead_minutes:
                    continue

                slug = f"{slug_prefix}{ts}"
                market = self._fetch_market_by_slug(slug)

                if market:
                    # Parse market into MarketState
                    market_state = self._parse_market(market, asset)
                    if market_state:
                        upcoming.append(market_state)

        # Sort by start time (soonest first)
        upcoming.sort(key=lambda m: m.start_time)

        logger.debug(f"Found {len(upcoming)} upcoming 15-minute markets")
        return upcoming

    def close(self):
        """Close HTTP session."""
        if self._session:
            self._session.close()
            logger.debug("Gamma API session closed")
