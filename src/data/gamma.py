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
    # Cache for target prices by slug to avoid repeated API calls
    _target_price_cache: dict = None

    def __post_init__(self):
        """Initialize HTTP session."""
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PolymarketArbitrageBot/1.0",
        })
        if self._target_price_cache is None:
            self._target_price_cache = {}

    def clear_stale_price_cache(self, current_period_ts: int) -> int:
        """
        Clear target price cache entries for expired periods.

        Called during period transitions to prevent using stale target prices
        from previous periods. The API sometimes takes time to update after
        a period transition, and we don't want to use cached old prices.

        Args:
            current_period_ts: Unix timestamp of the current period start

        Returns:
            Number of cache entries removed
        """
        if not self._target_price_cache:
            return 0

        # Keep only entries for the current and previous period
        # (previous period might still have active markets settling)
        valid_timestamps = {current_period_ts, current_period_ts - 900}

        stale_keys = []
        for key in list(self._target_price_cache.keys()):
            # Key format: "{asset}:{start_timestamp}"
            try:
                _, ts_str = key.split(":")
                ts = int(ts_str)
                if ts not in valid_timestamps:
                    stale_keys.append(key)
            except (ValueError, AttributeError):
                # Invalid key format, remove it
                stale_keys.append(key)

        for key in stale_keys:
            del self._target_price_cache[key]

        if stale_keys:
            logger.info(f"Cleared {len(stale_keys)} stale target price cache entries")

        return len(stale_keys)

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

        # Only check current and previous period - NOT future periods
        # Previous period may still be active in its final minutes
        timestamps = [
            base_ts - 900,   # Previous period (may still be active)
            base_ts,         # Current period
        ]

        logger.debug(f"Checking market timestamps: {timestamps} (base: {base_ts})")

        # Supported assets with their slug prefix
        asset_slugs = {
            "BTC": "btc-updown-15m-",
            "ETH": "eth-updown-15m-",
            "SOL": "sol-updown-15m-",
            "XRP": "xrp-updown-15m-",
        }

        import time as time_module

        for idx, (asset, slug_prefix) in enumerate(asset_slugs.items()):
            # Only fetch supported assets
            if asset not in self.config.supported_assets:
                continue

            # Small delay between assets to avoid rate limiting on price API
            if idx > 0:
                time_module.sleep(0.2)

            for ts in timestamps:
                slug = f"{slug_prefix}{ts}"
                market = self._fetch_market_by_slug(slug)

                if not market:
                    logger.debug(f"No market found for slug: {slug}")
                    continue

                # Check if market is active and not closed
                is_active = market.get("active", False)
                is_closed = market.get("closed", True)
                accepting_orders = market.get("acceptingOrders", False)

                logger.debug(
                    f"Market {slug}: active={is_active}, closed={is_closed}, "
                    f"accepting={accepting_orders}"
                )

                if not is_active or is_closed:
                    continue

                # Check if market is still accepting orders
                if not accepting_orders:
                    continue

                # Parse market into MarketState
                market_state = self._parse_market(market, asset)
                if not market_state:
                    logger.debug(f"Failed to parse market: {slug}")
                    continue

                # Only include if not already in list and has time remaining
                time_remaining = (market_state.end_time - now).total_seconds()
                time_since_start = (now - market_state.start_time).total_seconds()

                # Only trade markets that have STARTED and have time remaining
                if time_since_start < 0:
                    logger.debug(f"SKIP {slug}: hasn't started yet (starts in {-time_since_start:.0f}s)")
                    continue

                if time_remaining <= 0:
                    logger.debug(f"SKIP {slug}: already expired")
                    continue

                # Avoid duplicates
                if any(m.condition_id == market_state.condition_id for m in filtered):
                    continue

                filtered.append(market_state)
                logger.debug(f"[{asset}] Market active | {time_remaining:.0f}s remaining | Target: ${market_state.target_price:,.0f}")

        # Sort by end time (soonest first)
        filtered.sort(key=lambda m: m.end_time)

        # Only log if we found markets (avoid spam when no markets)
        if filtered:
            logger.debug(f"Found {len(filtered)} active 15-minute crypto markets")
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

            # Parse target price - different logic for 15-min vs other markets
            target_price = None
            start_ts = None
            cache_key = None

            # For 15-min "updown" markets, ALWAYS use the dedicated price API
            # Don't trust metadata startPrice/targetPrice as it may be stale
            if "-updown-15m-" in slug.lower():
                start_ts = self._extract_timestamp_from_slug(slug)
                cache_key = f"{asset}:{start_ts}" if start_ts else None

                # Check cache first (from previous API calls only)
                if cache_key and cache_key in self._target_price_cache:
                    cached = self._target_price_cache[cache_key]
                    if cached == 0:
                        # Already tried and failed, skip market silently
                        return None
                    elif cached > 0:
                        target_price = cached
                        logger.debug(f"Using cached target price for {asset}: ${target_price:,.2f}")

                # If not in cache, fetch from Polymarket price API
                if not target_price and start_ts and cache_key:
                    target_price = self.fetch_price_to_beat(asset, start_ts)
                    if target_price:
                        # Cache successful price
                        self._target_price_cache[cache_key] = target_price
                        logger.debug(f"Got target price for {asset} from API: ${target_price:,.2f}")
                    else:
                        # Cache failure to avoid repeated API calls (use 0 as marker)
                        self._target_price_cache[cache_key] = 0

                # If API fails, skip this market
                if not target_price or target_price <= 0:
                    logger.debug(
                        f"Skipping {asset} market (no price data available yet): "
                        f"slug={slug}, start_ts={start_ts}"
                    )
                    return None
            else:
                # For non-15-min markets, use question parsing or metadata
                target_price = self._extract_price_from_question(question)
                if target_price is None or target_price == 0:
                    target_price = market.get("startPrice") or market.get("targetPrice")

                if not target_price or target_price <= 0:
                    logger.warning(
                        f"Market missing valid target price: {condition_id} "
                        f"(extracted: {target_price})"
                    )
                    return None

            # Parse timestamps - try multiple field names
            # IMPORTANT: Use endDate first (full datetime), NOT endDateIso (date only)
            # API returns endDateIso as just "2026-01-21" without time!
            end_time_str = (
                market.get("endDate")
                or market.get("end_date_iso")
                or market.get("endDateIso")  # Last resort - date only
            )
            start_time_str = (
                market.get("startDate")
                or market.get("start_date_iso")
                or market.get("startDateIso")  # Last resort - date only
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
        timestamp = self._extract_timestamp_from_slug(slug)
        if timestamp:
            try:
                return datetime.fromtimestamp(timestamp, tz=timezone.utc)
            except (ValueError, OSError):
                pass
        return None

    def _extract_timestamp_from_slug(self, slug: str) -> Optional[int]:
        """
        Extract unix timestamp from 15-minute market slug.

        Slug format: {asset}-updown-15m-{unix_timestamp}
        Example: btc-updown-15m-1769027400 -> 1769027400

        Args:
            slug: Market slug

        Returns:
            Unix timestamp (seconds) or None
        """
        match = re.search(r"-(\d{10})$", slug)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
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
        # The actual target price should come from the crypto-price API
        placeholders = {
            "BTC": 100000.0,
            "ETH": 3500.0,
            "SOL": 200.0,
            "XRP": 2.5,
        }
        return placeholders.get(asset.upper())

    def fetch_price_to_beat(
        self,
        asset: str,
        market_start_timestamp: int,
        extended_retry: bool = False,
    ) -> Optional[float]:
        """
        Fetch the "price to beat" for a 15-minute market from Polymarket API.

        The price to beat is the Chainlink price at the market's START time,
        which equals the closePrice of the PREVIOUS 15-minute interval.

        API: https://polymarket.com/api/crypto/crypto-price
             ?symbol={BTC|ETH|SOL|XRP}
             &eventStartTime={timestamp_ms}
             &variant=fifteen

        Args:
            asset: Crypto symbol (BTC, ETH, SOL, XRP)
            market_start_timestamp: Unix timestamp (seconds) of market START
            extended_retry: If True, use longer retry logic for period boundaries

        Returns:
            Price to beat (float) or None if unavailable
        """
        import time
        from datetime import datetime, timezone

        # The previous market's timestamp (15 minutes = 900 seconds earlier)
        previous_market_timestamp = market_start_timestamp - 900

        # Convert to milliseconds for API
        timestamp_ms = previous_market_timestamp * 1000

        url = "https://polymarket.com/api/crypto/crypto-price"
        params = {
            "symbol": asset.upper(),
            "eventStartTime": timestamp_ms,
            "variant": "fifteen"
        }

        # Check if we're close to a period boundary (within 30 seconds of a new period)
        now = datetime.now(timezone.utc)
        current_ts = int(now.timestamp())
        current_period_start = (current_ts // 900) * 900
        seconds_into_period = current_ts - current_period_start
        is_period_start = seconds_into_period < 30

        # Use extended retry for period boundaries or when explicitly requested
        if extended_retry or is_period_start:
            max_retries = 6  # More retries at period boundary
            base_delay = 1.0  # Longer initial delay
            logger.debug(
                f"Using extended retry for {asset} (period_start={is_period_start}, "
                f"seconds_into_period={seconds_into_period})"
            )
        else:
            max_retries = 3
            base_delay = 0.5

        logger.debug(
            f"Fetching price to beat for {asset}: "
            f"market_start={market_start_timestamp}, prev_ts={previous_market_timestamp}, "
            f"timestamp_ms={timestamp_ms}"
        )

        for attempt in range(max_retries):
            try:
                response = self._session.get(url, params=params, timeout=10)
                response.raise_for_status()

                # Check for empty response
                if not response.text or not response.text.strip():
                    if attempt < max_retries - 1:
                        delay = base_delay * (attempt + 1)
                        logger.debug(
                            f"Empty response for {asset}, retrying in {delay:.1f}s "
                            f"({attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                        continue
                    logger.warning(f"Empty response from crypto-price API for {asset}")
                    return None

                data = response.json()

                logger.debug(f"Crypto-price API response for {asset}: {data}")

                # closePrice = price at END of previous interval = START of current interval
                # This is the "price to beat"
                price = data.get("closePrice")

                if price and isinstance(price, (int, float)) and price > 0:
                    logger.debug(f"Fetched price to beat for {asset}: ${price:,.2f}")
                    return float(price)

                # Fallback to openPrice if closePrice not available
                price = data.get("openPrice")
                if price and isinstance(price, (int, float)) and price > 0:
                    logger.debug(f"Using openPrice for {asset}: ${price:,.2f}")
                    return float(price)

                # At period boundaries, the API might not have data yet - keep retrying
                if is_period_start and attempt < max_retries - 1:
                    delay = base_delay * (attempt + 1)
                    logger.debug(
                        f"No price data yet for {asset} at period start, retrying in {delay:.1f}s "
                        f"({attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    continue

                # If this interval has no data, it might be too far in past/future
                # Don't log warning for expected "not available" cases
                if data.get("completed") is None:
                    logger.debug(f"No price data available for {asset} at timestamp {timestamp_ms}")
                else:
                    logger.warning(f"No valid price in API response for {asset}: {data}")
                return None

            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (attempt + 1)
                    logger.debug(
                        f"Request error for {asset}, retrying in {delay:.1f}s "
                        f"({attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(delay)
                    continue
                logger.warning(f"Failed to fetch price to beat for {asset}: {e}")
                return None
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (attempt + 1)
                    logger.debug(
                        f"JSON decode error for {asset}, retrying in {delay:.1f}s "
                        f"({attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    continue
                logger.warning(f"Invalid JSON response for {asset}: {e}")
                return None
            except Exception as e:
                logger.warning(f"Error parsing price to beat for {asset}: {e}")
                return None

        return None

    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse datetime string in various formats. Always returns UTC timezone-aware datetime."""
        if not dt_str:
            return None

        try:
            # Handle ISO format with Z suffix
            if dt_str.endswith("Z"):
                dt_str = dt_str[:-1] + "+00:00"
            dt = datetime.fromisoformat(dt_str)
            # Ensure timezone-aware (default to UTC if naive)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass

        # Try other common formats
        formats = [
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f",
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

        Note: For 15-minute "Up or Down" markets, there is NO price in the
        question text. These markets use the Chainlink price at market start
        as the reference. Return None for these cases.

        Args:
            question: Market question text

        Returns:
            Extracted price or None if no price found
        """
        # Skip "Up or Down" markets - they don't have a target price in the question
        # The target is the Chainlink price at market start
        if "Up or Down" in question:
            return None

        # Only match explicit dollar amounts like $97,500 or $3,245.50
        # Don't match arbitrary numbers (like dates: "January 27")
        pattern = r"\$([0-9,]+\.?[0-9]*)"
        match = re.search(pattern, question)
        if match:
            price_str = match.group(1).replace(",", "")
            try:
                price = float(price_str)
                # Sanity check: crypto prices should be reasonable
                # BTC: $10k-$500k, ETH: $100-$50k, SOL: $1-$1k, XRP: $0.1-$50
                if price >= 0.1:  # At least 10 cents
                    return price
            except ValueError:
                pass

        return None

    def get_market_by_id(self, condition_id: str) -> Optional[dict]:
        """
        Fetch a specific market by condition ID.

        Uses query parameter filtering instead of path parameter,
        as the Gamma API /markets/{id} endpoint expects a different format.

        Args:
            condition_id: Market condition ID

        Returns:
            Market dict or None if not found
        """
        # Use query parameter instead of path parameter
        # The Gamma API uses condition_ids (plural) as the filter parameter
        url = f"{self.base_url}/markets"
        params = {"condition_ids": condition_id}

        try:
            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            markets = response.json()
            if markets and len(markets) > 0:
                return markets[0]
            return None
        except requests.RequestException as e:
            logger.debug(f"Failed to fetch market by condition_id {condition_id}: {e}")
            return None

    def get_market_by_slug(self, slug: str) -> Optional[dict]:
        """
        Fetch a specific market by slug.

        Args:
            slug: Market slug (e.g., "btc-updown-15m-2024-01-29-1200")

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
            logger.debug(f"Failed to fetch market by slug {slug}: {e}")
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
            resolution_price = market.get("resolutionPrice")

            # Method 1: Standard resolved flag + tokens[].winner
            if resolved:
                tokens = market.get("tokens", [])
                winning_outcome = None

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

            # Method 2: Auto-resolved 15-min markets — resolved is None but
            # outcomePrices shows clear winner (["1","0"] or ["0","1"]).
            # These markets have automaticallyResolved=True and empty tokens[].
            outcome_prices_raw = market.get("outcomePrices")
            outcomes_raw = market.get("outcomes")
            is_closed = market.get("closed", False)

            if outcome_prices_raw and is_closed:
                try:
                    outcome_prices = json.loads(outcome_prices_raw) if isinstance(outcome_prices_raw, str) else outcome_prices_raw
                    outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else (outcomes_raw or [])
                except (json.JSONDecodeError, TypeError):
                    outcome_prices = []
                    outcomes = []

                if len(outcome_prices) >= 2 and len(outcomes) >= 2:
                    prices = [float(p) for p in outcome_prices]

                    # Clear resolution: one outcome at 1.0 and the other at 0.0
                    if max(prices) >= 0.99 and min(prices) <= 0.01:
                        winner_idx = prices.index(max(prices))
                        winner_label = str(outcomes[winner_idx]).lower()

                        winning_outcome = None
                        if "up" in winner_label or "yes" in winner_label or ">=" in winner_label:
                            winning_outcome = "UP"
                        elif "down" in winner_label or "no" in winner_label or "<" in winner_label:
                            winning_outcome = "DOWN"

                        if winning_outcome:
                            logger.info(
                                "RESOLVE_FROM_PRICES market=%s winner=%s "
                                "outcomePrices=%s outcomes=%s",
                                condition_id[:16], winning_outcome,
                                outcome_prices, outcomes,
                            )
                            return {
                                "resolved": True,
                                "winning_outcome": winning_outcome,
                                "resolution_price": resolution_price,
                            }

            return {"resolved": False, "winning_outcome": None, "resolution_price": None}

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
