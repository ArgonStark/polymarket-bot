"""
Historical price data fetching from CoinGecko API.

This module fetches historical price data at startup to pre-populate
price histories, allowing the bot to make better trading decisions
immediately without waiting to collect price data.
"""

import logging
import asyncio
from typing import Optional
import aiohttp
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# CoinGecko coin IDs for supported assets
COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
}

# CoinGecko API base URL (free tier)
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"


async def fetch_historical_prices(
    asset: str,
    hours: int = 1,
    session: Optional[aiohttp.ClientSession] = None,
) -> list[float]:
    """
    Fetch historical price data for an asset from CoinGecko.

    Args:
        asset: Asset symbol (BTC, ETH, SOL, XRP)
        hours: Number of hours of history to fetch (default: 1)
        session: Optional aiohttp session for reuse

    Returns:
        List of prices (most recent last), or empty list on error
    """
    coin_id = COINGECKO_IDS.get(asset.upper())
    if not coin_id:
        logger.warning(f"Unknown asset for CoinGecko: {asset}")
        return []

    # CoinGecko market_chart endpoint
    # days=1 with granularity gives ~5-minute intervals
    url = f"{COINGECKO_API_URL}/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "1",  # Last 24 hours
    }

    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True

    try:
        async with session.get(url, params=params, timeout=30) as response:
            if response.status != 200:
                logger.warning(
                    f"CoinGecko API error for {asset}: HTTP {response.status}"
                )
                return []

            data = await response.json()

            # Extract prices from response
            # Response format: {"prices": [[timestamp_ms, price], ...]}
            prices_data = data.get("prices", [])

            if not prices_data:
                logger.warning(f"No price data returned for {asset}")
                return []

            # Convert to just prices (sorted by time, most recent last)
            # Only take the most recent N hours
            now_ms = datetime.now(timezone.utc).timestamp() * 1000
            cutoff_ms = now_ms - (hours * 60 * 60 * 1000)

            prices = []
            for timestamp_ms, price in prices_data:
                if timestamp_ms >= cutoff_ms:
                    prices.append(price)

            logger.info(
                f"[{asset}] Fetched {len(prices)} historical prices "
                f"(last {hours}h from CoinGecko)"
            )

            return prices

    except asyncio.TimeoutError:
        logger.debug(f"Timeout fetching historical prices for {asset}")
        return []
    except Exception as e:
        # Log at debug level - this is expected to fail in sandboxed environments
        # The bot will collect prices during warm-up period instead
        logger.debug(f"Could not fetch historical prices for {asset}: {e}")
        return []
    finally:
        if close_session:
            await session.close()


async def fetch_all_historical_prices(
    assets: list[str],
    hours: int = 1,
) -> dict[str, list[float]]:
    """
    Fetch historical prices for multiple assets concurrently.

    Args:
        assets: List of asset symbols (BTC, ETH, SOL, XRP)
        hours: Number of hours of history to fetch

    Returns:
        Dict mapping asset symbols to price lists
    """
    logger.info(
        f"📚 Fetching historical prices for {', '.join(assets)} "
        f"(last {hours}h)..."
    )

    results = {}

    async with aiohttp.ClientSession() as session:
        # Fetch all assets concurrently
        tasks = []
        for asset in assets:
            tasks.append(fetch_historical_prices(asset, hours, session))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for asset, response in zip(assets, responses):
            if isinstance(response, Exception):
                logger.warning(f"Failed to fetch {asset}: {response}")
                results[asset] = []
            else:
                results[asset] = response

            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)

    # Summary
    total_prices = sum(len(p) for p in results.values())
    logger.info(f"📚 Historical data loaded: {total_prices} total price points")

    return results


def prepopulate_price_histories(
    signal_generator,
    historical_data: dict[str, list[float]],
):
    """
    Pre-populate the signal generator's price histories with historical data.

    Args:
        signal_generator: SignalGenerator instance
        historical_data: Dict mapping asset symbols to price lists
    """
    for asset, prices in historical_data.items():
        if not prices:
            continue

        symbol = f"{asset.lower()}/usd"

        # Pre-populate the price history
        if symbol not in signal_generator.price_histories:
            signal_generator.price_histories[symbol] = []

        # Add historical prices (keep only last 100 to match SignalGenerator limit)
        signal_generator.price_histories[symbol] = prices[-100:]

        # Also set the current price
        if prices:
            signal_generator.chainlink_prices[symbol] = prices[-1]

        # Update volatility estimate based on historical data
        if len(prices) >= 5:
            from ..probability import estimate_volatility
            signal_generator.volatilities[asset.lower()] = estimate_volatility(
                prices,
                window=20,
                default_vol=signal_generator.config.volatility.get(asset.upper()),
            )

        logger.debug(
            f"[{asset}] Pre-populated {len(prices)} prices, "
            f"latest: ${prices[-1]:,.2f}"
        )

    logger.info(
        f"✅ Price histories pre-populated for: "
        f"{', '.join(historical_data.keys())}"
    )
