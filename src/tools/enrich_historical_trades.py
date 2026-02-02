"""
Historical Trade Data Enrichment Tool

Fetches historical closed positions from Polymarket and enriches them with
market context from Binance for ML training.

Usage:
    python -m src.tools.enrich_historical_trades --wallet 0x63ce342161250d705dc0b16df89036c8e5f9ba9a

Data Sources:
    - Polymarket closed-positions API: Trade outcomes with P&L
    - Binance historical klines API: OHLCV data for indicators
"""

import os
import json
import time
import logging
import argparse
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
BINANCE_API = "https://api.binance.com/api/v3"
POLYMARKET_DATA_API = "https://data-api.polymarket.com"

# Asset detection patterns
ASSET_PATTERNS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
    "XRP": ["xrp", "ripple"],
}

# Asset to Binance symbol mapping
ASSET_TO_SYMBOL = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
}

# Rate limiting
BINANCE_RATE_LIMIT = 0.12  # seconds between requests
POLYMARKET_RATE_LIMIT = 0.25


@dataclass
class EnrichedTrade:
    """A trade enriched with market context for ML training."""

    # Basic trade info
    trade_id: str
    timestamp: str
    timestamp_ms: int
    asset: str
    side: str  # "UP" or "DOWN" (what they bet on)
    outcome: str  # "WIN" or "LOSS"
    entry_price: float
    size_usd: float
    realized_pnl: float

    # Price context at entry
    asset_price: float = 0.0
    btc_price: float = 0.0
    eth_price: float = 0.0
    sol_price: float = 0.0

    # Technical indicators
    rsi_14: float = 50.0
    rsi_7: float = 50.0
    macd_histogram: float = 0.0
    macd_signal: str = "none"

    # Bollinger Bands
    bb_bandwidth: float = 0.0
    bb_position: str = "middle"

    # Stochastic
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    stoch_signal: str = "neutral"

    # Moving averages
    sma_20: float = 0.0
    sma_50: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    price_vs_sma20: float = 0.0
    price_vs_sma50: float = 0.0

    # Trend
    trend_1h: float = 0.0
    trend_4h: float = 0.0
    trend_1d: float = 0.0
    trend_direction: str = "neutral"

    # Volatility
    volatility_1h: float = 0.0
    volatility_24h: float = 0.0
    atr_14: float = 0.0

    # Volume
    volume_ratio: float = 1.0
    volume_24h: float = 0.0

    # Time context
    hour_utc: int = 0
    day_of_week: int = 0
    is_weekend: bool = False
    is_asia_session: bool = False
    is_europe_session: bool = False
    is_us_session: bool = False

    # Market structure
    higher_high: bool = False
    lower_low: bool = False
    consolidating: bool = False


class BinanceHistoricalData:
    """Fetches historical OHLCV data from Binance."""

    def __init__(self):
        self.cache: Dict[str, List[Dict]] = {}
        self.last_request = 0
        self.price_cache: Dict[str, float] = {}

    def _rate_limit(self):
        elapsed = time.time() - self.last_request
        if elapsed < BINANCE_RATE_LIMIT:
            time.sleep(BINANCE_RATE_LIMIT - elapsed)
        self.last_request = time.time()

    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
        limit: int = 500
    ) -> List[Dict]:
        """Fetch historical klines from Binance."""
        cache_key = f"{symbol}_{interval}_{start_time}_{end_time}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        self._rate_limit()

        try:
            response = requests.get(
                f"{BINANCE_API}/klines",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": start_time,
                    "endTime": end_time,
                    "limit": limit,
                },
                timeout=10
            )
            response.raise_for_status()

            raw_klines = response.json()
            klines = []
            for k in raw_klines:
                klines.append({
                    "open_time": k[0],
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_time": k[6],
                })

            self.cache[cache_key] = klines
            return klines

        except Exception as e:
            logger.debug(f"Failed to fetch klines for {symbol}: {e}")
            return []

    def get_price_at_time(self, symbol: str, timestamp_ms: int) -> Optional[float]:
        """Get the price at a specific timestamp."""
        cache_key = f"{symbol}_{timestamp_ms // 60000}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]

        klines = self.get_klines(
            symbol=symbol,
            interval="1m",
            start_time=timestamp_ms - 60000,
            end_time=timestamp_ms + 60000,
            limit=3
        )

        if klines:
            price = klines[0]["close"]
            self.price_cache[cache_key] = price
            return price
        return None

    def get_ohlcv_context(
        self,
        symbol: str,
        timestamp_ms: int,
        lookback_hours: int = 48
    ) -> List[Dict]:
        """Get OHLCV data for context around a timestamp."""
        start_time = timestamp_ms - (lookback_hours * 60 * 60 * 1000)
        return self.get_klines(
            symbol=symbol,
            interval="1h",
            start_time=start_time,
            end_time=timestamp_ms,
            limit=lookback_hours
        )


class TechnicalIndicators:
    """Calculate technical indicators from OHLCV data."""

    @staticmethod
    def calculate_rsi(closes: List[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0

        changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [c if c > 0 else 0 for c in changes]
        losses = [-c if c < 0 else 0 for c in changes]

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 2)

    @staticmethod
    def calculate_sma(values: List[float], period: int) -> float:
        if len(values) < period:
            return values[-1] if values else 0.0
        return sum(values[-period:]) / period

    @staticmethod
    def calculate_ema(values: List[float], period: int) -> float:
        if len(values) < period:
            return values[-1] if values else 0.0

        multiplier = 2 / (period + 1)
        ema = sum(values[:period]) / period

        for price in values[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    @staticmethod
    def calculate_macd(closes: List[float]) -> Tuple[float, str]:
        if len(closes) < 26:
            return 0.0, "none"

        ema12 = TechnicalIndicators.calculate_ema(closes, 12)
        ema26 = TechnicalIndicators.calculate_ema(closes, 26)
        macd_line = ema12 - ema26

        # Previous MACD for crossover detection
        if len(closes) > 27:
            prev_ema12 = TechnicalIndicators.calculate_ema(closes[:-1], 12)
            prev_ema26 = TechnicalIndicators.calculate_ema(closes[:-1], 26)
            prev_macd = prev_ema12 - prev_ema26

            if macd_line > 0 and prev_macd <= 0:
                signal = "bullish_cross"
            elif macd_line < 0 and prev_macd >= 0:
                signal = "bearish_cross"
            else:
                signal = "bullish" if macd_line > 0 else "bearish"
        else:
            signal = "bullish" if macd_line > 0 else "bearish"

        return round(macd_line, 6), signal

    @staticmethod
    def calculate_bollinger_bands(closes: List[float], period: int = 20) -> Tuple[float, str]:
        if len(closes) < period:
            return 0.0, "middle"

        sma = sum(closes[-period:]) / period
        variance = sum((p - sma) ** 2 for p in closes[-period:]) / period
        std = variance ** 0.5

        upper = sma + (2 * std)
        lower = sma - (2 * std)
        bandwidth = (upper - lower) / sma if sma > 0 else 0.0

        current_price = closes[-1]
        if current_price > upper:
            position = "above"
        elif current_price < lower:
            position = "below"
        else:
            position = "middle"

        return round(bandwidth, 4), position

    @staticmethod
    def calculate_stochastic(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Tuple[float, float, str]:
        if len(closes) < period:
            return 50.0, 50.0, "neutral"

        highest_high = max(highs[-period:])
        lowest_low = min(lows[-period:])

        if highest_high == lowest_low:
            k = 50.0
        else:
            k = 100 * (closes[-1] - lowest_low) / (highest_high - lowest_low)

        d = k  # Simplified

        if k > 80:
            signal = "overbought"
        elif k < 20:
            signal = "oversold"
        else:
            signal = "neutral"

        return round(k, 2), round(d, 2), signal

    @staticmethod
    def calculate_volatility(closes: List[float], period: int = 24) -> float:
        if len(closes) < period + 1:
            return 0.0

        returns = [(closes[i] - closes[i-1]) / closes[i-1]
                   for i in range(1, len(closes)) if closes[i-1] != 0]

        if len(returns) < period:
            return 0.0

        recent_returns = returns[-period:]
        mean_return = sum(recent_returns) / len(recent_returns)
        variance = sum((r - mean_return) ** 2 for r in recent_returns) / len(recent_returns)

        return round(variance ** 0.5, 6)

    @staticmethod
    def calculate_trend(closes: List[float], period: int) -> float:
        if len(closes) < period or closes[-period] == 0:
            return 0.0
        return round((closes[-1] - closes[-period]) / closes[-period] * 100, 4)

    @staticmethod
    def calculate_volume_ratio(volumes: List[float], period: int = 20) -> float:
        if len(volumes) < period + 1:
            return 1.0
        avg_volume = sum(volumes[-period-1:-1]) / period
        if avg_volume == 0:
            return 1.0
        return round(volumes[-1] / avg_volume, 2)

    @staticmethod
    def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 0.0

        true_ranges = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            true_ranges.append(tr)

        if len(true_ranges) < period:
            return 0.0

        return round(sum(true_ranges[-period:]) / period, 4)


class PolymarketTradesFetcher:
    """Fetches historical closed positions from Polymarket."""

    def __init__(self, wallet_address: str):
        self.wallet_address = wallet_address.lower()
        self.last_request = 0

    def _rate_limit(self):
        elapsed = time.time() - self.last_request
        if elapsed < POLYMARKET_RATE_LIMIT:
            time.sleep(POLYMARKET_RATE_LIMIT - elapsed)
        self.last_request = time.time()

    def fetch_closed_positions(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Fetch closed positions with P&L data."""
        self._rate_limit()

        try:
            response = requests.get(
                f"{POLYMARKET_DATA_API}/closed-positions",
                params={
                    "user": self.wallet_address,
                    "limit": limit,
                    "offset": offset,
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch closed positions: {e}")
            return []

    def fetch_all_closed_positions(self, max_positions: int = 20000) -> List[Dict]:
        """Fetch all historical closed positions."""
        all_positions = []
        seen_ids = set()
        offset = 0
        limit = 50  # API seems to cap at 50

        while len(all_positions) < max_positions:
            logger.info(f"Fetching closed positions... offset={offset}, total={len(all_positions)}")

            positions = self.fetch_closed_positions(limit=limit, offset=offset)

            if not positions:
                break

            # Deduplicate by conditionId
            new_count = 0
            for pos in positions:
                cid = pos.get("conditionId", "")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    all_positions.append(pos)
                    new_count += 1

            logger.info(f"  Got {len(positions)} positions, {new_count} new (total unique: {len(all_positions)})")

            # If we got no new positions, we've hit the end
            if new_count == 0:
                offset += limit  # Try next page anyway
                if offset > len(all_positions) + 1000:  # Safety limit
                    break
            else:
                offset += limit

            if len(positions) < limit:
                break

        logger.info(f"Fetched {len(all_positions)} unique closed positions")
        return all_positions


class TradeEnricher:
    """Enriches trades with market context."""

    def __init__(self):
        self.binance = BinanceHistoricalData()
        self.indicators = TechnicalIndicators()

    def _detect_asset(self, title: str) -> Optional[str]:
        """Detect asset from market title."""
        title_lower = title.lower()
        for asset, patterns in ASSET_PATTERNS.items():
            for pattern in patterns:
                if pattern in title_lower:
                    return asset
        return None

    def _parse_side(self, outcome: str) -> Optional[str]:
        """Parse side from outcome field."""
        outcome_lower = outcome.lower()
        if "up" in outcome_lower or "yes" in outcome_lower or "higher" in outcome_lower:
            return "UP"
        elif "down" in outcome_lower or "no" in outcome_lower or "lower" in outcome_lower:
            return "DOWN"
        return None

    def _get_session_info(self, timestamp: datetime) -> Tuple[bool, bool, bool, bool]:
        """Determine trading session based on UTC hour."""
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5
        is_asia = 0 <= hour < 8
        is_europe = 7 <= hour < 16
        is_us = 13 <= hour < 22
        return is_weekend, is_asia, is_europe, is_us

    def enrich_position(self, position: Dict) -> Optional[EnrichedTrade]:
        """Enrich a closed position with market context."""
        try:
            # Parse timestamp (seconds to datetime)
            timestamp_sec = position.get("timestamp")
            if not timestamp_sec:
                return None

            timestamp = datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)
            timestamp_ms = timestamp_sec * 1000

            # Detect asset
            title = position.get("title", "")
            asset = self._detect_asset(title)
            if not asset:
                logger.debug(f"Could not detect asset from: {title}")
                return None

            # Parse side (what they bet on)
            outcome = position.get("outcome", "")
            side = self._parse_side(outcome)
            if not side:
                logger.debug(f"Could not parse side from: {outcome}")
                return None

            # Determine win/loss from curPrice (1 = won, 0 = lost)
            cur_price = position.get("curPrice", 0)
            realized_pnl = position.get("realizedPnl", 0)

            if cur_price == 1:
                trade_outcome = "WIN"
            elif cur_price == 0:
                trade_outcome = "LOSS"
            elif realized_pnl > 0:
                trade_outcome = "WIN"
            elif realized_pnl < 0:
                trade_outcome = "LOSS"
            else:
                # Can't determine outcome
                return None

            # Get trade details
            entry_price = float(position.get("avgPrice", 0.5))
            total_shares = float(position.get("totalBought", 0))

            # Cost = shares * entry price
            cost_usd = total_shares * entry_price

            # Calculate actual P&L based on outcome
            # If won: profit = shares * (1 - entry_price)
            # If lost: loss = shares * entry_price (lost the cost)
            if trade_outcome == "WIN":
                calculated_pnl = total_shares * (1.0 - entry_price)
            else:
                calculated_pnl = -cost_usd

            # Note: API realizedPnl seems inflated, use our calculation
            # For reference: API says ${realized_pnl:.2f}, we calculate ${calculated_pnl:.2f}

            # Create enriched trade
            enriched = EnrichedTrade(
                trade_id=position.get("conditionId", str(timestamp_ms)),
                timestamp=timestamp.isoformat(),
                timestamp_ms=timestamp_ms,
                asset=asset,
                side=side,
                outcome=trade_outcome,
                entry_price=entry_price,
                size_usd=cost_usd,
                realized_pnl=calculated_pnl,  # Use our calculated P&L
            )

            # Fetch historical OHLCV data from Binance
            symbol = ASSET_TO_SYMBOL.get(asset, "BTCUSDT")
            ohlcv = self.binance.get_ohlcv_context(symbol, timestamp_ms, lookback_hours=48)

            if not ohlcv or len(ohlcv) < 5:
                logger.debug(f"Insufficient OHLCV data for {asset} at {timestamp}")
                # Still return with basic info, just no indicators
                return enriched

            # Extract price arrays
            closes = [k["close"] for k in ohlcv]
            highs = [k["high"] for k in ohlcv]
            lows = [k["low"] for k in ohlcv]
            volumes = [k["volume"] for k in ohlcv]

            # Asset price
            enriched.asset_price = closes[-1] if closes else 0.0

            # Other asset prices (with caching)
            enriched.btc_price = self.binance.get_price_at_time("BTCUSDT", timestamp_ms) or 0.0
            if asset != "ETH":
                enriched.eth_price = self.binance.get_price_at_time("ETHUSDT", timestamp_ms) or 0.0
            if asset != "SOL":
                enriched.sol_price = self.binance.get_price_at_time("SOLUSDT", timestamp_ms) or 0.0

            # Calculate indicators
            enriched.rsi_14 = self.indicators.calculate_rsi(closes, 14)
            enriched.rsi_7 = self.indicators.calculate_rsi(closes, 7)

            enriched.macd_histogram, enriched.macd_signal = self.indicators.calculate_macd(closes)
            enriched.bb_bandwidth, enriched.bb_position = self.indicators.calculate_bollinger_bands(closes)
            enriched.stoch_k, enriched.stoch_d, enriched.stoch_signal = self.indicators.calculate_stochastic(highs, lows, closes)

            # Moving averages
            enriched.sma_20 = self.indicators.calculate_sma(closes, 20)
            enriched.sma_50 = self.indicators.calculate_sma(closes, min(50, len(closes)))
            enriched.ema_12 = self.indicators.calculate_ema(closes, 12)
            enriched.ema_26 = self.indicators.calculate_ema(closes, 26)

            current_price = closes[-1]
            if enriched.sma_20 > 0:
                enriched.price_vs_sma20 = round((current_price - enriched.sma_20) / enriched.sma_20 * 100, 2)
            if enriched.sma_50 > 0:
                enriched.price_vs_sma50 = round((current_price - enriched.sma_50) / enriched.sma_50 * 100, 2)

            # Trends
            enriched.trend_1h = self.indicators.calculate_trend(closes, min(1, len(closes)-1))
            enriched.trend_4h = self.indicators.calculate_trend(closes, min(4, len(closes)-1))
            enriched.trend_1d = self.indicators.calculate_trend(closes, min(24, len(closes)-1))

            if enriched.trend_1d > 1:
                enriched.trend_direction = "up"
            elif enriched.trend_1d < -1:
                enriched.trend_direction = "down"
            else:
                enriched.trend_direction = "neutral"

            # Volatility
            enriched.volatility_1h = self.indicators.calculate_volatility(closes, min(1, len(closes)-1))
            enriched.volatility_24h = self.indicators.calculate_volatility(closes, min(24, len(closes)-1))
            enriched.atr_14 = self.indicators.calculate_atr(highs, lows, closes, min(14, len(closes)-1))

            # Volume
            enriched.volume_ratio = self.indicators.calculate_volume_ratio(volumes)
            enriched.volume_24h = sum(volumes[-24:]) if len(volumes) >= 24 else sum(volumes)

            # Time context
            enriched.hour_utc = timestamp.hour
            enriched.day_of_week = timestamp.weekday()
            is_weekend, is_asia, is_europe, is_us = self._get_session_info(timestamp)
            enriched.is_weekend = is_weekend
            enriched.is_asia_session = is_asia
            enriched.is_europe_session = is_europe
            enriched.is_us_session = is_us

            # Market structure
            if len(highs) >= 3:
                enriched.higher_high = highs[-1] > highs[-2] > highs[-3]
                enriched.lower_low = lows[-1] < lows[-2] < lows[-3]
                price_range = (max(highs[-3:]) - min(lows[-3:])) / current_price if current_price > 0 else 0
                enriched.consolidating = price_range < 0.02

            return enriched

        except Exception as e:
            logger.debug(f"Error enriching position: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Enrich historical trades with market data")
    parser.add_argument("--wallet", required=True, help="Wallet address to fetch trades for")
    parser.add_argument("--output", default="data/enriched_trades.json", help="Output file path")
    parser.add_argument("--max-positions", type=int, default=20000, help="Maximum positions to fetch")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing data if resuming
    existing_ids = set()
    existing_data = []
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing_data = json.load(f)
            existing_ids = {t["trade_id"] for t in existing_data}
            logger.info(f"Resuming with {len(existing_data)} existing enriched trades")

    # Fetch closed positions
    logger.info(f"Fetching closed positions for wallet: {args.wallet}")
    fetcher = PolymarketTradesFetcher(args.wallet)
    positions = fetcher.fetch_all_closed_positions(max_positions=args.max_positions)
    logger.info(f"Found {len(positions)} closed positions")

    # Filter to crypto Up/Down markets only
    crypto_positions = []
    for pos in positions:
        title = pos.get("title", "").lower()
        if any(pattern in title for patterns in ASSET_PATTERNS.values() for pattern in patterns):
            if "up" in title or "down" in title:
                crypto_positions.append(pos)

    logger.info(f"Filtered to {len(crypto_positions)} crypto Up/Down positions")

    # Enrich positions
    enricher = TradeEnricher()
    enriched_trades = existing_data.copy()

    new_count = 0
    skip_count = 0
    error_count = 0
    total = len(crypto_positions)

    for i, position in enumerate(crypto_positions):
        trade_id = position.get("conditionId", "")

        if trade_id in existing_ids:
            skip_count += 1
            continue

        if (i + 1) % 100 == 0 or i == 0:
            logger.info(f"Progress: {i+1}/{total} | New: {new_count} | Skipped: {skip_count} | Errors: {error_count}")

        enriched = enricher.enrich_position(position)

        if enriched:
            enriched_trades.append(asdict(enriched))
            existing_ids.add(enriched.trade_id)
            new_count += 1
        else:
            error_count += 1

        # Save periodically
        if new_count > 0 and new_count % 200 == 0:
            with open(output_path, "w") as f:
                json.dump(enriched_trades, f, indent=2)
            logger.info(f"Checkpoint: Saved {len(enriched_trades)} enriched trades")

    # Final save
    with open(output_path, "w") as f:
        json.dump(enriched_trades, f, indent=2)

    # Summary
    logger.info("=" * 60)
    logger.info("ENRICHMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total enriched trades: {len(enriched_trades)}")
    logger.info(f"New trades enriched: {new_count}")
    logger.info(f"Skipped (already processed): {skip_count}")
    logger.info(f"Errors (non-crypto or invalid): {error_count}")
    logger.info(f"Output saved to: {output_path}")

    # Stats
    if enriched_trades:
        wins = sum(1 for t in enriched_trades if t["outcome"] == "WIN")
        losses = sum(1 for t in enriched_trades if t["outcome"] == "LOSS")
        total_pnl = sum(t.get("realized_pnl", 0) for t in enriched_trades)

        if wins + losses > 0:
            logger.info(f"Win rate: {wins}/{wins+losses} = {wins/(wins+losses)*100:.1f}%")
        logger.info(f"Wins: {wins} | Losses: {losses}")
        logger.info(f"Total P&L: ${total_pnl:,.2f}")

        # Asset breakdown
        by_asset = {}
        for t in enriched_trades:
            asset = t["asset"]
            if asset not in by_asset:
                by_asset[asset] = {"wins": 0, "losses": 0, "pnl": 0}
            if t["outcome"] == "WIN":
                by_asset[asset]["wins"] += 1
            else:
                by_asset[asset]["losses"] += 1
            by_asset[asset]["pnl"] += t.get("realized_pnl", 0)

        logger.info("\nBy Asset:")
        for asset, stats in sorted(by_asset.items()):
            total = stats["wins"] + stats["losses"]
            wr = stats["wins"] / total * 100 if total > 0 else 0
            logger.info(f"  {asset}: {stats['wins']}W/{stats['losses']}L ({wr:.1f}%) | P&L: ${stats['pnl']:,.2f}")


if __name__ == "__main__":
    main()
