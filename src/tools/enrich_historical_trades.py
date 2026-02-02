"""
Historical Trade Data Enrichment Tool

Fetches historical market data for past trades and enriches them with
technical indicators and market context for ML training.

Usage:
    python -m src.tools.enrich_historical_trades --wallet 0x63ce342161250d705dc0b16df89036c8e5f9ba9a

This tool:
1. Fetches historical trades from Polymarket Data API
2. For each trade, fetches Binance OHLCV data at that timestamp
3. Calculates all 82 ML features for that moment
4. Saves enriched dataset for ML training
"""

import os
import json
import time
import logging
import argparse
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
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
GAMMA_API = "https://gamma-api.polymarket.com"

# Asset to Binance symbol mapping
ASSET_TO_SYMBOL = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
}

# Rate limiting
BINANCE_RATE_LIMIT = 0.1  # seconds between requests
POLYMARKET_RATE_LIMIT = 0.2


@dataclass
class EnrichedTrade:
    """A trade enriched with market context for ML training."""

    # Basic trade info
    trade_id: str
    timestamp: str
    asset: str
    side: str  # "UP" or "DOWN"
    outcome: str  # "WIN", "LOSS", or "PENDING"
    entry_price: float
    size_usd: float

    # Price context
    btc_price: float = 0.0
    eth_price: float = 0.0
    sol_price: float = 0.0
    asset_price: float = 0.0

    # Technical indicators (at trade entry time)
    rsi_14: float = 50.0
    rsi_7: float = 50.0
    macd_histogram: float = 0.0
    macd_signal: str = "none"  # bullish_cross, bearish_cross, none

    # Bollinger Bands
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_middle: float = 0.0
    bb_bandwidth: float = 0.0
    bb_position: str = "middle"  # above, below, middle

    # Stochastic
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    stoch_signal: str = "neutral"  # overbought, oversold, neutral

    # Moving averages
    sma_20: float = 0.0
    sma_50: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    price_vs_sma20: float = 0.0  # % above/below
    price_vs_sma50: float = 0.0

    # Trend
    trend_1h: float = 0.0  # % change
    trend_4h: float = 0.0
    trend_1d: float = 0.0
    trend_direction: str = "neutral"  # up, down, neutral

    # Volatility
    volatility_1h: float = 0.0
    volatility_24h: float = 0.0
    atr_14: float = 0.0

    # Volume
    volume_ratio: float = 1.0  # vs 20-period average
    volume_24h: float = 0.0
    obv_trend: float = 0.0

    # Candlestick patterns
    candle_pattern: str = "none"
    heiken_ashi_trend: str = "neutral"
    ha_consecutive: int = 0

    # VWAP
    vwap: float = 0.0
    price_vs_vwap: float = 0.0

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

    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
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
        """
        Fetch historical klines from Binance.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval (e.g., "1m", "5m", "1h")
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Max number of klines

        Returns:
            List of kline dictionaries
        """
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

            # Parse klines into dictionaries
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
                    "quote_volume": float(k[7]),
                    "trades": k[8],
                })

            self.cache[cache_key] = klines
            return klines

        except Exception as e:
            logger.warning(f"Failed to fetch klines for {symbol}: {e}")
            return []

    def get_price_at_time(self, symbol: str, timestamp_ms: int) -> Optional[float]:
        """Get the price at a specific timestamp."""
        klines = self.get_klines(
            symbol=symbol,
            interval="1m",
            start_time=timestamp_ms - 60000,  # 1 minute before
            end_time=timestamp_ms + 60000,    # 1 minute after
            limit=3
        )

        if klines:
            return klines[0]["close"]
        return None

    def get_ohlcv_context(
        self,
        symbol: str,
        timestamp_ms: int,
        lookback_hours: int = 24
    ) -> List[Dict]:
        """Get OHLCV data for context around a timestamp."""
        start_time = timestamp_ms - (lookback_hours * 60 * 60 * 1000)

        # Fetch hourly data for longer context
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
        """Calculate RSI."""
        if len(closes) < period + 1:
            return 50.0

        changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [c if c > 0 else 0 for c in changes]
        losses = [-c if c < 0 else 0 for c in changes]

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return round(rsi, 2)

    @staticmethod
    def calculate_sma(values: List[float], period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(values) < period:
            return values[-1] if values else 0.0
        return sum(values[-period:]) / period

    @staticmethod
    def calculate_ema(values: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(values) < period:
            return values[-1] if values else 0.0

        multiplier = 2 / (period + 1)
        ema = sum(values[:period]) / period

        for price in values[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    @staticmethod
    def calculate_macd(closes: List[float]) -> tuple[float, float, str]:
        """Calculate MACD histogram and signal."""
        if len(closes) < 26:
            return 0.0, 0.0, "none"

        ema12 = TechnicalIndicators.calculate_ema(closes, 12)
        ema26 = TechnicalIndicators.calculate_ema(closes, 26)
        macd_line = ema12 - ema26

        # Calculate signal line (9-period EMA of MACD)
        # Simplified: just use current MACD as approximation
        signal_line = macd_line * 0.9  # Rough approximation

        histogram = macd_line - signal_line

        # Determine signal
        signal = "none"
        if histogram > 0 and macd_line > 0:
            signal = "bullish_cross"
        elif histogram < 0 and macd_line < 0:
            signal = "bearish_cross"

        return round(histogram, 6), round(macd_line, 6), signal

    @staticmethod
    def calculate_bollinger_bands(
        closes: List[float],
        period: int = 20,
        std_dev: float = 2.0
    ) -> tuple[float, float, float, float, str]:
        """Calculate Bollinger Bands."""
        if len(closes) < period:
            price = closes[-1] if closes else 0.0
            return price, price, price, 0.0, "middle"

        sma = sum(closes[-period:]) / period

        # Calculate standard deviation
        variance = sum((p - sma) ** 2 for p in closes[-period:]) / period
        std = variance ** 0.5

        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)

        bandwidth = (upper - lower) / sma if sma > 0 else 0.0

        # Current price position
        current_price = closes[-1]
        if current_price > upper:
            position = "above"
        elif current_price < lower:
            position = "below"
        else:
            position = "middle"

        return round(upper, 2), round(lower, 2), round(sma, 2), round(bandwidth, 4), position

    @staticmethod
    def calculate_stochastic(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        k_period: int = 14,
        d_period: int = 3
    ) -> tuple[float, float, str]:
        """Calculate Stochastic oscillator."""
        if len(closes) < k_period:
            return 50.0, 50.0, "neutral"

        highest_high = max(highs[-k_period:])
        lowest_low = min(lows[-k_period:])

        if highest_high == lowest_low:
            k = 50.0
        else:
            k = 100 * (closes[-1] - lowest_low) / (highest_high - lowest_low)

        # D is SMA of K (simplified)
        d = k  # Simplified

        # Signal
        if k > 80:
            signal = "overbought"
        elif k < 20:
            signal = "oversold"
        else:
            signal = "neutral"

        return round(k, 2), round(d, 2), signal

    @staticmethod
    def calculate_atr(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14
    ) -> float:
        """Calculate Average True Range."""
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

        return sum(true_ranges[-period:]) / period

    @staticmethod
    def calculate_volatility(closes: List[float], period: int = 24) -> float:
        """Calculate volatility as standard deviation of returns."""
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
        """Calculate trend as percentage change over period."""
        if len(closes) < period or closes[-period] == 0:
            return 0.0

        return round((closes[-1] - closes[-period]) / closes[-period] * 100, 4)

    @staticmethod
    def calculate_volume_ratio(volumes: List[float], period: int = 20) -> float:
        """Calculate volume ratio vs average."""
        if len(volumes) < period + 1:
            return 1.0

        avg_volume = sum(volumes[-period-1:-1]) / period
        if avg_volume == 0:
            return 1.0

        return round(volumes[-1] / avg_volume, 2)


class PolymarketTradesFetcher:
    """Fetches historical trades from Polymarket."""

    def __init__(self, wallet_address: str):
        self.wallet_address = wallet_address.lower()
        self.last_request = 0

    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        elapsed = time.time() - self.last_request
        if elapsed < POLYMARKET_RATE_LIMIT:
            time.sleep(POLYMARKET_RATE_LIMIT - elapsed)
        self.last_request = time.time()

    def fetch_activity(
        self,
        activity_type: str = "TRADE",
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """Fetch activity from Polymarket Data API."""
        self._rate_limit()

        try:
            response = requests.get(
                f"{POLYMARKET_DATA_API}/activity",
                params={
                    "user": self.wallet_address,
                    "type": activity_type,
                    "limit": limit,
                    "offset": offset,
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch activity: {e}")
            return []

    def fetch_all_trades(self, max_trades: int = 15000) -> List[Dict]:
        """Fetch all historical trades."""
        all_trades = []
        offset = 0
        limit = 100

        while len(all_trades) < max_trades:
            logger.info(f"Fetching trades... offset={offset}, total={len(all_trades)}")

            trades = self.fetch_activity("TRADE", limit=limit, offset=offset)

            if not trades:
                break

            all_trades.extend(trades)
            offset += limit

            if len(trades) < limit:
                break

        logger.info(f"Fetched {len(all_trades)} total trades")
        return all_trades

    def fetch_closed_positions(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Fetch closed positions with realized P&L."""
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


class TradeEnricher:
    """Enriches trades with market context."""

    def __init__(self):
        self.binance = BinanceHistoricalData()
        self.indicators = TechnicalIndicators()

    def _parse_asset_from_market(self, market_name: str) -> Optional[str]:
        """Extract asset (BTC, ETH, etc.) from market name."""
        market_upper = market_name.upper()

        for asset in ["BTC", "ETH", "SOL", "XRP"]:
            if asset in market_upper:
                return asset

        return None

    def _parse_side_from_outcome(self, outcome: str) -> Optional[str]:
        """Parse side (UP/DOWN) from outcome string."""
        outcome_upper = outcome.upper()

        if "YES" in outcome_upper or "UP" in outcome_upper or "HIGHER" in outcome_upper:
            return "UP"
        elif "NO" in outcome_upper or "DOWN" in outcome_upper or "LOWER" in outcome_upper:
            return "DOWN"

        return None

    def _get_session_info(self, timestamp: datetime) -> tuple[bool, bool, bool, bool]:
        """Determine trading session based on UTC hour."""
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5

        # Asia: 00:00 - 08:00 UTC
        is_asia = 0 <= hour < 8

        # Europe: 07:00 - 16:00 UTC
        is_europe = 7 <= hour < 16

        # US: 13:00 - 22:00 UTC
        is_us = 13 <= hour < 22

        return is_weekend, is_asia, is_europe, is_us

    def enrich_trade(self, trade: Dict) -> Optional[EnrichedTrade]:
        """Enrich a single trade with market context."""
        try:
            # Parse basic trade info
            timestamp_str = trade.get("timestamp") or trade.get("createdAt")
            if not timestamp_str:
                return None

            # Parse timestamp
            try:
                if "T" in timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                else:
                    timestamp = datetime.fromtimestamp(int(timestamp_str) / 1000, tz=timezone.utc)
            except:
                return None

            timestamp_ms = int(timestamp.timestamp() * 1000)

            # Parse asset
            market_name = trade.get("title") or trade.get("market", {}).get("question", "")
            asset = self._parse_asset_from_market(market_name)
            if not asset:
                return None

            # Parse side
            outcome = trade.get("outcome") or trade.get("side") or ""
            side = self._parse_side_from_outcome(outcome)
            if not side:
                return None

            # Determine win/loss
            trade_type = trade.get("type", "").upper()
            realized_pnl = trade.get("realizedPnl") or trade.get("pnl")

            if realized_pnl is not None:
                outcome_result = "WIN" if float(realized_pnl) > 0 else "LOSS"
            elif trade_type == "REDEEM":
                outcome_result = "WIN"  # Redeems are usually wins
            else:
                outcome_result = "PENDING"

            # Get entry price and size
            entry_price = float(trade.get("price") or trade.get("avgPrice") or 0.5)
            size_usd = float(trade.get("usdcSize") or trade.get("size") or 0)

            # Create enriched trade
            enriched = EnrichedTrade(
                trade_id=trade.get("id") or trade.get("transactionHash") or str(timestamp_ms),
                timestamp=timestamp.isoformat(),
                asset=asset,
                side=side,
                outcome=outcome_result,
                entry_price=entry_price,
                size_usd=size_usd,
            )

            # Fetch historical data
            symbol = ASSET_TO_SYMBOL.get(asset, "BTCUSDT")
            ohlcv = self.binance.get_ohlcv_context(symbol, timestamp_ms, lookback_hours=48)

            if not ohlcv:
                logger.warning(f"No OHLCV data for {asset} at {timestamp}")
                return enriched

            # Extract price arrays
            closes = [k["close"] for k in ohlcv]
            highs = [k["high"] for k in ohlcv]
            lows = [k["low"] for k in ohlcv]
            volumes = [k["volume"] for k in ohlcv]

            # Get prices for major assets
            enriched.asset_price = closes[-1] if closes else 0.0
            enriched.btc_price = self.binance.get_price_at_time("BTCUSDT", timestamp_ms) or 0.0
            enriched.eth_price = self.binance.get_price_at_time("ETHUSDT", timestamp_ms) or 0.0
            enriched.sol_price = self.binance.get_price_at_time("SOLUSDT", timestamp_ms) or 0.0

            # Calculate indicators
            enriched.rsi_14 = self.indicators.calculate_rsi(closes, 14)
            enriched.rsi_7 = self.indicators.calculate_rsi(closes, 7)

            macd_hist, macd_line, macd_sig = self.indicators.calculate_macd(closes)
            enriched.macd_histogram = macd_hist
            enriched.macd_signal = macd_sig

            bb_upper, bb_lower, bb_mid, bb_bw, bb_pos = self.indicators.calculate_bollinger_bands(closes)
            enriched.bb_upper = bb_upper
            enriched.bb_lower = bb_lower
            enriched.bb_middle = bb_mid
            enriched.bb_bandwidth = bb_bw
            enriched.bb_position = bb_pos

            stoch_k, stoch_d, stoch_sig = self.indicators.calculate_stochastic(highs, lows, closes)
            enriched.stoch_k = stoch_k
            enriched.stoch_d = stoch_d
            enriched.stoch_signal = stoch_sig

            # Moving averages
            enriched.sma_20 = self.indicators.calculate_sma(closes, 20)
            enriched.sma_50 = self.indicators.calculate_sma(closes, 50) if len(closes) >= 50 else enriched.sma_20
            enriched.ema_12 = self.indicators.calculate_ema(closes, 12)
            enriched.ema_26 = self.indicators.calculate_ema(closes, 26)

            current_price = closes[-1]
            if enriched.sma_20 > 0:
                enriched.price_vs_sma20 = (current_price - enriched.sma_20) / enriched.sma_20 * 100
            if enriched.sma_50 > 0:
                enriched.price_vs_sma50 = (current_price - enriched.sma_50) / enriched.sma_50 * 100

            # Trends
            enriched.trend_1h = self.indicators.calculate_trend(closes, 1)
            enriched.trend_4h = self.indicators.calculate_trend(closes, 4)
            enriched.trend_1d = self.indicators.calculate_trend(closes, 24)

            if enriched.trend_1d > 1:
                enriched.trend_direction = "up"
            elif enriched.trend_1d < -1:
                enriched.trend_direction = "down"
            else:
                enriched.trend_direction = "neutral"

            # Volatility
            enriched.volatility_1h = self.indicators.calculate_volatility(closes, 1)
            enriched.volatility_24h = self.indicators.calculate_volatility(closes, 24)
            enriched.atr_14 = self.indicators.calculate_atr(highs, lows, closes, 14)

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

            # Market structure (simplified)
            if len(highs) >= 3:
                enriched.higher_high = highs[-1] > highs[-2] > highs[-3]
                enriched.lower_low = lows[-1] < lows[-2] < lows[-3]
                enriched.consolidating = (max(highs[-3:]) - min(lows[-3:])) / current_price < 0.02

            return enriched

        except Exception as e:
            logger.error(f"Error enriching trade: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Enrich historical trades with market data")
    parser.add_argument("--wallet", required=True, help="Wallet address to fetch trades for")
    parser.add_argument("--output", default="data/enriched_trades.json", help="Output file path")
    parser.add_argument("--max-trades", type=int, default=15000, help="Maximum trades to fetch")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing data if resuming
    existing_trade_ids = set()
    existing_data = []
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing_data = json.load(f)
            existing_trade_ids = {t["trade_id"] for t in existing_data}
            logger.info(f"Resuming with {len(existing_data)} existing enriched trades")

    # Fetch trades
    logger.info(f"Fetching trades for wallet: {args.wallet}")
    fetcher = PolymarketTradesFetcher(args.wallet)

    trades = fetcher.fetch_all_trades(max_trades=args.max_trades)
    logger.info(f"Found {len(trades)} raw trades")

    # Also fetch closed positions for outcome data
    closed_positions = []
    offset = 0
    while True:
        batch = fetcher.fetch_closed_positions(limit=100, offset=offset)
        if not batch:
            break
        closed_positions.extend(batch)
        offset += 100
        if len(batch) < 100:
            break

    logger.info(f"Found {len(closed_positions)} closed positions")

    # Enrich trades
    enricher = TradeEnricher()
    enriched_trades = existing_data.copy()

    new_count = 0
    skip_count = 0
    error_count = 0

    all_items = trades + closed_positions
    total = len(all_items)

    for i, trade in enumerate(all_items):
        trade_id = trade.get("id") or trade.get("transactionHash") or ""

        if trade_id in existing_trade_ids:
            skip_count += 1
            continue

        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i+1}/{total} | New: {new_count} | Skipped: {skip_count} | Errors: {error_count}")

        enriched = enricher.enrich_trade(trade)

        if enriched:
            enriched_trades.append(asdict(enriched))
            existing_trade_ids.add(enriched.trade_id)
            new_count += 1
        else:
            error_count += 1

        # Save periodically
        if new_count > 0 and new_count % 100 == 0:
            with open(output_path, "w") as f:
                json.dump(enriched_trades, f, indent=2)
            logger.info(f"Saved {len(enriched_trades)} enriched trades to {output_path}")

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
    logger.info(f"Errors: {error_count}")
    logger.info(f"Output saved to: {output_path}")

    # Stats
    if enriched_trades:
        wins = sum(1 for t in enriched_trades if t["outcome"] == "WIN")
        losses = sum(1 for t in enriched_trades if t["outcome"] == "LOSS")
        pending = sum(1 for t in enriched_trades if t["outcome"] == "PENDING")

        logger.info(f"Win rate: {wins}/{wins+losses} = {wins/(wins+losses)*100:.1f}%" if wins+losses > 0 else "N/A")
        logger.info(f"Wins: {wins} | Losses: {losses} | Pending: {pending}")


if __name__ == "__main__":
    main()
