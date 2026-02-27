"""
Regime detection using ATR percentile + Kaufman Efficiency Ratio.

Classifies market state as TRENDING, RANGING, or CHOPPY to gate trading.
Choppy markets are coin-flip environments — skip them.

ATR (Average True Range): Measures volatility amplitude.
Efficiency Ratio: ER = |net_move| / total_path_length.
  ER=1.0 → perfect trend (all movement in one direction)
  ER=0.0 → pure noise (lots of movement, no net progress)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.data.binance import BinanceFeed, fetch_klines

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    CHOPPY = "choppy"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """Result of regime classification for one asset."""
    regime: RegimeType
    efficiency_ratio: float  # 0-1
    atr_percentile: float    # 0-100
    should_trade: bool
    kelly_multiplier: float  # 0.0 - 1.0, applied to Kelly fraction
    detail: str = ""


@dataclass
class RegimeDetector:
    """
    Classifies market regime per asset using Binance price data.

    Uses:
    - 1m klines for ATR calculation (fetched from REST API, cached)
    - BinanceFeed._price_history for Efficiency Ratio (real-time WebSocket data)
    """

    binance_feed: BinanceFeed

    # Thresholds (can be overridden via config)
    er_trending_threshold: float = 0.60
    er_ranging_threshold: float = 0.35
    atr_low_percentile: float = 20.0
    ranging_kelly_mult: float = 0.50

    # Cache
    _cache: dict[str, RegimeState] = field(default_factory=dict)
    _cache_ts: dict[str, float] = field(default_factory=dict)
    _cache_ttl: float = 5.0  # seconds

    def get_regime(self, asset: str) -> RegimeState:
        """
        Get current regime for an asset. Cached for _cache_ttl seconds.

        Returns RegimeState with should_trade and kelly_multiplier.
        """
        now = time.monotonic()
        cached_ts = self._cache_ts.get(asset, 0.0)
        if now - cached_ts < self._cache_ttl and asset in self._cache:
            return self._cache[asset]

        regime = self._classify(asset)
        self._cache[asset] = regime
        self._cache_ts[asset] = now
        return regime

    def _classify(self, asset: str) -> RegimeState:
        """Compute regime classification for one asset."""
        # --- Efficiency Ratio from BinanceFeed price history ---
        er, er_direction = self._compute_efficiency_ratio(asset)

        # --- ATR percentile from 1m klines ---
        atr_pct = self._compute_atr_percentile(asset)

        # --- Classification ---
        if atr_pct is not None and atr_pct < self.atr_low_percentile:
            # Low volatility — unpredictable, coin-flip outcomes
            return RegimeState(
                regime=RegimeType.CHOPPY,
                efficiency_ratio=er,
                atr_percentile=atr_pct if atr_pct is not None else 50.0,
                should_trade=False,
                kelly_multiplier=0.0,
                detail=f"low_vol atr_pct={atr_pct:.0f}",
            )

        if er >= self.er_trending_threshold:
            regime_type = RegimeType.TRENDING_UP if er_direction > 0 else RegimeType.TRENDING_DOWN
            return RegimeState(
                regime=regime_type,
                efficiency_ratio=er,
                atr_percentile=atr_pct if atr_pct is not None else 50.0,
                should_trade=True,
                kelly_multiplier=1.0,
                detail=f"trending er={er:.2f} dir={'up' if er_direction > 0 else 'down'}",
            )

        if er >= self.er_ranging_threshold:
            return RegimeState(
                regime=RegimeType.RANGING,
                efficiency_ratio=er,
                atr_percentile=atr_pct if atr_pct is not None else 50.0,
                should_trade=True,
                kelly_multiplier=self.ranging_kelly_mult,
                detail=f"ranging er={er:.2f}",
            )

        # ER < ranging threshold AND high ATR = volatile but directionless
        if atr_pct is not None and atr_pct > 50:
            return RegimeState(
                regime=RegimeType.CHOPPY,
                efficiency_ratio=er,
                atr_percentile=atr_pct,
                should_trade=False,
                kelly_multiplier=0.0,
                detail=f"volatile_chop er={er:.2f} atr_pct={atr_pct:.0f}",
            )

        # Low ER + moderate vol = ranging with reduced size
        return RegimeState(
            regime=RegimeType.RANGING,
            efficiency_ratio=er,
            atr_percentile=atr_pct if atr_pct is not None else 50.0,
            should_trade=True,
            kelly_multiplier=self.ranging_kelly_mult,
            detail=f"weak_range er={er:.2f}",
        )

    def _compute_efficiency_ratio(self, asset: str, bars: int = 10) -> tuple[float, float]:
        """
        Compute Kaufman Efficiency Ratio from BinanceFeed price history.

        Returns (er, direction) where direction > 0 = net up, < 0 = net down.
        """
        history = self.binance_feed.get_price_history(asset, limit=bars + 1)
        if len(history) < 2:
            return 0.5, 0.0  # Unknown — neutral

        prices = [p for _, p in history]

        # Net move = |first - last|
        net_move = abs(prices[-1] - prices[0])
        direction = prices[-1] - prices[0]

        # Total path length = sum of |consecutive differences|
        path_length = sum(abs(prices[i + 1] - prices[i]) for i in range(len(prices) - 1))

        if path_length == 0:
            return 1.0, 0.0  # No movement at all — treat as flat

        er = net_move / path_length
        return er, direction

    def _compute_atr_percentile(self, asset: str, period: int = 14, lookback: int = 50) -> Optional[float]:
        """
        Compute ATR percentile from 1m klines.

        Returns percentile (0-100) of current ATR vs recent ATR values,
        or None if insufficient data.
        """
        klines = fetch_klines(asset, interval="1m", limit=lookback + period)
        if len(klines) < period + 1:
            return None

        # Compute True Range for each bar
        true_ranges = []
        for i in range(1, len(klines)):
            high = klines[i]["high"]
            low = klines[i]["low"]
            prev_close = klines[i - 1]["close"]
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            true_ranges.append(tr)

        if len(true_ranges) < period:
            return None

        # Compute rolling ATR values
        atr_values = []
        for i in range(len(true_ranges) - period + 1):
            window = true_ranges[i:i + period]
            atr_values.append(sum(window) / period)

        if not atr_values:
            return None

        # Current ATR is the last one
        current_atr = atr_values[-1]

        # Percentile rank
        below = sum(1 for a in atr_values if a < current_atr)
        percentile = (below / len(atr_values)) * 100.0

        return percentile
