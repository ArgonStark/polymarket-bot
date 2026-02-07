"""Feature builder for ML pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: str
    categories: Optional[List[str]] = None


@dataclass
class FeatureSchema:
    features: List[FeatureSpec]

    def names(self) -> List[str]:
        return [f.name for f in self.features]

    def dtypes(self) -> List[str]:
        return [f.dtype for f in self.features]

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "features": [
                {"name": f.name, "dtype": f.dtype, "categories": f.categories}
                for f in self.features
            ]
        }

    @classmethod
    def from_manifest(cls, manifest: Dict[str, Any]) -> "FeatureSchema":
        feats = []
        for item in manifest["features"]:
            feats.append(FeatureSpec(item["name"], item["dtype"], item.get("categories")))
        return cls(feats)


class FeatureBuilder:
    """Builds features for training and inference with strict schema."""

    def __init__(self, depth_levels: int = 5, vol_window: int = 20):
        self.depth_levels = depth_levels
        self.vol_window = vol_window
        self.schema = self._build_schema(depth_levels)

    def build(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        required = [
            "timestamp",
            "market_open_ts",
            "market_end_ts",
            "yes_bid",
            "yes_ask",
            "no_bid",
            "no_ask",
            "orderbook_bids",
            "orderbook_asks",
            "reference_price",
            "reference_price_open",
            "reference_price_history",
        ]
        for key in required:
            if key not in snapshot:
                raise ValueError(f"Missing snapshot field: {key}")

        ts = _to_dt(snapshot["timestamp"])
        open_ts = _to_dt(snapshot["market_open_ts"])
        end_ts = _to_dt(snapshot["market_end_ts"])
        if end_ts <= open_ts:
            raise ValueError("market_end_ts must be after market_open_ts")

        yes_bid = _to_float(snapshot["yes_bid"])
        yes_ask = _to_float(snapshot["yes_ask"])
        no_bid = _to_float(snapshot["no_bid"])
        no_ask = _to_float(snapshot["no_ask"])
        if yes_bid <= 0 or yes_ask <= 0 or no_bid <= 0 or no_ask <= 0:
            raise ValueError("Invalid bid/ask prices")

        orderbook_bids = snapshot["orderbook_bids"]
        orderbook_asks = snapshot["orderbook_asks"]
        if len(orderbook_bids) < self.depth_levels or len(orderbook_asks) < self.depth_levels:
            raise ValueError("Insufficient orderbook depth")

        ref_price = _to_float(snapshot["reference_price"])
        ref_open = _to_float(snapshot["reference_price_open"])
        if ref_price <= 0 or ref_open <= 0:
            raise ValueError("Invalid reference prices")

        ref_hist = snapshot["reference_price_history"]
        ref_hist = _validate_history(ref_hist, ts)

        mid = (yes_bid + yes_ask) / 2
        spread = yes_ask - yes_bid

        bid_depth = sum(_to_float(s) for _, s in orderbook_bids[: self.depth_levels])
        ask_depth = sum(_to_float(s) for _, s in orderbook_asks[: self.depth_levels])
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0.0

        seconds_to_resolution = (end_ts - ts).total_seconds()
        seconds_since_open = (ts - open_ts).total_seconds()
        time_progress_pct = seconds_since_open / (end_ts - open_ts).total_seconds()

        ref_return_since_open = (ref_price - ref_open) / ref_open
        returns = _returns_from_history(ref_hist)
        vol = np.std(returns[-self.vol_window :]) if len(returns) > 1 else 0.0
        ref_return_z = ref_return_since_open / vol if vol > 0 else 0.0

        recent_prices = [p for _, p in ref_hist[-self.vol_window :]]
        mean_recent = float(np.mean(recent_prices)) if recent_prices else ref_price
        std_recent = float(np.std(recent_prices)) if len(recent_prices) > 1 else 0.0
        ref_price_z = (ref_price - mean_recent) / std_recent if std_recent > 0 else 0.0

        features: Dict[str, Any] = {
            "polymarket_yes_bid": yes_bid,
            "polymarket_yes_ask": yes_ask,
            "polymarket_no_bid": no_bid,
            "polymarket_no_ask": no_ask,
            "polymarket_mid": mid,
            "polymarket_spread": spread,
            "orderbook_bid_depth": bid_depth,
            "orderbook_ask_depth": ask_depth,
            "orderbook_imbalance": imbalance,
            "seconds_to_resolution": seconds_to_resolution,
            "seconds_since_open": seconds_since_open,
            "time_progress_pct": time_progress_pct,
            "reference_return_since_open": ref_return_since_open,
            "reference_return_zscore": ref_return_z,
            "reference_price_zscore": ref_price_z,
        }

        for i in range(self.depth_levels):
            bp, bs = orderbook_bids[i]
            ap, a_s = orderbook_asks[i]
            features[f"yes_bid_px_{i}"] = _to_float(bp)
            features[f"yes_bid_sz_{i}"] = _to_float(bs)
            features[f"yes_ask_px_{i}"] = _to_float(ap)
            features[f"yes_ask_sz_{i}"] = _to_float(a_s)

        self._validate_features(features)
        return features

    def to_numpy(self, features: Dict[str, Any]) -> np.ndarray:
        self._validate_features(features)
        values = []
        for spec in self.schema.features:
            values.append(_cast(features[spec.name], spec.dtype))
        return np.asarray(values, dtype=np.float32)

    def manifest(self) -> Dict[str, Any]:
        data = self.schema.to_manifest()
        data["depth_levels"] = self.depth_levels
        data["vol_window"] = self.vol_window
        return data

    def _validate_features(self, features: Dict[str, Any]) -> None:
        names = self.schema.names()
        missing = [n for n in names if n not in features]
        extra = [n for n in features.keys() if n not in names]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        if extra:
            raise ValueError(f"Unexpected features: {extra}")

    @staticmethod
    def _build_schema(depth_levels: int) -> FeatureSchema:
        base = [
            FeatureSpec("polymarket_yes_bid", "float32"),
            FeatureSpec("polymarket_yes_ask", "float32"),
            FeatureSpec("polymarket_no_bid", "float32"),
            FeatureSpec("polymarket_no_ask", "float32"),
            FeatureSpec("polymarket_mid", "float32"),
            FeatureSpec("polymarket_spread", "float32"),
            FeatureSpec("orderbook_bid_depth", "float32"),
            FeatureSpec("orderbook_ask_depth", "float32"),
            FeatureSpec("orderbook_imbalance", "float32"),
            FeatureSpec("seconds_to_resolution", "float32"),
            FeatureSpec("seconds_since_open", "float32"),
            FeatureSpec("time_progress_pct", "float32"),
            FeatureSpec("reference_return_since_open", "float32"),
            FeatureSpec("reference_return_zscore", "float32"),
            FeatureSpec("reference_price_zscore", "float32"),
        ]

        for i in range(depth_levels):
            base.extend(
                [
                    FeatureSpec(f"yes_bid_px_{i}", "float32"),
                    FeatureSpec(f"yes_bid_sz_{i}", "float32"),
                    FeatureSpec(f"yes_ask_px_{i}", "float32"),
                    FeatureSpec(f"yes_ask_sz_{i}", "float32"),
                ]
            )

        return FeatureSchema(base)


def _to_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    raise ValueError("Invalid timestamp")


def _to_float(value: Any) -> float:
    return float(value)


def _cast(value: Any, dtype: str) -> Any:
    if dtype.startswith("float"):
        return float(value)
    if dtype.startswith("int"):
        return int(value)
    return value


def _validate_history(history: Iterable[Any], ts: datetime) -> List[Tuple[datetime, float]]:
    parsed: List[Tuple[datetime, float]] = []
    for item in history:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            t = _to_dt(item[0])
            p = _to_float(item[1])
        elif isinstance(item, dict):
            t = _to_dt(item["ts"])
            p = _to_float(item["price"])
        else:
            raise ValueError("Invalid history item")
        if t > ts:
            raise ValueError("History contains future timestamp")
        parsed.append((t, p))
    if not parsed:
        raise ValueError("Empty reference_price_history")
    return parsed


def _returns_from_history(history: List[Tuple[datetime, float]]) -> List[float]:
    returns = []
    for i in range(1, len(history)):
        prev = history[i - 1][1]
        curr = history[i][1]
        if prev > 0:
            returns.append((curr - prev) / prev)
    return returns
