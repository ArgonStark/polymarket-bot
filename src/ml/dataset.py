"""Dataset loader for ML pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .feature_builder import FeatureBuilder


@dataclass
class DatasetSplit:
    X: np.ndarray
    y: np.ndarray
    entry_yes: np.ndarray
    entry_no: np.ndarray
    timestamps: np.ndarray


class Dataset:
    """Loads trade samples and builds feature matrix."""

    def __init__(self, feature_builder: FeatureBuilder):
        self.feature_builder = feature_builder

    def load(self, path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rows = []
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))

        features = []
        labels = []
        entry_yes = []
        entry_no = []
        timestamps = []

        for row in rows:
            snapshot, y, yes_price, no_price, ts = self._row_to_sample(row)
            feats = self.feature_builder.build(snapshot)
            features.append(self.feature_builder.to_numpy(feats))
            labels.append(y)
            entry_yes.append(yes_price)
            entry_no.append(no_price)
            timestamps.append(ts)

        X = np.vstack(features).astype(np.float32)
        y_arr = np.asarray(labels, dtype=np.int64)
        entry_yes_arr = np.asarray(entry_yes, dtype=np.float32)
        entry_no_arr = np.asarray(entry_no, dtype=np.float32)
        ts_arr = np.asarray(timestamps, dtype=np.float64)
        return X, y_arr, entry_yes_arr, entry_no_arr, ts_arr

    def split_time(
        self,
        X: np.ndarray,
        y: np.ndarray,
        entry_yes: np.ndarray,
        entry_no: np.ndarray,
        timestamps: np.ndarray,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
    ) -> Tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
        order = np.argsort(timestamps)
        X = X[order]
        y = y[order]
        entry_yes = entry_yes[order]
        entry_no = entry_no[order]
        timestamps = timestamps[order]

        n = len(X)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train = DatasetSplit(
            X=X[:n_train],
            y=y[:n_train],
            entry_yes=entry_yes[:n_train],
            entry_no=entry_no[:n_train],
            timestamps=timestamps[:n_train],
        )
        val = DatasetSplit(
            X=X[n_train : n_train + n_val],
            y=y[n_train : n_train + n_val],
            entry_yes=entry_yes[n_train : n_train + n_val],
            entry_no=entry_no[n_train : n_train + n_val],
            timestamps=timestamps[n_train : n_train + n_val],
        )
        test = DatasetSplit(
            X=X[n_train + n_val :],
            y=y[n_train + n_val :],
            entry_yes=entry_yes[n_train + n_val :],
            entry_no=entry_no[n_train + n_val :],
            timestamps=timestamps[n_train + n_val :],
        )
        return train, val, test

    def _row_to_sample(self, row: Dict[str, Any]) -> Tuple[Dict[str, Any], int, float, float, float]:
        ts = _to_ts(row.get("timestamp") or row.get("ts"))
        market_open = row.get("market_open_ts") or row.get("open_ts") or row.get("start_ts")
        market_end = row.get("market_end_ts") or row.get("end_ts") or row.get("resolution_ts")

        yes_bid = _num(row.get("polymarket_yes_bid") or row.get("yes_bid"))
        yes_ask = _num(row.get("polymarket_yes_ask") or row.get("yes_ask"))

        no_bid = row.get("polymarket_no_bid") or row.get("no_bid")
        no_ask = row.get("polymarket_no_ask") or row.get("no_ask")
        if no_bid is None or no_ask is None:
            no_bid = 1.0 - yes_ask
            no_ask = 1.0 - yes_bid

        orderbook_bids = row.get("orderbook_bids") or row.get("bids")
        orderbook_asks = row.get("orderbook_asks") or row.get("asks")

        reference_price = _num(row.get("reference_price") or row.get("ref_price"))
        reference_price_open = _num(row.get("reference_price_open") or row.get("ref_price_open") or row.get("target_price"))
        reference_price_history = row.get("reference_price_history") or row.get("ref_price_history")

        outcome = row.get("resolution") or row.get("outcome") or row.get("resolved")
        if isinstance(outcome, str):
            outcome = outcome.upper()
            y = 1 if outcome in ("UP", "YES", "TRUE", "1") else 0
        else:
            y = 1 if int(outcome) == 1 else 0

        snapshot = {
            "timestamp": ts,
            "market_open_ts": market_open,
            "market_end_ts": market_end,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "no_bid": no_bid,
            "no_ask": no_ask,
            "orderbook_bids": orderbook_bids,
            "orderbook_asks": orderbook_asks,
            "reference_price": reference_price,
            "reference_price_open": reference_price_open,
            "reference_price_history": reference_price_history,
        }

        entry_yes = float(yes_ask)
        entry_no = float(no_ask)
        return snapshot, y, entry_yes, entry_no, float(ts)


def _num(val: Any) -> float:
    return float(val)


def _to_ts(val: Any) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        return float(datetime.fromisoformat(val.replace("Z", "+00:00")).timestamp())
    raise ValueError("Invalid timestamp")
