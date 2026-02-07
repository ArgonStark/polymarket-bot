import json
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pytest

from src.ml.feature_builder import FeatureBuilder
from src.ml.infer import ModelBundle


def _snapshot(depth=5):
    now = datetime.now(timezone.utc)
    open_ts = now - timedelta(minutes=5)
    end_ts = now + timedelta(minutes=10)

    bids = [(0.45 - i * 0.01, 100 + i) for i in range(depth)]
    asks = [(0.55 + i * 0.01, 100 + i) for i in range(depth)]

    history = [(open_ts + timedelta(seconds=i * 10), 100 + i) for i in range(30)]

    return {
        "timestamp": now,
        "market_open_ts": open_ts,
        "market_end_ts": end_ts,
        "yes_bid": 0.45,
        "yes_ask": 0.55,
        "no_bid": 0.45,
        "no_ask": 0.55,
        "orderbook_bids": bids,
        "orderbook_asks": asks,
        "reference_price": 110,
        "reference_price_open": 100,
        "reference_price_history": history,
    }


def test_feature_skew_ordering_and_dtype():
    builder = FeatureBuilder(depth_levels=5, vol_window=20)
    snap = _snapshot()
    feats = builder.build(snap)
    vec1 = builder.to_numpy(feats)
    vec2 = builder.to_numpy(feats)
    assert vec1.shape == vec2.shape
    assert vec1.dtype == np.float32
    assert np.allclose(vec1, vec2)


def test_no_leakage_future_history():
    builder = FeatureBuilder(depth_levels=5, vol_window=20)
    snap = _snapshot()
    snap["reference_price_history"].append((snap["timestamp"] + timedelta(seconds=60), 999))
    with pytest.raises(ValueError):
        builder.build(snap)


def test_bundle_schema_mismatch(tmp_path):
    builder = FeatureBuilder(depth_levels=5, vol_window=20)
    manifest = builder.manifest()
    manifest["features"].append({"name": "extra", "dtype": "float32", "categories": None})

    joblib.dump(["model"], tmp_path / "model.bin")
    joblib.dump(["cal"], tmp_path / "calibrator.bin")
    joblib.dump(None, tmp_path / "encoder.bin")
    (tmp_path / "features_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "config.json").write_text(json.dumps({"depth_levels": 5, "vol_window": 20}))

    with pytest.raises(ValueError):
        ModelBundle(str(tmp_path))
