"""Inference for ML probability model."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

from .feature_builder import FeatureBuilder, FeatureSchema

logger = logging.getLogger(__name__)


@dataclass
class InferenceOutput:
    p_up: float
    uncertainty: float
    edge_up: float
    edge_down: float
    ev_up: float
    ev_down: float


class ModelBundle:
    """Loads model bundle and validates feature schema."""

    def __init__(self, bundle_path: str):
        self.bundle_path = Path(bundle_path)
        self.models = joblib.load(self.bundle_path / "model.bin")
        self.calibrators = joblib.load(self.bundle_path / "calibrator.bin")
        self.encoder = joblib.load(self.bundle_path / "encoder.bin")
        self.manifest = json.loads((self.bundle_path / "features_manifest.json").read_text())
        self.config = json.loads((self.bundle_path / "config.json").read_text())
        self.schema = FeatureSchema.from_manifest(self.manifest)
        self.builder = FeatureBuilder(
            depth_levels=self.manifest["depth_levels"],
            vol_window=self.manifest["vol_window"],
        )
        self._validate_schema()

    def _validate_schema(self) -> None:
        if self.schema.names() != self.builder.schema.names():
            raise ValueError("Feature schema mismatch")
        if self.schema.dtypes() != self.builder.schema.dtypes():
            raise ValueError("Feature dtype mismatch")


class InferenceEngine:
    """Predict probabilities and EV using a bundle."""

    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle

    def predict_proba(self, snapshot: Dict[str, Any]) -> Tuple[float, float]:
        features = self.bundle.builder.build(snapshot)
        vec = self.bundle.builder.to_numpy(features)
        # Wrap in DataFrame with feature names to avoid LightGBM warning:
        # "X does not have valid feature names, but LGBMClassifier was fitted with feature names"
        feature_names = self.bundle.schema.names()
        if _HAS_PANDAS:
            X = pd.DataFrame(vec.reshape(1, -1), columns=feature_names)
        else:
            X = vec.reshape(1, -1)
        preds = []
        for model, calibrator in zip(self.bundle.models, self.bundle.calibrators):
            proba = model.predict_proba(X)[:, 1][0]
            preds.append(float(calibrator.transform([proba])[0]))
        p_up = float(np.mean(preds))
        uncertainty = float(np.std(preds))
        return p_up, uncertainty

    def evaluate(self, snapshot: Dict[str, Any], fees_bps: float, slippage_bps: float) -> InferenceOutput:
        p_up, uncertainty = self.predict_proba(snapshot)
        yes_price = float(snapshot["yes_ask"])
        no_price = float(snapshot["no_ask"])
        fee = fees_bps / 10000.0
        slip = slippage_bps / 10000.0

        cost_yes = yes_price * (1 + fee + slip)
        cost_no = no_price * (1 + fee + slip)

        edge_up = p_up - yes_price
        edge_down = (1 - p_up) - no_price

        ev_up = p_up - cost_yes
        ev_down = (1 - p_up) - cost_no

        return InferenceOutput(
            p_up=p_up,
            uncertainty=uncertainty,
            edge_up=edge_up,
            edge_down=edge_down,
            ev_up=ev_up,
            ev_down=ev_down,
        )
