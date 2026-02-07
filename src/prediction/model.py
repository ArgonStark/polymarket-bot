"""Gradient boosting probability model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
from sklearn.ensemble import GradientBoostingClassifier

from .features import PredictionFeatures


logger = logging.getLogger(__name__)


@dataclass
class PredictionOutput:
    """Prediction result."""

    prob_yes: float
    confidence: float


class ProbabilityModel:
    """Lightweight gradient boosting model with confidence output."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
        )
        self._is_fitted = False

        if self.model_path:
            self.load(self.model_path)

    def fit(self, X: list[list[float]], y: list[int]):
        self.model.fit(X, y)
        self._is_fitted = True
        logger.info(f"Probability model trained on {len(y)} samples")

    def predict(self, features: PredictionFeatures) -> PredictionOutput:
        if not self._is_fitted:
            logger.debug("Probability model not fitted; returning neutral prediction")
            return PredictionOutput(prob_yes=0.5, confidence=0.0)

        proba = self.model.predict_proba([features.as_vector()])[0]
        prob_yes = float(proba[1]) if len(proba) > 1 else float(proba[0])
        confidence = self._confidence_from_proba(prob_yes)
        return PredictionOutput(prob_yes=prob_yes, confidence=confidence)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Saved probability model to {path}")

    def load(self, path: str):
        if not Path(path).exists():
            logger.warning(f"Probability model not found at {path}")
            return
        self.model = joblib.load(path)
        self._is_fitted = True
        logger.info(f"Loaded probability model from {path}")

    @staticmethod
    def _confidence_from_proba(prob_yes: float) -> float:
        prob_yes = min(max(prob_yes, 1e-6), 1 - 1e-6)
        entropy = -(prob_yes * _log2(prob_yes) + (1 - prob_yes) * _log2(1 - prob_yes))
        return max(0.0, 1.0 - entropy)


def _log2(x: float) -> float:
    import math
    return math.log(x, 2)
