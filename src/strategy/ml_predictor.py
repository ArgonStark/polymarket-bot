"""
Machine Learning Signal Quality Predictor

Learns from trade history to predict which signals are likely to win.
Uses a simple but effective approach that works without heavy ML dependencies.
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeFeatures:
    """Features extracted from a trade signal."""
    edge: float  # Expected edge/profit margin
    time_remaining: float  # Seconds until market closes
    volatility: float  # Asset volatility estimate
    price_momentum: float  # Recent price change direction (-1 to 1)
    hour_of_day: int  # 0-23 UTC
    day_of_week: int  # 0-6 (Monday=0)
    asset: str  # BTC, ETH, SOL, XRP
    side: str  # UP or DOWN

    def to_vector(self) -> list[float]:
        """Convert to feature vector for model."""
        # Normalize features to roughly 0-1 range
        return [
            min(self.edge * 10, 1.0),  # edge 0.1 -> 1.0
            min(self.time_remaining / 900, 1.0),  # 900s -> 1.0
            min(self.volatility * 100, 1.0),  # 0.01 -> 1.0
            (self.price_momentum + 1) / 2,  # -1,1 -> 0,1
            self.hour_of_day / 24,
            self.day_of_week / 7,
            1.0 if self.asset == "BTC" else 0.0,
            1.0 if self.asset == "ETH" else 0.0,
            1.0 if self.asset == "SOL" else 0.0,
            1.0 if self.asset == "XRP" else 0.0,
            1.0 if self.side == "UP" else 0.0,
        ]

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "edge": self.edge,
            "time_remaining": self.time_remaining,
            "volatility": self.volatility,
            "price_momentum": self.price_momentum,
            "hour_of_day": self.hour_of_day,
            "day_of_week": self.day_of_week,
            "asset": self.asset,
            "side": self.side,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TradeFeatures":
        """Create from dictionary."""
        return cls(
            edge=data["edge"],
            time_remaining=data["time_remaining"],
            volatility=data["volatility"],
            price_momentum=data["price_momentum"],
            hour_of_day=data["hour_of_day"],
            day_of_week=data["day_of_week"],
            asset=data["asset"],
            side=data["side"],
        )


@dataclass
class SimpleLogisticRegression:
    """
    A simple logistic regression model that doesn't require sklearn.
    Uses online learning to update weights incrementally.
    """
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    learning_rate: float = 0.1
    n_features: int = 11  # Number of features in TradeFeatures.to_vector()

    def __post_init__(self):
        if not self.weights:
            # Initialize weights with small random values
            import random
            random.seed(42)
            self.weights = [random.uniform(-0.1, 0.1) for _ in range(self.n_features)]

    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation function."""
        # Clip to avoid overflow
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))

    def predict_proba(self, features: list[float]) -> float:
        """Predict probability of win (class 1)."""
        if len(features) != len(self.weights):
            logger.warning(f"Feature mismatch: {len(features)} vs {len(self.weights)}")
            return 0.5

        # Linear combination
        z = self.bias + sum(w * f for w, f in zip(self.weights, features))
        return self._sigmoid(z)

    def update(self, features: list[float], outcome: int):
        """
        Update weights based on a single training example.

        Args:
            features: Feature vector
            outcome: 1 for win, 0 for loss
        """
        if len(features) != len(self.weights):
            return

        # Prediction
        pred = self.predict_proba(features)

        # Error
        error = outcome - pred

        # Update weights (gradient descent)
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error * features[i]
        self.bias += self.learning_rate * error

    def to_dict(self) -> dict:
        """Serialize model."""
        return {
            "weights": self.weights,
            "bias": self.bias,
            "learning_rate": self.learning_rate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimpleLogisticRegression":
        """Deserialize model."""
        return cls(
            weights=data.get("weights", []),
            bias=data.get("bias", 0.0),
            learning_rate=data.get("learning_rate", 0.1),
        )


@dataclass
class MLSignalPredictor:
    """
    Machine Learning predictor for signal quality.

    Features:
    - Learns from historical trade outcomes
    - Uses online learning (updates after each trade)
    - Persists model to disk
    - Provides win probability predictions
    """

    model: SimpleLogisticRegression = field(default_factory=SimpleLogisticRegression)
    min_confidence: float = 0.55  # Minimum predicted win probability to trade
    min_training_samples: int = 10  # Minimum trades before using ML filter
    training_samples: int = 0
    model_path: str = "ml_model.json"

    # Performance tracking
    predictions_made: int = 0
    correct_predictions: int = 0

    def __post_init__(self):
        """Load model if exists."""
        self._load_model()

    def extract_features(
        self,
        signal,  # Signal object
        volatility: float,
        price_momentum: float = 0.0,
    ) -> TradeFeatures:
        """
        Extract features from a trading signal.

        Args:
            signal: Trading signal
            volatility: Asset volatility estimate
            price_momentum: Recent price direction (-1 to 1)
        """
        now = datetime.now(timezone.utc)

        return TradeFeatures(
            edge=signal.edge,
            time_remaining=signal.market.time_remaining,
            volatility=volatility,
            price_momentum=price_momentum,
            hour_of_day=now.hour,
            day_of_week=now.weekday(),
            asset=signal.market.asset,
            side=signal.side.value,
        )

    def should_trade(
        self,
        signal,
        volatility: float,
        price_momentum: float = 0.0,
    ) -> tuple[bool, float, str]:
        """
        Decide if we should take this trade based on ML prediction.

        Args:
            signal: Trading signal
            volatility: Asset volatility
            price_momentum: Recent price direction

        Returns:
            Tuple of (should_trade, confidence, reason)
        """
        # Not enough training data yet - allow trades to collect data
        if self.training_samples < self.min_training_samples:
            return (True, 0.5, f"Learning mode ({self.training_samples}/{self.min_training_samples} samples)")

        # Extract features and predict
        features = self.extract_features(signal, volatility, price_momentum)
        feature_vector = features.to_vector()
        confidence = self.model.predict_proba(feature_vector)

        if confidence >= self.min_confidence:
            return (True, confidence, f"ML confidence: {confidence:.0%}")
        else:
            return (False, confidence, f"ML rejected: {confidence:.0%} < {self.min_confidence:.0%}")

    def record_outcome(
        self,
        signal,
        volatility: float,
        price_momentum: float,
        won: bool,
    ):
        """
        Record a trade outcome and update the model.

        Args:
            signal: Original trading signal
            volatility: Asset volatility at trade time
            price_momentum: Price momentum at trade time
            won: Whether the trade was profitable
        """
        features = self.extract_features(signal, volatility, price_momentum)
        feature_vector = features.to_vector()

        # Get prediction before updating (for accuracy tracking)
        if self.training_samples >= self.min_training_samples:
            pred_prob = self.model.predict_proba(feature_vector)
            predicted_win = pred_prob >= 0.5
            if predicted_win == won:
                self.correct_predictions += 1
            self.predictions_made += 1

        # Update model with this example
        self.model.update(feature_vector, 1 if won else 0)
        self.training_samples += 1

        # Log learning progress
        if self.training_samples % 5 == 0:
            accuracy = self.correct_predictions / max(1, self.predictions_made)
            logger.info(
                f"🤖 ML Model: {self.training_samples} samples | "
                f"Accuracy: {accuracy:.0%} ({self.correct_predictions}/{self.predictions_made})"
            )

        # Save model periodically
        if self.training_samples % 10 == 0:
            self._save_model()

    def get_model_stats(self) -> dict:
        """Get model statistics."""
        accuracy = self.correct_predictions / max(1, self.predictions_made)
        return {
            "training_samples": self.training_samples,
            "predictions_made": self.predictions_made,
            "correct_predictions": self.correct_predictions,
            "accuracy": accuracy,
            "min_confidence": self.min_confidence,
            "is_active": self.training_samples >= self.min_training_samples,
        }

    def _save_model(self):
        """Save model to disk."""
        try:
            data = {
                "model": self.model.to_dict(),
                "training_samples": self.training_samples,
                "predictions_made": self.predictions_made,
                "correct_predictions": self.correct_predictions,
                "min_confidence": self.min_confidence,
            }
            with open(self.model_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"ML model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save ML model: {e}")

    def _load_model(self):
        """Load model from disk if exists."""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, "r") as f:
                    data = json.load(f)
                self.model = SimpleLogisticRegression.from_dict(data["model"])
                self.training_samples = data.get("training_samples", 0)
                self.predictions_made = data.get("predictions_made", 0)
                self.correct_predictions = data.get("correct_predictions", 0)
                self.min_confidence = data.get("min_confidence", 0.55)
                logger.info(
                    f"🤖 ML model loaded: {self.training_samples} samples, "
                    f"{self.correct_predictions}/{self.predictions_made} correct"
                )
        except Exception as e:
            logger.warning(f"Could not load ML model: {e}")


# Singleton instance
_ml_predictor: Optional[MLSignalPredictor] = None


def get_ml_predictor() -> MLSignalPredictor:
    """Get or create the ML predictor singleton."""
    global _ml_predictor
    if _ml_predictor is None:
        _ml_predictor = MLSignalPredictor()
    return _ml_predictor
