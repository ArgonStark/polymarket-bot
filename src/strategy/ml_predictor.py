"""
Machine Learning Signal Quality Predictor

Learns from trade history to predict which signals are likely to win.
Uses a simple but effective approach that works without heavy ML dependencies.

ENHANCED: Now includes features for:
- Arbitrage signal types (binary, asymmetric, dump, hedge)
- Market spread and order book depth
- Price trend from historical data
- Distance from target price
- Better integration with all bot components
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

# Arbitrage signal types for one-hot encoding
ARB_TYPES = ["none", "binary_arb", "asymmetric", "dump", "hedge"]


@dataclass
class TradeFeatures:
    """
    Features extracted from a trade signal.

    ENHANCED with additional features:
    - arb_type: Type of arbitrage opportunity detected
    - spread: Market bid-ask spread
    - bid_depth: Order book bid depth
    - ask_depth: Order book ask depth
    - price_trend: Short-term price trend from historical data
    - distance_from_target: How far current price is from target
    """
    # Core features
    edge: float  # Expected edge/profit margin
    time_remaining: float  # Seconds until market closes
    volatility: float  # Asset volatility estimate
    price_momentum: float  # Recent price change direction (-1 to 1)
    hour_of_day: int  # 0-23 UTC
    day_of_week: int  # 0-6 (Monday=0)
    asset: str  # BTC, ETH, SOL, XRP
    side: str  # UP or DOWN

    # NEW: Enhanced features
    arb_type: str = "none"  # Type of arbitrage (none, binary_arb, asymmetric, dump, hedge)
    spread: float = 0.0  # Market bid-ask spread
    bid_depth: float = 0.0  # Order book bid depth
    ask_depth: float = 0.0  # Order book ask depth
    price_trend: float = 0.0  # Short-term price trend (-1 to 1)
    distance_from_target: float = 0.0  # Normalized distance from target price

    def to_vector(self) -> list[float]:
        """Convert to feature vector for model (21 features total)."""
        # Normalize features to roughly 0-1 range
        vector = [
            # Core features (6)
            min(self.edge * 10, 1.0),  # edge 0.1 -> 1.0
            min(self.time_remaining / 900, 1.0),  # 900s -> 1.0
            min(self.volatility * 100, 1.0),  # 0.01 -> 1.0
            (self.price_momentum + 1) / 2,  # -1,1 -> 0,1
            self.hour_of_day / 24,
            self.day_of_week / 7,
            # Asset one-hot (4)
            1.0 if self.asset == "BTC" else 0.0,
            1.0 if self.asset == "ETH" else 0.0,
            1.0 if self.asset == "SOL" else 0.0,
            1.0 if self.asset == "XRP" else 0.0,
            # Side (1)
            1.0 if self.side == "UP" else 0.0,
            # NEW: Arb type one-hot (5)
            1.0 if self.arb_type == "none" else 0.0,
            1.0 if self.arb_type == "binary_arb" else 0.0,
            1.0 if self.arb_type == "asymmetric" else 0.0,
            1.0 if self.arb_type == "dump" else 0.0,
            1.0 if self.arb_type == "hedge" else 0.0,
            # NEW: Market features (4)
            min(self.spread * 10, 1.0),  # spread 0.1 -> 1.0
            min(self.bid_depth / 10000, 1.0),  # depth 10k -> 1.0
            min(self.ask_depth / 10000, 1.0),
            (self.price_trend + 1) / 2,  # -1,1 -> 0,1
            # NEW: Distance from target (1)
            min(abs(self.distance_from_target) * 100, 1.0),  # 1% -> 1.0
        ]
        return vector

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
            "arb_type": self.arb_type,
            "spread": self.spread,
            "bid_depth": self.bid_depth,
            "ask_depth": self.ask_depth,
            "price_trend": self.price_trend,
            "distance_from_target": self.distance_from_target,
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
            arb_type=data.get("arb_type", "none"),
            spread=data.get("spread", 0.0),
            bid_depth=data.get("bid_depth", 0.0),
            ask_depth=data.get("ask_depth", 0.0),
            price_trend=data.get("price_trend", 0.0),
            distance_from_target=data.get("distance_from_target", 0.0),
        )


@dataclass
class SimpleLogisticRegression:
    """
    A simple logistic regression model that doesn't require sklearn.
    Uses online learning to update weights incrementally.

    UPDATED: Now supports 21 features (was 11) for enhanced ML integration.
    """
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    learning_rate: float = 0.1
    n_features: int = 21  # Number of features in TradeFeatures.to_vector()

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
        arb_type: str = "none",
        spread: float = 0.0,
        bid_depth: float = 0.0,
        ask_depth: float = 0.0,
        price_trend: float = 0.0,
        distance_from_target: float = 0.0,
    ) -> TradeFeatures:
        """
        Extract features from a trading signal.

        Args:
            signal: Trading signal
            volatility: Asset volatility estimate
            price_momentum: Recent price direction (-1 to 1)
            arb_type: Type of arbitrage signal (none, binary_arb, asymmetric, dump, hedge)
            spread: Market bid-ask spread
            bid_depth: Order book bid depth
            ask_depth: Order book ask depth
            price_trend: Short-term price trend from historical data
            distance_from_target: How far current price is from target
        """
        now = datetime.now(timezone.utc)

        # Get arb_type from signal if stored there
        if hasattr(signal, '_arb_type') and signal._arb_type:
            arb_type = signal._arb_type

        return TradeFeatures(
            edge=signal.edge,
            time_remaining=signal.market.time_remaining,
            volatility=volatility,
            price_momentum=price_momentum,
            hour_of_day=now.hour,
            day_of_week=now.weekday(),
            asset=signal.market.asset,
            side=signal.side.value,
            arb_type=arb_type,
            spread=spread,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            price_trend=price_trend,
            distance_from_target=distance_from_target,
        )

    def should_trade(
        self,
        signal,
        volatility: float,
        price_momentum: float = 0.0,
        arb_type: str = "none",
        spread: float = 0.0,
        bid_depth: float = 0.0,
        ask_depth: float = 0.0,
        price_trend: float = 0.0,
        distance_from_target: float = 0.0,
    ) -> tuple[bool, float, str]:
        """
        Decide if we should take this trade based on ML prediction.

        Args:
            signal: Trading signal
            volatility: Asset volatility
            price_momentum: Recent price direction
            arb_type: Type of arbitrage signal
            spread: Market bid-ask spread
            bid_depth: Order book bid depth
            ask_depth: Order book ask depth
            price_trend: Short-term price trend
            distance_from_target: Distance from target price

        Returns:
            Tuple of (should_trade, confidence, reason)
        """
        # Not enough training data yet - allow trades to collect data
        if self.training_samples < self.min_training_samples:
            return (True, 0.5, f"Learning mode ({self.training_samples}/{self.min_training_samples} samples)")

        # Extract features and predict
        features = self.extract_features(
            signal, volatility, price_momentum, arb_type,
            spread, bid_depth, ask_depth, price_trend, distance_from_target
        )
        feature_vector = features.to_vector()
        confidence = self.model.predict_proba(feature_vector)

        # Log feature importance info for debugging
        if confidence < self.min_confidence:
            arb_info = f" [ARB: {arb_type}]" if arb_type != "none" else ""
            return (False, confidence, f"ML rejected: {confidence:.0%} < {self.min_confidence:.0%}{arb_info}")
        else:
            arb_info = f" [ARB: {arb_type}]" if arb_type != "none" else ""
            return (True, confidence, f"ML confidence: {confidence:.0%}{arb_info}")

    def record_outcome(
        self,
        signal,
        volatility: float,
        price_momentum: float,
        won: bool,
        arb_type: str = "none",
        spread: float = 0.0,
        bid_depth: float = 0.0,
        ask_depth: float = 0.0,
        price_trend: float = 0.0,
        distance_from_target: float = 0.0,
    ):
        """
        Record a trade outcome and update the model.

        Args:
            signal: Original trading signal
            volatility: Asset volatility at trade time
            price_momentum: Price momentum at trade time
            won: Whether the trade was profitable
            arb_type: Type of arbitrage signal
            spread: Market bid-ask spread at trade time
            bid_depth: Order book bid depth at trade time
            ask_depth: Order book ask depth at trade time
            price_trend: Price trend at trade time
            distance_from_target: Distance from target at trade time
        """
        features = self.extract_features(
            signal, volatility, price_momentum, arb_type,
            spread, bid_depth, ask_depth, price_trend, distance_from_target
        )
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

                model_data = data.get("model", {})
                old_weights = model_data.get("weights", [])

                # Check if we need to migrate from old (11 features) to new (21 features)
                expected_features = 21
                if len(old_weights) == 11:
                    logger.info(
                        f"🤖 Migrating ML model from 11 to 21 features..."
                    )
                    # Extend weights with zeros for new features
                    import random
                    random.seed(42)
                    new_weights = old_weights + [random.uniform(-0.1, 0.1) for _ in range(10)]
                    model_data["weights"] = new_weights
                    # Reset training samples since feature set changed significantly
                    self.training_samples = max(0, data.get("training_samples", 0) // 2)
                    self.predictions_made = 0
                    self.correct_predictions = 0
                    logger.info(
                        f"🤖 Migration complete - model needs retraining with new features"
                    )
                else:
                    self.training_samples = data.get("training_samples", 0)
                    self.predictions_made = data.get("predictions_made", 0)
                    self.correct_predictions = data.get("correct_predictions", 0)

                self.model = SimpleLogisticRegression.from_dict(model_data)
                self.min_confidence = data.get("min_confidence", 0.55)

                logger.info(
                    f"🤖 ML model loaded: {self.training_samples} samples, "
                    f"{self.correct_predictions}/{self.predictions_made} correct, "
                    f"{len(self.model.weights)} features"
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


def calculate_price_trend(prices: list[float], window: int = 10) -> float:
    """
    Calculate price trend from historical prices.

    Returns:
        Trend value between -1 (strongly down) and 1 (strongly up)
    """
    if len(prices) < 2:
        return 0.0

    # Use the last N prices
    recent = prices[-min(window, len(prices)):]

    if len(recent) < 2:
        return 0.0

    # Calculate linear regression slope
    n = len(recent)
    x_mean = (n - 1) / 2
    y_mean = sum(recent) / n

    numerator = sum((i - x_mean) * (p - y_mean) for i, p in enumerate(recent))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return 0.0

    slope = numerator / denominator

    # Normalize by average price to get percentage change per sample
    if y_mean > 0:
        normalized_slope = slope / y_mean
        # Scale to -1, 1 range (assuming 1% per sample is extreme)
        trend = max(-1, min(1, normalized_slope * 100))
        return trend

    return 0.0


def extract_ml_features_from_market(
    signal,
    signal_generator,
    market,
) -> dict:
    """
    Extract all ML features from signal, market, and signal generator.

    This is a convenience function to gather all enhanced features.

    Args:
        signal: Trading signal
        signal_generator: SignalGenerator instance
        market: MarketState instance

    Returns:
        Dictionary with all ML feature parameters
    """
    # Get volatility
    volatility = signal_generator.get_volatility(market.asset)

    # Calculate price momentum
    current_price = signal_generator.get_price(market.asset)
    price_momentum = 0.0
    if current_price and market.target_price:
        price_momentum = (current_price - market.target_price) / market.target_price
        price_momentum = max(-1, min(1, price_momentum * 10))

    # Calculate price trend from historical data
    symbol = f"{market.asset.lower()}/usd"
    price_history = signal_generator.price_histories.get(symbol, [])
    price_trend = calculate_price_trend(price_history)

    # Get arb type from signal
    arb_type = getattr(signal, '_arb_type', "none") or "none"

    # Market spread and depth
    spread = market.best_ask - market.best_bid if market.best_ask and market.best_bid else 0.0
    bid_depth = market.bid_depth
    ask_depth = market.ask_depth

    # Distance from target
    distance_from_target = 0.0
    if current_price and market.target_price:
        distance_from_target = (current_price - market.target_price) / market.target_price

    return {
        "volatility": volatility,
        "price_momentum": price_momentum,
        "arb_type": arb_type,
        "spread": spread,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "price_trend": price_trend,
        "distance_from_target": distance_from_target,
    }
