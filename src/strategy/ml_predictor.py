"""
Machine Learning Signal Quality Predictor

Learns from trade history to predict which signals are likely to win.
Uses a simple but effective approach that works without heavy ML dependencies.

ENHANCED: Now includes features for:
- Arbitrage signal types (binary, asymmetric, dump, hedge)
- Market spread and order book depth
- Price trend from historical data
- Distance from target price
- Binance price lead (faster indicator)
- Binance confirmation signals (STRONG, MEDIUM, WEAK, NONE)
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

# Binance confirmation types for one-hot encoding
BINANCE_CONF_TYPES = ["NONE", "WEAK", "MEDIUM", "STRONG"]


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
    - binance_lead_pct: Binance price lead over Chainlink (faster indicator)
    - binance_confirmation: Confirmation signal type (STRONG, MEDIUM, WEAK, NONE)
    - trend_1h/4h/1d: Multi-timeframe trend context for better decision making
    - price_normalized: Current asset price (normalized 0-1)
    - price_above_target: Whether price is above target (binary)
    - price_range_position: Where price sits in recent high/low range (0-1)
    - price_velocity: How fast price is moving (magnitude)
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

    # Enhanced features
    arb_type: str = "none"  # Type of arbitrage (none, binary_arb, asymmetric, dump, hedge)
    spread: float = 0.0  # Market bid-ask spread
    bid_depth: float = 0.0  # Order book bid depth
    ask_depth: float = 0.0  # Order book ask depth
    price_trend: float = 0.0  # Short-term price trend (-1 to 1)
    distance_from_target: float = 0.0  # Normalized distance from target price

    # Binance confirmation features (leading indicator)
    binance_lead_pct: float = 0.0  # (Binance - Chainlink) / Chainlink
    binance_confirmation: str = "NONE"  # STRONG, MEDIUM, WEAK, NONE

    # Multi-timeframe trends (context for ML decision making)
    # These help the model understand broader market direction
    trend_1h: float = 0.0   # 1-hour trend (-1 to +1)
    trend_4h: float = 0.0   # 4-hour trend (-1 to +1)
    trend_1d: float = 0.0   # 1-day trend (-1 to +1)

    # NEW: Price observation features (helps ML understand price context)
    price_normalized: float = 0.0  # Current price normalized (0-1 scale per asset)
    price_above_target: float = 0.0  # 1.0 if price >= target, 0.0 otherwise
    price_range_position: float = 0.5  # Position in recent range (0=low, 1=high)
    price_velocity: float = 0.0  # Speed of price movement (normalized)

    def to_vector(self) -> list[float]:
        """Convert to feature vector for model (33 features total)."""
        # Normalize features to roughly 0-1 range
        vector = [
            # Core features (6)
            max(0.0, min(self.edge * 10, 1.0)),  # edge 0.1 -> 1.0, clamp negatives to 0
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
            # Arb type one-hot (5)
            1.0 if self.arb_type == "none" else 0.0,
            1.0 if self.arb_type == "binary_arb" else 0.0,
            1.0 if self.arb_type == "asymmetric" else 0.0,
            1.0 if self.arb_type == "dump" else 0.0,
            1.0 if self.arb_type == "hedge" else 0.0,
            # Market features (4)
            min(self.spread * 10, 1.0),  # spread 0.1 -> 1.0
            min(self.bid_depth / 10000, 1.0),  # depth 10k -> 1.0
            min(self.ask_depth / 10000, 1.0),
            (self.price_trend + 1) / 2,  # -1,1 -> 0,1
            # Distance from target (1)
            min(abs(self.distance_from_target) * 100, 1.0),  # 1% -> 1.0
            # Binance confirmation features (5)
            # Lead percentage normalized: -1% to +1% -> 0 to 1
            (max(-0.01, min(0.01, self.binance_lead_pct)) + 0.01) / 0.02,
            # Confirmation type one-hot (4)
            1.0 if self.binance_confirmation == "NONE" else 0.0,
            1.0 if self.binance_confirmation == "WEAK" else 0.0,
            1.0 if self.binance_confirmation == "MEDIUM" else 0.0,
            1.0 if self.binance_confirmation == "STRONG" else 0.0,
            # Multi-timeframe trends (3) - normalized from -1,1 to 0,1
            (self.trend_1h + 1) / 2,
            (self.trend_4h + 1) / 2,
            (self.trend_1d + 1) / 2,
            # NEW: Price observation features (4)
            min(max(self.price_normalized, 0.0), 1.0),  # Already 0-1
            self.price_above_target,  # Binary 0 or 1
            min(max(self.price_range_position, 0.0), 1.0),  # 0-1 range position
            min(abs(self.price_velocity) * 10, 1.0),  # Velocity normalized (0.1 = 1.0)
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
            "binance_lead_pct": self.binance_lead_pct,
            "binance_confirmation": self.binance_confirmation,
            "trend_1h": self.trend_1h,
            "trend_4h": self.trend_4h,
            "trend_1d": self.trend_1d,
            "price_normalized": self.price_normalized,
            "price_above_target": self.price_above_target,
            "price_range_position": self.price_range_position,
            "price_velocity": self.price_velocity,
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
            binance_lead_pct=data.get("binance_lead_pct", 0.0),
            binance_confirmation=data.get("binance_confirmation", "NONE"),
            trend_1h=data.get("trend_1h", 0.0),
            trend_4h=data.get("trend_4h", 0.0),
            trend_1d=data.get("trend_1d", 0.0),
            price_normalized=data.get("price_normalized", 0.0),
            price_above_target=data.get("price_above_target", 0.0),
            price_range_position=data.get("price_range_position", 0.5),
            price_velocity=data.get("price_velocity", 0.0),
        )


@dataclass
class SimpleLogisticRegression:
    """
    A simple logistic regression model that doesn't require sklearn.
    Uses online learning to update weights incrementally.

    UPDATED: Now supports 29 features including multi-timeframe trends (1h, 4h, 1d).
    """
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    learning_rate: float = 0.1
    n_features: int = 33  # Number of features in TradeFeatures.to_vector()

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
class SimpleNeuralNetwork:
    """
    Simple 2-layer neural network for online learning.
    More powerful than logistic regression, adapts to new patterns.

    Architecture: Input(33) -> Hidden(32) -> Output(1)
    Uses ReLU activation and online gradient descent.

    FIXED: Now uses all 33 features from TradeFeatures.to_vector()
    """
    input_size: int = 33  # Full feature vector (matching TradeFeatures.to_vector())
    hidden_size: int = 32  # Larger hidden layer for more features
    learning_rate: float = 0.03  # Slightly lower for stability

    # Weights
    weights_ih: list[list[float]] = field(default_factory=list)  # Input -> Hidden
    weights_ho: list[float] = field(default_factory=list)  # Hidden -> Output
    bias_h: list[float] = field(default_factory=list)  # Hidden bias
    bias_o: float = 0.0  # Output bias

    def __post_init__(self):
        """Initialize weights if not provided."""
        import random
        random.seed(42)

        if not self.weights_ih:
            # Xavier initialization
            scale = (2.0 / (self.input_size + self.hidden_size)) ** 0.5
            self.weights_ih = [
                [random.gauss(0, scale) for _ in range(self.input_size)]
                for _ in range(self.hidden_size)
            ]

        if not self.weights_ho:
            scale = (2.0 / (self.hidden_size + 1)) ** 0.5
            self.weights_ho = [random.gauss(0, scale) for _ in range(self.hidden_size)]

        if not self.bias_h:
            self.bias_h = [0.0] * self.hidden_size

    def _relu(self, x: float) -> float:
        """ReLU activation."""
        return max(0, x)

    def _relu_derivative(self, x: float) -> float:
        """ReLU derivative."""
        return 1.0 if x > 0 else 0.0

    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation for output."""
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, features: list[float]) -> tuple[float, list[float]]:
        """Forward pass, returns (output, hidden_activations)."""
        # Pad or truncate features to match input size
        if len(features) < self.input_size:
            features = features + [0.0] * (self.input_size - len(features))
        elif len(features) > self.input_size:
            features = features[:self.input_size]

        # Hidden layer
        hidden = []
        for i in range(self.hidden_size):
            z = self.bias_h[i] + sum(
                w * f for w, f in zip(self.weights_ih[i], features)
            )
            hidden.append(self._relu(z))

        # Output layer
        z_out = self.bias_o + sum(w * h for w, h in zip(self.weights_ho, hidden))
        output = self._sigmoid(z_out)

        return output, hidden

    def predict_proba(self, features: list[float]) -> float:
        """Predict probability of win."""
        output, _ = self.forward(features)
        return output

    def update(self, features: list[float], outcome: int):
        """Update weights using backpropagation."""
        # Pad or truncate features
        if len(features) < self.input_size:
            features = features + [0.0] * (self.input_size - len(features))
        elif len(features) > self.input_size:
            features = features[:self.input_size]

        # Forward pass
        output, hidden = self.forward(features)

        # Calculate output error
        output_error = outcome - output
        output_delta = output_error * output * (1 - output)  # Sigmoid derivative

        # Update output weights
        for i in range(self.hidden_size):
            self.weights_ho[i] += self.learning_rate * output_delta * hidden[i]
        self.bias_o += self.learning_rate * output_delta

        # Calculate hidden errors and update hidden weights
        for i in range(self.hidden_size):
            # Hidden layer pre-activation for ReLU derivative
            z_h = self.bias_h[i] + sum(
                w * f for w, f in zip(self.weights_ih[i], features)
            )
            hidden_delta = output_delta * self.weights_ho[i] * self._relu_derivative(z_h)

            for j in range(self.input_size):
                self.weights_ih[i][j] += self.learning_rate * hidden_delta * features[j]
            self.bias_h[i] += self.learning_rate * hidden_delta

    def to_dict(self) -> dict:
        """Serialize model."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "weights_ih": self.weights_ih,
            "weights_ho": self.weights_ho,
            "bias_h": self.bias_h,
            "bias_o": self.bias_o,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimpleNeuralNetwork":
        """Deserialize model."""
        return cls(
            input_size=data.get("input_size", 11),
            hidden_size=data.get("hidden_size", 16),
            learning_rate=data.get("learning_rate", 0.05),
            weights_ih=data.get("weights_ih", []),
            weights_ho=data.get("weights_ho", []),
            bias_h=data.get("bias_h", []),
            bias_o=data.get("bias_o", 0.0),
        )


class TraderModelLoader:
    """
    Loads the pre-trained Random Forest model from successful trader data.
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(project_root, "trader_model.json")

        self.model_path = model_path
        self.trees = []
        self.feature_names = []
        self.is_loaded = False
        self._load()

    def _load(self):
        """Load model from disk."""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, "r") as f:
                    data = json.load(f)

                model_data = data.get("model", {})
                self.trees = model_data.get("trees", [])
                self.feature_names = model_data.get("feature_names", [])
                self.is_loaded = len(self.trees) > 0

                if self.is_loaded:
                    metrics = data.get("metrics", {}).get("test", {})
                    logger.info(
                        f"Loaded trader model: {len(self.trees)} trees | "
                        f"Accuracy: {metrics.get('accuracy', 0)*100:.1f}%"
                    )
        except Exception as e:
            logger.warning(f"Could not load trader model: {e}")
            self.is_loaded = False

    def _predict_tree(self, tree: dict, features: list[float]) -> float:
        """Predict using a single tree."""
        if tree.get("leaf"):
            return tree.get("prediction", 0.5)

        feature_idx = tree.get("feature", 0)
        threshold = tree.get("threshold", 0.5)

        if feature_idx < len(features) and features[feature_idx] <= threshold:
            return self._predict_tree(tree.get("left", {"leaf": True, "prediction": 0.5}), features)
        else:
            return self._predict_tree(tree.get("right", {"leaf": True, "prediction": 0.5}), features)

    def predict_proba(self, features: list[float]) -> float:
        """Predict probability using Random Forest."""
        if not self.is_loaded or not self.trees:
            return 0.5

        predictions = [self._predict_tree(tree, features) for tree in self.trees]
        return sum(predictions) / len(predictions)

    def extract_simple_features(self, signal, market) -> list[float]:
        """
        Extract simplified features matching the trader model format.

        The trader model uses 11 features:
        [price, size, usdc_size, hour, day, time_remaining, is_btc, is_eth, is_sol, is_xrp, is_up]
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

        # Get price (use recommended_price from signal)
        price = getattr(signal, 'recommended_price', 0.5)

        # Size (normalized)
        size = getattr(signal, 'size_shares', 10) / 100
        usdc_size = getattr(signal, 'size_usd', 10) / 200

        # Time features
        hour = now.hour / 24
        day = now.weekday() / 7

        # Time remaining
        time_remaining = getattr(signal, 'time_remaining', 450) / 900

        # Asset one-hot
        asset = market.asset if hasattr(market, 'asset') else "BTC"
        is_btc = 1.0 if asset == "BTC" else 0.0
        is_eth = 1.0 if asset == "ETH" else 0.0
        is_sol = 1.0 if asset == "SOL" else 0.0
        is_xrp = 1.0 if asset == "XRP" else 0.0

        # Side
        side = getattr(signal, 'side', None)
        is_up = 1.0 if side and side.value == "UP" else 0.0

        return [price, size, usdc_size, hour, day, time_remaining,
                is_btc, is_eth, is_sol, is_xrp, is_up]


@dataclass
class HybridPredictor:
    """
    Hybrid ML predictor combining:
    1. Random Forest (pre-trained on successful trader)
    2. Neural Network (online learning from your trades)

    The combination provides stable base predictions while
    adapting to current market conditions.
    """

    # Base model weight (0.7 = 70% base, 30% adaptive)
    base_weight: float = 0.7

    # Models
    trader_model: TraderModelLoader = field(default_factory=TraderModelLoader)
    adaptive_model: SimpleNeuralNetwork = field(default_factory=SimpleNeuralNetwork)

    # Tracking
    predictions_made: int = 0
    correct_predictions: int = 0
    training_samples: int = 0

    def predict_proba(self, signal, market, full_features: list[float]) -> float:
        """
        Predict win probability using hybrid approach.

        Args:
            signal: Trading signal
            market: Market state
            full_features: Full 33-feature vector

        Returns:
            Combined probability estimate
        """
        # Use full 33-feature vector for adaptive model (neural network)
        # This ensures the model learns from ALL available information

        # Get base prediction from trader model (uses 11 simple features)
        if self.trader_model.is_loaded:
            simple_features = self.trader_model.extract_simple_features(signal, market)
            base_prob = self.trader_model.predict_proba(simple_features)
        else:
            base_prob = 0.5

        # Get adaptive prediction using FULL 33-feature vector
        # The neural network now has input_size=33 to handle this
        adaptive_prob = self.adaptive_model.predict_proba(full_features)

        # Combine predictions
        if self.trader_model.is_loaded:
            combined = self.base_weight * base_prob + (1 - self.base_weight) * adaptive_prob
        else:
            # If no trader model, use only adaptive (with full features)
            combined = adaptive_prob

        return combined

    def update(self, features: list[float], outcome: int, signal=None, market=None):
        """Update adaptive model with new outcome."""
        # Update adaptive model with FULL 33-feature vector
        # This ensures training and prediction use the same feature space
        self.adaptive_model.update(features, outcome)
        self.training_samples += 1

    def to_dict(self) -> dict:
        """Serialize hybrid predictor."""
        return {
            "base_weight": self.base_weight,
            "adaptive_model": self.adaptive_model.to_dict(),
            "predictions_made": self.predictions_made,
            "correct_predictions": self.correct_predictions,
            "training_samples": self.training_samples,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HybridPredictor":
        """Deserialize hybrid predictor."""
        predictor = cls(
            base_weight=data.get("base_weight", 0.7),
            predictions_made=data.get("predictions_made", 0),
            correct_predictions=data.get("correct_predictions", 0),
            training_samples=data.get("training_samples", 0),
        )

        if "adaptive_model" in data:
            predictor.adaptive_model = SimpleNeuralNetwork.from_dict(data["adaptive_model"])

        return predictor


@dataclass
class MLSignalPredictor:
    """
    Machine Learning predictor for signal quality.

    Features:
    - Learns from historical trade outcomes
    - Uses online learning (updates after each trade)
    - Persists model to disk
    - Provides win probability predictions
    - HYBRID MODE: Combines pre-trained trader model with adaptive neural network
    """

    model: SimpleLogisticRegression = field(default_factory=SimpleLogisticRegression)
    min_confidence: float = 0.55  # Minimum predicted win probability to trade
    min_training_samples: int = 10  # Minimum trades before using ML filter
    training_samples: int = 0
    model_path: str = ""  # Set in __post_init__

    # Performance tracking
    predictions_made: int = 0
    correct_predictions: int = 0

    # Hybrid predictor (combines trader model + adaptive neural network)
    hybrid_predictor: Optional[HybridPredictor] = None
    use_hybrid: bool = True  # Enable hybrid mode by default

    def __post_init__(self):
        """Initialize model path and load model if exists."""
        # Use absolute path based on project root (two levels up from this file)
        if not self.model_path:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.model_path = os.path.join(project_root, "ml_model.json")
        self._load_model()

        # Create the model file if it doesn't exist
        if not os.path.exists(self.model_path):
            self._save_model()
            logger.info(f"🤖 Created new ML model at: {self.model_path}")

        # Initialize hybrid predictor
        if self.use_hybrid and self.hybrid_predictor is None:
            self.hybrid_predictor = HybridPredictor()
            if self.hybrid_predictor.trader_model.is_loaded:
                logger.info("🤖 Hybrid ML enabled: Trader model + Adaptive NN")
            else:
                logger.info("🤖 Hybrid ML enabled: Adaptive NN only (no trader model found)")

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
        binance_lead_pct: float = 0.0,
        binance_confirmation: str = "NONE",
        trend_1h: float = 0.0,
        trend_4h: float = 0.0,
        trend_1d: float = 0.0,
        current_price: float = 0.0,
        target_price: float = 0.0,
        price_high: float = 0.0,
        price_low: float = 0.0,
        price_velocity: float = 0.0,
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
            binance_lead_pct: Binance price lead over Chainlink (as decimal)
            binance_confirmation: Confirmation type (STRONG, MEDIUM, WEAK, NONE)
            trend_1h: 1-hour price trend (-1 to +1)
            trend_4h: 4-hour price trend (-1 to +1)
            trend_1d: 1-day price trend (-1 to +1)
            current_price: Current asset price (e.g., BTC price in USD)
            target_price: Target price for the market
            price_high: Recent high price (for range calculation)
            price_low: Recent low price (for range calculation)
            price_velocity: Rate of price change (absolute)
        """
        now = datetime.now(timezone.utc)

        # Get arb_type from signal if stored there
        if hasattr(signal, '_arb_type') and signal._arb_type:
            arb_type = signal._arb_type

        # Get prices from signal if not provided
        if current_price == 0.0 and hasattr(signal, 'market'):
            target_price = getattr(signal.market, 'target_price', 0.0)

        # Calculate normalized price (0-1 scale based on asset)
        # BTC: ~100k range, ETH: ~10k range, SOL: ~500 range, XRP: ~5 range
        asset = signal.market.asset
        price_scales = {"BTC": 150000, "ETH": 10000, "SOL": 500, "XRP": 10}
        scale = price_scales.get(asset, 100000)
        price_normalized = min(current_price / scale, 1.0) if current_price > 0 else 0.0

        # Calculate price above target (binary)
        price_above_target = 1.0 if current_price >= target_price and target_price > 0 else 0.0

        # Calculate price range position (0 = at low, 1 = at high)
        if price_high > price_low and price_low > 0:
            price_range_position = (current_price - price_low) / (price_high - price_low)
            price_range_position = max(0.0, min(1.0, price_range_position))
        else:
            price_range_position = 0.5  # Default to middle if no range data

        return TradeFeatures(
            edge=signal.edge,
            time_remaining=signal.market.time_remaining,
            volatility=volatility,
            price_momentum=price_momentum,
            hour_of_day=now.hour,
            day_of_week=now.weekday(),
            asset=asset,
            side=signal.side.value,
            arb_type=arb_type,
            spread=spread,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            price_trend=price_trend,
            distance_from_target=distance_from_target,
            binance_lead_pct=binance_lead_pct,
            binance_confirmation=binance_confirmation,
            trend_1h=trend_1h,
            trend_4h=trend_4h,
            trend_1d=trend_1d,
            price_normalized=price_normalized,
            price_above_target=price_above_target,
            price_range_position=price_range_position,
            price_velocity=price_velocity,
        )

    def _get_gradual_threshold(self) -> float:
        """
        Get the ML confidence threshold based on training samples.

        Uses a gradual ramp-up to collect more diverse training data
        before applying strict filtering:
        - 0-25 samples: Learning mode (allow all) - extended for more exploration
        - 25-40 samples: 45% threshold (very lenient)
        - 40-60 samples: 50% threshold (lenient)
        - 60+ samples: 55% threshold (moderate)

        Returns:
            Current confidence threshold (0.0 to 1.0)
        """
        samples = self.training_samples

        if samples < 25:
            return 0.0  # Learning mode - allow all (extended from 10 to 25)
        elif samples < 40:
            return 0.45  # Early filtering - 45% (lowered from 50%)
        elif samples < 60:
            return 0.50  # Medium filtering - 50% (lowered from 55%)
        else:
            return 0.55  # Full filtering - 55% (lowered from 60%)

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
        binance_lead_pct: float = 0.0,
        binance_confirmation: str = "NONE",
        trend_1h: float = 0.0,
        trend_4h: float = 0.0,
        trend_1d: float = 0.0,
    ) -> tuple[bool, float, str]:
        """
        Decide if we should take this trade based on ML prediction.

        Uses gradual threshold ramp-up (lenient settings for more trades):
        - 0-25 samples: Learning mode (allow all)
        - 25-40 samples: 45% threshold
        - 40-60 samples: 50% threshold
        - 60+ samples: 55% threshold

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
            binance_lead_pct: Binance price lead over Chainlink
            binance_confirmation: Confirmation type (STRONG, MEDIUM, WEAK, NONE)
            trend_1h: 1-hour price trend (-1 to +1)
            trend_4h: 4-hour price trend (-1 to +1)
            trend_1d: 1-day price trend (-1 to +1)

        Returns:
            Tuple of (should_trade, confidence, reason)
        """
        # Get current threshold based on training samples
        threshold = self._get_gradual_threshold()

        # Extract features and predict
        features = self.extract_features(
            signal, volatility, price_momentum, arb_type,
            spread, bid_depth, ask_depth, price_trend, distance_from_target,
            binance_lead_pct, binance_confirmation,
            trend_1h, trend_4h, trend_1d
        )
        feature_vector = features.to_vector()

        # Use hybrid predictor if available, otherwise use logistic regression
        if self.use_hybrid and self.hybrid_predictor is not None:
            # Get market from signal for hybrid predictor
            market = getattr(signal, 'market', None)
            confidence = self.hybrid_predictor.predict_proba(signal, market, feature_vector)
            model_type = "Hybrid" if self.hybrid_predictor.trader_model.is_loaded else "NN"
        else:
            confidence = self.model.predict_proba(feature_vector)
            model_type = "LR"

        # Log feature importance info for debugging
        arb_info = f" [ARB: {arb_type}]" if arb_type != "none" else ""
        bn_info = f" [BN: {binance_confirmation}]" if binance_confirmation != "NONE" else ""

        if confidence < threshold:
            return (False, confidence, f"{model_type} rejected: {confidence:.0%} < {threshold:.0%}{arb_info}{bn_info}")
        else:
            return (True, confidence, f"{model_type} confidence: {confidence:.0%} (threshold: {threshold:.0%}){arb_info}{bn_info}")

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
        binance_lead_pct: float = 0.0,
        binance_confirmation: str = "NONE",
        trend_1h: float = 0.0,
        trend_4h: float = 0.0,
        trend_1d: float = 0.0,
        current_price: float = 0.0,
        target_price: float = 0.0,
        price_high: float = 0.0,
        price_low: float = 0.0,
        price_velocity: float = 0.0,
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
            binance_lead_pct: Binance price lead over Chainlink at trade time
            binance_confirmation: Confirmation type at trade time
            trend_1h: 1-hour price trend (-1 to +1)
            trend_4h: 4-hour price trend (-1 to +1)
            trend_1d: 1-day price trend (-1 to +1)
            current_price: Current asset price at trade time
            target_price: Target price for the market
            price_high: Recent high price
            price_low: Recent low price
            price_velocity: Rate of price change
        """
        features = self.extract_features(
            signal, volatility, price_momentum, arb_type,
            spread, bid_depth, ask_depth, price_trend, distance_from_target,
            binance_lead_pct, binance_confirmation,
            trend_1h, trend_4h, trend_1d,
            current_price, target_price, price_high, price_low, price_velocity
        )
        feature_vector = features.to_vector()

        # Get prediction before updating (for accuracy tracking)
        if self.training_samples >= self.min_training_samples:
            # Use hybrid predictor if available for accurate tracking
            if self.use_hybrid and self.hybrid_predictor is not None:
                market = getattr(signal, 'market', None)
                pred_prob = self.hybrid_predictor.predict_proba(signal, market, feature_vector)
            else:
                pred_prob = self.model.predict_proba(feature_vector)
            predicted_win = pred_prob >= 0.5
            if predicted_win == won:
                self.correct_predictions += 1
            self.predictions_made += 1

        # Update model with this example
        self.model.update(feature_vector, 1 if won else 0)
        self.training_samples += 1

        # Also update hybrid predictor's adaptive model
        if self.use_hybrid and self.hybrid_predictor is not None:
            market = getattr(signal, 'market', None)
            self.hybrid_predictor.update(feature_vector, 1 if won else 0, signal=signal, market=market)

        # Log outcome recording
        result_str = "WIN" if won else "LOSS"
        model_type = "Hybrid" if (self.use_hybrid and self.hybrid_predictor and self.hybrid_predictor.trader_model.is_loaded) else "ML"
        logger.info(f"🤖 {model_type} recorded: {result_str} | Sample #{self.training_samples}")

        # Log learning progress
        if self.training_samples % 5 == 0:
            accuracy = self.correct_predictions / max(1, self.predictions_made)
            logger.info(
                f"🤖 ML Model: {self.training_samples} samples | "
                f"Accuracy: {accuracy:.0%} ({self.correct_predictions}/{self.predictions_made})"
            )

        # Save model after EVERY trade (user might stop bot at any time)
        self._save_model()

    def get_model_stats(self) -> dict:
        """Get model statistics."""
        accuracy = self.correct_predictions / max(1, self.predictions_made)
        stats = {
            "training_samples": self.training_samples,
            "predictions_made": self.predictions_made,
            "correct_predictions": self.correct_predictions,
            "accuracy": accuracy,
            "min_confidence": self.min_confidence,
            "is_active": self.training_samples >= self.min_training_samples,
            "hybrid_enabled": self.use_hybrid,
        }

        # Add hybrid predictor stats
        if self.use_hybrid and self.hybrid_predictor is not None:
            stats["trader_model_loaded"] = self.hybrid_predictor.trader_model.is_loaded
            stats["hybrid_training_samples"] = self.hybrid_predictor.training_samples
            stats["model_type"] = "Hybrid RF+NN" if self.hybrid_predictor.trader_model.is_loaded else "Adaptive NN"
        else:
            stats["model_type"] = "Logistic Regression"

        return stats

    def is_prediction_accurate(self, min_accuracy: float = 0.45, min_samples: int = 10) -> tuple[bool, str]:
        """
        Check if model predictions are accurate enough to continue trading.

        Args:
            min_accuracy: Minimum required accuracy (default 45%)
            min_samples: Minimum predictions needed before checking (default 10)

        Returns:
            Tuple of (is_accurate, reason)
        """
        if self.predictions_made < min_samples:
            return (True, f"Not enough predictions yet ({self.predictions_made}/{min_samples})")

        accuracy = self.correct_predictions / self.predictions_made
        if accuracy < min_accuracy:
            return (
                False,
                f"Prediction accuracy {accuracy:.0%} below {min_accuracy:.0%} minimum "
                f"({self.correct_predictions}/{self.predictions_made} correct)"
            )

        return (True, f"Accuracy OK: {accuracy:.0%}")

    def save(self):
        """Public method to save model to disk."""
        self._save_model()

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

            # Save hybrid predictor state
            if self.use_hybrid and self.hybrid_predictor is not None:
                data["hybrid_predictor"] = self.hybrid_predictor.to_dict()

            with open(self.model_path, "w") as f:
                json.dump(data, f, indent=2)

            model_type = "Hybrid" if (self.use_hybrid and self.hybrid_predictor and self.hybrid_predictor.trader_model.is_loaded) else "ML"
            logger.info(f"🤖 {model_type} model saved: {self.training_samples} samples → {self.model_path}")
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

                # Check if we need to migrate feature counts
                # 11 -> 21: Added arb type and market features
                # 21 -> 26: Added Binance confirmation features
                # 26 -> 29: Added multi-timeframe trends (1h, 4h, 1d)
                # 29 -> 33: Added price observation features (price_normalized, price_above_target, price_range_position, price_velocity)
                expected_features = 33
                import random
                random.seed(42)

                if len(old_weights) == 11:
                    logger.info(
                        f"🤖 Migrating ML model from 11 to 33 features (adding arb + Binance + trends + price)..."
                    )
                    # Extend weights with small random values for new features
                    new_weights = old_weights + [random.uniform(-0.1, 0.1) for _ in range(22)]
                    model_data["weights"] = new_weights
                    # Reset training samples since feature set changed significantly
                    self.training_samples = max(0, data.get("training_samples", 0) // 2)
                    self.predictions_made = 0
                    self.correct_predictions = 0
                    logger.info(
                        f"🤖 Migration complete - model needs retraining with new features"
                    )
                elif len(old_weights) == 21:
                    logger.info(
                        f"🤖 Migrating ML model from 21 to 33 features (adding Binance + trends + price)..."
                    )
                    # Extend weights for Binance + trend + price features
                    new_weights = old_weights + [random.uniform(-0.1, 0.1) for _ in range(12)]
                    model_data["weights"] = new_weights
                    # Keep most training data, slight reset since new features added
                    self.training_samples = max(0, int(data.get("training_samples", 0) * 0.75))
                    self.predictions_made = int(data.get("predictions_made", 0) * 0.75)
                    self.correct_predictions = int(data.get("correct_predictions", 0) * 0.75)
                    logger.info(
                        f"🤖 Migration complete - model will retrain with Binance + trend + price features"
                    )
                elif len(old_weights) == 26:
                    logger.info(
                        f"🤖 Migrating ML model from 26 to 33 features (adding trends + price)..."
                    )
                    # Extend weights for trend + price features (7 new)
                    new_weights = old_weights + [random.uniform(-0.1, 0.1) for _ in range(7)]
                    model_data["weights"] = new_weights
                    # Keep most training data since this is a minor addition
                    self.training_samples = max(0, int(data.get("training_samples", 0) * 0.9))
                    self.predictions_made = int(data.get("predictions_made", 0) * 0.9)
                    self.correct_predictions = int(data.get("correct_predictions", 0) * 0.9)
                    logger.info(
                        f"🤖 Migration complete - model will retrain with trend + price features"
                    )
                elif len(old_weights) == 29:
                    logger.info(
                        f"🤖 Migrating ML model from 29 to 33 features (adding price observation)..."
                    )
                    # Extend weights for price features only (4 new: price_normalized, price_above_target, price_range_position, price_velocity)
                    new_weights = old_weights + [random.uniform(-0.1, 0.1) for _ in range(4)]
                    model_data["weights"] = new_weights
                    # Keep most training data since this is a minor addition
                    self.training_samples = max(0, int(data.get("training_samples", 0) * 0.9))
                    self.predictions_made = int(data.get("predictions_made", 0) * 0.9)
                    self.correct_predictions = int(data.get("correct_predictions", 0) * 0.9)
                    logger.info(
                        f"🤖 Migration complete - model will retrain with price observation features"
                    )
                else:
                    self.training_samples = data.get("training_samples", 0)
                    self.predictions_made = data.get("predictions_made", 0)
                    self.correct_predictions = data.get("correct_predictions", 0)

                self.model = SimpleLogisticRegression.from_dict(model_data)
                self.min_confidence = data.get("min_confidence", 0.55)

                # Load hybrid predictor state if present
                if self.use_hybrid and "hybrid_predictor" in data:
                    self.hybrid_predictor = HybridPredictor.from_dict(data["hybrid_predictor"])

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

    # Binance confirmation features
    binance_lead_pct = 0.0
    binance_confirmation = "NONE"

    # Get Binance confirmation if available
    if hasattr(signal_generator, 'get_binance_confirmation'):
        binance_conf = signal_generator.get_binance_confirmation(
            asset=market.asset,
            target_price=market.target_price,
        )
        binance_lead_pct = binance_conf.get("lead_pct", 0.0) or 0.0
        binance_confirmation = binance_conf.get("confirmation_type", "NONE")

    # Multi-timeframe trends from Binance (1h, 4h, 1d)
    # These provide broader market context for better ML decision making
    trend_1h = 0.0
    trend_4h = 0.0
    trend_1d = 0.0

    try:
        from ..data.binance import get_multi_timeframe_trends
        trends = get_multi_timeframe_trends(market.asset)
        trend_1h = trends.get("trend_1h", 0.0)
        trend_4h = trends.get("trend_4h", 0.0)
        trend_1d = trends.get("trend_1d", 0.0)
    except Exception as e:
        logger.debug(f"Could not fetch multi-timeframe trends: {e}")

    return {
        "volatility": volatility,
        "price_momentum": price_momentum,
        "arb_type": arb_type,
        "spread": spread,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "price_trend": price_trend,
        "distance_from_target": distance_from_target,
        "binance_lead_pct": binance_lead_pct,
        "binance_confirmation": binance_confirmation,
        "trend_1h": trend_1h,
        "trend_4h": trend_4h,
        "trend_1d": trend_1d,
    }
