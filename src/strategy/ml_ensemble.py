"""
Advanced ML Ensemble Module

Implements:
1. Ensemble Voting - Multiple models vote on predictions
2. Feature Importance - Track which features matter most
3. Gradient Boosting - Simple GBM implementation
4. Model Versioning - A/B testing between model versions

All implemented without heavy ML dependencies (no sklearn, no torch).
"""

import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# GRADIENT BOOSTING IMPLEMENTATION
# =============================================================================

@dataclass
class DecisionStump:
    """
    Simple decision stump (1-level decision tree).
    Used as weak learner in gradient boosting.
    """
    feature_idx: int = 0
    threshold: float = 0.5
    left_value: float = 0.0  # Value if feature <= threshold
    right_value: float = 0.0  # Value if feature > threshold

    def predict(self, features: List[float]) -> float:
        """Predict using this stump."""
        if self.feature_idx < len(features):
            if features[self.feature_idx] <= self.threshold:
                return self.left_value
            else:
                return self.right_value
        return 0.0

    def to_dict(self) -> dict:
        return {
            "feature_idx": self.feature_idx,
            "threshold": self.threshold,
            "left_value": self.left_value,
            "right_value": self.right_value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DecisionStump":
        return cls(
            feature_idx=data.get("feature_idx", 0),
            threshold=data.get("threshold", 0.5),
            left_value=data.get("left_value", 0.0),
            right_value=data.get("right_value", 0.0),
        )


@dataclass
class GradientBoostingClassifier:
    """
    Enhanced Gradient Boosting implementation for binary classification.

    Uses decision stumps as weak learners and gradient descent
    to minimize log-loss.

    IMPROVED:
    - L2 regularization to prevent overfitting
    - Early stopping based on validation loss
    - Feature subsampling (like XGBoost)
    - Feature importance tracking
    """
    n_estimators: int = 100  # Increased for better accuracy
    learning_rate: float = 0.08  # Slightly lower for stability
    max_features: int = 83  # Updated for 83 features

    # Regularization
    l2_lambda: float = 0.1  # L2 regularization strength
    min_samples_split: int = 5  # Minimum samples to make a split
    max_depth: int = 1  # Depth of weak learners (1 = stump)

    # Early stopping
    early_stopping: bool = True
    early_stopping_rounds: int = 10  # Stop if no improvement for N rounds
    validation_fraction: float = 0.2  # Fraction for validation

    # Trained stumps
    stumps: List[DecisionStump] = field(default_factory=list)

    # Initial prediction (log-odds of positive class)
    init_prediction: float = 0.0

    # Training data buffer (for online learning)
    X_buffer: List[List[float]] = field(default_factory=list)
    y_buffer: List[int] = field(default_factory=list)
    buffer_size: int = 300  # Increased buffer

    # Performance tracking
    training_samples: int = 0

    # Feature importance tracking
    feature_importance: List[float] = field(default_factory=list)
    feature_split_counts: List[int] = field(default_factory=list)

    def __post_init__(self):
        """Initialize feature importance arrays."""
        if not self.feature_importance:
            self.feature_importance = [0.0] * self.max_features
        if not self.feature_split_counts:
            self.feature_split_counts = [0] * self.max_features

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function."""
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))

    def _find_best_split(
        self,
        X: List[List[float]],
        residuals: List[float],
        feature_indices: List[int],
    ) -> DecisionStump:
        """Find the best split for a decision stump with L2 regularization."""
        best_stump = DecisionStump()
        best_gain = float('-inf')
        best_feature_idx = -1

        for feat_idx in feature_indices:
            # Get unique values for this feature
            values = sorted(set(x[feat_idx] for x in X if feat_idx < len(x)))

            if len(values) < 2:
                continue

            # Try different thresholds
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2

                # Split samples
                left_residuals = []
                right_residuals = []

                for j, x in enumerate(X):
                    if feat_idx < len(x) and x[feat_idx] <= threshold:
                        left_residuals.append(residuals[j])
                    else:
                        right_residuals.append(residuals[j])

                # Check minimum samples for split
                if len(left_residuals) < self.min_samples_split or len(right_residuals) < self.min_samples_split:
                    continue

                # Calculate gain with L2 regularization (like XGBoost)
                # Regularized leaf values: sum(residuals) / (count + lambda)
                left_sum = sum(left_residuals)
                right_sum = sum(right_residuals)
                n_left = len(left_residuals)
                n_right = len(right_residuals)

                # L2 regularized leaf values
                left_mean = left_sum / (n_left + self.l2_lambda)
                right_mean = right_sum / (n_right + self.l2_lambda)

                # Gain calculation with regularization
                # Gain = 0.5 * [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda)] - gamma
                # For regression on residuals: H = n (count), G = sum(residuals)
                gain_left = (left_sum ** 2) / (n_left + self.l2_lambda)
                gain_right = (right_sum ** 2) / (n_right + self.l2_lambda)
                total_sum = left_sum + right_sum
                total_n = n_left + n_right
                gain_total = (total_sum ** 2) / (total_n + self.l2_lambda)

                gain = 0.5 * (gain_left + gain_right - gain_total)

                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feat_idx
                    best_stump = DecisionStump(
                        feature_idx=feat_idx,
                        threshold=threshold,
                        left_value=left_mean,
                        right_value=right_mean,
                    )

        # Track feature importance
        if best_feature_idx >= 0 and best_feature_idx < len(self.feature_importance):
            self.feature_importance[best_feature_idx] += max(0, best_gain)
            self.feature_split_counts[best_feature_idx] += 1

        return best_stump

    def _calculate_log_loss(self, predictions: List[float], y: List[int]) -> float:
        """Calculate log loss for early stopping."""
        eps = 1e-15
        total_loss = 0.0
        for i, yi in enumerate(y):
            prob = self._sigmoid(predictions[i])
            prob = max(eps, min(1 - eps, prob))
            total_loss -= yi * math.log(prob) + (1 - yi) * math.log(1 - prob)
        return total_loss / len(y) if y else 0.0

    def fit(self, X: List[List[float]], y: List[int]):
        """Fit the model on training data with early stopping."""
        if not X or not y:
            return

        n_samples = len(X)

        # Split into train/validation if early stopping enabled
        if self.early_stopping and n_samples >= 20:
            val_size = int(n_samples * self.validation_fraction)
            indices = list(range(n_samples))
            random.shuffle(indices)
            train_indices = indices[val_size:]
            val_indices = indices[:val_size]
            X_train = [X[i] for i in train_indices]
            y_train = [y[i] for i in train_indices]
            X_val = [X[i] for i in val_indices]
            y_val = [y[i] for i in val_indices]
        else:
            X_train, y_train = X, y
            X_val, y_val = [], []

        n_train = len(X_train)

        # Initialize prediction with log-odds
        pos_count = sum(y_train)
        neg_count = n_train - pos_count
        if pos_count > 0 and neg_count > 0:
            self.init_prediction = math.log(pos_count / neg_count)
        else:
            self.init_prediction = 0.0

        # Current predictions
        train_predictions = [self.init_prediction] * n_train
        val_predictions = [self.init_prediction] * len(X_val) if X_val else []

        # Clear existing stumps and reset feature importance
        self.stumps = []
        self.feature_importance = [0.0] * self.max_features
        self.feature_split_counts = [0] * self.max_features

        # Feature indices to sample from
        all_features = list(range(min(self.max_features, len(X_train[0]) if X_train else self.max_features)))

        # Early stopping tracking
        best_val_loss = float('inf')
        rounds_without_improvement = 0
        best_n_stumps = 0

        # Build stumps iteratively
        for estimator_idx in range(self.n_estimators):
            # Calculate residuals (gradient of log-loss)
            residuals = []
            for i in range(n_train):
                prob = self._sigmoid(train_predictions[i])
                residuals.append(y_train[i] - prob)

            # Sample features (like XGBoost/random forest)
            # Use sqrt(n_features) as default, but can sample more for better coverage
            n_features_sample = max(1, int(math.sqrt(len(all_features)) * 1.5))
            feature_subset = random.sample(all_features, min(n_features_sample, len(all_features)))

            # Find best stump
            stump = self._find_best_split(X_train, residuals, feature_subset)

            # Update training predictions
            for i in range(n_train):
                pred = stump.predict(X_train[i])
                train_predictions[i] += self.learning_rate * pred

            self.stumps.append(stump)

            # Early stopping check
            if self.early_stopping and X_val:
                # Update validation predictions
                for i in range(len(X_val)):
                    pred = stump.predict(X_val[i])
                    val_predictions[i] += self.learning_rate * pred

                val_loss = self._calculate_log_loss(val_predictions, y_val)

                if val_loss < best_val_loss - 0.0001:  # Improvement threshold
                    best_val_loss = val_loss
                    rounds_without_improvement = 0
                    best_n_stumps = len(self.stumps)
                else:
                    rounds_without_improvement += 1

                if rounds_without_improvement >= self.early_stopping_rounds:
                    # Rollback to best model
                    self.stumps = self.stumps[:best_n_stumps]
                    logger.debug(f"Early stopping at {estimator_idx + 1} estimators, best was {best_n_stumps}")
                    break

        self.training_samples = n_samples

    def get_feature_importance(self) -> Dict[int, float]:
        """Get feature importance scores normalized to sum to 1."""
        total = sum(self.feature_importance)
        if total <= 0:
            return {}
        return {i: imp / total for i, imp in enumerate(self.feature_importance) if imp > 0}

    def partial_fit(self, features: List[float], outcome: int):
        """Online learning - add sample and periodically retrain."""
        self.X_buffer.append(features)
        self.y_buffer.append(outcome)

        # Keep buffer size limited
        if len(self.X_buffer) > self.buffer_size:
            self.X_buffer.pop(0)
            self.y_buffer.pop(0)

        # Retrain periodically
        if len(self.X_buffer) >= 20 and len(self.X_buffer) % 10 == 0:
            self.fit(self.X_buffer, self.y_buffer)

    def predict_proba(self, features: List[float]) -> float:
        """Predict probability of positive class."""
        if not self.stumps:
            return 0.5

        # Start with initial prediction
        pred = self.init_prediction

        # Add contributions from all stumps
        for stump in self.stumps:
            pred += self.learning_rate * stump.predict(features)

        return self._sigmoid(pred)

    def to_dict(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_features": self.max_features,
            "l2_lambda": self.l2_lambda,
            "min_samples_split": self.min_samples_split,
            "early_stopping": self.early_stopping,
            "early_stopping_rounds": self.early_stopping_rounds,
            "validation_fraction": self.validation_fraction,
            "init_prediction": self.init_prediction,
            "stumps": [s.to_dict() for s in self.stumps],
            "training_samples": self.training_samples,
            "X_buffer": self.X_buffer[-100:],  # Save last 100 samples
            "y_buffer": self.y_buffer[-100:],
            "feature_importance": self.feature_importance,
            "feature_split_counts": self.feature_split_counts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GradientBoostingClassifier":
        model = cls(
            n_estimators=data.get("n_estimators", 100),
            learning_rate=data.get("learning_rate", 0.08),
            max_features=data.get("max_features", 83),
            l2_lambda=data.get("l2_lambda", 0.1),
            min_samples_split=data.get("min_samples_split", 5),
            early_stopping=data.get("early_stopping", True),
            early_stopping_rounds=data.get("early_stopping_rounds", 10),
            validation_fraction=data.get("validation_fraction", 0.2),
        )
        model.init_prediction = data.get("init_prediction", 0.0)
        model.stumps = [DecisionStump.from_dict(s) for s in data.get("stumps", [])]
        model.training_samples = data.get("training_samples", 0)
        model.X_buffer = data.get("X_buffer", [])
        model.y_buffer = data.get("y_buffer", [])
        model.feature_importance = data.get("feature_importance", [0.0] * model.max_features)
        model.feature_split_counts = data.get("feature_split_counts", [0] * model.max_features)
        return model


# =============================================================================
# FEATURE IMPORTANCE TRACKING
# =============================================================================

@dataclass
class FeatureImportanceTracker:
    """
    Tracks feature importance based on prediction outcomes.

    Uses a simple correlation-based approach:
    - For each feature, track correlation with outcomes
    - Higher correlation = more important feature

    UPDATED: Now supports 83 features including new indicators + variant.
    """

    n_features: int = 83
    feature_names: List[str] = field(default_factory=list)

    # Running statistics for each feature
    feature_sums: List[float] = field(default_factory=list)
    feature_sq_sums: List[float] = field(default_factory=list)
    outcome_sums: List[float] = field(default_factory=list)  # Sum of feature * outcome

    total_samples: int = 0
    total_outcomes: float = 0.0
    total_outcomes_sq: float = 0.0

    def __post_init__(self):
        if not self.feature_names:
            self.feature_names = [
                # Core features (6)
                "edge", "time_remaining", "volatility", "price_momentum",
                "hour_of_day", "day_of_week",
                # Asset one-hot (4)
                "is_btc", "is_eth", "is_sol", "is_xrp",
                # Side (1)
                "is_up",
                # Market variant (1)
                "is_five_min",
                # Arb type one-hot (5)
                "arb_none", "arb_binary", "arb_asymmetric", "arb_dump", "arb_hedge",
                # Market features (4)
                "spread", "bid_depth", "ask_depth", "price_trend",
                # Distance (1)
                "distance_from_target",
                # Binance features (5)
                "binance_lead", "conf_none", "conf_weak", "conf_medium", "conf_strong",
                # Multi-timeframe trends (3)
                "trend_1h", "trend_4h", "trend_1d",
                # Price features (4)
                "price_normalized", "price_above_target", "price_range_position", "price_velocity",
                # Chart analysis (13)
                "chart_rsi", "chart_trend_strength", "chart_is_uptrend", "chart_is_downtrend",
                "chart_is_ranging", "chart_bullish_reversal", "chart_bearish_reversal",
                "chart_momentum", "chart_bias_bullish", "chart_bias_bearish",
                "chart_confidence", "chart_bullish_pattern", "chart_bearish_pattern",
                # Advanced chart (7)
                "chart_uncertainty", "chart_position_mult", "chart_tf_aligned",
                "chart_alignment", "chart_trend_breaking", "chart_resume_ready", "chart_resume_conf",
                # NEW: MACD features (4)
                "macd_histogram", "macd_cross_bullish", "macd_cross_bearish", "macd_cross_none",
                # NEW: Bollinger Bands (4)
                "bb_bandwidth", "bb_above", "bb_below", "bb_middle",
                # NEW: Stochastic (5)
                "stoch_k", "stoch_d", "stoch_overbought", "stoch_oversold", "stoch_neutral",
                # NEW: RSI Divergence (4)
                "rsi_div_bullish", "rsi_div_bearish", "rsi_div_none", "rsi_div_strength",
                # NEW: Volume (3)
                "volume_ratio", "is_high_volume", "obv_trend",
                # NEW: Heiken Ashi (5)
                "ha_bullish", "ha_bearish", "ha_neutral", "ha_consecutive", "ha_strength",
                # NEW: VWAP (4)
                "vwap_distance", "vwap_above", "vwap_below", "vwap_at",
            ]

        if not self.feature_sums:
            self.feature_sums = [0.0] * self.n_features
            self.feature_sq_sums = [0.0] * self.n_features
            self.outcome_sums = [0.0] * self.n_features

    def update(self, features: List[float], outcome: int):
        """Update feature statistics with new sample."""
        self.total_samples += 1
        self.total_outcomes += outcome
        self.total_outcomes_sq += outcome * outcome

        for i, f in enumerate(features[:self.n_features]):
            if i < len(self.feature_sums):
                self.feature_sums[i] += f
                self.feature_sq_sums[i] += f * f
                self.outcome_sums[i] += f * outcome

    def get_importance(self) -> Dict[str, float]:
        """
        Calculate feature importance scores.

        Uses Pearson correlation coefficient between feature and outcome.
        """
        if self.total_samples < 10:
            return {name: 0.0 for name in self.feature_names}

        importance = {}
        n = self.total_samples

        # Outcome statistics
        y_mean = self.total_outcomes / n
        y_var = (self.total_outcomes_sq / n) - (y_mean ** 2)
        y_std = math.sqrt(max(0, y_var))

        for i, name in enumerate(self.feature_names[:self.n_features]):
            if i >= len(self.feature_sums):
                importance[name] = 0.0
                continue

            # Feature statistics
            x_mean = self.feature_sums[i] / n
            x_var = (self.feature_sq_sums[i] / n) - (x_mean ** 2)
            x_std = math.sqrt(max(0, x_var))

            # Covariance
            cov = (self.outcome_sums[i] / n) - (x_mean * y_mean)

            # Correlation
            if x_std > 0 and y_std > 0:
                correlation = cov / (x_std * y_std)
                # Use absolute correlation as importance
                importance[name] = abs(correlation)
            else:
                importance[name] = 0.0

        return importance

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        importance = self.get_importance()
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]

    def to_dict(self) -> dict:
        return {
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "feature_sums": self.feature_sums,
            "feature_sq_sums": self.feature_sq_sums,
            "outcome_sums": self.outcome_sums,
            "total_samples": self.total_samples,
            "total_outcomes": self.total_outcomes,
            "total_outcomes_sq": self.total_outcomes_sq,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureImportanceTracker":
        tracker = cls(
            n_features=data.get("n_features", 83),
            feature_names=data.get("feature_names", []),
        )
        # Handle migration from smaller feature sets
        old_sums = data.get("feature_sums", [])
        old_sq_sums = data.get("feature_sq_sums", [])
        old_outcome_sums = data.get("outcome_sums", [])

        # Extend arrays if needed
        tracker.feature_sums = old_sums + [0.0] * (tracker.n_features - len(old_sums)) if len(old_sums) < tracker.n_features else old_sums[:tracker.n_features]
        tracker.feature_sq_sums = old_sq_sums + [0.0] * (tracker.n_features - len(old_sq_sums)) if len(old_sq_sums) < tracker.n_features else old_sq_sums[:tracker.n_features]
        tracker.outcome_sums = old_outcome_sums + [0.0] * (tracker.n_features - len(old_outcome_sums)) if len(old_outcome_sums) < tracker.n_features else old_outcome_sums[:tracker.n_features]

        tracker.total_samples = data.get("total_samples", 0)
        tracker.total_outcomes = data.get("total_outcomes", 0.0)
        tracker.total_outcomes_sq = data.get("total_outcomes_sq", 0.0)
        return tracker


# =============================================================================
# ENSEMBLE VOTING
# =============================================================================

@dataclass
class ModelWrapper:
    """Wrapper for different model types to provide uniform interface."""
    name: str
    model: any
    weight: float = 1.0
    predictions_made: int = 0
    correct_predictions: int = 0

    def predict_proba(self, features: List[float]) -> float:
        """Get prediction from wrapped model."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(features)
        return 0.5

    def update(self, features: List[float], outcome: int):
        """Update wrapped model if it supports online learning."""
        if hasattr(self.model, 'update'):
            self.model.update(features, outcome)
        elif hasattr(self.model, 'partial_fit'):
            self.model.partial_fit(features, outcome)

    def record_prediction(self, predicted_win: bool, actual_win: bool):
        """Track prediction accuracy."""
        self.predictions_made += 1
        if predicted_win == actual_win:
            self.correct_predictions += 1

    @property
    def accuracy(self) -> float:
        if self.predictions_made == 0:
            return 0.5
        return self.correct_predictions / self.predictions_made


@dataclass
class EnsembleVoter:
    """
    Ensemble that combines multiple models using voting.

    Supports:
    - Weighted average (soft voting)
    - Majority vote (hard voting)
    - Dynamic weight adjustment based on recent performance
    """

    models: List[ModelWrapper] = field(default_factory=list)
    voting_method: str = "soft"  # "soft" (weighted avg) or "hard" (majority)
    dynamic_weights: bool = True  # Adjust weights based on performance

    # Tracking
    total_predictions: int = 0
    ensemble_correct: int = 0

    def add_model(self, name: str, model: any, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models.append(ModelWrapper(name=name, model=model, weight=weight))

    def _update_weights(self):
        """Update model weights based on recent accuracy."""
        if not self.dynamic_weights:
            return

        # Only update if we have enough predictions
        min_predictions = 10
        eligible_models = [m for m in self.models if m.predictions_made >= min_predictions]

        if not eligible_models:
            return

        # Calculate accuracy-based weights
        accuracies = [m.accuracy for m in eligible_models]
        min_acc = min(accuracies)
        max_acc = max(accuracies)

        # Normalize to 0.5-1.5 range
        for model in eligible_models:
            if max_acc > min_acc:
                normalized = (model.accuracy - min_acc) / (max_acc - min_acc)
                model.weight = 0.5 + normalized  # Range: 0.5 to 1.5
            else:
                model.weight = 1.0

    def predict_proba(self, features: List[float]) -> Tuple[float, Dict[str, float]]:
        """
        Get ensemble prediction.

        Returns:
            Tuple of (probability, individual_predictions_dict)
        """
        if not self.models:
            return 0.5, {}

        individual_preds = {}

        if self.voting_method == "hard":
            # Hard voting - majority vote
            votes_for = 0
            votes_against = 0

            for model in self.models:
                prob = model.predict_proba(features)
                individual_preds[model.name] = prob

                if prob >= 0.5:
                    votes_for += model.weight
                else:
                    votes_against += model.weight

            total_votes = votes_for + votes_against
            ensemble_prob = votes_for / total_votes if total_votes > 0 else 0.5

        else:
            # Soft voting - weighted average
            weighted_sum = 0.0
            total_weight = 0.0

            for model in self.models:
                prob = model.predict_proba(features)
                individual_preds[model.name] = prob
                weighted_sum += prob * model.weight
                total_weight += model.weight

            ensemble_prob = weighted_sum / total_weight if total_weight > 0 else 0.5

        return ensemble_prob, individual_preds

    def update(self, features: List[float], outcome: int):
        """Update all models and track accuracy."""
        # Get predictions before update
        ensemble_prob, individual_preds = self.predict_proba(features)
        ensemble_predicted_win = ensemble_prob >= 0.5
        actual_win = outcome == 1

        # Track ensemble accuracy
        self.total_predictions += 1
        if ensemble_predicted_win == actual_win:
            self.ensemble_correct += 1

        # Update individual models and track their accuracy
        for model in self.models:
            model_prob = individual_preds.get(model.name, 0.5)
            model_predicted_win = model_prob >= 0.5
            model.record_prediction(model_predicted_win, actual_win)
            model.update(features, outcome)

        # Periodically update weights
        if self.total_predictions % 5 == 0:
            self._update_weights()

    def get_stats(self) -> dict:
        """Get ensemble statistics."""
        model_stats = []
        for m in self.models:
            model_stats.append({
                "name": m.name,
                "weight": m.weight,
                "accuracy": m.accuracy,
                "predictions": m.predictions_made,
            })

        return {
            "voting_method": self.voting_method,
            "n_models": len(self.models),
            "ensemble_accuracy": self.ensemble_correct / max(1, self.total_predictions),
            "total_predictions": self.total_predictions,
            "models": model_stats,
        }

    def to_dict(self) -> dict:
        return {
            "voting_method": self.voting_method,
            "dynamic_weights": self.dynamic_weights,
            "total_predictions": self.total_predictions,
            "ensemble_correct": self.ensemble_correct,
            # Note: Individual models are saved separately
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EnsembleVoter":
        return cls(
            voting_method=data.get("voting_method", "soft"),
            dynamic_weights=data.get("dynamic_weights", True),
            total_predictions=data.get("total_predictions", 0),
            ensemble_correct=data.get("ensemble_correct", 0),
        )


# =============================================================================
# MODEL VERSIONING / A/B TESTING
# =============================================================================

@dataclass
class ModelVersion:
    """Represents a version of the ML model for A/B testing."""
    version_id: str
    created_at: str
    description: str

    # Performance metrics
    predictions: int = 0
    correct: int = 0
    total_pnl: float = 0.0

    # Model state (serialized)
    model_state: dict = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / max(1, self.predictions)

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / max(1, self.predictions)

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "created_at": self.created_at,
            "description": self.description,
            "predictions": self.predictions,
            "correct": self.correct,
            "total_pnl": self.total_pnl,
            "model_state": self.model_state,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelVersion":
        return cls(
            version_id=data.get("version_id", "v1"),
            created_at=data.get("created_at", ""),
            description=data.get("description", ""),
            predictions=data.get("predictions", 0),
            correct=data.get("correct", 0),
            total_pnl=data.get("total_pnl", 0.0),
            model_state=data.get("model_state", {}),
        )


@dataclass
class ModelVersionManager:
    """
    Manages multiple model versions for A/B testing.

    Features:
    - Track multiple model versions
    - Compare performance between versions
    - Automatically promote best performing version
    - Rollback to previous version if needed
    """

    versions: Dict[str, ModelVersion] = field(default_factory=dict)
    active_version: str = "v1"
    challenger_version: Optional[str] = None

    # A/B test settings
    challenger_traffic_pct: float = 0.2  # 20% traffic to challenger
    min_samples_for_comparison: int = 50
    auto_promote: bool = True
    promotion_threshold: float = 0.05  # Challenger must be 5% better

    # Path for persistence
    versions_path: str = ""

    def __post_init__(self):
        if not self.versions_path:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.versions_path = os.path.join(project_root, "model_versions.json")
        self._load_versions()

    def create_version(
        self,
        version_id: str,
        description: str,
        model_state: dict,
    ) -> ModelVersion:
        """Create a new model version."""
        version = ModelVersion(
            version_id=version_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            description=description,
            model_state=model_state,
        )
        self.versions[version_id] = version
        self._save_versions()
        logger.info(f"Created model version: {version_id}")
        return version

    def get_active_version(self) -> Optional[ModelVersion]:
        """Get the currently active model version."""
        return self.versions.get(self.active_version)

    def get_model_for_prediction(self) -> Tuple[str, Optional[ModelVersion]]:
        """
        Get which model version to use for a prediction.

        Returns:
            Tuple of (version_id, version_object)
        """
        # If no challenger, always use active
        if not self.challenger_version or self.challenger_version not in self.versions:
            return self.active_version, self.versions.get(self.active_version)

        # Randomly assign to challenger based on traffic percentage
        if random.random() < self.challenger_traffic_pct:
            return self.challenger_version, self.versions.get(self.challenger_version)

        return self.active_version, self.versions.get(self.active_version)

    def record_prediction(
        self,
        version_id: str,
        predicted_win: bool,
        actual_win: bool,
        pnl: float = 0.0,
    ):
        """Record a prediction outcome for a version."""
        if version_id not in self.versions:
            return

        version = self.versions[version_id]
        version.predictions += 1
        if predicted_win == actual_win:
            version.correct += 1
        version.total_pnl += pnl

        # Check for auto-promotion
        if self.auto_promote:
            self._check_promotion()

        # Save periodically
        if version.predictions % 10 == 0:
            self._save_versions()

    def _check_promotion(self):
        """Check if challenger should be promoted to active."""
        if not self.challenger_version:
            return

        active = self.versions.get(self.active_version)
        challenger = self.versions.get(self.challenger_version)

        if not active or not challenger:
            return

        # Need minimum samples
        if challenger.predictions < self.min_samples_for_comparison:
            return

        # Compare accuracy
        if challenger.accuracy > active.accuracy + self.promotion_threshold:
            logger.info(
                f"Promoting challenger {self.challenger_version} "
                f"(accuracy: {challenger.accuracy:.1%}) over "
                f"{self.active_version} (accuracy: {active.accuracy:.1%})"
            )
            self.active_version = self.challenger_version
            self.challenger_version = None
            self._save_versions()

    def set_challenger(self, version_id: str):
        """Set a version as the challenger for A/B testing."""
        if version_id in self.versions and version_id != self.active_version:
            self.challenger_version = version_id
            logger.info(f"Set challenger version: {version_id}")
            self._save_versions()

    def get_comparison(self) -> dict:
        """Get comparison between active and challenger."""
        active = self.versions.get(self.active_version)
        challenger = self.versions.get(self.challenger_version) if self.challenger_version else None

        result = {
            "active": {
                "version_id": self.active_version,
                "accuracy": active.accuracy if active else 0,
                "predictions": active.predictions if active else 0,
                "avg_pnl": active.avg_pnl if active else 0,
            }
        }

        if challenger:
            result["challenger"] = {
                "version_id": self.challenger_version,
                "accuracy": challenger.accuracy,
                "predictions": challenger.predictions,
                "avg_pnl": challenger.avg_pnl,
            }
            result["accuracy_diff"] = challenger.accuracy - active.accuracy if active else 0

        return result

    def _save_versions(self):
        """Save versions to disk."""
        try:
            data = {
                "active_version": self.active_version,
                "challenger_version": self.challenger_version,
                "versions": {k: v.to_dict() for k, v in self.versions.items()},
            }
            with open(self.versions_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model versions: {e}")

    def _load_versions(self):
        """Load versions from disk."""
        try:
            if os.path.exists(self.versions_path):
                with open(self.versions_path, "r") as f:
                    data = json.load(f)

                self.active_version = data.get("active_version", "v1")
                self.challenger_version = data.get("challenger_version")
                self.versions = {
                    k: ModelVersion.from_dict(v)
                    for k, v in data.get("versions", {}).items()
                }
                logger.info(f"Loaded {len(self.versions)} model versions")
        except Exception as e:
            logger.warning(f"Could not load model versions: {e}")


# =============================================================================
# ENHANCED ML PREDICTOR (COMBINES ALL ENHANCEMENTS)
# =============================================================================

@dataclass
class EnhancedMLPredictor:
    """
    Enhanced ML Predictor that combines all ML improvements:
    - Ensemble voting with multiple model types
    - Feature importance tracking
    - Gradient boosting model
    - Model versioning and A/B testing
    - Kelly criterion integration
    """

    # Core components
    ensemble: EnsembleVoter = field(default_factory=EnsembleVoter)
    feature_tracker: FeatureImportanceTracker = field(default_factory=FeatureImportanceTracker)
    version_manager: ModelVersionManager = field(default_factory=ModelVersionManager)

    # Models in ensemble
    gbm: GradientBoostingClassifier = field(default_factory=GradientBoostingClassifier)

    # Settings
    min_confidence: float = 0.55
    min_training_samples: int = 10
    use_kelly: bool = True

    # Tracking
    training_samples: int = 0
    model_path: str = ""

    def __post_init__(self):
        """Initialize the enhanced predictor."""
        if not self.model_path:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.model_path = os.path.join(project_root, "ml_enhanced.json")

        # Add models to ensemble
        self.ensemble.add_model("gbm", self.gbm, weight=1.0)

        # Load saved state
        self._load()

    def predict_proba(self, features: List[float]) -> Tuple[float, dict]:
        """
        Get prediction from enhanced ensemble.

        Returns:
            Tuple of (probability, details_dict)
        """
        ensemble_prob, individual_preds = self.ensemble.predict_proba(features)

        # Get feature importance
        top_features = self.feature_tracker.get_top_features(5)

        details = {
            "ensemble_probability": ensemble_prob,
            "individual_predictions": individual_preds,
            "top_features": top_features,
            "training_samples": self.training_samples,
            "ensemble_stats": self.ensemble.get_stats(),
        }

        return ensemble_prob, details

    def should_trade(
        self,
        features: List[float],
        edge: float,
        market_price: float,
    ) -> Tuple[bool, float, str, dict]:
        """
        Decide if we should take this trade.

        Returns:
            Tuple of (should_trade, confidence, reason, details)
        """
        # Get prediction
        confidence, details = self.predict_proba(features)

        # Check minimum samples
        if self.training_samples < self.min_training_samples:
            return (
                True,
                confidence,
                f"Learning mode ({self.training_samples}/{self.min_training_samples} samples)",
                details,
            )

        # Check confidence threshold
        if confidence < self.min_confidence:
            return (
                False,
                confidence,
                f"Ensemble rejected: {confidence:.1%} < {self.min_confidence:.1%}",
                details,
            )

        # Calculate Kelly position size if enabled
        if self.use_kelly:
            from .kelly import calculate_kelly_position
            # Kelly calculation would go here
            details["kelly_enabled"] = True

        return (
            True,
            confidence,
            f"Ensemble approved: {confidence:.1%}",
            details,
        )

    def record_outcome(self, features: List[float], outcome: int, pnl: float = 0.0):
        """Record trade outcome and update all components."""
        # Update ensemble
        self.ensemble.update(features, outcome)

        # Update feature tracker
        self.feature_tracker.update(features, outcome)

        # Update gradient boosting
        self.gbm.partial_fit(features, outcome)

        # Track for version manager
        version_id, _ = self.version_manager.get_model_for_prediction()
        self.version_manager.record_prediction(
            version_id=version_id,
            predicted_win=self.ensemble.predict_proba(features)[0] >= 0.5,
            actual_win=outcome == 1,
            pnl=pnl,
        )

        self.training_samples += 1

        # Save periodically
        if self.training_samples % 5 == 0:
            self._save()

        # Log progress
        if self.training_samples % 10 == 0:
            stats = self.ensemble.get_stats()
            logger.info(
                f"🤖 Enhanced ML: {self.training_samples} samples | "
                f"Ensemble accuracy: {stats['ensemble_accuracy']:.1%}"
            )

    def get_stats(self) -> dict:
        """Get comprehensive statistics."""
        return {
            "training_samples": self.training_samples,
            "ensemble": self.ensemble.get_stats(),
            "feature_importance": self.feature_tracker.get_top_features(10),
            "version_comparison": self.version_manager.get_comparison(),
            "gbm_samples": self.gbm.training_samples,
        }

    def _save(self):
        """Save state to disk."""
        try:
            data = {
                "training_samples": self.training_samples,
                "ensemble": self.ensemble.to_dict(),
                "feature_tracker": self.feature_tracker.to_dict(),
                "gbm": self.gbm.to_dict(),
            }
            with open(self.model_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Enhanced ML saved: {self.training_samples} samples")
        except Exception as e:
            logger.error(f"Failed to save enhanced ML: {e}")

    def _load(self):
        """Load state from disk."""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, "r") as f:
                    data = json.load(f)

                self.training_samples = data.get("training_samples", 0)

                if "ensemble" in data:
                    self.ensemble = EnsembleVoter.from_dict(data["ensemble"])

                if "feature_tracker" in data:
                    self.feature_tracker = FeatureImportanceTracker.from_dict(data["feature_tracker"])

                if "gbm" in data:
                    self.gbm = GradientBoostingClassifier.from_dict(data["gbm"])
                    # Re-add to ensemble
                    self.ensemble.models = []
                    self.ensemble.add_model("gbm", self.gbm, weight=1.0)

                logger.info(f"Enhanced ML loaded: {self.training_samples} samples")
        except Exception as e:
            logger.warning(f"Could not load enhanced ML: {e}")


# Singleton instance
_enhanced_predictor: Optional[EnhancedMLPredictor] = None


def get_enhanced_predictor() -> EnhancedMLPredictor:
    """Get or create the enhanced ML predictor singleton."""
    global _enhanced_predictor
    if _enhanced_predictor is None:
        _enhanced_predictor = EnhancedMLPredictor()
    return _enhanced_predictor
