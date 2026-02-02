"""
Train ML Model from Enriched Historical Data

This tool trains the ML model using enriched historical trade data
that includes full market context (indicators, prices, etc.).

Usage:
    python -m src.tools.train_from_enriched --input data/enriched_trades.json

Features:
- Converts enriched trade data to TradeFeatures format
- Splits data into train/validation/test sets
- Trains with cross-validation
- Reports feature importance
- Saves trained model
"""

import json
import logging
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingStats:
    """Statistics from training."""
    total_samples: int = 0
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    test_accuracy: float = 0.0
    win_rate: float = 0.0
    feature_importance: Dict[str, float] = None


def load_enriched_data(filepath: str) -> List[Dict]:
    """Load enriched trade data from JSON."""
    with open(filepath) as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} enriched trades from {filepath}")
    return data


def filter_valid_trades(trades: List[Dict]) -> List[Dict]:
    """Filter to only trades with known outcomes and valid data."""
    valid = []
    for trade in trades:
        # Must have outcome
        if trade.get("outcome") not in ["WIN", "LOSS"]:
            continue

        # Must have price data
        if trade.get("asset_price", 0) <= 0:
            continue

        # Must have basic indicators
        if trade.get("rsi_14") is None:
            continue

        valid.append(trade)

    logger.info(f"Filtered to {len(valid)} valid trades (from {len(trades)})")
    return valid


def enriched_to_features(trade: Dict) -> List[float]:
    """
    Convert enriched trade to 82-feature vector.

    This maps the enriched data fields to the TradeFeatures format
    used by the ML predictor.
    """
    # Helper functions
    def safe_float(val, default=0.0):
        try:
            return float(val) if val is not None else default
        except:
            return default

    def one_hot_side(side: str) -> Tuple[float, float]:
        """One-hot encode side: (is_up, is_down)"""
        if side == "UP":
            return (1.0, 0.0)
        elif side == "DOWN":
            return (0.0, 1.0)
        return (0.0, 0.0)

    def one_hot_arb_type(arb: str) -> Tuple[float, float, float, float]:
        """One-hot encode arb type."""
        # For historical data, assume "synced" since we don't have arb info
        return (0.0, 0.0, 1.0, 0.0)  # none, binary, synced, lagging

    def one_hot_confirmation(conf: str) -> Tuple[float, float, float, float]:
        """One-hot encode confirmation type."""
        # For historical, use MEDIUM as default
        return (0.0, 1.0, 0.0, 0.0)  # none, weak, medium, strong

    def one_hot_macd_cross(signal: str) -> Tuple[float, float, float]:
        """One-hot MACD crossover."""
        if signal == "bullish_cross":
            return (1.0, 0.0, 0.0)
        elif signal == "bearish_cross":
            return (0.0, 1.0, 0.0)
        return (0.0, 0.0, 1.0)  # none

    def one_hot_bb_position(pos: str) -> Tuple[float, float, float]:
        """One-hot BB position."""
        if pos == "above":
            return (1.0, 0.0, 0.0)
        elif pos == "below":
            return (0.0, 1.0, 0.0)
        return (0.0, 0.0, 1.0)  # middle

    def one_hot_stoch_signal(sig: str) -> Tuple[float, float, float]:
        """One-hot stochastic signal."""
        if sig == "overbought":
            return (1.0, 0.0, 0.0)
        elif sig == "oversold":
            return (0.0, 1.0, 0.0)
        return (0.0, 0.0, 1.0)  # neutral

    def one_hot_rsi_div(div: str) -> Tuple[float, float, float]:
        """One-hot RSI divergence."""
        if div == "bullish":
            return (1.0, 0.0, 0.0)
        elif div == "bearish":
            return (0.0, 1.0, 0.0)
        return (0.0, 0.0, 1.0)  # none

    def one_hot_ha_trend(trend: str) -> Tuple[float, float, float]:
        """One-hot Heiken Ashi trend."""
        if trend == "bullish":
            return (1.0, 0.0, 0.0)
        elif trend == "bearish":
            return (0.0, 1.0, 0.0)
        return (0.0, 0.0, 1.0)  # neutral

    def one_hot_vwap_pos(pos: str) -> Tuple[float, float, float]:
        """One-hot VWAP position."""
        if pos == "above":
            return (1.0, 0.0, 0.0)
        elif pos == "below":
            return (0.0, 1.0, 0.0)
        return (0.0, 0.0, 1.0)  # at

    # Build feature vector (must match TradeFeatures.to_vector() order)
    features = []

    # 1. Volatility (normalized)
    volatility = safe_float(trade.get("volatility_24h"), 0.02)
    features.append(min(1.0, volatility * 10))  # Normalize

    # 2. Price momentum (normalized)
    momentum = safe_float(trade.get("trend_1h"), 0) / 100
    features.append(max(-1, min(1, momentum)))

    # 3-6. Side one-hot (up, down)
    is_up, is_down = one_hot_side(trade.get("side", ""))
    features.extend([is_up, is_down])

    # 7-10. Arb type one-hot
    features.extend(one_hot_arb_type("synced"))

    # 11. Spread (normalized)
    spread = 0.02  # Assume typical spread for historical
    features.append(min(1.0, spread * 10))

    # 12-13. Depth (normalized)
    features.append(0.5)  # bid_depth normalized
    features.append(0.5)  # ask_depth normalized

    # 14. Price trend
    price_trend = safe_float(trade.get("trend_4h"), 0) / 100
    features.append(max(-1, min(1, price_trend)))

    # 15. Distance from target
    # For 15-min markets, use entry price deviation from 0.5
    entry_price = safe_float(trade.get("entry_price"), 0.5)
    distance = abs(entry_price - 0.5)
    features.append(min(1.0, distance * 2))

    # 16. Binance lead percentage
    features.append(0.0)  # Not available in historical

    # 17-20. Binance confirmation one-hot
    features.extend(one_hot_confirmation("MEDIUM"))

    # 21-23. Multi-timeframe trends
    features.append(max(-1, min(1, safe_float(trade.get("trend_1h"), 0) / 100)))
    features.append(max(-1, min(1, safe_float(trade.get("trend_4h"), 0) / 100)))
    features.append(max(-1, min(1, safe_float(trade.get("trend_1d"), 0) / 100)))

    # 24-27. Price observation features
    asset_price = safe_float(trade.get("asset_price"), 50000)
    btc_price = safe_float(trade.get("btc_price"), 50000)

    # Normalize price (log scale relative to typical range)
    price_normalized = min(1.0, max(0.0, asset_price / 100000))
    features.append(price_normalized)

    # Price above/below target (use entry price as proxy)
    features.append(1.0 if entry_price > 0.5 else 0.0)

    # Price range position
    features.append(0.5)  # Default to middle

    # Price velocity
    features.append(max(-1, min(1, safe_float(trade.get("trend_1h"), 0) / 10)))

    # 28. Edge (use entry price deviation as proxy)
    edge = abs(entry_price - 0.5) * 2
    features.append(min(1.0, edge))

    # 29. Time remaining (normalized) - assume mid-market
    features.append(0.5)

    # 30. RSI (normalized 0-1)
    rsi = safe_float(trade.get("rsi_14"), 50)
    features.append(rsi / 100)

    # 31. Trend strength (from RSI deviation)
    trend_strength = abs(rsi - 50) / 50
    features.append(trend_strength)

    # 32-34. Trend direction one-hot
    trend_dir = trade.get("trend_direction", "neutral")
    features.append(1.0 if trend_dir == "up" else 0.0)
    features.append(1.0 if trend_dir == "down" else 0.0)
    features.append(1.0 if trend_dir == "neutral" else 0.0)

    # 35-36. Reversal signals (from stochastic)
    stoch_sig = trade.get("stoch_signal", "neutral")
    features.append(1.0 if stoch_sig == "oversold" else 0.0)  # bullish reversal
    features.append(1.0 if stoch_sig == "overbought" else 0.0)  # bearish reversal

    # 37. Momentum (from MACD)
    macd = safe_float(trade.get("macd_histogram"), 0)
    features.append(max(-1, min(1, macd * 1000)))

    # 38-39. Bias one-hot
    features.append(1.0 if macd > 0 else 0.0)  # bullish bias
    features.append(1.0 if macd < 0 else 0.0)  # bearish bias

    # 40. Confidence (from indicator agreement)
    confidence = 0.5
    if (trend_dir == "up" and stoch_sig == "oversold") or (trend_dir == "down" and stoch_sig == "overbought"):
        confidence = 0.7
    features.append(confidence)

    # 41-42. Pattern signals (from candlestick patterns)
    pattern = trade.get("candle_pattern", "none")
    features.append(1.0 if "bullish" in pattern.lower() else 0.0)
    features.append(1.0 if "bearish" in pattern.lower() else 0.0)

    # 43. Uncertainty score (inverse of confidence)
    features.append(1.0 - confidence)

    # 44. Position multiplier (default 1.0)
    features.append(1.0)

    # 45. Timeframe aligned (default 1.0)
    features.append(1.0)

    # 46. Alignment score
    features.append(0.8)  # Assume reasonable alignment

    # 47. Trend breaking (from consolidation)
    features.append(1.0 if trade.get("consolidating", False) else 0.0)

    # 48. Resume ready
    features.append(1.0)

    # 49. Resume confidence
    features.append(0.8)

    # ========== NEW INDICATOR FEATURES (50-82) ==========

    # 50. MACD histogram (normalized)
    features.append(max(-1, min(1, macd * 1000)))

    # 51-53. MACD crossover one-hot
    features.extend(one_hot_macd_cross(trade.get("macd_signal", "none")))

    # 54. BB bandwidth
    bb_bw = safe_float(trade.get("bb_bandwidth"), 0.1)
    features.append(min(1.0, bb_bw * 5))

    # 55-57. BB position one-hot
    features.extend(one_hot_bb_position(trade.get("bb_position", "middle")))

    # 58. Stochastic K (normalized)
    stoch_k = safe_float(trade.get("stoch_k"), 50)
    features.append(stoch_k / 100)

    # 59. Stochastic D (normalized)
    stoch_d = safe_float(trade.get("stoch_d"), 50)
    features.append(stoch_d / 100)

    # 60-62. Stochastic signal one-hot
    features.extend(one_hot_stoch_signal(trade.get("stoch_signal", "neutral")))

    # 63-65. RSI divergence one-hot (derive from RSI vs price)
    # If price making highs but RSI lower = bearish divergence
    # If price making lows but RSI higher = bullish divergence
    rsi_div = "none"
    if trade.get("higher_high") and rsi < 50:
        rsi_div = "bearish"
    elif trade.get("lower_low") and rsi > 50:
        rsi_div = "bullish"
    features.extend(one_hot_rsi_div(rsi_div))

    # 66. RSI divergence strength
    features.append(abs(rsi - 50) / 50 if rsi_div != "none" else 0.0)

    # 67. Volume ratio (normalized)
    vol_ratio = safe_float(trade.get("volume_ratio"), 1.0)
    features.append(min(1.0, vol_ratio / 3))

    # 68. Is high volume
    features.append(1.0 if vol_ratio > 1.5 else 0.0)

    # 69. OBV trend (normalized)
    obv_trend = safe_float(trade.get("obv_trend"), 0)
    features.append(max(-1, min(1, obv_trend)))

    # 70-72. Heiken Ashi trend one-hot
    ha_trend = trade.get("heiken_ashi_trend", "neutral")
    features.extend(one_hot_ha_trend(ha_trend))

    # 73. HA consecutive candles (normalized)
    ha_consec = safe_float(trade.get("ha_consecutive"), 0)
    features.append(min(1.0, ha_consec / 10))

    # 74. HA strength
    features.append(0.5 if ha_trend != "neutral" else 0.0)

    # 75. VWAP distance % (normalized)
    vwap_dist = safe_float(trade.get("price_vs_vwap"), 0)
    features.append(max(-1, min(1, vwap_dist / 5)))

    # 76-78. VWAP position one-hot
    vwap_pos = "at"
    if vwap_dist > 1:
        vwap_pos = "above"
    elif vwap_dist < -1:
        vwap_pos = "below"
    features.extend(one_hot_vwap_pos(vwap_pos))

    # Verify we have exactly 82 features
    assert len(features) == 82, f"Expected 82 features, got {len(features)}"

    return features


def split_data(
    trades: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = True
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data into train/val/test sets."""
    data = trades.copy()

    if shuffle:
        random.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    return train, val, test


def train_model(
    train_data: List[Dict],
    val_data: List[Dict],
    learning_rate: float = 0.01,
    epochs: int = 100
) -> Tuple[List[float], float]:
    """
    Train a simple logistic regression model.

    Returns:
        Tuple of (weights, val_accuracy)
    """
    # Initialize weights
    n_features = 82
    weights = [random.uniform(-0.1, 0.1) for _ in range(n_features)]
    bias = 0.0

    def sigmoid(x):
        if x > 500:
            return 1.0
        elif x < -500:
            return 0.0
        return 1.0 / (1.0 + (2.71828 ** -x))

    def predict(features):
        z = sum(w * f for w, f in zip(weights, features)) + bias
        return sigmoid(z)

    # Convert to features
    train_X = [enriched_to_features(t) for t in train_data]
    train_y = [1.0 if t["outcome"] == "WIN" else 0.0 for t in train_data]

    val_X = [enriched_to_features(t) for t in val_data]
    val_y = [1.0 if t["outcome"] == "WIN" else 0.0 for t in val_data]

    best_val_acc = 0.0
    best_weights = weights.copy()

    for epoch in range(epochs):
        # Shuffle training data
        combined = list(zip(train_X, train_y))
        random.shuffle(combined)
        train_X_shuffled, train_y_shuffled = zip(*combined)

        # Training
        for features, label in zip(train_X_shuffled, train_y_shuffled):
            pred = predict(features)
            error = label - pred

            # Update weights (gradient descent)
            for i in range(n_features):
                weights[i] += learning_rate * error * features[i]
            bias += learning_rate * error

        # Validation accuracy
        correct = 0
        for features, label in zip(val_X, val_y):
            pred = 1.0 if predict(features) >= 0.5 else 0.0
            if pred == label:
                correct += 1

        val_acc = correct / len(val_y) if val_y else 0.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = weights.copy()

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} | Val accuracy: {val_acc:.2%}")

    return best_weights, best_val_acc


def evaluate_model(
    weights: List[float],
    test_data: List[Dict]
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model on test data."""
    bias = 0.0

    def sigmoid(x):
        if x > 500:
            return 1.0
        elif x < -500:
            return 0.0
        return 1.0 / (1.0 + (2.71828 ** -x))

    def predict(features):
        z = sum(w * f for w, f in zip(weights, features)) + bias
        return sigmoid(z)

    test_X = [enriched_to_features(t) for t in test_data]
    test_y = [1.0 if t["outcome"] == "WIN" else 0.0 for t in test_data]

    correct = 0
    tp = fp = tn = fn = 0

    for features, label in zip(test_X, test_y):
        pred = 1.0 if predict(features) >= 0.5 else 0.0
        if pred == label:
            correct += 1
        if pred == 1.0 and label == 1.0:
            tp += 1
        elif pred == 1.0 and label == 0.0:
            fp += 1
        elif pred == 0.0 and label == 0.0:
            tn += 1
        else:
            fn += 1

    accuracy = correct / len(test_y) if test_y else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }

    return accuracy, metrics


def get_feature_importance(weights: List[float]) -> Dict[str, float]:
    """Get feature importance from weights."""
    feature_names = [
        "volatility", "momentum", "side_up", "side_down",
        "arb_none", "arb_binary", "arb_synced", "arb_lagging",
        "spread", "bid_depth", "ask_depth", "price_trend", "distance_target",
        "binance_lead", "conf_none", "conf_weak", "conf_medium", "conf_strong",
        "trend_1h", "trend_4h", "trend_1d",
        "price_norm", "price_above", "price_range", "price_velocity",
        "edge", "time_remaining",
        "rsi", "trend_strength", "trend_up", "trend_down", "trend_neutral",
        "reversal_bullish", "reversal_bearish", "momentum_chart",
        "bias_bullish", "bias_bearish", "confidence",
        "pattern_bullish", "pattern_bearish",
        "uncertainty", "pos_mult", "tf_aligned", "align_score",
        "trend_break", "resume_ready", "resume_conf",
        # New indicators
        "macd_hist", "macd_bull", "macd_bear", "macd_none",
        "bb_bandwidth", "bb_above", "bb_below", "bb_middle",
        "stoch_k", "stoch_d", "stoch_ob", "stoch_os", "stoch_neutral",
        "rsi_div_bull", "rsi_div_bear", "rsi_div_none", "rsi_div_strength",
        "vol_ratio", "high_vol", "obv_trend",
        "ha_bull", "ha_bear", "ha_neutral", "ha_consec", "ha_strength",
        "vwap_dist", "vwap_above", "vwap_below", "vwap_at",
    ]

    # Ensure we have enough names
    while len(feature_names) < len(weights):
        feature_names.append(f"feature_{len(feature_names)}")

    importance = {}
    for name, weight in zip(feature_names, weights):
        importance[name] = abs(weight)

    # Sort by importance
    sorted_importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

    return sorted_importance


def save_model(
    weights: List[float],
    output_path: str,
    metrics: Dict[str, float],
    feature_importance: Dict[str, float]
):
    """Save trained model to file."""
    model_data = {
        "model": {
            "weights": weights,
            "bias": 0.0,
            "n_features": len(weights),
        },
        "training_metrics": metrics,
        "feature_importance": dict(list(feature_importance.items())[:20]),  # Top 20
        "trained_at": datetime.now().isoformat(),
        "training_samples": metrics.get("train_samples", 0),
        "predictions_made": 0,
        "correct_predictions": 0,
        "min_confidence": 0.52,
    }

    with open(output_path, "w") as f:
        json.dump(model_data, f, indent=2)

    logger.info(f"Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ML model from enriched trades")
    parser.add_argument("--input", default="data/enriched_trades.json", help="Input enriched trades file")
    parser.add_argument("--output", default="models/ml_model.json", help="Output model file")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    trades = load_enriched_data(args.input)

    # Filter valid trades
    valid_trades = filter_valid_trades(trades)

    if len(valid_trades) < 100:
        logger.error(f"Not enough valid trades for training ({len(valid_trades)})")
        return

    # Split data
    train, val, test = split_data(valid_trades)

    # Calculate win rate
    total_wins = sum(1 for t in valid_trades if t["outcome"] == "WIN")
    win_rate = total_wins / len(valid_trades)
    logger.info(f"Overall win rate: {win_rate:.2%}")

    # Train model
    logger.info("Training model...")
    weights, val_acc = train_model(train, val, learning_rate=args.learning_rate, epochs=args.epochs)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_acc, metrics = evaluate_model(weights, test)

    # Get feature importance
    feature_importance = get_feature_importance(weights)

    # Print results
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Training samples: {len(train)}")
    logger.info(f"Validation accuracy: {val_acc:.2%}")
    logger.info(f"Test accuracy: {test_acc:.2%}")
    logger.info(f"Test precision: {metrics['precision']:.2%}")
    logger.info(f"Test recall: {metrics['recall']:.2%}")
    logger.info(f"Test F1: {metrics['f1']:.2%}")
    logger.info("")
    logger.info("Top 10 Important Features:")
    for i, (name, imp) in enumerate(list(feature_importance.items())[:10]):
        logger.info(f"  {i+1}. {name}: {imp:.4f}")

    # Add training info to metrics
    metrics["train_samples"] = len(train)
    metrics["val_samples"] = len(val)
    metrics["test_samples"] = len(test)
    metrics["val_accuracy"] = val_acc
    metrics["win_rate"] = win_rate

    # Save model
    save_model(weights, args.output, metrics, feature_importance)

    # If accuracy is good, copy to main model location
    if test_acc > 0.55:
        main_model_path = "models/ml_model.json"
        save_model(weights, main_model_path, metrics, feature_importance)
        logger.info(f"Model also saved to {main_model_path} (accuracy > 55%)")
    else:
        logger.warning(f"Model accuracy ({test_acc:.2%}) below 55% threshold - not deploying")


if __name__ == "__main__":
    main()
