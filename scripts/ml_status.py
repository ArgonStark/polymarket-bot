#!/usr/bin/env python3
"""
ML Model Status Monitor

Shows current state of the ML predictor including:
- Training samples and accuracy
- Hybrid model status
- Feature weights (top influencers)
- Recent predictions
"""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_ml_model():
    """Load ML model from disk."""
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ml_model.json"
    )

    if not os.path.exists(model_path):
        return None

    with open(model_path, "r") as f:
        return json.load(f)


def load_trader_model():
    """Load trader model from disk."""
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "trader_model.json"
    )

    if not os.path.exists(model_path):
        return None

    with open(model_path, "r") as f:
        return json.load(f)


def get_feature_names():
    """Get names for the 29 features."""
    return [
        # Core features (6)
        "edge", "time_remaining", "volatility", "price_momentum", "hour", "day",
        # Asset one-hot (4)
        "is_BTC", "is_ETH", "is_SOL", "is_XRP",
        # Side (1)
        "is_UP",
        # Arb type one-hot (5)
        "arb_none", "arb_binary", "arb_asymmetric", "arb_dump", "arb_hedge",
        # Market features (4)
        "spread", "bid_depth", "ask_depth", "price_trend",
        # Distance (1)
        "distance_from_target",
        # Binance features (5)
        "binance_lead", "bn_NONE", "bn_WEAK", "bn_MEDIUM", "bn_STRONG",
        # Trends (3)
        "trend_1h", "trend_4h", "trend_1d",
    ]


def get_simple_feature_names():
    """Get names for the 11 simple features used by hybrid model."""
    return [
        "price", "size", "usdc_size", "hour", "day", "time_remaining",
        "is_BTC", "is_ETH", "is_SOL", "is_XRP", "is_UP"
    ]


def print_header(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_ml_status():
    """Print ML model status."""
    data = load_ml_model()

    print_header("ML MODEL STATUS")

    if data is None:
        print("\n  No ML model found (ml_model.json)")
        print("  The model will be created after your first trade.")
        return

    # Basic stats
    samples = data.get("training_samples", 0)
    predictions = data.get("predictions_made", 0)
    correct = data.get("correct_predictions", 0)
    accuracy = correct / predictions if predictions > 0 else 0

    print(f"\n  Training Samples: {samples}")
    print(f"  Predictions Made: {predictions}")
    print(f"  Correct Predictions: {correct}")
    print(f"  Accuracy: {accuracy:.1%}")

    # Learning mode status
    if samples < 25:
        print(f"\n  Mode: LEARNING ({samples}/25 samples)")
        print("  Status: Taking ALL trades to collect training data")
    elif samples < 40:
        print(f"\n  Mode: EARLY FILTERING (threshold: 45%)")
    elif samples < 60:
        print(f"\n  Mode: MEDIUM FILTERING (threshold: 50%)")
    else:
        print(f"\n  Mode: FULL FILTERING (threshold: 55%)")

    # Logistic regression weights
    model_data = data.get("model", {})
    weights = model_data.get("weights", [])
    bias = model_data.get("bias", 0)

    if weights:
        feature_names = get_feature_names()

        # Get top positive and negative weights
        weight_pairs = list(zip(feature_names[:len(weights)], weights))
        sorted_weights = sorted(weight_pairs, key=lambda x: abs(x[1]), reverse=True)

        print(f"\n  Logistic Regression Bias: {bias:.4f}")
        print("\n  Top 10 Feature Weights (most influential):")
        for name, weight in sorted_weights[:10]:
            direction = "+" if weight > 0 else ""
            bar = "█" * int(abs(weight) * 20)
            print(f"    {name:20s}: {direction}{weight:+.4f} {bar}")

    # Hybrid predictor status
    hybrid_data = data.get("hybrid_predictor")
    if hybrid_data:
        print_header("HYBRID PREDICTOR")

        hybrid_samples = hybrid_data.get("training_samples", 0)
        base_weight = hybrid_data.get("base_weight", 0.7)

        print(f"\n  Hybrid Training Samples: {hybrid_samples}")
        print(f"  Base Weight: {base_weight:.0%} (trader model)")
        print(f"  Adaptive Weight: {1-base_weight:.0%} (neural network)")

        # Neural network info
        nn_data = hybrid_data.get("adaptive_model", {})
        if nn_data:
            input_size = nn_data.get("input_size", 11)
            hidden_size = nn_data.get("hidden_size", 16)
            lr = nn_data.get("learning_rate", 0.05)
            print(f"\n  Neural Network Architecture: {input_size} -> {hidden_size} -> 1")
            print(f"  Learning Rate: {lr}")


def print_trader_model_status():
    """Print trader model status."""
    data = load_trader_model()

    print_header("TRADER MODEL (Pre-trained)")

    if data is None:
        print("\n  No trader model found (trader_model.json)")
        print("  Run: python scripts/train_from_trader.py")
        print("  This trains on successful trader patterns for better predictions.")
        return

    # Model info
    model_data = data.get("model", {})
    trees = model_data.get("trees", [])
    feature_names = model_data.get("feature_names", [])

    print(f"\n  Random Forest Trees: {len(trees)}")
    print(f"  Features: {len(feature_names)}")

    # Training info
    training = data.get("training_info", {})
    if training:
        total_trades = training.get("total_trades", 0)
        wins = training.get("wins", 0)
        losses = training.get("losses", 0)
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

        print(f"\n  Trained on: {total_trades} trades")
        print(f"  Trader Win Rate: {win_rate:.1%}")

    # Metrics
    metrics = data.get("metrics", {})
    if metrics:
        test = metrics.get("test", {})
        if test:
            print(f"\n  Test Accuracy: {test.get('accuracy', 0):.1%}")
            print(f"  Test Precision: {test.get('precision', 0):.1%}")
            print(f"  Test Recall: {test.get('recall', 0):.1%}")


def print_recommendations():
    """Print recommendations based on current state."""
    data = load_ml_model()
    trader_data = load_trader_model()

    print_header("RECOMMENDATIONS")

    recommendations = []

    if data is None:
        recommendations.append("Start the bot to begin ML training")
        recommendations.append("Or run: python scripts/sync_ml_from_trades.py to backfill from history")
    else:
        samples = data.get("training_samples", 0)
        predictions = data.get("predictions_made", 0)
        correct = data.get("correct_predictions", 0)
        accuracy = correct / predictions if predictions > 0 else 0

        if samples < 25:
            recommendations.append(f"Collect {25 - samples} more trades for ML filtering to activate")
            recommendations.append("Tip: Run 'python scripts/sync_ml_from_trades.py' to backfill from history")

        if predictions >= 10 and accuracy < 0.45:
            recommendations.append("WARNING: Accuracy below 45% - consider pausing to analyze")

        if predictions >= 20 and accuracy > 0.55:
            recommendations.append("Good accuracy! Consider increasing position size")

    if trader_data is None:
        recommendations.append("Train trader model: python scripts/train_from_trader.py")

    if not recommendations:
        recommendations.append("Everything looks good!")

    for i, rec in enumerate(recommendations, 1):
        print(f"\n  {i}. {rec}")


def main():
    print("\n" + "="*60)
    print("         POLYMARKET BOT - ML STATUS MONITOR")
    print("="*60)

    print_ml_status()
    print_trader_model_status()
    print_recommendations()

    print("\n" + "="*60)
    print("  Run this anytime: python scripts/ml_status.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
