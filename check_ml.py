#!/usr/bin/env python3
"""Check ML model learning status."""

import json
import os

MODEL_PATH = "ml_model.json"

FEATURE_NAMES = [
    "edge", "time_remaining", "up_price", "spread", "hour_of_day",
    "volatility", "momentum", "is_btc", "is_eth", "is_sol", "is_xrp",
    "arb_none", "arb_binary", "arb_multi", "arb_stale", "arb_reverse",
    "market_spread", "bid_depth", "ask_depth", "price_trend", "distance_from_target"
]

def main():
    if not os.path.exists(MODEL_PATH):
        print("No ML model found yet.")
        print("The model is created after your first completed trade.")
        return

    with open(MODEL_PATH) as f:
        data = json.load(f)

    samples = data.get("training_samples", 0)
    predictions = data.get("predictions_made", 0)
    correct = data.get("correct_predictions", 0)
    min_conf = data.get("min_confidence", 0.55)

    model = data.get("model", {})
    weights = model.get("weights", [])
    bias = model.get("bias", 0)

    print("=" * 50)
    print("ML MODEL STATUS")
    print("=" * 50)
    print(f"Training samples:    {samples}")
    print(f"Predictions made:    {predictions}")
    print(f"Correct predictions: {correct}")
    if predictions > 0:
        accuracy = correct / predictions * 100
        print(f"Accuracy:            {accuracy:.1f}%")
    print(f"Min confidence:      {min_conf:.0%}")
    print(f"Model features:      {len(weights)}")
    print()

    # Learning phase
    if samples < 10:
        print(f"Phase: LEARNING ({samples}/10 trades)")
        print("       All signals allowed while gathering data")
    else:
        print(f"Phase: ACTIVE FILTERING")
        print(f"       Signals below {min_conf:.0%} confidence are blocked")

    print()
    print("=" * 50)
    print("FEATURE WEIGHTS (what the model learned)")
    print("=" * 50)

    # Sort by absolute weight (importance)
    if weights:
        weight_pairs = list(zip(FEATURE_NAMES[:len(weights)], weights))
        weight_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        for name, weight in weight_pairs:
            direction = "+" if weight > 0 else "-"
            bar = "█" * min(int(abs(weight) * 10), 20)
            print(f"  {name:25} {weight:+.3f} {bar}")

    print()
    print(f"Bias: {bias:.3f}")
    print()

    # Interpretation
    print("=" * 50)
    print("INTERPRETATION")
    print("=" * 50)
    if weights:
        positive = [(n, w) for n, w in weight_pairs if w > 0.1]
        negative = [(n, w) for n, w in weight_pairs if w < -0.1]

        if positive:
            print("Factors that INCREASE win probability:")
            for name, w in positive[:5]:
                print(f"  + {name}")

        if negative:
            print("Factors that DECREASE win probability:")
            for name, w in negative[:5]:
                print(f"  - {name}")

if __name__ == "__main__":
    main()
