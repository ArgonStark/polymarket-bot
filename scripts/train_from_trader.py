#!/usr/bin/env python3
"""
Train ML Model from Successful Trader's Data

Downloads trade history from a successful Polymarket trader and trains
a Random Forest model to learn their winning patterns.

Usage:
    python scripts/train_from_trader.py

This will:
1. Download all trades from the successful trader (@0x8dxd)
2. Calculate win/loss for each trade
3. Extract features (time, asset, price, etc.)
4. Train a Random Forest model
5. Save the model for use in the bot
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Successful trader's wallet address
TRADER_WALLET = "0x63ce342161250d705dc0b16df89036c8e5f9ba9a"
TRADER_NAME = "@0x8dxd"

# Polymarket Data API
DATA_API_BASE = "https://data-api.polymarket.com"

# Output paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_OUTPUT = PROJECT_ROOT / "trader_model.json"
DATA_CACHE = PROJECT_ROOT / "trader_data.json"


def fetch_trades(wallet: str, limit: int = 500, offset: int = 0) -> list[dict]:
    """Fetch trades from Polymarket Data API."""
    url = f"{DATA_API_BASE}/activity?user={wallet}&limit={limit}&offset={offset}&type=TRADE"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PolymarketBot/1.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
            return data if isinstance(data, list) else []
    except Exception as e:
        logger.error(f"Failed to fetch trades: {e}")
        return []


def download_all_trades(wallet: str, max_trades: int = 15000) -> list[dict]:
    """Download all trades for a wallet with pagination."""
    all_trades = []
    offset = 0
    limit = 500

    logger.info(f"Downloading trades for {TRADER_NAME} ({wallet[:10]}...)")

    while offset < max_trades:
        trades = fetch_trades(wallet, limit=limit, offset=offset)

        if not trades:
            break

        all_trades.extend(trades)
        offset += limit

        logger.info(f"  Downloaded {len(all_trades)} trades...")

        # Rate limiting
        time.sleep(0.5)

    logger.info(f"Total trades downloaded: {len(all_trades)}")
    return all_trades


def extract_asset_from_slug(slug: str) -> Optional[str]:
    """Extract asset (BTC, ETH, SOL, XRP) from market slug."""
    slug_lower = slug.lower()

    if "btc" in slug_lower or "bitcoin" in slug_lower:
        return "BTC"
    elif "eth" in slug_lower or "ethereum" in slug_lower:
        return "ETH"
    elif "sol" in slug_lower or "solana" in slug_lower:
        return "SOL"
    elif "xrp" in slug_lower or "ripple" in slug_lower:
        return "XRP"

    return None


def is_15min_market(slug: str, title: str = "") -> bool:
    """Check if this is a 15-minute crypto market."""
    combined = f"{slug} {title}".lower()
    return "15m" in combined or "15-min" in combined or "updown-15m" in combined


def extract_market_timestamp(slug: str) -> Optional[int]:
    """Extract market timestamp from slug (e.g., btc-updown-15m-1706123400)."""
    try:
        parts = slug.split("-")
        if len(parts) >= 4:
            return int(parts[-1])
    except (ValueError, IndexError):
        pass
    return None


def determine_outcome(trade: dict, all_trades: list[dict]) -> Optional[bool]:
    """
    Determine if a trade was a win or loss.

    For 15-min markets:
    - If trader bought and later sold at higher price → WIN
    - If market settled and their side won → WIN
    - Otherwise → LOSS
    """
    # Look for corresponding sell or redemption
    condition_id = trade.get("conditionId", "")
    trade_time = trade.get("timestamp", 0)
    side = trade.get("side", "").upper()
    buy_price = float(trade.get("price", 0))

    if side != "BUY":
        return None  # Only analyze buy trades

    # Look for sells of the same position
    for other in all_trades:
        if other.get("conditionId") != condition_id:
            continue
        if other.get("timestamp", 0) <= trade_time:
            continue
        if other.get("side", "").upper() == "SELL":
            sell_price = float(other.get("price", 0))
            return sell_price > buy_price

    # If no sell found, check if it's a 15-min market that settled
    # For 15-min markets, UP wins if price went up, DOWN wins otherwise
    # We'll use a heuristic: if they held to settlement, assume 60% win rate
    # (This trader has 96% win rate, so most holds are wins)

    # Better approach: look for REDEEM transactions
    for other in all_trades:
        if other.get("conditionId") != condition_id:
            continue
        if other.get("type") == "REDEEM":
            # Redemption means they won
            return True

    # Default: assume win if price was good (< 0.5 = undervalued)
    return buy_price < 0.50


def extract_features(trade: dict) -> Optional[dict]:
    """Extract ML features from a trade."""
    slug = trade.get("slug", "") or trade.get("market", "")
    title = trade.get("title", "") or trade.get("name", "")

    # Only process 15-min crypto markets
    if not is_15min_market(slug, title):
        return None

    asset = extract_asset_from_slug(slug)
    if not asset:
        return None

    # Extract basic features
    timestamp = trade.get("timestamp", 0)
    price = float(trade.get("price", 0))
    size = float(trade.get("size", 0))
    usdc_size = float(trade.get("usdcSize", 0))
    side = trade.get("side", "BUY").upper()
    outcome = trade.get("outcome", "").upper()  # "Yes" or "No"

    # Time features
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    hour_of_day = dt.hour
    day_of_week = dt.weekday()

    # Market timestamp for time remaining calculation
    market_ts = extract_market_timestamp(slug)
    if market_ts:
        # Time remaining when trade was made
        time_remaining = max(0, (market_ts + 900) - timestamp)
    else:
        time_remaining = 450  # Default to middle of period

    # Determine if UP or DOWN position
    is_up = outcome.lower() in ["yes", "up"] if outcome else True

    return {
        "asset": asset,
        "side": "UP" if is_up else "DOWN",
        "price": price,
        "size": size,
        "usdc_size": usdc_size,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "time_remaining": time_remaining,
        "timestamp": timestamp,
        "slug": slug,
        "buy_or_sell": side,
    }


def process_trades(trades: list[dict]) -> tuple[list[dict], list[bool]]:
    """Process trades into features and outcomes."""
    features_list = []
    outcomes = []

    # Sort by timestamp
    sorted_trades = sorted(trades, key=lambda t: t.get("timestamp", 0))

    for trade in sorted_trades:
        # Only process BUY trades (entries)
        if trade.get("side", "").upper() != "BUY":
            continue

        features = extract_features(trade)
        if not features:
            continue

        outcome = determine_outcome(trade, sorted_trades)
        if outcome is None:
            continue

        features_list.append(features)
        outcomes.append(outcome)

    return features_list, outcomes


def features_to_vector(features: dict) -> list[float]:
    """Convert features dict to numeric vector for ML."""
    # Normalize features to 0-1 range
    vector = [
        # Price (0-1)
        features["price"],

        # Size (normalized, cap at 100 shares)
        min(features["size"] / 100, 1.0),

        # USD size (normalized, cap at $200)
        min(features["usdc_size"] / 200, 1.0),

        # Time features
        features["hour_of_day"] / 24,
        features["day_of_week"] / 7,

        # Time remaining (normalized to 900s)
        min(features["time_remaining"] / 900, 1.0),

        # Asset one-hot (BTC, ETH, SOL, XRP)
        1.0 if features["asset"] == "BTC" else 0.0,
        1.0 if features["asset"] == "ETH" else 0.0,
        1.0 if features["asset"] == "SOL" else 0.0,
        1.0 if features["asset"] == "XRP" else 0.0,

        # Side (UP=1, DOWN=0)
        1.0 if features["side"] == "UP" else 0.0,
    ]

    return vector


class SimpleRandomForest:
    """
    Simple Random Forest implementation without sklearn.
    Uses decision stumps (single-feature splits) for simplicity.
    """

    def __init__(self, n_trees: int = 50, max_depth: int = 5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        self.feature_names = [
            "price", "size", "usdc_size", "hour", "day", "time_remaining",
            "is_btc", "is_eth", "is_sol", "is_xrp", "is_up"
        ]

    def _bootstrap_sample(self, X: list, y: list) -> tuple[list, list]:
        """Create bootstrap sample."""
        import random
        n = len(X)
        indices = [random.randint(0, n-1) for _ in range(n)]
        return [X[i] for i in indices], [y[i] for i in indices]

    def _gini(self, y: list) -> float:
        """Calculate Gini impurity."""
        if not y:
            return 0
        p = sum(y) / len(y)
        return 2 * p * (1 - p)

    def _find_best_split(self, X: list, y: list, features_to_try: list) -> tuple:
        """Find best split for a node."""
        best_gain = -1
        best_feature = 0
        best_threshold = 0.5

        current_gini = self._gini(y)

        for feature_idx in features_to_try:
            # Get unique values for this feature
            values = sorted(set(x[feature_idx] for x in X))

            # Try thresholds between consecutive values
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i+1]) / 2

                # Split data
                left_y = [y[j] for j in range(len(X)) if X[j][feature_idx] <= threshold]
                right_y = [y[j] for j in range(len(X)) if X[j][feature_idx] > threshold]

                if not left_y or not right_y:
                    continue

                # Calculate information gain
                n = len(y)
                gain = current_gini - (
                    len(left_y)/n * self._gini(left_y) +
                    len(right_y)/n * self._gini(right_y)
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: list, y: list, depth: int = 0) -> dict:
        """Build a decision tree recursively."""
        import random

        # Base cases
        if depth >= self.max_depth or len(set(y)) == 1 or len(y) < 5:
            return {"leaf": True, "prediction": sum(y) / len(y) if y else 0.5}

        # Random feature subset (sqrt of total features)
        n_features = len(X[0])
        n_try = max(1, int(n_features ** 0.5))
        features_to_try = random.sample(range(n_features), min(n_try, n_features))

        # Find best split
        feature, threshold, gain = self._find_best_split(X, y, features_to_try)

        if gain <= 0:
            return {"leaf": True, "prediction": sum(y) / len(y)}

        # Split data
        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(len(X)):
            if X[i][feature] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])

        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(left_X, left_y, depth + 1),
            "right": self._build_tree(right_X, right_y, depth + 1),
        }

    def _predict_tree(self, tree: dict, x: list) -> float:
        """Predict using a single tree."""
        if tree["leaf"]:
            return tree["prediction"]

        if x[tree["feature"]] <= tree["threshold"]:
            return self._predict_tree(tree["left"], x)
        else:
            return self._predict_tree(tree["right"], x)

    def fit(self, X: list, y: list):
        """Train the random forest."""
        import random
        random.seed(42)

        self.trees = []
        for i in range(self.n_trees):
            # Bootstrap sample
            X_sample, y_sample = self._bootstrap_sample(X, y)

            # Build tree
            tree = self._build_tree(X_sample, y_sample)
            self.trees.append(tree)

            if (i + 1) % 10 == 0:
                logger.info(f"  Built {i+1}/{self.n_trees} trees...")

    def predict_proba(self, x: list) -> float:
        """Predict probability of win."""
        if not self.trees:
            return 0.5

        predictions = [self._predict_tree(tree, x) for tree in self.trees]
        return sum(predictions) / len(predictions)

    def predict(self, X: list) -> list[float]:
        """Predict for multiple samples."""
        return [self.predict_proba(x) for x in X]

    def feature_importance(self) -> dict[str, float]:
        """Calculate feature importance (simplified)."""
        importance = {name: 0.0 for name in self.feature_names}

        def count_feature_uses(tree: dict, counts: dict):
            if tree["leaf"]:
                return
            feature_name = self.feature_names[tree["feature"]]
            counts[feature_name] = counts.get(feature_name, 0) + 1
            count_feature_uses(tree["left"], counts)
            count_feature_uses(tree["right"], counts)

        for tree in self.trees:
            count_feature_uses(tree, importance)

        # Normalize
        total = sum(importance.values()) or 1
        return {k: v/total for k, v in importance.items()}

    def to_dict(self) -> dict:
        """Serialize model."""
        return {
            "n_trees": self.n_trees,
            "max_depth": self.max_depth,
            "trees": self.trees,
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimpleRandomForest":
        """Deserialize model."""
        model = cls(n_trees=data["n_trees"], max_depth=data["max_depth"])
        model.trees = data["trees"]
        model.feature_names = data.get("feature_names", model.feature_names)
        return model


def evaluate_model(model: SimpleRandomForest, X: list, y: list) -> dict:
    """Evaluate model performance."""
    predictions = model.predict(X)

    # Convert to binary predictions
    pred_binary = [1 if p >= 0.5 else 0 for p in predictions]

    # Calculate metrics
    correct = sum(1 for p, actual in zip(pred_binary, y) if p == actual)
    accuracy = correct / len(y) if y else 0

    # Precision, recall for wins
    true_positives = sum(1 for p, a in zip(pred_binary, y) if p == 1 and a == 1)
    predicted_positives = sum(pred_binary)
    actual_positives = sum(y)

    precision = true_positives / predicted_positives if predicted_positives else 0
    recall = true_positives / actual_positives if actual_positives else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "total_samples": len(y),
        "wins": sum(y),
        "losses": len(y) - sum(y),
    }


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info(f"Training ML Model from {TRADER_NAME}'s Trade History")
    logger.info("=" * 60)

    # Check for cached data
    if DATA_CACHE.exists():
        logger.info(f"Loading cached data from {DATA_CACHE}")
        with open(DATA_CACHE) as f:
            trades = json.load(f)
    else:
        # Download trades
        trades = download_all_trades(TRADER_WALLET)

        if not trades:
            logger.error("No trades downloaded. Check internet connection.")
            return

        # Cache the data
        logger.info(f"Caching data to {DATA_CACHE}")
        with open(DATA_CACHE, "w") as f:
            json.dump(trades, f)

    logger.info(f"\nTotal trades in dataset: {len(trades)}")

    # Process trades
    logger.info("\nProcessing trades...")
    features_list, outcomes = process_trades(trades)

    logger.info(f"Processed {len(features_list)} 15-min crypto trades")
    logger.info(f"  Wins: {sum(outcomes)} ({sum(outcomes)/len(outcomes)*100:.1f}%)")
    logger.info(f"  Losses: {len(outcomes) - sum(outcomes)}")

    if len(features_list) < 50:
        logger.warning("Not enough trades to train a reliable model.")
        return

    # Convert to vectors
    X = [features_to_vector(f) for f in features_list]
    y = [1 if o else 0 for o in outcomes]

    # Split into train/test (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"\nTraining set: {len(X_train)} trades")
    logger.info(f"Test set: {len(X_test)} trades")

    # Train model
    logger.info("\nTraining Random Forest...")
    model = SimpleRandomForest(n_trees=50, max_depth=5)
    model.fit(X_train, y_train)

    # Evaluate
    logger.info("\nEvaluating model...")
    train_metrics = evaluate_model(model, X_train, y_train)
    test_metrics = evaluate_model(model, X_test, y_test)

    logger.info(f"\nTraining Accuracy: {train_metrics['accuracy']*100:.1f}%")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']*100:.1f}%")
    logger.info(f"Test Precision: {test_metrics['precision']*100:.1f}%")
    logger.info(f"Test Recall: {test_metrics['recall']*100:.1f}%")

    # Feature importance
    importance = model.feature_importance()
    logger.info("\nFeature Importance:")
    for name, imp in sorted(importance.items(), key=lambda x: -x[1]):
        if imp > 0.01:
            logger.info(f"  {name}: {imp*100:.1f}%")

    # Save model
    logger.info(f"\nSaving model to {MODEL_OUTPUT}")
    model_data = {
        "model": model.to_dict(),
        "metrics": {
            "train": train_metrics,
            "test": test_metrics,
        },
        "feature_importance": importance,
        "trained_on": TRADER_NAME,
        "total_trades": len(features_list),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(MODEL_OUTPUT, "w") as f:
        json.dump(model_data, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {MODEL_OUTPUT}")
    logger.info("=" * 60)

    # Summary
    logger.info(f"""
Next steps:
1. The model has learned from {TRADER_NAME}'s {len(features_list)} trades
2. Test accuracy: {test_metrics['accuracy']*100:.1f}%
3. The bot will now use this model to filter signals
4. Run the bot to see it in action!
""")


if __name__ == "__main__":
    main()
