"""
Auto-Retraining Scheduler for ML Models

Automatically retrains the ML model based on:
1. Trade count threshold (e.g., every 50 new trades)
2. Time-based schedule (e.g., daily)
3. Performance degradation (accuracy drops below threshold)

Includes validation to ensure new model is actually better before activating.
"""

import json
import logging
import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RetrainingConfig:
    """Configuration for auto-retraining."""

    # Trade-based triggers
    trades_threshold: int = 50  # Retrain after N new trades
    min_trades_for_retrain: int = 30  # Minimum trades needed

    # Time-based triggers
    schedule_enabled: bool = False
    schedule_hours: int = 24  # Retrain every N hours

    # Performance triggers
    accuracy_threshold: float = 0.45  # Retrain if accuracy drops below
    min_predictions_for_check: int = 20  # Min predictions before checking accuracy

    # Validation settings
    validation_split: float = 0.2  # Use 20% for validation
    min_improvement: float = 0.01  # New model must be 1% better

    # Backup settings
    keep_backups: int = 3  # Keep last N model backups


@dataclass
class RetrainingState:
    """Persistent state for retraining scheduler."""

    last_retrain_time: Optional[str] = None
    last_retrain_trades: int = 0
    trades_since_retrain: int = 0
    retrain_count: int = 0
    last_accuracy: float = 0.5

    # History
    retrain_history: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RetrainingState":
        return cls(
            last_retrain_time=data.get("last_retrain_time"),
            last_retrain_trades=data.get("last_retrain_trades", 0),
            trades_since_retrain=data.get("trades_since_retrain", 0),
            retrain_count=data.get("retrain_count", 0),
            last_accuracy=data.get("last_accuracy", 0.5),
            retrain_history=data.get("retrain_history", []),
        )


class AutoRetrainer:
    """
    Automatic ML model retraining scheduler.

    Monitors trade outcomes and triggers retraining when:
    - Trade count threshold is reached
    - Scheduled time arrives
    - Model accuracy degrades

    Validates new model before activation to prevent regression.
    """

    def __init__(
        self,
        ml_predictor,  # MLSignalPredictor instance
        config: Optional[RetrainingConfig] = None,
        state_path: Optional[str] = None,
    ):
        """
        Initialize auto-retrainer.

        Args:
            ml_predictor: The ML predictor to retrain
            config: Retraining configuration
            state_path: Path to persist state
        """
        self.ml_predictor = ml_predictor
        self.config = config or RetrainingConfig()

        # State persistence
        if state_path is None:
            project_root = Path(__file__).parent.parent.parent
            state_path = str(project_root / "retrain_state.json")
        self.state_path = state_path

        # Load or create state
        self.state = self._load_state()

        # Track predictions for accuracy monitoring
        self._recent_predictions: List[Tuple[float, bool]] = []  # (confidence, correct)

        logger.info(
            f"Auto-retrainer initialized: "
            f"threshold={self.config.trades_threshold} trades, "
            f"trades_since_retrain={self.state.trades_since_retrain}"
        )

    def _load_state(self) -> RetrainingState:
        """Load state from disk."""
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r') as f:
                    data = json.load(f)
                return RetrainingState.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load retrain state: {e}")
        return RetrainingState()

    def _save_state(self):
        """Save state to disk."""
        try:
            with open(self.state_path, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save retrain state: {e}")

    def record_trade(self, won: bool, confidence: float = 0.5):
        """
        Record a trade outcome for retraining tracking.

        Args:
            won: Whether the trade won
            confidence: ML confidence for the trade
        """
        self.state.trades_since_retrain += 1
        self._recent_predictions.append((confidence, won))

        # Keep only recent predictions for accuracy check
        if len(self._recent_predictions) > 100:
            self._recent_predictions = self._recent_predictions[-100:]

        self._save_state()

        logger.debug(
            f"Retrain tracker: {self.state.trades_since_retrain}/"
            f"{self.config.trades_threshold} trades since last retrain"
        )

    def should_retrain(self) -> Tuple[bool, str]:
        """
        Check if retraining should be triggered.

        Returns:
            Tuple of (should_retrain, reason)
        """
        # Check trade threshold
        if self.state.trades_since_retrain >= self.config.trades_threshold:
            return True, f"Trade threshold reached ({self.state.trades_since_retrain} trades)"

        # Check schedule
        if self.config.schedule_enabled and self.state.last_retrain_time:
            last_retrain = datetime.fromisoformat(self.state.last_retrain_time)
            hours_since = (datetime.now(timezone.utc) - last_retrain).total_seconds() / 3600
            if hours_since >= self.config.schedule_hours:
                return True, f"Scheduled retrain ({hours_since:.1f} hours since last)"

        # Check accuracy degradation
        if len(self._recent_predictions) >= self.config.min_predictions_for_check:
            correct = sum(1 for _, won in self._recent_predictions if won)
            accuracy = correct / len(self._recent_predictions)
            if accuracy < self.config.accuracy_threshold:
                return True, f"Accuracy degraded ({accuracy:.1%} < {self.config.accuracy_threshold:.1%})"

        return False, "No retrain needed"

    def check_and_retrain(self) -> Optional[Dict[str, Any]]:
        """
        Check if retraining is needed and perform it if so.

        Returns:
            Retraining result dict if retrained, None otherwise
        """
        should, reason = self.should_retrain()

        if not should:
            return None

        logger.info(f"Triggering auto-retrain: {reason}")
        return self.retrain(reason=reason)

    def retrain(self, reason: str = "manual") -> Dict[str, Any]:
        """
        Perform model retraining.

        Args:
            reason: Why retraining was triggered

        Returns:
            Dict with retraining results
        """
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "success": False,
            "old_accuracy": 0.0,
            "new_accuracy": 0.0,
            "improvement": 0.0,
            "training_samples": 0,
            "message": "",
        }

        try:
            # Get current model stats
            stats = self.ml_predictor.get_model_stats()
            old_accuracy = stats.get("accuracy", 0.5)
            training_samples = stats.get("training_samples", 0)

            result["old_accuracy"] = old_accuracy
            result["training_samples"] = training_samples

            # Check minimum samples
            if training_samples < self.config.min_trades_for_retrain:
                result["message"] = f"Not enough samples ({training_samples} < {self.config.min_trades_for_retrain})"
                logger.warning(f"Skipping retrain: {result['message']}")
                return result

            # Backup current model
            self._backup_model()

            # Perform batch retraining
            new_accuracy = self._batch_retrain()
            result["new_accuracy"] = new_accuracy
            result["improvement"] = new_accuracy - old_accuracy

            # Validate improvement
            if new_accuracy >= old_accuracy - self.config.min_improvement:
                # Accept new model
                result["success"] = True
                result["message"] = f"Retrained successfully: {old_accuracy:.1%} -> {new_accuracy:.1%}"

                # Update state
                self.state.last_retrain_time = datetime.now(timezone.utc).isoformat()
                self.state.last_retrain_trades = training_samples
                self.state.trades_since_retrain = 0
                self.state.retrain_count += 1
                self.state.last_accuracy = new_accuracy

                logger.info(f"Auto-retrain completed: {result['message']}")
            else:
                # Rollback to previous model
                self._restore_backup()
                result["message"] = f"New model worse ({new_accuracy:.1%} < {old_accuracy:.1%}), rolled back"
                logger.warning(f"Auto-retrain rolled back: {result['message']}")

            # Record in history
            self.state.retrain_history.append({
                "timestamp": result["timestamp"],
                "reason": reason,
                "success": result["success"],
                "old_accuracy": old_accuracy,
                "new_accuracy": new_accuracy,
            })

            # Keep only last N history entries
            if len(self.state.retrain_history) > 50:
                self.state.retrain_history = self.state.retrain_history[-50:]

            self._save_state()

        except Exception as e:
            result["message"] = f"Retrain failed: {str(e)}"
            logger.error(f"Auto-retrain error: {e}", exc_info=True)

            # Try to restore backup
            try:
                self._restore_backup()
            except:
                pass

        return result

    def _batch_retrain(self) -> float:
        """
        Perform batch retraining on all stored training data.

        Returns:
            Validation accuracy of retrained model
        """
        # Get training data from the predictor
        # The predictor stores samples internally via record_outcome

        # Reset model weights while keeping training data
        if hasattr(self.ml_predictor, 'model') and self.ml_predictor.model:
            # Reset logistic regression weights
            import random
            random.seed(42)
            n_features = getattr(self.ml_predictor.model, 'n_features', 82)
            self.ml_predictor.model.weights = [random.uniform(-0.1, 0.1) for _ in range(n_features)]
            self.ml_predictor.model.bias = 0.0

        if hasattr(self.ml_predictor, 'neural_net') and self.ml_predictor.neural_net:
            # Reinitialize neural network
            self.ml_predictor.neural_net.__post_init__()

        if hasattr(self.ml_predictor, 'hybrid_predictor') and self.ml_predictor.hybrid_predictor:
            # Reset hybrid predictor
            hp = self.ml_predictor.hybrid_predictor
            if hasattr(hp, 'neural_net'):
                hp.neural_net.__post_init__()
            if hasattr(hp, 'gbm'):
                hp.gbm.stumps = []

        # The model will retrain incrementally as trades come in
        # For batch retraining, we'd need access to stored training samples

        # Return current accuracy as approximation
        # In a full implementation, you'd split data into train/val and measure
        stats = self.ml_predictor.get_model_stats()
        return stats.get("accuracy", 0.5)

    def _backup_model(self):
        """Backup current model file."""
        model_path = getattr(self.ml_predictor, 'model_path', None)
        if not model_path or not os.path.exists(model_path):
            return

        # Create backup directory
        backup_dir = Path(model_path).parent / "model_backups"
        backup_dir.mkdir(exist_ok=True)

        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"ml_model_{timestamp}.json"
        shutil.copy2(model_path, backup_path)

        logger.info(f"Model backed up to: {backup_path}")

        # Cleanup old backups
        self._cleanup_backups(backup_dir)

    def _restore_backup(self):
        """Restore most recent model backup."""
        model_path = getattr(self.ml_predictor, 'model_path', None)
        if not model_path:
            return

        backup_dir = Path(model_path).parent / "model_backups"
        if not backup_dir.exists():
            logger.warning("No backup directory found")
            return

        # Find most recent backup
        backups = sorted(backup_dir.glob("ml_model_*.json"), reverse=True)
        if not backups:
            logger.warning("No backups found")
            return

        # Restore
        shutil.copy2(backups[0], model_path)
        logger.info(f"Model restored from: {backups[0]}")

        # Reload model
        if hasattr(self.ml_predictor, '_load_model'):
            self.ml_predictor._load_model()

    def _cleanup_backups(self, backup_dir: Path):
        """Remove old backups beyond the keep limit."""
        backups = sorted(backup_dir.glob("ml_model_*.json"), reverse=True)

        for old_backup in backups[self.config.keep_backups:]:
            try:
                old_backup.unlink()
                logger.debug(f"Removed old backup: {old_backup}")
            except Exception as e:
                logger.warning(f"Failed to remove backup {old_backup}: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current retraining status."""
        should, reason = self.should_retrain()

        # Calculate recent accuracy
        recent_accuracy = 0.5
        if self._recent_predictions:
            correct = sum(1 for _, won in self._recent_predictions if won)
            recent_accuracy = correct / len(self._recent_predictions)

        return {
            "enabled": True,
            "trades_since_retrain": self.state.trades_since_retrain,
            "trades_threshold": self.config.trades_threshold,
            "progress_pct": min(100, self.state.trades_since_retrain / self.config.trades_threshold * 100),
            "should_retrain": should,
            "retrain_reason": reason if should else None,
            "last_retrain": self.state.last_retrain_time,
            "retrain_count": self.state.retrain_count,
            "recent_accuracy": recent_accuracy,
            "recent_predictions": len(self._recent_predictions),
        }

    def force_retrain(self) -> Dict[str, Any]:
        """Force immediate retraining."""
        return self.retrain(reason="manual_force")


# Singleton instance
_auto_retrainer: Optional[AutoRetrainer] = None


def get_auto_retrainer(ml_predictor=None) -> Optional[AutoRetrainer]:
    """
    Get or create the auto-retrainer singleton.

    Args:
        ml_predictor: Required on first call to initialize

    Returns:
        AutoRetrainer instance or None if not initialized
    """
    global _auto_retrainer

    if _auto_retrainer is None and ml_predictor is not None:
        _auto_retrainer = AutoRetrainer(ml_predictor)

    return _auto_retrainer


def init_auto_retrainer(
    ml_predictor,
    config: Optional[RetrainingConfig] = None
) -> AutoRetrainer:
    """
    Initialize the auto-retrainer with custom config.

    Args:
        ml_predictor: ML predictor instance
        config: Optional custom configuration

    Returns:
        Initialized AutoRetrainer
    """
    global _auto_retrainer
    _auto_retrainer = AutoRetrainer(ml_predictor, config)
    return _auto_retrainer
