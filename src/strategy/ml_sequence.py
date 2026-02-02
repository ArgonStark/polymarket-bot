"""
LSTM-Style Sequence Memory Module

Implements pattern matching on price sequences without heavy ML dependencies.
Stores recent price sequences per asset and finds similar historical patterns
to predict future movements.

Key Features:
1. Sequence storage - Last N price sequences per asset
2. Pattern matching - Find similar historical sequences using DTW-like distance
3. Outcome tracking - Track what happened after each pattern
4. Recurrent memory - Simple recurrent state for continuity
"""

import json
import logging
import math
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PriceSequence:
    """
    Represents a price sequence with its outcome.
    """
    asset: str
    timestamp: str
    prices: List[float]  # Normalized price changes
    volumes: List[float]  # Normalized volumes
    outcome: Optional[int] = None  # 1 = price went up, 0 = price went down, None = unknown
    outcome_magnitude: float = 0.0  # How much it moved (%)

    def to_dict(self) -> dict:
        return {
            "asset": self.asset,
            "timestamp": self.timestamp,
            "prices": self.prices,
            "volumes": self.volumes,
            "outcome": self.outcome,
            "outcome_magnitude": self.outcome_magnitude,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PriceSequence":
        return cls(
            asset=data.get("asset", ""),
            timestamp=data.get("timestamp", ""),
            prices=data.get("prices", []),
            volumes=data.get("volumes", []),
            outcome=data.get("outcome"),
            outcome_magnitude=data.get("outcome_magnitude", 0.0),
        )


@dataclass
class RecurrentState:
    """
    Simple recurrent state that captures recent market behavior.
    Acts as a "memory" of what's been happening.
    """
    # Momentum (exponential moving average of returns)
    momentum: float = 0.0

    # Volatility state (recent variability)
    volatility_state: float = 0.0

    # Trend state (-1 to +1)
    trend_state: float = 0.0

    # Last N outcomes for this asset
    recent_outcomes: List[int] = field(default_factory=list)

    # Decay factor for updating states
    decay: float = 0.9

    def update(self, price_change: float, outcome: Optional[int] = None):
        """Update recurrent state with new observation."""
        # Update momentum (EMA of price changes)
        self.momentum = self.decay * self.momentum + (1 - self.decay) * price_change

        # Update volatility state (EMA of absolute changes)
        self.volatility_state = self.decay * self.volatility_state + (1 - self.decay) * abs(price_change)

        # Update trend state based on direction
        direction = 1.0 if price_change > 0 else (-1.0 if price_change < 0 else 0.0)
        self.trend_state = self.decay * self.trend_state + (1 - self.decay) * direction

        # Track outcomes
        if outcome is not None:
            self.recent_outcomes.append(outcome)
            # Keep only last 20 outcomes
            if len(self.recent_outcomes) > 20:
                self.recent_outcomes = self.recent_outcomes[-20:]

    def get_win_rate(self) -> float:
        """Get recent win rate."""
        if not self.recent_outcomes:
            return 0.5
        return sum(self.recent_outcomes) / len(self.recent_outcomes)

    def to_dict(self) -> dict:
        return {
            "momentum": self.momentum,
            "volatility_state": self.volatility_state,
            "trend_state": self.trend_state,
            "recent_outcomes": self.recent_outcomes,
            "decay": self.decay,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RecurrentState":
        state = cls(
            momentum=data.get("momentum", 0.0),
            volatility_state=data.get("volatility_state", 0.0),
            trend_state=data.get("trend_state", 0.0),
            decay=data.get("decay", 0.9),
        )
        state.recent_outcomes = data.get("recent_outcomes", [])
        return state


@dataclass
class SequenceMemory:
    """
    LSTM-style sequence memory for price pattern matching.

    Stores sequences per asset and finds similar historical patterns
    to help predict future movements.
    """
    # Settings
    sequence_length: int = 15  # Number of price points in each sequence
    max_sequences_per_asset: int = 200  # Maximum sequences to store per asset
    similarity_threshold: float = 0.7  # Minimum similarity to consider a match
    min_matches_for_prediction: int = 5  # Minimum matches needed to make prediction

    # Stored sequences per asset
    sequences: Dict[str, List[PriceSequence]] = field(default_factory=dict)

    # Recurrent state per asset
    states: Dict[str, RecurrentState] = field(default_factory=dict)

    # Persistence path
    model_path: str = ""

    def __post_init__(self):
        """Initialize and load saved state."""
        if not self.model_path:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.model_path = os.path.join(project_root, "ml_sequences.json")
        self._load()

    def _normalize_sequence(self, prices: List[float]) -> List[float]:
        """
        Normalize a price sequence to relative changes.
        This makes patterns comparable across different price levels.
        """
        if len(prices) < 2:
            return []

        # Convert to percentage changes
        changes = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0:
                change = (prices[i] - prices[i - 1]) / prices[i - 1]
                # Clip extreme values
                change = max(-0.1, min(0.1, change))
                changes.append(change)
            else:
                changes.append(0.0)

        return changes

    def _calculate_similarity(self, seq1: List[float], seq2: List[float]) -> float:
        """
        Calculate similarity between two sequences using dynamic time warping-like approach.

        Returns similarity score between 0 and 1.
        """
        if not seq1 or not seq2:
            return 0.0

        # Simple approach: correlation coefficient
        n = min(len(seq1), len(seq2))
        if n < 3:
            return 0.0

        seq1 = seq1[:n]
        seq2 = seq2[:n]

        # Calculate means
        mean1 = sum(seq1) / n
        mean2 = sum(seq2) / n

        # Calculate correlation
        numerator = sum((seq1[i] - mean1) * (seq2[i] - mean2) for i in range(n))
        denom1 = math.sqrt(sum((x - mean1) ** 2 for x in seq1))
        denom2 = math.sqrt(sum((x - mean2) ** 2 for x in seq2))

        if denom1 < 1e-10 or denom2 < 1e-10:
            return 0.0

        correlation = numerator / (denom1 * denom2)

        # Convert correlation (-1 to 1) to similarity (0 to 1)
        # correlation of 1 = similarity of 1
        # correlation of -1 = similarity of 0 (opposite pattern)
        # correlation of 0 = similarity of 0.5 (no relationship)
        similarity = (correlation + 1) / 2

        return similarity

    def add_sequence(
        self,
        asset: str,
        prices: List[float],
        volumes: Optional[List[float]] = None,
        outcome: Optional[int] = None,
        outcome_magnitude: float = 0.0,
    ):
        """
        Add a new price sequence to memory.

        Args:
            asset: Asset symbol (BTC, ETH, etc.)
            prices: Raw prices (will be normalized)
            volumes: Optional volume data
            outcome: 1 if price went up, 0 if down, None if unknown
            outcome_magnitude: Percentage move that occurred
        """
        # Normalize prices to changes
        normalized = self._normalize_sequence(prices)
        if len(normalized) < 5:  # Need minimum sequence length
            return

        # Normalize volumes if provided
        norm_volumes = []
        if volumes and len(volumes) > 1:
            avg_vol = sum(volumes) / len(volumes) if volumes else 1.0
            norm_volumes = [v / avg_vol if avg_vol > 0 else 1.0 for v in volumes]

        # Create sequence
        seq = PriceSequence(
            asset=asset,
            timestamp=datetime.now(timezone.utc).isoformat(),
            prices=normalized,
            volumes=norm_volumes,
            outcome=outcome,
            outcome_magnitude=outcome_magnitude,
        )

        # Add to asset's sequences
        if asset not in self.sequences:
            self.sequences[asset] = []

        self.sequences[asset].append(seq)

        # Limit memory per asset
        if len(self.sequences[asset]) > self.max_sequences_per_asset:
            self.sequences[asset] = self.sequences[asset][-self.max_sequences_per_asset:]

        # Update recurrent state
        if asset not in self.states:
            self.states[asset] = RecurrentState()

        if normalized:
            self.states[asset].update(normalized[-1], outcome)

        # Log sequence addition
        outcome_str = "WIN" if outcome == 1 else ("LOSS" if outcome == 0 else "PENDING")
        logger.debug(
            f"📊 SEQ [{asset}]: Added sequence | "
            f"len={len(normalized)} | outcome={outcome_str} | "
            f"total_seqs={len(self.sequences[asset])}"
        )

        # Save periodically
        total_seqs = sum(len(s) for s in self.sequences.values())
        if total_seqs % 10 == 0:
            self._save()

    def find_similar_patterns(
        self,
        asset: str,
        current_prices: List[float],
        top_k: int = 10,
    ) -> List[Tuple[PriceSequence, float]]:
        """
        Find similar historical patterns for the current sequence.

        Args:
            asset: Asset symbol
            current_prices: Current price sequence (raw prices)
            top_k: Number of top matches to return

        Returns:
            List of (sequence, similarity_score) tuples
        """
        if asset not in self.sequences:
            return []

        # Normalize current sequence
        current_normalized = self._normalize_sequence(current_prices)
        if len(current_normalized) < 5:
            return []

        # Find similar sequences
        matches = []
        for seq in self.sequences[asset]:
            if seq.outcome is None:  # Skip sequences without known outcomes
                continue

            similarity = self._calculate_similarity(current_normalized, seq.prices)

            if similarity >= self.similarity_threshold:
                matches.append((seq, similarity))

        # Sort by similarity (descending)
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches[:top_k]

    def predict_from_patterns(
        self,
        asset: str,
        current_prices: List[float],
    ) -> Tuple[float, float, str]:
        """
        Predict outcome based on similar historical patterns.

        Args:
            asset: Asset symbol
            current_prices: Current price sequence

        Returns:
            Tuple of (probability_up, confidence, reason)
        """
        matches = self.find_similar_patterns(asset, current_prices)

        if len(matches) < self.min_matches_for_prediction:
            logger.debug(
                f"📊 SEQ [{asset}]: Insufficient pattern matches | "
                f"found={len(matches)} | required={self.min_matches_for_prediction}"
            )
            return 0.5, 0.0, f"Insufficient matches ({len(matches)}/{self.min_matches_for_prediction})"

        # Calculate weighted outcome based on similarity
        weighted_up = 0.0
        total_weight = 0.0

        for seq, similarity in matches:
            if seq.outcome is not None:
                # Weight by similarity squared (emphasize closer matches)
                weight = similarity ** 2
                weighted_up += weight * seq.outcome
                total_weight += weight

        if total_weight < 0.01:
            return 0.5, 0.0, "No weighted outcomes"

        prob_up = weighted_up / total_weight

        # Calculate confidence based on:
        # 1. Number of matches
        # 2. Consistency of outcomes
        # 3. Average similarity
        n_matches = len(matches)
        avg_similarity = sum(s for _, s in matches) / n_matches

        # Check outcome consistency
        outcomes = [seq.outcome for seq, _ in matches if seq.outcome is not None]
        if outcomes:
            outcome_variance = sum((o - prob_up) ** 2 for o in outcomes) / len(outcomes)
            consistency = 1.0 - min(1.0, outcome_variance * 4)  # 0 to 1
        else:
            consistency = 0.0

        # Combine factors into confidence
        match_factor = min(1.0, n_matches / 20)  # Max out at 20 matches
        confidence = (avg_similarity * 0.4 + consistency * 0.4 + match_factor * 0.2)

        reason = f"Pattern match: {n_matches} similar, avg_sim={avg_similarity:.2f}, consistency={consistency:.2f}"

        # Log prediction result
        direction = "UP" if prob_up > 0.5 else ("DOWN" if prob_up < 0.5 else "NEUTRAL")
        logger.info(
            f"📊 SEQ PREDICT [{asset}]: {direction} {prob_up:.1%} | "
            f"confidence={confidence:.1%} | matches={n_matches} | "
            f"avg_sim={avg_similarity:.2f}"
        )

        return prob_up, confidence, reason

    def get_recurrent_features(self, asset: str) -> Dict[str, float]:
        """
        Get recurrent state features for an asset.

        These can be added to the ML feature vector.
        """
        if asset not in self.states:
            return {
                "seq_momentum": 0.0,
                "seq_volatility": 0.0,
                "seq_trend": 0.5,  # Neutral
                "seq_win_rate": 0.5,
            }

        state = self.states[asset]
        return {
            "seq_momentum": max(-1.0, min(1.0, state.momentum * 100)),  # Normalize
            "seq_volatility": min(1.0, state.volatility_state * 100),
            "seq_trend": (state.trend_state + 1) / 2,  # -1,1 -> 0,1
            "seq_win_rate": state.get_win_rate(),
        }

    def get_stats(self) -> dict:
        """Get memory statistics."""
        total_sequences = sum(len(s) for s in self.sequences.values())
        sequences_with_outcomes = sum(
            sum(1 for seq in seqs if seq.outcome is not None)
            for seqs in self.sequences.values()
        )

        return {
            "total_sequences": total_sequences,
            "sequences_with_outcomes": sequences_with_outcomes,
            "assets_tracked": list(self.sequences.keys()),
            "sequences_per_asset": {k: len(v) for k, v in self.sequences.items()},
        }

    def _save(self):
        """Save memory to disk."""
        try:
            data = {
                "sequence_length": self.sequence_length,
                "max_sequences_per_asset": self.max_sequences_per_asset,
                "similarity_threshold": self.similarity_threshold,
                "sequences": {
                    asset: [s.to_dict() for s in seqs[-100:]]  # Save last 100 per asset
                    for asset, seqs in self.sequences.items()
                },
                "states": {
                    asset: state.to_dict()
                    for asset, state in self.states.items()
                },
            }
            with open(self.model_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Sequence memory saved: {sum(len(s) for s in self.sequences.values())} sequences")
        except Exception as e:
            logger.error(f"Failed to save sequence memory: {e}")

    def _load(self):
        """Load memory from disk."""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, "r") as f:
                    data = json.load(f)

                self.sequence_length = data.get("sequence_length", 15)
                self.max_sequences_per_asset = data.get("max_sequences_per_asset", 200)
                self.similarity_threshold = data.get("similarity_threshold", 0.7)

                self.sequences = {
                    asset: [PriceSequence.from_dict(s) for s in seqs]
                    for asset, seqs in data.get("sequences", {}).items()
                }

                self.states = {
                    asset: RecurrentState.from_dict(state)
                    for asset, state in data.get("states", {}).items()
                }

                logger.info(f"Sequence memory loaded: {sum(len(s) for s in self.sequences.values())} sequences")
        except Exception as e:
            logger.warning(f"Could not load sequence memory: {e}")


# Singleton instance
_sequence_memory: Optional[SequenceMemory] = None


def get_sequence_memory() -> SequenceMemory:
    """Get or create the sequence memory singleton."""
    global _sequence_memory
    if _sequence_memory is None:
        _sequence_memory = SequenceMemory()
    return _sequence_memory


def update_sequence_memory(
    asset: str,
    prices: List[float],
    volumes: Optional[List[float]] = None,
    outcome: Optional[int] = None,
    outcome_magnitude: float = 0.0,
):
    """Convenience function to update sequence memory."""
    memory = get_sequence_memory()
    memory.add_sequence(asset, prices, volumes, outcome, outcome_magnitude)


def predict_from_sequences(
    asset: str,
    current_prices: List[float],
) -> Tuple[float, float, str]:
    """Convenience function to get prediction from sequences."""
    memory = get_sequence_memory()
    return memory.predict_from_patterns(asset, current_prices)
