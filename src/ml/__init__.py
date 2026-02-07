"""ML package."""

from .feature_builder import FeatureBuilder, FeatureSchema
from .dataset import Dataset, DatasetSplit
from .interface import (
    MLInterface,
    NoOpML,
    MLInput,
    MLDecision,
    TradeSide,
    MarketContext,
    TrendContext,
    ChartContext,
    IndicatorContext,
    PriceContext,
)
from .predictor import MLPredictor, create_ml_input_from_signal
from .train import main as train_main
from .infer import ModelBundle, InferenceEngine, InferenceOutput
from .policy import Policy, PolicyConfig, PolicyDecision

__all__ = [
    "MLInterface",
    "MLPredictor",
    "NoOpML",
    "MLInput",
    "MLDecision",
    "TradeSide",
    "MarketContext",
    "TrendContext",
    "ChartContext",
    "IndicatorContext",
    "PriceContext",
    "create_ml_input_from_signal",
    "FeatureBuilder",
    "FeatureSchema",
    "Dataset",
    "DatasetSplit",
    "ModelBundle",
    "InferenceEngine",
    "InferenceOutput",
    "Policy",
    "PolicyConfig",
    "PolicyDecision",
    "train_main",
]

# Exported symbols: MLInterface, MLInput, MLDecision, MLPredictor, NoOpML, TradeSide,
# MarketContext, TrendContext, ChartContext, IndicatorContext, PriceContext,
# create_ml_input_from_signal, FeatureBuilder, FeatureSchema, Dataset, DatasetSplit,
# ModelBundle, InferenceEngine, InferenceOutput, Policy, PolicyConfig, PolicyDecision, train_main
