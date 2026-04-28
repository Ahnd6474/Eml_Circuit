"""Residual EML circuit layers and benchmark utilities."""

from .benchmarks import BenchmarkDataset, make_benchmark_dataset
from .functional import eml, positive_log_branch, softsign_clip, stabilized_eml
from .layers import EMLResidualBlock, RMSNorm
from .models import EMLRegressor, EMLStack, MLPRegressor
from .training import (
    BenchmarkTrainingRun,
    RegressionMetrics,
    RegressionTrainingConfig,
    build_regression_model,
    evaluate_regression_mse,
    fit_regression_model,
    save_training_checkpoint,
    train_benchmark_regressor,
)

__all__ = [
    "BenchmarkDataset",
    "EMLResidualBlock",
    "EMLRegressor",
    "EMLStack",
    "MLPRegressor",
    "RMSNorm",
    "BenchmarkTrainingRun",
    "RegressionMetrics",
    "RegressionTrainingConfig",
    "build_regression_model",
    "eml",
    "evaluate_regression_mse",
    "fit_regression_model",
    "make_benchmark_dataset",
    "positive_log_branch",
    "save_training_checkpoint",
    "softsign_clip",
    "stabilized_eml",
    "train_benchmark_regressor",
]
