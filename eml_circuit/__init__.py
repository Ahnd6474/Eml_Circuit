"""Residual EML circuit layers and benchmark utilities."""

from .benchmarks import BenchmarkDataset, make_benchmark_dataset
from .functional import eml, positive_log_branch, softsign_clip, stabilized_eml
from .layers import EMLResidualBlock, RMSNorm
from .models import EMLRegressor, EMLStack, MLPRegressor
from .training import RegressionMetrics, evaluate_regression_mse, fit_regression_model

__all__ = [
    "BenchmarkDataset",
    "EMLResidualBlock",
    "EMLRegressor",
    "EMLStack",
    "MLPRegressor",
    "RMSNorm",
    "RegressionMetrics",
    "eml",
    "evaluate_regression_mse",
    "fit_regression_model",
    "make_benchmark_dataset",
    "positive_log_branch",
    "softsign_clip",
    "stabilized_eml",
]
