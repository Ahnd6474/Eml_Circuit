from __future__ import annotations

import importlib.util
import unittest


if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

import torch

from eml_circuit import (
    EMLRegressor,
    EMLResidualBlock,
    EMLStack,
    make_benchmark_dataset,
    softsign_clip,
)


class FunctionalTests(unittest.TestCase):
    def test_softsign_clip_is_bounded(self) -> None:
        x = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
        clipped = softsign_clip(x, c=5.0)
        self.assertTrue(torch.all(clipped.abs() < 5.0))


class ModelTests(unittest.TestCase):
    def test_block_preserves_shape(self) -> None:
        block = EMLResidualBlock(dim=16, width=32)
        x = torch.randn(8, 16)
        y = block(x)
        self.assertEqual(y.shape, x.shape)

    def test_stack_preserves_shape(self) -> None:
        stack = EMLStack(dim=16, depth=3, width=32)
        x = torch.randn(8, 16)
        y = stack(x)
        self.assertEqual(y.shape, x.shape)

    def test_regressor_outputs_scalar(self) -> None:
        model = EMLRegressor(hidden_dim=32, depth=2, width=32)
        x = torch.randn(8, 2)
        y = model(x)
        self.assertEqual(y.shape, (8, 1))


class BenchmarkTests(unittest.TestCase):
    def test_shared_dataset_shapes(self) -> None:
        dataset = make_benchmark_dataset("shared", n_train=64, n_extrap=16, seed=1)
        self.assertEqual(dataset.train_inputs.shape, (64, 2))
        self.assertEqual(dataset.train_targets.shape, (64, 1))
        self.assertEqual(dataset.extrap_inputs.shape, (16, 2))
        self.assertEqual(dataset.extrap_targets.shape, (16, 1))
        self.assertTrue(torch.isfinite(dataset.train_targets).all())


if __name__ == "__main__":
    unittest.main()
