from __future__ import annotations

import argparse

import torch

from eml_circuit import EMLRegressor, MLPRegressor, fit_regression_model, make_benchmark_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EMLStack on README benchmarks.")
    parser.add_argument("--benchmark", choices=["shared", "deep", "circuit"], default="shared")
    parser.add_argument("--model", choices=["emlstack", "mlp"], default="emlstack")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dataset = make_benchmark_dataset(args.benchmark, seed=args.seed)
    if args.model == "emlstack":
        model = EMLRegressor(
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            width=args.width,
        )
    else:
        model = MLPRegressor(
            hidden_dim=args.hidden_dim,
            depth=args.depth,
        )

    metrics = fit_regression_model(
        model,
        dataset.train_inputs,
        dataset.train_targets,
        extrap_inputs=dataset.extrap_inputs,
        extrap_targets=dataset.extrap_targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    print(f"benchmark={dataset.name}")
    print(f"model={args.model}")
    print(f"train_mse={metrics.train_mse:.6f}")
    print(f"extrap_mse={metrics.extrap_mse:.6f}")


if __name__ == "__main__":
    main()
