from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import torch

from eml_circuit import (
    RegressionTrainingConfig,
    infer_model_device,
    save_training_checkpoint,
    train_benchmark_regressor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EMLStack, MLP, and tree-search baselines across available devices."
    )
    parser.add_argument("--benchmark", choices=["shared", "deep", "circuit"], default="shared")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["emlstack", "mlp", "eml_tree"],
        default=["emlstack", "mlp", "eml_tree"],
    )
    parser.add_argument("--devices", nargs="*", default=None)
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-extrap", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--c", type=float, default=5.0)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--init-scale", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--selection-metric", choices=["train", "extrap"], default="extrap")
    parser.add_argument("--tree-max-depth", type=int, default=4)
    parser.add_argument("--tree-beam-width", type=int, default=32)
    parser.add_argument("--tree-max-basis-size", type=int, default=4)
    parser.add_argument("--tree-min-improvement", type=float, default=1e-6)
    parser.add_argument("--tree-selection-pool-size", type=int, default=128)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def detect_devices(requested_devices: list[str] | None) -> list[str]:
    if requested_devices:
        return requested_devices
    if torch.cuda.is_available():
        return [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    return ["cpu"]


def build_base_config(args: argparse.Namespace) -> RegressionTrainingConfig:
    return RegressionTrainingConfig(
        benchmark=args.benchmark,
        n_train=args.n_train,
        n_extrap=args.n_extrap,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        width=args.width,
        c=args.c,
        eps=args.eps,
        init_scale=args.init_scale,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        eval_every=args.eval_every,
        print_every=args.print_every,
        show_progress=not args.no_progress,
        selection_metric=args.selection_metric,
        tree_max_depth=args.tree_max_depth,
        tree_beam_width=args.tree_beam_width,
        tree_max_basis_size=args.tree_max_basis_size,
        tree_min_improvement=args.tree_min_improvement,
        tree_selection_pool_size=args.tree_selection_pool_size,
        seed=args.seed,
    )


def _run_single_job(
    config: RegressionTrainingConfig,
    save_path: str | None,
) -> dict[str, object]:
    run = train_benchmark_regressor(config)
    if save_path is not None:
        save_training_checkpoint(save_path, run)

    summary: dict[str, object] = {
        "benchmark": run.dataset.name,
        "model": config.model,
        "device": str(infer_model_device(run.model)),
        "train_mse": run.metrics.train_mse,
        "extrap_mse": run.metrics.extrap_mse,
        "best_epoch": run.metrics.best_epoch,
        "best_score": run.metrics.best_score,
        "checkpoint": save_path,
    }
    selected_expressions = getattr(run.model, "selected_expressions", None)
    if selected_expressions:
        summary["selected_expressions"] = list(selected_expressions)
    return summary


def main() -> None:
    args = parse_args()
    devices = detect_devices(args.devices)
    base_config = build_base_config(args)
    jobs: list[tuple[RegressionTrainingConfig, str | None]] = []
    save_dir = None if args.save_dir is None else Path(args.save_dir)

    for index, model_name in enumerate(args.models):
        assigned_device = devices[index % len(devices)]
        config = replace(
            base_config,
            model=model_name,
            device=assigned_device,
        )
        if len(devices) > 1 and len(args.models) > 1:
            config = replace(config, show_progress=False)

        save_path = None
        if save_dir is not None:
            save_path = str(save_dir / f"{args.benchmark}_{model_name}.pt")
        jobs.append((config, save_path))

    if len(devices) > 1 and len(jobs) > 1:
        max_workers = min(len(devices), len(jobs))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_run_single_job, config, save_path)
                for config, save_path in jobs
            ]
            for future in as_completed(futures):
                summary = future.result()
                print_summary(summary)
    else:
        for config, save_path in jobs:
            summary = _run_single_job(config, save_path)
            print_summary(summary)


def print_summary(summary: dict[str, object]) -> None:
    print(
        " | ".join(
            [
                f"benchmark={summary['benchmark']}",
                f"model={summary['model']}",
                f"device={summary['device']}",
                f"train_mse={summary['train_mse']:.6f}",
                f"extrap_mse={summary['extrap_mse']:.6f}",
            ]
        )
    )
    if summary["best_epoch"] is not None:
        print(f"best_epoch={summary['best_epoch']} best_score={summary['best_score']:.6f}")
    if summary.get("selected_expressions"):
        print(f"selected_expressions={summary['selected_expressions']}")
    if summary.get("checkpoint"):
        print(f"checkpoint={summary['checkpoint']}")


if __name__ == "__main__":
    main()
