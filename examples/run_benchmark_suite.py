from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import torch

from eml_circuit import (
    RegressionTrainingConfig,
    infer_model_device,
    list_benchmark_groups,
    list_benchmark_names,
    save_training_checkpoint,
    train_benchmark_regressor,
)


def parse_args() -> argparse.Namespace:
    benchmark_groups = list_benchmark_groups()
    benchmark_names = sorted(set(list_benchmark_names() + ["shared", "deep", "circuit"]))
    parser = argparse.ArgumentParser(
        description="Run grouped benchmark suites across available devices."
    )
    parser.add_argument("--benchmark-group", choices=benchmark_groups, default=None)
    parser.add_argument("--benchmarks", nargs="*", choices=benchmark_names, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["emlstack", "mlp", "eml_tree"],
        default=None,
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


def select_benchmarks(args: argparse.Namespace) -> list[str]:
    if args.benchmarks:
        return args.benchmarks
    if args.benchmark_group:
        return list_benchmark_names(args.benchmark_group)
    return list_benchmark_names()


def select_models(args: argparse.Namespace) -> list[str]:
    if args.models:
        return args.models
    if args.benchmark_group == "tree":
        return ["emlstack", "eml_tree"]
    if args.benchmark_group == "mlp":
        return ["emlstack", "mlp"]
    return ["emlstack", "mlp", "eml_tree"]


def build_base_config(args: argparse.Namespace) -> RegressionTrainingConfig:
    return RegressionTrainingConfig(
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
    benchmarks = select_benchmarks(args)
    models = select_models(args)
    base_config = build_base_config(args)
    save_dir = None if args.save_dir is None else Path(args.save_dir)
    jobs: list[tuple[RegressionTrainingConfig, str | None]] = []

    for benchmark_index, benchmark_name in enumerate(benchmarks):
        for model_index, model_name in enumerate(models):
            job_index = benchmark_index * len(models) + model_index
            assigned_device = devices[job_index % len(devices)]
            config = replace(
                base_config,
                benchmark=benchmark_name,
                model=model_name,
                device=assigned_device,
            )
            if len(devices) > 1 and len(benchmarks) * len(models) > 1:
                config = replace(config, show_progress=False)

            save_path = None
            if save_dir is not None:
                save_path = str(save_dir / f"{benchmark_name}_{model_name}.pt")
            jobs.append((config, save_path))

    if len(devices) > 1 and len(jobs) > 1:
        max_workers = min(len(devices), len(jobs))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_run_single_job, config, save_path)
                for config, save_path in jobs
            ]
            for future in as_completed(futures):
                print_summary(future.result())
    else:
        for config, save_path in jobs:
            print_summary(_run_single_job(config, save_path))


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
