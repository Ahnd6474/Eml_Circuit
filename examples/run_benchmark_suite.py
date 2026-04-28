from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import torch

from eml_circuit import (
    count_trainable_parameters,
    RegressionTrainingConfig,
    infer_eml_width,
    infer_model_device,
    list_benchmark_groups,
    list_benchmark_names,
    save_training_checkpoint,
    should_normalize_targets,
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
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--hidden-dims", nargs="*", type=int, default=None)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--widths", nargs="*", type=int, default=None)
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
    parser.add_argument("--normalize-targets", choices=["auto", "never", "always"], default="auto")
    parser.add_argument("--tree-max-depth", type=int, default=4)
    parser.add_argument("--tree-beam-width", type=int, default=32)
    parser.add_argument("--tree-max-basis-size", type=int, default=4)
    parser.add_argument("--tree-min-improvement", type=float, default=1e-6)
    parser.add_argument("--tree-selection-pool-size", type=int, default=128)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--log-dir", default=None)
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
        normalize_targets=args.normalize_targets,
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
    log_path: str | None,
) -> dict[str, object]:
    if log_path is None:
        return _execute_job(config, save_path, None)

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as handle:
        with redirect_stdout(handle), redirect_stderr(handle):
            return _execute_job(config, save_path, str(log_file))


def _execute_job(
    config: RegressionTrainingConfig,
    save_path: str | None,
    log_path: str | None,
) -> dict[str, object]:
    run = train_benchmark_regressor(config)
    if save_path is not None:
        save_training_checkpoint(save_path, run)

    model_metadata_fn = getattr(run.model, "export_metadata", None)
    model_metadata = model_metadata_fn() if model_metadata_fn is not None else None
    summary: dict[str, object] = {
        "benchmark": run.dataset.name,
        "model": config.model,
        "device": str(infer_model_device(run.model)),
        "normalized_targets": should_normalize_targets(
            config.benchmark,
            config.normalize_targets,
        ),
        "hidden_dim": config.hidden_dim,
        "width": (
            infer_eml_width(config.hidden_dim, config.width)
            if config.model == "emlstack"
            else None
        ),
        "trainable_parameters": count_trainable_parameters(run.model),
        "train_mse": run.metrics.train_mse,
        "extrap_mse": run.metrics.extrap_mse,
        "best_epoch": run.metrics.best_epoch,
        "best_score": run.metrics.best_score,
        "checkpoint": save_path,
        "log_path": log_path,
    }
    selected_expressions = getattr(run.model, "selected_expressions", None)
    if selected_expressions:
        summary["selected_expressions"] = list(selected_expressions)
    if model_metadata is not None:
        summary["selected_expression_count"] = model_metadata.get("selected_expression_count")
        summary["selected_total_nodes"] = model_metadata.get("selected_total_nodes")
        summary["selected_max_depth"] = model_metadata.get("selected_max_depth")
    return summary


def select_hidden_dims(args: argparse.Namespace) -> list[int]:
    if args.hidden_dims:
        return args.hidden_dims
    return [args.hidden_dim]


def select_widths(args: argparse.Namespace) -> list[int | None]:
    if args.widths:
        return list(args.widths)
    return [args.width]


def main() -> None:
    args = parse_args()
    devices = detect_devices(args.devices)
    benchmarks = select_benchmarks(args)
    models = select_models(args)
    hidden_dims = select_hidden_dims(args)
    widths = select_widths(args)
    base_config = build_base_config(args)
    save_dir = None if args.save_dir is None else Path(args.save_dir)
    total_jobs = len(benchmarks) * sum(
        len(widths) * len(hidden_dims) if model_name == "emlstack" else len(hidden_dims)
        for model_name in models
    )
    default_log_dir = None
    if args.log_dir is not None:
        default_log_dir = Path(args.log_dir)
    elif total_jobs > 1:
        default_log_dir = (save_dir / "logs") if save_dir is not None else Path("suite_logs")
    jobs: list[tuple[RegressionTrainingConfig, str | None, str | None]] = []

    for benchmark_index, benchmark_name in enumerate(benchmarks):
        for model_name in models:
            model_widths = widths if model_name == "emlstack" else [None]
            for hidden_dim in hidden_dims:
                for width in model_widths:
                    job_index = len(jobs)
                    assigned_device = devices[job_index % len(devices)]
                    config = replace(
                        base_config,
                        benchmark=benchmark_name,
                        model=model_name,
                        hidden_dim=hidden_dim,
                        width=width,
                        device=assigned_device,
                    )
                    if len(devices) > 1 and total_jobs > 1:
                        config = replace(config, show_progress=False)

                    capacity_tag = f"h{hidden_dim}"
                    if model_name == "emlstack":
                        resolved_width = infer_eml_width(hidden_dim, width)
                        capacity_tag += f"_w{resolved_width}"

                    save_path = None
                    if save_dir is not None:
                        save_path = str(save_dir / f"{benchmark_name}_{model_name}_{capacity_tag}.pt")
                    log_path = None
                    if default_log_dir is not None:
                        log_path = str(default_log_dir / f"{benchmark_name}_{model_name}_{capacity_tag}.log")
                    jobs.append((config, save_path, log_path))

    if len(devices) > 1 and len(jobs) > 1:
        max_workers = min(len(devices), len(jobs))
        spawn_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=spawn_context,
        ) as executor:
            futures = [
                executor.submit(_run_single_job, config, save_path, log_path)
                for config, save_path, log_path in jobs
            ]
            for future in as_completed(futures):
                print_summary(future.result())
    else:
        for config, save_path, log_path in jobs:
            print_summary(_run_single_job(config, save_path, log_path))


def print_summary(summary: dict[str, object]) -> None:
    print(
        " | ".join(
            [
                f"benchmark={summary['benchmark']}",
                f"model={summary['model']}",
                f"device={summary['device']}",
                f"normalized_targets={summary['normalized_targets']}",
                f"hidden_dim={summary['hidden_dim']}",
                f"width={summary['width'] if summary['width'] is not None else 'n/a'}",
                f"params={summary['trainable_parameters']}",
                f"train_mse={summary['train_mse']:.6f}",
                f"extrap_mse={summary['extrap_mse']:.6f}",
            ]
        )
    )
    if summary["best_epoch"] is not None:
        print(f"best_epoch={summary['best_epoch']} best_score={summary['best_score']:.6f}")
    if summary.get("selected_expressions"):
        print(f"selected_expressions={summary['selected_expressions']}")
    if summary.get("selected_expression_count") is not None:
        print(
            "complexity="
            f"exprs:{summary['selected_expression_count']} "
            f"nodes:{summary['selected_total_nodes']} "
            f"depth:{summary['selected_max_depth']}"
        )
    if summary.get("checkpoint"):
        print(f"checkpoint={summary['checkpoint']}")
    if summary.get("log_path"):
        print(f"log={summary['log_path']}")


if __name__ == "__main__":
    main()
