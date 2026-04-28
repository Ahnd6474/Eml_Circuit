from __future__ import annotations

import argparse

from eml_circuit import (
    RegressionTrainingConfig,
    infer_model_device,
    save_training_checkpoint,
    train_benchmark_regressor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EMLStack on README benchmarks.")
    parser.add_argument("--benchmark", choices=["shared", "deep", "circuit"], default="shared")
    parser.add_argument("--model", choices=["emlstack", "mlp", "eml_tree"], default="emlstack")
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
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RegressionTrainingConfig(
        benchmark=args.benchmark,
        model=args.model,
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
        device=args.device,
    )
    run = train_benchmark_regressor(config)

    if args.save_path is not None:
        save_training_checkpoint(args.save_path, run)

    print(f"benchmark={run.dataset.name}")
    print(f"model={config.model}")
    print(f"device={infer_model_device(run.model)}")
    print(f"train_mse={run.metrics.train_mse:.6f}")
    print(f"extrap_mse={run.metrics.extrap_mse:.6f}")
    if run.metrics.best_epoch is not None:
        print(f"best_epoch={run.metrics.best_epoch}")
        print(f"best_score={run.metrics.best_score:.6f}")
    if args.save_path is not None:
        print(f"checkpoint={args.save_path}")


if __name__ == "__main__":
    main()
