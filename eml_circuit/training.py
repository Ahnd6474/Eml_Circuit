from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .benchmarks import BenchmarkDataset, make_benchmark_dataset
from .models import EMLRegressor, MLPRegressor


@dataclass
class RegressionMetrics:
    train_mse: float
    extrap_mse: float
    history: list[float] = field(default_factory=list)
    extrap_history: list[float] = field(default_factory=list)
    best_epoch: int | None = None
    best_score: float | None = None


@dataclass
class RegressionTrainingConfig:
    benchmark: str = "shared"
    model: str = "emlstack"
    n_train: int = 2048
    n_extrap: int = 512
    hidden_dim: int = 128
    depth: int = 4
    width: int | None = 128
    output_dim: int = 1
    c: float = 5.0
    eps: float = 1e-4
    init_scale: float = 1e-3
    epochs: int = 500
    batch_size: int = 256
    lr: float = 3e-3
    weight_decay: float = 0.0
    grad_clip_norm: float | None = 1.0
    eval_every: int = 1
    print_every: int = 50
    restore_best: bool = True
    selection_metric: str = "extrap"
    seed: int = 0
    device: str | torch.device | None = None
    dtype: torch.dtype = torch.float32


@dataclass
class BenchmarkTrainingRun:
    config: RegressionTrainingConfig
    dataset: BenchmarkDataset
    model: torch.nn.Module
    metrics: RegressionMetrics


def evaluate_regression_mse(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        preds = model(inputs)
        mse = F.mse_loss(preds, targets).item()
    model.train(was_training)
    return mse


def fit_regression_model(
    model: torch.nn.Module,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    *,
    extrap_inputs: torch.Tensor | None = None,
    extrap_targets: torch.Tensor | None = None,
    epochs: int = 1000,
    batch_size: int = 256,
    lr: float = 3e-3,
    weight_decay: float = 0.0,
    grad_clip_norm: float | None = 1.0,
    eval_every: int = 1,
    print_every: int | None = None,
    restore_best: bool = True,
    selection_metric: str = "extrap",
) -> RegressionMetrics:
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if eval_every <= 0:
        raise ValueError(f"eval_every must be positive, got {eval_every}")
    if print_every is not None and print_every <= 0:
        raise ValueError(f"print_every must be positive, got {print_every}")
    if selection_metric not in {"train", "extrap"}:
        raise ValueError(
            f"selection_metric must be 'train' or 'extrap', got {selection_metric}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    history: list[float] = []
    extrap_history: list[float] = []
    n_samples = train_inputs.shape[0]
    has_extrap = extrap_inputs is not None and extrap_targets is not None
    effective_selection_metric = selection_metric if has_extrap else "train"
    best_epoch: int | None = None
    best_score: float | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n_samples, device=train_inputs.device)
        epoch_loss = 0.0

        for start in range(0, n_samples, batch_size):
            index = permutation[start : start + batch_size]
            batch_inputs = train_inputs[index]
            batch_targets = train_targets[index]

            optimizer.zero_grad(set_to_none=True)
            preds = model(batch_inputs)
            loss = F.mse_loss(preds, batch_targets)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            epoch_loss += loss.item() * batch_inputs.shape[0]

        mean_epoch_loss = epoch_loss / n_samples
        history.append(mean_epoch_loss)

        should_eval = (epoch + 1) % eval_every == 0 or epoch + 1 == epochs
        train_mse_epoch = float("nan")
        extrap_mse_epoch = float("nan")

        if should_eval:
            train_mse_epoch = evaluate_regression_mse(model, train_inputs, train_targets)
            if has_extrap:
                extrap_mse_epoch = evaluate_regression_mse(
                    model,
                    extrap_inputs,
                    extrap_targets,
                )
            score = (
                train_mse_epoch
                if effective_selection_metric == "train"
                else extrap_mse_epoch
            )
            if best_score is None or score < best_score:
                best_score = score
                best_epoch = epoch + 1
                if restore_best:
                    best_state_dict = copy.deepcopy(model.state_dict())

        extrap_history.append(extrap_mse_epoch)

        should_print = (
            print_every is not None
            and ((epoch + 1) == 1 or (epoch + 1) % print_every == 0 or (epoch + 1) == epochs)
        )
        if should_print:
            if train_mse_epoch != train_mse_epoch:
                train_mse_epoch = evaluate_regression_mse(
                    model,
                    train_inputs,
                    train_targets,
                )
            if has_extrap and extrap_mse_epoch != extrap_mse_epoch:
                extrap_mse_epoch = evaluate_regression_mse(
                    model,
                    extrap_inputs,
                    extrap_targets,
                )

            message = (
                f"[epoch {epoch + 1:04d}/{epochs:04d}] "
                f"loss={mean_epoch_loss:.6f} train_mse={train_mse_epoch:.6f}"
            )
            if has_extrap:
                message += f" extrap_mse={extrap_mse_epoch:.6f}"
            print(message)

    if restore_best and best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    train_mse = evaluate_regression_mse(model, train_inputs, train_targets)
    if not has_extrap:
        extrap_mse = float("nan")
    else:
        extrap_mse = evaluate_regression_mse(model, extrap_inputs, extrap_targets)

    return RegressionMetrics(
        train_mse=train_mse,
        extrap_mse=extrap_mse,
        history=history,
        extrap_history=extrap_history,
        best_epoch=best_epoch,
        best_score=best_score,
    )


def build_regression_model(config: RegressionTrainingConfig) -> torch.nn.Module:
    if config.model == "emlstack":
        return EMLRegressor(
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            width=config.width,
            output_dim=config.output_dim,
            c=config.c,
            eps=config.eps,
            init_scale=config.init_scale,
        )
    if config.model == "mlp":
        return MLPRegressor(
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            output_dim=config.output_dim,
        )
    raise ValueError(f"Unknown model type: {config.model}")


def train_benchmark_regressor(
    config: RegressionTrainingConfig,
) -> BenchmarkTrainingRun:
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)

    dataset = make_benchmark_dataset(
        config.benchmark,
        n_train=config.n_train,
        n_extrap=config.n_extrap,
        seed=config.seed,
        device=device,
        dtype=config.dtype,
    )
    model = build_regression_model(config).to(device=device, dtype=config.dtype)
    metrics = fit_regression_model(
        model,
        dataset.train_inputs,
        dataset.train_targets,
        extrap_inputs=dataset.extrap_inputs,
        extrap_targets=dataset.extrap_targets,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        weight_decay=config.weight_decay,
        grad_clip_norm=config.grad_clip_norm,
        eval_every=config.eval_every,
        print_every=config.print_every,
        restore_best=config.restore_best,
        selection_metric=config.selection_metric,
    )
    return BenchmarkTrainingRun(
        config=config,
        dataset=dataset,
        model=model,
        metrics=metrics,
    )


def save_training_checkpoint(
    path: str | Path,
    run: BenchmarkTrainingRun,
) -> None:
    checkpoint = {
        "model_state_dict": run.model.state_dict(),
        "config": {
            "benchmark": run.config.benchmark,
            "model": run.config.model,
            "n_train": run.config.n_train,
            "n_extrap": run.config.n_extrap,
            "hidden_dim": run.config.hidden_dim,
            "depth": run.config.depth,
            "width": run.config.width,
            "output_dim": run.config.output_dim,
            "c": run.config.c,
            "eps": run.config.eps,
            "init_scale": run.config.init_scale,
            "epochs": run.config.epochs,
            "batch_size": run.config.batch_size,
            "lr": run.config.lr,
            "weight_decay": run.config.weight_decay,
            "grad_clip_norm": run.config.grad_clip_norm,
            "eval_every": run.config.eval_every,
            "print_every": run.config.print_every,
            "restore_best": run.config.restore_best,
            "selection_metric": run.config.selection_metric,
            "seed": run.config.seed,
            "device": str(run.config.device),
            "dtype": str(run.config.dtype),
        },
        "metrics": {
            "train_mse": run.metrics.train_mse,
            "extrap_mse": run.metrics.extrap_mse,
            "history": run.metrics.history,
            "extrap_history": run.metrics.extrap_history,
            "best_epoch": run.metrics.best_epoch,
            "best_score": run.metrics.best_score,
        },
        "dataset_name": run.dataset.name,
    }
    torch.save(checkpoint, Path(path))
