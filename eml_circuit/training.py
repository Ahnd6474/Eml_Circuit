from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .benchmarks import BenchmarkDataset, make_benchmark_dataset
from .models import EMLRegressor, MLPRegressor
from .progress import maybe_tqdm
from .symbolic import EMLTreeSearchRegressor


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
    benchmark: str = "tree_shared_subexpr_a"
    model: str = "emlstack"
    n_train: int = 2048
    n_extrap: int = 512
    hidden_dim: int = 16
    depth: int = 2
    width: int | None = None
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
    show_progress: bool = True
    restore_best: bool = True
    selection_metric: str = "extrap"
    tree_max_depth: int = 4
    tree_beam_width: int = 32
    tree_max_basis_size: int = 4
    tree_min_improvement: float = 1e-6
    tree_selection_pool_size: int = 128
    seed: int = 0
    device: str | torch.device | None = None
    dtype: torch.dtype = torch.float32


@dataclass
class BenchmarkTrainingRun:
    config: RegressionTrainingConfig
    dataset: BenchmarkDataset
    model: torch.nn.Module
    metrics: RegressionMetrics


def infer_eml_width(hidden_dim: int, width: int | None) -> int:
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
    if width is not None:
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        return width
    return max(4, hidden_dim // 2)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


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
    show_progress: bool = True,
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
    progress = maybe_tqdm(
        range(epochs),
        disable=not show_progress,
        desc=f"train:{model.__class__.__name__}",
        leave=False,
    )

    for epoch in progress:
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
            if hasattr(progress, "write"):
                progress.write(message)
            else:
                print(message)

        if hasattr(progress, "set_postfix"):
            postfix = {"loss": f"{mean_epoch_loss:.6f}"}
            if train_mse_epoch == train_mse_epoch:
                postfix["train_mse"] = f"{train_mse_epoch:.6f}"
            if has_extrap and extrap_mse_epoch == extrap_mse_epoch:
                postfix["extrap_mse"] = f"{extrap_mse_epoch:.6f}"
            progress.set_postfix(postfix)

    if restore_best and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    progress.close()

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
        width = infer_eml_width(config.hidden_dim, config.width)
        return EMLRegressor(
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            width=width,
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
    if config.model == "eml_tree":
        return EMLTreeSearchRegressor(
            max_depth=config.tree_max_depth,
            beam_width=config.tree_beam_width,
            max_basis_size=config.tree_max_basis_size,
            min_improvement=config.tree_min_improvement,
            selection_pool_size=config.tree_selection_pool_size,
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
    if isinstance(model, EMLTreeSearchRegressor):
        symbolic_result = model.fit(
            dataset.train_inputs,
            dataset.train_targets,
            extrap_inputs=dataset.extrap_inputs,
            extrap_targets=dataset.extrap_targets,
            show_progress=config.show_progress,
            progress_desc=f"tree:{config.benchmark}",
        )
        metrics = RegressionMetrics(
            train_mse=symbolic_result.train_mse,
            extrap_mse=symbolic_result.extrap_mse,
            history=symbolic_result.history,
            extrap_history=symbolic_result.extrap_history,
            best_epoch=symbolic_result.best_depth,
            best_score=min(symbolic_result.history),
        )
    else:
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
            show_progress=config.show_progress,
            restore_best=config.restore_best,
            selection_metric=config.selection_metric,
        )
    return BenchmarkTrainingRun(
        config=config,
        dataset=dataset,
        model=model,
        metrics=metrics,
    )


def infer_model_device(model: torch.nn.Module) -> torch.device:
    for parameter in model.parameters():
        return parameter.device
    for buffer in model.buffers():
        return buffer.device
    return torch.device("cpu")


def save_training_checkpoint(
    path: str | Path,
    run: BenchmarkTrainingRun,
) -> None:
    path = Path(path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)

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
            "resolved_width": (
                infer_eml_width(run.config.hidden_dim, run.config.width)
                if run.config.model == "emlstack"
                else run.config.width
            ),
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
            "show_progress": run.config.show_progress,
            "restore_best": run.config.restore_best,
            "selection_metric": run.config.selection_metric,
            "tree_max_depth": run.config.tree_max_depth,
            "tree_beam_width": run.config.tree_beam_width,
            "tree_max_basis_size": run.config.tree_max_basis_size,
            "tree_min_improvement": run.config.tree_min_improvement,
            "tree_selection_pool_size": run.config.tree_selection_pool_size,
            "seed": run.config.seed,
            "device": str(infer_model_device(run.model)),
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
        "trainable_parameters": count_trainable_parameters(run.model),
    }
    if hasattr(run.model, "export_metadata"):
        checkpoint["model_metadata"] = run.model.export_metadata()
    torch.save(checkpoint, path)
