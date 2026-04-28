from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class RegressionMetrics:
    train_mse: float
    extrap_mse: float
    history: list[float] = field(default_factory=list)


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
) -> RegressionMetrics:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    history: list[float] = []
    n_samples = train_inputs.shape[0]

    for _ in range(epochs):
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

        history.append(epoch_loss / n_samples)

    train_mse = evaluate_regression_mse(model, train_inputs, train_targets)
    if extrap_inputs is None or extrap_targets is None:
        extrap_mse = float("nan")
    else:
        extrap_mse = evaluate_regression_mse(model, extrap_inputs, extrap_targets)

    return RegressionMetrics(
        train_mse=train_mse,
        extrap_mse=extrap_mse,
        history=history,
    )
