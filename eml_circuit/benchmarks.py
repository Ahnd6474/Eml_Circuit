from __future__ import annotations

from dataclasses import dataclass

import torch

from .functional import eml, softsign_clip


@dataclass
class BenchmarkDataset:
    name: str
    train_inputs: torch.Tensor
    train_targets: torch.Tensor
    extrap_inputs: torch.Tensor
    extrap_targets: torch.Tensor


def shared_subexpression_family(
    inputs: torch.Tensor,
    *,
    a: tuple[float, ...] = (0.9, -0.4, 0.6),
    alpha: tuple[float, ...] = (0.6, -0.35, 0.2),
    beta: tuple[float, ...] = (1.2, 1.5, 0.8),
    gamma: tuple[float, ...] = (0.7, 0.4, 0.9),
) -> torch.Tensor:
    x = inputs[..., 0]
    y = inputs[..., 1]
    g = eml(x, y)

    total = torch.zeros_like(g)
    for coeff, alpha_i, beta_i, gamma_i in zip(a, alpha, beta, gamma, strict=True):
        total = total + coeff * (
            torch.exp(alpha_i * g) - torch.log(beta_i + gamma_i * g.square())
        )
    return total.unsqueeze(-1)


def deep_eml_chain(
    inputs: torch.Tensor,
    *,
    depth: int = 5,
    c: float = 5.0,
) -> torch.Tensor:
    x = inputs[..., 0]
    y = inputs[..., 1]
    g = x
    for _ in range(depth):
        g = eml(softsign_clip(g, c), y)
    return g.unsqueeze(-1)


def low_depth_circuit_high_size_tree(
    inputs: torch.Tensor,
    *,
    weights: tuple[float, float, float, float] = (0.6, -0.15, 0.2, 0.08),
) -> torch.Tensor:
    x = inputs[..., 0]
    y = inputs[..., 1]

    z1 = eml(x, y)
    z2 = eml(z1, torch.ones_like(z1))
    z3 = eml(torch.ones_like(z1), z1)
    z4 = eml(z2, z3)

    w1, w2, w3, w4 = weights
    return (w1 * z1 + w2 * z2 + w3 * z3 + w4 * z4).unsqueeze(-1)


def make_benchmark_dataset(
    name: str,
    *,
    n_train: int = 2048,
    n_extrap: int = 512,
    seed: int = 0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> BenchmarkDataset:
    name = name.lower()
    device = device or "cpu"

    if name == "shared":
        fn = shared_subexpression_family
        train_domain = ((-1.25, 1.25), (0.75, 2.25))
        extrap_domain = ((1.25, 2.5), (2.25, 4.0))
    elif name == "deep":
        fn = deep_eml_chain
        train_domain = ((-0.6, 0.6), (0.9, 1.5))
        extrap_domain = ((0.6, 1.2), (1.5, 2.2))
    elif name == "circuit":
        fn = low_depth_circuit_high_size_tree
        train_domain = ((0.2, 0.9), (1.0, 1.8))
        extrap_domain = ((0.9, 1.4), (1.8, 2.4))
    else:
        raise ValueError(f"Unknown benchmark name: {name}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    train_inputs, train_targets = _sample_valid_regression_points(
        fn,
        n_samples=n_train,
        domain=train_domain,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    extrap_inputs, extrap_targets = _sample_valid_regression_points(
        fn,
        n_samples=n_extrap,
        domain=extrap_domain,
        generator=generator,
        device=device,
        dtype=dtype,
    )

    return BenchmarkDataset(
        name=name,
        train_inputs=train_inputs,
        train_targets=train_targets,
        extrap_inputs=extrap_inputs,
        extrap_targets=extrap_targets,
    )


def _sample_valid_regression_points(
    fn,
    *,
    n_samples: int,
    domain: tuple[tuple[float, float], tuple[float, float]],
    generator: torch.Generator,
    device: torch.device | str,
    dtype: torch.dtype,
    max_rounds: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    accepted_inputs: list[torch.Tensor] = []
    accepted_targets: list[torch.Tensor] = []
    total = 0

    for _ in range(max_rounds):
        if total >= n_samples:
            break
        needed = n_samples - total
        candidate_inputs = _sample_box(
            needed * 4,
            domain=domain,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        candidate_targets = fn(candidate_inputs)
        valid_mask = torch.isfinite(candidate_targets).all(dim=-1)
        if valid_mask.any():
            accepted_inputs.append(candidate_inputs[valid_mask])
            accepted_targets.append(candidate_targets[valid_mask])
            total += int(valid_mask.sum().item())

    if total < n_samples:
        raise RuntimeError(
            f"Could not sample {n_samples} valid points for domain {domain}; "
            f"collected only {total}"
        )

    inputs = torch.cat(accepted_inputs, dim=0)[:n_samples]
    targets = torch.cat(accepted_targets, dim=0)[:n_samples]
    return inputs, targets


def _sample_box(
    n_samples: int,
    *,
    domain: tuple[tuple[float, float], tuple[float, float]],
    generator: torch.Generator,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    (x_low, x_high), (y_low, y_high) = domain
    x = torch.rand(n_samples, 1, generator=generator, dtype=dtype)
    y = torch.rand(n_samples, 1, generator=generator, dtype=dtype)
    x = x * (x_high - x_low) + x_low
    y = y * (y_high - y_low) + y_low
    return torch.cat([x, y], dim=-1).to(device=device)
