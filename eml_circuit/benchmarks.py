from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .functional import eml, softsign_clip


Domain2D = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    train_inputs: torch.Tensor
    train_targets: torch.Tensor
    extrap_inputs: torch.Tensor
    extrap_targets: torch.Tensor
    target_mean: float = 0.0
    target_std: float = 1.0
    normalized_targets: bool = False


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    group: str
    fn: Callable[[torch.Tensor], torch.Tensor]
    train_domain: Domain2D
    extrap_domain: Domain2D
    description: str


def tree_shared_subexpression_a(inputs: torch.Tensor) -> torch.Tensor:
    x = inputs[..., 0]
    y = inputs[..., 1]
    g = eml(x, y)
    total = (
        0.9 * (torch.exp(0.6 * g) - torch.log(1.2 + 0.7 * g.square()))
        - 0.4 * (torch.exp(-0.35 * g) - torch.log(1.5 + 0.4 * g.square()))
        + 0.6 * (torch.exp(0.2 * g) - torch.log(0.8 + 0.9 * g.square()))
    )
    return total.unsqueeze(-1)


def tree_deep_chain_b(inputs: torch.Tensor, *, depth: int = 5, c: float = 5.0) -> torch.Tensor:
    x = inputs[..., 0]
    y = inputs[..., 1]
    g = x
    for _ in range(depth):
        g = eml(softsign_clip(g, c), y)
    return g.unsqueeze(-1)


def tree_circuit_reuse_c(inputs: torch.Tensor) -> torch.Tensor:
    x = inputs[..., 0]
    y = inputs[..., 1]
    z1 = eml(x, y)
    z2 = eml(z1, torch.ones_like(z1))
    z3 = eml(torch.ones_like(z1), z1)
    z4 = eml(z2, z3)
    return (0.6 * z1 - 0.15 * z2 + 0.2 * z3 + 0.08 * z4).unsqueeze(-1)


def mlp_gelu_mix_a(inputs: torch.Tensor) -> torch.Tensor:
    x = inputs[..., 0]
    y = inputs[..., 1]
    u = 1.15 * x - 0.45 * y
    v = 0.55 * x + 1.10 * y
    total = F.gelu(u) + 0.7 * F.gelu(v) - 0.25 * u * v
    return total.unsqueeze(-1)


def mlp_relu_piecewise_b(inputs: torch.Tensor) -> torch.Tensor:
    x = inputs[..., 0]
    y = inputs[..., 1]
    ridge = 1.3 * x - 0.8 * y - 0.15
    corner = -0.9 * x + 1.1 * y + 0.2
    total = F.relu(ridge) - 0.6 * F.relu(corner) + 0.12 * (x + y).square()
    return total.unsqueeze(-1)


def mlp_silu_gate_c(inputs: torch.Tensor) -> torch.Tensor:
    x = inputs[..., 0]
    y = inputs[..., 1]
    u = 0.9 * x + 0.6 * y
    v = -0.7 * x + 1.2 * y
    total = F.silu(u) * (1.0 + 0.15 * x) + 0.8 * F.silu(v) - 0.1 * x * y
    return total.unsqueeze(-1)


BENCHMARK_SPECS: dict[str, BenchmarkSpec] = {
    "tree_shared_subexpr_a": BenchmarkSpec(
        name="tree_shared_subexpr_a",
        group="tree",
        fn=tree_shared_subexpression_a,
        train_domain=((-1.25, 1.25), (0.75, 2.25)),
        extrap_domain=((1.25, 2.5), (2.25, 4.0)),
        description="Shared EML subexpressions reused across several branches.",
    ),
    "tree_deep_chain_b": BenchmarkSpec(
        name="tree_deep_chain_b",
        group="tree",
        fn=tree_deep_chain_b,
        train_domain=((-0.6, 0.6), (0.9, 1.5)),
        extrap_domain=((0.6, 1.2), (1.5, 2.2)),
        description="Deep EML chain that stresses tree depth search.",
    ),
    "tree_circuit_reuse_c": BenchmarkSpec(
        name="tree_circuit_reuse_c",
        group="tree",
        fn=tree_circuit_reuse_c,
        train_domain=((0.2, 0.9), (1.0, 1.8)),
        extrap_domain=((0.9, 1.4), (1.8, 2.4)),
        description="Low-depth circuit with repeated intermediate EML reuse.",
    ),
    "mlp_gelu_mix_a": BenchmarkSpec(
        name="mlp_gelu_mix_a",
        group="mlp",
        fn=mlp_gelu_mix_a,
        train_domain=((-2.0, 2.0), (-2.0, 2.0)),
        extrap_domain=((2.0, 3.25), (2.0, 3.25)),
        description="GELU-composed target for MLP-family comparison.",
    ),
    "mlp_relu_piecewise_b": BenchmarkSpec(
        name="mlp_relu_piecewise_b",
        group="mlp",
        fn=mlp_relu_piecewise_b,
        train_domain=((-2.5, 2.5), (-2.5, 2.5)),
        extrap_domain=((2.5, 4.0), (2.5, 4.0)),
        description="ReLU piecewise target with mild quadratic interaction.",
    ),
    "mlp_silu_gate_c": BenchmarkSpec(
        name="mlp_silu_gate_c",
        group="mlp",
        fn=mlp_silu_gate_c,
        train_domain=((-2.25, 2.25), (-2.25, 2.25)),
        extrap_domain=((2.25, 3.75), (2.25, 3.75)),
        description="SiLU-gated smooth target for neural baseline comparison.",
    ),
}


BENCHMARK_ALIASES = {
    "shared": "tree_shared_subexpr_a",
    "deep": "tree_deep_chain_b",
    "circuit": "tree_circuit_reuse_c",
}


def canonicalize_benchmark_name(name: str) -> str:
    lowered = name.lower()
    return BENCHMARK_ALIASES.get(lowered, lowered)


def get_benchmark_spec(name: str) -> BenchmarkSpec:
    canonical_name = canonicalize_benchmark_name(name)
    try:
        return BENCHMARK_SPECS[canonical_name]
    except KeyError as exc:
        available = ", ".join(sorted(BENCHMARK_SPECS))
        raise ValueError(f"Unknown benchmark name: {name}. Available: {available}") from exc


def list_benchmark_names(group: str | None = None) -> list[str]:
    specs = BENCHMARK_SPECS.values()
    if group is not None:
        specs = [spec for spec in specs if spec.group == group]
    return sorted(spec.name for spec in specs)


def list_benchmark_groups() -> list[str]:
    return sorted({spec.group for spec in BENCHMARK_SPECS.values()})


def list_benchmark_specs(group: str | None = None) -> list[BenchmarkSpec]:
    specs = list(BENCHMARK_SPECS.values())
    if group is not None:
        specs = [spec for spec in specs if spec.group == group]
    return sorted(specs, key=lambda spec: spec.name)


def make_benchmark_dataset(
    name: str,
    *,
    n_train: int = 2048,
    n_extrap: int = 512,
    seed: int = 0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    normalize_targets: bool = False,
) -> BenchmarkDataset:
    spec = get_benchmark_spec(name)
    device = device or "cpu"

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    train_inputs, train_targets = _sample_valid_regression_points(
        spec.fn,
        n_samples=n_train,
        domain=spec.train_domain,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    extrap_inputs, extrap_targets = _sample_valid_regression_points(
        spec.fn,
        n_samples=n_extrap,
        domain=spec.extrap_domain,
        generator=generator,
        device=device,
        dtype=dtype,
    )

    target_mean = 0.0
    target_std = 1.0
    if normalize_targets:
        target_mean = float(train_targets.mean().item())
        target_std = float(train_targets.std(unbiased=False).item())
        target_std = max(target_std, 1e-6)
        train_targets = (train_targets - target_mean) / target_std
        extrap_targets = (extrap_targets - target_mean) / target_std

    return BenchmarkDataset(
        name=spec.name,
        train_inputs=train_inputs,
        train_targets=train_targets,
        extrap_inputs=extrap_inputs,
        extrap_targets=extrap_targets,
        target_mean=target_mean,
        target_std=target_std,
        normalized_targets=normalize_targets,
    )


def _sample_valid_regression_points(
    fn,
    *,
    n_samples: int,
    domain: Domain2D,
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
    domain: Domain2D,
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
