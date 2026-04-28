from __future__ import annotations

import torch
import torch.nn.functional as F


def softsign_clip(x: torch.Tensor, c: float) -> torch.Tensor:
    """Clamp to (-c, c) with slower saturation than tanh."""
    if c <= 0:
        raise ValueError(f"c must be positive, got {c}")
    return x / (1.0 + x.abs() / c)


def positive_log_branch(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Project logits into the positive reals for a stable log branch."""
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    return F.softplus(x) + eps


def eml(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Raw EML operator from the paper."""
    return torch.exp(u) - torch.log(v)


def stabilized_eml(
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    c: float = 5.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Numerically safer EML variant used inside EMLStack."""
    u_safe = softsign_clip(u, c)
    v_pos = positive_log_branch(v, eps)
    return eml(u_safe, v_pos)
