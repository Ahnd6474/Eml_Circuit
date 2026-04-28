from __future__ import annotations

import torch
import torch.nn as nn

from .functional import stabilized_eml


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


class EMLResidualBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        width: int | None = None,
        *,
        c: float = 5.0,
        eps: float = 1e-4,
        init_scale: float = 1e-3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        width = width or dim
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")

        self.norm = RMSNorm(dim)
        self.pair_proj = nn.Linear(dim, 2 * width, bias=bias)
        self.out_proj = nn.Linear(width, dim, bias=bias)
        self.c = c
        self.eps = eps
        self.res_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.norm(h)
        u, v = self.pair_proj(z).chunk(2, dim=-1)
        e = stabilized_eml(u, v, c=self.c, eps=self.eps)
        return h + self.res_scale * self.out_proj(e)
