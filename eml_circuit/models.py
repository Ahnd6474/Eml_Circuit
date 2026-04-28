from __future__ import annotations

import torch
import torch.nn as nn

from .layers import EMLResidualBlock


class EMLStack(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        *,
        width: int | None = None,
        c: float = 5.0,
        eps: float = 1e-4,
        init_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        self.blocks = nn.ModuleList(
            [
                EMLResidualBlock(
                    dim=dim,
                    width=width,
                    c=c,
                    eps=eps,
                    init_scale=init_scale,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            h = block(h)
        return h


class EMLRegressor(nn.Module):
    """Regression wrapper for the 2D benchmark families in the README."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 16,
        depth: int = 2,
        *,
        width: int | None = None,
        output_dim: int = 1,
        c: float = 5.0,
        eps: float = 1e-4,
        init_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.stack = EMLStack(
            dim=hidden_dim,
            depth=depth,
            width=width,
            c=c,
            eps=eps,
            init_scale=init_scale,
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.stack(h)
        return self.output_proj(h)


class MLPRegressor(nn.Module):
    """Simple MLP + GELU baseline from the README baseline table."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 16,
        depth: int = 2,
        *,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
