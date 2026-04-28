from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional import eml
from .progress import maybe_tqdm


@dataclass(frozen=True)
class TreeSearchNode:
    kind: str
    left: TreeSearchNode | None = None
    right: TreeSearchNode | None = None

    def to_string(self) -> str:
        if self.kind in {"x0", "x1", "one"}:
            return self.kind
        if self.left is None or self.right is None:
            raise ValueError("eml node requires left and right children")
        return f"eml({self.left.to_string()}, {self.right.to_string()})"

    def to_dict(self) -> dict[str, object]:
        if self.kind in {"x0", "x1", "one"}:
            return {"kind": self.kind}
        if self.left is None or self.right is None:
            raise ValueError("eml node requires left and right children")
        return {
            "kind": self.kind,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    def node_count(self) -> int:
        if self.kind in {"x0", "x1", "one"}:
            return 1
        if self.left is None or self.right is None:
            raise ValueError("eml node requires left and right children")
        return 1 + self.left.node_count() + self.right.node_count()

    def tree_depth(self) -> int:
        if self.kind in {"x0", "x1", "one"}:
            return 0
        if self.left is None or self.right is None:
            raise ValueError("eml node requires left and right children")
        return 1 + max(self.left.tree_depth(), self.right.tree_depth())


@dataclass
class TreeSearchFitResult:
    train_mse: float
    extrap_mse: float
    history: list[float]
    extrap_history: list[float]
    best_depth: int | None
    selected_expressions: list[str]


@dataclass
class _Candidate:
    node: TreeSearchNode
    depth: int
    train_values: torch.Tensor
    extrap_values: torch.Tensor | None
    score: float

    @property
    def expression(self) -> str:
        return self.node.to_string()


class EMLTreeSearchRegressor(nn.Module):
    """Beam-search baseline over raw EML trees plus a linear readout."""

    def __init__(
        self,
        *,
        max_depth: int = 4,
        beam_width: int = 32,
        max_basis_size: int = 4,
        min_improvement: float = 1e-6,
        selection_pool_size: int = 128,
    ) -> None:
        super().__init__()
        if max_depth < 0:
            raise ValueError(f"max_depth must be non-negative, got {max_depth}")
        if beam_width <= 0:
            raise ValueError(f"beam_width must be positive, got {beam_width}")
        if max_basis_size <= 0:
            raise ValueError(f"max_basis_size must be positive, got {max_basis_size}")
        if selection_pool_size <= 0:
            raise ValueError(
                f"selection_pool_size must be positive, got {selection_pool_size}"
            )

        self.max_depth = max_depth
        self.beam_width = beam_width
        self.max_basis_size = max_basis_size
        self.min_improvement = min_improvement
        self.selection_pool_size = selection_pool_size

        self.selected_nodes: list[TreeSearchNode] = []
        self.selected_expressions: list[str] = []
        self.register_buffer("coefficients", torch.zeros(0))
        self.register_buffer("bias", torch.tensor(0.0))

    def fit(
        self,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        *,
        extrap_inputs: torch.Tensor | None = None,
        extrap_targets: torch.Tensor | None = None,
        show_progress: bool = True,
        progress_desc: str = "tree-search",
    ) -> TreeSearchFitResult:
        y_train = train_targets.squeeze(-1)
        y_extrap = None if extrap_targets is None else extrap_targets.squeeze(-1)
        has_extrap = extrap_inputs is not None and extrap_targets is not None

        library = self._make_terminal_candidates(train_inputs, extrap_inputs, y_train)
        seen = {candidate.expression for candidate in library}
        depth_history: list[float] = []
        extrap_history: list[float] = []
        best_depth: int | None = None
        best_train = float("inf")

        depth_progress = maybe_tqdm(
            range(self.max_depth + 1),
            disable=not show_progress,
            desc=progress_desc,
            leave=False,
        )

        for depth in depth_progress:
            if depth > 0:
                pool = sorted(library, key=lambda candidate: candidate.score)[
                    : self.beam_width
                ]
                new_candidates = self._expand_depth(
                    depth,
                    pool,
                    y_train,
                    seen,
                )
                library.extend(new_candidates)

            current_pool = sorted(library, key=lambda candidate: candidate.score)[
                : self.selection_pool_size
            ]
            selection = self._greedy_select(current_pool, y_train, y_extrap)
            depth_history.append(selection["train_mse"])
            extrap_history.append(selection["extrap_mse"])

            if selection["train_mse"] < best_train:
                best_train = selection["train_mse"]
                best_depth = depth
                self._commit_selection(selection)

            if hasattr(depth_progress, "set_postfix"):
                postfix = {"train_mse": f"{selection['train_mse']:.6f}"}
                if has_extrap and selection["extrap_mse"] == selection["extrap_mse"]:
                    postfix["extrap_mse"] = f"{selection['extrap_mse']:.6f}"
                depth_progress.set_postfix(postfix)

        depth_progress.close()

        train_mse = F.mse_loss(self(train_inputs), train_targets).item()
        extrap_mse = (
            F.mse_loss(self(extrap_inputs), extrap_targets).item()
            if has_extrap
            else float("nan")
        )
        return TreeSearchFitResult(
            train_mse=train_mse,
            extrap_mse=extrap_mse,
            history=depth_history,
            extrap_history=extrap_history,
            best_depth=best_depth,
            selected_expressions=list(self.selected_expressions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.selected_nodes:
            raise RuntimeError("EMLTreeSearchRegressor must be fit before calling forward")

        feature_columns = [
            self._evaluate_node(node, x).unsqueeze(-1) for node in self.selected_nodes
        ]
        features = torch.cat(feature_columns, dim=-1)
        coefficients = self.coefficients.to(device=x.device, dtype=x.dtype)
        bias = self.bias.to(device=x.device, dtype=x.dtype)
        return (features @ coefficients.unsqueeze(-1) + bias).to(dtype=x.dtype)

    def export_metadata(self) -> dict[str, object]:
        selected_node_counts = [node.node_count() for node in self.selected_nodes]
        selected_depths = [node.tree_depth() for node in self.selected_nodes]
        return {
            "selected_expressions": list(self.selected_expressions),
            "selected_nodes": [node.to_dict() for node in self.selected_nodes],
            "selected_expression_count": len(self.selected_nodes),
            "selected_node_counts": selected_node_counts,
            "selected_total_nodes": sum(selected_node_counts),
            "selected_max_depth": max(selected_depths, default=0),
            "max_depth": self.max_depth,
            "beam_width": self.beam_width,
            "max_basis_size": self.max_basis_size,
            "min_improvement": self.min_improvement,
            "selection_pool_size": self.selection_pool_size,
        }

    def _make_terminal_candidates(
        self,
        train_inputs: torch.Tensor,
        extrap_inputs: torch.Tensor | None,
        train_targets: torch.Tensor,
    ) -> list[_Candidate]:
        terminals = [
            (TreeSearchNode("x0"), train_inputs[:, 0]),
            (TreeSearchNode("x1"), train_inputs[:, 1]),
            (TreeSearchNode("one"), torch.ones_like(train_inputs[:, 0])),
        ]
        candidates: list[_Candidate] = []
        for node, values_train in terminals:
            values_extrap = None
            if extrap_inputs is not None:
                if node.kind == "x0":
                    values_extrap = extrap_inputs[:, 0]
                elif node.kind == "x1":
                    values_extrap = extrap_inputs[:, 1]
                else:
                    values_extrap = torch.ones_like(extrap_inputs[:, 0])
            score = self._single_feature_mse(values_train, train_targets)
            candidates.append(
                _Candidate(
                    node=node,
                    depth=0,
                    train_values=values_train,
                    extrap_values=values_extrap,
                    score=score,
                )
            )
        return candidates

    def _expand_depth(
        self,
        depth: int,
        pool: list[_Candidate],
        train_targets: torch.Tensor,
        seen: set[str],
    ) -> list[_Candidate]:
        new_candidates: list[_Candidate] = []
        for left in pool:
            for right in pool:
                if max(left.depth, right.depth) + 1 != depth:
                    continue

                node = TreeSearchNode("eml", left=left.node, right=right.node)
                expression = node.to_string()
                if expression in seen:
                    continue
                seen.add(expression)

                train_values = self._safe_eml(left.train_values, right.train_values)
                if train_values is None:
                    continue

                extrap_values = None
                if left.extrap_values is not None and right.extrap_values is not None:
                    extrap_values = self._safe_eml(left.extrap_values, right.extrap_values)
                    if extrap_values is None:
                        continue

                score = self._single_feature_mse(train_values, train_targets)
                new_candidates.append(
                    _Candidate(
                        node=node,
                        depth=depth,
                        train_values=train_values,
                        extrap_values=extrap_values,
                        score=score,
                    )
                )

        new_candidates.sort(key=lambda candidate: candidate.score)
        return new_candidates[: self.beam_width]

    def _greedy_select(
        self,
        candidates: list[_Candidate],
        train_targets: torch.Tensor,
        extrap_targets: torch.Tensor | None,
    ) -> dict[str, object]:
        remaining = list(candidates)
        selected: list[_Candidate] = []
        best_train = self._bias_only_mse(train_targets)
        best_selection = {
            "selected": [],
            "coefficients": torch.zeros(0, dtype=train_targets.dtype, device=train_targets.device),
            "bias": train_targets.mean(),
            "train_mse": best_train,
            "extrap_mse": float("nan"),
        }

        for _ in range(min(self.max_basis_size, len(remaining))):
            best_candidate_selection = None
            for candidate in remaining:
                proposed = selected + [candidate]
                coefficients, bias, train_mse = self._solve_linear_readout(
                    [item.train_values for item in proposed],
                    train_targets,
                )
                if best_candidate_selection is None or train_mse < best_candidate_selection["train_mse"]:
                    best_candidate_selection = {
                        "candidate": candidate,
                        "coefficients": coefficients,
                        "bias": bias,
                        "train_mse": train_mse,
                    }

            if best_candidate_selection is None:
                break
            if best_train - best_candidate_selection["train_mse"] < self.min_improvement:
                break

            selected.append(best_candidate_selection["candidate"])
            remaining = [candidate for candidate in remaining if candidate is not best_candidate_selection["candidate"]]
            best_train = best_candidate_selection["train_mse"]
            best_selection = {
                "selected": list(selected),
                "coefficients": best_candidate_selection["coefficients"],
                "bias": best_candidate_selection["bias"],
                "train_mse": best_candidate_selection["train_mse"],
                "extrap_mse": self._evaluate_selection_extrap(
                    selected,
                    best_candidate_selection["coefficients"],
                    best_candidate_selection["bias"],
                    extrap_targets,
                ),
            }

        if not best_selection["selected"] and candidates:
            fallback = min(candidates, key=lambda candidate: candidate.score)
            coefficients, bias, train_mse = self._solve_linear_readout(
                [fallback.train_values],
                train_targets,
            )
            best_selection = {
                "selected": [fallback],
                "coefficients": coefficients,
                "bias": bias,
                "train_mse": train_mse,
                "extrap_mse": self._evaluate_selection_extrap(
                    [fallback],
                    coefficients,
                    bias,
                    extrap_targets,
                ),
            }

        return best_selection

    def _evaluate_selection_extrap(
        self,
        selected: list[_Candidate],
        coefficients: torch.Tensor,
        bias: torch.Tensor,
        extrap_targets: torch.Tensor | None,
    ) -> float:
        if extrap_targets is None or not selected:
            return float("nan")

        if any(candidate.extrap_values is None for candidate in selected):
            return float("nan")
        features = torch.stack(
            [candidate.extrap_values for candidate in selected],
            dim=-1,
        )
        predictions = features @ coefficients + bias
        return F.mse_loss(predictions, extrap_targets).item()

    def _commit_selection(self, selection: dict[str, object]) -> None:
        selected = selection["selected"]
        coefficients = selection["coefficients"]
        bias = selection["bias"]

        self.selected_nodes = [candidate.node for candidate in selected]
        self.selected_expressions = [candidate.expression for candidate in selected]
        self.coefficients = coefficients.detach().clone()
        self.bias = bias.detach().clone()

    def _single_feature_mse(
        self,
        values: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        coefficients, bias, mse = self._solve_linear_readout([values], targets)
        del coefficients, bias
        return mse

    def _solve_linear_readout(
        self,
        features: list[torch.Tensor],
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        design = torch.stack(features, dim=-1)
        ones = torch.ones(
            design.shape[0],
            1,
            device=design.device,
            dtype=design.dtype,
        )
        system = torch.cat([design, ones], dim=-1).to(dtype=torch.float64)
        target_column = targets.unsqueeze(-1).to(dtype=torch.float64)
        solution = torch.linalg.lstsq(system, target_column).solution.squeeze(-1)
        coefficients = solution[:-1].to(dtype=design.dtype)
        bias = solution[-1].to(dtype=design.dtype)
        predictions = design @ coefficients + bias
        mse = F.mse_loss(predictions, targets).item()
        return coefficients, bias, mse

    def _bias_only_mse(self, targets: torch.Tensor) -> float:
        baseline = torch.full_like(targets, float(targets.mean()))
        return F.mse_loss(baseline, targets).item()

    def _safe_eml(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor | None:
        if not torch.all(right > 0):
            return None
        values = eml(left, right)
        if not torch.isfinite(values).all():
            return None
        return values

    def _evaluate_node(
        self,
        node: TreeSearchNode,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        if node.kind == "x0":
            return inputs[:, 0]
        if node.kind == "x1":
            return inputs[:, 1]
        if node.kind == "one":
            return torch.ones_like(inputs[:, 0])
        if node.left is None or node.right is None:
            raise ValueError("eml node requires left and right children")

        left = self._evaluate_node(node.left, inputs)
        right = self._evaluate_node(node.right, inputs)
        if not torch.all(right > 0):
            raise RuntimeError("Encountered non-positive right branch while evaluating EML tree")
        return eml(left, right)
