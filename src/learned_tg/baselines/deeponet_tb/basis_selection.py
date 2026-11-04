"""SVD trunk-basis selection from the trained solution-DeepONet trunk.

After training, the coordinate-only trunk is evaluated on a fixed offline
selection grid, giving ``T_sel`` of shape ``(N_sel, p)``.  The retained method
uses the top-``m`` right singular vectors of the row-weighted trunk matrix.

Row weighting acts on *rows* (the spatial quadrature weight ``sqrt(cell_volume)``);
for the uniform cell-centered grid this is a constant ``h`` and therefore leaves
the right singular vectors invariant, but it is applied explicitly so the path
is correct for any possibly non-uniform target grid.  SVD selection is an
offline operation that never touches test coefficient fields or solutions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def row_weight(T: np.ndarray, cell_volume: float) -> np.ndarray:
    """Apply the spatial quadrature weight to rows: ``sqrt(cell_volume) * T``."""
    return float(np.sqrt(cell_volume)) * T


@dataclass
class SelectionResult:
    method: str
    W: np.ndarray               # (p, m) SVD mixing matrix
    m: int


def select_svd(T_weighted: np.ndarray, m: int) -> SelectionResult:
    """Top-``m`` right singular vectors of the weighted trunk matrix.

    Returns W of shape (128, m) so that on any target grid ``V_h = T_h @ W``.
    """
    if T_weighted.ndim != 2:
        raise ValueError(f"T_weighted must be 2-D (N, p), got {tuple(T_weighted.shape)}")
    p = int(T_weighted.shape[1])
    if not (0 < m <= p):
        raise ValueError(f"svd selection requires 0 < m <= p, got m={m}, p={p}")
    _, _, vh = np.linalg.svd(T_weighted, full_matrices=False)
    W = np.ascontiguousarray(vh.T[:, :m])  # (p, m)
    return SelectionResult(method="svd", W=W, m=m)


def select_trunk_basis(
    T_sel: np.ndarray,
    m: int,
    *,
    cell_volume: float = 1.0,
) -> SelectionResult:
    """Select the retained TB-SVD basis on the offline selection grid."""
    return select_svd(row_weight(T_sel, cell_volume), m)


def apply_selection(T_target: np.ndarray, selection: SelectionResult) -> np.ndarray:
    """Build the coarse basis ``V`` on a target grid from a selection.

    The retained SVD selection returns ``T_target @ W`` with shape
    ``(N_target, m)``.
    """
    if selection.method != "svd":
        raise ValueError(f"only the retained SVD selection is supported, got {selection.method!r}")
    return np.ascontiguousarray(T_target @ selection.W)
