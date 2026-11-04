"""Numerical routines used by the paper; no experiment orchestration."""
from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
import torch
from learned_tg.baselines.deeponet_tb.basis_selection import SelectionResult
from learned_tg.baselines.deeponet_tb.solution_deeponet import SolutionDeepONet, cell_center_coords
from learned_tg.coarse.pod import _apply_A_interface, _make_prepared_coarse_basis
from learned_tg.model.interface import JumpingCoefficient

DEFAULT_TB_QR_EPSILON = 1e-10

def evaluate_trunk_matrix(
    model: SolutionDeepONet,
    n: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Evaluate the coordinate-only trunk on an n x n grid: (N, p) with N = n*n."""
    coords = cell_center_coords(n, device=device, dtype=torch.float32)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        T = model.trunk_matrix(coords)  # (N, p) float32
    return T.to(dtype=dtype).contiguous()

def qr_truncate_basis(
    candidate: torch.Tensor,
    epsilon: float = DEFAULT_TB_QR_EPSILON,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Thin QR followed by the paper's absolute ``R_jj`` rank threshold.

    ``candidate`` must have shape ``(1, N, m)``.  The returned basis has shape
    ``(1, N, r)`` with orthonormal columns and ``r <= m``.  An empty retained
    space is rejected because it cannot define a coarse correction.
    """
    if candidate.dim() != 3 or int(candidate.shape[0]) != 1:
        raise ValueError(
            "candidate trunk basis must have shape (1,N,m), "
            f"got {tuple(candidate.shape)}"
        )
    _, num_nodes, raw_dim = candidate.shape
    if int(raw_dim) <= 0 or int(num_nodes) < int(raw_dim):
        raise ValueError(
            "candidate trunk basis requires 0 < m <= N, "
            f"got N={int(num_nodes)}, m={int(raw_dim)}"
        )
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError(f"qr epsilon must be finite and non-negative, got {epsilon}")

    Q, R = torch.linalg.qr(candidate[0], mode="reduced")
    r_diagonal = torch.abs(torch.diagonal(R, offset=0))
    keep = r_diagonal >= epsilon
    retained_dim = int(keep.sum().item())
    if retained_dim == 0:
        raise ValueError(
            "all candidate trunk basis columns were rejected by QR: "
            f"max |R_jj|={float(r_diagonal.max().item()):.6e}, epsilon={epsilon:.6e}"
        )
    kept_indices = torch.nonzero(keep, as_tuple=False).flatten()
    dropped_indices = torch.nonzero(~keep, as_tuple=False).flatten()
    basis = Q.index_select(1, kept_indices).unsqueeze(0).contiguous()
    info: Dict[str, Any] = {
        "raw_dimension": int(raw_dim),
        "retained_dimension": retained_dim,
        "qr_epsilon": epsilon,
        "r_diagonal": [float(value) for value in r_diagonal.detach().cpu().tolist()],
        "kept_indices": [int(value) for value in kept_indices.detach().cpu().tolist()],
        "dropped_indices": [int(value) for value in dropped_indices.detach().cpu().tolist()],
    }
    return basis, info

def build_tb_basis_with_qr_info(
    T_h: torch.Tensor,
    selection: SelectionResult,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    qr_epsilon: float = DEFAULT_TB_QR_EPSILON,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Build a selected candidate basis and return its QR diagnostics."""
    if selection.method != "svd":
        raise ValueError(
            f"only the retained SVD trunk-basis selection is supported, got {selection.method!r}"
        )
    T_np = T_h.detach().cpu().numpy()
    V_np = np.ascontiguousarray(T_np @ selection.W)
    V = torch.from_numpy(V_np).to(device=device, dtype=dtype).unsqueeze(0)
    return qr_truncate_basis(V.contiguous(), epsilon=qr_epsilon)

def prepare_tb_coarse_basis(
    operator: JumpingCoefficient,
    V: torch.Tensor,
    coarse_basis_mode: str = "a_orthonormal",
) -> Dict[str, Any]:
    """Prepare the TB coarse basis through the existing A-orthonormalization path.

    Returns the same prepared-coarse-basis dict as
    ``_make_prepared_coarse_basis`` (Q, AQ, A_rom, coarse_factor, ...).
    """
    AV = _apply_A_interface(operator.asm, V)
    return _make_prepared_coarse_basis(
        V,
        AV,
        reg=1e-10,
        coarse_basis_mode=coarse_basis_mode,
    )
