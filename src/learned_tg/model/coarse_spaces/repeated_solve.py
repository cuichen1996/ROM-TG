"""Numerical routines used by the paper; no experiment orchestration."""
from __future__ import annotations
from typing import Any, Dict, Mapping, Tuple
import torch
from learned_tg.model.coarse_spaces.local_spectral import build_modified_geneo_coarse_space

ALL_DIRICHLET_BOUNDARY_TYPES = {
    "west": "dirichlet",
    "east": "dirichlet",
    "north": "dirichlet",
    "south": "dirichlet",
}

def build_threshold_geneo_raw_basis(
    asm: Any,
    *,
    threshold: float,
    boundary_types: Mapping[str, str],
    sample_id: int,
    max_local_modes: int = 64,
    threshold_initial_modes: int = 8,
    dense_cutoff: int = 512,
    eig_tol: float = 1e-9,
    eigen_residual_tol: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Build the standard threshold-selected modified GenEO span."""

    basis, diagnostics = build_modified_geneo_coarse_space(
        asm,
        selection="threshold_modified_geneo",
        threshold=float(threshold),
        threshold_initial_modes=int(threshold_initial_modes),
        max_local_modes=int(max_local_modes),
        dense_cutoff=int(dense_cutoff),
        eig_tol=float(eig_tol),
        eigen_residual_tol=float(eigen_residual_tol),
        boundary_types=dict(boundary_types),
        sample_id=int(sample_id),
    )
    if not diagnostics.get("threshold_complete", False):
        raise RuntimeError(
            "formal repeated-solve benchmark requires every local threshold to be "
            f"strictly bracketed; sample_id={sample_id}"
        )
    if float(diagnostics.get("max_selected_direct_eig_residual", 0.0)) > float(
        eigen_residual_tol
    ):
        raise RuntimeError(
            "modified GenEO accepted a finite eigenpair above the direct-pencil "
            f"residual tolerance for sample_id={sample_id}"
        )
    return basis, diagnostics
