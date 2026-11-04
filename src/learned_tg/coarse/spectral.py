"""Numerical routines used by the paper; no experiment orchestration."""
from __future__ import annotations
import argparse
import copy
import hashlib
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple
import numpy as np
import torch
from learned_tg.coarse.pod import _make_prepared_coarse_basis
from learned_tg.problems.static_helpers import _apply_A_interface, _sync_if_needed, generate_interface_samples_for_grid
from learned_tg.model.coarse_spaces.local_spectral import build_modified_geneo_coarse_space

def _sample_field_sha256(sample: Mapping[str, Any]) -> str:
    """Hash the canonical CPU coefficient field used by every compute device."""
    field = sample["a_field"]
    if not isinstance(field, torch.Tensor):
        raise TypeError("sample a_field must be a torch.Tensor")
    array = np.ascontiguousarray(field.detach().cpu().to(torch.float64).numpy())
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()

def _generate_canonical_interface_samples(
    n: int,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """Generate held-out coefficients on CPU, then move solver fields to device.

    PyTorch CPU and CUDA random-number streams are not bitwise identical.  A
    CPU-only generation stage plus a saved field hash makes a seed identify the
    same coefficient sample on every compute device and machine.
    """
    generation_args = copy.copy(args)
    generation_args.device = str(args.sample_generation_device)
    samples = generate_interface_samples_for_grid(
        int(n),
        generation_args,
        num_samples=int(args.num_samples),
        seed=int(args.seed),
    )
    target = torch.device(args.device)
    for sample in samples:
        if bool(sample.get("failed", False)):
            continue
        sample["sample_generation_device"] = str(args.sample_generation_device)
        sample["sample_field_sha256"] = _sample_field_sha256(sample)
        for name in ("phi", "a_field", "a_hard", "b", "rhs"):
            value = sample.get(name)
            if isinstance(value, torch.Tensor):
                sample[name] = value.to(device=target, dtype=torch.float64)
    return samples

@torch.no_grad()
def _prepare_raw_basis_staged(
    raw_basis: torch.Tensor,
    operator: Any,
    args: argparse.Namespace,
    *,
    coarse_basis_mode: str = "a_orthonormal",
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Prepare the common TG basis while timing, without changing its algebra."""

    device = torch.device(args.device)
    basis = raw_basis.to(device=device, dtype=torch.float64)
    _sync_if_needed(device)
    started = time.perf_counter()
    basis_A = _apply_A_interface(operator.asm, basis)
    _sync_if_needed(device)
    basis_A_apply_time = time.perf_counter() - started
    prepare_timing: Dict[str, float] = {}
    prepared = _make_prepared_coarse_basis(
        basis,
        basis_A,
        reg=1e-10,
        coarse_basis_mode=coarse_basis_mode,
        timing_diagnostics=prepare_timing,
    )
    timing = {
        "basis_A_apply_time": float(basis_A_apply_time),
        "global_A_orthonormalization_time": float(
            prepare_timing.get("global_A_orthonormalization_time", 0.0)
        ),
        "Galerkin_assembly_time": float(
            prepare_timing.get("Galerkin_assembly_time", 0.0)
        ),
        "basis_prepare_time": float(
            basis_A_apply_time + prepare_timing.get("total_prepare_coarse_basis_time", 0.0)
        ),
    }
    Q, AQ = prepared["Q"], prepared["AQ"]
    if str(coarse_basis_mode).lower() in {"a_orthonormal", "a_orth", "orthonormal"}:
        eye = torch.eye(Q.shape[-1], device=Q.device, dtype=Q.dtype).unsqueeze(0)
        timing["A_orthonormality_error"] = float(
            torch.linalg.matrix_norm(Q.transpose(1, 2) @ AQ - eye).amax().item()
        )
    else:
        timing["A_orthonormality_error"] = float("nan")
    timing["coarse_basis_mode"] = str(prepared.get("coarse_basis_mode", coarse_basis_mode))
    timing["final_A_rank"] = int(
        prepare_timing.get("input_A_numerical_rank", prepared["basis_dim"])
    )
    timing["wrapper_basis_dimension"] = int(prepared["basis_dim"])
    timing["input_A_rank_floor"] = float(
        prepare_timing.get("input_A_rank_floor", 0.0)
    )
    return prepared, timing

def _build_local_spectral(
    operator: Any,
    args: argparse.Namespace,
    *,
    selection: str,
    sample_id: int,
    threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    kwargs: Dict[str, Any] = {
        "selection": selection,
        "dictionary_modes_per_patch": int(args.dictionary_modes_per_patch),
        "threshold_initial_modes": int(args.threshold_initial_modes),
        "max_local_modes": int(args.max_local_modes),
        "dense_cutoff": int(args.dense_cutoff),
        "eig_tol": float(args.local_eig_tol),
        "eigen_residual_tol": float(args.local_eigen_residual_tol),
        "diagnostic_spectrum_modes": int(args.diagnostic_spectrum_modes),
        "sample_id": int(sample_id),
        "seed": int(args.seed),
    }
    if selection == "compressed_local_spectral_dictionary":
        kwargs["rank"] = int(args.m_dim)
    else:
        kwargs["threshold"] = args.spectral_threshold if threshold is None else float(threshold)
    return build_modified_geneo_coarse_space(operator.asm, **kwargs)
