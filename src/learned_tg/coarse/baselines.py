"""Numerical routines used by the paper; no experiment orchestration."""
from __future__ import annotations
import argparse
from typing import Any, Dict, Sequence, Tuple
import torch
from learned_tg.problems.static_helpers import _time_call

@torch.no_grad()
def _compress_global_bases(
    bases: Sequence[torch.Tensor],
    requested_dims: Sequence[int],
    device: torch.device,
) -> Tuple[Dict[int, torch.Tensor], float, int]:
    requested_dims = sorted(set(int(dim) for dim in requested_dims))
    if not requested_dims or min(requested_dims) <= 0:
        raise ValueError("global dimensions must be positive")

    def work() -> Tuple[Dict[int, torch.Tensor], int]:
        columns = torch.cat([basis.squeeze(0) for basis in bases], dim=1)
        norms = torch.linalg.vector_norm(columns, dim=0).clamp_min(1e-30)
        columns = columns / norms.unsqueeze(0)
        gram = torch.matmul(columns.transpose(0, 1), columns)
        gram = 0.5 * (gram + gram.transpose(0, 1))
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        threshold = max(float(eigenvalues[0].item()), 1.0) * 1e-12
        numerical_rank = int((eigenvalues > threshold).sum().item())
        outputs: Dict[int, torch.Tensor] = {}
        for requested in requested_dims:
            effective = min(int(requested), numerical_rank)
            if effective < requested:
                raise RuntimeError(
                    f"global pool rank {numerical_rank} is smaller than requested dimension {requested}"
                )
            vectors = eigenvectors[:, :effective]
            scales = eigenvalues[:effective].clamp_min(threshold).rsqrt()
            basis = torch.matmul(columns, vectors) * scales.unsqueeze(0)
            outputs[requested] = basis.unsqueeze(0).contiguous()
        return outputs, numerical_rank

    (outputs, numerical_rank), elapsed = _time_call(device, work)
    return outputs, float(elapsed), int(numerical_rank)

def _learned_inputs(sample: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, ...]:
    a_field = sample["a_field"].to(device=device, dtype=torch.float64)
    phi = sample["phi"].to(device=device, dtype=torch.float64)
    X = sample["X"].to(device=device, dtype=torch.float64)
    Y = sample["Y"].to(device=device, dtype=torch.float64)
    if X.dim() == 2:
        X = X.unsqueeze(0).unsqueeze(0)
    elif X.dim() == 3:
        X = X.unsqueeze(1)
    if Y.dim() == 2:
        Y = Y.unsqueeze(0).unsqueeze(0)
    elif Y.dim() == 3:
        Y = Y.unsqueeze(1)
    return a_field, phi, X, Y

@torch.no_grad()
def _predict_learned_basis(
    learned_model: Any,
    sample: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, float]:
    device = torch.device(args.device)
    inputs = _learned_inputs(sample, device)

    def work() -> torch.Tensor:
        basis = learned_model.predict_basis(*inputs, active_dim=int(args.m_dim))
        return torch.nan_to_num(basis.to(dtype=torch.float64), nan=0.0, posinf=0.0, neginf=0.0)

    basis, elapsed = _time_call(device, work)
    return basis, float(elapsed)
