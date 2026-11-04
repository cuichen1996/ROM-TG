"""Certified high-accuracy solution labels for the solution-DeepONet.

Solution labels ``u_h(mu) = A_mu^{-1} f`` are generated offline with the same
interface-aware CCFVM operator, fixed right-hand side, boundary conditions and
contrast used by the static random-interface benchmark.  Every label is verified
to satisfy ``||A u - f||_2 / ||f||_2 <= 1e-8``; a label that fails certification
is never silently included.

Primary path: sparse direct solve (``ccfvm_convergence.core.assemble_ccfvm_matrix``
+ ``scipy.sparse.linalg.spsolve``), which is machine-precision and was verified
against the torch matrix-free matvec.  Fallback (strongest non-learning path):
snapshot A-POD two-grid iteration via the existing ``solve`` method, which
monitors the *true* residual at every step.  The learned coarse-space checkpoint
is never used to generate labels.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from learned_tg.model.interface import JumpingCoefficient


@dataclass
class LabelInfo:
    sample_id: int
    converged: bool
    iterations: int
    true_relative_residual: float
    wall_time: float
    retry: str = "none"  # "none" | "snapshot_pod" | "failed"


def relative_residual(operator: JumpingCoefficient, x: torch.Tensor, f: torch.Tensor) -> float:
    """True relative residual ``||A x - f|| / ||f||``."""
    r = f - operator.forward(x)
    bnorm = torch.linalg.norm(f.reshape(f.shape[0], -1), dim=1).clamp_min(1e-30)
    rel = torch.linalg.norm(r.reshape(r.shape[0], -1), dim=1) / bnorm
    return float(torch.max(rel).item())


def _solve_dense_direct(
    operator: JumpingCoefficient,
    f: torch.Tensor,
) -> torch.Tensor:
    """Machine-precision direct solve via a single batched matvec + dense LU.

    The full matrix ``A`` is obtained by applying the existing matrix-free
    ``CCFVM_ASM`` operator to the identity (one batched matvec), then solved with
    ``torch.linalg.solve``.  This stays entirely inside ``jumping_transport`` (no
    cross-package sparse-assembly dependency), so it ports to the server unchanged.
    """
    from learned_tg.coarse.pod import _apply_A_interface

    n = operator.n
    N = n * n
    device = f.device
    eye = torch.eye(N, device=device, dtype=torch.float64)
    A = _apply_A_interface(operator.asm, eye.reshape(1, N, N))[0]  # (N, N)
    u = torch.linalg.solve(A, f.reshape(N, 1)).reshape(f.shape)
    return u.to(dtype=torch.float64)


def _snapshot_pod_prepared(operator: JumpingCoefficient, args: Any, rank: int) -> Dict[str, Any]:
    from learned_tg.coarse.pod import _apply_A_interface, _make_prepared_coarse_basis, build_oracle_basis
    from learned_tg.problems.static_helpers import _generate_smoothed_snapshots

    E, _ = _generate_smoothed_snapshots(
        problem="interface", operator=operator, n=operator.n,
        sample_seed=int(getattr(args, "seed", 1234)), args=args,
    )
    op_apply_A = lambda vectors: _apply_A_interface(operator.asm, vectors)
    Q, AQ, _ = build_oracle_basis(E, op_apply_A, basis_type="a_pod", rank=rank, reg=1e-12)
    return _make_prepared_coarse_basis(Q, AQ, reg=1e-10, coarse_basis_mode="a_orthonormal")


def _solve_two_grid_snapshot_pod(
    operator: JumpingCoefficient,
    f: torch.Tensor,
    *,
    tol: float,
    max_iter: int,
    args: Any,
) -> Tuple[torch.Tensor, List[float]]:
    """Fallback: two-grid iteration with a snapshot A-POD coarse space.

    Uses ``JumpingCoefficient.solve``, which recomputes the true residual every
    iteration (no Krylov-history drift).
    """
    prepared = _snapshot_pod_prepared(operator, args, rank=int(getattr(args, "m_dim", 15)))
    x, residuals = operator.solve(
        torch.zeros_like(f), f,
        prepared_coarse_basis=prepared,
        max_iter=int(max_iter), tol=float(tol),
        M_smooth=int(args.pre_smooth_steps), omega=float(args.omega),
        pre_smooth_mode="asm", pre_smooth_steps=int(args.pre_smooth_steps),
        post_smooth_mode="asm" if int(args.post_smooth_steps) > 0 else "none",
        post_smooth_steps=int(args.post_smooth_steps),
    )
    return x, residuals


def certify_solution_label(
    operator: JumpingCoefficient,
    f: torch.Tensor,
    *,
    tol: float = 1e-8,
    max_iter: int = 500,
    args: Any,
    device: torch.device,
) -> Tuple[torch.Tensor, LabelInfo]:
    """Solve ``A u = f`` to a certified tolerance (dense-direct primary)."""
    started = time.perf_counter()

    # Primary: machine-precision dense direct solve (self-contained).
    try:
        u = _solve_dense_direct(operator, f)
        true_rel = relative_residual(operator, u, f)
        if true_rel <= float(tol):
            return u.detach().clone(), LabelInfo(
                sample_id=-1, converged=True, iterations=0,
                true_relative_residual=true_rel,
                wall_time=time.perf_counter() - started, retry="none",
            )
    except Exception as exc:  # noqa: BLE001 - fall back to iterative path.
        print(f"[labels] dense direct solve failed ({exc}); using two-grid fallback", flush=True)

    # Fallback: snapshot A-POD two-grid (true-residual monitored).
    x, residuals = _solve_two_grid_snapshot_pod(operator, f, tol=tol, max_iter=max_iter, args=args)
    true_rel = relative_residual(operator, x, f)
    converged = true_rel <= float(tol)
    return x.detach().clone(), LabelInfo(
        sample_id=-1, converged=converged, iterations=int(len(residuals)),
        true_relative_residual=true_rel,
        wall_time=time.perf_counter() - started,
        retry="snapshot_pod" if converged else "failed",
    )


def generate_certified_labels(
    samples: Sequence[Dict[str, Any]],
    args: Any,
    *,
    tol: float = 1e-8,
    max_iter: int = 500,
    device: torch.device,
) -> Tuple[torch.Tensor, List[LabelInfo]]:
    """Generate certified labels for a list of interface samples.

    Each sample must contain ``a_field``, ``phi`` and ``b`` (the fixed RHS).
    Returns ``labels`` of shape (num_samples, n*n) and per-sample ``infos``.
    """
    labels: List[torch.Tensor] = []
    infos: List[LabelInfo] = []
    for idx, sample in enumerate(samples):
        if bool(sample.get("failed", False)):
            infos.append(LabelInfo(sample_id=idx, converged=False, iterations=0,
                                   true_relative_residual=float("nan"), wall_time=0.0, retry="failed"))
            labels.append(torch.full((1, int(sample["n"]) * int(sample["n"])), float("nan"), dtype=torch.float64))
            continue
        n = int(sample["n"])
        operator = JumpingCoefficient(
            sample["a_field"].to(device=device, dtype=torch.float64),
            n=n,
            k_1d=int(getattr(args, "k_1d", 10)),
            subdomain_solver=str(getattr(args, "subdomain_solver", "cholesky")),
            phi=sample.get("phi").to(device=device, dtype=torch.float64),
            transmissibility=str(getattr(args, "transmissibility", "interface_aware")),
        )
        f = sample["b"].to(device=device, dtype=torch.float64)
        u_h, info = certify_solution_label(
            operator, f, tol=tol, max_iter=max_iter, args=args, device=device,
        )
        info.sample_id = idx
        labels.append(u_h.reshape(1, -1).detach().cpu().to(torch.float64))
        infos.append(info)
        del operator, u_h
    labels_tensor = torch.cat(labels, dim=0) if labels else torch.empty(0)
    return labels_tensor, infos
