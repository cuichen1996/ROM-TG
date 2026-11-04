"""
Orthogonality and subspace-overlap diagnostics for coarse bases.

All functions are operator-agnostic: operator applications (AQ, AX, …) are
either pre-computed by the caller or computed on demand via op_solver.apply_A.
No imports from model.interface; dependency goes: diagnostics → coarse_loss only.
"""

import numpy as np
import torch


def _symmetrize(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.transpose(-1, -2))


def build_orth_debug_dict(
    G_raw,
    reg,
    backend,
    eigvals_before=None,
    eigvals_after_reg=None,
    eigvals_clamped=None,
):
    G_dbg = _symmetrize(G_raw.detach())
    batch_size, basis_dim, _ = G_dbg.shape
    I = torch.eye(basis_dim, device=G_dbg.device, dtype=G_dbg.dtype).unsqueeze(0)
    G_reg_dbg = G_dbg + float(reg) * I

    diag = torch.diagonal(G_reg_dbg, dim1=-2, dim2=-1)
    offdiag = G_reg_dbg - torch.diag_embed(diag)

    eigvals_before_dbg = None if eigvals_before is None else eigvals_before.detach()
    eigvals_after_reg_dbg = None if eigvals_after_reg is None else eigvals_after_reg.detach()
    eigvals_clamped_dbg = None if eigvals_clamped is None else eigvals_clamped.detach()

    if eigvals_before_dbg is None:
        try:
            eigvals_before_dbg = torch.linalg.eigvalsh(G_dbg)
        except RuntimeError:
            eigvals_before_dbg = None

    if eigvals_after_reg_dbg is None and eigvals_before_dbg is not None:
        eigvals_after_reg_dbg = eigvals_before_dbg + float(reg)

    if eigvals_clamped_dbg is None and eigvals_after_reg_dbg is not None:
        eigvals_clamped_dbg = eigvals_after_reg_dbg.clamp_min(float(reg))

    if eigvals_clamped_dbg is not None:
        cond_den = eigvals_clamped_dbg.amin(dim=-1).clamp_min(max(float(reg), 1e-30))
        cond_est = (eigvals_clamped_dbg.amax(dim=-1) / cond_den).amax().item()
        min_clamped_eig = eigvals_clamped_dbg.amin().item()
    else:
        cond_est = float("nan")
        min_clamped_eig = float("nan")

    return {
        "backend": str(backend),
        "min_eig_before_reg": (
            eigvals_before_dbg.amin().item()
            if eigvals_before_dbg is not None else float("nan")
        ),
        "min_eig_after_reg": (
            eigvals_after_reg_dbg.amin().item()
            if eigvals_after_reg_dbg is not None else float("nan")
        ),
        "min_clamped_eig": min_clamped_eig,
        "min_diag": diag.amin().item(),
        "max_abs_offdiag": offdiag.abs().amax().item(),
        "cond_est": cond_est,
    }


def format_orth_debug(debug):
    if not debug:
        return "backend=unknown"
    return (
        "backend={backend}, min_eig_before_reg={min_eig_before_reg:.3e}, "
        "min_eig_after_reg={min_eig_after_reg:.3e}, min_clamped_eig={min_clamped_eig:.3e}, "
        "min_diag={min_diag:.3e}, max_abs_offdiag={max_abs_offdiag:.3e}, "
        "cond_est={cond_est:.3e}"
    ).format(**debug)


def metric_cross_gram(Q0, X, metric="l2", op_solver=None, AX=None):
    metric = str(metric).lower()

    if metric == "l2":
        return torch.matmul(Q0.transpose(1, 2), X)

    if metric == "a":
        if AX is None:
            if op_solver is None:
                raise ValueError("A-metric operations require a valid op_solver.")
            AX = op_solver.apply_A(X)
        return torch.matmul(Q0.transpose(1, 2), AX)

    raise ValueError(f"Unsupported metric={metric!r}. Expected one of {{'l2', 'a'}}.")


def subspace_shift_metric(Q0, Q, metric="l2", op_solver=None, AQ=None):
    gram = metric_cross_gram(Q0, Q, metric=metric, op_solver=op_solver, AX=AQ)
    residual = Q - torch.matmul(Q0, gram)
    return torch.linalg.matrix_norm(residual, ord="fro", dim=(-2, -1)) / np.sqrt(Q.shape[-1])


def euclidean_orthogonality_error(Q):
    batch_size, _, basis_dim = Q.shape
    I = torch.eye(basis_dim, device=Q.device, dtype=Q.dtype).unsqueeze(0)
    gram = torch.matmul(Q.transpose(1, 2), Q)
    return torch.linalg.matrix_norm(gram - I, ord="fro", dim=(-2, -1))


def basis_orthogonality_error(Q, metric="l2", op_solver=None, AQ=None):
    metric = str(metric).lower()

    if metric == "l2":
        return euclidean_orthogonality_error(Q)

    if metric == "a":
        if AQ is None:
            if op_solver is None:
                raise ValueError("A-metric orthogonality checks require a valid op_solver.")
            AQ = op_solver.apply_A(Q)

        batch_size, _, basis_dim = Q.shape
        I = torch.eye(basis_dim, device=Q.device, dtype=Q.dtype).unsqueeze(0)
        gram = _symmetrize(torch.matmul(Q.transpose(1, 2), AQ))
        return torch.linalg.matrix_norm(gram - I, ord="fro", dim=(-2, -1))

    raise ValueError(f"Unsupported metric={metric!r}. Expected one of {{'l2', 'a'}}.")


def max_metric_overlap(Q0, X, metric="l2", op_solver=None, AX=None):
    gram = metric_cross_gram(Q0, X, metric=metric, op_solver=op_solver, AX=AX)
    return gram.abs().amax(dim=(-2, -1))
