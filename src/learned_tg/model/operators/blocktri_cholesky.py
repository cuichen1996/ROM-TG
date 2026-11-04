"""Batched block-tridiagonal Cholesky for five-point subdomain matrices.

A five-point matrix on an ``rows x cols`` patch, ordered row-major, is block
tridiagonal with tridiagonal ``cols x cols`` diagonal blocks T_i and diagonal
couplings diag(n_i) between patch rows i and i-1.  Its Cholesky factor keeps
that structure,
    C_i = diag(n_i) L_{i-1}^{-T},   L_i = chol(T_i - C_i C_i^T),
so a factor costs O(rows * cols^2) storage instead of O((rows * cols)^2).  The
diagonal blocks are stored inverted, together with G_i = L_i^{-1} C_i and
H_i = L_i^{-T} C_{i+1}^T, so each substitution step is one batched GEMM:
    y_i = L_i^{-1} r_i - G_i y_{i-1},   x_i = L_i^{-T} y_i - H_i x_{i+1}.
Everything stays on the device of the inputs.
"""

from typing import Tuple

import torch


def blocktri_cholesky_factor(
    diag: torch.Tensor, west: torch.Tensor, south: torch.Tensor, shift: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Factor the batch of SPD matrices A + shift * I given by their lower triangle.

    diag, west, south: (batch, rows, cols) with diag[:, i, j] = A[(i, j), (i, j)],
    west[:, i, j] = A[(i, j), (i, j - 1)] (column 0 unused) and
    south[:, i, j] = A[(i, j), (i - 1, j)] (row 0 unused).
    Returns (Linv, G, H), each of shape (batch, rows, cols, cols).
    """
    batch, rows, cols = diag.shape
    Linv = torch.empty((batch, rows, cols, cols), device=diag.device, dtype=diag.dtype)
    G = torch.zeros_like(Linv)
    H = torch.zeros_like(Linv)
    eye = torch.eye(cols, device=diag.device, dtype=diag.dtype).expand(batch, cols, cols)
    for i in range(rows):
        T_low = torch.diag_embed(west[:, i, 1:], offset=-1)
        T = torch.diag_embed(diag[:, i] + shift) + T_low + T_low.mT
        if i > 0:
            Ct = Linv[:, i - 1] * south[:, i, None, :]  # C_i^T = L_{i-1}^{-1} diag(n_i)
            T = T - Ct.mT @ Ct
        Linv[:, i] = torch.linalg.solve_triangular(torch.linalg.cholesky(T), eye, upper=False)
        if i > 0:
            G[:, i] = Linv[:, i] @ Ct.mT
            H[:, i - 1] = Linv[:, i - 1].mT @ Ct
    return Linv, G, H


def blocktri_cholesky_solve(
    Linv: torch.Tensor, G: torch.Tensor, H: torch.Tensor, rhs: torch.Tensor
) -> torch.Tensor:
    """Solve with the factors of blocktri_cholesky_factor; rhs (batch, rows, cols, C)."""
    rows = rhs.shape[1]
    y = Linv @ rhs
    for i in range(1, rows):
        y[:, i] = torch.baddbmm(y[:, i], G[:, i], y[:, i - 1], alpha=-1.0)
    x = Linv.mT @ y
    for i in range(rows - 2, -1, -1):
        x[:, i] = torch.baddbmm(x[:, i], H[:, i], x[:, i + 1], alpha=-1.0)
    return x
