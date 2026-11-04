"""Batched 2D geometric multigrid V-cycle helpers for ASM subdomain solves.

All routines operate on tensors of shape (NB, 1, h, w) where NB is the
combined batch × subdomain dimension. This keeps everything as a flat batch
on GPU and avoids per-subdomain loops.
"""

import torch
import torch.nn.functional as F


def stencil_matvec_2d(u, D, W, E, N, S):
    """Apply 5-point stencil with zero-Dirichlet at boundary.

    u, D, W, E, N, S: all shape (NB, 1, h, w).
    Returns: A u, shape (NB, 1, h, w).
    """
    u_pad = F.pad(u, (1, 1, 1, 1), mode="constant", value=0)
    return (
        D * u
        + W * u_pad[..., 1:-1, :-2]
        + E * u_pad[..., 1:-1, 2:]
        + N * u_pad[..., :-2, 1:-1]
        + S * u_pad[..., 2:, 1:-1]
    )


def weighted_jacobi_smooth(u, rhs, D, W, E, N, S, omega, num_steps):
    """Apply num_steps of weighted Jacobi: u <- u + omega * (1/D) * (rhs - A u)."""
    if num_steps <= 0:
        return u
    inv_D = 1.0 / D.clamp_min(1e-30)
    for _ in range(num_steps):
        Au = stencil_matvec_2d(u, D, W, E, N, S)
        u = u + omega * inv_D * (rhs - Au)
    return u


def galerkin_coarsen_stencil(D_h, W_h, E_h, N_h, S_h):
    """Galerkin coarsen 5-point stencil under 2×2 piecewise-constant agglomeration.

    Input stencils: (NB, 1, h, w) where h and w must be even.
    Returns: 5 coarse stencils each shape (NB, 1, h/2, w/2).

    Coarse cell (I,J) aggregates fine cells (2I,2J), (2I,2J+1), (2I+1,2J), (2I+1,2J+1)
    named NW, NE, SW, SE respectively.

        E_H = E_NE + E_SE
        W_H = W_NW + W_SW
        N_H = N_NW + N_NE
        S_H = S_SW + S_SE
        D_H = sum(D corners) + internal coupling terms
    """
    assert D_h.shape[-1] % 2 == 0 and D_h.shape[-2] % 2 == 0, \
        f"Coarsening requires even spatial dims, got {D_h.shape}"

    def _s(t, di, dj):
        return t[..., di::2, dj::2]

    # 4 corners: NW=(0,0), NE=(0,1), SW=(1,0), SE=(1,1)
    E_H = _s(E_h, 0, 1) + _s(E_h, 1, 1)          # east flux out of coarse cell
    W_H = _s(W_h, 0, 0) + _s(W_h, 1, 0)          # west flux
    N_H = _s(N_h, 0, 0) + _s(N_h, 0, 1)          # north flux
    S_H = _s(S_h, 1, 0) + _s(S_h, 1, 1)          # south flux

    D_H = (
        _s(D_h, 0, 0) + _s(D_h, 0, 1) + _s(D_h, 1, 0) + _s(D_h, 1, 1)  # 4 diagonal sums
        + _s(E_h, 0, 0) + _s(W_h, 0, 1)   # NW-NE horizontal internal coupling
        + _s(E_h, 1, 0) + _s(W_h, 1, 1)   # SW-SE horizontal internal coupling
        + _s(S_h, 0, 0) + _s(N_h, 1, 0)   # NW-SW vertical internal coupling
        + _s(S_h, 0, 1) + _s(N_h, 1, 1)   # NE-SE vertical internal coupling
    )

    return D_H, W_H, E_H, N_H, S_H


def restrict_2d(r):
    """R operator: 2×2 sum aggregation.

    r: (NB, 1, h, w) with h, w even.
    Returns: (NB, 1, h/2, w/2).
    """
    NB, C, h, w = r.shape
    return r.reshape(NB, C, h // 2, 2, w // 2, 2).sum(dim=(3, 5))


def prolong_2d(e):
    """P = R^T operator: piecewise-constant injection.

    e: (NB, 1, h, w).
    Returns: (NB, 1, 2h, 2w) — each coarse value broadcast to its 2×2 fine block.
    """
    NB, C, h, w = e.shape
    out = e.unsqueeze(-1).unsqueeze(-3).expand(NB, C, h, 2, w, 2)
    return out.reshape(NB, C, 2 * h, 2 * w)


def build_subdomain_gmg_hierarchy(
    D_lvl0, W_lvl0, E_lvl0, N_lvl0, S_lvl0,
    coarsest_threshold=24,
):
    """Build full GMG hierarchy by repeated Galerkin coarsening.

    Input stencils at finest level (level 0): shape (NB, 1, s, s).

    Returns list of dicts, each with keys 'D', 'W', 'E', 'N', 'S'.
    Stops coarsening when next level size < coarsest_threshold or size is odd.
    """
    levels = [{"D": D_lvl0, "W": W_lvl0, "E": E_lvl0, "N": N_lvl0, "S": S_lvl0}]
    while True:
        cur = levels[-1]
        cur_size = cur["D"].shape[-1]
        if cur_size % 2 != 0:
            break
        next_size = cur_size // 2
        if next_size < coarsest_threshold:
            break
        D_c, W_c, E_c, N_c, S_c = galerkin_coarsen_stencil(
            cur["D"], cur["W"], cur["E"], cur["N"], cur["S"]
        )
        levels.append({"D": D_c, "W": W_c, "E": E_c, "N": N_c, "S": S_c})
    return levels


def assemble_coarsest_dense_matrix(D, W, E, N, S):
    """Assemble a dense matrix from a 5-point stencil for batched Cholesky.

    D, W, E, N, S: (NB, 1, s, s) — must be small enough for dense storage.
    Returns: (NB, s*s, s*s) dense matrix.
    """
    NB, _, s, _ = D.shape
    n_loc = s * s

    D_flat = D.reshape(NB, n_loc)
    W_flat = W.reshape(NB, n_loc)
    E_flat = E.reshape(NB, n_loc)
    N_flat = N.reshape(NB, n_loc)
    S_flat = S.reshape(NB, n_loc)

    A = torch.zeros(NB, n_loc, n_loc, device=D.device, dtype=D.dtype)
    idx = torch.arange(n_loc, device=D.device)

    A[:, idx, idx] = D_flat

    idx_w = idx[idx % s != 0]
    A[:, idx_w, idx_w - 1] = W_flat[:, idx_w]

    idx_e = idx[idx % s != s - 1]
    A[:, idx_e, idx_e + 1] = E_flat[:, idx_e]

    idx_n = idx[idx >= s]
    A[:, idx_n, idx_n - s] = N_flat[:, idx_n]

    idx_s = idx[idx < n_loc - s]
    A[:, idx_s, idx_s + s] = S_flat[:, idx_s]

    A = A + 1e-10 * torch.eye(n_loc, device=A.device, dtype=A.dtype).unsqueeze(0)
    return A


def subdomain_vcycle(
    rhs, levels, coarsest_cholesky_L,
    smooth_steps_pre=2, smooth_steps_post=2, omega=2.0 / 3.0,
):
    """Run one V-cycle on batched subdomain problems.

    rhs: (NB, C, s, s) — RHS at finest level; C channels processed in parallel.
    levels: list of stencil dicts from build_subdomain_gmg_hierarchy
    coarsest_cholesky_L: (NB, s_c*s_c, s_c*s_c) — lower Cholesky factor at coarsest level

    Returns: e of shape (NB, C, s, s) — approximate solution to A e = rhs
    """
    num_levels = len(levels)

    def _coarsest_solve(r):
        NB, C, s_c, _ = r.shape
        r_flat = r.reshape(NB, C, s_c * s_c).permute(0, 2, 1)   # (NB, s_c*s_c, C)
        e_flat = torch.cholesky_solve(r_flat, coarsest_cholesky_L)
        return e_flat.permute(0, 2, 1).reshape(NB, C, s_c, s_c)

    if num_levels == 1:
        return _coarsest_solve(rhs)

    r_lvl = [None] * num_levels
    e_lvl = [None] * num_levels

    r_lvl[0] = rhs
    e_lvl[0] = torch.zeros_like(rhs)

    # Downward sweep
    for lvl in range(num_levels - 1):
        stns = levels[lvl]
        e_lvl[lvl] = weighted_jacobi_smooth(
            e_lvl[lvl], r_lvl[lvl],
            stns["D"], stns["W"], stns["E"], stns["N"], stns["S"],
            omega=omega, num_steps=smooth_steps_pre,
        )
        Ae = stencil_matvec_2d(
            e_lvl[lvl],
            stns["D"], stns["W"], stns["E"], stns["N"], stns["S"],
        )
        r_lvl[lvl + 1] = restrict_2d(r_lvl[lvl] - Ae)
        e_lvl[lvl + 1] = torch.zeros_like(r_lvl[lvl + 1])

    # Coarsest solve
    e_lvl[-1] = _coarsest_solve(r_lvl[-1])

    # Upward sweep
    for lvl in range(num_levels - 2, -1, -1):
        e_lvl[lvl] = e_lvl[lvl] + prolong_2d(e_lvl[lvl + 1])
        stns = levels[lvl]
        e_lvl[lvl] = weighted_jacobi_smooth(
            e_lvl[lvl], r_lvl[lvl],
            stns["D"], stns["W"], stns["E"], stns["N"], stns["S"],
            omega=omega, num_steps=smooth_steps_post,
        )

    return e_lvl[0]
