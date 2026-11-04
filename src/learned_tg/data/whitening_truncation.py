"""
Adaptive whitening truncation utilities for error-family snapshot caching.

Provides:
  - adaptive_truncation_rank: per-sample spectral truncation using log-gap detection
  - compute_a_whitened_basis_with_eigvals: A-metric whitened basis + Gram eigenvalues
  - compute_l2_whitened_basis_with_eigvals: L2 whitened basis + squared singular values
  - apply_adaptive_padding: zero-pad whitened basis beyond per-sample k_mu
"""

import math
from typing import Tuple

import torch


@torch.no_grad()
def adaptive_truncation_rank(
    lambdas: torch.Tensor,
    floor_relative: float = 1e-12,
    min_rank: int = 4,
    max_rank: int = 30,
) -> int:
    """
    Two-step adaptive spectral truncation rank k*.

    Step 1: exclude modes where lambda_k / lambda_1 < floor_relative (numerical noise).
    Step 2: among remaining modes, find the position of the largest log-gap
            log(lambda_k) - log(lambda_{k+1}).  That position is k*.

    The search range for k* is [min_rank, min(q_eff, max_rank)], ensuring
    min_rank <= k* <= max_rank.

    Parameters
    ----------
    lambdas:
        Eigenvalues (any order, shape ``(q,)``).  Need not be normalised.
    floor_relative:
        Relative floor below which eigenvalues are treated as numerical noise.
        Conservative default ``1e-12`` avoids cutting true signal.
    min_rank:
        Minimum allowed k*.
    max_rank:
        Maximum allowed k*.

    Returns
    -------
    int
        Adaptive truncation rank k*, satisfying min_rank <= k* <= max_rank.
    """
    lambdas = lambdas.detach().double()
    q = int(lambdas.shape[0])

    lambdas, _ = torch.sort(lambdas, descending=True)

    min_rank = max(1, int(min_rank))
    max_rank = max(min_rank, min(int(max_rank), q))

    lam1 = float(lambdas[0].item())
    if lam1 <= 0.0:
        return min_rank

    # Step 1: number of effective (non-noise) modes
    threshold = float(floor_relative) * lam1
    q_eff = int((lambdas > threshold).sum().item())
    q_eff = max(q_eff, min_rank)

    # Step 2: search for maximum log-gap.
    # Gap at 0-indexed position i: log(lambda[i]) - log(lambda[i+1])
    # This corresponds to keeping i+1 modes (truncation rank = i+1).
    # We want k* in [min_rank, min(q_eff, max_rank)], so we search
    # i in [min_rank-1, min(q_eff, max_rank)-1] and need i+1 < q.
    search_lo = min_rank - 1
    search_hi = min(q_eff, max_rank) - 1
    search_hi = min(search_hi, q - 2)  # need lambda[i+1] to exist

    if search_lo > search_hi:
        return min(min_rank, max_rank)

    lam_slice = lambdas[search_lo : search_hi + 2].clamp_min(1e-300)
    log_lam = torch.log(lam_slice)
    gaps = log_lam[:-1] - log_lam[1:]  # shape: (search_hi - search_lo + 1,)

    best_local_idx = int(gaps.argmax().item())
    k_star = search_lo + 1 + best_local_idx

    return int(max(min_rank, min(k_star, max_rank)))


@torch.no_grad()
def compute_a_whitened_basis_with_eigvals(
    E: torch.Tensor,
    AE: torch.Tensor,
    rank: int,
    reg: float = 1e-12,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A-metric whitened basis and descending Gram eigenvalues.

    Parameters
    ----------
    E:
        Snapshot family, shape ``(B, N, q)``.
    AE:
        ``A @ E``, shape ``(B, N, q)``.
    rank:
        Number of whitened basis columns to compute (upper bound; actual
        output has ``min(rank, q)`` columns).
    reg:
        Floor applied to eigenvalues before taking the inverse square root.
    beta:
        Whitening strength in [0, 1].  beta=1 is full whitening (each column
        has A-norm 1, identical to the previous implementation).  beta=0
        leaves the spectral scaling untouched (columns have A-norm sigma_j).
        For general beta, column j has A-norm sigma_j^{1-beta}.  beta=1 uses
        a fast path that is bit-identical to the pre-beta implementation.

    Returns
    -------
    U_k : (B, N, r)
        Partially whitened basis, ``r = min(rank, q)``.
    eigvals_desc : (B, q)
        Gram eigenvalues in descending order (all q values for diagnostics).
    """
    if not (0.0 <= float(beta) <= 1.0):
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    E = E.double()
    AE = AE.to(dtype=E.dtype, device=E.device)
    B, N, q = E.shape

    gram = 0.5 * (
        torch.matmul(E.transpose(1, 2), AE) + torch.matmul(AE.transpose(1, 2), E)
    )
    eigvals, eigvecs = torch.linalg.eigh(gram)
    eigvals_desc, sort_idx = torch.sort(eigvals, dim=-1, descending=True)
    eigvecs_desc = torch.gather(
        eigvecs,
        -1,
        sort_idx.unsqueeze(-2).expand(-1, q, -1),
    )

    r = min(int(rank), q)
    sigma = eigvals_desc[:, :r].clamp_min(float(reg))
    V_k = eigvecs_desc[:, :, :r]
    if math.isclose(beta, 1.0, abs_tol=1e-12):
        # Fast path: exactly the pre-beta implementation, bit-identical.
        scale = sigma.rsqrt()
    else:
        # Partial whitening: column j is scaled by lambda_j^{-beta/2}.
        # beta=0 => scale=1 (no whitening); beta=1 => lambda^{-1/2} (full whitening).
        scale = sigma.pow(-0.5 * float(beta))
    U_k = torch.matmul(E, V_k * scale.unsqueeze(-2))

    return U_k.contiguous(), eigvals_desc.contiguous()


@torch.no_grad()
def compute_l2_whitened_basis_with_eigvals(
    E: torch.Tensor,
    rank: int,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    L2 whitened basis (left singular vectors) and squared singular values.

    Parameters
    ----------
    E:
        Snapshot family, shape ``(B, N, q)``.
    rank:
        Number of whitened basis columns.
    beta:
        Whitening strength in [0, 1].  beta=1 returns the L2-orthonormal left
        singular vectors (identical to the pre-beta implementation).  beta=0
        scales column j by sigma_j so its L2-norm equals sigma_j.  beta=1 uses
        a fast path that is bit-identical to the pre-beta implementation.

    Returns
    -------
    U_k : (B, N, r)
        Partially whitened basis, ``r = min(rank, q, N)``.
    sigma_sq_desc : (B, q)
        Squared singular values in descending order.
    """
    if not (0.0 <= float(beta) <= 1.0):
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    E = E.double()
    U, S, _ = torch.linalg.svd(E, full_matrices=False)
    r = min(int(rank), int(U.shape[-1]))
    if math.isclose(beta, 1.0, abs_tol=1e-12):
        # Fast path: exactly the pre-beta implementation, bit-identical.
        return U[:, :, :r].contiguous(), (S ** 2).contiguous()
    # Partial whitening: scale column j by sigma_j^{1-beta} (exponent >= 0).
    scale = S[:, :r].pow(1.0 - float(beta))  # (B, r)
    return (U[:, :, :r] * scale.unsqueeze(-2)).contiguous(), (S ** 2).contiguous()


@torch.no_grad()
def apply_adaptive_padding(
    U_k: torch.Tensor,
    eigvals_desc: torch.Tensor,
    floor_relative: float,
    min_rank: int,
    max_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply per-sample adaptive truncation and zero-pad columns beyond k_mu.

    Parameters
    ----------
    U_k:
        Whitened basis, shape ``(B, N, k_max)``.
    eigvals_desc:
        Eigenvalues in descending order, shape ``(B, q)``.
    floor_relative, min_rank, max_rank:
        Passed verbatim to :func:`adaptive_truncation_rank`.

    Returns
    -------
    U_padded : (B, N, k_max)
        Zero-padded whitened basis.
    k_per_sample : (B,) int64 tensor
        Per-sample truncation rank k_mu.
    """
    B, N, k_max = U_k.shape
    U_padded = U_k.clone()
    k_list = []

    for i in range(B):
        k_mu = adaptive_truncation_rank(
            eigvals_desc[i],
            floor_relative=floor_relative,
            min_rank=min_rank,
            max_rank=max_rank,
        )
        if k_mu < k_max:
            U_padded[i, :, k_mu:] = 0.0
        k_list.append(k_mu)

    k_per_sample = torch.tensor(k_list, dtype=torch.long, device=U_k.device)
    return U_padded, k_per_sample
