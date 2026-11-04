"""Shared coarse-space loss utilities.

All functions in this module are operator-agnostic tensor routines.  Operator
applications such as ``A @ V`` or ``A @ E`` must be computed by the caller and
passed in as ``AV`` / ``AE``.

Shape convention:
    V, Q, E:  (B, N, m_or_q)
    AV, AQ, AE: same shape as the tensor they correspond to
    reduced Gram matrices: (B, m_or_q, m_or_q)
"""

import torch


def require_finite(x: torch.Tensor, name: str) -> torch.Tensor:
    """Assert tensor is finite after a numerical decomposition; raise on NaN/Inf."""
    if not torch.isfinite(x).all():
        raise RuntimeError(
            f"{name} contains non-finite values after numerical decomposition — "
            f"shape={tuple(x.shape)}, dtype={x.dtype}, "
            f"nan={x.isnan().sum().item()}, inf={x.isinf().sum().item()}"
        )
    return x


def canonical_metric(metric, name="metric"):
    key = str(metric).lower()
    if key in {"l2", "euclidean", "2"}:
        return "l2"
    if key in {"a", "energy", "a_inner", "a-inner"}:
        return "a"
    raise ValueError(f"{name} must be one of {{'l2', 'a'}}, got {metric!r}")


_canonical_metric = canonical_metric  # backward-compat alias


def _canonical_reduction(reduction):
    key = str(reduction).lower()
    if key in {"mean", "per_sample"}:
        return "per_sample_mean"
    if key in {"sum_ratio", "global"}:
        return "batch_weighted"
    if key not in {"batch_weighted", "per_sample_mean"}:
        raise ValueError(
            "reduction must be one of {'batch_weighted', 'per_sample_mean'}, "
            f"got {reduction!r}"
        )
    return key


def _check_snapshot_tensor(name, tensor):
    if tensor.dim() != 3:
        raise ValueError(f"{name} must have shape (B, N, k), got {tuple(tensor.shape)}")


def _clean_float64(tensor):
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float64)


def symmetrize_matrix(M):
    """Return the symmetric part of a batched square matrix."""
    return 0.5 * (M + M.transpose(-1, -2))


def _capture_clip_stats(captured_raw_nonnegative, total_nonnegative):
    capture_excess = (captured_raw_nonnegative - total_nonnegative).clamp_min(0.0)
    num_capture_clipped = (capture_excess > 0.0).sum().to(
        dtype=total_nonnegative.dtype,
    )
    max_capture_excess = (
        capture_excess.amax()
        if capture_excess.numel() > 0
        else total_nonnegative.new_tensor(0.0)
    )
    return num_capture_clipped, max_capture_excess


def a_orthonormalize_from_AV(V, AV, reg=1e-10):
    """A-orthonormalize columns of ``V`` using precomputed ``AV``.

    Parameters
    ----------
    V:
        Candidate basis with shape ``(B, N, m)``.
    AV:
        Operator-applied basis with shape ``(B, N, m)``.
    reg:
        Positive diagonal regularization for the local Gram matrix.

    Returns
    -------
    Q, AQ, A_rom:
        ``Q`` and ``AQ`` have shape ``(B, N, m)`` and satisfy
        ``Q.transpose(1, 2) @ AQ ~= I``. ``A_rom`` is that reduced Gram matrix.
    """
    _check_snapshot_tensor("V", V)
    _check_snapshot_tensor("AV", AV)
    if V.shape != AV.shape:
        raise ValueError(f"V and AV must have the same shape, got {tuple(V.shape)} vs {tuple(AV.shape)}")

    V = _clean_float64(V)
    AV = _clean_float64(AV).to(device=V.device)
    batch_size, _, basis_dim = V.shape
    eye = torch.eye(basis_dim, device=V.device, dtype=V.dtype).unsqueeze(0)
    G_raw = symmetrize_matrix(torch.matmul(V.transpose(1, 2), AV))
    G_reg = G_raw + float(reg) * eye

    try:
        chol = torch.linalg.cholesky(G_reg)
        Qt = torch.linalg.solve_triangular(
            chol,
            V.transpose(1, 2).contiguous(),
            upper=False,
        )
        AQt = torch.linalg.solve_triangular(
            chol,
            AV.transpose(1, 2).contiguous(),
            upper=False,
        )
        Q = Qt.transpose(1, 2)
        AQ = AQt.transpose(1, 2)
    except RuntimeError:
        eigvals, eigvecs = torch.linalg.eigh(G_raw)
        eigvals = eigvals.clamp_min(float(reg))
        G_inv_sqrt = torch.matmul(
            eigvecs * eigvals.rsqrt().unsqueeze(-2),
            eigvecs.transpose(-1, -2),
        )
        Q = torch.matmul(V, G_inv_sqrt)
        AQ = torch.matmul(AV, G_inv_sqrt)

    Q = require_finite(Q, "Q")
    AQ = require_finite(AQ, "AQ")
    A_rom = symmetrize_matrix(torch.matmul(Q.transpose(1, 2), AQ))
    return Q, AQ, require_finite(A_rom, "A_rom")


def euclidean_orthonormalize(V, reg=1e-12):
    """Return a Euclidean-orthonormal basis spanning the columns of ``V``."""
    _check_snapshot_tensor("V", V)
    V = _clean_float64(V)
    try:
        Q, _ = torch.linalg.qr(V, mode="reduced")
    except RuntimeError:
        basis_dim = int(V.shape[-1])
        eye = torch.eye(basis_dim, device=V.device, dtype=V.dtype).unsqueeze(0)
        G = symmetrize_matrix(torch.matmul(V.transpose(1, 2), V))
        eigvals, eigvecs = torch.linalg.eigh(G + float(reg) * eye)
        eigvals = eigvals.clamp_min(float(reg))
        G_inv_sqrt = torch.matmul(
            eigvecs * eigvals.rsqrt().unsqueeze(-2),
            eigvecs.transpose(-1, -2),
        )
        Q = torch.matmul(V, G_inv_sqrt)
    return require_finite(Q, "Q")


def _resolve_whiten_rank(E, rank):
    if rank is None:
        return min(int(E.shape[1]), int(E.shape[2]))
    rank = int(rank)
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    return max(1, min(rank, int(E.shape[1]), int(E.shape[2])))


def compute_metric_whitened_snapshot_family(
    E,
    rank,
    metric="l2",
    AE=None,
    reg=1e-12,
    output_layout="family",
):
    """Construct a metric-whitened snapshot family.

    Parameters
    ----------
    E:
        Snapshot family with shape ``(B, N, q)``.
    rank:
        Target whitening rank, clipped to ``min(N, q)``.
    metric:
        ``"l2"`` or ``"a"``.
    AE:
        Required for ``metric="a"``; must be ``A @ E`` with shape ``(B, N, q)``.
    output_layout:
        ``"family"`` returns ``(B, N, q)`` via ``U_k V_k^T`` and preserves the
        caller-facing snapshot count. ``"basis"`` returns ``(B, N, rank)`` with
        metric-orthonormal whitened columns.
    """
    metric = _canonical_metric(metric)
    _check_snapshot_tensor("E", E)
    E = _clean_float64(E)
    max_rank = _resolve_whiten_rank(E, rank)
    output_key = str(output_layout).lower()
    if output_key not in {"family", "basis", "rank"}:
        raise ValueError(
            "output_layout must be one of {'family', 'basis'}, "
            f"got {output_layout!r}"
        )

    if metric == "l2":
        U, _, Vh = torch.linalg.svd(E, full_matrices=False)
        U_k = U[:, :, :max_rank].contiguous()
        if output_key in {"basis", "rank"}:
            return require_finite(U_k, "U_k")
        Vh_k = Vh[:, :max_rank, :].contiguous()
        E_white = torch.matmul(U_k, Vh_k)
        return require_finite(E_white, "E_white")

    if AE is None:
        raise ValueError("A-metric whitening requires precomputed AE.")
    _check_snapshot_tensor("AE", AE)
    if AE.shape != E.shape:
        raise ValueError(f"E and AE must have the same shape, got {tuple(E.shape)} vs {tuple(AE.shape)}")
    AE = _clean_float64(AE).to(device=E.device)

    gram = symmetrize_matrix(torch.matmul(E.transpose(1, 2), AE))
    eigvals, eigvecs = torch.linalg.eigh(gram)
    eigvals_desc, sort_idx = torch.sort(eigvals, dim=-1, descending=True)
    eigvecs_desc = torch.gather(
        eigvecs,
        -1,
        sort_idx.unsqueeze(-2).expand(-1, eigvecs.shape[1], -1),
    )
    sigma = eigvals_desc[:, :max_rank].clamp_min(float(reg))
    V_k = eigvecs_desc[:, :, :max_rank].contiguous()
    U_A_k = torch.matmul(E, V_k * sigma.rsqrt().unsqueeze(-2))
    U_A_k = require_finite(U_A_k, "U_A_k")
    if output_key in {"basis", "rank"}:
        return U_A_k
    E_white = torch.matmul(U_A_k, V_k.transpose(1, 2))
    return require_finite(E_white, "E_white")


def capture_loss_from_precomputed(
    Q,
    BQ,
    E,
    BE,
    metric="l2",
    reduction="batch_weighted",
    eps=1e-12,
):
    """Compute relative projection loss from pre-orthonormalized data.

    ``Q`` must be orthonormal in the selected metric.  For ``metric="a"``,
    ``BE`` must be supplied and represents ``A @ E``.  ``BQ`` is accepted for
    API symmetry and diagnostics but is not needed for the coefficient formula.
    """
    metric = _canonical_metric(metric)
    reduction = _canonical_reduction(reduction)

    _check_snapshot_tensor("Q", Q)
    _check_snapshot_tensor("E", E)
    Q = _clean_float64(Q)
    E = _clean_float64(E).to(device=Q.device)
    if Q.shape[0] != E.shape[0] or Q.shape[1] != E.shape[1]:
        raise ValueError(f"Q and E shape mismatch: {tuple(Q.shape)} vs {tuple(E.shape)}")

    if metric == "l2":
        coeff = torch.matmul(Q.transpose(1, 2), E)
        total_energy = (E * E).sum(dim=1)
    else:
        if BE is None:
            raise ValueError("A-metric capture loss requires precomputed BE.")
        _check_snapshot_tensor("BE", BE)
        BE = _clean_float64(BE).to(device=Q.device)
        if BE.shape != E.shape:
            raise ValueError(f"E and BE must have the same shape, got {tuple(E.shape)} vs {tuple(BE.shape)}")
        coeff = torch.matmul(Q.transpose(1, 2), BE)
        total_energy = (E * BE).sum(dim=1)
        if BQ is not None and BQ.shape != Q.shape:
            raise ValueError(f"Q and BQ must have the same shape, got {tuple(Q.shape)} vs {tuple(BQ.shape)}")

    captured_energy_raw = (coeff * coeff).sum(dim=1)
    total_nonnegative = total_energy.clamp_min(0.0)
    upper_tol = 1e-8 * torch.maximum(total_nonnegative.detach(), torch.ones_like(total_nonnegative))
    captured_raw_nonnegative = captured_energy_raw.clamp_min(0.0)
    num_capture_clipped, max_capture_excess = _capture_clip_stats(
        captured_raw_nonnegative,
        total_nonnegative,
    )
    captured_energy = torch.minimum(
        captured_raw_nonnegative,
        total_nonnegative + upper_tol,
    )
    residual_energy = (total_nonnegative - captured_energy).clamp_min(0.0)

    sample_total_raw = total_nonnegative.sum(dim=-1)
    sample_total = sample_total_raw.clamp_min(float(eps))
    sample_captured = captured_energy.sum(dim=-1)
    sample_residual = residual_energy.sum(dim=-1)
    captured_ratio = sample_captured / sample_total
    per_sample_loss = sample_residual / sample_total
    per_sample_loss_mean = per_sample_loss.mean()

    if reduction == "per_sample_mean":
        loss = per_sample_loss_mean
    else:
        loss = sample_residual.sum() / sample_total_raw.sum().clamp_min(float(eps))

    stats = {
        "total_energy": total_energy,
        "captured_energy": captured_energy,
        "residual_energy": residual_energy,
        "captured_ratio": captured_ratio,
        "loss": loss,
        "per_sample_loss_mean": per_sample_loss_mean,
        "sum_total_energy": total_nonnegative.sum(),
        "sum_captured_energy": captured_energy.sum(),
        "sum_residual_energy": residual_energy.sum(),
        "num_snapshots": total_energy.new_tensor(float(total_energy.numel())),
        "num_capture_clipped": num_capture_clipped,
        "max_capture_excess": max_capture_excess,
    }
    return loss, stats


def stable_relative_projection_loss(
    V,
    BV,
    E,
    BE=None,
    *,
    metric="a",
    reg=1e-10,
    reduction="batch_weighted",
    eps=1e-12,
):
    """Stable relative projection loss for a raw, non-orthonormal basis.

    For ``metric="a"`` this evaluates

        ``captured = diag(E^T B V (V^T B V + reg I)^{-1} V^T B E)``

    snapshot-by-snapshot and then applies the same relative reduction as
    :func:`capture_loss_from_precomputed`.
    """
    metric = _canonical_metric(metric)
    reduction = _canonical_reduction(reduction)
    _check_snapshot_tensor("V", V)
    _check_snapshot_tensor("E", E)
    V = _clean_float64(V)
    E = _clean_float64(E).to(device=V.device)
    if V.shape[0] != E.shape[0] or V.shape[1] != E.shape[1]:
        raise ValueError(f"V and E shape mismatch: {tuple(V.shape)} vs {tuple(E.shape)}")

    if metric == "l2":
        BV = V
        BE = E
    else:
        if BV is None or BE is None:
            raise ValueError("A-metric raw projection requires BV and BE.")
        _check_snapshot_tensor("BV", BV)
        _check_snapshot_tensor("BE", BE)
        if BV.shape != V.shape:
            raise ValueError(f"V and BV must have the same shape, got {tuple(V.shape)} vs {tuple(BV.shape)}")
        if BE.shape != E.shape:
            raise ValueError(f"E and BE must have the same shape, got {tuple(E.shape)} vs {tuple(BE.shape)}")
        BV = _clean_float64(BV).to(device=V.device)
        BE = _clean_float64(BE).to(device=V.device)

    basis_dim = int(V.shape[-1])
    eye = torch.eye(basis_dim, device=V.device, dtype=V.dtype).unsqueeze(0)
    G = symmetrize_matrix(torch.matmul(V.transpose(1, 2), BV)) + float(reg) * eye
    rhs = torch.matmul(V.transpose(1, 2), BE)
    try:
        y = torch.linalg.solve(G, rhs)
    except RuntimeError:
        y = torch.matmul(torch.linalg.pinv(G), rhs)

    captured = (rhs * y).sum(dim=1)
    if metric == "l2":
        total = (E * E).sum(dim=1)
    else:
        total = (E * BE).sum(dim=1)

    total_nonnegative = total.clamp_min(0.0)
    captured_raw_nonnegative = captured.clamp_min(0.0)
    num_capture_clipped, max_capture_excess = _capture_clip_stats(
        captured_raw_nonnegative,
        total_nonnegative,
    )
    captured = torch.minimum(captured_raw_nonnegative, total_nonnegative)
    residual = (total_nonnegative - captured).clamp_min(0.0)
    sample_total_raw = total_nonnegative.sum(dim=-1)
    sample_total = total_nonnegative.sum(dim=-1).clamp_min(float(eps))
    sample_captured = captured.sum(dim=-1)
    sample_residual = residual.sum(dim=-1)
    captured_ratio = sample_captured / sample_total
    per_sample_loss = sample_residual / sample_total
    per_sample_loss_mean = per_sample_loss.mean()
    if reduction == "per_sample_mean":
        loss = per_sample_loss_mean
    else:
        loss = sample_residual.sum() / sample_total_raw.sum().clamp_min(float(eps))
    stats = {
        "total_energy": total,
        "captured_energy": captured,
        "residual_energy": residual,
        "captured_ratio": captured_ratio,
        "loss": loss,
        "per_sample_loss_mean": per_sample_loss_mean,
        "sum_total_energy": total_nonnegative.sum(),
        "sum_captured_energy": captured.sum(),
        "sum_residual_energy": residual.sum(),
        "num_snapshots": total.new_tensor(float(total.numel())),
        "num_capture_clipped": num_capture_clipped,
        "max_capture_excess": max_capture_excess,
    }
    return loss, stats
