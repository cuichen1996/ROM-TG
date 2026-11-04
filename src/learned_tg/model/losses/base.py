"""
Base loss utilities: snapshot normalization, A-projection base class.
"""

import torch
import torch.nn as nn
from contextlib import contextmanager

from learned_tg.model.losses import primitives as coarse_loss
from learned_tg.model.operators.ccfvm_asm import CCFVM_ASM
from learned_tg.model.basis_utils.diagnostics import build_orth_debug_dict as _build_orth_debug_dict, format_orth_debug as _format_orth_debug


# ── Shared helpers ────────────────────────────────────────────────────────────

def _symmetrize_matrix(M):
    return coarse_loss.symmetrize_matrix(M)


def _canonical_l2_a_metric(metric, name="metric"):
    key = str(metric).lower()
    if key in {"a", "energy", "a_inner", "a-inner"}:
        return "a"
    if key in {"l2", "euclidean", "2"}:
        return "l2"
    raise ValueError(f"{name} must be one of {{'a', 'l2'}}, got {metric!r}")


def _canonical_snapshot_whiten_metric(metric):
    return _canonical_l2_a_metric(metric, name="snapshot whitening metric")


def _resolve_snapshot_whiten_rank(E, rank):
    if rank is None:
        return None
    rank = int(rank)
    if rank <= 0:
        return None
    if E.dim() != 3:
        raise ValueError(
            f"Expected snapshot family with shape (B, N, q), got {tuple(E.shape)}"
        )
    max_rank = min(rank, int(E.shape[1]), int(E.shape[2]))
    if max_rank <= 0:
        raise ValueError(
            f"Unable to whiten snapshot family with rank={rank} for shape {tuple(E.shape)}"
        )
    return max_rank


def _compute_metric_whitened_snapshot_family(
    E,
    rank,
    metric="l2",
    op_solver=None,
    reg=1e-12,
):
    metric = _canonical_snapshot_whiten_metric(metric)
    max_rank = _resolve_snapshot_whiten_rank(E, rank)
    if max_rank is None:
        return None

    if metric == "l2":
        return coarse_loss.compute_metric_whitened_snapshot_family(
            E,
            rank=max_rank,
            metric="l2",
            reg=reg,
            output_layout="family",
        )

    if op_solver is None:
        raise ValueError("A-metric whitening requires a valid op_solver.")

    AE = op_solver.apply_A(E)
    return coarse_loss.compute_metric_whitened_snapshot_family(
        E,
        rank=max_rank,
        metric="a",
        AE=AE,
        reg=reg,
        output_layout="family",
    )


def normalize_snapshot_family(e_family, expected_num_nodes=None):
    """
    Normalize snapshot families to the canonical internal layout E: (B, N, q).

    Supported input layouts:
      - (B, N, q)        # canonical internal layout
      - (B, N, 1)
      - (B, N)
      - (B, q, N, 1)     # explicit legacy compatibility layout
      - (B, q, ny, nx)   # rectangular grid layout; ny*nx == N

    ``expected_num_nodes`` may be an int ``N`` or a tuple ``(ny, nx)``.
    """
    if e_family.dim() == 4:
        if e_family.shape[-1] == 1:
            # Legacy layout (B, q, N, 1) → (B, N, q)
            E = e_family.squeeze(-1).transpose(1, 2)
        else:
            # Rectangular grid layout (B, q, ny, nx) → (B, ny*nx, q)
            B, q, ny, nx = e_family.shape
            E = e_family.reshape(B, q, ny * nx).transpose(1, 2)
    elif e_family.dim() == 3:
        E = e_family
    elif e_family.dim() == 2:
        E = e_family.unsqueeze(-1)
    else:
        raise ValueError(
            "Unsupported e_family shape. Expected one of "
            "{(B, N, q), (B, N, 1), (B, N), (B, q, N, 1), (B, q, ny, nx)}, "
            f"got {tuple(e_family.shape)}"
        )

    if expected_num_nodes is not None:
        if isinstance(expected_num_nodes, (tuple, list)):
            N = int(expected_num_nodes[0]) * int(expected_num_nodes[1])
        else:
            N = int(expected_num_nodes)
        if E.shape[1] != N:
            raise ValueError(
                "e_family must follow the canonical layout (B, N, q) with "
                f"N={N}, got normalized shape {tuple(E.shape)}. "
                "If you still have a legacy 3D tensor shaped (B, q, N), transpose it "
                "explicitly before passing it here."
            )
    return E.contiguous()


# ── AProjectionLossBase ───────────────────────────────────────────────────────

class AProjectionLossBase(nn.Module):
    """
    提供:
      1) 批量施加 A 算子
      2) A-内积下的正交化
      3) 投影/正交化的数值诊断
    """
    def __init__(self, k_1d=10, target_overlap=0.3, reg=1e-10, dtype=torch.float64):
        super().__init__()
        self.k_1d = k_1d
        self.target_overlap = target_overlap
        self.reg = reg
        self.dtype = dtype
        self._forward_cache = None
        self._forward_cache_depth = 0
        self._last_orth_debug = None
        self._last_projection_debug = None

    def _build_solver(self, a_field):
        return CCFVM_ASM(
            a=a_field.to(self.dtype),
            k_1d=self.k_1d,
            target_overlap=self.target_overlap,
            build_local_solver=False,
        ).to(a_field.device)

    def _tensor_cache_key(self, tensor):
        return (
            int(tensor.data_ptr()),
            int(tensor.storage_offset()),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.device.type,
            tensor.device.index,
            str(tensor.dtype),
            int(getattr(tensor, "_version", 0)),
        )

    @contextmanager
    def _forward_cache_scope(self):
        self._forward_cache_depth += 1
        if self._forward_cache is None:
            self._forward_cache = {}
        try:
            yield
        finally:
            self._forward_cache_depth -= 1
            if self._forward_cache_depth <= 0:
                self._forward_cache = None
                self._forward_cache_depth = 0

    def _get_solver(self, a_field):
        if self._forward_cache is None:
            return self._build_solver(a_field)

        solver_cache = self._forward_cache.setdefault("solvers", {})
        cache_key = self._tensor_cache_key(a_field)
        cached_solver = solver_cache.get(cache_key)
        if cached_solver is None:
            cached_solver = self._build_solver(a_field)
            solver_cache[cache_key] = cached_solver
        return cached_solver

    def _apply_A(self, X, a_field):
        """
        X: (B, N, K)
        return AX: (B, N, K)
        """
        B, N, K = X.shape
        n = int(round(N ** 0.5))
        assert n * n == N, f"N={N} 不是平方数"

        apply_A_cache = None
        cache_key = None
        if self._forward_cache is not None:
            apply_A_cache = self._forward_cache.setdefault("apply_A", {})
            cache_key = (self._tensor_cache_key(a_field), self._tensor_cache_key(X))
            cached_AX = apply_A_cache.get(cache_key)
            if cached_AX is not None:
                return cached_AX

        solver = self._get_solver(a_field)

        X_img = X.transpose(1, 2).reshape(B, K, n, n).to(self.dtype)
        AX_img = solver.forward(X_img)
        AX = AX_img.reshape(B, K, N).transpose(1, 2)

        AX = AX.to(X.dtype)
        if apply_A_cache is not None:
            apply_A_cache[cache_key] = AX
        return AX

    def _a_orthonormalize(self, V, AV):
        """
        给定 V 和 AV，返回 A-正交归一后的 Q, AQ, G
        满足 Q^T A Q ≈ I

        V:  (B, N, m)
        AV: (B, N, m)
        """
        B, N, m = V.shape
        I = torch.eye(m, device=V.device, dtype=V.dtype).unsqueeze(0)

        G_raw = _symmetrize_matrix(torch.matmul(V.transpose(1, 2), AV))   # (B, m, m)
        G = G_raw + self.reg * I
        self._last_orth_debug = _build_orth_debug_dict(
            G_raw,
            reg=self.reg,
            backend="cholesky",
        )

        try:
            L = torch.linalg.cholesky(G)
            backend = "cholesky"

            Qt = torch.linalg.solve_triangular(
                L, V.transpose(1, 2), upper=False
            )                                         # (B, m, N)
            AQt = torch.linalg.solve_triangular(
                L, AV.transpose(1, 2), upper=False
            )                                         # (B, m, N)

            Q = Qt.transpose(1, 2)                    # (B, N, m)
            AQ = AQt.transpose(1, 2)                  # (B, N, m)
        except RuntimeError as e:
            try:
                eigvals, eigvecs = torch.linalg.eigh(G_raw)
            except RuntimeError as eig_err:
                self._last_orth_debug = _build_orth_debug_dict(
                    G_raw,
                    reg=self.reg,
                    backend="eigh_fallback_failed",
                )
                raise RuntimeError(
                    "A-正交化失败：Cholesky 与 eigh fallback 均失败。"
                    f"诊断: {_format_orth_debug(self._last_orth_debug)}"
                ) from eig_err

            eigvals_after_reg = eigvals + self.reg
            eigvals_clamped = eigvals.clamp_min(self.reg)
            inv_sqrt = eigvals_clamped.rsqrt()
            G_inv_sqrt = torch.matmul(
                eigvecs * inv_sqrt.unsqueeze(-2),
                eigvecs.transpose(-1, -2),
            )

            Q = torch.matmul(V, G_inv_sqrt)
            AQ = torch.matmul(AV, G_inv_sqrt)
            backend = "eigh_fallback"
            self._last_orth_debug = _build_orth_debug_dict(
                G_raw,
                reg=self.reg,
                backend=backend,
                eigvals_before=eigvals,
                eigvals_after_reg=eigvals_after_reg,
                eigvals_clamped=eigvals_clamped,
            )

            if (not torch.isfinite(Q).all()) or (not torch.isfinite(AQ).all()):
                bad_q = (~torch.isfinite(Q)).sum().item()
                bad_aq = (~torch.isfinite(AQ)).sum().item()
                raise RuntimeError(
                    "A-正交化失败：eigh fallback 后仍产生非有限值。"
                    f" bad(Q)={bad_q}, bad(AQ)={bad_aq}. "
                    f"诊断: {_format_orth_debug(self._last_orth_debug)}"
                ) from e

        if backend == "cholesky":
            self._last_orth_debug = _build_orth_debug_dict(
                G_raw,
                reg=self.reg,
                backend=backend,
            )

        return Q, AQ, G

    def _capture_upper_tolerance(self, total_energy):
        return 1e-8 * torch.maximum(
            total_energy.detach().abs(),
            torch.ones_like(total_energy),
        )

    def _stabilize_captured_energy(self, captured_raw, total_energy, metric_name):
        tol = self._capture_upper_tolerance(total_energy)
        upper = total_energy.clamp_min(0.0) + tol
        captured_nonneg = captured_raw.clamp_min(0.0)
        upper_excess = (captured_nonneg - upper).clamp_min(0.0)
        negative_excess = (-captured_raw).clamp_min(0.0)
        captured = torch.minimum(captured_nonneg, upper)

        self._last_projection_debug = {
            "metric": str(metric_name),
            "num_negative_clamped": float((captured_raw < 0.0).sum().item()),
            "num_upper_clipped": float((captured_nonneg > upper).sum().item()),
            "max_negative_before_clamp": float(negative_excess.max().item()),
            "max_excess_before_clip": float(upper_excess.max().item()),
            "max_tol": float(tol.max().item()),
        }
        return captured

    def _normalize_e_family_shape(self, e_family, a_field=None):
        expected_num_nodes = None
        if a_field is not None:
            expected_num_nodes = int(a_field.shape[-2] * a_field.shape[-1])
        E = normalize_snapshot_family(e_family, expected_num_nodes=expected_num_nodes)
        return E

    def _canonical_loss_type(self, loss_type):
        lt = str(loss_type).lower()
        if lt in {"a", "energy", "a_inner", "a-inner"}:
            return "a"
        if lt in {"l2", "euclidean", "2"}:
            return "l2"
        raise ValueError(f"Unknown loss_type={loss_type!r}, expected 'A' or 'l2'")

    def _canonical_batch_reduction(self, batch_reduction):
        reduction = str(batch_reduction).lower()
        if reduction not in {"mean", "sum_ratio", "blended"}:
            raise ValueError(
                "batch_reduction must be one of {'mean', 'sum_ratio', 'blended'}, "
                f"got {batch_reduction!r}"
            )
        return reduction

    def _validate_batch_reduction_alpha(self, alpha):
        alpha = float(alpha)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must satisfy 0 <= alpha <= 1, got {alpha!r}")
        return alpha

    def _reduce_ratio_over_batch(
        self,
        numerator,
        denominator,
        *,
        eps,
        batch_reduction,
        alpha,
    ):
        if numerator.shape != denominator.shape:
            raise ValueError(
                "numerator and denominator must have the same shape, "
                f"got {tuple(numerator.shape)} vs {tuple(denominator.shape)}"
            )

        numerator_flat = numerator.reshape(numerator.shape[0], -1)
        denominator_flat = denominator.reshape(denominator.shape[0], -1)

        sample_numerator = numerator_flat.sum(dim=-1)
        sample_denominator_raw = denominator_flat.sum(dim=-1)
        sample_denominator = sample_denominator_raw.clamp_min(float(eps))
        per_sample_ratio = sample_numerator / sample_denominator
        ratio_per_sample_mean = per_sample_ratio.mean()

        global_numerator = sample_numerator.sum()
        global_denominator = sample_denominator_raw.sum().clamp_min(float(eps))
        ratio_global_sum = global_numerator / global_denominator

        if batch_reduction == "mean":
            ratio = ratio_per_sample_mean
        elif batch_reduction == "sum_ratio":
            ratio = ratio_global_sum
        else:
            blend_alpha = numerator.new_tensor(float(alpha))
            ratio = (
                (1.0 - blend_alpha) * ratio_per_sample_mean
                + blend_alpha * ratio_global_sum
            )

        return {
            "ratio": ratio,
            "per_sample_ratio": per_sample_ratio,
            "per_sample_mean": ratio_per_sample_mean,
            "global_sum_ratio": ratio_global_sum,
        }

    def _resolve_orth_reg_metric(self, orth_reg_metric, default=None):
        if orth_reg_metric is None:
            orth_reg_metric = default
        if orth_reg_metric is None:
            return None
        return self._canonical_loss_type(orth_reg_metric)

    def _euclidean_orthonormalize(self, V):
        """
        对 V 做 Euclidean 正交化，返回 Q，使 Q^T Q = I
        V: (B, N, m)
        """
        Q, _ = torch.linalg.qr(V, mode="reduced")
        return Q

    def _metric_gram(self, V, a_field, metric, MV=None):
        metric = self._canonical_loss_type(metric)
        if metric == "a":
            if MV is None:
                MV = self._apply_A(V, a_field)
            return torch.matmul(V.transpose(1, 2), MV)
        return torch.matmul(V.transpose(1, 2), V)

    def _normalized_orth_regularizer(self, V, a_field, metric, eps, MV=None):
        G = self._metric_gram(V, a_field, metric, MV=MV)  # (B, m, m)
        d = torch.sqrt(torch.diagonal(G, dim1=-2, dim2=-1).clamp_min(eps))
        G_norm = G / (d.unsqueeze(-1) * d.unsqueeze(-2)).clamp_min(eps)

        m = G.shape[-1]
        I = torch.eye(m, device=G.device, dtype=G.dtype).unsqueeze(0)
        return ((G_norm - I) ** 2).mean()

    def build_metric_orthonormal_basis(self, V, a_field, loss_type="A", AV=None):
        """
        根据 loss_type 构造对应度量下的正交基:
          - loss_type='A'  : Q^T A Q = I
          - loss_type='l2' : Q^T Q = I
        """
        lt = self._canonical_loss_type(loss_type)
        a_field = a_field.to(self.dtype)

        if lt == "a":
            if AV is None:
                AV = self._apply_A(V, a_field)
            Q, AQ, G = self._a_orthonormalize(V, AV)
            return Q, AQ, G

        Q = self._euclidean_orthonormalize(V)
        return Q, None, None

    def metric_energy(self, X, a_field, loss_type="A", MX=None):
        """
        返回对应度量下每个 snapshot 的能量:
          - A:  <x,x>_A  = x^T A x
          - l2: <x,x>_2  = x^T x
        X: (B, N, q)
        """
        lt = self._canonical_loss_type(loss_type)
        X = self._normalize_e_family_shape(X, a_field=a_field)
        a_field = a_field.to(self.dtype)

        if lt == "a":
            if MX is None:
                MX = self._apply_A(X, a_field)
            energy = (X * MX).sum(dim=1)   # (B, q)
            return energy, MX

        energy = (X * X).sum(dim=1)        # (B, q)
        return energy, X

    def project_onto_basis(
        self,
        E,
        a_field,
        V=None,
        Q=None,
        AQ=None,
        ME=None,
        AV=None,
        loss_type="A",
    ):
        """
        对 snapshot family E 做对应度量下的正交投影:

          - A-loss:
              Pi(E) = Q (Q^T A E),   with Q^T A Q = I
              captured = <Pi(E), Pi(E)>_A

          - l2-loss:
              Pi(E) = Q (Q^T E),     with Q^T Q = I
              captured = ||Pi(E)||_2^2
        """
        lt = self._canonical_loss_type(loss_type)
        E = self._normalize_e_family_shape(E, a_field=a_field)
        a_field = a_field.to(self.dtype)
        self._last_projection_debug = None

        if Q is None:
            if V is None:
                raise ValueError("Either raw basis V or orthonormal basis Q must be provided")
            Q, AQ, _ = self.build_metric_orthonormal_basis(V, a_field, loss_type=lt, AV=AV)

        if lt == "a":
            if ME is None:
                ME = self._apply_A(E, a_field)
            coeff = torch.matmul(Q.transpose(1, 2), ME)   # (B, m, q)
            projection = torch.matmul(Q, coeff)           # (B, N, q)
            if AQ is None:
                Mprojection = self._apply_A(projection, a_field)
            else:
                Mprojection = torch.matmul(AQ, coeff)
            total_energy = (E * ME).sum(dim=1)
            captured_raw = (projection * Mprojection).sum(dim=1)
            captured = self._stabilize_captured_energy(
                captured_raw,
                total_energy,
                metric_name="a",
            )
            return projection, captured, coeff, Q, AQ, ME

        # l2
        coeff = torch.matmul(Q.transpose(1, 2), E)        # (B, m, q)
        projection = torch.matmul(Q, coeff)               # (B, N, q)
        total_energy = (E * E).sum(dim=1)
        captured_raw = (projection * projection).sum(dim=1)
        captured = self._stabilize_captured_energy(
            captured_raw,
            total_energy,
            metric_name="l2",
        )
        return projection, captured, coeff, Q, None, E
