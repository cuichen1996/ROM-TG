"""
Snapshot-based subspace training loss (capture / SVD / NLSS variants).
"""

import warnings

import torch
import torch.nn as nn

from learned_tg.model.losses.base import normalize_snapshot_family, _canonical_snapshot_whiten_metric, _compute_metric_whitened_snapshot_family, _symmetrize_matrix
from learned_tg.model.losses.primitives import require_finite
from learned_tg.model.basis_utils.orthonormalize import a_orthonormalize_from_AV as _a_orth_impl, euclidean_orthonormalize_qr as _eucl_orth_impl
from learned_tg.model.basis_utils.reference_basis import _broadcast_reference_basis
from learned_tg.model.operators.ccfvm_asm import CCFVM_ASM


class SubspaceLoss(nn.Module):
    """
    Snapshot-based subspace training loss.

    Supports three loss variants:
      - `capture`: minimize the fraction of snapshot energy not captured by the
        predicted subspace; supports l2 and A metrics.
      - `svd`: align the predicted subspace with the leading SVD modes of the
        snapshot family (Euclidean / l2 metric only).
      - `nlss`: non-linear subspace streaming loss, prefix-capture variant
        (Euclidean / l2 metric only).
    """
    def __init__(
        self,
        loss_type="capture",
        metric="l2",
        lambda_orth=0.0,
        lambda_decorrelate=0.0,
        use_prefix_truncation=True,
        effective_rank_mode="truncate",
        capture_whiten=False,
        capture_whiten_rank=None,
        capture_whiten_metric=None,
        svd_teacher_rank=None,
        svd_allow_embedding=True,
        svd_loss_variant="projection",
        log_extra_metrics=True,
        k_1d=10,
        target_overlap=0.3,
        orth_reg=None,
        eps=1e-8,
        dtype=torch.float64,
    ):
        super().__init__()
        self.loss_type = self._canonical_loss_type(loss_type)
        self.metric = self._canonical_metric(metric)
        self.lambda_orth = float(lambda_orth)
        self.lambda_decorrelate = float(lambda_decorrelate)
        self.use_prefix_truncation = bool(use_prefix_truncation)
        self.effective_rank_mode = self._canonical_effective_rank_mode(
            effective_rank_mode
        )
        self.capture_whiten = bool(capture_whiten)
        self.capture_whiten_rank = self._canonical_capture_whiten_rank(
            capture_whiten_rank
        )
        self.capture_whiten_metric = (
            None if capture_whiten_metric is None
            else _canonical_snapshot_whiten_metric(capture_whiten_metric)
        )
        self.svd_teacher_rank = self._canonical_svd_teacher_rank(svd_teacher_rank)
        self.svd_allow_embedding = bool(svd_allow_embedding)
        self.svd_loss_variant = self._canonical_svd_loss_variant(svd_loss_variant)
        self.log_extra_metrics = bool(log_extra_metrics)
        self.k_1d = int(k_1d)
        self.target_overlap = float(target_overlap)
        self.eps = float(eps)
        self.orth_reg = float(self.eps if orth_reg is None else orth_reg)
        self.dtype = dtype
        self._warned_effective_rank = False
        self._warning_flags = set()

        if self.loss_type in {"svd", "nlss"} and self.metric != "l2":
            raise ValueError("svd/nlss loss types only support l2 metric; use capture for A-metric.")

    def _canonical_loss_type(self, loss_type):
        lt = str(loss_type).lower()
        if lt not in {"capture", "svd", "nlss"}:
            raise ValueError(
                "loss_type must be one of {'capture', 'svd', 'nlss'}, "
                f"got {loss_type!r}"
            )
        return lt

    def _canonical_metric(self, metric):
        key = str(metric).lower()
        if key in {"l2", "euclidean", "2"}:
            return "l2"
        if key in {"a", "energy", "a_inner", "a-inner"}:
            return "a"
        raise ValueError("metric must be 'l2' or 'a'")

    def _canonical_effective_rank_mode(self, mode):
        key = str(mode).lower()
        if key not in {"truncate", "error"}:
            raise ValueError(
                "effective_rank_mode must be one of {'truncate', 'error'}, "
                f"got {mode!r}"
            )
        return key

    def _canonical_capture_whiten_rank(self, whiten_rank):
        if whiten_rank is None:
            return None
        whiten_rank = int(whiten_rank)
        if whiten_rank <= 0:
            raise ValueError(
                "capture_whiten_rank must be a positive integer or None, "
                f"got {whiten_rank!r}"
            )
        return whiten_rank

    def _canonical_svd_teacher_rank(self, teacher_rank):
        if teacher_rank is None:
            return None
        teacher_rank = int(teacher_rank)
        if teacher_rank <= 0:
            raise ValueError(
                f"svd_teacher_rank must be a positive integer or None, got {teacher_rank!r}"
            )
        return teacher_rank

    def _canonical_svd_loss_variant(self, variant):
        key = str(variant).lower()
        if key not in {"projection", "projector_fro"}:
            raise ValueError(
                "svd_loss_variant must be one of {'projection', 'projector_fro'}, "
                f"got {variant!r}"
            )
        return key

    def _warn_once(self, flag, message):
        if flag in self._warning_flags:
            return
        warnings.warn(message, stacklevel=2)
        self._warning_flags.add(flag)

    def _canonicalize_snapshot_family(self, e_family, expected_num_nodes=None):
        E = normalize_snapshot_family(
            e_family,
            expected_num_nodes=expected_num_nodes,
        )
        return torch.nan_to_num(
            E,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).to(self.dtype)

    def _warn_effective_rank_if_needed(self, m_cfg, max_rank):
        if self._warned_effective_rank:
            return
        self._warn_once(
            "effective_rank_truncated",
            "Effective rank truncated because the network output dimension "
            f"m_cfg={m_cfg} exceeds min(q, N)={max_rank}. Only the first {max_rank} "
            "columns participate in the subspace loss.",
        )
        self._warned_effective_rank = True

    def _resolve_effective_rank(self, num_columns, num_nodes, num_snapshots):
        m_cfg = int(num_columns)
        max_rank = min(int(num_nodes), int(num_snapshots))
        if max_rank <= 0:
            raise ValueError(
                f"Subspace loss requires positive min(q, N), got N={num_nodes}, q={num_snapshots}"
            )
        if m_cfg > max_rank:
            if self.effective_rank_mode == "error":
                raise ValueError(
                    "Effective rank exceeded the available snapshot rank: "
                    f"m_cfg={m_cfg}, min(q, N)={max_rank}. "
                    "Set effective_rank_mode='truncate' to enable truncation."
                )
            self._warn_effective_rank_if_needed(m_cfg, max_rank)
        return min(m_cfg, max_rank), m_cfg, max_rank

    def _truncate_rank_if_needed(self, basis, E, basis_name="basis"):
        basis = torch.nan_to_num(
            basis,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).to(self.dtype)
        if basis.dim() != 3:
            raise ValueError(
                f"Expected {basis_name} to have shape (B, N, m), got {tuple(basis.shape)}"
            )

        m_eff, m_cfg, max_rank = self._resolve_effective_rank(
            basis.shape[-1],
            E.shape[1],
            E.shape[-1],
        )
        if self.use_prefix_truncation:
            basis_eff = basis[:, :, :m_eff].contiguous()
        else:
            basis_eff = basis

        return basis_eff, m_eff, {
            "snapshot/m_cfg": float(m_cfg),
            "snapshot/max_effective_rank": float(max_rank),
            "snapshot/was_truncated": float(m_cfg > m_eff),
        }

    def _euclidean_q(self, V_eff):
        Q = _eucl_orth_impl(V_eff)
        return require_finite(Q, "Q")

    def _build_metric_solver(self, a_field):
        if a_field is None:
            raise ValueError("A-metric loss requires a_field")
        a_field = torch.nan_to_num(
            a_field,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).to(self.dtype)
        return CCFVM_ASM(
            a=a_field,
            k_1d=self.k_1d,
            target_overlap=self.target_overlap,
            build_local_solver=False,
        ).to(a_field.device)

    def _metric_orthonormalize(self, V, a_field=None, op_solver=None):
        if V.shape[-1] <= 0:
            raise ValueError("Predicted basis rank must be positive")

        if self.metric == "l2":
            return self._euclidean_q(V), None, op_solver

        if op_solver is None:
            op_solver = self._build_metric_solver(a_field)
        AV = op_solver.apply_A(V)
        Q, AQ, _ = _a_orth_impl(V, AV, reg=self.orth_reg)
        return (
            require_finite(Q, "Q"),
            require_finite(AQ, "AQ"),
            op_solver,
        )

    def _normalized_metric_orth_regularizer(self, V, a_field=None, op_solver=None):
        if self.lambda_orth <= 0.0:
            return V.new_tensor(0.0)

        if self.metric == "a":
            if op_solver is None:
                op_solver = self._build_metric_solver(a_field)
            AV = op_solver.apply_A(V)
            gram = _symmetrize_matrix(torch.matmul(V.transpose(1, 2), AV))
        else:
            gram = torch.matmul(V.transpose(1, 2), V)

        diag = torch.diagonal(gram, dim1=-2, dim2=-1).clamp_min(self.eps)
        scale = torch.sqrt(diag)
        gram_norm = gram / (scale.unsqueeze(-1) * scale.unsqueeze(-2)).clamp_min(self.eps)
        basis_dim = gram.shape[-1]
        eye = torch.eye(basis_dim, device=V.device, dtype=V.dtype).unsqueeze(0)
        return ((gram_norm - eye) ** 2).mean()

    def _resolve_capture_whiten_rank(self, E):
        available_rank = min(int(E.shape[1]), int(E.shape[2]))
        if available_rank <= 0:
            raise ValueError(
                f"Capture whitening requires positive min(N, q), got {tuple(E.shape)}"
            )

        if self.capture_whiten_rank is None:
            return available_rank

        whiten_rank = min(int(self.capture_whiten_rank), available_rank)
        if whiten_rank < int(self.capture_whiten_rank):
            self._warn_once(
                "capture_whiten_rank_truncated",
                "Capture whitening rank exceeds the available snapshot rank; "
                f"truncating k from {int(self.capture_whiten_rank)} to {whiten_rank}.",
            )
        return whiten_rank

    def _resolve_capture_whiten_metric(self, metric=None):
        if metric is None:
            metric = self.capture_whiten_metric
        if metric is None:
            metric = self.metric
        return _canonical_snapshot_whiten_metric(metric)

    def _whiten_error_snapshots(self, E, metric=None, op_solver=None):
        if not self.capture_whiten:
            return E, {
                "active": 0.0,
                "rank": 0.0,
                "metric": "l2",
            }

        whiten_rank = self._resolve_capture_whiten_rank(E)
        whiten_metric = self._resolve_capture_whiten_metric(metric)
        try:
            E_white = _compute_metric_whitened_snapshot_family(
                E,
                rank=whiten_rank,
                metric=whiten_metric,
                op_solver=op_solver,
                reg=self.orth_reg,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Capture whitening failed while constructing the metric-aware "
                f"whitened e_family. E.shape={tuple(E.shape)}, whiten_rank={whiten_rank}, "
                f"metric={whiten_metric!r}"
            ) from exc

        E_white = require_finite(E_white, "E_white")
        return E_white, {
            "active": 1.0,
            "rank": float(whiten_rank),
            "metric": whiten_metric,
        }

    def _prepare_capture_family(
        self,
        E,
        e_family_white=None,
        e_family_white_metric=None,
        op_solver=None,
    ):
        if not self.capture_whiten:
            return E, {
                "active": 0.0,
                "rank": 0.0,
                "precomputed": 0.0,
                "metric": self._resolve_capture_whiten_metric("l2"),
            }

        expected_whiten_metric = self._resolve_capture_whiten_metric()
        if e_family_white is None:
            E_white, whiten_stats = self._whiten_error_snapshots(
                E,
                metric=expected_whiten_metric,
                op_solver=op_solver,
            )
            whiten_stats["precomputed"] = 0.0
            return E_white, whiten_stats

        if e_family_white_metric is not None:
            provided_metric = _canonical_snapshot_whiten_metric(e_family_white_metric)
            if provided_metric != expected_whiten_metric:
                self._warn_once(
                    "capture_whiten_metric_mismatch",
                    "Cached whitened snapshot family metric does not match the requested "
                    f"capture whitening metric ({provided_metric!r} vs {expected_whiten_metric!r}); "
                    "falling back to online reconstruction.",
                )
                E_white, whiten_stats = self._whiten_error_snapshots(
                    E,
                    metric=expected_whiten_metric,
                    op_solver=op_solver,
                )
                whiten_stats["precomputed"] = 0.0
                return E_white, whiten_stats

        E_white = self._canonicalize_snapshot_family(
            e_family_white,
            expected_num_nodes=int(E.shape[1]),
        )
        return E_white, {
            "active": 1.0,
            "rank": float(self._resolve_capture_whiten_rank(E)),
            "precomputed": 1.0,
            "metric": expected_whiten_metric,
        }

    def _metric_total_energy(self, E, metric="l2", op_solver=None):
        metric = self._canonical_metric(metric)
        if metric == "l2":
            return (E * E).sum(dim=(1, 2)).clamp_min(self.eps), E

        if op_solver is None:
            raise ValueError("A-metric energy evaluation requires op_solver")
        AE = op_solver.apply_A(E)
        total_energy = (E * AE).sum(dim=(1, 2)).clamp_min(self.eps)
        return total_energy, AE

    def _capture_loss_from_QE(self, Q, E, metric="l2", op_solver=None):
        metric = self._canonical_metric(metric)

        if metric == "a":
            total_energy, AE = self._metric_total_energy(E, metric="a", op_solver=op_solver)
            coeff = torch.matmul(Q.transpose(1, 2), AE)
        else:
            total_energy, _ = self._metric_total_energy(E, metric="l2")
            coeff = torch.matmul(Q.transpose(1, 2), E)

        captured_energy = (coeff * coeff).sum(dim=(1, 2))
        capture_ratio = captured_energy / total_energy
        loss_per_sample = 1.0 - capture_ratio
        return {
            "loss": loss_per_sample.mean(),
            "loss_per_sample": loss_per_sample,
            "capture_ratio": capture_ratio,
            "captured_energy": captured_energy,
            "total_energy": total_energy,
            "coeff": coeff,
        }

    def _available_svd_basis_rank(self, svd_basis, E):
        if svd_basis is None:
            return min(int(E.shape[1]), int(E.shape[2]))

        if not torch.is_tensor(svd_basis):
            svd_basis = torch.as_tensor(svd_basis, device=E.device, dtype=self.dtype)
        else:
            svd_basis = svd_basis.to(device=E.device, dtype=self.dtype)

        if svd_basis.dim() == 2:
            if svd_basis.shape[0] != E.shape[1]:
                raise ValueError(
                    "svd_basis shape mismatch: expected first dimension to equal num_nodes "
                    f"{E.shape[1]}, got {tuple(svd_basis.shape)}"
                )
            return int(svd_basis.shape[1])

        if svd_basis.dim() == 3:
            if svd_basis.shape[1] != E.shape[1]:
                raise ValueError(
                    "svd_basis shape mismatch: expected num_nodes dimension to equal "
                    f"{E.shape[1]}, got {tuple(svd_basis.shape)}"
                )
            if svd_basis.shape[0] not in {1, E.shape[0]}:
                raise ValueError(
                    "svd_basis batch mismatch: expected batch dimension 1 or "
                    f"{E.shape[0]}, got {tuple(svd_basis.shape)}"
                )
            return int(svd_basis.shape[2])

        raise ValueError(
            "svd_basis must have shape (N, k0) or (B, N, k0), "
            f"got {tuple(svd_basis.shape)}"
        )

    def _resolve_svd_teacher_rank(self, available_teacher_rank, num_nodes, pred_rank_eff):
        if self.svd_teacher_rank is None:
            return min(int(available_teacher_rank), int(num_nodes), int(pred_rank_eff))
        return min(
            int(self.svd_teacher_rank),
            int(available_teacher_rank),
            int(num_nodes),
        )

    def _svd_teacher_from_E(self, E, pred_rank_eff, svd_basis=None):
        available_teacher_rank = self._available_svd_basis_rank(svd_basis, E)
        teacher_rank_raw = self._resolve_svd_teacher_rank(
            available_teacher_rank=available_teacher_rank,
            num_nodes=E.shape[1],
            pred_rank_eff=pred_rank_eff,
        )
        if teacher_rank_raw <= 0:
            raise ValueError(
                "Resolved non-positive teacher rank for SVD supervision: "
                f"available_teacher_rank={available_teacher_rank}, pred_rank_eff={pred_rank_eff}, "
                f"svd_teacher_rank={self.svd_teacher_rank}"
            )

        if svd_basis is not None:
            teacher = _broadcast_reference_basis(
                svd_basis,
                batch_size=E.shape[0],
                num_nodes=E.shape[1],
                basis_dim=teacher_rank_raw,
                device=E.device,
                dtype=self.dtype,
            )
            teacher = self._euclidean_q(teacher[:, :, :teacher_rank_raw].contiguous())
            return teacher.detach(), int(teacher_rank_raw)

        with torch.no_grad():
            try:
                U, _, _ = torch.linalg.svd(E, full_matrices=False)
            except RuntimeError as exc:
                raise RuntimeError(
                    "SVD teacher construction failed for e_family. "
                    f"E.shape={tuple(E.shape)}"
                ) from exc
            teacher = require_finite(U[:, :, :teacher_rank_raw].contiguous(), "teacher")
        return teacher.detach(), int(teacher_rank_raw)

    def _truncate_pred_subspace_for_svd(self, Q_pred, teacher_rank):
        pred_rank_raw = int(Q_pred.shape[-1])
        teacher_rank = int(teacher_rank)
        if teacher_rank <= 0:
            raise ValueError(
                f"teacher_rank must be positive in _truncate_pred_subspace_for_svd, got {teacher_rank}"
            )
        if pred_rank_raw <= 0:
            raise ValueError("Q_pred must have a positive number of columns for SVD supervision.")
        if pred_rank_raw <= teacher_rank:
            return Q_pred[:, :, :pred_rank_raw].contiguous(), pred_rank_raw

        column_priority = torch.linspace(
            float(pred_rank_raw),
            1.0,
            pred_rank_raw,
            device=Q_pred.device,
            dtype=Q_pred.dtype,
        )
        column_priority = column_priority / column_priority.amax().clamp_min(self.eps)
        Q_weighted = Q_pred * column_priority.view(1, 1, -1)
        try:
            U, _, _ = torch.linalg.svd(Q_weighted, full_matrices=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to truncate the predicted SVD subspace with sample-wise SVD. "
                f"Q_pred.shape={tuple(Q_pred.shape)}, teacher_rank={teacher_rank}"
            ) from exc

        return require_finite(U[:, :, :teacher_rank].contiguous(), "U_truncated"), teacher_rank

    def _svd_supervised_loss(self, Q_pred, E, svd_basis=None):
        pred_rank_raw = int(Q_pred.shape[-1])
        if pred_rank_raw <= 0:
            raise ValueError("Predicted SVD supervision rank must be positive")

        Q_teacher, _ = self._svd_teacher_from_E(
            E,
            pred_rank_eff=pred_rank_raw,
            svd_basis=svd_basis,
        )
        teacher_rank = int(Q_teacher.shape[-1])
        if teacher_rank <= 0:
            raise ValueError("Teacher rank must be positive")

        if teacher_rank > pred_rank_raw:
            warn_flag = (
                "svd_teacher_rank_exceeds_pred_embedding"
                if self.svd_allow_embedding else
                "svd_teacher_rank_exceeds_pred_equal"
            )
            warn_message = (
                "teacher rank exceeds predicted rank; truncating teacher to the available "
                "predicted rank before SVD supervision."
                if self.svd_allow_embedding else
                "svd_allow_embedding=False; truncating teacher to the available "
                "predicted rank for equal-rank SVD supervision."
            )
            self._warn_once(warn_flag, warn_message)
            Q_teacher = Q_teacher[:, :, :pred_rank_raw].contiguous()
            teacher_rank = int(Q_teacher.shape[-1])
        elif (not self.svd_allow_embedding) and teacher_rank != pred_rank_raw:
            self._warn_once(
                "svd_equal_rank_truncate",
                "svd_allow_embedding=False; truncating predicted SVD supervision to the "
                "teacher rank before computing the loss.",
            )

        Q_pred_loss, pred_rank_used = self._truncate_pred_subspace_for_svd(
            Q_pred,
            teacher_rank=teacher_rank,
        )

        if teacher_rank <= 0:
            raise ValueError("Resolved non-positive final teacher rank for svd supervision")
        if pred_rank_used <= 0:
            raise ValueError("Resolved non-positive final predicted rank for svd supervision")

        overlap = torch.matmul(Q_teacher.transpose(1, 2), Q_pred_loss)
        overlap_sq = (overlap * overlap).sum(dim=(1, 2))
        svd_loss_variant = self.svd_loss_variant
        if svd_loss_variant == "projector_fro" and pred_rank_used != teacher_rank:
            self._warn_once(
                "svd_projector_fro_fallback",
                "svd_loss_variant='projector_fro' requires equal teacher/pred rank; falling back to 'projection'.",
            )
            svd_loss_variant = "projection"

        if svd_loss_variant == "projector_fro":
            projector_pred = torch.matmul(Q_pred_loss, Q_pred_loss.transpose(1, 2))
            projector_teacher = torch.matmul(Q_teacher, Q_teacher.transpose(1, 2))
            diff = projector_pred - projector_teacher
            loss_per_sample = 0.5 * (diff * diff).sum(dim=(1, 2))
        else:
            loss_per_sample = float(teacher_rank) - overlap_sq

        total_energy, _ = self._metric_total_energy(E, metric="l2")
        model_coeff = torch.matmul(Q_pred_loss.transpose(1, 2), E)
        teacher_coeff = torch.matmul(Q_teacher.transpose(1, 2), E)
        model_capture = (model_coeff * model_coeff).sum(dim=(1, 2)) / total_energy
        teacher_capture = (teacher_coeff * teacher_coeff).sum(dim=(1, 2)) / total_energy
        principal_overlap = overlap_sq / max(float(teacher_rank), 1.0)
        teacher_in_pred_residual = float(teacher_rank) - overlap_sq
        svd_extra_dims = max(pred_rank_raw - pred_rank_used, 0)
        embedding_active = float(self.svd_allow_embedding and svd_extra_dims > 0)

        return {
            "loss": loss_per_sample.mean(),
            "loss_per_sample": loss_per_sample,
            "principal_overlap": principal_overlap,
            "teacher_capture_ratio": teacher_capture,
            "model_capture_ratio": model_capture,
            "teacher_in_pred_residual": teacher_in_pred_residual,
            "teacher_rank": teacher_rank,
            "pred_rank": pred_rank_used,
            "pred_rank_raw": pred_rank_raw,
            "pred_rank_used": pred_rank_used,
            "embedding_active": embedding_active,
            "extra_dims": float(svd_extra_dims),
            "svd_loss_variant": svd_loss_variant,
        }

    def _nlss_loss(self, Q, E, m_eff):
        coeff = torch.matmul(Q.transpose(1, 2), E)
        total_energy, _ = self._metric_total_energy(E, metric="l2")
        coeff_energy = (coeff * coeff).sum(dim=-1)
        prefix_captured = torch.cumsum(coeff_energy, dim=-1)
        prefix_capture_ratio = prefix_captured / total_energy.unsqueeze(-1)
        prefix_loss = 1.0 - prefix_capture_ratio
        loss_per_sample = prefix_loss.mean(dim=-1)
        return {
            "loss": loss_per_sample.mean(),
            "loss_per_sample": loss_per_sample,
            "prefix_capture_ratio": prefix_capture_ratio,
            "final_capture_ratio": prefix_capture_ratio[:, -1],
        }

    def forward(
        self,
        V_raw=None,
        Q=None,
        e_family=None,
        e_family_white=None,
        e_family_white_metric=None,
        a_field=None,
        svd_basis=None,
        op_solver=None,
        return_stats=False,
    ):
        if e_family is None:
            raise ValueError("e_family must be provided for subspace training")
        if V_raw is None and Q is None:
            raise ValueError("Either V_raw or Q must be provided for subspace training")

        expected_num_nodes = None
        if V_raw is not None:
            expected_num_nodes = int(V_raw.shape[1])
        elif Q is not None:
            expected_num_nodes = int(Q.shape[1])

        E = self._canonicalize_snapshot_family(
            e_family,
            expected_num_nodes=expected_num_nodes,
        )

        if V_raw is not None:
            V_candidate, m_eff, rank_stats = self._truncate_rank_if_needed(
                V_raw,
                E,
                basis_name="V_raw",
            )
        else:
            V_candidate, m_eff, rank_stats = self._truncate_rank_if_needed(
                Q,
                E,
                basis_name="Q",
            )

        Q_pred, _, op_solver = self._metric_orthonormalize(
            V_candidate,
            a_field=a_field,
            op_solver=op_solver,
        )
        if (
            self.loss_type == "capture"
            and self.capture_whiten
            and self._resolve_capture_whiten_metric() == "a"
            and op_solver is None
        ):
            op_solver = self._build_metric_solver(a_field)
        if Q_pred.shape[-1] > m_eff:
            Q_pred = Q_pred[:, :, :m_eff].contiguous()
        if Q_pred.shape[-1] <= 0:
            raise ValueError("Resolved non-positive predicted rank after orthonormalization")
        if V_candidate.shape[-1] > m_eff:
            V_candidate = V_candidate[:, :, :m_eff].contiguous()

        # --- V column effective-rank diagnostic (no gradient, always computed) ---
        with torch.no_grad():
            G_l2_diag = torch.matmul(
                V_candidate.transpose(1, 2).to(torch.float32),
                V_candidate.to(torch.float32),
            )  # (B, m_eff, m_eff)
            eigvals_diag = torch.linalg.eigvalsh(G_l2_diag).clamp_min(1e-30)  # (B, m_eff)
            eff_rank_per_sample = (
                eigvals_diag.sum(dim=-1) ** 2
            ) / (eigvals_diag ** 2).sum(dim=-1)
            v_col_eff_rank = eff_rank_per_sample.mean().to(V_candidate.dtype)

        # --- Decorrelate regularization on raw V columns ---
        if self.lambda_decorrelate > 0.0:
            V_f32 = V_candidate.to(torch.float32)
            G_l2 = torch.matmul(V_f32.transpose(1, 2), V_f32)  # (B, m_eff, m_eff)
            diag = torch.diagonal(G_l2, dim1=-2, dim2=-1).clamp_min(1e-12)  # (B, m_eff)
            norm_outer = torch.sqrt(diag.unsqueeze(-1) * diag.unsqueeze(-2))  # (B, m_eff, m_eff)
            G_norm = G_l2 / norm_outer.clamp_min(1e-12)
            m_eff_dim = G_l2.shape[-1]
            eye = torch.eye(m_eff_dim, device=G_norm.device, dtype=G_norm.dtype).unsqueeze(0)
            off_diag = (G_norm * (1.0 - eye)).abs().sum(dim=(-1, -2)) / max(m_eff_dim * (m_eff_dim - 1), 1)
            decorrelate_term = off_diag.mean().to(V_candidate.dtype)
        else:
            decorrelate_term = V_candidate.new_tensor(0.0)

        loss_orth = self._normalized_metric_orth_regularizer(
            V_candidate,
            a_field=a_field,
            op_solver=op_solver,
        )

        if self.loss_type == "capture":
            E_capture, whiten_stats = self._prepare_capture_family(
                E,
                e_family_white=e_family_white,
                e_family_white_metric=e_family_white_metric,
                op_solver=op_solver,
            )
            main = self._capture_loss_from_QE(
                Q_pred,
                E_capture,
                metric=self.metric,
                op_solver=op_solver,
            )
            loss_main = main["loss"]
            extra_stats = {
                "snapshot/capture_ratio": main["capture_ratio"].mean().detach(),
                "snapshot/residual_ratio": main["loss_per_sample"].mean().detach(),
                "snapshot/capture_whiten_active": loss_main.new_tensor(
                    float(whiten_stats["active"])
                ),
                "snapshot/capture_whiten_rank": loss_main.new_tensor(
                    float(whiten_stats["rank"])
                ),
                "snapshot/capture_whiten_precomputed": loss_main.new_tensor(
                    float(whiten_stats["precomputed"])
                ),
            }
        elif self.loss_type == "svd":
            main = self._svd_supervised_loss(
                Q_pred,
                E,
                svd_basis=svd_basis,
            )
            loss_main = main["loss"]
            extra_stats = {
                "snapshot/principal_overlap": main["principal_overlap"].mean().detach(),
                "snapshot/teacher_capture_ratio": main["teacher_capture_ratio"].mean().detach(),
                "snapshot/model_capture_ratio": main["model_capture_ratio"].mean().detach(),
                "snapshot/teacher_in_pred_residual": main["teacher_in_pred_residual"].mean().detach(),
                "snapshot/svd_teacher_rank": loss_main.new_tensor(float(main["teacher_rank"])),
                "snapshot/svd_pred_rank": loss_main.new_tensor(float(main["pred_rank"])),
                "snapshot/svd_pred_rank_raw": loss_main.new_tensor(float(main["pred_rank_raw"])),
                "snapshot/svd_pred_rank_used": loss_main.new_tensor(float(main["pred_rank_used"])),
                "snapshot/svd_embedding_active": loss_main.new_tensor(float(main["embedding_active"])),
                "snapshot/svd_extra_dims": loss_main.new_tensor(float(main["extra_dims"])),
            }
        else:
            main = self._nlss_loss(Q_pred, E, m_eff=m_eff)
            loss_main = main["loss"]
            extra_stats = {
                "snapshot/nlss_loss": main["loss_per_sample"].mean().detach(),
                "snapshot/nlss_final_capture_ratio": main["final_capture_ratio"].mean().detach(),
            }

        loss_total = (
            loss_main
            + self.lambda_orth * loss_orth
            + self.lambda_decorrelate * decorrelate_term
        )
        if not return_stats:
            return loss_total

        stats = {
            "snapshot/loss": loss_total.detach(),
            "snapshot/loss_main": loss_main.detach(),
            "snapshot/loss_orth": loss_orth.detach(),
            "snapshot/loss_decorrelate": decorrelate_term.detach(),
            "snapshot/lambda_decorrelate": loss_total.new_tensor(self.lambda_decorrelate),
            "snapshot/v_col_eff_rank": v_col_eff_rank.detach(),
            "snapshot/m_eff": loss_total.new_tensor(float(m_eff)),
        }
        stats.update({
            key: loss_total.new_tensor(value)
            for key, value in rank_stats.items()
        })

        if self.log_extra_metrics:
            stats.update(extra_stats)
            if self.loss_type == "nlss":
                prefix_capture_ratio = main["prefix_capture_ratio"].detach()
                for prefix_idx in range(prefix_capture_ratio.shape[-1]):
                    stats[
                        f"snapshot/prefix_capture_ratio_{prefix_idx + 1}"
                    ] = prefix_capture_ratio[:, prefix_idx].mean()

        return loss_total, stats

