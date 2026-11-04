"""
Unified energy-projection loss (A-inner-product and Euclidean variants).
"""

import torch

from learned_tg.model.losses.base import AProjectionLossBase


class EnergyProjectionLoss(AProjectionLossBase):
    """
    Unified projection loss:
      - loss_type='A'  : A-inner-product projection loss
      - loss_type='l2' : Euclidean projection loss
    """
    def __init__(
        self,
        k_1d=10,
        target_overlap=0.3,
        epsilon=1e-6,
        orth_weight=0.01,
        loss_type="A",
        orth_reg_metric=None,
        batch_reduction="mean",
        alpha=0.0,
        dtype=torch.float64,
    ):
        super().__init__(
            k_1d=k_1d,
            target_overlap=target_overlap,
            reg=epsilon,
            dtype=dtype,
        )
        self.eps = float(epsilon)
        self.orth_weight = float(orth_weight)
        self.loss_type = self._canonical_loss_type(loss_type)
        self.orth_reg_metric = self._resolve_orth_reg_metric(
            orth_reg_metric,
            default=self.loss_type,
        )
        self.batch_reduction = self._canonical_batch_reduction(batch_reduction)
        self.alpha = self._validate_batch_reduction_alpha(alpha)

    def _orth_regularizer(self, V, a_field, MV=None):
        if self.orth_weight <= 0.0:
            return V.new_tensor(0.0)
        return self._normalized_orth_regularizer(
            V,
            a_field,
            metric=self.orth_reg_metric,
            eps=self.eps,
            MV=MV,
        )

    def _projection_capture_terms(
        self,
        E,
        a_field,
        V=None,
        Q=None,
        AQ=None,
        ME=None,
        AV=None,
    ):
        total_energy, ME = self.metric_energy(
            E,
            a_field,
            loss_type=self.loss_type,
            MX=ME,
        )
        _, captured_energy, _, Q, AQ, ME = self.project_onto_basis(
            E,
            a_field,
            V=V,
            Q=Q,
            AQ=AQ,
            ME=ME,
            AV=AV,
            loss_type=self.loss_type,
        )
        residual_energy = total_energy - captured_energy
        return total_energy, captured_energy, residual_energy, Q, AQ, ME

    def forward(
        self,
        V=None,
        e_family=None,
        a_field=None,
        Q=None,
        AQ=None,
        ME=None,
        return_stats=False,
    ):
        with self._forward_cache_scope():
            if e_family is None:
                raise ValueError("e_family must be provided")
            if a_field is None:
                raise ValueError("a_field must be provided")

            E = self._normalize_e_family_shape(e_family, a_field=a_field)
            a_field = a_field.to(self.dtype)
            use_explicit_basis = (Q is not None)

            if V is None and not use_explicit_basis:
                raise ValueError("Either raw basis V or orthonormal basis Q must be provided")

            if V is not None:
                V = torch.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0).to(self.dtype)
            if use_explicit_basis:
                Q = torch.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0).to(self.dtype)
            if AQ is not None:
                AQ = torch.nan_to_num(AQ, nan=0.0, posinf=0.0, neginf=0.0).to(self.dtype)
            if ME is not None:
                ME = torch.nan_to_num(ME, nan=0.0, posinf=0.0, neginf=0.0).to(self.dtype)

            AV = None
            if (not use_explicit_basis) and (
                self.loss_type == "a" or (
                self.orth_weight > 0.0 and self.orth_reg_metric == "a"
                )
            ):
                AV = self._apply_A(V, a_field)

            total_energy, captured_energy, residual_energy, _, _, _ = self._projection_capture_terms(
                E,
                a_field,
                V=V,
                Q=Q,
                AQ=AQ,
                ME=ME,
                AV=AV,
            )

            capture_stats = self._reduce_ratio_over_batch(
                captured_energy,
                total_energy,
                eps=self.eps,
                batch_reduction=self.batch_reduction,
                alpha=self.alpha,
            )
            residual_stats = self._reduce_ratio_over_batch(
                residual_energy,
                total_energy,
                eps=self.eps,
                batch_reduction=self.batch_reduction,
                alpha=self.alpha,
            )

            sum_total = total_energy.sum()
            sum_captured = captured_energy.sum()
            sum_residual = residual_energy.sum()
            capture_ratio = capture_stats["ratio"]
            loss_proj = residual_stats["ratio"]
            loss_per_sample_mean = residual_stats["per_sample_mean"]
            loss_global_sum_ratio = residual_stats["global_sum_ratio"]

            if use_explicit_basis:
                loss = loss_proj
            else:
                loss_orth = self._orth_regularizer(V, a_field, MV=AV)
                loss = loss_proj + self.orth_weight * loss_orth

            if not return_stats:
                return loss

            stats = {
                "loss_type": self.loss_type,
                "batch_reduction": self.batch_reduction,
                "alpha": float(self.alpha),
                "capture_ratio": capture_ratio.detach(),
                "capture_ratio_per_sample_mean": capture_stats["per_sample_mean"].detach(),
                "capture_ratio_global_sum": capture_stats["global_sum_ratio"].detach(),
                "loss_proj": loss_proj.detach(),
                "loss_per_sample_mean": loss_per_sample_mean.detach(),
                "loss_global_sum_ratio": loss_global_sum_ratio.detach(),
                "mean_captured_energy": captured_energy.detach().mean(),
                "mean_total_energy": total_energy.detach().mean(),
                "mean_residual_energy": residual_energy.detach().mean(),
                "sum_captured": sum_captured.detach(),
                "sum_total": sum_total.detach(),
                "sum_captured_energy": sum_captured.detach(),
                "sum_total_energy": sum_total.detach(),
                "sum_residual_energy": sum_residual.detach(),
                "num_snapshots": captured_energy.new_tensor(float(captured_energy.numel())),
            }
            return loss, stats
