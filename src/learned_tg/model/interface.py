import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from contextlib import contextmanager
from torch.nn.init import constant_, xavier_uniform_

from learned_tg.model.losses import primitives as coarse_loss
from learned_tg.model.losses.primitives import require_finite, symmetrize_matrix, a_orthonormalize_from_AV, euclidean_orthonormalize, compute_metric_whitened_snapshot_family, canonical_metric
from learned_tg.model.deeponet import InterfaceDeepONet
from learned_tg.model.fno import InterfaceFNO

from learned_tg.model.operators.ccfvm_asm import CCFVM_ASM

from learned_tg.model.operators.gmg import InterfaceGMGHierarchy, _interface_gmg_down, _interface_gmg_up, _safe_interface_gmg_subdomain_count
from learned_tg.model.basis_utils.smoothing import smooth_basis as _smooth_basis_impl
from learned_tg.model.basis_utils.reference_basis import _infer_grid_side_from_num_nodes, _zero_dirichlet_boundary_columns, _flatten_coord_field, _broadcast_reference_basis, _make_constant_reference_columns, _make_high_contrast_reference_columns, _make_block_reference_columns, _make_low_frequency_reference_columns, _make_smoothed_random_reference_columns
from learned_tg.model.losses.base import AProjectionLossBase, normalize_snapshot_family
from learned_tg.model.losses.energy_projection import EnergyProjectionLoss
from learned_tg.model.losses.subspace import SubspaceLoss


# ============================================================
#  Utilities for basis smoothing / A-orthonormalization
# ============================================================
from learned_tg.model.basis_utils.diagnostics import build_orth_debug_dict as _build_orth_debug_dict, format_orth_debug as _format_orth_debug, metric_cross_gram as _metric_cross_gram, subspace_shift_metric as _subspace_shift_metric, euclidean_orthogonality_error as _euclidean_orthogonality_error, basis_orthogonality_error as _basis_orthogonality_error, max_metric_overlap as _max_metric_overlap


def _smooth_basis_with_solver(
    op_solver,
    a_field,
    V,
    mode="asm",
    omega=0.6,
    num_steps=1,
    k_1d=10,
    target_overlap=0.3,
):
    """
    V: (B, N, m) — thin wrapper around model.basis_utils.smoothing.smooth_basis.
    """
    return _smooth_basis_impl(op_solver, V, mode=mode, num_steps=num_steps, omega=omega)


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
    metric = canonical_metric(metric, name="snapshot whitening metric")
    max_rank = _resolve_snapshot_whiten_rank(E, rank)
    if max_rank is None:
        return None

    if metric == "l2":
        return compute_metric_whitened_snapshot_family(
            E,
            rank=max_rank,
            metric="l2",
            reg=reg,
            output_layout="family",
        )

    if op_solver is None:
        raise ValueError("A-metric whitening requires a valid op_solver.")

    AE = op_solver.apply_A(E)
    return compute_metric_whitened_snapshot_family(
        E,
        rank=max_rank,
        metric="a",
        AE=AE,
        reg=reg,
        output_layout="family",
    )



class JumpingCoefficient(nn.Module):
    """
    跳跃系数扩散算子包裹器
    """
    def __init__(
        self,
        a_field,
        n=63,
        k_1d=10,
        coarse_space_append_island_basis=False,
        coarse_space_island_basis_threshold=500.0,
        subdomain_solver="cholesky",
        jacobi_pcg_iters=10,
        vcycle_coarsest_threshold=24,
        vcycle_smooth_steps=2,
        vcycle_iter=2,
        vcycle_omega=2.0 / 3.0,
        vcycle_compile=True,
        phi=None,
        transmissibility="harmonic",
    ):
        super(JumpingCoefficient, self).__init__()
        self.a_field = a_field.double()
        self.device = a_field.device
        self.n = n
        self.transmissibility = str(transmissibility).lower()
        self.phi = None if phi is None else phi.to(
            device=self.device,
            dtype=torch.float64,
        )
        self.coarse_space_append_island_basis = bool(coarse_space_append_island_basis)
        self.coarse_space_island_basis_threshold = float(coarse_space_island_basis_threshold)
        self.subdomain_solver = subdomain_solver
        self.jacobi_pcg_iters = int(jacobi_pcg_iters)

        self.asm = CCFVM_ASM(
            a_field,
            k_1d=k_1d,
            target_overlap=0.3,
            subdomain_solver=self.subdomain_solver,
            jacobi_pcg_iters=self.jacobi_pcg_iters,
            vcycle_coarsest_threshold=int(vcycle_coarsest_threshold),
            vcycle_smooth_steps=int(vcycle_smooth_steps),
            vcycle_iter=int(vcycle_iter),
            vcycle_omega=float(vcycle_omega),
            vcycle_compile=bool(vcycle_compile),
            phi=self.phi,
            transmissibility=self.transmissibility,
        )
        self._gmg_hierarchy_cache = {}

    def forward(self, x):
        """
        全域矩阵向量乘法 (直接透传给 ASM 算子的 forward)
        """
        return self.asm(x)

    def asm_smooth(self, x, f, M_smooth=1, omega=0.8):
        return self.asm.asm_smooth(x, f, M_smooth=M_smooth, omega=omega)

    def jacobi_smooth(self, x, f, M_smooth=1, omega=0.8):
        """
        对解向量 x 做带权 Jacobi 磨光:
            r = f - A x
            x <- x + omega * D^{-1} r
        其中 D 使用 self.asm.sc
        """
        x = x.double()
        f = f.double()

        if M_smooth is None:
            M_smooth = 1
        if M_smooth <= 0:
            return x

        D = self.asm.sc.abs() + 1e-12
        for _ in range(int(M_smooth)):
            r = f - self.forward(x)
            x = x + omega * (r / D)
        return x

    def _get_gmg_hierarchy(
        self,
        gmg_num_levels=None,
        gmg_min_coarse_size=4,
        gmg_smoother="jacobi",
        gmg_pre_smooth_steps=2,
        gmg_post_smooth_steps=2,
        gmg_omega=0.8,
        gmg_coarse_solve="direct",
    ):
        cache_key = (
            None if gmg_num_levels is None else int(gmg_num_levels),
            int(gmg_min_coarse_size),
            str(gmg_smoother).lower(),
            int(gmg_pre_smooth_steps),
            int(gmg_post_smooth_steps),
            float(gmg_omega),
            str(gmg_coarse_solve).lower(),
        )
        hierarchy = self._gmg_hierarchy_cache.get(cache_key)
        if hierarchy is None:
            hierarchy = InterfaceGMGHierarchy(
                self.asm,
                num_levels=cache_key[0],
                min_coarse_size=cache_key[1],
                smoother=cache_key[2],
                pre_smooth_steps=cache_key[3],
                post_smooth_steps=cache_key[4],
                omega=cache_key[5],
                coarse_solve=cache_key[6],
            )
            self._gmg_hierarchy_cache[cache_key] = hierarchy
        return hierarchy

    def gmg_smooth(
        self,
        x,
        f,
        M_smooth=1,
        omega=0.8,
        gmg_num_levels=None,
        gmg_min_coarse_size=4,
        gmg_smoother="jacobi",
        gmg_pre_smooth_steps=2,
        gmg_post_smooth_steps=2,
        gmg_omega=0.8,
        gmg_coarse_solve="direct",
    ):
        """
        对解向量 x 做几何多重网格 V-cycle 平滑:
            r = f - A x
            x <- x + omega * GMG(r)
        """
        x = x.double()
        f = f.double()

        if M_smooth is None:
            M_smooth = 1
        if M_smooth <= 0:
            return x

        hierarchy = self._get_gmg_hierarchy(
            gmg_num_levels=gmg_num_levels,
            gmg_min_coarse_size=gmg_min_coarse_size,
            gmg_smoother=gmg_smoother,
            gmg_pre_smooth_steps=gmg_pre_smooth_steps,
            gmg_post_smooth_steps=gmg_post_smooth_steps,
            gmg_omega=gmg_omega,
            gmg_coarse_solve=gmg_coarse_solve,
        )
        for _ in range(int(M_smooth)):
            r = f - self.forward(x)
            z = hierarchy.apply(r)
            x = x + float(omega) * z
        return x

    def solution_smooth(
        self,
        x,
        f,
        mode="asm",
        M_smooth=1,
        omega=0.8,
        gmg_num_levels=None,
        gmg_min_coarse_size=4,
        gmg_smoother="jacobi",
        gmg_pre_smooth_steps=2,
        gmg_post_smooth_steps=2,
        gmg_omega=0.8,
        gmg_coarse_solve="direct",
    ):
        """
        mode in {"none", "jacobi", "asm", "gmg"}
        - none: 直接返回 x
        - jacobi: 调用 jacobi_smooth
        - asm: 调用现有 asm_smooth
        - gmg: 调用几何多重网格 V-cycle smoother
        """
        mode = str(mode).lower()
        if mode not in {"none", "jacobi", "asm", "gmg"}:
            raise ValueError(
                "solution smoothing mode must be one of "
                f"{{'none', 'jacobi', 'asm', 'gmg'}}, got {mode!r}"
            )

        x = x.double()
        f = f.double()
        if M_smooth is None:
            M_smooth = 1

        if mode == "none" or M_smooth <= 0:
            return x
        if mode == "jacobi":
            return self.jacobi_smooth(x, f, M_smooth=M_smooth, omega=omega)
        if mode == "asm":
            return self.asm_smooth(x, f, M_smooth=M_smooth, omega=omega)
        if mode == "gmg":
            return self.gmg_smooth(
                x,
                f,
                M_smooth=M_smooth,
                omega=omega,
                gmg_num_levels=gmg_num_levels,
                gmg_min_coarse_size=gmg_min_coarse_size,
                gmg_smoother=gmg_smoother,
                gmg_pre_smooth_steps=gmg_pre_smooth_steps,
                gmg_post_smooth_steps=gmg_post_smooth_steps,
                gmg_omega=gmg_omega,
                gmg_coarse_solve=gmg_coarse_solve,
            )

    def smooth_basis(self, V, mode="asm", omega=0.6, num_steps=1):
        """
        对 coarse basis 做后处理：
        - mode='none'   : 不做处理
        - mode='jacobi' : SA-AMG 风格的一步/多步 weighted Jacobi
        - mode='asm'    : 与训练数据/在线平滑器一致的一步/多步 ASM smoothing

        注意：
        这一步会改变子空间本身，不是单纯换基。
        """
        return _smooth_basis_with_solver(
            op_solver=self.asm,
            a_field=self.a_field,
            V=V,
            mode=mode,
            omega=omega,
            num_steps=num_steps,
            k_1d=self.asm.k_1d,
            target_overlap=0.3,
        )

    def a_orthonormalize(self, V, reg=1e-10):
        """
        对给定 coarse basis 做 A-正交化，返回:
        Q, AQ, A_rom = Q^T A Q
        """
        V = torch.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0).double()
        AV = self.asm.apply_A(V)
        return a_orthonormalize_from_AV(V, AV, reg=reg)

    def _maybe_append_island_basis(self, V):
        if not self.coarse_space_append_island_basis:
            return V

        batch_size, num_nodes, _ = V.shape
        island_basis = _make_high_contrast_reference_columns(
            self.a_field,
            threshold=self.coarse_space_island_basis_threshold,
            zero_boundary=False,
        ).to(device=V.device, dtype=V.dtype)

        if island_basis.shape[1] != num_nodes:
            raise ValueError(
                "Island basis num_nodes mismatch: "
                f"expected {num_nodes}, got {island_basis.shape[1]}"
            )
        if island_basis.shape[0] == 1 and batch_size > 1:
            island_basis = island_basis.expand(batch_size, -1, -1)
        elif island_basis.shape[0] != batch_size:
            raise ValueError(
                "Island basis batch mismatch: "
                f"expected {batch_size}, got {island_basis.shape[0]}"
            )

        return torch.cat([V, island_basis.contiguous()], dim=2)

    def prepare_coarse_basis(
        self,
        V,
        basis_smooth_mode="asm",
        basis_smooth_steps=1,
        basis_smooth_omega=0.6,
        orthonormalize_metric="a",
        orth_reg=1e-10,
    ):
        """
        Prepare a coarse-space object for repeated online solves.

        This performs all basis-dependent work once:
          1) basis smoothing / post-processing
          2) orthonormalization under the requested metric ('a' or 'l2')
          3) coarse operator regularization + factorization

        Returns
        -------
        dict with keys:
          - Q, AQ, A_rom, A_rom_reg
          - coarse_solver: "cholesky" or "pinv"
          - coarse_factor: Cholesky factor L or pseudo-inverse
          - metadata for validation/debugging
        """
        V = self._maybe_append_island_basis(V)
        batch_size, num_nodes, basis_dim = V.shape
        device = V.device
        dtype = V.dtype
        metric_key = str(orthonormalize_metric).strip().lower()
        direct_galerkin = metric_key in {"none", "raw", "direct_galerkin"}
        if not direct_galerkin:
            orthonormalize_metric = canonical_metric(
                orthonormalize_metric, name="coarse basis orthonormalization metric"
            )
        else:
            orthonormalize_metric = "direct_galerkin"

        V_tilde = self.smooth_basis(
            V,
            mode=basis_smooth_mode,
            omega=basis_smooth_omega,
            num_steps=basis_smooth_steps,
        )
        if direct_galerkin:
            # Local-spectral/GenEO bases already have meaningful local support.
            # Preserve their columns and assemble V^T A V directly instead of
            # applying the common dense global A-orthonormalization wrapper.
            Q = V_tilde
            AQ = self.asm.apply_A(Q)
            A_rom = symmetrize_matrix(torch.matmul(Q.transpose(1, 2), AQ))
        elif orthonormalize_metric == "a":
            Q, AQ, A_rom = self.a_orthonormalize(V_tilde, reg=orth_reg)
        else:
            Q = euclidean_orthonormalize(V_tilde)
            AQ = self.asm.apply_A(Q)
            A_rom = symmetrize_matrix(torch.matmul(Q.transpose(1, 2), AQ))

        I = torch.eye(basis_dim, device=device, dtype=dtype).unsqueeze(0)
        A_rom_reg = A_rom + orth_reg * I

        try:
            coarse_factor = torch.linalg.cholesky(A_rom_reg)
            coarse_solver = "cholesky"
        except RuntimeError:
            coarse_factor = torch.linalg.pinv(A_rom_reg)
            coarse_solver = "pinv"

        return {
            "Q": Q,
            "AQ": AQ,
            "A_rom": A_rom,
            "A_rom_reg": A_rom_reg,
            "coarse_solver": coarse_solver,
            "coarse_factor": coarse_factor,
            "orth_reg": float(orth_reg),
            "basis_smooth_mode": basis_smooth_mode,
            "basis_smooth_steps": int(basis_smooth_steps),
            "basis_smooth_omega": float(basis_smooth_omega),
            "orthonormalize_metric": orthonormalize_metric,
            "batch_size": int(batch_size),
            "num_nodes": int(num_nodes),
            "basis_dim": int(basis_dim),
        }

    def _validate_prepared_coarse_basis(self, prepared_coarse_basis, x):
        required_keys = {
            "Q",
            "AQ",
            "A_rom",
            "A_rom_reg",
            "coarse_solver",
            "coarse_factor",
            "batch_size",
            "num_nodes",
            "basis_dim",
        }
        missing_keys = sorted(required_keys.difference(prepared_coarse_basis.keys()))
        if missing_keys:
            raise KeyError(
                f"prepared_coarse_basis is missing required keys: {missing_keys}"
            )

        Q = prepared_coarse_basis["Q"]
        batch_size = int(prepared_coarse_basis["batch_size"])
        num_nodes = int(prepared_coarse_basis["num_nodes"])
        basis_dim = int(prepared_coarse_basis["basis_dim"])

        if Q.shape != (batch_size, num_nodes, basis_dim):
            raise ValueError(
                "prepared_coarse_basis['Q'] shape is inconsistent with its metadata: "
                f"Q.shape={tuple(Q.shape)}, metadata={(batch_size, num_nodes, basis_dim)}"
            )

        x_batch = int(x.shape[0])
        x_nodes = int(x.shape[-2] * x.shape[-1])
        if batch_size != x_batch or num_nodes != x_nodes:
            raise ValueError(
                "prepared_coarse_basis does not match the current solve batch/grid: "
                f"prepared (B={batch_size}, N={num_nodes}) vs current (B={x_batch}, N={x_nodes})"
            )

        if Q.device != x.device:
            raise ValueError(
                "prepared_coarse_basis must be on the same device as the solve tensors: "
                f"{Q.device} vs {x.device}"
            )

    def _solve_prepared_coarse_system(self, prepared_coarse_basis, rhs_rom):
        coarse_solver = prepared_coarse_basis["coarse_solver"]
        coarse_factor = prepared_coarse_basis["coarse_factor"]
        if coarse_solver == "identity":
            # For an A-orthonormal basis Q, Q^T A Q = I and the Galerkin
            # inverse is already absorbed into Q.  The coarse coefficients are
            # therefore exactly Q^T r (up to the orthonormalization error).
            return rhs_rom
        if coarse_solver == "cholesky":
            return torch.cholesky_solve(rhs_rom, coarse_factor)
        if coarse_solver == "pinv":
            return torch.matmul(coarse_factor, rhs_rom)
        raise ValueError(f"Unknown coarse solver type: {coarse_solver!r}")

    def two_grid(
        self,
        x,
        f,
        V=None,
        prepared_coarse_basis=None,
        M_smooth=1,
        omega=0.8,
        pre_smooth_mode="asm",
        pre_smooth_steps=None,
        pre_smooth_omega=None,
        post_smooth_mode="none",
        post_smooth_steps=0,
        post_smooth_omega=0.8,
        basis_smooth_mode="asm",
        basis_smooth_steps=1,
        basis_smooth_omega=0.6,
        coarse_basis_orth_metric="a",
        orth_reg=1e-10,
        gmg_num_levels=None,
        gmg_min_coarse_size=4,
        gmg_smoother="jacobi",
        gmg_pre_smooth_steps=2,
        gmg_post_smooth_steps=2,
        gmg_omega=0.8,
        gmg_coarse_solve="direct",
    ):
        batch_size = x.shape[0]
        n = self.n

        x = x.double()
        f = f.double()
        if V is not None:
            V = V.double()

        if pre_smooth_steps is None:
            pre_smooth_steps = M_smooth
        if pre_smooth_omega is None:
            pre_smooth_omega = omega

        # 1. 预平滑
        x = self.solution_smooth(
            x,
            f,
            mode=pre_smooth_mode,
            M_smooth=pre_smooth_steps,
            omega=pre_smooth_omega,
            gmg_num_levels=gmg_num_levels,
            gmg_min_coarse_size=gmg_min_coarse_size,
            gmg_smoother=gmg_smoother,
            gmg_pre_smooth_steps=gmg_pre_smooth_steps,
            gmg_post_smooth_steps=gmg_post_smooth_steps,
            gmg_omega=gmg_omega,
            gmg_coarse_solve=gmg_coarse_solve,
        )

        # 2. 残差
        r = f - self.forward(x)
        r_vec = r.view(batch_size, -1, 1)

        if prepared_coarse_basis is None:
            if V is None:
                raise ValueError("Either V or prepared_coarse_basis must be provided")
            prepared_coarse_basis = self.prepare_coarse_basis(
                V,
                basis_smooth_mode=basis_smooth_mode,
                basis_smooth_steps=basis_smooth_steps,
                basis_smooth_omega=basis_smooth_omega,
                orthonormalize_metric=coarse_basis_orth_metric,
                orth_reg=orth_reg,
            )
        else:
            self._validate_prepared_coarse_basis(prepared_coarse_basis, x)

        Q = prepared_coarse_basis["Q"]

        # 4. 粗问题右端项
        rhs_rom = torch.matmul(Q.transpose(1, 2), r_vec)  # (B, m+1, 1)

        y = self._solve_prepared_coarse_system(prepared_coarse_basis, rhs_rom)

        # 5. 粗校正
        correction = torch.matmul(Q, y).view(batch_size, 1, n, n)
        x = x + correction

        # 6. 后平滑
        x = self.solution_smooth(
            x,
            f,
            mode=post_smooth_mode,
            M_smooth=post_smooth_steps,
            omega=post_smooth_omega,
            gmg_num_levels=gmg_num_levels,
            gmg_min_coarse_size=gmg_min_coarse_size,
            gmg_smoother=gmg_smoother,
            gmg_pre_smooth_steps=gmg_pre_smooth_steps,
            gmg_post_smooth_steps=gmg_post_smooth_steps,
            gmg_omega=gmg_omega,
            gmg_coarse_solve=gmg_coarse_solve,
        )

        return x


    def solve(
        self,
        x0,
        f,
        V=None,
        prepared_coarse_basis=None,
        max_iter=200,
        tol=1e-6,
        M_smooth=1,
        omega=0.8,
        pre_smooth_mode="asm",
        pre_smooth_steps=None,
        pre_smooth_omega=None,
        post_smooth_mode="none",
        post_smooth_steps=0,
        post_smooth_omega=0.8,
        basis_smooth_mode="asm",
        basis_smooth_steps=1,
        basis_smooth_omega=0.6,
        coarse_basis_orth_metric="a",
        orth_reg=1e-10,
        gmg_num_levels=None,
        gmg_min_coarse_size=4,
        gmg_smoother="jacobi",
        gmg_pre_smooth_steps=2,
        gmg_post_smooth_steps=2,
        gmg_omega=0.8,
        gmg_coarse_solve="direct",
    ):
        """
        相对残量控制的 Two-Grid Solver
        """
        x = x0.clone().double()
        f = f.double()
        if V is not None:
            V = V.double()

        residuals = []

        f_norm = torch.sqrt(torch.sum(f ** 2, dim=(1, 2, 3)))
        f_norm = torch.where(f_norm > 1e-12, f_norm, torch.ones_like(f_norm))

        if prepared_coarse_basis is None:
            if V is None:
                raise ValueError("Either V or prepared_coarse_basis must be provided")
            prepared_coarse_basis = self.prepare_coarse_basis(
                V,
                basis_smooth_mode=basis_smooth_mode,
                basis_smooth_steps=basis_smooth_steps,
                basis_smooth_omega=basis_smooth_omega,
                orthonormalize_metric=coarse_basis_orth_metric,
                orth_reg=orth_reg,
            )
        else:
            self._validate_prepared_coarse_basis(prepared_coarse_basis, x)

        for i in range(max_iter):
            Ax = self.forward(x)
            r = f - Ax
            res_norm = torch.sqrt(torch.sum(r ** 2, dim=(1, 2, 3)))

            relative_res = res_norm / f_norm
            residuals.append(relative_res.mean().item())

            if residuals[-1] < tol:
                print(f"[TG] 收敛于第 {i+1} 步 | 相对残差 = {residuals[-1]:.6e}")
                break

            x = self.two_grid(
                x, f, V,
                prepared_coarse_basis=prepared_coarse_basis,
                M_smooth=M_smooth,
                omega=omega,
                pre_smooth_mode=pre_smooth_mode,
                pre_smooth_steps=pre_smooth_steps,
                pre_smooth_omega=pre_smooth_omega,
                post_smooth_mode=post_smooth_mode,
                post_smooth_steps=post_smooth_steps,
                post_smooth_omega=post_smooth_omega,
                basis_smooth_mode=basis_smooth_mode,
                basis_smooth_steps=basis_smooth_steps,
                basis_smooth_omega=basis_smooth_omega,
                coarse_basis_orth_metric=coarse_basis_orth_metric,
                orth_reg=orth_reg,
                gmg_num_levels=gmg_num_levels,
                gmg_min_coarse_size=gmg_min_coarse_size,
                gmg_smoother=gmg_smoother,
                gmg_pre_smooth_steps=gmg_pre_smooth_steps,
                gmg_post_smooth_steps=gmg_post_smooth_steps,
                gmg_omega=gmg_omega,
                gmg_coarse_solve=gmg_coarse_solve,
            )

        return x, residuals


    def fcg_solve(
            self, x0, f, V=None, prepared_coarse_basis=None,
            max_iter=200, m_max=20, tol=1e-6, M_smooth=1, omega=0.8,
            pre_smooth_mode="asm", pre_smooth_steps=None, pre_smooth_omega=None,
            post_smooth_mode="none", post_smooth_steps=0, post_smooth_omega=0.8,
            basis_smooth_mode="none", basis_smooth_steps=1, basis_smooth_omega=0.6,
            coarse_basis_orth_metric="a", orth_reg=1e-10,
            gmg_num_levels=None, gmg_min_coarse_size=4, gmg_smoother="jacobi",
            gmg_pre_smooth_steps=2, gmg_post_smooth_steps=2, gmg_omega=0.8,
            gmg_coarse_solve="direct", reorthogonalize=True, breakdown_tol=1e-28,
            preconditioner=None, verbose=True, *, return_info=False,
            krylov_method="fgmres", krylov_diagnostics=False,
        ):
        """Compatibility entry point for verified right FGMRES (default).

        ``krylov_method='fgcr'`` selects the tested image-recomputed GCR.
        Both use two-pass orthogonalization and verify the true final residual.
        ``max_iter`` counts Krylov updates across all full restarts. The list
        history includes the initial state; use ``history.num_updates`` or
        ``return_info=True`` instead of inferring iterations from its length.
        The original unsafe algorithm is retained as ``fcg_solve_legacy``.
        """
        from learned_tg.model.krylov import solve_batched, KrylovHistory
        if not reorthogonalize:
            raise ValueError("the verified Krylov backend requires two-pass orthogonalization")
        x0=x0.double();f=f.double()
        if preconditioner is None:
            if prepared_coarse_basis is None:
                if V is None:raise ValueError("provide V, prepared_coarse_basis, or a preconditioner callback")
                prepared_coarse_basis=self.prepare_coarse_basis(V.double(),
                    basis_smooth_mode=basis_smooth_mode,basis_smooth_steps=basis_smooth_steps,
                    basis_smooth_omega=basis_smooth_omega,
                    orthonormalize_metric=coarse_basis_orth_metric,orth_reg=orth_reg)
            else:self._validate_prepared_coarse_basis(prepared_coarse_basis,x0)
            def apply_preconditioner(r):
                return self.two_grid(torch.zeros_like(r),r,prepared_coarse_basis=prepared_coarse_basis,
                    M_smooth=M_smooth,omega=omega,
                    pre_smooth_mode=pre_smooth_mode,pre_smooth_steps=pre_smooth_steps,pre_smooth_omega=pre_smooth_omega,
                    post_smooth_mode=post_smooth_mode,post_smooth_steps=post_smooth_steps,post_smooth_omega=post_smooth_omega,
                    basis_smooth_mode=basis_smooth_mode,basis_smooth_steps=basis_smooth_steps,basis_smooth_omega=basis_smooth_omega,
                    coarse_basis_orth_metric=coarse_basis_orth_metric,orth_reg=orth_reg,
                    gmg_num_levels=gmg_num_levels,gmg_min_coarse_size=gmg_min_coarse_size,gmg_smoother=gmg_smoother,
                    gmg_pre_smooth_steps=gmg_pre_smooth_steps,gmg_post_smooth_steps=gmg_post_smooth_steps,
                    gmg_omega=gmg_omega,gmg_coarse_solve=gmg_coarse_solve).double()
        else:apply_preconditioner=preconditioner
        method="fgcr" if m_max==0 else krylov_method
        result=solve_batched(self.forward,apply_preconditioner,f,x0,method=method,
            tol=tol,maxiter=max_iter,restart=m_max,diagnostics=krylov_diagnostics)
        self.last_krylov_info=result.info
        history=KrylovHistory(result.info)
        if verbose:
            print(f"[{method.upper()}] {result.info['status']}: updates={result.info['updates']}, true relative residual={result.info['true_final_residual']:.6e}")
        if return_info:return result.x,history,result.info
        return result.x,history

    def fgmres_solve(self, *args, **kwargs):
        """Explicit name for the verified production default."""
        kwargs['krylov_method']='fgmres'
        return self.fcg_solve(*args,**kwargs)

    def fgcr_solve(self, *args, **kwargs):
        """Verified GCR alternative, with recomputed operator images."""
        kwargs['krylov_method']='fgcr'
        return self.fcg_solve(*args,**kwargs)

    def fgmres_split_solve(self, initial, f, prepared_coarse_basis, *,
                           max_iter=300, m_max=50, tol=1e-6, safety=.5):
        """Opt-in TG(1,1) split-pressure API; never returns a collapsed tensor."""
        from learned_tg.model.split_pressure import SplitPressure, StoredFVMResidual, fgmres_split
        from learned_tg.model.accurate_matvec import AccurateFVMAction
        if not isinstance(initial,SplitPressure):
            raise TypeError('provide both float64 pressure components')
        self._validate_prepared_coarse_basis(prepared_coarse_basis,initial.high)
        def B(r):
            return self.two_grid(torch.zeros_like(r),r,prepared_coarse_basis=prepared_coarse_basis,
                M_smooth=1,omega=.8,pre_smooth_mode='asm',pre_smooth_steps=1,pre_smooth_omega=.8,
                post_smooth_mode='asm',post_smooth_steps=1,post_smooth_omega=.8,
                basis_smooth_mode='none',basis_smooth_steps=0,coarse_basis_orth_metric='a',
                orth_reg=prepared_coarse_basis['orth_reg'])
        result=fgmres_split(AccurateFVMAction(self.asm),B,StoredFVMResidual(self.asm),f,
            initial=initial,tol=tol,safety=safety,max_updates=max_iter,restart=m_max)
        self.last_krylov_info=result.info
        return result

    def fcg_solve_legacy(
            self,
            x0,
            f,
            V=None,
            prepared_coarse_basis=None,
            max_iter=200,
            m_max=20,
            tol=1e-6,
            M_smooth=1,
            omega=0.8,
            pre_smooth_mode="asm",
            pre_smooth_steps=None,
            pre_smooth_omega=None,
            post_smooth_mode="none",
            post_smooth_steps=0,
            post_smooth_omega=0.8,
            basis_smooth_mode="none",
            basis_smooth_steps=1,
            basis_smooth_omega=0.6,
            coarse_basis_orth_metric="a",
            orth_reg=1e-10,
            gmg_num_levels=None,
            gmg_min_coarse_size=4,
            gmg_smoother="jacobi",
            gmg_pre_smooth_steps=2,
            gmg_post_smooth_steps=2,
            gmg_omega=0.8,
            gmg_coarse_solve="direct",
            reorthogonalize=True,
            breakdown_tol=1e-28,
            preconditioner=None,
            verbose=True,
        ):
            """
            Batched Flexible Generalized Conjugate Residual (F-GCR) solver.
            (Adapted for robust deep learning Batched Tensor environments and jumping-coefficients)
            """
            # 1. 强制全程双精度
            x = x0.clone().double()
            f = f.clone().double()
            if V is not None:
                V = V.double()

            batch_size = f.shape[0]

            # 辅助函数：安全的 Batch 维度内积
            def batch_dot(u, v):
                return torch.sum(u * v, dim=(1, 2, 3), keepdim=True)

            # 提前准备粗空间基底（复用你原本的严谨逻辑）。可选的
            # preconditioner 回调用于 one-level ASM-FGCR 公平基线；TG 调用
            # 不传该参数，因而保留原有路径和全部参数。
            use_external_preconditioner = preconditioner is not None
            missing_preconditioner = (
                not use_external_preconditioner
                and prepared_coarse_basis is None
                and V is None
            )
            if missing_preconditioner:
                raise ValueError(
                    "gcr_solve requires a preconditioner: provide V, "
                    "prepared_coarse_basis, or a preconditioner callback."
                )

            if not use_external_preconditioner and prepared_coarse_basis is None:
                prepared_coarse_basis = self.prepare_coarse_basis(
                    V,
                    basis_smooth_mode=basis_smooth_mode,
                    basis_smooth_steps=basis_smooth_steps,
                    basis_smooth_omega=basis_smooth_omega,
                    orthonormalize_metric=coarse_basis_orth_metric,
                    orth_reg=orth_reg,
                )
            elif not use_external_preconditioner:
                self._validate_prepared_coarse_basis(prepared_coarse_basis, x)

            # 初始残差计算
            r = f - self.forward(x).double()
            f_norm = torch.sqrt(batch_dot(f, f)).clamp(min=1e-12)

            P_list = []
            Q_list = []
            residuals = []

            # 活动掩码：记录哪些样本还在迭代 (1.0表示未收敛，0.0表示已收敛)
            active_mask = torch.ones((batch_size, 1, 1, 1), device=x.device, dtype=torch.float64)

            for it in range(int(max_iter)):
                res_norm = torch.sqrt(batch_dot(r, r))
                rel_res = res_norm / f_norm

                # 更新当前 Batch 的收敛状态掩码
                active_mask = (rel_res > tol).double()
                res_max = rel_res.max().item()
                res_mean = rel_res.mean().item()
                residuals.append(res_mean)

                if active_mask.sum() == 0:
                    if verbose:
                        print(
                            f"[GCR] Converged at iter {it} | "
                            f"max rel res = {res_max:.6e}, mean = {res_mean:.6e}"
                        )
                    break

                # 2. 预条件求解：z = B(r)
                # 将 kwargs 打包传入 two_grid
                if use_external_preconditioner:
                    z = preconditioner(r).double()
                else:
                    z = self.two_grid(
                        torch.zeros_like(r),
                        r,
                        V=V,
                        prepared_coarse_basis=prepared_coarse_basis,
                        M_smooth=M_smooth,
                        omega=omega,
                        pre_smooth_mode=pre_smooth_mode,
                        pre_smooth_steps=pre_smooth_steps,
                        pre_smooth_omega=pre_smooth_omega,
                        post_smooth_mode=post_smooth_mode,
                        post_smooth_steps=post_smooth_steps,
                        post_smooth_omega=post_smooth_omega,
                        basis_smooth_mode=basis_smooth_mode,
                        basis_smooth_steps=basis_smooth_steps,
                        basis_smooth_omega=basis_smooth_omega,
                        coarse_basis_orth_metric=coarse_basis_orth_metric,
                        orth_reg=orth_reg,
                        gmg_num_levels=gmg_num_levels,
                        gmg_min_coarse_size=gmg_min_coarse_size,
                        gmg_smoother=gmg_smoother,
                        gmg_pre_smooth_steps=gmg_pre_smooth_steps,
                        gmg_post_smooth_steps=gmg_post_smooth_steps,
                        gmg_omega=gmg_omega,
                        gmg_coarse_solve=gmg_coarse_solve,
                    ).double()

                p = z.clone()
                q = self.forward(p).double()

                # 3. 首次正交化 (Modified Gram-Schmidt)
                for p_k, q_k in zip(P_list, Q_list):
                    qk_qk = batch_dot(q_k, q_k)
                    valid_qk = (qk_qk > breakdown_tol).double()

                    beta = batch_dot(q, q_k) / qk_qk.clamp(min=breakdown_tol)
                    beta = beta * valid_qk  # 屏蔽发生除零击穿的历史向量

                    p = p - beta * p_k
                    q = q - beta * q_k

                # 4. 可选：二次正交化 (Re-orthogonalization)
                if reorthogonalize:
                    for p_k, q_k in zip(P_list, Q_list):
                        qk_qk = batch_dot(q_k, q_k)
                        valid_qk = (qk_qk > breakdown_tol).double()

                        beta = batch_dot(q, q_k) / qk_qk.clamp(min=breakdown_tol)
                        beta = beta * valid_qk

                        p = p - beta * p_k
                        q = q - beta * q_k

                # 5. 最优步长计算与掩码更新
                qq = batch_dot(q, q)
                valid_q = (qq > breakdown_tol).double()  # 检查当前步是否击穿

                alpha = batch_dot(r, q) / qq.clamp(min=breakdown_tol)

                # 【核心逻辑】：只更新 (1)未收敛 且 (2)未击穿 的样本
                alpha = alpha * valid_q * active_mask

                x = x + alpha * p
                r = r - alpha * q

                # 6. 残差精准修正 (Residual Drift Correction)
                # 每隔15步计算真实残差，防止单步累加在高对比度下导致浮点截断灾难
                # if (it + 1) % 30 == 0:
                #     r = f - self.forward(x).double()

                # 7. 截断/保存搜索历史
                if m_max is not None and m_max > 0:
                    P_list.append(p)
                    Q_list.append(q)
                    if len(P_list) >= m_max:
                        # FGCR(m) is a restarted method.  A rolling window can
                        # permanently annihilate the next direction once the
                        # stored image space saturates, leaving a flat residual
                        # curve instead of beginning a fresh Krylov cycle.
                        P_list.clear()
                        Q_list.clear()
                elif m_max is None:
                    # 保持全部历史
                    P_list.append(p)
                    Q_list.append(q)
                # 如果 m_max == 0, 不保存历史，退化为带预条件的 Richardson 迭代

            return x, residuals



# ==========================================
#  求解器类
# ==========================================
class Solver(nn.Module):
    """
    跳跃系数求解器
    结合生成的 ROM 基底进行训练
    """
    def __init__(self, config):
        super(Solver, self).__init__()

        self.error_threshold = config.get('error_threshold', 1e-6)
        self.max_iter_num = config.get('max_iter_num', 200)
        self.M_smooth = config.get('M_smooth_solve', 1)
        self.omega = config.get('omega', 0.8)

        self.k_1d = config.get('k_1d', 10)
        self.transmissibility = str(
            config.get("transmissibility", "harmonic")
        ).lower()
        if self.transmissibility not in {
            "harmonic",
            "interface",
            "interface_aware",
        }:
            raise ValueError(
                "transmissibility must be 'harmonic', 'interface', or "
                "'interface_aware'"
            )
        self.subdomain_solver = str(config.get('subdomain_solver', 'cholesky'))
        self.jacobi_pcg_iters = int(config.get('jacobi_pcg_iters', 10))
        self.m_dim = config.get('m_dim', 10)

        loss_objective_config = config.get("loss_objective", "capture")
        self.loss_objective = str(loss_objective_config).lower()
        if self.loss_objective not in {"capture", "svd", "nlss"}:
            raise ValueError(
                "loss_objective must be one of {'capture', 'svd', 'nlss'}, "
                f"got {loss_objective_config!r}"
            )

        self.snapshot_lambda_orth = float(config.get("snapshot_lambda_orth", 0.0))
        self.snapshot_use_prefix_truncation = bool(
            config.get("nlss_use_prefix_truncation", True)
        )
        self.snapshot_effective_rank_mode = str(
            config.get("snapshot_effective_rank_mode", "truncate")
        ).lower()
        self.svd_teacher_rank = config.get("svd_teacher_rank")
        self.svd_allow_embedding = bool(config.get("svd_allow_embedding", True))
        self.svd_loss_variant = str(config.get("svd_loss_variant", "projection")).lower()
        self.log_extra_loss_metrics = bool(config.get("log_extra_loss_metrics", True))

        self.basis_smooth_mode_train = config.get('basis_smooth_mode_train', 'jacobi')
        self.basis_smooth_steps_train = config.get('basis_smooth_steps_train', 1)
        self.basis_smooth_omega_train = config.get('basis_smooth_omega_train', 0.6)

        self.basis_smooth_mode_solve = config.get('basis_smooth_mode_solve', 'jacobi')
        self.basis_smooth_steps_solve = config.get('basis_smooth_steps_solve', 1)
        self.basis_smooth_omega_solve = config.get('basis_smooth_omega_solve', 0.6)
        self.gmg_num_levels = config.get('solve_gmg_num_levels', config.get('gmg_num_levels'))
        self.gmg_min_coarse_size = int(
            config.get('solve_gmg_min_coarse_size', config.get('gmg_min_coarse_size', 4))
        )
        self.gmg_smoother = str(
            config.get('solve_gmg_smoother', config.get('gmg_smoother', 'jacobi'))
        ).lower()
        self.gmg_pre_smooth_steps = int(
            config.get(
                'solve_gmg_pre_smooth_steps',
                config.get('gmg_pre_smooth_steps', 2),
            )
        )
        self.gmg_post_smooth_steps = int(
            config.get(
                'solve_gmg_post_smooth_steps',
                config.get('gmg_post_smooth_steps', 2),
            )
        )
        self.gmg_omega = float(config.get('solve_gmg_omega', config.get('gmg_omega', 0.8)))
        self.gmg_coarse_solve = str(
            config.get(
                'solve_gmg_coarse_solve',
                config.get('gmg_coarse_solve', 'direct'),
            )
        ).lower()
        if self.gmg_smoother not in {"jacobi", "asm"}:
            raise ValueError(
                f"solve GMG smoother must be one of {{'jacobi', 'asm'}}, got {self.gmg_smoother!r}"
            )
        if self.gmg_coarse_solve not in {"direct", "jacobi"}:
            raise ValueError(
                "solve GMG coarse_solve must be one of "
                f"{{'direct', 'jacobi'}}, got {self.gmg_coarse_solve!r}"
            )

        self.basis_orth_reg = config.get('basis_orth_reg', 1e-8)
        self.coarse_basis_orth_metric = canonical_metric(
            config.get('coarse_basis_orth_metric', 'a'),
            name="coarse basis orthonormalization metric",
        )
        self.basis_mode = str(config.get('basis_mode', 'direct')).lower()
        if self.basis_mode not in {'direct', 'transport'}:
            raise ValueError(
                f"basis_mode must be one of {{'direct', 'transport'}}, got {self.basis_mode!r}"
            )

        self.transport_retraction = str(config.get('transport_retraction', 'qr')).lower()
        if self.transport_retraction != 'qr':
            raise ValueError(
                f"Only QR retraction is currently supported, got {self.transport_retraction!r}"
            )
        self.transport_metric = str(config.get('transport_metric', 'l2')).lower()
        if self.transport_metric not in {'l2', 'a'}:
            raise ValueError(
                f"transport_metric must be one of {{'l2', 'a'}}, got {self.transport_metric!r}"
            )

        self.lambda_tan = float(config.get('lambda_tan', 1e-4))
        self.reference_basis_type = str(config.get('reference_basis_type', 'hybrid')).lower()
        self.reference_basis_components = self._normalize_reference_basis_components(
            config.get('reference_basis_components')
        )
        self.reference_num_low_freq = int(config.get('reference_num_low_freq', 4))
        self.reference_block_count_1d = int(
            config.get('reference_block_count_1d', max(2, int(round(np.sqrt(self.m_dim)))))
        )
        self.reference_high_contrast_threshold = float(
            config.get('reference_high_contrast_threshold', 500.0)
        )
        self.coarse_space_append_island_basis = bool(
            config.get('coarse_space_append_island_basis', False)
        )
        self.coarse_space_island_basis_threshold = float(
            config.get(
                'coarse_space_island_basis_threshold',
                self.reference_high_contrast_threshold,
            )
        )
        # Legacy knobs kept for config/checkpoint compatibility. smoothed_random
        # now prefers drawing columns directly from e_family when it is available.
        self.reference_random_smooth_steps = int(
            config.get('reference_random_smooth_steps', 50)
        )
        self.reference_random_smooth_omega = float(
            config.get('reference_random_smooth_omega', 0.8)
        )
        random_seed = config.get('reference_random_seed', config.get('seed', 1234))
        self.reference_random_seed = None if random_seed is None else int(random_seed)
        self.reference_zero_boundary = bool(config.get('reference_zero_boundary', True))
        self.reference_basis_cache_enabled = bool(config.get('reference_basis_cache_enabled', True))
        self._reference_basis_cache = {}
        self._last_basis_debug = {
            "basis_mode": self.basis_mode,
            "transport_metric": self.transport_metric,
        }
        self._lightweight_operator_log_contexts = set()
        self._warning_flags = set()

        capture_family_config = config.get("capture_family", "raw")
        self.capture_family = str(capture_family_config).lower()
        if self.capture_family not in {"raw", "white"}:
            raise ValueError(
                "capture_family must be one of {'raw', 'white'}, "
                f"got {self.capture_family!r}"
            )
        capture_metric_config = config.get("capture_metric", config.get("loss_type", "A"))
        self.capture_metric = canonical_metric(
            capture_metric_config,
            name="capture metric",
        )
        whiten_metric_config = config.get("whiten_metric", self.capture_metric)
        self.whiten_metric = canonical_metric(whiten_metric_config, name="snapshot whitening metric")
        self.whiten_rank = config.get("whiten_rank", self.m_dim)
        self.use_cached_whitened_family = bool(
            config.get("use_cached_whitened_family", True)
        )
        self.loss_type = self.capture_metric
        self.batch_reduction = config.get("batch_reduction", "mean")
        self.batch_reduction_alpha = config.get("alpha", 0.0)
        self.criterion = EnergyProjectionLoss(
            k_1d=self.k_1d,
            target_overlap=0.3,
            epsilon=config.get('loss_reg', 1e-6),
            orth_weight=config.get('orth_weight', 0.01),
            loss_type=self.capture_metric,
            orth_reg_metric=config.get('orth_reg_metric'),
            batch_reduction=self.batch_reduction,
            alpha=self.batch_reduction_alpha,
            dtype=torch.float64,
        )
        subspace_metric = self.capture_metric if self.loss_objective == "capture" else "l2"
        self.snapshot_subspace_criterion = SubspaceLoss(
            loss_type=self.loss_objective,
            metric=subspace_metric,
            lambda_orth=self.snapshot_lambda_orth,
            lambda_decorrelate=float(config.get('lambda_decorrelate', 0.0)),
            use_prefix_truncation=self.snapshot_use_prefix_truncation,
            effective_rank_mode=self.snapshot_effective_rank_mode,
            capture_whiten=(self.capture_family == "white"),
            capture_whiten_rank=self.whiten_rank,
            capture_whiten_metric=self.whiten_metric,
            svd_teacher_rank=self.svd_teacher_rank,
            svd_allow_embedding=self.svd_allow_embedding,
            svd_loss_variant=self.svd_loss_variant,
            log_extra_metrics=self.log_extra_loss_metrics,
            k_1d=self.k_1d,
            target_overlap=0.3,
            orth_reg=config.get('basis_orth_reg', config.get('loss_reg', 1e-8)),
            eps=config.get('loss_reg', 1e-8),
            dtype=torch.float64,
        )

        deeponet_branch_latent_dim = config.get(
            'branch_latent_dim',
            config.get('branch_hidden_dim', 256),
        )
        deeponet_common_kwargs = dict(
            num_fractures=config.get('num_fractures', 1),
            m_dim=self.m_dim,
            p_dim=config.get('p_dim', 64),
            activation=config.get('activation', 'gelu'),
            trunk_width=config.get('trunk_width', 256),
            trunk_depth=config.get('trunk_depth', 4),
            trunk_num_fourier_freqs=config.get('trunk_num_fourier_freqs', 6),
            trunk_use_film=config.get('trunk_use_film', True),
            branch_channels=config.get('branch_channels', [32, 64, 128, 128]),
            branch_blocks_per_stage=config.get('branch_blocks_per_stage', 2),
            branch_latent_dim=deeponet_branch_latent_dim,
            head_hidden_dim=config.get('head_hidden_dim', 256),
            head_depth=config.get('head_depth', 3),
            column_embedding_dim=config.get('column_embedding_dim', 32),
            enforce_zero_boundary=config.get('enforce_zero_boundary', True),
            branch_canonical_input_size=config.get('branch_canonical_input_size', 64),
        )

        if config["V_net"] == "fno":
            self.V_net = InterfaceFNO(
                num_fractures=config.get('num_fractures', 1),
                m_dim=self.m_dim,
                modes=config.get('modes', 12),
                width=config.get('width', 32),
                activation=config.get('activation', 'gelu'),
                enforce_zero_boundary=config.get('enforce_zero_boundary', True),
            )
        elif config["V_net"] == "deeponet":
            self.V_net = InterfaceDeepONet(
                use_column_embedding=config.get('use_column_embedding', True),
                **deeponet_common_kwargs,
            )
        else:
            raise ValueError(f"Unknown V_net type: {config['V_net']}. Expected 'fno' or 'deeponet'.")

        self.xavier_init = config.get('xavier_init', 0)
        if self.xavier_init > 0:
            self._reset_parameters()

    def _warn_once(self, flag, message):
        if flag in self._warning_flags:
            return
        warnings.warn(message, stacklevel=2)
        self._warning_flags.add(flag)

    def _canonical_reference_component_name(self, name):
        key = str(name).strip().lower()
        alias_map = {
            "constant": "constant",
            "low_freq": "low_frequency",
            "low_frequency": "low_frequency",
            "block": "block",
            "piecewise_constant": "block",
            "partition_of_unity": "block",
            "pou": "block",
            "high_contrast": "island",
            "island": "island",
            "smoothed_random": "smoothed_random",
            "random_smoothed": "smoothed_random",
            "random_smooth": "smoothed_random",
            "random_error": "smoothed_random",
            "smoothed_error": "smoothed_random",
            "asm_random": "smoothed_random",
        }
        if key not in alias_map:
            raise ValueError(
                "Unsupported reference basis component/type {!r}. Expected one of "
                "{'constant', 'low_frequency', 'block', 'island', 'smoothed_random', 'hybrid'}.".format(name)
            )
        return alias_map[key]

    def _normalize_reference_basis_components(self, components):
        """
        Normalize the optional hybrid component list to a canonical tuple.
        """
        if components is None:
            return ("constant", "low_frequency", "block", "island")

        if isinstance(components, str):
            raw_components = [
                item.strip() for item in components.replace("+", ",").split(",") if item.strip()
            ]
        else:
            try:
                raw_components = list(components)
            except TypeError as exc:
                raise ValueError(
                    "reference_basis_components must be a string or an iterable of component names"
                ) from exc

        normalized = []
        for item in raw_components:
            canonical = self._canonical_reference_component_name(item)
            if canonical not in normalized:
                normalized.append(canonical)

        if not normalized:
            raise ValueError("reference_basis_components cannot be empty")
        return tuple(normalized)

    def _resolve_reference_basis_components(self):
        """
        Return the canonical reference components used by the current basis_mode setup.
        """
        if self.reference_basis_type == "hybrid":
            return self.reference_basis_components
        return (self._canonical_reference_component_name(self.reference_basis_type),)

    def _align_model_inputs(self, a_field, phi_field):
        if a_field.dim() == 3:
            a_field = a_field.unsqueeze(1)
        if phi_field.dim() == 3:
            phi_field = phi_field.unsqueeze(1)
        return a_field, phi_field

    def _prepare_network_inputs(self, a_field, phi_field, X, Y):
        a_field, phi_field = self._align_model_inputs(a_field, phi_field)
        a_field_net = a_field.to(torch.float32)
        phi_field_net = phi_field.to(torch.float32)
        X_net = X.to(torch.float32)
        Y_net = Y.to(torch.float32)
        return a_field, phi_field, a_field_net, phi_field_net, X_net, Y_net

    def get_last_basis_debug(self):
        return dict(self._last_basis_debug)

    def _orthonormalize_transport_basis(self, basis, op_solver=None):
        if self.transport_metric == "l2":
            return euclidean_orthonormalize(basis)
        if op_solver is None:
            raise ValueError("transport_metric='a' requires op_solver for A-orthonormalization.")
        # Precondition rank-deficient / duplicated candidate columns with Euclidean QR first.
        # This keeps the l2 path unchanged while giving the A-metric path a numerically
        # stable full-column frame before generalized QR.
        basis = euclidean_orthonormalize(basis)
        AV = op_solver.apply_A(basis)
        Q, _, _ = a_orthonormalize_from_AV(basis, AV, reg=self.basis_orth_reg)
        return require_finite(Q, "Q")

    def _build_reference_basis_candidates(
        self,
        a_field,
        phi_field,
        X,
        Y,
        basis_dim,
        e_family=None,
        op_solver=None,
    ):
        batch_size = a_field.shape[0]
        num_nodes = int(a_field.shape[-2] * a_field.shape[-1])
        device = a_field.device
        dtype = torch.float64
        active_components = self._resolve_reference_basis_components()
        basis_op = op_solver

        x_coord = _flatten_coord_field(X, batch_size, num_nodes, device, dtype)
        y_coord = _flatten_coord_field(Y, batch_size, num_nodes, device, dtype)

        candidates = []
        if "constant" in active_components:
            candidates.append(
                _make_constant_reference_columns(batch_size, num_nodes, device, dtype)
            )

        if "low_frequency" in active_components:
            candidates.append(
                _make_low_frequency_reference_columns(
                    batch_size,
                    num_nodes,
                    x_coord,
                    y_coord,
                    num_columns=max(self.reference_num_low_freq, 0),
                    device=device,
                    dtype=dtype,
                    zero_boundary=self.reference_zero_boundary,
                )
            )

        if "block" in active_components:
            candidates.append(
                _make_block_reference_columns(
                    batch_size,
                    num_nodes,
                    num_blocks_1d=self.reference_block_count_1d,
                    device=device,
                    dtype=dtype,
                    zero_boundary=self.reference_zero_boundary,
                )
            )

        if "island" in active_components:
            candidates.append(
                _make_high_contrast_reference_columns(
                    a_field,
                    threshold=self.reference_high_contrast_threshold,
                    zero_boundary=self.reference_zero_boundary,
                ).to(device=device, dtype=dtype)
            )

        if "smoothed_random" in active_components:
            if e_family is None and basis_op is None:
                basis_op = self._build_basis_operator(
                    a_field,
                    build_local_solver=True,
                    phi_field=phi_field,
                )
            candidates.append(
                _make_smoothed_random_reference_columns(
                    batch_size,
                    num_nodes,
                    num_columns=max(basis_dim, 1),
                    device=device,
                    dtype=dtype,
                    e_family=e_family,
                    op_solver=basis_op,
                    smooth_steps=self.reference_random_smooth_steps,
                    smooth_omega=self.reference_random_smooth_omega,
                    zero_boundary=self.reference_zero_boundary,
                    seed=self.reference_random_seed,
                )
            )

        candidates = [col for col in candidates if col is not None and col.shape[-1] > 0]
        if not candidates:
            raise ValueError(
                "Could not build any reference basis columns for reference_basis_type={!r} "
                "with components={}.".format(
                    self.reference_basis_type,
                    active_components,
                )
            )

        reference = torch.cat(candidates, dim=-1)

        if reference.shape[-1] < basis_dim and e_family is None:
            if basis_op is None:
                basis_op = self._build_basis_operator(
                    a_field,
                    build_local_solver=True,
                    phi_field=phi_field,
                )
            extra_smoothed_random = _make_smoothed_random_reference_columns(
                batch_size,
                num_nodes,
                num_columns=basis_dim - reference.shape[-1],
                device=device,
                dtype=dtype,
                e_family=None,
                op_solver=basis_op,
                smooth_steps=self.reference_random_smooth_steps,
                smooth_omega=self.reference_random_smooth_omega,
                zero_boundary=self.reference_zero_boundary,
                seed=self.reference_random_seed,
            )
            reference = torch.cat([reference, extra_smoothed_random], dim=-1)

        if reference.shape[-1] < basis_dim:
            extra_low_freq = _make_low_frequency_reference_columns(
                batch_size,
                num_nodes,
                x_coord,
                y_coord,
                num_columns=max(basis_dim - reference.shape[-1] + self.reference_num_low_freq, 1),
                device=device,
                dtype=dtype,
                zero_boundary=self.reference_zero_boundary,
            )
            reference = torch.cat([reference, extra_low_freq], dim=-1)

        if reference.shape[-1] < basis_dim:
            extra_blocks = _make_block_reference_columns(
                batch_size,
                num_nodes,
                num_blocks_1d=max(
                    self.reference_block_count_1d + 1,
                    int(np.ceil(np.sqrt(basis_dim))),
                ),
                device=device,
                dtype=dtype,
                zero_boundary=self.reference_zero_boundary,
            )
            reference = torch.cat([reference, extra_blocks], dim=-1)

        if reference.shape[-1] < basis_dim:
            raise RuntimeError(
                f"Could not assemble enough reference columns for basis_dim={basis_dim}; "
                f"only built {reference.shape[-1]} candidate columns."
            )

        return reference

    def get_reference_basis(
        self,
        a_field,
        phi_field,
        X,
        Y,
        basis_dim,
        e_family=None,
        reference_basis=None,
        cache_key=None,
        op_solver=None,
    ):
        batch_size = a_field.shape[0]
        num_nodes = int(a_field.shape[-2] * a_field.shape[-1])
        device = a_field.device
        dtype = torch.float64
        active_components = self._resolve_reference_basis_components()
        allow_reference_cache = (
            self.reference_basis_cache_enabled and self.transport_metric == "l2"
        )
        if (
            cache_key is None
            and allow_reference_cache
            and set(active_components).issubset({"constant", "block", "low_frequency"})
        ):
            cache_key = (
                tuple(active_components),
                int(batch_size),
                int(num_nodes),
                int(basis_dim),
                int(self.reference_num_low_freq),
                int(self.reference_block_count_1d),
                bool(self.reference_zero_boundary),
            )

        if reference_basis is not None:
            reference = _broadcast_reference_basis(
                reference_basis,
                batch_size=batch_size,
                num_nodes=num_nodes,
                basis_dim=basis_dim,
                device=device,
                dtype=dtype,
            )
            return self._orthonormalize_transport_basis(reference, op_solver=op_solver)

        if (
            allow_reference_cache
            and cache_key is not None
            and cache_key in self._reference_basis_cache
        ):
            return _broadcast_reference_basis(
                self._reference_basis_cache[cache_key],
                batch_size=batch_size,
                num_nodes=num_nodes,
                basis_dim=basis_dim,
                device=device,
                dtype=dtype,
            )

        candidate_reference = self._build_reference_basis_candidates(
            a_field,
            phi_field,
            X,
            Y,
            basis_dim=basis_dim,
            e_family=e_family,
            op_solver=op_solver,
        )
        Q0_full = self._orthonormalize_transport_basis(
            candidate_reference,
            op_solver=op_solver,
        )
        Q0 = Q0_full[:, :, :basis_dim].contiguous()

        # A-metric reference bases depend on the current coefficient field, so reusing
        # cached Q0 across parameter samples is unsafe. We therefore cache only l2 bases.
        if allow_reference_cache and cache_key is not None:
            self._reference_basis_cache[cache_key] = Q0.detach().cpu()

        return Q0

    def project_to_tangent(self, Q0, W, op_solver=None):
        """
        Grassmann-style horizontal projection under the configured transport metric.
        For metric='l2': Xi = W - Q0 (Q0^T W)
        For metric='a' : Xi = W - Q0 (Q0^T A W)
        """
        Q0 = torch.nan_to_num(Q0, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float64)
        W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float64)
        if self.transport_metric == "l2":
            Xi = W - torch.matmul(Q0, torch.matmul(Q0.transpose(1, 2), W))
            return Xi

        if op_solver is None:
            raise ValueError("transport_metric='a' requires op_solver in project_to_tangent.")

        AW = op_solver.apply_A(W)
        Xi = W - torch.matmul(Q0, torch.matmul(Q0.transpose(1, 2), AW))
        return Xi

    def retract_qr(self, Q0, Xi, op_solver=None):
        """
        Use a QR-style retraction under the configured transport metric.
        When transport_metric='a', the existing 'qr' backend name is interpreted as
        generalized QR, implemented via A-orthonormalization of Q0 + Xi.
        """
        if self.transport_retraction != 'qr':
            raise RuntimeError(
                f"Unsupported transport retraction backend: {self.transport_retraction!r}"
            )

        Q_candidate = (Q0 + Xi).to(torch.float64)
        if self.transport_metric == "l2":
            Q, _ = torch.linalg.qr(Q_candidate, mode="reduced")
            return require_finite(Q, "Q")

        if op_solver is None:
            raise ValueError("transport_metric='a' requires op_solver in retract_qr.")

        AY = op_solver.apply_A(Q_candidate)
        Q, _, _ = a_orthonormalize_from_AV(
            Q_candidate,
            AY,
            reg=self.basis_orth_reg,
        )
        return require_finite(Q, "Q")

    def build_transport_basis(
        self,
        a_field,
        phi_field,
        X,
        Y,
        W,
        e_family=None,
        reference_basis=None,
        cache_key=None,
    ):
        W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float64)
        basis_dim = int(W.shape[-1])
        op_solver = (
            self._build_basis_operator(
                a_field,
                build_local_solver=self._reference_basis_needs_local_solver(
                    e_family=e_family,
                ),
                log_context="transport A-ops",
                phi_field=phi_field,
            )
            if self.transport_metric == "a" else None
        )

        Q0 = self.get_reference_basis(
            a_field,
            phi_field,
            X,
            Y,
            basis_dim=basis_dim,
            e_family=e_family,
            reference_basis=reference_basis,
            cache_key=cache_key,
            op_solver=op_solver,
        )
        Xi = self.project_to_tangent(Q0, W, op_solver=op_solver)
        Q = self.retract_qr(Q0, Xi, op_solver=op_solver)

        xi_fro = torch.linalg.matrix_norm(Xi, ord="fro", dim=(-2, -1))
        zero = W.new_zeros(W.shape[0])
        orth_error_l2 = _basis_orthogonality_error(Q, metric="l2")
        if self.transport_metric == "a":
            AQ = op_solver.apply_A(Q)
            AXi = op_solver.apply_A(Xi)
            orth_error_a = _basis_orthogonality_error(
                Q,
                metric="a",
                op_solver=op_solver,
                AQ=AQ,
            )
            tangent_violation = _max_metric_overlap(
                Q0,
                Xi,
                metric="a",
                op_solver=op_solver,
                AX=AXi,
            )
            subspace_shift = _subspace_shift_metric(
                Q0,
                Q,
                metric="a",
                op_solver=op_solver,
                AQ=AQ,
            )
            orth_error = orth_error_a
        else:
            orth_error_a = zero
            tangent_violation = _max_metric_overlap(Q0, Xi, metric="l2")
            subspace_shift = _subspace_shift_metric(Q0, Q, metric="l2")
            orth_error = orth_error_l2
        tan_reg = self.lambda_tan * torch.mean(xi_fro ** 2)

        return {
            "basis_mode": "transport",
            "transport_metric": self.transport_metric,
            "basis": Q,
            "W": W,
            "Q0": Q0,
            "Xi": Xi,
            "Q": Q,
            "xi_fro_norm": xi_fro,
            "xi_fro_mean": xi_fro.mean(),
            "subspace_shift": subspace_shift,
            "subspace_shift_mean": subspace_shift.mean(),
            "tangent_violation": tangent_violation,
            "tangent_violation_mean": tangent_violation.mean(),
            "orth_error": orth_error,
            "orth_error_mean": orth_error.mean(),
            "orth_error_l2": orth_error_l2,
            "orth_error_l2_mean": orth_error_l2.mean(),
            "orth_error_a": orth_error_a,
            "orth_error_a_mean": orth_error_a.mean(),
            "transport_regularization": tan_reg,
        }

    def _direct_basis_info(self, basis):
        zero = basis.new_tensor(0.0)
        return {
            "basis_mode": "direct",
            "transport_metric": self.transport_metric,
            "basis": basis,
            "W": basis,
            "Q0": None,
            "Xi": None,
            "Q": None,
            "xi_fro_norm": zero.expand(basis.shape[0]),
            "xi_fro_mean": zero,
            "subspace_shift": zero.expand(basis.shape[0]),
            "subspace_shift_mean": zero,
            "tangent_violation": zero.expand(basis.shape[0]),
            "tangent_violation_mean": zero,
            "orth_error": zero.expand(basis.shape[0]),
            "orth_error_mean": zero,
            "orth_error_l2": zero.expand(basis.shape[0]),
            "orth_error_l2_mean": zero,
            "orth_error_a": zero.expand(basis.shape[0]),
            "orth_error_a_mean": zero,
            "transport_regularization": zero,
        }

    def _cache_basis_debug(self, basis_info):
        debug = {
            "basis_mode": basis_info.get("basis_mode", self.basis_mode),
            "transport_metric": basis_info.get("transport_metric", self.transport_metric),
            "xi_fro_mean": float(basis_info["xi_fro_mean"].detach().item()),
            "subspace_shift_mean": float(basis_info["subspace_shift_mean"].detach().item()),
            "tangent_violation_mean": float(basis_info["tangent_violation_mean"].detach().item()),
            "orth_error_mean": float(basis_info["orth_error_mean"].detach().item()),
            "orth_error_l2_mean": float(basis_info["orth_error_l2_mean"].detach().item()),
            "orth_error_a_mean": float(basis_info["orth_error_a_mean"].detach().item()),
            "transport_regularization": float(
                basis_info["transport_regularization"].detach().item()
            ),
        }
        self._last_basis_debug = debug
        return debug

    def _merge_basis_stats(self, stats, basis_info):
        if stats is None:
            stats = {}
        stats = dict(stats)
        stats.update({
            "transport_is_active": float(basis_info.get("basis_mode") == "transport"),
            "transport_xi_fro_mean": basis_info["xi_fro_mean"].detach(),
            "transport_subspace_shift": basis_info["subspace_shift_mean"].detach(),
            "transport_tangent_violation": basis_info["tangent_violation_mean"].detach(),
            "transport_q_orth_error": basis_info["orth_error_mean"].detach(),
            "transport_q_orth_error_l2": basis_info["orth_error_l2_mean"].detach(),
            "transport_q_orth_error_a": basis_info["orth_error_a_mean"].detach(),
            "transport_regularization": basis_info["transport_regularization"].detach(),
        })
        return stats

    def _augment_loss_with_transport_regularization(self, loss_output, basis_info, return_stats):
        if isinstance(loss_output, tuple):
            loss, stats = loss_output
        else:
            loss, stats = loss_output, None

        total_loss = loss + basis_info["transport_regularization"]
        if not return_stats:
            return total_loss

        return total_loss, self._merge_basis_stats(stats, basis_info)

    def _augment_loss_stats_with_capture_family(self, loss_output, capture_family_info, return_stats):
        if not return_stats:
            return loss_output
        if isinstance(loss_output, tuple):
            loss, stats = loss_output
        else:
            loss, stats = loss_output, None
        stats = {} if stats is None else dict(stats)
        device = loss.device if torch.is_tensor(loss) else torch.device("cpu")
        dtype = loss.dtype if torch.is_tensor(loss) else torch.float64
        stats.update({
            "capture_family_is_white": torch.tensor(
                float(capture_family_info.get("capture_family") == "white"),
                device=device,
                dtype=dtype,
            ),
            "used_cached_whitened_family": torch.tensor(
                float(capture_family_info.get("used_cached_whitened_family", False)),
                device=device,
                dtype=dtype,
            ),
            "whiten_rank": torch.tensor(
                float(capture_family_info.get("whiten_rank", 0)),
                device=device,
                dtype=dtype,
            ),
            "cached_white_rank_or_cols": torch.tensor(
                float(capture_family_info.get("cached_white_rank_or_cols", 0)),
                device=device,
                dtype=dtype,
            ),
            "requested_whiten_rank": torch.tensor(
                float(capture_family_info.get("requested_whiten_rank", 0)),
                device=device,
                dtype=dtype,
            ),
        })
        return loss, stats

    def _predict_raw_basis(self, a_field, phi_field, X, Y, active_dim=None):
        a_field, phi_field, a_field_net, phi_field_net, X_net, Y_net = self._prepare_network_inputs(
            a_field,
            phi_field,
            X,
            Y,
        )
        V = self.V_net(a_field_net, X_net, Y_net, phi_field_net)
        if active_dim is not None:
            active_dim = int(active_dim)
            if active_dim < 1 or active_dim > self.m_dim:
                raise ValueError(
                    f"active_dim must satisfy 1 <= active_dim <= {self.m_dim}, got {active_dim}"
                )
            V = V[:, :, :active_dim]
        return a_field, phi_field, V.to(torch.float64)

    def forward_basis(
        self,
        a_field,
        phi_field,
        X,
        Y,
        active_dim=None,
        e_family=None,
        reference_basis=None,
        cache_key=None,
        return_info=False,
    ):
        a_field, phi_field, raw_basis = self._predict_raw_basis(
            a_field,
            phi_field,
            X,
            Y,
            active_dim=active_dim,
        )

        if self.basis_mode == "transport":
            basis_info = self.build_transport_basis(
                a_field,
                phi_field,
                X,
                Y,
                raw_basis,
                e_family=e_family,
                reference_basis=reference_basis,
                cache_key=cache_key,
            )
            basis = basis_info["basis"]
        else:
            basis = raw_basis
            basis_info = self._direct_basis_info(basis)

        self._cache_basis_debug(basis_info)
        if return_info:
            return basis, basis_info
        return basis

    def predict_basis(
        self,
        a_field,
        phi_field,
        X,
        Y,
        active_dim=None,
        e_family=None,
        reference_basis=None,
        cache_key=None,
        return_info=False,
    ):
        return self.forward_basis(
            a_field,
            phi_field,
            X,
            Y,
            active_dim=active_dim,
            e_family=e_family,
            reference_basis=reference_basis,
            cache_key=cache_key,
            return_info=return_info,
        )

    def _get_snapshot_l2_reference_basis(
        self,
        a_field,
        phi_field,
        X,
        Y,
        basis_dim,
        e_family=None,
        reference_basis=None,
        cache_key=None,
    ):
        batch_size = a_field.shape[0]
        num_nodes = int(a_field.shape[-2] * a_field.shape[-1])
        device = a_field.device
        dtype = torch.float64
        active_components = self._resolve_reference_basis_components()
        allow_reference_cache = self.reference_basis_cache_enabled

        if (
            cache_key is None
            and allow_reference_cache
            and set(active_components).issubset({"constant", "block", "low_frequency"})
        ):
            cache_key = (
                "snapshot_l2",
                tuple(active_components),
                int(batch_size),
                int(num_nodes),
                int(basis_dim),
                int(self.reference_num_low_freq),
                int(self.reference_block_count_1d),
                bool(self.reference_zero_boundary),
            )

        if reference_basis is not None:
            reference = _broadcast_reference_basis(
                reference_basis,
                batch_size=batch_size,
                num_nodes=num_nodes,
                basis_dim=basis_dim,
                device=device,
                dtype=dtype,
            )
            return euclidean_orthonormalize(reference)

        if (
            allow_reference_cache
            and cache_key is not None
            and cache_key in self._reference_basis_cache
        ):
            return _broadcast_reference_basis(
                self._reference_basis_cache[cache_key],
                batch_size=batch_size,
                num_nodes=num_nodes,
                basis_dim=basis_dim,
                device=device,
                dtype=dtype,
            )

        candidate_reference = self._build_reference_basis_candidates(
            a_field,
            phi_field,
            X,
            Y,
            basis_dim=basis_dim,
            e_family=e_family,
        )
        Q0_full = euclidean_orthonormalize(candidate_reference)
        Q0 = Q0_full[:, :, :basis_dim].contiguous()

        if allow_reference_cache and cache_key is not None:
            self._reference_basis_cache[cache_key] = Q0.detach().cpu()

        return Q0

    def _build_snapshot_transport_basis(
        self,
        a_field,
        phi_field,
        X,
        Y,
        W,
        e_family=None,
        reference_basis=None,
        cache_key=None,
    ):
        basis_dim = int(W.shape[-1])

        Q0 = self._get_snapshot_l2_reference_basis(
            a_field,
            phi_field,
            X,
            Y,
            basis_dim=basis_dim,
            e_family=e_family,
            reference_basis=reference_basis,
            cache_key=cache_key,
        )
        Xi = W - torch.matmul(Q0, torch.matmul(Q0.transpose(1, 2), W))
        Q = euclidean_orthonormalize(Q0 + Xi)

        xi_fro = torch.linalg.matrix_norm(Xi, ord="fro", dim=(-2, -1))
        zero = W.new_zeros(W.shape[0])
        zero_scalar = W.new_tensor(0.0)
        orth_error_l2 = _basis_orthogonality_error(Q, metric="l2")
        tangent_violation = _max_metric_overlap(Q0, Xi, metric="l2")
        subspace_shift = _subspace_shift_metric(Q0, Q, metric="l2")
        tan_reg = self.lambda_tan * torch.mean(xi_fro ** 2)

        return {
            "basis_mode": "transport",
            "transport_metric": "l2",
            "basis": Q,
            "W": W,
            "Q0": Q0,
            "Xi": Xi,
            "Q": Q,
            "xi_fro_norm": xi_fro,
            "xi_fro_mean": xi_fro.mean(),
            "subspace_shift": subspace_shift,
            "subspace_shift_mean": subspace_shift.mean(),
            "tangent_violation": tangent_violation,
            "tangent_violation_mean": tangent_violation.mean(),
            "orth_error": orth_error_l2,
            "orth_error_mean": orth_error_l2.mean(),
            "orth_error_l2": orth_error_l2,
            "orth_error_l2_mean": orth_error_l2.mean(),
            "orth_error_a": zero,
            "orth_error_a_mean": zero_scalar,
            "transport_regularization": tan_reg,
        }

    def _canonical_training_loss_type(self, loss_type=None):
        if loss_type is None:
            loss_type = self.loss_type
        return self.criterion._canonical_loss_type(loss_type)

    def _needs_training_basis_operator(self, loss_type=None):
        lt = self._canonical_training_loss_type(loss_type)
        if lt == "a":
            return True
        return not (
            self.basis_smooth_mode_train == "none" or self.basis_smooth_steps_train <= 0
        )

    def _training_operator_needs_local_solver(self):
        return (
            str(self.basis_smooth_mode_train).lower() == "asm"
            and int(self.basis_smooth_steps_train) > 0
        )

    def _reference_basis_needs_local_solver(self, e_family=None):
        active_components = self._resolve_reference_basis_components()
        return "smoothed_random" in active_components and e_family is None

    def _build_basis_operator(
        self,
        a_field,
        build_local_solver=None,
        log_context=None,
        phi_field=None,
    ):
        if build_local_solver is None:
            build_local_solver = self._training_operator_needs_local_solver()
        build_local_solver = bool(build_local_solver)
        if (
            log_context is not None
            and not build_local_solver
            and log_context not in self._lightweight_operator_log_contexts
        ):
            print(
                f"[Operator] CCFVM_ASM build_local_solver=False for {log_context}"
            )
            self._lightweight_operator_log_contexts.add(log_context)
        return CCFVM_ASM(
            a_field,
            k_1d=self.k_1d,
            target_overlap=0.3,
            build_local_solver=build_local_solver,
            subdomain_solver=self.subdomain_solver,
            jacobi_pcg_iters=self.jacobi_pcg_iters,
            phi=phi_field if self.transmissibility != "harmonic" else None,
            transmissibility=self.transmissibility,
        )

    def _maybe_build_training_basis_operator(
        self,
        a_field,
        loss_type=None,
        phi_field=None,
    ):
        if not self._needs_training_basis_operator(loss_type):
            return None
        return self._build_basis_operator(
            a_field,
            build_local_solver=self._training_operator_needs_local_solver(),
            log_context="training loss",
            phi_field=phi_field,
        )

    def a_orthonormalize_basis(
        self,
        a_field,
        V,
        reg=None,
        op_solver=None,
        phi_field=None,
    ):
        if reg is None:
            reg = self.basis_orth_reg
        if op_solver is None:
            op_solver = self._build_basis_operator(
                a_field,
                build_local_solver=False,
                log_context="training loss",
                phi_field=phi_field,
            )
        AV = op_solver.apply_A(V)
        Q, _, _ = a_orthonormalize_from_AV(V, AV, reg=reg)
        return Q

    def _smooth_basis_for_training(self, a_field, V, op_solver=None):
        if self.basis_smooth_mode_train == "none" or self.basis_smooth_steps_train <= 0:
            return V
        if op_solver is None:
            op_solver = self._build_basis_operator(
                a_field,
                build_local_solver=str(self.basis_smooth_mode_train).lower() == "asm",
                log_context="training loss",
            )
        return _smooth_basis_with_solver(
            op_solver=op_solver,
            a_field=a_field,
            V=V,
            mode=self.basis_smooth_mode_train,
            omega=self.basis_smooth_omega_train,
            num_steps=self.basis_smooth_steps_train,
        )

    def _prepare_metric_basis_for_training(self, a_field, V, op_solver=None):
        """
        A-loss 专用: basis smoothing + AV + A-orthonormalization.
        """
        if op_solver is None:
            op_solver = self._build_basis_operator(
                a_field,
                build_local_solver=self._training_operator_needs_local_solver(),
                log_context="training loss",
            )
        V_tilde = self._smooth_basis_for_training(a_field, V, op_solver=op_solver)
        AV = op_solver.apply_A(V_tilde)
        Q, AQ, A_rom = a_orthonormalize_from_AV(
            V_tilde,
            AV,
            reg=self.basis_orth_reg,
        )
        return V_tilde, AV, Q, AQ, A_rom

    def _prepare_l2_basis_for_training(self, a_field, V, op_solver=None):
        """
        l2-loss 专用: 仅做必要的 basis smoothing，然后做 Euclidean QR.
        """
        V_tilde = self._smooth_basis_for_training(a_field, V, op_solver=op_solver)
        Q = euclidean_orthonormalize(V_tilde)
        return V_tilde, Q

    def _prepare_basis_for_training(self, a_field, V, op_solver=None, loss_type=None):
        lt = self._canonical_training_loss_type(loss_type)
        if lt == "a":
            _, _, Q, _, _ = self._prepare_metric_basis_for_training(
                a_field,
                V,
                op_solver=op_solver,
            )
            return Q

        _, Q = self._prepare_l2_basis_for_training(
            a_field,
            V,
            op_solver=op_solver,
        )
        return Q

    def _prepare_a_capture_inputs(self, a_field, e_family, V_raw, op_solver=None):
        if op_solver is None:
            op_solver = self._build_basis_operator(
                a_field,
                build_local_solver=self._training_operator_needs_local_solver(),
                log_context="training loss",
            )

        _, _, Q, AQ, A_rom = self._prepare_metric_basis_for_training(
            a_field,
            V_raw,
            op_solver=op_solver,
        )

        expected_num_nodes = int(a_field.shape[-2] * a_field.shape[-1])
        E = normalize_snapshot_family(
            e_family,
            expected_num_nodes=expected_num_nodes,
        )
        E = torch.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float64)
        AE = op_solver.apply_A(E)

        return Q, AQ, A_rom, E, AE

    def _prepare_l2_capture_inputs(self, a_field, e_family, V_raw, op_solver=None):
        Q = self._prepare_basis_for_training(
            a_field,
            V_raw,
            op_solver=op_solver,
            loss_type="l2",
        )

        expected_num_nodes = int(a_field.shape[-2] * a_field.shape[-1])
        E = normalize_snapshot_family(
            e_family,
            expected_num_nodes=expected_num_nodes,
        )
        E = torch.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float64)

        return Q, E

    def _resolve_whiten_rank(self, E):
        available_rank = min(int(E.shape[1]), int(E.shape[2]))
        if available_rank <= 0:
            raise ValueError(
                f"Unable to whiten empty capture family with shape {tuple(E.shape)}"
            )
        if self.whiten_rank is None:
            return available_rank
        requested_rank = int(self.whiten_rank)
        if requested_rank <= 0:
            raise ValueError(f"whiten_rank must be positive or None, got {self.whiten_rank!r}")
        return min(requested_rank, available_rank)

    def _cached_white_family_matches(
        self,
        e_family_white,
        e_family_white_metric,
        e_family_white_rank=None,
        E_raw=None,
    ):
        if e_family_white is None or not self.use_cached_whitened_family:
            return False
        if e_family_white_metric is None:
            return False
        try:
            provided_metric = canonical_metric(e_family_white_metric, name="snapshot whitening metric")
        except ValueError:
            return False
        if provided_metric != self.whiten_metric:
            return False
        if e_family_white_rank is not None:
            if E_raw is not None:
                requested_rank = self._resolve_whiten_rank(E_raw)
            elif self.whiten_rank is not None:
                requested_rank = int(self.whiten_rank)
            else:
                requested_rank = None
            if requested_rank is not None and int(e_family_white_rank) != int(requested_rank):
                return False
        return True

    def _prepare_capture_family_for_loss(
        self,
        a_field,
        e_family,
        e_family_white=None,
        e_family_white_metric=None,
        e_family_white_rank=None,
        op_solver=None,
    ):
        expected_num_nodes = int(a_field.shape[-2] * a_field.shape[-1])
        E_raw = normalize_snapshot_family(
            e_family,
            expected_num_nodes=expected_num_nodes,
        )
        E_raw = torch.nan_to_num(E_raw, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float64)

        if self.capture_family == "raw":
            return E_raw, {
                "capture_family": "raw",
                "whiten_metric": None,
                "whiten_rank": 0,
                "used_cached_whitened_family": False,
                "cached_white_metric": None,
                "requested_whiten_metric": None,
                "cached_white_rank_or_cols": 0,
                "requested_whiten_rank": 0,
            }

        requested_whiten_rank = self._resolve_whiten_rank(E_raw)
        cached_white_rank_or_cols = (
            int(e_family_white.shape[-1])
            if torch.is_tensor(e_family_white) and e_family_white.dim() >= 1
            else 0
        )

        if self._cached_white_family_matches(
            e_family_white,
            e_family_white_metric,
            e_family_white_rank=e_family_white_rank,
            E_raw=E_raw,
        ):
            E_white = normalize_snapshot_family(
                e_family_white,
                expected_num_nodes=expected_num_nodes,
            )
            E_white = torch.nan_to_num(
                E_white,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).to(torch.float64)
            return E_white, {
                "capture_family": "white",
                "whiten_metric": self.whiten_metric,
                "whiten_rank": int(E_white.shape[-1]),
                "used_cached_whitened_family": True,
                "cached_white_metric": canonical_metric(e_family_white_metric, name="snapshot whitening metric"),
                "requested_whiten_metric": self.whiten_metric,
                "cached_white_rank_or_cols": int(E_white.shape[-1]),
                "requested_whiten_rank": int(requested_whiten_rank),
            }

        if e_family_white is not None and self.use_cached_whitened_family:
            self._warn_once(
                "white_family_metadata_mismatch",
                "Cached e_family_white metadata does not match the requested snapshot "
                f"whiten_metric={self.whiten_metric!r}; recomputing whitening online.",
            )

        whiten_rank = requested_whiten_rank
        if self.whiten_metric == "l2":
            E_white = coarse_loss.compute_metric_whitened_snapshot_family(
                E_raw,
                rank=whiten_rank,
                metric="l2",
                reg=self.basis_orth_reg,
                output_layout="family",
            )
        else:
            if op_solver is None:
                op_solver = self._build_basis_operator(
                    a_field,
                    build_local_solver=False,
                    log_context="training loss",
                )
            AE_raw = op_solver.apply_A(E_raw)
            E_white = coarse_loss.compute_metric_whitened_snapshot_family(
                E_raw,
                rank=whiten_rank,
                metric="a",
                AE=AE_raw,
                reg=self.basis_orth_reg,
                output_layout="family",
            )

        try:
            cached_white_metric = (
                None if e_family_white_metric is None
                else canonical_metric(e_family_white_metric, name="snapshot whitening metric")
            )
        except ValueError:
            cached_white_metric = None

        return require_finite(E_white, "E_white"), {
            "capture_family": "white",
            "whiten_metric": self.whiten_metric,
            "whiten_rank": int(whiten_rank),
            "used_cached_whitened_family": False,
            "cached_white_metric": cached_white_metric,
            "requested_whiten_metric": self.whiten_metric,
            "cached_white_rank_or_cols": int(cached_white_rank_or_cols),
            "requested_whiten_rank": int(requested_whiten_rank),
        }

    def _prepare_stage2_a_capture_inputs(self, *args, **kwargs):
        self._warn_once(
            "deprecated_prepare_stage2_a_capture_inputs",
            "_prepare_stage2_a_capture_inputs is deprecated; use _prepare_a_capture_inputs.",
        )
        return self._prepare_a_capture_inputs(*args, **kwargs)

    def _prepare_stage2_l2_capture_inputs(self, *args, **kwargs):
        self._warn_once(
            "deprecated_prepare_stage2_l2_capture_inputs",
            "_prepare_stage2_l2_capture_inputs is deprecated; use _prepare_l2_capture_inputs.",
        )
        return self._prepare_l2_capture_inputs(*args, **kwargs)

    def _resolve_stage2_whiten_rank(self, *args, **kwargs):
        self._warn_once(
            "deprecated_resolve_stage2_whiten_rank",
            "_resolve_stage2_whiten_rank is deprecated; use _resolve_whiten_rank.",
        )
        return self._resolve_whiten_rank(*args, **kwargs)

    def _prepare_stage2_capture_family(self, *args, **kwargs):
        self._warn_once(
            "deprecated_prepare_stage2_capture_family",
            "_prepare_stage2_capture_family is deprecated; use _prepare_capture_family_for_loss.",
        )
        return self._prepare_capture_family_for_loss(*args, **kwargs)

    def _snapshotize_subspace_loss_output(self, loss_output, return_stats):
        return loss_output

    def _forward_snapshot_subspace_supervision(
        self,
        a_field,
        phi_field,
        X,
        Y,
        e_family,
        e_family_white=None,
        e_family_white_metric=None,
        e_family_white_rank=None,
        svd_basis=None,
        return_stats=False,
    ):
        del e_family_white_rank
        a_field, phi_field = self._align_model_inputs(a_field, phi_field)
        _, _, V_raw = self._predict_raw_basis(
            a_field,
            phi_field,
            X,
            Y,
        )
        V_raw = torch.nan_to_num(V_raw, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float64)
        subspace_op = None
        if (
            self.snapshot_subspace_criterion.metric == "a"
            or (
                self.snapshot_subspace_criterion.capture_whiten
                and self.snapshot_subspace_criterion._resolve_capture_whiten_metric() == "a"
            )
        ):
            subspace_op = self._build_basis_operator(
                a_field,
                build_local_solver=False,
                log_context="snapshot subspace loss",
                phi_field=phi_field,
            )
        if self.basis_mode == "transport":
            basis_info = self._build_snapshot_transport_basis(
                a_field,
                phi_field,
                X,
                Y,
                V_raw,
                e_family=e_family,
            )
            loss_output = self.snapshot_subspace_criterion(
                Q=basis_info["basis"],
                e_family=e_family,
                e_family_white=e_family_white,
                e_family_white_metric=e_family_white_metric,
                a_field=a_field,
                svd_basis=svd_basis,
                op_solver=subspace_op,
                return_stats=return_stats,
            )
        else:
            basis_info = self._direct_basis_info(V_raw)
            loss_output = self.snapshot_subspace_criterion(
                V_raw=V_raw,
                e_family=e_family,
                e_family_white=e_family_white,
                e_family_white_metric=e_family_white_metric,
                a_field=a_field,
                svd_basis=svd_basis,
                op_solver=subspace_op,
                return_stats=return_stats,
            )
        self._cache_basis_debug(basis_info)
        loss_output = self._snapshotize_subspace_loss_output(
            loss_output,
            return_stats=return_stats,
        )
        return self._augment_loss_with_transport_regularization(
            loss_output,
            basis_info,
            return_stats=return_stats,
        )

    def _forward_snapshot_capture(
        self,
        a_field,
        phi_field,
        X,
        Y,
        e_family,
        e_family_white=None,
        e_family_white_metric=None,
        e_family_white_rank=None,
        return_stats=False,
    ):
        a_field, phi_field = self._align_model_inputs(a_field, phi_field)
        V_raw, basis_info = self.forward_basis(
            a_field,
            phi_field,
            X,
            Y,
            e_family=e_family,
            return_info=True,
        )

        training_needs_operator = (
            self.capture_metric == "a"
            or (self.capture_family == "white" and self.whiten_metric == "a")
            or self._needs_training_basis_operator(self.capture_metric)
        )
        op_solver = None
        if training_needs_operator:
            op_solver = self._build_basis_operator(
                a_field,
                build_local_solver=self._training_operator_needs_local_solver(),
                log_context="training loss",
                phi_field=phi_field,
            )

        E_capture, capture_family_info = self._prepare_capture_family_for_loss(
            a_field,
            e_family,
            e_family_white=e_family_white,
            e_family_white_metric=e_family_white_metric,
            e_family_white_rank=e_family_white_rank,
            op_solver=op_solver,
        )

        if self.criterion.loss_type == "a":
            Q, AQ, _, E, AE = self._prepare_a_capture_inputs(
                a_field,
                E_capture,
                V_raw,
                op_solver=op_solver,
            )
            loss_output = self.criterion(
                V=None,
                e_family=E,
                a_field=a_field,
                Q=Q,
                AQ=AQ,
                ME=AE,
                return_stats=return_stats,
            )
            loss_output = self._augment_loss_stats_with_capture_family(
                loss_output,
                capture_family_info,
                return_stats=return_stats,
            )
            return self._augment_loss_with_transport_regularization(
                loss_output,
                basis_info,
                return_stats=return_stats,
            )

        if self.criterion.loss_type == "l2":
            Q, E = self._prepare_l2_capture_inputs(
                a_field,
                E_capture,
                V_raw,
                op_solver=op_solver,
            )
            loss_output = self.criterion(
                V=None,
                e_family=E,
                a_field=a_field,
                Q=Q,
                ME=E,
                return_stats=return_stats,
            )
            loss_output = self._augment_loss_stats_with_capture_family(
                loss_output,
                capture_family_info,
                return_stats=return_stats,
            )
            return self._augment_loss_with_transport_regularization(
                loss_output,
                basis_info,
                return_stats=return_stats,
            )

        raise RuntimeError(
            f"Unsupported snapshot capture loss_type={self.criterion.loss_type!r}"
        )

    def forward(
        self,
        a_field,
        phi_field,
        X,
        Y,
        e_family,
        epoch,
        e_family_white=None,
        e_family_white_metric=None,
        e_family_white_rank=None,
        svd_basis=None,
        return_stats=False,
        training_stage=None,
    ):
        """
        前向传播：计算基于 smoothed error snapshots 的 capture/subspace loss。
        """
        del epoch

        if training_stage is not None:
            self._warn_once(
                "deprecated_training_stage_forward",
                "training_stage is deprecated and ignored. Snapshot training is "
                "controlled by loss_objective/capture_family/capture_metric.",
            )

        if self.loss_objective == "capture":
            return self._forward_snapshot_capture(
                a_field,
                phi_field,
                X,
                Y,
                e_family,
                e_family_white=e_family_white,
                e_family_white_metric=e_family_white_metric,
                e_family_white_rank=e_family_white_rank,
                return_stats=return_stats,
            )
        if self.loss_objective in {"svd", "nlss"}:
            return self._forward_snapshot_subspace_supervision(
                a_field,
                phi_field,
                X,
                Y,
                e_family,
                e_family_white=e_family_white,
                e_family_white_metric=e_family_white_metric,
                e_family_white_rank=e_family_white_rank,
                svd_basis=svd_basis,
                return_stats=return_stats,
            )

        raise ValueError(
            "loss_objective must be one of {'capture', 'svd', 'nlss'}, "
            f"got {self.loss_objective!r}"
        )

    def smooth_basis(
        self,
        a_field,
        V,
        mode="asm",
        omega=0.6,
        num_steps=1,
        phi_field=None,
    ):
        """
        训练端对网络输出 basis 的后处理。
        """
        op_solver = self._build_basis_operator(
            a_field,
            build_local_solver=(str(mode).lower() == "asm" and int(num_steps) > 0),
            phi_field=phi_field,
        )

        return _smooth_basis_with_solver(
            op_solver=op_solver,
            a_field=a_field,
            V=V,
            mode=mode,
            omega=omega,
            num_steps=num_steps,
            k_1d=self.k_1d,
            target_overlap=0.3,
        )

    def _reset_parameters(self):
        """Xavier 初始化"""
        for param in self.V_net.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=self.xavier_init)
            else:
                constant_(param, 1e-2)
