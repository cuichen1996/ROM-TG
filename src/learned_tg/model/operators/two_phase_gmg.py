from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from learned_tg.model.operators.two_phase_fvm import MobilityFVMOperatorASM


def gmg_down(matrix: torch.Tensor) -> torch.Tensor:
    """
    Full-weighting grid restriction used by the standard GMG baseline.

    Input and output are cell-centered tensors with shape (B, C, ny, nx).
    """
    if matrix.dim() != 4:
        raise ValueError(f"gmg_down expects shape (B,C,ny,nx), got {tuple(matrix.shape)}")
    channels = int(matrix.shape[1])
    base_kernel = torch.tensor(
        [[[[1, 2, 1],
           [2, 4, 2],
           [1, 2, 1]]]],
        device=matrix.device,
        dtype=matrix.dtype,
    ) / 16.0
    kernel = base_kernel.repeat(channels, 1, 1, 1)
    return F.conv2d(matrix, kernel, stride=2, padding=1, groups=channels)


def gmg_up(
    matrix: torch.Tensor,
    target_shape: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Bilinear-like GMG prolongation using the requested transposed-convolution kernel.

    Input and output are cell-centered tensors with shape (B, C, ny, nx).
    """
    if matrix.dim() != 4:
        raise ValueError(f"gmg_up expects shape (B,C,ny,nx), got {tuple(matrix.shape)}")
    channels = int(matrix.shape[1])
    base_kernel = torch.tensor(
        [[[[1, 2, 1],
           [2, 4, 2],
           [1, 2, 1]]]],
        device=matrix.device,
        dtype=matrix.dtype,
    ) / 4.0
    kernel = base_kernel.repeat(channels, 1, 1, 1)
    out = F.conv_transpose2d(
        matrix,
        kernel,
        stride=2,
        padding=1,
        output_padding=1,
        groups=channels,
    )
    if target_shape is not None:
        target_ny, target_nx = int(target_shape[0]), int(target_shape[1])
        out = out[..., :target_ny, :target_nx]
    return out


class StandardGMGHierarchy:
    """
    Standard cell-centered geometric multigrid baseline for MobilityFVMOperatorASM.

    Coarse operators are rediscretized from restricted mobility fields. This is
    intentionally a plain geometric hierarchy, without coefficient-aware
    interpolation, so it can serve as a baseline for high-contrast interface cases.
    """

    def __init__(
        self,
        fine_op: MobilityFVMOperatorASM,
        num_levels: Optional[int] = None,
        min_coarse_size: int = 4,
        smoother: str = "jacobi",
        pre_smooth_steps: int = 2,
        post_smooth_steps: int = 2,
        omega: float = 0.8,
        coarse_solve: str = "direct",
    ):
        self.fine_op = fine_op
        self.device = fine_op.device
        self.dtype = fine_op.dtype
        self.Lx = float(fine_op.hx * fine_op.nx)
        self.Ly = float(fine_op.hy * fine_op.ny)
        self.p_left = float(fine_op.p_left)
        self.p_right = float(fine_op.p_right)
        self.target_overlap = float(getattr(fine_op, "target_overlap", 0.3))
        self.asm_reg = float(fine_op.asm_reg)
        self.transmissibility = str(getattr(fine_op, "transmissibility", "harmonic"))

        self.smoother = str(smoother).lower()
        if self.smoother not in {"jacobi", "asm"}:
            raise ValueError(f"GMG smoother must be 'jacobi' or 'asm', got {smoother!r}")
        self.pre_smooth_steps = int(pre_smooth_steps)
        self.post_smooth_steps = int(post_smooth_steps)
        self.omega = float(omega)
        self.coarse_solve = str(coarse_solve).lower()
        if self.coarse_solve not in {"direct", "jacobi"}:
            raise ValueError(
                f"GMG coarse_solve must be 'direct' or 'jacobi', got {coarse_solve!r}"
            )

        self.min_coarse_size = max(2, int(min_coarse_size))
        if num_levels is not None and int(num_levels) <= 0:
            raise ValueError(f"num_levels must be positive when provided, got {num_levels}")
        self.num_levels_requested = None if num_levels is None else int(num_levels)

        build_local_solver = self.smoother == "asm"
        if build_local_solver and not (
            fine_op.build_local_solver and fine_op.subdomain_solver == "cholesky"
        ):
            level0 = self._make_operator(
                fine_op.lam,
                phi=getattr(fine_op, "phi", None),
                build_local_solver=True,
            )
        else:
            level0 = fine_op

        self.ops: List[MobilityFVMOperatorASM] = [level0]
        self._build_levels(build_local_solver=build_local_solver)

        self.coarse_direct_matrix: Optional[torch.Tensor] = None
        coarse_op = self.ops[-1]
        if self.coarse_solve == "direct" and coarse_op.N <= 1024:
            self.coarse_direct_matrix = self._assemble_dense_matrix(coarse_op)

    @staticmethod
    def _safe_subdomain_count(n: int, requested: int) -> int:
        n = int(n)
        requested = int(requested)
        return max(1, min(requested, max(1, n // 2)))

    def _make_operator(
        self,
        lam: torch.Tensor,
        *,
        phi: Optional[torch.Tensor] = None,
        build_local_solver: bool,
    ) -> MobilityFVMOperatorASM:
        ny = int(lam.shape[-2])
        nx = int(lam.shape[-1])
        return MobilityFVMOperatorASM(
            lam=lam.to(device=self.device, dtype=self.dtype),
            hx=self.Lx / float(nx),
            hy=self.Ly / float(ny),
            p_left=self.p_left,
            p_right=self.p_right,
            kx_sub=self._safe_subdomain_count(nx, self.fine_op.kx_sub),
            ky_sub=self._safe_subdomain_count(ny, self.fine_op.ky_sub),
            target_overlap=self.target_overlap,
            asm_reg=self.asm_reg,
            build_local_solver=bool(build_local_solver),
            phi=phi,
            transmissibility=self.transmissibility,
        )

    def _build_levels(self, *, build_local_solver: bool) -> None:
        while True:
            if (
                self.num_levels_requested is not None
                and len(self.ops) >= self.num_levels_requested
            ):
                break

            current = self.ops[-1]
            if min(current.ny, current.nx) <= self.min_coarse_size:
                break

            lam_c = gmg_down(current.lam).clamp_min(1e-30)
            phi_c = None
            if self.transmissibility != "harmonic":
                if current.phi is None:
                    raise ValueError(
                        "interface-aware GMG requires phi on every hierarchy level"
                    )
                phi_c = gmg_down(current.phi)
            next_ny = int(lam_c.shape[-2])
            next_nx = int(lam_c.shape[-1])
            if (next_ny, next_nx) == (current.ny, current.nx):
                break
            if min(next_ny, next_nx) < self.min_coarse_size:
                break

            self.ops.append(
                self._make_operator(
                    lam_c,
                    phi=phi_c,
                    build_local_solver=build_local_solver,
                )
            )

    def _assemble_dense_matrix(self, op: MobilityFVMOperatorASM) -> torch.Tensor:
        num_nodes = int(op.N)
        eye = torch.eye(num_nodes, device=op.device, dtype=op.dtype)
        basis = eye.reshape(num_nodes, op.ny, op.nx).unsqueeze(0)
        basis = basis.expand(op.B, num_nodes, op.ny, op.nx).contiguous()
        columns = op.forward(basis).reshape(op.B, num_nodes, num_nodes)
        return columns.transpose(1, 2).contiguous()

    def _coarse_solve_direct(
        self,
        op: MobilityFVMOperatorASM,
        b: torch.Tensor,
    ) -> torch.Tensor:
        if self.coarse_direct_matrix is None:
            return self.smooth(
                op,
                torch.zeros_like(b),
                b,
                steps=20,
                omega=self.omega,
                smoother="jacobi",
            )

        b4, _ = op._to_4d(b)
        batch_size, channels, _, _ = b4.shape
        matrix = self.coarse_direct_matrix
        if matrix.shape[0] == 1 and batch_size > 1:
            matrix = matrix.expand(batch_size, -1, -1)
        elif matrix.shape[0] != batch_size:
            raise ValueError(
                "GMG coarse solve batch mismatch: "
                f"matrix batch={matrix.shape[0]}, rhs batch={batch_size}"
            )

        rhs = b4.reshape(batch_size, channels, op.N).transpose(1, 2).contiguous()
        try:
            sol = torch.linalg.solve(matrix, rhs)
        except RuntimeError:
            sol = torch.matmul(torch.linalg.pinv(matrix), rhs)
        return sol.transpose(1, 2).reshape(batch_size, channels, op.ny, op.nx)

    def restrict(self, x_f: torch.Tensor) -> torch.Tensor:
        return gmg_down(x_f)

    def prolong(
        self,
        x_c: torch.Tensor,
        target_shape: Tuple[int, int],
    ) -> torch.Tensor:
        return gmg_up(x_c, target_shape=target_shape)

    def smooth(
        self,
        op: MobilityFVMOperatorASM,
        x: torch.Tensor,
        b: torch.Tensor,
        steps: int,
        omega: float,
        smoother: str,
    ) -> torch.Tensor:
        smoother = str(smoother).lower()
        if int(steps) <= 0:
            return x.to(device=op.device, dtype=op.dtype)
        x4, _ = op._to_4d(x)
        b4, _ = op._to_4d(b)
        if smoother == "jacobi":
            for _ in range(int(steps)):
                r = b4 - op.forward(x4)
                x4 = x4 + float(omega) * op.apply_jacobi(r)
            return x4
        if smoother == "asm":
            return op.asm_smooth(x4, b4, steps=int(steps), omega=float(omega))
        raise ValueError(f"Unknown GMG smoother={smoother!r}")

    def v_cycle(self, level: int, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        level = int(level)
        op = self.ops[level]
        x4, _ = op._to_4d(x)
        b4, _ = op._to_4d(b)

        if level == len(self.ops) - 1:
            if self.coarse_solve == "direct":
                return self._coarse_solve_direct(op, b4)
            return self.smooth(
                op,
                x4,
                b4,
                steps=20,
                omega=self.omega,
                smoother="jacobi",
            )

        x4 = self.smooth(
            op,
            x4,
            b4,
            steps=self.pre_smooth_steps,
            omega=self.omega,
            smoother=self.smoother,
        )
        r_f = b4 - op.forward(x4)
        r_c = self.restrict(r_f)
        e_c0 = torch.zeros_like(r_c)
        e_c = self.v_cycle(level + 1, e_c0, r_c)
        e_f = self.prolong(e_c, target_shape=(op.ny, op.nx))
        x4 = x4 + e_f
        x4 = self.smooth(
            op,
            x4,
            b4,
            steps=self.post_smooth_steps,
            omega=self.omega,
            smoother=self.smoother,
        )
        return x4

    def apply(self, r: torch.Tensor) -> torch.Tensor:
        op = self.ops[0]
        r4, original_shape = op._to_4d(r)
        z0 = torch.zeros_like(r4)
        z4 = self.v_cycle(0, z0, r4)
        return op._from_4d(z4, original_shape)
