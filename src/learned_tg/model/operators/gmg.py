"""
Geometric multigrid (GMG) hierarchy for the interface FVM operator.

Restriction (_interface_gmg_down) and prolongation (_interface_gmg_up) use a
3×3 bilinear kernel with stride-2 convolution — consistent with the online
learned two-grid smoother in model/interface.py.

NOTE: data/gen_data_interface.py has its own _gmg_down/_gmg_up that use
avg_pool2d / F.interpolate (different numerics).  Those are intentionally
kept separate and are NOT aliases of these functions.
"""

import torch
import torch.nn.functional as F


def _safe_interface_gmg_subdomain_count(n: int, requested: int) -> int:
    n = int(n)
    requested = int(requested)
    return max(1, min(requested, max(1, n - 1)))


def _interface_gmg_down(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.dim() != 4:
        raise ValueError(f"GMG restriction expects shape (B,C,n,n), got {tuple(matrix.shape)}")
    if min(matrix.shape[-2:]) <= 2:
        return matrix
    channels = int(matrix.shape[1])
    base_kernel = torch.tensor(
        [[[[1, 2, 1],
           [2, 4, 2],
           [1, 2, 1]]]],
        device=matrix.device,
        dtype=matrix.dtype,
    ) / 4.0
    kernel = base_kernel.repeat(channels, 1, 1, 1)
    return F.conv2d(matrix, kernel, stride=2, padding=1, groups=channels)


def _interface_gmg_up(matrix: torch.Tensor, target_shape) -> torch.Tensor:
    if matrix.dim() != 4:
        raise ValueError(f"GMG prolongation expects shape (B,C,n,n), got {tuple(matrix.shape)}")
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
    target_n, target_m = int(target_shape[0]), int(target_shape[1])
    return out[..., :target_n, :target_m]


class InterfaceGMGHierarchy:
    """
    Standard geometric multigrid V-cycle for the interface FVM operator.

    Uses bilinear-kernel restriction/prolongation consistent with the online
    two-grid solver.  Build with a CCFVM_ASM fine operator.
    """

    def __init__(
        self,
        fine_op,
        num_levels=None,
        min_coarse_size=4,
        smoother="jacobi",
        pre_smooth_steps=2,
        post_smooth_steps=2,
        omega=0.8,
        coarse_solve="direct",
    ):
        self.fine_op = fine_op
        self.device = fine_op.device
        self.dtype = fine_op.dtype
        self.k_1d = int(getattr(fine_op, "k_1d", 10))
        self.target_overlap = float(getattr(fine_op, "target_overlap", 0.3))
        self.transmissibility = str(
            getattr(fine_op, "transmissibility", "harmonic")
        ).lower()
        self.subdomain_solver = str(
            getattr(fine_op, "subdomain_solver", "cholesky")
        ).lower()
        self.jacobi_pcg_iters = int(getattr(fine_op, "jacobi_pcg_iters", 10))
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
            raise ValueError(f"gmg_num_levels must be positive when provided, got {num_levels}")
        self.num_levels_requested = None if num_levels is None else int(num_levels)

        # Derive operator class from fine_op to avoid importing CCFVM_ASM here
        # (prevents circular imports when model.interface imports this module).
        self._CCFVM_ASM = type(fine_op)
        build_local_solver = self.smoother == "asm"
        if build_local_solver and not (
            fine_op.build_local_solver and fine_op.subdomain_solver == "cholesky"
        ):
            level0 = self._make_operator(
                fine_op.a,
                phi_field=fine_op.phi,
                build_local_solver=True,
            )
        else:
            level0 = fine_op
        self.ops = [level0]
        self._build_levels(build_local_solver=build_local_solver)

        self.coarse_direct_matrix = None
        coarse_op = self.ops[-1]
        if self.coarse_solve == "direct" and coarse_op.n * coarse_op.n <= 1024:
            self.coarse_direct_matrix = self._assemble_dense_matrix(coarse_op)

    def _make_operator(self, a_field, *, phi_field=None, build_local_solver):
        n = int(a_field.shape[-1])
        return self._CCFVM_ASM(
            a_field.to(device=self.device, dtype=self.dtype),
            k_1d=_safe_interface_gmg_subdomain_count(n, self.k_1d),
            target_overlap=self.target_overlap,
            build_local_solver=bool(build_local_solver),
            subdomain_solver=self.subdomain_solver,
            jacobi_pcg_iters=self.jacobi_pcg_iters,
            phi=(
                None
                if phi_field is None
                else phi_field.to(device=self.device, dtype=self.dtype)
            ),
            transmissibility=self.transmissibility,
        )

    def _build_levels(self, *, build_local_solver):
        while True:
            if self.num_levels_requested is not None and len(self.ops) >= self.num_levels_requested:
                break
            current = self.ops[-1]
            if current.n <= self.min_coarse_size:
                break
            a_c = _interface_gmg_down(current.a).clamp_min(1e-30)
            phi_c = None if current.phi is None else _interface_gmg_down(current.phi)
            next_n = int(a_c.shape[-1])
            if next_n == current.n or next_n < self.min_coarse_size:
                break
            self.ops.append(
                self._make_operator(
                    a_c,
                    phi_field=phi_c,
                    build_local_solver=build_local_solver,
                )
            )

    def _assemble_dense_matrix(self, op):
        num_nodes = int(op.n * op.n)
        eye = torch.eye(num_nodes, device=op.device, dtype=op.dtype)
        basis = eye.reshape(num_nodes, op.n, op.n).unsqueeze(0)
        basis = basis.expand(op.a.shape[0], num_nodes, op.n, op.n).contiguous()
        columns = op.forward(basis).reshape(op.a.shape[0], num_nodes, num_nodes)
        return columns.transpose(1, 2).contiguous()

    def _smooth(self, op, x, b, steps, omega, smoother):
        if int(steps) <= 0:
            return x.to(device=op.device, dtype=op.dtype)
        x = x.to(device=op.device, dtype=op.dtype)
        b = b.to(device=op.device, dtype=op.dtype)
        smoother = str(smoother).lower()
        if smoother == "jacobi":
            diag = op.sc.abs().clamp_min(1e-30)
            for _ in range(int(steps)):
                r = b - op.forward(x)
                x = x + float(omega) * (r / diag)
            return x
        if smoother == "asm":
            return op.asm_smooth(x, b, M_smooth=int(steps), omega=float(omega))
        raise ValueError(f"Unknown GMG smoother={smoother!r}")

    def _coarse_solve(self, op, b):
        if self.coarse_direct_matrix is None:
            return self._smooth(
                op,
                torch.zeros_like(b),
                b,
                steps=20,
                omega=self.omega,
                smoother="jacobi",
            )

        batch_size, channels, _, _ = b.shape
        matrix = self.coarse_direct_matrix
        if matrix.shape[0] == 1 and batch_size > 1:
            matrix = matrix.expand(batch_size, -1, -1)
        elif matrix.shape[0] != batch_size:
            raise ValueError(
                "GMG coarse solve batch mismatch: "
                f"matrix batch={matrix.shape[0]}, rhs batch={batch_size}"
            )
        rhs = b.reshape(batch_size, channels, op.n * op.n).transpose(1, 2).contiguous()
        try:
            sol = torch.linalg.solve(matrix, rhs)
        except RuntimeError:
            sol = torch.matmul(torch.linalg.pinv(matrix), rhs)
        return sol.transpose(1, 2).reshape(batch_size, channels, op.n, op.n)

    def v_cycle(self, level, x, b):
        op = self.ops[int(level)]
        x = x.to(device=op.device, dtype=op.dtype)
        b = b.to(device=op.device, dtype=op.dtype)
        if int(level) == len(self.ops) - 1:
            if self.coarse_solve == "direct":
                return self._coarse_solve(op, b)
            return self._smooth(
                op,
                x,
                b,
                steps=20,
                omega=self.omega,
                smoother="jacobi",
            )

        x = self._smooth(
            op,
            x,
            b,
            steps=self.pre_smooth_steps,
            omega=self.omega,
            smoother=self.smoother,
        )
        r = b - op.forward(x)
        r_c = _interface_gmg_down(r)
        e_c = self.v_cycle(level + 1, torch.zeros_like(r_c), r_c)
        e_f = _interface_gmg_up(e_c, target_shape=(op.n, op.n))
        x = x + e_f
        x = self._smooth(
            op,
            x,
            b,
            steps=self.post_smooth_steps,
            omega=self.omega,
            smoother=self.smoother,
        )
        return x

    def apply(self, r):
        op = self.ops[0]
        r = r.to(device=op.device, dtype=op.dtype)
        return self.v_cycle(0, torch.zeros_like(r), r)
