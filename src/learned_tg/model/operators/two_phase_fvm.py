import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from learned_tg.model.operators.blocktri_cholesky import blocktri_cholesky_factor, blocktri_cholesky_solve
from learned_tg.model.operators.mobility_coefficients import build_mobility_coefficients, harmonic_mean


def _compute_subdomain_starts_1d(n: int, s: int, k: int) -> list:
    """Return k integer start positions covering [0, n-s] with first=0, last=n-s."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k == 1:
        return [max(0, (n - s) // 2)]
    if k == 2:
        return [0, n - s]
    starts = [round(i * (n - s) / (k - 1)) for i in range(k)]
    starts[0] = 0
    starts[-1] = n - s
    for i in range(1, len(starts)):
        if starts[i] < starts[i - 1]:
            raise ValueError(f"non-monotonic starts at n={n}, s={s}, k={k}")
    return starts


def get_fixed_subdomains_params_1d(n: int, k: int, target_overlap: float = 0.3):
    """Return (s, starts) where s is patch size and starts is a list of k start positions
    satisfying starts[0]=0, starts[-1]=n-s, ensuring full coverage of [0, n).
    """
    if k <= 1:
        return n, [0]

    target_overlap = float(max(0.0, min(0.9, target_overlap)))
    denom = (k - 1) + 1.0 / max(1.0 - target_overlap, 1e-8)
    d = max(1, int(round(n / denom)))
    s = n - d * (k - 1)

    if s <= 1:
        d = max(1, n // k)
        s = n - d * (k - 1)

    if s <= 1:
        raise ValueError(f"Invalid subdomain parameters: n={n}, k={k}, s={s}, d={d}")

    starts = _compute_subdomain_starts_1d(n, s, k)
    return s, starts


def _sync_if_needed(device):
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device, fn):
    _sync_if_needed(device)
    tic = time.perf_counter()
    out = fn()
    _sync_if_needed(device)
    return out, time.perf_counter() - tic


@dataclass
class CGInfo:
    num_iter: int
    rel_res: float
    converged: bool
    residual_history: List[float]


class MobilityFVMOperatorASM(nn.Module):
    """
    Matrix-free cell-centered FVM operator for

        - div(lambda grad p) = q

    on Omega = [0, Lx] x [0, Ly].

    Pressure boundary conditions:
        x = 0:   p = p_left
        x = Lx:  p = p_right
        y = 0,Ly: no-flow Neumann.

    Unknowns are cell-centered values p with shape (B, 1, ny, nx).

    The linear system is
        A p = q + b_bc.
    For the closed-droplet benchmark, q = 0 and the flow is driven by p_left - p_right.
    """

    def __init__(
        self,
        lam: torch.Tensor,
        hx: float,
        hy: float,
        p_left: float = 1.0,
        p_right: float = 0.0,
        kx_sub: int = 8,
        ky_sub: int = 8,
        target_overlap: float = 0.3,
        asm_reg: float = 1e-10,
        build_local_solver: bool = True,
        subdomain_solver: str = "cholesky",
        jacobi_pcg_iters: int = 10,
        phi: Optional[torch.Tensor] = None,
        transmissibility: str = "harmonic",
        cholesky_backend: str = "blocktri",
    ):
        super().__init__()

        if lam.dim() == 2:
            lam = lam[None, None, :, :]
        elif lam.dim() == 3:
            lam = lam[:, None, :, :]
        elif lam.dim() != 4:
            raise ValueError(f"lam must have dim 2, 3, or 4, got shape={tuple(lam.shape)}")

        self.lam = lam.double()
        self.device = self.lam.device
        self.dtype = self.lam.dtype
        self.transmissibility = str(transmissibility).lower()
        if self.transmissibility == "interface":
            self.transmissibility = "interface_aware"
        if self.transmissibility not in {"harmonic", "interface_aware"}:
            raise ValueError(
                "transmissibility must be 'harmonic', 'interface', or "
                f"'interface_aware', got {transmissibility!r}"
            )
        self.phi = None if phi is None else phi.to(
            device=self.device,
            dtype=self.dtype,
        )
        if self.transmissibility != "harmonic" and self.phi is None:
            raise ValueError("phi is required for interface-aware transmissibility")

        self.B = self.lam.shape[0]
        self.ny = self.lam.shape[-2]
        self.nx = self.lam.shape[-1]
        self.N = self.nx * self.ny

        self.hx = float(hx)
        self.hy = float(hy)
        self.p_left = float(p_left)
        self.p_right = float(p_right)

        self.kx_sub = int(kx_sub)
        self.ky_sub = int(ky_sub)
        self.target_overlap = float(target_overlap)
        self.sx, starts_x = get_fixed_subdomains_params_1d(self.nx, self.kx_sub, self.target_overlap)
        self.sy, starts_y = get_fixed_subdomains_params_1d(self.ny, self.ky_sub, self.target_overlap)
        self.starts_x = starts_x
        self.starts_y = starts_y

        self.Kx = self.kx_sub
        self.Ky = self.ky_sub
        self.K = self.Kx * self.Ky

        self.asm_reg = float(asm_reg)
        self.build_local_solver = bool(build_local_solver)
        self.subdomain_solver = str(subdomain_solver).lower()
        if self.subdomain_solver not in {"cholesky", "jacobi_pcg"}:
            raise ValueError(f"unknown subdomain_solver={subdomain_solver!r}")
        # "blocktri": exact block-tridiagonal Cholesky, O(K sy sx^2) storage;
        # "dense": batched dense factor, O(K (sy sx)^2), for small-grid cross-checks.
        self.cholesky_backend = str(cholesky_backend).lower()
        if self.cholesky_backend not in {"blocktri", "dense"}:
            raise ValueError(f"unknown cholesky_backend={cholesky_backend!r}")
        self.jacobi_pcg_iters = int(jacobi_pcg_iters)

        self._build_coefficients()
        self._build_overlap_mask()
        if self.build_local_solver:
            if self.subdomain_solver == "cholesky" and self.cholesky_backend == "blocktri":
                self.register_buffer("local_cholesky_factor", None, persistent=False)
                Linv, G, H = self._build_local_blocktri_factor()
                self.register_buffer("local_bt_linv", Linv, persistent=False)
                self.register_buffer("local_bt_g", G, persistent=False)
                self.register_buffer("local_bt_h", H, persistent=False)
            elif self.subdomain_solver == "cholesky":
                self.register_buffer(
                    "local_cholesky_factor",
                    self._build_local_cholesky_factor(),
                    persistent=False,
                )
            else:
                self.register_buffer("local_cholesky_factor", None, persistent=False)
                self._cache_local_stencils()
        else:
            self.register_buffer("local_cholesky_factor", None, persistent=False)

    def _build_coefficients(self) -> None:
        coeffs = build_mobility_coefficients(
            self.lam,
            self.hx,
            self.hy,
            self.p_left,
            self.p_right,
            phi=self.phi,
            transmissibility=self.transmissibility,
        )
        for name, tensor in coeffs.items():
            self.register_buffer(name, tensor, persistent=False)

    def _unfold_subdomains(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, ny, nx) → (B, C, sy*sx, K); layout matches F.unfold per channel."""
        B, C = x.shape[:2]
        sy, sx = self.sy, self.sx
        starts_y, starts_x = self.starts_y, self.starts_x
        ky, kx = self.ky_sub, self.kx_sub
        patches = x.new_empty(B, C, ky, kx, sy, sx)
        for i, iy in enumerate(starts_y):
            for j, ix in enumerate(starts_x):
                patches[:, :, i, j] = x[:, :, iy:iy + sy, ix:ix + sx]
        return patches.permute(0, 1, 4, 5, 2, 3).contiguous().reshape(B, C, sy * sx, ky * kx)

    def _fold_subdomains(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: (B, C, sy*sx, K) → (B, C, ny, nx) with overlapping contributions summed."""
        B, C = patches.shape[:2]
        sy, sx = self.sy, self.sx
        starts_y, starts_x = self.starts_y, self.starts_x
        ky, kx = self.ky_sub, self.kx_sub
        out = patches.new_zeros(B, C, self.ny, self.nx)
        p = patches.reshape(B, C, sy, sx, ky, kx).permute(0, 1, 4, 5, 2, 3).contiguous()
        for i, iy in enumerate(starts_y):
            for j, ix in enumerate(starts_x):
                out[:, :, iy:iy + sy, ix:ix + sx] += p[:, :, i, j]
        return out

    def _build_overlap_mask(self) -> None:
        ones = torch.ones(1, 1, self.sy * self.sx, self.K, device=self.device, dtype=self.dtype)
        mask = self._fold_subdomains(ones)
        self.register_buffer("overlap_mask", mask.clamp_min(1.0), persistent=False)

    def _to_4d(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        x = x.to(device=self.device, dtype=self.dtype)
        original_shape = tuple(x.shape)

        if x.dim() == 1:
            x = x.view(1, 1, self.ny, self.nx)
        elif x.dim() == 2:
            x = x.view(x.shape[0], 1, self.ny, self.nx)
        elif x.dim() == 3:
            if x.shape[-1] == self.N:
                x = x.view(x.shape[0], x.shape[1], self.ny, self.nx)
            elif x.shape[-2:] == (self.ny, self.nx):
                x = x[:, None, :, :]
            else:
                raise ValueError(f"Cannot parse x shape {original_shape}")
        elif x.dim() == 4:
            pass
        else:
            raise ValueError(f"Unsupported x shape {original_shape}")

        return x, original_shape

    def _from_4d(self, x4: torch.Tensor, original_shape: Tuple[int, ...]) -> torch.Tensor:
        if len(original_shape) == 1:
            return x4.reshape(-1)
        if len(original_shape) == 2:
            return x4.reshape(original_shape[0], -1)
        if len(original_shape) == 3:
            if original_shape[-1] == self.N:
                return x4.reshape(original_shape[0], original_shape[1], -1)
            return x4[:, 0, :, :]
        if len(original_shape) == 4:
            return x4
        raise ValueError(f"Unsupported original_shape={original_shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply A x."""
        x4, original_shape = self._to_4d(x)

        Ax = self.diag * x4
        Ax[..., :, 1:] -= self.aw[..., :, 1:] * x4[..., :, :-1]
        Ax[..., :, :-1] -= self.ae[..., :, :-1] * x4[..., :, 1:]
        Ax[..., 1:, :] -= self.as_[..., 1:, :] * x4[..., :-1, :]
        Ax[..., :-1, :] -= self.an[..., :-1, :] * x4[..., 1:, :]

        return self._from_4d(Ax, original_shape)

    def rhs(self, q: Optional[torch.Tensor] = None) -> torch.Tensor:
        if q is None:
            q4 = torch.zeros_like(self.lam)
        else:
            q4, _ = self._to_4d(q)
        return q4 + self.bc_rhs

    def _build_local_cholesky_factor(self) -> torch.Tensor:
        B = self.lam.shape[0]
        sy, sx = self.sy, self.sx
        nloc = sy * sx

        D_p = self._unfold_subdomains(self.diag).squeeze(1)
        W_p = self._unfold_subdomains(self.aw).squeeze(1)
        E_p = self._unfold_subdomains(self.ae).squeeze(1)
        S_p = self._unfold_subdomains(self.as_).squeeze(1)
        N_p = self._unfold_subdomains(self.an).squeeze(1)

        D_t = D_p.transpose(1, 2)
        W_t = W_p.transpose(1, 2)
        E_t = E_p.transpose(1, 2)
        S_t = S_p.transpose(1, 2)
        N_t = N_p.transpose(1, 2)

        A_local = torch.zeros(B, self.K, nloc, nloc, device=self.device, dtype=self.dtype)
        idx = torch.arange(nloc, device=self.device)
        A_local[:, :, idx, idx] = D_t

        idx_w = idx[idx % sx != 0]
        idx_e = idx[idx % sx != sx - 1]
        idx_s = idx[idx >= sx]
        idx_n = idx[idx < nloc - sx]

        A_local[:, :, idx_w, idx_w - 1] = -W_t[:, :, idx_w]
        A_local[:, :, idx_e, idx_e + 1] = -E_t[:, :, idx_e]
        A_local[:, :, idx_s, idx_s - sx] = -S_t[:, :, idx_s]
        A_local[:, :, idx_n, idx_n + sx] = -N_t[:, :, idx_n]

        eye = torch.eye(nloc, device=self.device, dtype=self.dtype)[None, None, :, :]
        A_local = A_local + self.asm_reg * eye
        return torch.linalg.cholesky(A_local.reshape(B * self.K, nloc, nloc).contiguous())

    def _build_local_blocktri_factor(self):
        """Block-tridiagonal Cholesky factors of the shifted local matrices; see
        model.operators.blocktri_cholesky."""
        BK = self.B * self.K
        sy, sx = self.sy, self.sx

        def _rows(x):
            return self._unfold_subdomains(x).squeeze(1).transpose(1, 2).reshape(BK, sy, sx)

        return blocktri_cholesky_factor(
            _rows(self.diag), -_rows(self.aw), -_rows(self.as_), self.asm_reg
        )

    def _apply_asm_blocktri_cholesky(self, patches, B, C, nloc):
        """Block-tridiagonal Cholesky local solve; (B, C, nloc, K) → (B, C, nloc, K)."""
        sy, sx, K = self.sy, self.sx, self.K
        if B == self.B:
            rhs = patches.view(B, C, sy, sx, K).permute(0, 4, 2, 3, 1).reshape(B * K, sy, sx, C)
        elif self.B == 1:  # fold the residual batch into the right-hand-side columns
            rhs = patches.view(B, C, sy, sx, K).permute(4, 2, 3, 0, 1).reshape(K, sy, sx, B * C)
        else:
            raise ValueError(f"ASM batch mismatch: residual B={B}, operator B={self.B}")
        x = blocktri_cholesky_solve(self.local_bt_linv, self.local_bt_g, self.local_bt_h, rhs)
        if B == self.B:
            z = x.reshape(B, K, sy, sx, C).permute(0, 4, 2, 3, 1)
        else:
            z = x.reshape(K, sy, sx, B, C).permute(3, 4, 1, 2, 0)
        return z.reshape(B, C, nloc, K).contiguous()

    def _cache_local_stencils(self):
        """Pre-compute unfolded stencil patches for use in batched PCG matvec."""
        B = self.B
        sy, sx = self.sy, self.sx
        K = self.K

        def _unfold_to_BKsysx(x):
            patches = self._unfold_subdomains(x)  # (B, 1, sy*sx, K)
            return patches.squeeze(1).permute(0, 2, 1).reshape(B, K, sy, sx).contiguous()

        self.register_buffer("D_loc", _unfold_to_BKsysx(self.diag), persistent=False)
        self.register_buffer("W_loc", _unfold_to_BKsysx(self.aw), persistent=False)
        self.register_buffer("E_loc", _unfold_to_BKsysx(self.ae), persistent=False)
        self.register_buffer("S_loc", _unfold_to_BKsysx(self.as_), persistent=False)
        self.register_buffer("N_loc", _unfold_to_BKsysx(self.an), persistent=False)

    def _local_stencil_matvec(self, u_BKCsysx):
        """Apply A_loc to (B, K, C, sy, sx) subdomain patches with zero-Dirichlet padding."""
        u_pad = F.pad(u_BKCsysx, (1, 1, 1, 1), mode="constant", value=0)
        D = self.D_loc.unsqueeze(2)  # (B, K, 1, sy, sx)
        W = self.W_loc.unsqueeze(2)  # aw (positive, subtracted for west neighbor)
        E = self.E_loc.unsqueeze(2)  # ae
        S = self.S_loc.unsqueeze(2)  # as_ (row i-1 in array)
        N = self.N_loc.unsqueeze(2)  # an  (row i+1 in array)
        # Matches forward(): Ax = diag*u - aw*u_w - ae*u_e - as_*u_{i-1} - an*u_{i+1}
        return (
            D * u_BKCsysx
            - W * u_pad[..., 1:-1, :-2]
            - E * u_pad[..., 1:-1, 2:]
            - S * u_pad[..., :-2, 1:-1]
            - N * u_pad[..., 2:, 1:-1]
        )

    def _apply_jacobi_pcg_subdomain(self, rhs_BKCsysx):
        """k-step Jacobi-PCG in each (B, K) subdomain; all batch/channel dims are parallel."""
        eps = 1e-30
        inv_D = 1.0 / self.D_loc.unsqueeze(2).clamp_min(eps)  # (B, K, 1, sy, sx)
        z = torch.zeros_like(rhs_BKCsysx)
        r = rhs_BKCsysx.clone()
        z_pre = inv_D * r
        p = z_pre.clone()
        rs_old = (r * z_pre).sum(dim=(-1, -2), keepdim=True)  # (B, K, C, 1, 1)
        for _ in range(self.jacobi_pcg_iters):
            Ap = self._local_stencil_matvec(p)
            pAp = (p * Ap).sum(dim=(-1, -2), keepdim=True).clamp_min(eps)
            alpha = rs_old / pAp
            z = z + alpha * p
            r = r - alpha * Ap
            z_new = inv_D * r
            rs_new = (r * z_new).sum(dim=(-1, -2), keepdim=True)
            beta = rs_new / rs_old.clamp_min(eps)
            p = z_new + beta * p
            rs_old = rs_new
        return z

    def _apply_asm_cholesky(self, patches, B, C, nloc):
        """Cholesky local solve; patches (B, C, nloc, K) → z_patches (B, C, nloc, K)."""
        r_b = patches.view(B, C, nloc, self.K).permute(0, 1, 3, 2).unsqueeze(-1)
        if B == self.B:
            L = self.local_cholesky_factor.view(B, self.K, nloc, nloc)
        elif self.B == 1:
            L = self.local_cholesky_factor.view(1, self.K, nloc, nloc).expand(B, self.K, nloc, nloc)
        else:
            raise ValueError(f"ASM batch mismatch: residual B={B}, operator B={self.B}")
        L = L[:, None, :, :, :].expand(B, C, self.K, nloc, nloc).reshape(B * C * self.K, nloc, nloc)
        r_flat = r_b.reshape(B * C * self.K, nloc, 1)
        z_flat = torch.cholesky_solve(r_flat, L)
        return z_flat.reshape(B, C, self.K, nloc).permute(0, 1, 3, 2).contiguous()

    def apply_jacobi(self, r: torch.Tensor) -> torch.Tensor:
        r4, original_shape = self._to_4d(r)
        z = r4 / self.diag.clamp_min(1e-30)
        return self._from_4d(z, original_shape)

    def apply_asm(self, r: torch.Tensor, average: bool = False) -> torch.Tensor:
        """
        Additive Schwarz preconditioner; dispatches to cholesky or jacobi_pcg.

        average=False gives the standard additive Schwarz sum, which is the safer
        option for PCG symmetry. average=True is useful as a damped smoother.
        """
        if not self.build_local_solver:
            raise RuntimeError(
                "ASM local solver was not built. Construct with build_local_solver=True "
                "when using apply_asm/asm_smooth."
            )
        r4, original_shape = self._to_4d(r)
        B, C, _, _ = r4.shape
        sy, sx = self.sy, self.sx
        nloc = sy * sx

        patches = self._unfold_subdomains(r4)  # (B, C, nloc, K)

        if self.subdomain_solver == "cholesky" and self.cholesky_backend == "blocktri":
            z_patches = self._apply_asm_blocktri_cholesky(patches, B, C, nloc)
        elif self.subdomain_solver == "cholesky":
            z_patches = self._apply_asm_cholesky(patches, B, C, nloc)  # (B, C, nloc, K)
        else:
            # (B, C, nloc, K) → (B, K, C, sy, sx)
            rhs = patches.permute(0, 3, 1, 2).reshape(B, self.K, C, sy, sx).contiguous()
            z_rhs = self._apply_jacobi_pcg_subdomain(rhs)  # (B, K, C, sy, sx)
            # (B, K, C, sy, sx) → (B, C, nloc, K)
            z_patches = z_rhs.permute(0, 2, 3, 4, 1).contiguous().reshape(B, C, nloc, self.K)

        z = self._fold_subdomains(z_patches)
        if average:
            z = z / self.overlap_mask
        return self._from_4d(z, original_shape)

    def apply_preconditioner(
        self,
        r: torch.Tensor,
        kind: str = "asm",
        *,
        prepared_coarse_basis: Optional[Dict[str, object]] = None,
        tg_config: Optional[Dict[str, object]] = None,
        gmg_hierarchy: Optional["StandardGMGHierarchy"] = None,
        gmg_config: Optional[Dict[str, object]] = None,
    ) -> torch.Tensor:
        kind = str(kind).lower()
        if kind == "none":
            return r
        if kind == "jacobi":
            return self.apply_jacobi(r)
        if kind == "asm":
            return self.apply_asm(r, average=False)
        if kind in {"asm_avg", "ras"}:
            return self.apply_asm(r, average=True)
        if kind in {"tg", "learned_tg"}:
            if prepared_coarse_basis is None:
                raise ValueError(
                    "preconditioner='tg' requires a prepared_coarse_basis. "
                    "Call prepare_two_phase_coarse_basis once before iteration "
                    "and pass it into apply_preconditioner/fcg."
            )
            tg_config = {} if tg_config is None else dict(tg_config)
            smoother = tg_config.get("smoother", tg_config.get("smooth_mode", "asm"))
            pre_smoother = tg_config.get(
                "pre_smoother",
                tg_config.get("pre_smooth_mode", None),
            )
            post_smoother = tg_config.get(
                "post_smoother",
                tg_config.get("post_smooth_mode", None),
            )
            r4, original_shape = self._to_4d(r)
            z0 = torch.zeros_like(r4)
            import learned_tg.model.two_phase as _tp
            z4 = _tp.two_phase_two_grid_sweep(
                self,
                z0,
                r4,
                prepared_coarse_basis,
                pre_smooth_steps=int(tg_config.get("pre_smooth_steps", 1)),
                pre_smooth_omega=float(tg_config.get("pre_smooth_omega", 0.8)),
                post_smooth_steps=int(tg_config.get("post_smooth_steps", 0)),
                post_smooth_omega=float(tg_config.get("post_smooth_omega", 0.8)),
                smoother=str(smoother),
                pre_smoother=None if pre_smoother is None else str(pre_smoother),
                post_smoother=None if post_smoother is None else str(post_smoother),
                gmg_hierarchy=gmg_hierarchy,
            )
            return self._from_4d(z4, original_shape)
        if kind == "gmg":
            if gmg_hierarchy is None:
                raise ValueError(
                    "preconditioner='gmg' requires a pre-built StandardGMGHierarchy. "
                    "Construct the hierarchy once before iteration and pass it into "
                    "apply_preconditioner/fcg."
                )
            _ = gmg_config  # Reserved for API symmetry; hierarchy stores V-cycle settings.
            return gmg_hierarchy.apply(r)
        raise ValueError(f"Unknown preconditioner kind={kind}")

    @torch.no_grad()
    def pcg(
        self,
        b: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        tol: float = 1e-8,
        maxiter: int = 500,
        preconditioner: str = "asm",
        gmg_hierarchy: Optional["StandardGMGHierarchy"] = None,
        gmg_config: Optional[Dict[str, object]] = None,
    ) -> Tuple[torch.Tensor, CGInfo]:
        preconditioner = str(preconditioner).lower()
        if preconditioner in {"tg", "learned_tg"}:
            raise ValueError(
                "PCG only supports fixed SPD-style preconditioners here. "
                f"preconditioner={preconditioner!r} may be non-SPD; use fcg instead."
            )
        b4, _ = self._to_4d(b)
        x = torch.zeros_like(b4) if x0 is None else self._to_4d(x0)[0]

        r = b4 - self.forward(x)
        z = self.apply_preconditioner(
            r,
            preconditioner,
            gmg_hierarchy=gmg_hierarchy,
            gmg_config=gmg_config,
        )
        p = z.clone()
        rz = torch.sum(r * z, dim=(1, 2, 3), keepdim=True)
        bnorm = torch.linalg.norm(b4.reshape(b4.shape[0], -1), dim=1).clamp_min(1e-30)

        residual_history: List[float] = []
        converged = False
        it = 0

        for it in range(1, maxiter + 1):
            Ap = self.forward(p)
            denom = torch.sum(p * Ap, dim=(1, 2, 3), keepdim=True)
            denom = torch.where(torch.abs(denom) < 1e-300, torch.full_like(denom, 1e-300), denom)
            alpha = rz / denom

            x = x + alpha * p
            r = r - alpha * Ap

            rnorm = torch.linalg.norm(r.reshape(r.shape[0], -1), dim=1)
            rel = torch.max(rnorm / bnorm).item()
            residual_history.append(rel)
            if rel < tol:
                converged = True
                break

            z = self.apply_preconditioner(
                r,
                preconditioner,
                gmg_hierarchy=gmg_hierarchy,
                gmg_config=gmg_config,
            )
            rz_new = torch.sum(r * z, dim=(1, 2, 3), keepdim=True)
            beta = rz_new / rz.clamp_min(1e-300)
            p = z + beta * p
            rz = rz_new

        return x, CGInfo(
            num_iter=it,
            rel_res=residual_history[-1] if residual_history else float("nan"),
            converged=converged,
            residual_history=residual_history,
        )

    @torch.no_grad()
    def fcg(
        self,
        b: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        tol: float = 1e-8,
        maxiter: int = 200,
        preconditioner: str = "tg",
        m_max: Optional[int] = None,
        prepared_coarse_basis: Optional[Dict[str, object]] = None,
        tg_config: Optional[Dict[str, object]] = None,
        gmg_hierarchy: Optional["StandardGMGHierarchy"] = None,
        gmg_config: Optional[Dict[str, object]] = None,
        reorthogonalize: bool = True,
        breakdown_tol: float = 1e-28,
    ) -> Tuple[torch.Tensor, CGInfo]:
        """
        Batched Flexible GCR (F-GCR) solver.

        Runs in double precision throughout. Uses a per-sample active mask so
        converged samples are frozen rather than over-corrected. Orthogonalization
        shields against numerical breakdown via explicit validity masking.
        m_max=None keeps full history; m_max=0 degrades to preconditioned Richardson.
        """
        preconditioner = str(preconditioner).lower()
        if m_max is not None and int(m_max) < 0:
            raise ValueError(f"m_max must be non-negative when provided, got {m_max}")
        m_max_int = None if m_max is None else int(m_max)

        if preconditioner in {"tg", "learned_tg"} and gmg_hierarchy is None:
            tg_config_for_smoother = {} if tg_config is None else dict(tg_config)
            smoother = tg_config_for_smoother.get(
                "smoother",
                tg_config_for_smoother.get("smooth_mode", "asm"),
            )
            pre_smoother = tg_config_for_smoother.get(
                "pre_smoother",
                tg_config_for_smoother.get("pre_smooth_mode", None),
            )
            post_smoother = tg_config_for_smoother.get(
                "post_smoother",
                tg_config_for_smoother.get("post_smooth_mode", None),
            )
            import learned_tg.model.two_phase as _tp
            pre_mode = _tp._two_phase_smoother_mode(pre_smoother if pre_smoother is not None else smoother)
            post_mode = _tp._two_phase_smoother_mode(post_smoother if post_smoother is not None else smoother)
            if pre_mode == "gmg" or post_mode == "gmg":
                gmg_hierarchy = _tp._gmg_hierarchy_from_config(
                    self,
                    tg_config_for_smoother.get("gmg_config", gmg_config),
                )

        def _batch_dot(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return torch.sum(u * v, dim=(1, 2, 3), keepdim=True)

        # Force double precision for numerical stability at high contrast ratios
        b4, _ = self._to_4d(b)
        b4 = b4.double()
        x = (torch.zeros_like(b4) if x0 is None else self._to_4d(x0)[0].double())
        r = b4 - self.forward(x).double()
        f_norm = torch.sqrt(_batch_dot(b4, b4)).clamp(min=1e-30)

        residual_history: List[float] = []
        batch_size = b4.shape[0]

        rel0_vec = torch.sqrt(_batch_dot(r, r)) / f_norm
        rel0 = float(rel0_vec.max().item())
        if rel0 < tol:
            return x, CGInfo(
                num_iter=0,
                rel_res=float(rel0),
                converged=True,
                residual_history=residual_history,
            )

        p_history: List[torch.Tensor] = []
        q_history: List[torch.Tensor] = []
        converged = False
        current_rel = float(rel0)

        # Per-sample active mask: 1.0 = not yet converged, 0.0 = converged
        active_mask = torch.ones((batch_size, 1, 1, 1), device=b4.device, dtype=torch.float64)

        for _ in range(1, int(maxiter) + 1):
            z = self.apply_preconditioner(
                r,
                kind=preconditioner,
                prepared_coarse_basis=prepared_coarse_basis,
                tg_config=tg_config,
                gmg_hierarchy=gmg_hierarchy,
                gmg_config=gmg_config,
            ).double()
            if not bool(torch.isfinite(z).all().item()):
                break

            p = z.clone()
            q = self.forward(p).double()
            if not bool(torch.isfinite(q).all().item()):
                break

            # Orthogonalization (first pass, Modified Gram-Schmidt)
            for p_k, q_k in zip(p_history, q_history):
                qk_qk = _batch_dot(q_k, q_k)
                valid_qk = (qk_qk > breakdown_tol).double()
                beta = _batch_dot(q, q_k) / qk_qk.clamp(min=breakdown_tol) * valid_qk
                p = p - beta * p_k
                q = q - beta * q_k

            # Optional re-orthogonalization (second pass)
            if reorthogonalize:
                for p_k, q_k in zip(p_history, q_history):
                    qk_qk = _batch_dot(q_k, q_k)
                    valid_qk = (qk_qk > breakdown_tol).double()
                    beta = _batch_dot(q, q_k) / qk_qk.clamp(min=breakdown_tol) * valid_qk
                    p = p - beta * p_k
                    q = q - beta * q_k

            # Step size: only update unconverged samples that haven't broken down
            qq = _batch_dot(q, q)
            valid_q = (qq > breakdown_tol).double()
            alpha = _batch_dot(r, q) / qq.clamp(min=breakdown_tol) * valid_q * active_mask
            if not bool(torch.isfinite(alpha).all().item()):
                break

            x = x + alpha * p
            r = r - alpha * q

            rel_res = torch.sqrt(_batch_dot(r, r)) / f_norm
            active_mask = (rel_res > tol).double()
            current_rel = float(rel_res.max().item())
            residual_history.append(current_rel)

            if m_max_int is None or m_max_int > 0:
                p_history.append(p)
                q_history.append(q)
                if m_max_int is not None and len(p_history) >= m_max_int:
                    # True FGCR(m) restart; a rolling window can annihilate all
                    # subsequent directions after the stored space saturates.
                    p_history.clear()
                    q_history.clear()

            if active_mask.sum() == 0:
                converged = True
                break

        return x, CGInfo(
            num_iter=len(residual_history),
            rel_res=float(current_rel),
            converged=bool(converged),
            residual_history=residual_history,
        )

    def asm_smooth(self, x: torch.Tensor, f: torch.Tensor, steps: int = 1, omega: float = 0.8, M_smooth: int = None) -> torch.Tensor:
        if not self.build_local_solver:
            raise RuntimeError(
                "ASM local solver was not built. Construct with build_local_solver=True "
                "when using apply_asm/asm_smooth."
            )
        # Accept M_smooth as an alias for steps (SupportsApplyA protocol compatibility).
        if M_smooth is not None:
            steps = int(M_smooth)
        x4, _ = self._to_4d(x)
        f4, _ = self._to_4d(f)
        for _ in range(steps):
            r = f4 - self.forward(x4)
            z = self.apply_asm(r, average=True)
            x4 = x4 + omega * z
        return x4

    def apply_A(self, X: torch.Tensor) -> torch.Tensor:
        """
        Batch matrix-vector product for the SupportsApplyA protocol.

        X:  (B, N, m) with N = ny * nx
        AX: (B, N, m)
        """
        X = X.to(
            device=self.device, dtype=torch.float64
        )
        if X.dim() != 3:
            raise ValueError(f"X must have shape (B, N, m), got {tuple(X.shape)}")
        B, N, m = X.shape
        X_img = X.transpose(1, 2).reshape(B, m, self.ny, self.nx)
        AX_img = self.forward(X_img)
        AX = AX_img.reshape(B, m, N).transpose(1, 2)
        return AX

    def diag_for_jacobi(self) -> torch.Tensor:
        """Return (B, N, 1) diagonal for weighted-Jacobi smoothing."""
        return self.diag.reshape(self.diag.shape[0], -1, 1).clamp_min(1e-30)

    def grid_shape(self):
        """Return (ny, nx) for the SupportsApplyA protocol."""
        return (self.ny, self.nx)
