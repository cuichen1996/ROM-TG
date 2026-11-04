"""
Cell-centered FVM operator for the interface (jumping-coefficient) problem,
with additive Schwarz smoother.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from learned_tg.model.operators.blocktri_cholesky import blocktri_cholesky_factor, blocktri_cholesky_solve


def _face_transmissibility(a0, a1, phi0=None, phi1=None, mode="harmonic"):
    """Return positive torch transmissibilities for neighboring cell centers.

    The default harmonic branch deliberately retains the historical ``1e-16``
    denominator regularization so existing callers reproduce the old stencil.
    The interface-aware aliases replace only sign-changing faces with the
    level-set intersection-distance formula used by the sparse reference.
    """
    mode = str(mode).lower()
    if mode not in {"harmonic", "interface", "interface_aware"}:
        raise ValueError(
            "transmissibility must be 'harmonic', 'interface', or "
            "'interface_aware'"
        )
    if a0.shape != a1.shape:
        raise ValueError("neighbor coefficient tensors must have the same shape")
    if not bool(torch.isfinite(a0).all()) or not bool(torch.isfinite(a1).all()):
        raise ValueError("face coefficients must be finite")
    if bool((a0 <= 0).any()) or bool((a1 <= 0).any()):
        raise ValueError("face coefficients must be strictly positive")

    if mode == "harmonic":
        transmissibility = 2.0 * a0 * a1 / (a0 + a1 + 1e-16)
    else:
        if phi0 is None or phi1 is None:
            raise ValueError(f"phi is required for transmissibility={mode!r}")
        if phi0.shape != a0.shape or phi1.shape != a1.shape:
            raise ValueError("neighbor phi tensors must match coefficient shapes")
        if not bool(torch.isfinite(phi0).all()) or not bool(torch.isfinite(phi1).all()):
            raise ValueError("face level-set values must be finite")

        harmonic = 2.0 * a0 * a1 / (a0 + a1)
        crossing = (phi0 * phi1) < 0.0
        eps = torch.finfo(a0.dtype).eps
        theta = phi0.abs() / (phi0.abs() + phi1.abs()).clamp_min(eps)
        interface_denominator = theta / a0 + (1.0 - theta) / a1
        interface_value = interface_denominator.clamp_min(eps).reciprocal()
        transmissibility = torch.where(crossing, interface_value, harmonic)

    if not bool(torch.isfinite(transmissibility).all()):
        raise RuntimeError("face transmissibilities must be finite")
    if bool((transmissibility <= 0).any()):
        raise RuntimeError("face transmissibilities must be strictly positive")
    return transmissibility


def get_fixed_subdomains_params(n: int, k_1d: int = 10, target_overlap: float = 0.3):
    """Return (s, d, n_pad) where:
    - s is the constant subdomain side length (kernel_size), **enforced to be even**
      so the GMG V-cycle hierarchy can coarsen without trimming.
    - d is the constant stride
    - n_pad is the padded grid size that perfectly fits k_1d patches
    """
    if k_1d == 1:
        # Single subdomain: enforce s even by bumping if needed
        s = n if n % 2 == 0 else n + 1
        return s, s, max(s, n)

    denom = 1.0 + (k_1d - 1) * (1.0 - target_overlap)
    s = max(2, min(n, round(n / denom)))

    # Force s to be even so GMG V-cycle can coarsen without trimming.
    if s % 2 != 0:
        s += 1

    d = max(1, round(s * (1.0 - target_overlap)))
    n_pad = s + (k_1d - 1) * d
    if n_pad < n:
        import math
        d = max(1, math.ceil((n - s) / (k_1d - 1)))
        n_pad = s + (k_1d - 1) * d
    return s, d, n_pad


class CCFVM_ASM(nn.Module):
    """
    Cell-centered FVM stiffness operator for -div(a grad u) = f with zero
    Dirichlet boundary conditions.  Optionally builds additive Schwarz (ASM)
    local solvers for smoothing.  The default face transmissibility is the
    historical harmonic mean; a supplied level set enables interface-aware
    crossing distances without changing the matrix-free stencil interface.

    ``subdomain_solver="cholesky"`` solves the local problems exactly.  The
    default ``cholesky_backend="blocktri"`` factors each local matrix in its
    block-tridiagonal (patch-row) form on the device, storing O(K s^3) entries;
    ``"dense"`` keeps the original batched dense factor, O(K s^4) entries
    (about 307 GB at n=1023), for small-grid cross-checks.
    """

    def __init__(self, a, k_1d=10, target_overlap=0.3, build_local_solver=True,
                 subdomain_solver="cholesky", jacobi_pcg_iters=10,
                 vcycle_coarsest_threshold=24, vcycle_smooth_steps=2,
                 vcycle_iter=2, vcycle_omega=2.0 / 3.0, vcycle_compile=True,
                 phi=None, transmissibility="harmonic", cholesky_backend="blocktri"):
        super(CCFVM_ASM, self).__init__()
        self.a = a.unsqueeze(1).double() if a.dim() == 3 else a.double()
        self.device = a.device
        self.dtype = self.a.dtype
        self.n = self.a.shape[-1]
        if self.a.dim() != 4 or self.a.shape[1] != 1:
            raise ValueError("a must have shape (B,n,n) or (B,1,n,n)")
        if self.a.shape[-2] != self.n:
            raise ValueError("CCFVM_ASM requires a square coefficient grid")
        if not bool(torch.isfinite(self.a).all()) or bool((self.a <= 0).any()):
            raise ValueError("a must contain finite, strictly positive coefficients")

        self.transmissibility = str(transmissibility).lower()
        if self.transmissibility not in {
            "harmonic",
            "interface",
            "interface_aware",
        }:
            raise ValueError(
                "transmissibility must be 'harmonic', 'interface', or "
                "'interface_aware'"
            )
        if phi is None:
            if self.transmissibility != "harmonic":
                raise ValueError(
                    "phi is required for interface/interface_aware transmissibility"
                )
            self.phi = None
        else:
            if phi.dim() == 3:
                phi_4d = phi.unsqueeze(1)
            elif phi.dim() == 4 and phi.shape[1] == 1:
                phi_4d = phi
            else:
                raise ValueError("phi must have shape (B,n,n) or (B,1,n,n)")
            self.phi = phi_4d.to(device=self.device, dtype=self.dtype)
            if self.phi.shape != self.a.shape:
                raise ValueError(
                    "phi must have the same batch and spatial shape as a"
                )
            if not bool(torch.isfinite(self.phi).all()):
                raise ValueError("phi must contain only finite values")

        self.k_1d = k_1d
        self.target_overlap = float(target_overlap)
        self.build_local_solver = bool(build_local_solver)
        self.subdomain_solver = str(subdomain_solver).lower()
        if self.subdomain_solver not in {"cholesky", "jacobi_pcg", "gmg_vcycle"}:
            raise ValueError(f"unknown subdomain_solver={subdomain_solver!r}")
        self.cholesky_backend = str(cholesky_backend).lower()
        if self.cholesky_backend not in {"blocktri", "dense"}:
            raise ValueError(f"unknown cholesky_backend={cholesky_backend!r}")
        self.jacobi_pcg_iters = int(jacobi_pcg_iters)
        self.vcycle_coarsest_threshold = int(vcycle_coarsest_threshold)
        self.vcycle_smooth_steps = int(vcycle_smooth_steps)
        self.vcycle_iter = int(vcycle_iter)
        self.vcycle_omega = float(vcycle_omega)
        self.vcycle_compile = bool(vcycle_compile)

        # FVM flux coefficients.  Keep the historical padded arithmetic for
        # default harmonic mode.  Interface-aware mode evaluates each shared
        # interior face once and installs that same tensor value in both
        # directions, making the matrix strictly symmetric.
        if self.transmissibility == "harmonic":
            a_w = F.pad(self.a, [1, 0, 0, 0], mode="replicate")[:, :, :, :-1]
            a_e = F.pad(self.a, [0, 1, 0, 0], mode="replicate")[:, :, :, 1:]
            a_n = F.pad(self.a, [0, 0, 1, 0], mode="replicate")[:, :, :-1, :]
            a_s = F.pad(self.a, [0, 0, 0, 1], mode="replicate")[:, :, 1:, :]
            self.sw = -_face_transmissibility(a_w, self.a, mode="harmonic")
            self.se = -_face_transmissibility(a_e, self.a, mode="harmonic")
            self.sn = -_face_transmissibility(a_n, self.a, mode="harmonic")
            self.ss = -_face_transmissibility(a_s, self.a, mode="harmonic")
        else:
            tx = _face_transmissibility(
                self.a[..., :-1],
                self.a[..., 1:],
                self.phi[..., :-1],
                self.phi[..., 1:],
                mode=self.transmissibility,
            )
            ty = _face_transmissibility(
                self.a[..., :-1, :],
                self.a[..., 1:, :],
                self.phi[..., :-1, :],
                self.phi[..., 1:, :],
                mode=self.transmissibility,
            )
            self.sw = -torch.cat((self.a[..., :1], tx), dim=-1)
            self.se = -torch.cat((tx, self.a[..., -1:]), dim=-1)
            self.sn = -torch.cat((self.a[..., :1, :], ty), dim=-2)
            self.ss = -torch.cat((ty, self.a[..., -1:, :]), dim=-2)

        # FIX Task 1: Double fluxes at Dirichlet boundaries (h/2 distance correction)
        # Distance from cell center to physical boundary is h/2, not h
        self.sw[..., 0] *= 2.0      # West boundary (left column)
        self.se[..., -1] *= 2.0     # East boundary (right column)
        self.sn[..., 0, :] *= 2.0   # North boundary (top row)
        self.ss[..., -1, :] *= 2.0  # South boundary (bottom row)

        # Central coefficient after boundary correction
        self.sc = -(self.sw + self.se + self.sn + self.ss)

        if self.build_local_solver:
            # FIX Task 2: Get fixed stride parameters instead of dynamic starts
            self.s, self.d, self.n_pad = get_fixed_subdomains_params(
                self.n, self.k_1d, self.target_overlap
            )
            self.K = self.k_1d ** 2

            # FIX Task 2.4: Generate overlap_mask using F.fold on all-ones tensor
            # F.fold expects input shape (B, C*s*s, L) where L is number of patches (K)
            # We want to count how many times each pixel is covered by patches
            ones = torch.ones(1, self.s * self.s, self.K, device=self.device, dtype=self.dtype)
            # Use F.fold to create the mask at n_pad size
            mask_full = F.fold(
                ones,
                output_size=(self.n_pad, self.n_pad),
                kernel_size=self.s,
                stride=self.d
            )
            # Crop to original size n x n
            mask = mask_full[..., :self.n, :self.n]
            self.register_buffer("overlap_mask", mask, persistent=False)

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
            elif self.subdomain_solver == "jacobi_pcg":
                self.register_buffer("local_cholesky_factor", None, persistent=False)
                self._cache_local_stencils()
            else:  # gmg_vcycle
                self.register_buffer("local_cholesky_factor", None, persistent=False)
                self._compiled_vcycle = None
                self._build_subdomain_gmg_hierarchy()
        else:
            self.s = None
            self.d = None
            self.n_pad = None
            self.K = None
            self.register_buffer("overlap_mask", None, persistent=False)
            self.register_buffer("local_cholesky_factor", None, persistent=False)

    def _require_local_solver(self, caller):
        if not self.build_local_solver:
            raise RuntimeError(f"{caller} requires build_local_solver=True")

    def _unfold_subdomains(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unfold subdomains using F.unfold with constant stride.
        x: (B, C, n, n) → (B, C, s*s, K)

        Zero-pads input from (n, n) to (n_pad, n_pad) before unfolding.
        """
        B, C = x.shape[:2]
        s = self.s
        d = self.d
        n_pad = self.n_pad
        k_1d = self.k_1d

        # FIX: Zero-pad from (n, n) to (n_pad, n_pad)
        pad_right = n_pad - self.n
        pad_bottom = n_pad - self.n
        x_padded = F.pad(x, (0, pad_right, 0, pad_bottom), mode="constant", value=0)

        # Use F.unfold with constant stride
        # kernel_size=s, stride=d, output should be (B, C, s*s, k_1d*k_1d)
        patches = F.unfold(
            x_padded,
            kernel_size=s,
            stride=d
        )
        # patches shape: (B, C*s*s, k_1d*k_1d)
        # Reshape to (B, C, s*s, K)
        patches = patches.view(B, C, s * s, k_1d * k_1d)
        return patches

    def _fold_subdomains(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Fold subdomains using F.fold with constant stride.
        patches: (B, C, s*s, K) → (B, C, n, n)

        Applies F.fold to get (n_pad, n_pad), then crops back to (n, n).
        """
        B, C = patches.shape[:2]
        s = self.s
        d = self.d
        n_pad = self.n_pad

        # Reshape patches for F.fold: (B, C*s*s, K)
        patches_fold = patches.view(B, C * s * s, self.K)

        # Use F.fold to get output at n_pad size
        folded = F.fold(
            patches_fold,
            output_size=(n_pad, n_pad),
            kernel_size=s,
            stride=d
        )
        # FIX: Crop from (n_pad, n_pad) back to (n, n)
        out = folded[..., :self.n, :self.n]
        return out

    def _local_stencil_fields(self):
        """Return stencil fields suitable for restricted local matrices.

        The global matvec suppresses physical-boundary neighbors through the
        zero padding of ``x``.  A subdomain can, however, extend into the
        right/bottom computational padding when ``n_pad > n``.  In that case
        an unmodified east/south boundary coefficient would become a spurious
        coupling to a padded local unknown.  Zeroing only those off-diagonal
        boundary entries makes every local stencil exactly ``R_k A R_k^T``;
        the boundary contribution remains in ``sc``, so the h/2 correction is
        unchanged.
        """
        west = self.sw.clone()
        east = self.se.clone()
        north = self.sn.clone()
        south = self.ss.clone()
        west[..., 0] = 0.0
        east[..., -1] = 0.0
        north[..., 0, :] = 0.0
        south[..., -1, :] = 0.0
        return self.sc, west, east, north, south

    def _build_local_cholesky_factor(self):
        """Build local Cholesky factors for each subdomain using fixed stride/unfold."""
        B = self.a.shape[0]
        s = self.s

        diagonal, west, east, north, south = self._local_stencil_fields()

        # FIX Task 2.5: Use updated _unfold_subdomains for zero-padded physics
        D_p = self._unfold_subdomains(diagonal).squeeze(1)
        W_p = self._unfold_subdomains(west).squeeze(1)
        E_p = self._unfold_subdomains(east).squeeze(1)
        N_p = self._unfold_subdomains(north).squeeze(1)
        S_p = self._unfold_subdomains(south).squeeze(1)

        A_local = torch.zeros((B, self.K, s * s, s * s), device=self.device, dtype=self.dtype)
        idx = torch.arange(s * s, device=self.device)

        D_t = D_p.transpose(1, 2)
        W_t = W_p.transpose(1, 2)
        E_t = E_p.transpose(1, 2)
        N_t = N_p.transpose(1, 2)
        S_t = S_p.transpose(1, 2)

        A_local[:, :, idx, idx] = D_t

        idx_W = idx[idx % s != 0]
        A_local[:, :, idx_W, idx_W - 1] = W_t[:, :, idx_W]

        idx_E = idx[idx % s != s - 1]
        A_local[:, :, idx_E, idx_E + 1] = E_t[:, :, idx_E]

        idx_N = idx[idx >= s]
        A_local[:, :, idx_N, idx_N - s] = N_t[:, :, idx_N]

        idx_S = idx[idx < s * s - s]
        A_local[:, :, idx_S, idx_S + s] = S_t[:, :, idx_S]

        eye = torch.eye(s * s, device=self.device, dtype=self.dtype).unsqueeze(0).unsqueeze(0)
        A_local = A_local + 1e-6 * eye
        A_local_flat = A_local.view(B * self.K, s * s, s * s).contiguous()
        return torch.linalg.cholesky(A_local_flat)

    def _cache_local_stencils(self):
        """Pre-compute unfolded stencil patches for use in batched PCG matvec."""
        B = self.a.shape[0]
        s = self.s
        K = self.K

        diagonal, west, east, north, south = self._local_stencil_fields()

        def _unfold_to_BKss(x):
            patches = self._unfold_subdomains(x)  # (B, 1, s*s, K)
            return patches.squeeze(1).permute(0, 2, 1).reshape(B, K, s, s).contiguous()

        self.register_buffer("D_loc", _unfold_to_BKss(diagonal), persistent=False)
        self.register_buffer("W_loc", _unfold_to_BKss(west), persistent=False)
        self.register_buffer("E_loc", _unfold_to_BKss(east), persistent=False)
        self.register_buffer("N_loc", _unfold_to_BKss(north), persistent=False)
        self.register_buffer("S_loc", _unfold_to_BKss(south), persistent=False)

    def _local_stencil_matvec(self, u_BKCss):
        """Apply A_loc to (B, K, C, s, s) subdomain patches with zero-Dirichlet padding."""
        u_pad = F.pad(u_BKCss, (1, 1, 1, 1), mode="constant", value=0)
        D = self.D_loc.unsqueeze(2)  # (B, K, 1, s, s)
        W = self.W_loc.unsqueeze(2)
        E = self.E_loc.unsqueeze(2)
        N = self.N_loc.unsqueeze(2)
        S = self.S_loc.unsqueeze(2)
        # sw/se/sn/ss are already negative, so signs here match forward()
        return (
            D * u_BKCss
            + W * u_pad[..., 1:-1, :-2]
            + E * u_pad[..., 1:-1, 2:]
            + N * u_pad[..., :-2, 1:-1]
            + S * u_pad[..., 2:, 1:-1]
        )

    def _apply_jacobi_pcg_subdomain(self, rhs_BKCss):
        """k-step Jacobi-PCG in each (B, K) subdomain; all batch/channel dims are parallel."""
        eps = 1e-30
        inv_D = 1.0 / self.D_loc.unsqueeze(2).clamp_min(eps)  # (B, K, 1, s, s)
        z = torch.zeros_like(rhs_BKCss)
        r = rhs_BKCss.clone()
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

    def _build_subdomain_gmg_hierarchy(self):
        """Build per-subdomain V-cycle hierarchy and register stencils as named buffers.

        If the subdomain size s is odd we trim one row/column to get s_gmg = s-1
        (even), allowing 2×2 Galerkin coarsening.  The trimmed cells sit at the
        local Dirichlet boundary (u≈0 there) so the approximation error is O(1/s).
        """
        from learned_tg.model.operators.subdomain_gmg import build_subdomain_gmg_hierarchy, assemble_coarsest_dense_matrix

        B = self.a.shape[0]
        K = self.K
        s = self.s
        NB = B * K

        # Round s down to the nearest even number so 2×2 coarsening is possible.
        s_gmg = (s // 2) * 2   # s if s is even, else s-1 (safeguard; s should always be even)
        assert s_gmg == s, (
            f"GMG V-cycle requires even subdomain size, got s={s}. "
            f"This should be guaranteed by get_fixed_subdomains_params."
        )
        self.s_gmg = s_gmg

        def _unfold_BK11(stencil_global):
            patches = self._unfold_subdomains(stencil_global)  # (B, 1, s*s, K)
            x = patches.squeeze(1).permute(0, 2, 1).reshape(B, K, s, s).contiguous()
            t = x.reshape(NB, 1, s, s)
            # Trim to s_gmg × s_gmg when s is odd
            return t[..., :s_gmg, :s_gmg]

        diagonal, west, east, north, south = self._local_stencil_fields()
        D0 = _unfold_BK11(diagonal)
        W0 = _unfold_BK11(west)
        E0 = _unfold_BK11(east)
        N0 = _unfold_BK11(north)
        S0 = _unfold_BK11(south)

        levels_list = build_subdomain_gmg_hierarchy(
            D0, W0, E0, N0, S0,
            coarsest_threshold=self.vcycle_coarsest_threshold,
        )

        # Register each level's stencils as named buffers so device migration works.
        for lvl_idx, lvl in enumerate(levels_list):
            self.register_buffer(f"_gmg_D_lvl{lvl_idx}", lvl["D"], persistent=False)
            self.register_buffer(f"_gmg_W_lvl{lvl_idx}", lvl["W"], persistent=False)
            self.register_buffer(f"_gmg_E_lvl{lvl_idx}", lvl["E"], persistent=False)
            self.register_buffer(f"_gmg_N_lvl{lvl_idx}", lvl["N"], persistent=False)
            self.register_buffer(f"_gmg_S_lvl{lvl_idx}", lvl["S"], persistent=False)
        self._gmg_num_levels = len(levels_list)

        # Precompute Cholesky at coarsest level
        coarsest = levels_list[-1]
        A_coarsest = assemble_coarsest_dense_matrix(
            coarsest["D"], coarsest["W"], coarsest["E"],
            coarsest["N"], coarsest["S"],
        )
        self.register_buffer(
            "_gmg_coarsest_L",
            torch.linalg.cholesky(A_coarsest),
            persistent=False,
        )

        # Sanity check the properties that remain valid with Dirichlet rows.
        # A row touching a physical boundary has a positive row sum, so the
        # older D = -(W+E+N+S) test is not a valid invariant there.
        for lvl_idx, lvl in enumerate(levels_list):
            tensors = tuple(lvl[name] for name in ("D", "W", "E", "N", "S"))
            if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
                raise RuntimeError(
                    f"Galerkin coarsening produced non-finite stencil values at level {lvl_idx}"
                )
            if bool((lvl["D"] < 0.0).any()):
                raise RuntimeError(
                    f"Galerkin coarsening produced a negative diagonal at level {lvl_idx}"
                )
            if any(bool((lvl[name] > 0.0).any()) for name in ("W", "E", "N", "S")):
                raise RuntimeError(
                    f"Galerkin coarsening produced a positive off-diagonal at level {lvl_idx}"
                )
            east_west_error = (
                lvl["E"][..., :-1] - lvl["W"][..., 1:]
            ).abs().max()
            south_north_error = (
                lvl["S"][..., :-1, :] - lvl["N"][..., 1:, :]
            ).abs().max()
            symmetry_scale = lvl["D"].abs().max().clamp_min(1e-30)
            symmetry_relative = float(
                torch.maximum(east_west_error, south_north_error) / symmetry_scale
            )
            if symmetry_relative > 1e-12:
                raise RuntimeError(
                    f"Galerkin coarsening broke stencil symmetry at level {lvl_idx}: "
                    f"relative error {symmetry_relative:.2e}"
                )

        sizes = [lvl["D"].shape[-1] for lvl in levels_list]
        print(
            f"[CCFVM_ASM local V-cycle] n={self.n}  s={s}  s_local={s_gmg}  "
            f"num_levels={self._gmg_num_levels}  sizes={sizes}  "
            f"coarsest_Cholesky={sizes[-1]}x{sizes[-1]}={sizes[-1]**2} dof"
        )
        self._gmg_levels_cache = self._get_gmg_levels_as_list()
        if self.vcycle_compile:
            print("[CCFVM_ASM local V-cycle] torch.compile requested (reduce-overhead).")
        else:
            print("[CCFVM_ASM local V-cycle] eager execution selected.")

    def _get_gmg_levels_as_list(self):
        """Reconstruct list-of-dicts from registered named buffers."""
        return [
            {
                "D": getattr(self, f"_gmg_D_lvl{lvl}"),
                "W": getattr(self, f"_gmg_W_lvl{lvl}"),
                "E": getattr(self, f"_gmg_E_lvl{lvl}"),
                "N": getattr(self, f"_gmg_N_lvl{lvl}"),
                "S": getattr(self, f"_gmg_S_lvl{lvl}"),
            }
            for lvl in range(self._gmg_num_levels)
        ]

    def _apply_gmg_vcycle_subdomain(self, rhs_BCKss):
        """Solve A_loc · z = rhs in each (B, K) subdomain via vcycle_iter V-cycles.

        rhs_BCKss: (B, K, C, s, s)
        Returns: z of same shape.

        When s_gmg < s (s was odd), the V-cycle operates on the s_gmg×s_gmg
        interior; the trimmed boundary strip is left as zero (Dirichlet ≈ 0).
        """
        from learned_tg.model.operators.subdomain_gmg import subdomain_vcycle, stencil_matvec_2d

        if self._compiled_vcycle is None:
            if not self.vcycle_compile:
                self._compiled_vcycle = subdomain_vcycle
            else:
                try:
                    self._compiled_vcycle = torch.compile(
                        subdomain_vcycle,
                        mode="reduce-overhead",
                        dynamic=False,
                        fullgraph=False,
                    )
                except Exception as e:
                    print(f"[CCFVM_ASM local V-cycle] torch.compile failed ({e!r}); "
                          f"falling back to eager subdomain_vcycle")
                    self._compiled_vcycle = subdomain_vcycle

        B, K, C, s, _ = rhs_BCKss.shape
        NB = B * K
        s_gmg = self.s_gmg
        levels = self._gmg_levels_cache
        lvl0 = levels[0]

        # All C channels are solved in parallel; the [s_gmg:, :] and [:, s_gmg:]
        # strip stays zero when s_gmg < s (Dirichlet boundary approximation).
        z_BCKss = torch.zeros_like(rhs_BCKss)
        rhs_gmg = rhs_BCKss[:, :, :, :s_gmg, :s_gmg].reshape(NB, C, s_gmg, s_gmg)

        # First iteration: z=0 means residual == rhs, skip the wasted matvec.
        z_gmg = self._compiled_vcycle(
            rhs_gmg, levels, self._gmg_coarsest_L,
            smooth_steps_pre=self.vcycle_smooth_steps,
            smooth_steps_post=self.vcycle_smooth_steps,
            omega=self.vcycle_omega,
        )
        for _ in range(self.vcycle_iter - 1):
            Az = stencil_matvec_2d(z_gmg, lvl0["D"], lvl0["W"], lvl0["E"], lvl0["N"], lvl0["S"])
            dz = self._compiled_vcycle(
                rhs_gmg - Az, levels, self._gmg_coarsest_L,
                smooth_steps_pre=self.vcycle_smooth_steps,
                smooth_steps_post=self.vcycle_smooth_steps,
                omega=self.vcycle_omega,
            )
            z_gmg = z_gmg + dz
        z_BCKss[:, :, :, :s_gmg, :s_gmg] = z_gmg.reshape(B, K, C, s_gmg, s_gmg)

        return z_BCKss

    def _build_local_blocktri_factor(self):
        """Block-tridiagonal Cholesky factors of R_k A R_k^T + 1e-6 I (lower triangle,
        as in the dense branch); see model.operators.blocktri_cholesky."""
        s = self.s
        BK = self.a.shape[0] * self.K
        D, W, _, N, _ = (
            self._unfold_subdomains(f).squeeze(1).transpose(1, 2).reshape(BK, s, s)
            for f in self._local_stencil_fields()
        )  # (B*K, row, col)
        return blocktri_cholesky_factor(D, W, N, 1e-6)

    def _apply_asm_blocktri_cholesky(self, r_patches, B, C, s, K):
        """Block-tridiagonal Cholesky local solve; (B, C, s*s, K) → (B, C, s*s, K)."""
        Bop = self.a.shape[0]
        if Bop == B:
            rhs = r_patches.view(B, C, s, s, K).permute(0, 4, 2, 3, 1).reshape(B * K, s, s, C)
        elif Bop == 1:  # fold the residual batch into the right-hand-side columns
            rhs = r_patches.view(B, C, s, s, K).permute(4, 2, 3, 0, 1).reshape(K, s, s, B * C)
        else:
            raise ValueError(
                f"apply_asm batch size mismatch: residual batch={B}, "
                f"operator batch={Bop}"
            )
        x = blocktri_cholesky_solve(self.local_bt_linv, self.local_bt_g, self.local_bt_h, rhs)
        if Bop == B:
            z = x.reshape(B, K, s, s, C).permute(0, 4, 2, 3, 1)
        else:
            z = x.reshape(K, s, s, B, C).permute(3, 4, 1, 2, 0)
        return z.reshape(B, C, s * s, K).contiguous()

    def _apply_asm_cholesky(self, r_patches, B, C, s, K):
        """Cholesky local solve; r_patches (B, C, s*s, K) → z_patches (B, C, s*s, K)."""
        nloc = s * s
        rhs = r_patches.view(B, C, nloc, K).permute(0, 3, 2, 1).contiguous()  # (B, K, nloc, C)
        if B == self.a.shape[0]:
            L = self.local_cholesky_factor.view(B, K, nloc, nloc)
            rhs_flat = rhs.reshape(B * K, nloc, C)
            L_flat = L.reshape(B * K, nloc, nloc)
            z_flat = torch.cholesky_solve(rhs_flat, L_flat)
            z = z_flat.reshape(B, K, nloc, C)
        elif self.a.shape[0] == 1:
            L0 = self.local_cholesky_factor.view(K, nloc, nloc)
            z_list = [torch.cholesky_solve(rhs[b], L0) for b in range(B)]
            z = torch.stack(z_list, dim=0)
        else:
            raise ValueError(
                f"apply_asm batch size mismatch: residual batch={B}, "
                f"operator batch={self.a.shape[0]}"
            )
        return z.permute(0, 3, 2, 1).contiguous()  # (B, C, nloc, K)

    def forward(self, x):
        """
        Global matrix-vector product A·x.

        Accepts (B, n*n), (B, m, n*n), or (B, m, n, n).
        Returns in the same layout.
        """
        x = x.to(self.dtype)
        xdim = x.dim()

        if xdim == 1:
            x = x.reshape(1, 1, self.n, self.n)
            batch_size, num_channels = 1, 1
        elif xdim == 2:
            x = x.reshape(x.shape[0], 1, self.n, self.n)
            batch_size, num_channels = x.shape[0], 1
        elif xdim == 3:
            batch_size, num_channels, _ = x.shape
            x = x.reshape(batch_size, num_channels, self.n, self.n)
        elif xdim == 4:
            batch_size, num_channels, _, _ = x.shape

        x_pad = F.pad(x, (1, 1, 1, 1), mode="constant", value=0)
        u_w = self.sw * x_pad[:, :, 1:-1, :-2]
        u_e = self.se * x_pad[:, :, 1:-1, 2:]
        u_n = self.sn * x_pad[:, :, :-2, 1:-1]
        u_s = self.ss * x_pad[:, :, 2:, 1:-1]
        u_c = self.sc * x
        Ax = u_w + u_e + u_n + u_s + u_c

        if xdim == 1:
            return Ax.flatten()
        elif xdim == 2:
            return Ax.view(batch_size, -1)
        elif xdim == 3:
            return Ax.view(batch_size, num_channels, -1)
        else:
            return Ax

    def apply_A(self, X: torch.Tensor) -> torch.Tensor:
        """
        Batch matrix-vector product for the basis-utils interface.

        X:  (B, N, K)   — N = n*n grid nodes, K = number of vectors
        AX: (B, N, K)
        """
        B, N, K = X.shape
        n = int(round(N ** 0.5))
        assert n * n == N, f"N={N} is not a perfect square"
        X_img = X.transpose(1, 2).reshape(B, K, n, n)
        # The stencil is elementwise, so applying it to blocks of columns gives the
        # same result while bounding the padded temporaries (wide GenEO bases at
        # n=1023 otherwise need ~8 copies of the full basis).
        AX_img = torch.empty_like(X_img)
        for k0 in range(0, K, 32):
            AX_img[:, k0:k0 + 32] = self.forward(X_img[:, k0:k0 + 32])
        AX = AX_img.reshape(B, K, N).transpose(1, 2)
        return AX

    def diag_for_jacobi(self) -> torch.Tensor:
        """Return (B, N, 1) diagonal for weighted-Jacobi smoothing."""
        B = self.a.shape[0]
        return self.sc.view(B, -1, 1).abs().clamp_min(1e-30)

    def grid_shape(self):
        """Return (ny, nx) = (n, n) for the SupportsApplyA protocol."""
        return (self.n, self.n)

    def _apply_local_solver_to_patches(self, r_patches, B, C, s, K):
        """Apply the configured block-local solver without global averaging."""
        s = self.s
        K = self.K

        if self.subdomain_solver == "cholesky" and self.cholesky_backend == "blocktri":
            return self._apply_asm_blocktri_cholesky(r_patches, B, C, s, K)
        if self.subdomain_solver == "cholesky":
            return self._apply_asm_cholesky(r_patches, B, C, s, K)
        elif self.subdomain_solver == "jacobi_pcg":
            # (B, C, s*s, K) → (B, K, C, s, s)
            rhs = r_patches.permute(0, 3, 1, 2).reshape(B, K, C, s, s).contiguous()
            z = self._apply_jacobi_pcg_subdomain(rhs)  # (B, K, C, s, s)
            # (B, K, C, s, s) → (B, C, s*s, K)
            return z.permute(0, 2, 3, 4, 1).contiguous().reshape(B, C, s * s, K)
        else:  # gmg_vcycle
            # (B, C, s*s, K) → (B, K, C, s, s)
            rhs = r_patches.permute(0, 3, 1, 2).reshape(B, K, C, s, s).contiguous()
            z = self._apply_gmg_vcycle_subdomain(rhs)  # (B, K, C, s, s)
            # (B, K, C, s, s) → (B, C, s*s, K)
            return z.permute(0, 2, 3, 4, 1).contiguous().reshape(B, C, s * s, K)

    def apply_asm(self, r):
        """Apply the averaged additive-Schwarz approximate inverse ``M^{-1}``."""
        self._require_local_solver("apply_asm")
        r = r.to(self.dtype)

        B, C, n, _ = r.shape
        s = self.s
        K = self.K

        r_patches = self._unfold_subdomains(r)  # (B, C, s*s, K)
        z_patches = self._apply_local_solver_to_patches(r_patches, B, C, s, K)

        return self._fold_subdomains(z_patches) / self.overlap_mask

    def apply_asm_transpose(self, r):
        """Apply the Euclidean adjoint ``M^{-T}`` of averaged ASM.

        With ``U`` denoting patch extraction, ``U^T`` patch folding, and
        ``D`` the overlap-count diagonal, the implemented averaged ASM is

            M^{-1} = D^{-1} U^T B^{-1} U.

        The exact adjoint is therefore

            M^{-T} = U^T B^{-T} U D^{-1}.

        The Cholesky and symmetric GMG V-cycle local actions used by the paper
        satisfy ``B^{-T}=B^{-1}``.  Averaging must consequently be applied to
        the *input* of the transpose, not to its output.  Fixed-iteration PCG
        has right-hand-side-dependent coefficients and is not a linear map, so
        it has no single matrix transpose and is rejected here.
        """
        self._require_local_solver("apply_asm_transpose")
        if self.subdomain_solver == "jacobi_pcg":
            raise NotImplementedError(
                "apply_asm_transpose is undefined for the nonlinear "
                "fixed-iteration jacobi_pcg local action"
            )
        r = r.to(self.dtype)
        B, C, n, _ = r.shape
        s = self.s
        K = self.K
        weighted = r / self.overlap_mask
        r_patches = self._unfold_subdomains(weighted)
        z_patches = self._apply_local_solver_to_patches(r_patches, B, C, s, K)
        return self._fold_subdomains(z_patches)

    def asm_smooth(self, x, f, M_smooth=1, omega=0.8):
        """ASM smoothing with under-relaxation."""
        self._require_local_solver("asm_smooth")
        x = x.to(self.dtype)
        f = f.to(self.dtype)
        for _ in range(M_smooth):
            Ax = self.forward(x)
            r = f - Ax
            z = self.apply_asm(r)
            x = x + omega * z
        return x
