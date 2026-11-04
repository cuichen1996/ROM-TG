import os
import math
import inspect
import warnings
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from learned_tg.model.losses import primitives as coarse_loss
from learned_tg.model.losses.primitives import require_finite
from learned_tg.model.operators.mobility_coefficients import build_mobility_coefficients
from learned_tg.model.two_phase import MobilityFVMOperatorASM, StandardGMGHierarchy, TwoPhaseClosedDropletFVM, _apply_A_with_operator
from learned_tg.data.whitening_truncation import adaptive_truncation_rank, apply_adaptive_padding, compute_a_whitened_basis_with_eigvals


SUPPORTED_TWO_PHASE_SHAPES = {
    "random_field",
    "fourier",
    "interface_random",
    "circle",
    "ellipse",
    "star",
    "flower",
    "mixed",
}

SUPPORTED_SNAPSHOT_SMOOTHERS = {"asm", "jacobi", "gmg"}


def _validate_positive_count(name, value):
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _as_device(device):
    return device if isinstance(device, torch.device) else torch.device(device)


def _empty_cuda_cache_if_needed(device):
    device = _as_device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _detach_cpu(tensor):
    return tensor.detach().cpu()


def _log_message(logger, message: str):
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def _operator_accepts_build_local_solver() -> bool:
    try:
        return "build_local_solver" in inspect.signature(MobilityFVMOperatorASM.__init__).parameters
    except Exception:
        return False


def _snapshot_smoother_from_config(config) -> str:
    if "snapshot_smooth_mode" in config and str(config.get("snapshot_smoother", "asm")).lower() == "asm":
        smoother = str(config.get("snapshot_smooth_mode"))
    elif (
        "snapshot_smoothing" in config
        and str(config.get("snapshot_smoother", "asm")).lower() == "asm"
        and str(config.get("snapshot_smoothing")).lower() in SUPPORTED_SNAPSHOT_SMOOTHERS
    ):
        smoother = str(config.get("snapshot_smoothing"))
    else:
        smoother = str(config.get("snapshot_smoother", "asm"))
    smoother = smoother.lower()
    if smoother in {"asm_avg", "ras"}:
        smoother = "asm"
    if smoother not in SUPPORTED_SNAPSHOT_SMOOTHERS:
        raise ValueError(
            "snapshot_smoother must be one of {'asm', 'jacobi', 'gmg'}, "
            f"got {smoother!r}"
        )
    return smoother


def _float_tag(value) -> str:
    return f"{float(value):.3g}".replace("+", "").replace("-", "m").replace(".", "p")


def _snapshot_smoothing_cache_tag(config) -> str:
    smoother = _snapshot_smoother_from_config(config)
    if smoother == "asm":
        return ""
    if smoother == "jacobi":
        return "_smjacobi"
    return (
        "_smgmg"
        f"_pre{int(config.get('snapshot_gmg_pre_smooth_steps', 2))}"
        f"_post{int(config.get('snapshot_gmg_post_smooth_steps', 2))}"
        f"_om{_float_tag(config.get('snapshot_gmg_omega', 0.8))}"
    )


class _LightweightMobilityFVMOperator:
    """Minimal two-phase cell-centered FVM operator for training losses.

    It implements only the pieces needed by _apply_A_with_operator and
    prepare_two_phase_coarse_basis when basis_smooth_mode is "none" or
    "jacobi": attributes nx/ny/N/device/dtype/diag and forward/_to_4d.
    It deliberately does not build ASM local Cholesky factors.
    """

    def __init__(
        self,
        lam,
        hx,
        hy,
        p_left=1.0,
        p_right=0.0,
        phi=None,
        transmissibility="harmonic",
        **_,
    ):
        if lam.dim() == 2:
            lam = lam[None, None, :, :]
        elif lam.dim() == 3:
            lam = lam[:, None, :, :]
        elif lam.dim() != 4:
            raise ValueError(f"lam must have dim 2, 3, or 4, got shape={tuple(lam.shape)}")
        self.lam = lam.to(dtype=torch.float64)
        self.device = self.lam.device
        self.dtype = self.lam.dtype
        self.B = int(self.lam.shape[0])
        self.ny = int(self.lam.shape[-2])
        self.nx = int(self.lam.shape[-1])
        self.N = self.nx * self.ny
        self.hx = float(hx)
        self.hy = float(hy)
        self.p_left = float(p_left)
        self.p_right = float(p_right)
        self.phi = None if phi is None else phi.to(
            device=self.device,
            dtype=self.dtype,
        )
        self.transmissibility = str(transmissibility).lower()
        self._build_coefficients()

    def _build_coefficients(self):
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
            setattr(self, name, tensor)

    def _to_4d(self, x):
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

    def _from_4d(self, x4, original_shape):
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

    def forward(self, x):
        x4, original_shape = self._to_4d(x)
        Ax = self.diag * x4
        Ax[..., :, 1:] -= self.aw[..., :, 1:] * x4[..., :, :-1]
        Ax[..., :, :-1] -= self.ae[..., :, :-1] * x4[..., :, 1:]
        Ax[..., 1:, :] -= self.as_[..., 1:, :] * x4[..., :-1, :]
        Ax[..., :-1, :] -= self.an[..., :-1, :] * x4[..., 1:, :]
        return self._from_4d(Ax, original_shape)

    def rhs(self, q=None):
        if q is None:
            return self.bc_rhs
        q4, _ = self._to_4d(q)
        return q4 + self.bc_rhs

    def apply_jacobi(self, r):
        r4, original_shape = self._to_4d(r)
        z = r4 / self.diag.clamp_min(1e-30)
        return self._from_4d(z, original_shape)

    def apply_asm(self, *_args, **_kwargs):
        raise RuntimeError("Lightweight operator has no ASM local solver. Use MobilityFVMOperatorASM with build_local_solver=True.")

    def asm_smooth(self, *_args, **_kwargs):
        raise RuntimeError("Lightweight operator has no ASM local solver. Use MobilityFVMOperatorASM with build_local_solver=True.")


def _make_mobility_operator(
    lam: torch.Tensor,
    config,
    *,
    phi: torch.Tensor = None,
    build_local_solver: bool = True,
    force_full_operator: bool = False,
):
    """Build the two-phase pressure operator with optional lightweight mode."""
    if lam.dim() == 3:
        lam = lam.unsqueeze(1)
    nx = int(lam.shape[-1])
    ny = int(lam.shape[-2])
    kwargs = dict(
        lam=lam.to(torch.float64),
        hx=float(config.get("Lx", 1.0)) / float(nx),
        hy=float(config.get("Ly", 1.0)) / float(ny),
        p_left=float(config.get("p_in", 1.0)),
        p_right=float(config.get("p_out", 0.0)),
        kx_sub=int(config.get("k_1d", config.get("kx_sub", 8))),
        ky_sub=int(config.get("k_1d", config.get("ky_sub", 8))),
        target_overlap=float(config.get("target_overlap", 0.3)),
        phi=phi,
        transmissibility=str(config.get("transmissibility", "harmonic")),
    )
    if not bool(build_local_solver) and not bool(force_full_operator):
        return _LightweightMobilityFVMOperator(**kwargs)
    if _operator_accepts_build_local_solver():
        kwargs["build_local_solver"] = bool(build_local_solver)
        kwargs["subdomain_solver"] = str(config.get("subdomain_solver", "cholesky"))
        kwargs["jacobi_pcg_iters"] = int(config.get("jacobi_pcg_iters", 10))
    try:
        return MobilityFVMOperatorASM(**kwargs)
    except TypeError:
        kwargs.pop("build_local_solver", None)
        kwargs.pop("subdomain_solver", None)
        kwargs.pop("jacobi_pcg_iters", None)
        return MobilityFVMOperatorASM(**kwargs)


@torch.no_grad()
def smooth_two_phase_snapshot_family(
    op,
    e0: torch.Tensor,
    *,
    M_smooth: int,
    omega: float = 0.8,
    smoother: str = "asm",
    config=None,
) -> torch.Tensor:
    """
    Smooth random error probes for two-phase family generation.

    The default "asm" branch preserves the historical data-generation path.
    "jacobi" applies weighted Jacobi to A e = 0, and "gmg" applies repeated
    standard GMG error-propagation steps using one V-cycle per step.
    """
    smoother = str(smoother).lower()
    if smoother not in SUPPORTED_SNAPSHOT_SMOOTHERS:
        raise ValueError(
            "snapshot smoother must be one of {'asm', 'jacobi', 'gmg'}, "
            f"got {smoother!r}"
        )
    M_smooth = int(M_smooth)
    if M_smooth <= 0:
        return e0.to(device=op.device, dtype=op.dtype)

    x = e0.to(device=op.device, dtype=op.dtype)
    if smoother == "asm":
        f_zero = torch.zeros_like(x)
        try:
            return op.asm_smooth(x, f_zero, steps=M_smooth, omega=float(omega))
        except TypeError:
            return op.asm_smooth(x, f_zero, M_smooth=M_smooth, omega=float(omega))

    if smoother == "jacobi":
        for _ in range(M_smooth):
            r = -op.forward(x)
            x = x + float(omega) * op.apply_jacobi(r)
        return x

    config = {} if config is None else dict(config)
    gmg = StandardGMGHierarchy(
        op,
        num_levels=config.get("snapshot_gmg_num_levels"),
        min_coarse_size=int(config.get("snapshot_gmg_min_coarse_size", 4)),
        smoother=str(config.get("snapshot_gmg_smoother", "jacobi")),
        pre_smooth_steps=int(config.get("snapshot_gmg_pre_smooth_steps", 2)),
        post_smooth_steps=int(config.get("snapshot_gmg_post_smooth_steps", 2)),
        omega=float(config.get("snapshot_gmg_omega", 0.8)),
        coarse_solve=str(config.get("snapshot_gmg_coarse_solve", "direct")),
    )
    outer_omega = float(config.get("snapshot_gmg_outer_omega", omega))
    for _ in range(M_smooth):
        r = -op.forward(x)
        x = x + outer_omega * gmg.apply(r)
    return x


def _estimate_tensors_size_bytes(tensors) -> int:
    total = 0
    for tensor in tensors:
        if torch.is_tensor(tensor):
            total += int(tensor.numel()) * int(tensor.element_size())
    return int(total)


def _maybe_preload_dataset_tensors_to_gpu(
    tensors,
    *,
    device,
    enabled: bool,
    safety_margin_gb: float = 8.0,
    logger=None,
):
    device = _as_device(device)
    tensors = tuple(tensors)
    if not bool(enabled):
        _log_message(logger, "[TwoPhaseData] dataset preload to GPU disabled.")
        return tensors, False
    if device.type != "cuda" or not torch.cuda.is_available():
        _log_message(logger, "[TwoPhaseData] dataset preload requested, but compute device is not CUDA; keeping tensors on CPU.")
        return tensors, False

    required_bytes = _estimate_tensors_size_bytes(tensors)
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device.index if device.index is not None else torch.cuda.current_device())
    except Exception:
        props = torch.cuda.get_device_properties(device)
        total_bytes = int(props.total_memory)
        free_bytes = total_bytes - int(torch.cuda.memory_reserved(device))
    safety_bytes = int(float(safety_margin_gb) * (1024 ** 3))
    if free_bytes - required_bytes <= safety_bytes:
        _log_message(
            logger,
            "[TwoPhaseData] dataset preload to GPU skipped: "
            f"required={required_bytes / 1024 ** 3:.2f} GiB, "
            f"free={free_bytes / 1024 ** 3:.2f} GiB, "
            f"safety_margin={float(safety_margin_gb):.2f} GiB.",
        )
        return tensors, False

    moved = tuple(
        tensor.to(device=device, non_blocking=False) if torch.is_tensor(tensor) else tensor
        for tensor in tensors
    )
    _log_message(
        logger,
        "[TwoPhaseData] dataset preload to GPU: True | "
        f"tensors={len(tensors)} | size={required_bytes / 1024 ** 3:.2f} GiB | device={device}.",
    )
    return moved, True


def _loss_type(config) -> str:
    return str(config.get("two_phase_loss_type", "capture_whiten")).lower()


def _needs_svd_basis(config) -> bool:
    return bool(config.get("compute_svd_basis", False)) or _loss_type(config) == "svd"


def _needs_whitened_family(config) -> bool:
    return bool(config.get("compute_whitened_family", False)) or (
        bool(config.get("cache_capture_whiten", True)) and _loss_type(config) == "capture_whiten"
    )


def _whiten_rank_from_config(config, num_nodes: int, num_snapshots: int) -> int:
    rank = int(config.get("whiten_rank", config.get("m_dim", min(num_nodes, num_snapshots))))
    return max(1, min(rank, int(num_nodes), int(num_snapshots)))


def _compute_a_whitened_family_with_operator(op, e_family, *, rank: int, whiten_reg: float):
    e_family = e_family.to(device=op.device, dtype=torch.float64)
    AE = _apply_A_with_operator(op, e_family)
    return coarse_loss.compute_metric_whitened_snapshot_family(
        e_family,
        rank=rank,
        metric="a",
        AE=AE,
        reg=float(whiten_reg),
        output_layout="basis",
    )


# ---------------------------------------------------------------------------
# Adaptive whitening helpers
# ---------------------------------------------------------------------------

def _resolve_two_phase_whiten_rank_mode(config) -> str:
    return str(config.get("whiten_rank_mode", "fixed")).lower()


def _resolve_two_phase_adaptive_whiten_params(config, num_nodes: int, num_snapshots: int):
    min_rank = max(1, int(config.get("whiten_rank_min", 4)))
    max_rank = max(min_rank, int(config.get("whiten_rank_max", 30)))
    max_rank = min(max_rank, int(num_snapshots))
    floor_relative = float(config.get("whiten_rank_floor_relative", 1e-12))
    return min_rank, max_rank, floor_relative


def _resolve_two_phase_whiten_beta(config) -> float:
    """Return whiten_beta from config, defaulting to 1.0 (full whitening)."""
    beta = float(config.get("whiten_beta", 1.0))
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"whiten_beta must be in [0, 1], got {beta}")
    return beta


def _two_phase_whiten_mode_cache_tag(config) -> str:
    mode = _resolve_two_phase_whiten_rank_mode(config)
    if mode == "adaptive":
        beta = _resolve_two_phase_whiten_beta(config)
        return f"_whitenadaptive_beta{round(beta * 100):03d}"
    return ""


@torch.no_grad()
def _compute_two_phase_whitened_family_adaptive(
    e_family: torch.Tensor,
    a_field: torch.Tensor,
    phi_field: torch.Tensor,
    config,
    *,
    k_max: int,
    min_rank: int,
    max_rank: int,
    floor_relative: float,
    beta: float = 1.0,
    compute_device="cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Adaptive-rank A-metric whitening for two-phase datasets.

    Returns
    -------
    e_white : (B, N, k_max) zero-padded whitened basis
    k_per_sample : (B,) int64 per-sample truncation ranks
    eigvals_per_sample : (B, q) Gram eigenvalues (descending)
    """
    device = _as_device(compute_device)
    chunk_size = int(config.get("whiten_cache_chunk_size", config.get("chunk_size", 20)))
    whiten_reg = float(config.get("whiten_reg", 1e-10))
    beta = float(beta)
    white_parts, k_parts, eig_parts = [], [], []

    for start_idx in range(0, int(e_family.shape[0]), chunk_size):
        end_idx = min(start_idx + chunk_size, int(e_family.shape[0]))
        lam_chunk = a_field[start_idx:end_idx].to(device=device, dtype=torch.float64)
        phi_chunk = phi_field[start_idx:end_idx].to(device=device, dtype=torch.float64)
        E_chunk = e_family[start_idx:end_idx].to(device=device, dtype=torch.float64)
        op = _make_mobility_operator(
            lam_chunk,
            config,
            phi=phi_chunk,
            build_local_solver=False,
        )
        AE = _apply_A_with_operator(op, E_chunk)
        U_k, eigvals = compute_a_whitened_basis_with_eigvals(
            E_chunk, AE, rank=k_max, reg=whiten_reg, beta=beta
        )
        U_padded, k_chunk = apply_adaptive_padding(
            U_k, eigvals,
            floor_relative=floor_relative,
            min_rank=min_rank,
            max_rank=max_rank,
        )
        white_parts.append(U_padded.detach().cpu())
        k_parts.append(k_chunk.cpu())
        eig_parts.append(eigvals.detach().cpu())
        del op, E_chunk, lam_chunk, phi_chunk, AE, U_k, U_padded, k_chunk, eigvals
        _empty_cuda_cache_if_needed(device)

    return (
        torch.cat(white_parts, dim=0).to(torch.float64),
        torch.cat(k_parts, dim=0),
        torch.cat(eig_parts, dim=0).to(torch.float64),
    )


def _shape_key(shape: str) -> str:
    key = str(shape).lower()
    if key == "flower":
        key = "star"
    if key in {"fourier", "interface_random"}:
        key = "random_field"
    if key not in SUPPORTED_TWO_PHASE_SHAPES:
        raise ValueError(
            f"Unsupported two-phase droplet shape {shape!r}; "
            f"expected one of {sorted(SUPPORTED_TWO_PHASE_SHAPES)}."
        )
    return key


def _sample_uniform(generator: torch.Generator, low: float, high: float) -> float:
    return float(low + (high - low) * torch.rand((), generator=generator).item())


def _sample_int(generator: torch.Generator, values) -> int:
    values = list(values)
    idx = int(torch.randint(0, len(values), (), generator=generator).item())
    return int(values[idx])


def _phi_is_closed_island(phi: torch.Tensor, min_area_cells: int = 8) -> bool:
    phi2 = phi.squeeze()
    if phi2.dim() != 2:
        raise ValueError(f"Expected squeezable 2D phi, got {tuple(phi.shape)}")

    inside = phi2 < 0.0
    if int(inside.sum().item()) < int(min_area_cells):
        return False
    touches_boundary = (
        inside[0, :].any()
        | inside[-1, :].any()
        | inside[:, 0].any()
        | inside[:, -1].any()
    )
    return not bool(touches_boundary.item())


def _closed_island_batch_mask(phi: torch.Tensor, min_area_cells: int = 8) -> torch.Tensor:
    if phi.dim() == 4:
        phi2 = phi[:, 0]
    elif phi.dim() == 3:
        phi2 = phi
    else:
        raise ValueError(f"Expected phi with shape (B,1,n,n) or (B,n,n), got {tuple(phi.shape)}")

    inside = phi2 < 0.0
    enough_area = inside.reshape(inside.shape[0], -1).sum(dim=1) >= int(min_area_cells)
    touches_boundary = (
        inside[:, 0, :].any(dim=1)
        | inside[:, -1, :].any(dim=1)
        | inside[:, :, 0].any(dim=1)
        | inside[:, :, -1].any(dim=1)
    )
    return enough_area & (~touches_boundary)


def generate_random_two_phase_closed_droplet_fields_with_topology(
    *,
    batch_size: int,
    solver: TwoPhaseClosedDropletFVM,
    generator: torch.Generator,
    alpha_range: Tuple[float, float] = (0.05, 0.35),
    r0_range: Tuple[float, float] = (0.12, 0.25),
    center_range: Tuple[float, float] = (0.30, 0.70),
    max_retries: int = 200,
    num_fourier_modes: int = 4,
    coeff_scale: float = 0.08,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Generate closed-droplet level sets using the same spirit as
    generate_random_interface_coeffs_with_topology: a random polar interface
    R(theta) around a random center, filtered by an island topology check.
    """
    batch_size = _validate_positive_count("batch_size", batch_size)
    num_fourier_modes = max(int(num_fourier_modes), 1)
    coeff_scale = float(coeff_scale)
    min_area_cells = max(8, int(0.002 * solver.nx * solver.ny))
    device = solver.device
    dtype = solver.dtype

    accepted_phi = []
    accepted_count = 0
    retries = 0

    while accepted_count < batch_size:
        retries += 1
        if retries > int(max_retries):
            raise RuntimeError(
                f"Failed to generate {batch_size} closed two-phase droplets within "
                f"{max_retries} retries; accepted {accepted_count}."
            )

        remaining = batch_size - accepted_count
        candidate_count = max(remaining, min(batch_size, 32))

        centers = (
            center_range[0]
            + (center_range[1] - center_range[0])
            * torch.rand(candidate_count, 2, 1, 1, generator=generator, dtype=dtype)
        ).to(device=device)
        R0 = (
            r0_range[0]
            + (r0_range[1] - r0_range[0])
            * torch.rand(candidate_count, 1, 1, 1, generator=generator, dtype=dtype)
        ).to(device=device)

        Xc = solver.X - centers[:, 0:1]
        Yc = solver.Y - centers[:, 1:2]
        radius = torch.sqrt(Xc ** 2 + Yc ** 2 + 1e-30)
        theta = torch.atan2(Yc, Xc)

        radial_factor = torch.ones(candidate_count, 1, solver.ny, solver.nx, device=device, dtype=dtype)
        for mode_idx in range(1, num_fourier_modes + 1):
            a_coeff = (
                coeff_scale
                * torch.randn(candidate_count, 1, 1, 1, generator=generator, dtype=dtype)
            ).to(device=device)
            b_coeff = (
                coeff_scale
                * torch.randn(candidate_count, 1, 1, 1, generator=generator, dtype=dtype)
            ).to(device=device)
            radial_factor = radial_factor + a_coeff * torch.cos(mode_idx * theta)
            radial_factor = radial_factor + b_coeff * torch.sin(mode_idx * theta)

        radial_factor = radial_factor.clamp_min(0.35)
        phi_candidate = radius - R0 * radial_factor
        accept_mask = _closed_island_batch_mask(phi_candidate, min_area_cells=min_area_cells)

        if accept_mask.any():
            accepted = phi_candidate[accept_mask]
            accepted_phi.append(accepted[:remaining].detach())
            accepted_count += int(min(remaining, accepted.shape[0]))

    phi = torch.cat(accepted_phi, dim=0)[:batch_size].contiguous()
    meta = {
        "shape": "random_field",
        "num_fourier_modes": int(num_fourier_modes),
        "coeff_scale": float(coeff_scale),
        "alpha_range": tuple(float(v) for v in alpha_range),
        "r0_range": tuple(float(v) for v in r0_range),
        "center_range": tuple(float(v) for v in center_range),
    }
    return phi, meta


def _sample_closed_droplet_phi(
    solver: TwoPhaseClosedDropletFVM,
    *,
    shape: str,
    generator: torch.Generator,
    alpha_range: Tuple[float, float],
    r0_range: Tuple[float, float],
    center_range: Tuple[float, float],
    max_retries: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    shape = _shape_key(shape)
    choices = ("circle", "ellipse", "star")
    min_area_cells = max(8, int(0.002 * solver.nx * solver.ny))

    for _ in range(int(max_retries)):
        active_shape = choices[_sample_int(generator, range(len(choices)))] if shape == "mixed" else shape
        xc = _sample_uniform(generator, center_range[0], center_range[1])
        yc = _sample_uniform(generator, center_range[0], center_range[1])
        rotation = _sample_uniform(generator, 0.0, 2.0 * math.pi)
        alpha = _sample_uniform(generator, alpha_range[0], alpha_range[1])
        r0 = _sample_uniform(generator, r0_range[0], r0_range[1])
        params: Dict[str, object] = {
            "shape": active_shape,
            "xc": xc,
            "yc": yc,
            "R0": r0,
            "rotation": rotation,
        }

        if active_shape == "circle":
            params["alpha"] = 0.0
            phi = solver.initial_droplet_phi(
                shape="circle",
                xc=xc,
                yc=yc,
                R0=r0,
            )
        elif active_shape == "ellipse":
            aspect = _sample_uniform(generator, 0.55, 1.65)
            a = r0 * math.sqrt(aspect)
            b = r0 / math.sqrt(aspect)
            max_radius = max(a, b)
            if max_radius >= min(xc, 1.0 - xc, yc, 1.0 - yc) - 2.0 * max(solver.hx, solver.hy):
                continue
            params.update({"ellipse_a": a, "ellipse_b": b, "alpha": 0.0})
            phi = solver.initial_droplet_phi(
                shape="ellipse",
                xc=xc,
                yc=yc,
                ellipse_a=a,
                ellipse_b=b,
                rotation=rotation,
            )
        else:
            star_k = _sample_int(generator, (3, 4, 5, 6))
            max_radius = r0 * (1.0 + alpha)
            if max_radius >= min(xc, 1.0 - xc, yc, 1.0 - yc) - 2.0 * max(solver.hx, solver.hy):
                continue
            params.update({"star_k": star_k, "alpha": alpha})
            phi = solver.initial_droplet_phi(
                shape="star",
                xc=xc,
                yc=yc,
                R0=r0,
                star_k=star_k,
                alpha=alpha,
                rotation=rotation,
            )

        if _phi_is_closed_island(phi, min_area_cells=min_area_cells):
            return phi, params

    raise RuntimeError(
        f"Failed to generate a closed two-phase droplet after {max_retries} retries. "
        "Try reducing r0_range/alpha_range or widening center_range."
    )


@torch.no_grad()
def generate_two_phase_family_data(
    *,
    config,
    num_samples: int,
    n: int,
    M_smooth: int,
    num_probes: int,
    device="cpu",
    chunk_size: int = 20,
):
    num_samples = _validate_positive_count("num_samples", num_samples)
    n = _validate_positive_count("n", n)
    M_smooth = int(M_smooth)
    num_probes = _validate_positive_count("num_probes", num_probes)
    chunk_size = _validate_positive_count("chunk_size", chunk_size)
    device = _as_device(device)

    shape = _shape_key(config.get("two_phase_shape", config.get("shape", "random_field")))
    max_retries = int(config.get("max_topology_retries", config.get("max_shape_retries", 200)))
    alpha_range = tuple(config.get("alpha_range", (0.05, 0.35)))
    r0_range = tuple(config.get("r0_range", (0.12, 0.25)))
    center_range = tuple(config.get("center_range", (0.30, 0.70)))
    omega = float(config.get("snapshot_smooth_omega", 0.8))
    snapshot_smoother = _snapshot_smoother_from_config(config)

    compute_svd_basis = _needs_svd_basis(config)
    compute_whitened_family = _needs_whitened_family(config)
    svd_metric = str(config.get("svd_basis_metric", "l2")).lower()
    if svd_metric not in {"l2", "a"}:
        raise ValueError("svd_basis_metric must be one of {'l2', 'a'}")
    m_dim = int(config.get("m_dim", min(num_probes, n * n)))
    svd_rank = max(1, min(m_dim, num_probes, n * n))
    whiten_rank = _whiten_rank_from_config(config, n * n, num_probes)
    whiten_reg = float(config.get("whiten_reg", 1e-10))

    solver = TwoPhaseClosedDropletFVM(
        nx=n,
        ny=int(config.get("ny", n)),
        Lx=float(config.get("Lx", 1.0)),
        Ly=float(config.get("Ly", 1.0)),
        mu_background=float(config.get("mu_background", 1.0)),
        mu_droplet=float(config.get("mu_droplet", 1e-4)),
        permeability=float(config.get("permeability", 1.0)),
        gap_width=config.get("gap_width", None),
        p_in=float(config.get("p_in", 1.0)),
        p_out=float(config.get("p_out", 0.0)),
        device=str(device),
        dtype=torch.float64,
        kx_sub=int(config.get("k_1d", config.get("kx_sub", 8))),
        ky_sub=int(config.get("k_1d", config.get("ky_sub", 8))),
        target_overlap=float(config.get("target_overlap", 0.3)),
        transmissibility=str(config.get("transmissibility", "harmonic")),
    )
    if solver.nx != solver.ny:
        raise ValueError("two-phase training data currently expects a square nx=ny grid")

    seed = int(config.get("seed", 1234))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + int(config.get("dataset_seed_offset", 0)))

    phi_parts = []
    lambda_parts = []
    e_family_parts = []
    svd_parts = [] if compute_svd_basis else None
    e_white_parts = [] if compute_whitened_family else None
    shape_params = []

    print(
        "[TwoPhaseData] Generating "
        f"{num_samples} closed-droplet samples, n={n}, q={num_probes}, "
        f"M_smooth={M_smooth}, smoother={snapshot_smoother}, shape={shape}."
    )

    for start_idx in range(0, num_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, num_samples)
        current_batch = end_idx - start_idx

        if shape == "random_field":
            phi_chunk, random_field_meta = generate_random_two_phase_closed_droplet_fields_with_topology(
                batch_size=current_batch,
                solver=solver,
                generator=generator,
                alpha_range=alpha_range,
                r0_range=r0_range,
                center_range=center_range,
                max_retries=max_retries,
                num_fourier_modes=int(config.get("random_field_num_modes", 4)),
                coeff_scale=float(config.get("random_field_coeff_scale", 0.08)),
            )
            params_chunk = [dict(random_field_meta) for _ in range(current_batch)]
        else:
            phi_chunk_list = []
            params_chunk = []
            for _ in range(current_batch):
                phi_i, params_i = _sample_closed_droplet_phi(
                    solver,
                    shape=shape,
                    generator=generator,
                    alpha_range=alpha_range,
                    r0_range=r0_range,
                    center_range=center_range,
                    max_retries=max_retries,
                )
                phi_chunk_list.append(phi_i)
                params_chunk.append(params_i)
            phi_chunk = torch.cat(phi_chunk_list, dim=0)

        phi_chunk = phi_chunk.to(device=device, dtype=torch.float64)
        lambda_chunk = solver.mobility(phi_chunk)
        op = _make_mobility_operator(
            lambda_chunk,
            config,
            phi=phi_chunk,
            build_local_solver=(snapshot_smoother == "asm"),
            force_full_operator=(snapshot_smoother == "gmg"),
        )
        lambda_chunk = op.lam

        e0 = torch.randn(
            current_batch,
            num_probes,
            n,
            n,
            device=device,
            dtype=torch.float64,
        )
        e_smooth = smooth_two_phase_snapshot_family(
            op,
            e0,
            M_smooth=M_smooth,
            omega=omega,
            smoother=snapshot_smoother,
            config=config,
        )
        e_family_chunk = e_smooth.reshape(current_batch, num_probes, n * n).transpose(1, 2).contiguous()

        phi_parts.append(_detach_cpu(phi_chunk))
        lambda_parts.append(_detach_cpu(lambda_chunk))
        e_family_parts.append(_detach_cpu(e_family_chunk))
        shape_params.extend(params_chunk)

        if svd_parts is not None:
            if svd_metric == "l2":
                U, _, _ = torch.linalg.svd(e_family_chunk, full_matrices=False)
                svd_basis_chunk = U[:, :, :svd_rank].contiguous()
            else:
                AE = _apply_A_with_operator(op, e_family_chunk)
                gram = 0.5 * (
                    torch.matmul(e_family_chunk.transpose(1, 2), AE)
                    + torch.matmul(AE.transpose(1, 2), e_family_chunk)
                )
                eigvals, eigvecs = torch.linalg.eigh(gram)
                eigvals, sort_idx = torch.sort(eigvals, dim=-1, descending=True)
                eigvecs = torch.gather(
                    eigvecs,
                    -1,
                    sort_idx.unsqueeze(-2).expand(-1, eigvecs.shape[1], -1),
                )
                eigvals = eigvals[:, :svd_rank].clamp_min(float(config.get("svd_basis_reg", 1e-12)))
                eigvecs = eigvecs[:, :, :svd_rank]
                svd_basis_chunk = torch.matmul(
                    e_family_chunk,
                    eigvecs * eigvals.rsqrt().unsqueeze(-2),
                ).contiguous()
            svd_parts.append(_detach_cpu(require_finite(svd_basis_chunk, "svd_basis_chunk")))

        if e_white_parts is not None:
            e_white_chunk = _compute_a_whitened_family_with_operator(
                op,
                e_family_chunk,
                rank=whiten_rank,
                whiten_reg=whiten_reg,
            )
            e_white_parts.append(_detach_cpu(e_white_chunk))

        del op, e0, e_smooth, e_family_chunk, phi_chunk, lambda_chunk
        _empty_cuda_cache_if_needed(device)
        print(f"[TwoPhaseData] Progress: {end_idx}/{num_samples}")

    phi = torch.cat(phi_parts, dim=0)
    a_field = torch.cat(lambda_parts, dim=0)
    e_family = torch.cat(e_family_parts, dim=0)

    X = solver.X.detach().cpu()[0, 0]
    Y = solver.Y.detach().cpu()[0, 0]
    X_flat = X.flatten().view(1, -1, 1).expand(num_samples, -1, -1).contiguous()
    Y_flat = Y.flatten().view(1, -1, 1).expand(num_samples, -1, -1).contiguous()
    snapshot_smoothing_steps = torch.full((num_samples, 1), int(M_smooth), dtype=torch.long)

    payload = {
        "a_field": a_field.to(torch.float64),
        "phi": phi.to(torch.float64),
        "X_flat": X_flat.to(torch.float64),
        "Y_flat": Y_flat.to(torch.float64),
        "e_family": e_family.to(torch.float64),
        "snapshot_smoothing_steps": snapshot_smoothing_steps,
        "meta": {
            "problem_type": "two_phase_closed_droplet",
            "mu_background": float(config.get("mu_background", 1.0)),
            "mu_droplet": float(config.get("mu_droplet", 1e-4)),
            "mobility_background": float(solver.lambda1),
            "mobility_droplet": float(solver.lambda2),
            "lambda_contrast": float(max(solver.lambda1, solver.lambda2) / max(min(solver.lambda1, solver.lambda2), 1e-300)),
            "boundary_condition": "left/right Dirichlet, top/bottom no-flow",
            "topology": "closed_droplet",
            "q_source": 0.0,
            "shape": shape,
            "alpha_range": tuple(float(v) for v in alpha_range),
            "r0_range": tuple(float(v) for v in r0_range),
            "center_range": tuple(float(v) for v in center_range),
            "snapshot_smoother": snapshot_smoother,
            "snapshot_smoothing": (
                "MobilityFVMOperatorASM.asm_smooth"
                if snapshot_smoother == "asm"
                else f"{snapshot_smoother}_error_smoothing"
            ),
            "snapshot_smoothing_steps": int(M_smooth),
            "snapshot_smoothing_omega": omega,
            "snapshot_gmg_config": {
                "num_levels": config.get("snapshot_gmg_num_levels"),
                "min_coarse_size": int(config.get("snapshot_gmg_min_coarse_size", 4)),
                "smoother": str(config.get("snapshot_gmg_smoother", "jacobi")),
                "pre_smooth_steps": int(config.get("snapshot_gmg_pre_smooth_steps", 2)),
                "post_smooth_steps": int(config.get("snapshot_gmg_post_smooth_steps", 2)),
                "omega": float(config.get("snapshot_gmg_omega", 0.8)),
                "outer_omega": float(config.get("snapshot_gmg_outer_omega", omega)),
                "coarse_solve": str(config.get("snapshot_gmg_coarse_solve", "direct")),
            } if snapshot_smoother == "gmg" else None,
            "operator_type": "MobilityFVMOperatorASM",
            "transmissibility": str(config.get("transmissibility", "harmonic")),
        },
        "shape_params": shape_params,
    }
    if svd_parts is not None:
        payload["svd_basis"] = torch.cat(svd_parts, dim=0).to(torch.float64)
        payload["svd_rank"] = int(payload["svd_basis"].shape[-1])
        payload["svd_basis_metric"] = svd_metric
    if e_white_parts is not None:
        payload["e_family_white_a"] = torch.cat(e_white_parts, dim=0).to(torch.float64)
        payload["e_family_white_a_rank"] = int(payload["e_family_white_a"].shape[-1])
        payload["e_family_white_metric"] = "a"
    return payload


def _cache_filename(config, split, num_samples, n, M_smooth, num_probes):
    shape = _shape_key(config.get("two_phase_shape", config.get("shape", "random_field")))
    mu_bg = float(config.get("mu_background", 1.0))
    mu_drop = float(config.get("mu_droplet", 1e-4))
    k_1d = int(config.get("k_1d", config.get("kx_sub", 8)))
    m_dim = int(config.get("m_dim", min(num_probes, n * n)))
    svd_flag = "svd" if _needs_svd_basis(config) else "nosvd"
    white_flag = (
        f"whiteA{_whiten_rank_from_config(config, n * n, num_probes)}"
        if _needs_whitened_family(config)
        else "nowhiteA"
    )
    smoothing_flag = _snapshot_smoothing_cache_tag(config)
    whiten_mode_flag = _two_phase_whiten_mode_cache_tag(config)
    tx_mode = str(config.get("transmissibility", "harmonic")).lower()
    tx_flag = "" if tx_mode == "harmonic" else f"_tx{tx_mode}"
    return (
        f"dataset_{split}_N{int(num_samples)}_n{int(n)}_M{int(M_smooth)}_q{int(num_probes)}_"
        f"m{m_dim}_{shape}_mubg{mu_bg:.3g}_mud{mu_drop:.3g}_k{k_1d}_"
        f"{svd_flag}_{white_flag}{smoothing_flag}{whiten_mode_flag}{tx_flag}.pt"
    ).replace("+", "")


def _legacy_cache_filename(config, split, num_samples, n, M_smooth, num_probes):
    shape = _shape_key(config.get("two_phase_shape", config.get("shape", "random_field")))
    mu_bg = float(config.get("mu_background", 1.0))
    mu_drop = float(config.get("mu_droplet", 1e-4))
    k_1d = int(config.get("k_1d", config.get("kx_sub", 8)))
    m_dim = int(config.get("m_dim", min(num_probes, n * n)))
    svd_flag = "svd" if _needs_svd_basis(config) else "nosvd"
    return (
        f"dataset_{split}_N{int(num_samples)}_n{int(n)}_M{int(M_smooth)}_q{int(num_probes)}_"
        f"m{m_dim}_{shape}_mubg{mu_bg:.3g}_mud{mu_drop:.3g}_k{k_1d}_{svd_flag}.pt"
    ).replace("+", "")


def _ensure_svd_basis_if_requested(dataset_dict, config, file_path=None):
    if not _needs_svd_basis(config):
        return dataset_dict
    if dataset_dict.get("svd_basis") is not None:
        return dataset_dict

    e_family = dataset_dict["e_family"].to(torch.float64)
    rank = max(1, min(int(config.get("m_dim", e_family.shape[-1])), e_family.shape[1], e_family.shape[2]))
    U, _, _ = torch.linalg.svd(e_family, full_matrices=False)
    dataset_dict["svd_basis"] = U[:, :, :rank].contiguous().cpu()
    dataset_dict["svd_rank"] = int(rank)
    dataset_dict["svd_basis_metric"] = "l2"
    if file_path is not None:
        torch.save(dataset_dict, file_path)
    return dataset_dict


def _ensure_cached_two_phase_whitened_family(
    dataset_dict,
    config,
    *,
    file_path=None,
    split="train",
    compute_device="cpu",
):
    if not _needs_whitened_family(config):
        return dataset_dict

    e_family = dataset_dict["e_family"]
    logger = config.get("logger")
    whiten_rank_mode = _resolve_two_phase_whiten_rank_mode(config)
    min_rank, max_rank_adaptive, floor_relative = _resolve_two_phase_adaptive_whiten_params(
        config,
        num_nodes=int(e_family.shape[1]),
        num_snapshots=int(e_family.shape[2]),
    )

    whiten_beta = _resolve_two_phase_whiten_beta(config)

    if whiten_rank_mode == "adaptive":
        if config.get("whiten_rank") is not None:
            warnings.warn(
                "whiten_rank_mode='adaptive': 'whiten_rank' is ignored. "
                "Use whiten_rank_min / whiten_rank_max.",
                UserWarning,
                stacklevel=3,
            )
        k_max = int(max_rank_adaptive)
        # Skip if already computed with same mode and beta.
        if (
            dataset_dict.get("e_family_white_a") is not None
            and dataset_dict.get("whiten_rank_mode") == "adaptive"
            and dataset_dict.get("whiten_rank_per_sample_a") is not None
            and abs(float(dataset_dict.get("whiten_beta", 1.0)) - whiten_beta) < 1e-12
        ):
            return dataset_dict

        _log_message(
            logger,
            f"[{split}] Adaptive A-whitening (k_max={k_max}, "
            f"min={min_rank}, floor={floor_relative:.1e}, beta={whiten_beta:.4f}), "
            f"samples={int(e_family.shape[0])}.",
        )
        e_white, k_per_sample, eigvals = _compute_two_phase_whitened_family_adaptive(
            e_family,
            dataset_dict["a_field"],
            dataset_dict["phi"],
            config,
            k_max=k_max,
            min_rank=min_rank,
            max_rank=max_rank_adaptive,
            floor_relative=floor_relative,
            beta=whiten_beta,
            compute_device=compute_device,
        )
        dataset_dict["e_family_white_a"] = e_white
        dataset_dict["e_family_white_a_rank"] = k_max
        dataset_dict["e_family_white_metric"] = "a"
        dataset_dict["whiten_rank_per_sample_a"] = k_per_sample
        dataset_dict["whiten_eigvals_per_sample_a"] = eigvals
        dataset_dict["whiten_rank_mode"] = "adaptive"
        dataset_dict["whiten_beta"] = whiten_beta
        dataset_dict["whiten_mode_signature"] = f"adaptive_beta{whiten_beta:.2f}"
        if file_path is not None:
            torch.save(dataset_dict, file_path)
            k_stats = (
                f"min={int(k_per_sample.min())} "
                f"median={int(k_per_sample.float().median())} "
                f"max={int(k_per_sample.max())}"
            )
            _log_message(
                logger,
                f"[{split}] Adaptive two-phase cache written: {file_path} "
                f"(k_max={k_max}, beta={whiten_beta:.2f}, per-sample k: {k_stats})",
            )
        return dataset_dict

    # Fixed mode: existing logic unchanged for byte-identical compatibility.
    rank = _whiten_rank_from_config(config, int(e_family.shape[1]), int(e_family.shape[2]))
    existing = dataset_dict.get("e_family_white_a")
    if torch.is_tensor(existing) and existing.dim() == 3 and int(existing.shape[-1]) >= rank:
        if int(existing.shape[-1]) > rank:
            dataset_dict["e_family_white_a"] = existing[:, :, :rank].contiguous()
            dataset_dict["e_family_white_a_rank"] = int(rank)
            if file_path is not None:
                torch.save(dataset_dict, file_path)
        return dataset_dict

    _log_message(
        logger,
        f"[{split}] Computing cached A-whitened two-phase family: rank={rank}, "
        f"samples={int(e_family.shape[0])}.",
    )
    device = _as_device(compute_device)
    chunk_size = int(config.get("whiten_cache_chunk_size", config.get("chunk_size", 20)))
    whiten_reg = float(config.get("whiten_reg", 1e-10))
    a_field = dataset_dict["a_field"]
    phi_field = dataset_dict["phi"]
    white_parts = []
    for start_idx in range(0, int(e_family.shape[0]), chunk_size):
        end_idx = min(start_idx + chunk_size, int(e_family.shape[0]))
        lam_chunk = a_field[start_idx:end_idx].to(device=device, dtype=torch.float64)
        phi_chunk = phi_field[start_idx:end_idx].to(device=device, dtype=torch.float64)
        E_chunk = e_family[start_idx:end_idx].to(device=device, dtype=torch.float64)
        op = _make_mobility_operator(
            lam_chunk,
            config,
            phi=phi_chunk,
            build_local_solver=False,
        )
        E_white = _compute_a_whitened_family_with_operator(
            op,
            E_chunk,
            rank=rank,
            whiten_reg=whiten_reg,
        )
        white_parts.append(E_white.detach().cpu())
        del op, E_chunk, lam_chunk, phi_chunk, E_white
        _empty_cuda_cache_if_needed(device)
        _log_message(logger, f"[{split}] A-whiten cache progress: {end_idx}/{int(e_family.shape[0])}")

    dataset_dict["e_family_white_a"] = torch.cat(white_parts, dim=0).to(torch.float64)
    dataset_dict["e_family_white_a_rank"] = int(rank)
    dataset_dict["e_family_white_metric"] = "a"
    if file_path is not None:
        torch.save(dataset_dict, file_path)
        _log_message(logger, f"[{split}] Updated two-phase cache with e_family_white_a: {file_path}")
    return dataset_dict


def create_cached_two_phase_family_data_loader(
    config,
    batch_size,
    n,
    M_smooth,
    num_samples,
    num_probes,
    split="train",
    cache_dir=None,
    **_,
):
    compute_device = config.get("device", "cpu")
    logger = config.get("logger")
    cache_dir = cache_dir or config.get("dataset_cache_dir", "./datasets/two_phase_family")
    os.makedirs(cache_dir, exist_ok=True)

    num_samples = _validate_positive_count("num_samples", num_samples)
    batch_size = min(_validate_positive_count("batch_size", batch_size), num_samples)

    # Log adaptive whitening config
    _wmode = _resolve_two_phase_whiten_rank_mode(config)
    _wmin, _wmax, _wfloor = _resolve_two_phase_adaptive_whiten_params(
        config, num_nodes=n * n, num_snapshots=num_probes
    )
    _wbeta = _resolve_two_phase_whiten_beta(config)
    _log_message(
        logger,
        f"[WhitenConfig] mode={_wmode!r} "
        f"rank={config.get('whiten_rank', 'N/A')} "
        f"min={_wmin} max={_wmax} "
        f"floor_relative={_wfloor:.1e} "
        f"metric={config.get('whiten_metric', 'a')!r} "
        f"beta={_wbeta:.4f}",
    )

    filename = _cache_filename(config, split, num_samples, n, M_smooth, num_probes)
    file_path = os.path.join(cache_dir, filename)

    load_path = file_path
    if not os.path.exists(load_path):
        legacy_path = os.path.join(cache_dir, _legacy_cache_filename(config, split, num_samples, n, M_smooth, num_probes))
        if _snapshot_smoother_from_config(config) == "asm" and os.path.exists(legacy_path):
            load_path = legacy_path

    if os.path.exists(load_path):
        message = f"[{split}] Loading cached two-phase dataset from {load_path}"
        logger.info(message) if logger is not None else print(message)
        dataset_dict = torch.load(load_path, map_location="cpu")
        dataset_dict = _ensure_svd_basis_if_requested(dataset_dict, config, file_path=load_path)
        dataset_dict = _ensure_cached_two_phase_whitened_family(
            dataset_dict,
            config,
            file_path=file_path,
            split=split,
            compute_device=compute_device,
        )
        if load_path != file_path and not os.path.exists(file_path):
            torch.save(dataset_dict, file_path)
    else:
        message = f"[{split}] Two-phase cache not found. Generating {file_path}"
        logger.info(message) if logger is not None else print(message)
        generation_config = dict(config)
        split_seed_offsets = {"train": 0, "val": 100000, "valid": 100000, "validation": 100000, "test": 200000}
        generation_config["dataset_seed_offset"] = int(config.get("dataset_seed_offset", 0)) + split_seed_offsets.get(str(split), 300000)
        dataset_dict = generate_two_phase_family_data(
            config=generation_config,
            num_samples=num_samples,
            n=n,
            M_smooth=M_smooth,
            num_probes=num_probes,
            device=compute_device,
            chunk_size=int(config.get("chunk_size", 20)),
        )
        dataset_dict = _ensure_cached_two_phase_whitened_family(
            dataset_dict,
            config,
            file_path=None,
            split=split,
            compute_device=compute_device,
        )
        torch.save(dataset_dict, file_path)

    required = ("a_field", "phi", "X_flat", "Y_flat", "e_family", "snapshot_smoothing_steps")
    missing = [key for key in required if key not in dataset_dict]
    if missing:
        raise KeyError(f"Cached two-phase dataset is missing keys: {missing}")

    tensors = [
        dataset_dict["a_field"],
        dataset_dict["phi"],
        dataset_dict["X_flat"],
        dataset_dict["Y_flat"],
        dataset_dict["e_family"],
    ]
    if dataset_dict.get("e_family_white_a") is not None:
        tensors.append(dataset_dict["e_family_white_a"])
    if dataset_dict.get("svd_basis") is not None:
        tensors.append(dataset_dict["svd_basis"])
    tensors.append(dataset_dict["snapshot_smoothing_steps"])

    tensors, dataset_on_gpu = _maybe_preload_dataset_tensors_to_gpu(
        tensors,
        device=compute_device,
        enabled=bool(config.get("preload_dataset_to_gpu", True)),
        safety_margin_gb=float(config.get("preload_dataset_safety_margin_gb", 10.0)),
        logger=logger,
    )

    dataset = TensorDataset(*tensors)
    if dataset_on_gpu:
        num_workers = 0
        pin_memory = False
        persistent_workers = False
        prefetch_factor = None
    else:
        num_workers = int(config.get("num_workers", config.get("dataloader_num_workers", 4)))
        pin_memory_config = config.get("pin_memory", None)
        pin_memory = (
            _as_device(compute_device).type == "cuda"
            if pin_memory_config is None
            else bool(pin_memory_config)
        )
        persistent_workers = bool(config.get("persistent_workers", True)) and num_workers > 0
        prefetch_factor = int(config.get("prefetch_factor", 2)) if num_workers > 0 else None

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(**loader_kwargs)
    return loader, file_path
