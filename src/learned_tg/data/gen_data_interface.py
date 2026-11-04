import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from learned_tg.model.interface import CCFVM_ASM, _compute_metric_whitened_snapshot_family
from learned_tg.model.losses.primitives import require_finite
from learned_tg.data.whitening_truncation import adaptive_truncation_rank, apply_adaptive_padding, compute_a_whitened_basis_with_eigvals, compute_l2_whitened_basis_with_eigvals
from learned_tg.data.channelized import generate_channelized_fields

SUPPORTED_DATASET_TYPES = {"interface", "channelized"}
SUPPORTED_TOPOLOGY_MODES = {"mixed", "island", "connected"}
SUPPORTED_SNAPSHOT_SMOOTHERS = {"none", "asm", "jacobi", "gmg"}
SUPPORTED_TRANSMISSIBILITIES = {"harmonic", "interface", "interface_aware"}


def _canonical_transmissibility(value):
    mode = str(value or "harmonic").strip().lower()
    if mode not in SUPPORTED_TRANSMISSIBILITIES:
        raise ValueError(
            "transmissibility must be one of "
            f"{sorted(SUPPORTED_TRANSMISSIBILITIES)}, got {value!r}"
        )
    return "interface_aware" if mode == "interface" else mode


def _canonical_whiten_metric(metric):
    key = str(metric).lower()
    if key in {"l2", "euclidean", "2"}:
        return "l2"
    if key in {"a", "energy", "a_inner", "a-inner"}:
        return "a"
    raise ValueError(
        f"capture whitening metric must be one of {{'l2', 'a'}}, got {metric!r}"
    )


def _validate_positive_count(name, value):
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _canonical_dataset_type(dataset_type):
    dataset_type = str(dataset_type or "interface").strip().lower()
    if dataset_type not in SUPPORTED_DATASET_TYPES:
        raise ValueError(
            f"dataset_type must be one of {sorted(SUPPORTED_DATASET_TYPES)}, "
            f"got {dataset_type!r}"
        )
    return dataset_type


def _get_config_value(config, primary_key, fallback_key, default):
    if config is None:
        return default
    if primary_key in config:
        return config[primary_key]
    if fallback_key in config:
        return config[fallback_key]
    return default


def _resolve_channelized_config(config=None):
    config = {} if config is None else config
    style = str(
        _get_config_value(config, "channel_style", "style", "legacy_full")
    ).strip().lower()
    paper_like = style in {"paper", "paper_internal", "paper_internal_thin"}
    return {
        "max_horizontal": int(
            _get_config_value(config, "channel_max_horizontal", "max_horizontal", 5)
        ),
        "max_vertical": int(
            _get_config_value(config, "channel_max_vertical", "max_vertical", 5)
        ),
        "width_range": tuple(
            _get_config_value(
                config,
                "channel_width_range",
                "width_range",
                (0.012, 0.035) if paper_like else (0.04, 0.10),
            )
        ),
        "min_width_cells": int(
            _get_config_value(
                config,
                "channel_min_width_cells",
                "min_width_cells",
                2 if paper_like else 3,
            )
        ),
        "max_width_cells": _get_config_value(
            config,
            "channel_max_width_cells",
            "max_width_cells",
            3 if paper_like else None,
        ),
        "allow_disconnected": bool(
            _get_config_value(
                config,
                "channel_allow_disconnected",
                "allow_disconnected",
                True,
            )
        ),
        "touch_boundary": bool(
            _get_config_value(
                config,
                "channel_touch_boundary",
                "touch_boundary",
                not paper_like,
            )
        ),
        "boundary_margin": float(
            _get_config_value(
                config,
                "channel_boundary_margin",
                "boundary_margin",
                0.05 if paper_like else 0.0,
            )
        ),
        "style": style,
    }


def _dataset_parameter_cache_tag(config, dataset_type, contrast, topology_mode):
    dataset_type = _canonical_dataset_type(dataset_type)
    if dataset_type == "interface":
        return f"c{int(contrast)}_topo{topology_mode}"

    channel_cfg = _resolve_channelized_config(config)
    width_lo, width_hi = channel_cfg["width_range"]
    disconnected_tag = "disc" if channel_cfg["allow_disconnected"] else "conn"
    tag = (
        f"channelized_c{int(contrast)}"
        f"_h{channel_cfg['max_horizontal']}"
        f"_v{channel_cfg['max_vertical']}"
        f"_w{_float_tag(width_lo)}-{_float_tag(width_hi)}"
        f"_min{channel_cfg['min_width_cells']}"
        f"_{disconnected_tag}"
    )
    if channel_cfg["style"] not in {"legacy", "legacy_full", "full", "full_span"}:
        tag += (
            f"_{channel_cfg['style']}"
            f"_touch{int(channel_cfg['touch_boundary'])}"
            f"_mar{_float_tag(channel_cfg['boundary_margin'])}"
        )
        if channel_cfg["max_width_cells"] is not None:
            tag += f"_max{int(channel_cfg['max_width_cells'])}"
    return tag


def _resolve_loader_batch_size(num_samples, batch_size):
    num_samples = _validate_positive_count("num_samples", num_samples)
    batch_size = _validate_positive_count("batch_size", batch_size)
    return min(num_samples, batch_size)


def _as_torch_device(device):
    """Normalize a user-provided device string/object to torch.device."""
    return device if isinstance(device, torch.device) else torch.device(device)


def _is_cuda_device(device):
    return _as_torch_device(device).type == "cuda"


def _empty_cuda_cache_if_needed(device):
    """Release PyTorch's cached CUDA blocks after large temporary chunks."""
    if _is_cuda_device(device) and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _detach_to_cpu(tensor):
    """Detach a tensor and move it to CPU for long-lived dataset storage."""
    if tensor is None:
        return None
    return tensor.detach().cpu()


def _move_to_compute_device(tensor, device, dtype=None):
    """Move only the current small chunk to the compute device."""
    if tensor is None:
        return None
    target = _as_torch_device(device)
    return tensor.to(device=target, dtype=dtype) if dtype is not None else tensor.to(device=target)


def _snapshot_smoother_from_config(config):
    if config is None:
        config = {}
    for aligned_key in ("interface_smoother", "smoother_mode", "pre_smooth_mode"):
        if config.get(aligned_key) is not None:
            smoother = str(config.get(aligned_key))
            break
    else:
        smoother = None

    if smoother is None:
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
            "snapshot_smoother must be one of {'none', 'asm', 'jacobi', 'gmg'}, "
            f"got {smoother!r}"
        )
    return smoother


def _float_tag(value):
    return f"{float(value):.3g}".replace("+", "").replace("-", "m").replace(".", "p")


def _snapshot_smoothing_cache_tag(config):
    smoother = _snapshot_smoother_from_config(config)
    if smoother == "none":
        return "_smnone"
    if smoother == "asm":
        return "_smasm"
    if smoother == "jacobi":
        return "_smjacobi"
    return (
        "_smgmg"
        f"_pre{int(config.get('snapshot_gmg_pre_smooth_steps', 2))}"
        f"_post{int(config.get('snapshot_gmg_post_smooth_steps', 2))}"
        f"_om{_float_tag(config.get('snapshot_gmg_omega', 0.8))}"
    )


def _safe_subdomain_count(n, requested):
    n = int(n)
    requested = int(requested)
    return max(1, min(requested, max(1, (n - 1) // 2)))


def _gmg_down(matrix):
    if matrix.dim() != 4:
        raise ValueError(f"gmg_down expects shape (B,C,n,n), got {tuple(matrix.shape)}")
    if min(matrix.shape[-2:]) <= 2:
        return matrix
    return F.avg_pool2d(matrix, kernel_size=2, stride=2, ceil_mode=False)


def _gmg_up(matrix, target_shape):
    if matrix.dim() != 4:
        raise ValueError(f"gmg_up expects shape (B,C,n,n), got {tuple(matrix.shape)}")
    return F.interpolate(
        matrix,
        size=(int(target_shape[0]), int(target_shape[1])),
        mode="bilinear",
        align_corners=False,
    )


class InterfaceGMGHierarchy:
    """
    Lightweight geometric multigrid error smoother for interface snapshots.

    It rediscretizes the same zero-Dirichlet interface operator on averaged
    coefficient fields.  This is used only for generating random error families,
    not for online learned two-grid solves.
    """

    def __init__(
        self,
        fine_op,
        *,
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
        self.subdomain_solver = str(getattr(fine_op, "subdomain_solver", "cholesky"))
        self.jacobi_pcg_iters = int(getattr(fine_op, "jacobi_pcg_iters", 10))
        self.transmissibility = _canonical_transmissibility(
            getattr(fine_op, "transmissibility", "harmonic")
        )
        self.smoother = str(smoother).lower()
        if self.smoother not in {"jacobi", "asm"}:
            raise ValueError(f"snapshot GMG smoother must be 'jacobi' or 'asm', got {smoother!r}")
        self.pre_smooth_steps = int(pre_smooth_steps)
        self.post_smooth_steps = int(post_smooth_steps)
        self.omega = float(omega)
        self.coarse_solve = str(coarse_solve).lower()
        if self.coarse_solve not in {"direct", "jacobi"}:
            raise ValueError(
                f"snapshot GMG coarse_solve must be 'direct' or 'jacobi', got {coarse_solve!r}"
            )

        self.min_coarse_size = max(2, int(min_coarse_size))
        if num_levels is not None and int(num_levels) <= 0:
            raise ValueError(f"snapshot_gmg_num_levels must be positive when provided, got {num_levels}")
        self.num_levels_requested = None if num_levels is None else int(num_levels)

        build_local_solver = self.smoother == "asm"
        if build_local_solver and not (
            getattr(fine_op, "build_local_solver", False) and self.subdomain_solver == "cholesky"
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
        return CCFVM_ASM(
            a_field.to(device=self.device, dtype=self.dtype),
            k_1d=_safe_subdomain_count(n, self.k_1d),
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
            a_c = _gmg_down(current.a).clamp_min(1e-30)
            phi_c = None if current.phi is None else _gmg_down(current.phi)
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
        raise ValueError(f"Unknown snapshot GMG smoother={smoother!r}")

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
                "Interface GMG coarse solve batch mismatch: "
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
        r_c = _gmg_down(r)
        e_c = self.v_cycle(level + 1, torch.zeros_like(r_c), r_c)
        e_f = _gmg_up(e_c, target_shape=(op.n, op.n))
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


@torch.no_grad()
def smooth_interface_snapshot_family(
    op_solver,
    e0,
    *,
    M_smooth,
    omega=0.8,
    smoother="asm",
    config=None,
):
    """
    Smooth random interface error probes for snapshot-family generation.

    The default "asm" branch preserves the historical generator.  "jacobi"
    applies weighted Jacobi to A e = 0.  "gmg" applies repeated geometric
    multigrid error-propagation steps using one V-cycle per outer step.
    """
    smoother = str(smoother).lower()
    if smoother not in SUPPORTED_SNAPSHOT_SMOOTHERS:
        raise ValueError(
            "snapshot smoother must be one of {'none', 'asm', 'jacobi', 'gmg'}, "
            f"got {smoother!r}"
        )
    M_smooth = int(M_smooth)
    x = e0.to(device=op_solver.device, dtype=op_solver.dtype)
    if smoother == "none" or M_smooth <= 0:
        return x

    if smoother == "asm":
        f_zero = torch.zeros_like(x)
        return op_solver.asm_smooth(x, f_zero, M_smooth=M_smooth, omega=float(omega))

    if smoother == "jacobi":
        diag = op_solver.sc.abs().clamp_min(1e-30)
        for _ in range(M_smooth):
            r = -op_solver.forward(x)
            x = x + float(omega) * (r / diag)
        return x

    config = {} if config is None else dict(config)
    gmg = InterfaceGMGHierarchy(
        op_solver,
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
        r = -op_solver.forward(x)
        x = x + outer_omega * gmg.apply(r)
    return x



def _tensor_nbytes(tensor):
    if not torch.is_tensor(tensor):
        return 0
    return int(tensor.numel()) * int(tensor.element_size())


def _format_gib(num_bytes):
    return float(num_bytes) / (1024.0 ** 3)


def _maybe_preload_dataset_tensors_to_gpu(
    tensors,
    *,
    device,
    enabled=False,
    safety_margin_gb=8.0,
    logger=None,
):
    """Optionally move the final cached dataset tensors to GPU for fast training."""
    target = _as_torch_device(device)
    if not enabled or target.type != "cuda":
        return tensors, False
    if not torch.cuda.is_available():
        return tensors, False

    try:
        torch.cuda.set_device(target)
        free_bytes, total_bytes = torch.cuda.mem_get_info(target)
    except Exception:
        free_bytes, total_bytes = torch.cuda.mem_get_info()

    required_bytes = sum(_tensor_nbytes(t) for t in tensors)
    margin_bytes = int(float(safety_margin_gb) * (1024 ** 3))
    message = (
        f"[DataLoader] Requested GPU dataset preload: "
        f"need={_format_gib(required_bytes):.2f} GiB, "
        f"free={_format_gib(free_bytes):.2f} GiB, "
        f"margin={float(safety_margin_gb):.2f} GiB."
    )
    if logger is not None:
        logger.info(message)
    else:
        print(message)

    if required_bytes + margin_bytes >= free_bytes:
        warning = (
            "[DataLoader] Skip GPU dataset preload because the final dataset "
            "does not fit with the configured safety margin. Training will use "
            "CPU tensors plus pinned-memory batch transfer."
        )
        if logger is not None:
            logger.warning(warning)
        else:
            print(warning)
        return tensors, False

    gpu_tensors = tuple(
        t.to(device=target, non_blocking=True) if torch.is_tensor(t) else t
        for t in tensors
    )
    torch.cuda.synchronize(target)
    print("[DataLoader] Dataset tensors preloaded to GPU for fast training.")
    return gpu_tensors, True


@torch.no_grad()
def _compute_truncated_left_svd_basis(e_family, rank):
    if rank is None:
        return None

    rank = int(rank)
    if rank <= 0:
        return None
    if e_family.dim() != 3:
        raise ValueError(
            f"Expected e_family with shape (B, N, q), got {tuple(e_family.shape)}"
        )

    max_rank = min(rank, int(e_family.shape[1]), int(e_family.shape[2]))
    if max_rank <= 0:
        raise ValueError(
            f"Unable to build SVD basis with rank={rank} for e_family shape {tuple(e_family.shape)}"
        )

    E = torch.nan_to_num(
        e_family,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).to(torch.float64)
    try:
        U, _, _ = torch.linalg.svd(E, full_matrices=False)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to compute truncated SVD basis for cached e_family chunk. "
            f"e_family.shape={tuple(e_family.shape)}, requested_rank={rank}"
        ) from exc

    return require_finite(U[:, :, :max_rank].contiguous(), "svd_basis_chunk")


@torch.no_grad()
def _compute_whitened_error_family(
    e_family,
    rank,
    *,
    metric="l2",
    op_solver=None,
):
    if rank is None:
        return None

    metric = _canonical_whiten_metric(metric)
    E = torch.nan_to_num(
        e_family,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).to(torch.float64)
    try:
        return _compute_metric_whitened_snapshot_family(
            E,
            rank=rank,
            metric=metric,
            op_solver=op_solver,
            reg=1e-12,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to compute metric-aware whitened cached e_family chunk. "
            f"e_family.shape={tuple(e_family.shape)}, requested_rank={rank}, metric={metric!r}"
        ) from exc


@torch.no_grad()
def _compute_dataset_svd_basis(
    e_family,
    rank,
    batch_chunk_size=20,
    compute_device="cpu",
):
    """Compute per-sample truncated SVD bases chunk-by-chunk.

    The returned tensor is intentionally kept on CPU. This prevents the full
    cached dataset from occupying GPU memory while still allowing each small
    chunk to be computed on CUDA when requested.
    """
    if rank is None:
        return None

    batch_chunk_size = max(int(batch_chunk_size), 1)
    parts = []
    for start_idx in range(0, e_family.shape[0], batch_chunk_size):
        end_idx = min(start_idx + batch_chunk_size, e_family.shape[0])
        e_chunk = _move_to_compute_device(
            e_family[start_idx:end_idx],
            compute_device,
            dtype=torch.float64,
        )
        basis_chunk = _compute_truncated_left_svd_basis(
            e_chunk,
            rank=rank,
        )
        parts.append(_detach_to_cpu(basis_chunk))
        del e_chunk, basis_chunk
        _empty_cuda_cache_if_needed(compute_device)

    return torch.cat(parts, dim=0)


@torch.no_grad()
def _compute_dataset_whitened_error_family(
    e_family,
    rank,
    *,
    metric="l2",
    a_field=None,
    phi=None,
    transmissibility="harmonic",
    batch_chunk_size=20,
    k_1d=10,
    compute_device="cpu",
):
    """Compute cached whitened error families without storing them on GPU."""
    if rank is None:
        return None

    metric = _canonical_whiten_metric(metric)
    batch_chunk_size = max(int(batch_chunk_size), 1)
    parts = []
    for start_idx in range(0, e_family.shape[0], batch_chunk_size):
        end_idx = min(start_idx + batch_chunk_size, e_family.shape[0])
        op_solver = None
        e_chunk = _move_to_compute_device(
            e_family[start_idx:end_idx],
            compute_device,
            dtype=torch.float64,
        )
        if metric == "a":
            if a_field is None:
                raise ValueError("A-metric cached whitening requires a_field.")
            a_chunk = _move_to_compute_device(
                a_field[start_idx:end_idx],
                compute_device,
                dtype=torch.float64,
            )
            phi_chunk = None
            if _canonical_transmissibility(transmissibility) != "harmonic":
                if phi is None:
                    raise ValueError(
                        "interface-aware A-metric whitening requires phi"
                    )
                phi_chunk = _move_to_compute_device(
                    phi[start_idx:end_idx],
                    compute_device,
                    dtype=torch.float64,
                )
            op_solver = CCFVM_ASM(
                a=a_chunk,
                k_1d=k_1d,
                build_local_solver=False,
                phi=phi_chunk,
                transmissibility=transmissibility,
            ).to(a_chunk.device)

        white_chunk = _compute_whitened_error_family(
            e_chunk,
            rank=rank,
            metric=metric,
            op_solver=op_solver,
        )
        parts.append(_detach_to_cpu(white_chunk))

        if op_solver is not None:
            del op_solver
        if metric == "a":
            del a_chunk, phi_chunk
        del e_chunk, white_chunk
        _empty_cuda_cache_if_needed(compute_device)

    return torch.cat(parts, dim=0)


@torch.no_grad()
def _compute_dataset_whitened_error_family_adaptive(
    e_family,
    *,
    metric="a",
    k_max,
    min_rank,
    max_rank,
    floor_relative,
    beta=1.0,
    a_field=None,
    phi=None,
    transmissibility="harmonic",
    batch_chunk_size=20,
    k_1d=10,
    compute_device="cpu",
):
    """
    Adaptive-rank whitening: per-sample truncation via log-gap detection, zero-padded
    to ``(B, N, k_max)`` so the downstream DataLoader sees a uniform shape.

    Returns
    -------
    e_white : (B, N, k_max) tensor on CPU
    k_per_sample : (B,) int64 tensor on CPU
    eigvals_per_sample : (B, q) float64 tensor on CPU
    """
    metric = _canonical_whiten_metric(metric)
    beta = float(beta)
    batch_chunk_size = max(int(batch_chunk_size), 1)
    parts, k_parts, eig_parts = [], [], []

    for start_idx in range(0, e_family.shape[0], batch_chunk_size):
        end_idx = min(start_idx + batch_chunk_size, e_family.shape[0])
        e_chunk = _move_to_compute_device(
            e_family[start_idx:end_idx], compute_device, dtype=torch.float64
        )

        if metric == "a":
            if a_field is None:
                raise ValueError("A-metric adaptive whitening requires a_field.")
            a_chunk = _move_to_compute_device(
                a_field[start_idx:end_idx], compute_device, dtype=torch.float64
            )
            phi_chunk = None
            if _canonical_transmissibility(transmissibility) != "harmonic":
                if phi is None:
                    raise ValueError(
                        "interface-aware adaptive A-metric whitening requires phi"
                    )
                phi_chunk = _move_to_compute_device(
                    phi[start_idx:end_idx], compute_device, dtype=torch.float64
                )
            op = CCFVM_ASM(
                a=a_chunk,
                k_1d=k_1d,
                build_local_solver=False,
                phi=phi_chunk,
                transmissibility=transmissibility,
            ).to(a_chunk.device)
            AE = op.apply_A(e_chunk)
            U_k, eigvals = compute_a_whitened_basis_with_eigvals(e_chunk, AE, rank=k_max, beta=beta)
            del op, a_chunk, phi_chunk, AE
        else:
            U_k, eigvals = compute_l2_whitened_basis_with_eigvals(e_chunk, rank=k_max, beta=beta)

        U_padded, k_chunk = apply_adaptive_padding(
            U_k, eigvals,
            floor_relative=floor_relative,
            min_rank=min_rank,
            max_rank=max_rank,
        )
        parts.append(_detach_to_cpu(U_padded))
        k_parts.append(k_chunk.cpu())
        eig_parts.append(_detach_to_cpu(eigvals))

        del e_chunk, U_k, U_padded, k_chunk, eigvals
        _empty_cuda_cache_if_needed(compute_device)

    return (
        torch.cat(parts, dim=0),
        torch.cat(k_parts, dim=0),
        torch.cat(eig_parts, dim=0),
    )


def _resolve_cached_svd_rank(config, n, num_probes):
    requested_rank = int(config.get("m_dim", num_probes))
    return max(1, min(requested_rank, int(n) * int(n), int(num_probes)))


def _resolve_cached_capture_whiten_rank(config, n, num_probes):
    capture_family_config = config.get("capture_family")
    capture_family = str(capture_family_config or "raw").lower()
    use_cached_whitened_family = bool(config.get("use_cached_whitened_family", True))
    if capture_family == "white" and use_cached_whitened_family:
        requested_rank = config.get("whiten_rank", config.get("m_dim", num_probes))
        if requested_rank is None:
            requested_rank = min(int(n) * int(n), int(num_probes))
        requested_rank = int(requested_rank)
        return max(1, min(requested_rank, int(n) * int(n), int(num_probes)))
    if capture_family not in {"raw", "white"}:
        raise ValueError(
            f"capture_family must be one of {{'raw', 'white'}}, got {capture_family!r}"
        )
    return None


def _resolve_cached_capture_whiten_metric(config):
    capture_family_config = config.get("capture_family")
    capture_family = str(capture_family_config or "raw").lower()
    use_cached_whitened_family = bool(config.get("use_cached_whitened_family", True))
    if capture_family == "white" and use_cached_whitened_family:
        requested_metric = config.get("whiten_metric", config.get("capture_metric"))
        if requested_metric is None:
            requested_metric = config.get("loss_type", "l2")
        return _canonical_whiten_metric(requested_metric)
    if capture_family not in {"raw", "white"}:
        raise ValueError(
            f"capture_family must be one of {{'raw', 'white'}}, got {capture_family!r}"
        )
    return None


def _whitened_family_field_names(metric):
    metric = _canonical_whiten_metric(metric)
    field_name = f"e_family_white_{metric}"
    rank_name = f"{field_name}_rank"
    return field_name, rank_name


# ---------------------------------------------------------------------------
# Adaptive whitening config helpers
# ---------------------------------------------------------------------------

def _resolve_whiten_rank_mode(config):
    """Return "fixed" or "adaptive" from config, with backward-compat default "fixed"."""
    return str(config.get("whiten_rank_mode", "fixed")).lower()


def _resolve_adaptive_whiten_params(config, num_probes):
    """Parse adaptive whitening hyper-parameters from config."""
    min_rank = max(1, int(config.get("whiten_rank_min", 4)))
    max_rank = max(min_rank, int(config.get("whiten_rank_max", 30)))
    max_rank = min(max_rank, int(num_probes))
    floor_relative = float(config.get("whiten_rank_floor_relative", 1e-12))
    return min_rank, max_rank, floor_relative


def _resolve_whiten_beta(config) -> float:
    """Return whiten_beta from config, defaulting to 1.0 (full whitening)."""
    beta = float(config.get("whiten_beta", 1.0))
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"whiten_beta must be in [0, 1], got {beta}")
    return beta


def _whiten_mode_cache_tag(config):
    """
    Return a filename suffix for the whitening mode.

    Fixed mode: empty string (preserves legacy filenames).
    Adaptive mode: "_whitenadaptive_beta{int(beta*100):03d}".
    """
    mode = _resolve_whiten_rank_mode(config)
    if mode == "adaptive":
        beta = _resolve_whiten_beta(config)
        return f"_whitenadaptive_beta{round(beta * 100):03d}"
    return ""


def _ensure_cached_svd_basis(
    dataset_dict,
    *,
    requested_rank,
    chunk_size,
    file_path=None,
    split="train",
    compute_device="cpu",
):
    requested_rank = int(requested_rank)
    existing_basis = dataset_dict.get("svd_basis")
    if existing_basis is not None and int(existing_basis.shape[-1]) >= requested_rank:
        dataset_dict["svd_rank"] = int(existing_basis.shape[-1])
        return dataset_dict, False

    e_family = dataset_dict["e_family"]
    svd_basis_cpu = _compute_dataset_svd_basis(
        e_family,
        rank=requested_rank,
        batch_chunk_size=chunk_size,
        compute_device=compute_device,
    )
    dataset_dict["svd_basis"] = svd_basis_cpu
    dataset_dict["svd_rank"] = int(svd_basis_cpu.shape[-1])
    if file_path is not None:
        dataset_to_save = {
            key: value.cpu() if torch.is_tensor(value) else value
            for key, value in dataset_dict.items()
        }
        torch.save(dataset_to_save, file_path)
        print(
            f"[{split.capitalize()}] Cached svd_basis updated at {file_path} "
            f"(rank={dataset_dict['svd_rank']})"
        )
    return dataset_dict, True


def _ensure_cached_whitened_family(
    dataset_dict,
    *,
    requested_rank,
    requested_metric,
    chunk_size,
    k_1d,
    file_path=None,
    split="train",
    compute_device="cpu",
    whiten_rank_mode="fixed",
    whiten_rank_min=4,
    whiten_rank_max=30,
    whiten_rank_floor_relative=1e-12,
    whiten_beta=1.0,
    transmissibility="harmonic",
):
    if requested_rank is None or requested_metric is None:
        return dataset_dict, False

    requested_rank = int(requested_rank)
    requested_metric = _canonical_whiten_metric(requested_metric)
    whiten_beta = float(whiten_beta)
    field_name, rank_name = _whitened_family_field_names(requested_metric)

    # -----------------------------------------------------------------------
    # Adaptive mode: write to dataset and return early if already cached.
    # -----------------------------------------------------------------------
    if whiten_rank_mode == "adaptive":
        k_max = int(whiten_rank_max)
        k_field = f"whiten_rank_per_sample_{requested_metric}"
        eig_field = f"whiten_eigvals_per_sample_{requested_metric}"

        # Already cached if field exists, mode matches, and beta matches.
        if (
            dataset_dict.get(field_name) is not None
            and dataset_dict.get("whiten_rank_mode") == "adaptive"
            and dataset_dict.get(k_field) is not None
            and abs(float(dataset_dict.get("whiten_beta", 1.0)) - whiten_beta) < 1e-12
        ):
            return dataset_dict, False

        e_family = dataset_dict["e_family"]
        e_white, k_per_sample, eigvals = _compute_dataset_whitened_error_family_adaptive(
            e_family,
            metric=requested_metric,
            k_max=k_max,
            min_rank=whiten_rank_min,
            max_rank=whiten_rank_max,
            floor_relative=whiten_rank_floor_relative,
            beta=whiten_beta,
            a_field=dataset_dict.get("a_field"),
            phi=dataset_dict.get("phi"),
            transmissibility=transmissibility,
            batch_chunk_size=chunk_size,
            k_1d=k_1d,
            compute_device=compute_device,
        )
        dataset_dict[field_name] = e_white
        dataset_dict[rank_name] = k_max
        dataset_dict[k_field] = k_per_sample
        dataset_dict[eig_field] = eigvals
        dataset_dict["whiten_rank_mode"] = "adaptive"
        dataset_dict["whiten_beta"] = whiten_beta
        dataset_dict["whiten_mode_signature"] = f"adaptive_beta{whiten_beta:.2f}"

        if file_path is not None:
            dataset_to_save = {
                key: value.cpu() if torch.is_tensor(value) else value
                for key, value in dataset_dict.items()
            }
            torch.save(dataset_to_save, file_path)
            k_stats = f"min={int(k_per_sample.min())} median={int(k_per_sample.float().median())} max={int(k_per_sample.max())}"
            print(
                f"[{split.capitalize()}] Adaptive cached {field_name} at {file_path} "
                f"(k_max={k_max}, beta={whiten_beta:.2f}, per-sample k: {k_stats})"
            )
        return dataset_dict, True

    # -----------------------------------------------------------------------
    # Fixed mode: existing logic, unchanged for byte-identical compatibility.
    # -----------------------------------------------------------------------
    existing_white = dataset_dict.get(field_name)
    existing_rank = dataset_dict.get(rank_name)
    if existing_white is None and requested_metric == "l2":
        legacy_white = dataset_dict.get("e_family_white")
        legacy_rank = dataset_dict.get("e_family_white_rank")
        if legacy_white is not None and legacy_rank is not None:
            dataset_dict[field_name] = legacy_white
            dataset_dict[rank_name] = legacy_rank
            existing_white = legacy_white
            existing_rank = legacy_rank
            if file_path is not None:
                dataset_to_save = {
                    key: value.cpu() if torch.is_tensor(value) else value
                    for key, value in dataset_dict.items()
                }
                torch.save(dataset_to_save, file_path)

    if (
        existing_white is not None
        and existing_rank is not None
        and int(existing_rank) == requested_rank
    ):
        return dataset_dict, False

    e_family = dataset_dict["e_family"]
    e_family_white_cpu = _compute_dataset_whitened_error_family(
        e_family,
        rank=requested_rank,
        metric=requested_metric,
        a_field=dataset_dict["a_field"],
        phi=dataset_dict.get("phi"),
        transmissibility=transmissibility,
        batch_chunk_size=chunk_size,
        k_1d=k_1d,
        compute_device=compute_device,
    )
    dataset_dict[field_name] = e_family_white_cpu
    dataset_dict[rank_name] = int(requested_rank)
    if file_path is not None:
        dataset_to_save = {
            key: value.cpu() if torch.is_tensor(value) else value
            for key, value in dataset_dict.items()
        }
        torch.save(dataset_to_save, file_path)
        print(
            f"[{split.capitalize()}] Cached {field_name} updated at {file_path} "
            f"(rank={dataset_dict[rank_name]})"
        )
    return dataset_dict, True

# ==============================================================================
# 1. 核心物理生成（🚨 绝对保留：严格的 float32 逻辑，保证物理几何与老版 100% 对齐）
# ==============================================================================
def generate_random_interface_coeffs(batch_size, n, contrast=1e4, device='cpu'):
    h = 1.0 / n
    x_1d = torch.linspace(h/2, 1.0 - h/2, n, device=device)
    Y, X = torch.meshgrid(x_1d, x_1d, indexing='ij')

    cX = X - (0.5 + (torch.rand(batch_size, 1, 1, device=device) - 0.5) * 0.2)
    cY = Y - (0.5 + (torch.rand(batch_size, 1, 1, device=device) - 0.5) * 0.2)
    R_grid = torch.sqrt(cX**2 + cY**2)
    Theta_grid = torch.atan2(cY, cX)

    R0 = 0.25 + torch.rand(batch_size, 1, 1, device=device) * 0.1
    a_coeffs = torch.randn(batch_size, 4, 1, 1, device=device) * 0.05
    b_coeffs = torch.randn(batch_size, 4, 1, 1, device=device) * 0.05

    R_interface = R0.clone()
    for k in range(1, 5):
        R_interface = R_interface + a_coeffs[:, k-1] * torch.cos(k * Theta_grid) + b_coeffs[:, k-1] * torch.sin(k * Theta_grid)

    phi = R_grid - R_interface

    a_field = torch.ones_like(phi)
    flip = torch.randint(0, 2, (batch_size, 1, 1), device=device, dtype=torch.float64)

    mask_inside = (phi < 0)
    mask_outside = (phi >= 0)

    a_inside = 1.0 * flip + contrast * (1.0 - flip)
    a_outside = contrast * flip + 1.0 * (1.0 - flip)

    a_field = a_inside * mask_inside + a_outside * mask_outside

    return phi, a_field, X, Y


def _validate_topology_mode(topology_mode):
    if topology_mode not in SUPPORTED_TOPOLOGY_MODES:
        raise ValueError(
            f"Unsupported topology_mode={topology_mode!r}. "
            f"Expected one of {sorted(SUPPORTED_TOPOLOGY_MODES)}."
        )


def check_topology(a_field, bg_val=None):
    """
    Classify a single coefficient field by whether its high-conductivity region
    touches the physical boundary.
    """
    if not torch.is_tensor(a_field):
        a_field = torch.as_tensor(a_field)

    a_field_2d = a_field.squeeze()
    if a_field_2d.ndim != 2:
        raise ValueError(
            f"check_topology expects a single 2D field (or squeezable equivalent), got shape {tuple(a_field.shape)}."
        )

    if bg_val is None:
        bg_val = a_field_2d.amin()

    high_cond_mask = a_field_2d > bg_val
    touches_boundary = (
        high_cond_mask[0, :].any()
        | high_cond_mask[-1, :].any()
        | high_cond_mask[:, 0].any()
        | high_cond_mask[:, -1].any()
    )
    return "connected" if bool(touches_boundary.item()) else "island"


def _match_topology_mode(a_field_batch, topology_mode, island_margin=0.1):
    """
    分类并匹配当前的拓扑模式。
    增加了 island_margin 参数，用于强制 island 模式下高导电区域与边界保持最小安全距离。
    """
    _validate_topology_mode(topology_mode)

    batch_size = a_field_batch.shape[0]
    if topology_mode == "mixed":
        return torch.ones(batch_size, dtype=torch.bool, device=a_field_batch.device)

    # 找到每个样本中的基态(低导电)值
    bg_vals = a_field_batch.reshape(batch_size, -1).amin(dim=1).view(-1, 1, 1)
    high_cond_mask = a_field_batch > bg_vals

    if topology_mode == "island":
        # 计算 0.1 物理距离对应占据的网格点数（至少为 1）
        n = a_field_batch.shape[-1]
        margin_idx = max(1, int(n * island_margin))

        # 检查高导电区域是否侵入了四周的 margin 缓冲区
        touches_margin = (
            high_cond_mask[:, :margin_idx, :].reshape(batch_size, -1).any(dim=1)
            | high_cond_mask[:, -margin_idx:, :].reshape(batch_size, -1).any(dim=1)
            | high_cond_mask[:, :, :margin_idx].reshape(batch_size, -1).any(dim=1)
            | high_cond_mask[:, :, -margin_idx:].reshape(batch_size, -1).any(dim=1)
        )
        return ~touches_margin
    else:
        # connected 模式保持原样：必须触碰最外层物理边界（index=0或-1）
        touches_boundary = (
            high_cond_mask[:, 0, :].any(dim=1)
            | high_cond_mask[:, -1, :].any(dim=1)
            | high_cond_mask[:, :, 0].any(dim=1)
            | high_cond_mask[:, :, -1].any(dim=1)
        )
        return touches_boundary


def generate_random_interface_coeffs_with_topology(
    batch_size,
    n,
    contrast=1e4,
    device='cpu',
    topology_mode='mixed',
    max_retries=100,
    island_margin=0.1,  # <--- 新增参数，默认强制 0.1 的边界距离
):
    """
    Wrap the original random generator with rejection sampling so we can enforce
    the topology of the high-conductivity region without touching the shape logic.
    """
    _validate_topology_mode(topology_mode)
    if max_retries <= 0:
        raise ValueError(f"max_retries must be positive, got {max_retries}.")

    if topology_mode == "mixed":
        return generate_random_interface_coeffs(
            batch_size=batch_size,
            n=n,
            contrast=contrast,
            device=device,
        )

    accepted_phi = []
    accepted_a_field = []
    accepted_count = 0
    retries = 0
    X = None
    Y = None

    while accepted_count < batch_size:
        retries += 1
        remaining = batch_size - accepted_count
        phi_candidate, a_field_candidate, X, Y = generate_random_interface_coeffs(
            batch_size=remaining,
            n=n,
            contrast=contrast,
            device=device,
        )

        # 传入 island_margin 进行带缓冲区的拒绝采样判定
        accept_mask = _match_topology_mode(
            a_field_candidate,
            topology_mode,
            island_margin=island_margin
        )

        if accept_mask.any():
            accepted_phi.append(phi_candidate[accept_mask])
            accepted_a_field.append(a_field_candidate[accept_mask])
            accepted_count += int(accept_mask.sum().item())

        if retries >= max_retries and accepted_count < batch_size:
            raise RuntimeError(
                f"Failed to generate {batch_size} '{topology_mode}' samples within {max_retries} retries. "
                f"Only accepted {accepted_count} samples. Consider relaxing the topology constraint, "
                f"decreasing island_margin, or increasing max_retries."
            )

    phi = torch.cat(accepted_phi, dim=0)[:batch_size]
    a_field = torch.cat(accepted_a_field, dim=0)[:batch_size]
    return phi, a_field, X, Y

# ==============================================================================
# 2. 多误差流形生成（Error Family Generation）
# ==============================================================================
def generate_data_family(
    batch_size=100,
    n=63,
    M_smooth=2,
    num_probes=10,
    contrast=1e4,
    device='cpu',
    chunk_size=20,
    k_1d=10,
    omega=0.8,
    topology_mode='mixed',
    max_retries=100,
    svd_rank=None,
    whiten_rank=None,
    whiten_metric=None,
    vectorized_probes=True,
    snapshot_smoother="asm",
    snapshot_config=None,
    dataset_type="interface",
    channel_max_horizontal=5,
    channel_max_vertical=5,
    channel_width_range=(0.04, 0.10),
    channel_min_width_cells=3,
    channel_max_width_cells=None,
    channel_allow_disconnected=True,
    channel_touch_boundary=True,
    channel_boundary_margin=0.0,
    channel_style="legacy_full",
    transmissibility="harmonic",
):
    batch_size = _validate_positive_count("batch_size", batch_size)
    chunk_size = _validate_positive_count("chunk_size", chunk_size)
    num_probes = _validate_positive_count("num_probes", num_probes)
    snapshot_config = {} if snapshot_config is None else dict(snapshot_config)
    transmissibility = _canonical_transmissibility(
        snapshot_config.get("transmissibility", transmissibility)
    )
    snapshot_config["transmissibility"] = transmissibility
    snapshot_config.setdefault("channel_max_horizontal", channel_max_horizontal)
    snapshot_config.setdefault("channel_max_vertical", channel_max_vertical)
    snapshot_config.setdefault("channel_width_range", channel_width_range)
    snapshot_config.setdefault("channel_min_width_cells", channel_min_width_cells)
    snapshot_config.setdefault("channel_max_width_cells", channel_max_width_cells)
    snapshot_config.setdefault("channel_allow_disconnected", channel_allow_disconnected)
    snapshot_config.setdefault("channel_touch_boundary", channel_touch_boundary)
    snapshot_config.setdefault("channel_boundary_margin", channel_boundary_margin)
    snapshot_config.setdefault("channel_style", channel_style)
    dataset_type = _canonical_dataset_type(snapshot_config.get("dataset_type", dataset_type))
    if dataset_type == "interface":
        _validate_topology_mode(topology_mode)
    channelized_config = _resolve_channelized_config(snapshot_config)
    if whiten_rank is not None and whiten_metric is None:
        whiten_metric = "l2"
    if whiten_metric is not None:
        whiten_metric = _canonical_whiten_metric(whiten_metric)
    vectorized_probes = bool(vectorized_probes)
    subdomain_solver_config = str(snapshot_config.get('subdomain_solver', 'cholesky'))
    jacobi_pcg_iters_config = int(snapshot_config.get('jacobi_pcg_iters', 10))
    vcycle_coarsest_threshold_config = int(snapshot_config.get('vcycle_coarsest_threshold', 24))
    vcycle_smooth_steps_config = int(snapshot_config.get('vcycle_smooth_steps', 2))
    vcycle_iter_config = int(snapshot_config.get('vcycle_iter', 2))
    vcycle_omega_config = float(snapshot_config.get('vcycle_omega', 2.0 / 3.0))
    vcycle_compile_config = bool(snapshot_config.get('vcycle_compile', True))
    if snapshot_smoother is None:
        snapshot_smoother = _snapshot_smoother_from_config(snapshot_config)
    else:
        snapshot_config["snapshot_smoother"] = snapshot_smoother
        snapshot_smoother = _snapshot_smoother_from_config(snapshot_config)
    e_family_list = []
    svd_basis_list = [] if svd_rank is not None and int(svd_rank) > 0 else None
    e_family_white_list = [] if whiten_rank is not None and int(whiten_rank) > 0 else None
    phi_list = []
    a_field_list = []

    print(
        f"    [Generator] Starting generation of {batch_size} samples with "
        f"{num_probes} probes each (dataset_type={dataset_type}, "
        f"topology_mode={topology_mode}, "
        f"smoother={snapshot_smoother}, transmissibility={transmissibility})..."
    )
    if dataset_type == "channelized":
        print(
            "    [Generator] channelized config: "
            f"max_horizontal={channelized_config['max_horizontal']}, "
            f"max_vertical={channelized_config['max_vertical']}, "
            f"width_range={channelized_config['width_range']}, "
            f"min_width_cells={channelized_config['min_width_cells']}, "
            f"max_width_cells={channelized_config['max_width_cells']}, "
            f"allow_disconnected={channelized_config['allow_disconnected']}"
            f", style={channelized_config['style']}, "
            f"touch_boundary={channelized_config['touch_boundary']}, "
            f"boundary_margin={channelized_config['boundary_margin']}"
        )
    print(f"    [DataGen] vectorized_probes={vectorized_probes}, num_probes={num_probes}")

    for start_idx in range(0, batch_size, chunk_size):
        end_idx = min(start_idx + chunk_size, batch_size)
        chunk_batch_size = end_idx - start_idx

        # 1. 生成物理场
        if dataset_type == "interface":
            phi_chunk, a_field_chunk, X_chunk, Y_chunk = generate_random_interface_coeffs_with_topology(
                batch_size=chunk_batch_size,
                n=n,
                contrast=contrast,
                device=device,
                topology_mode=topology_mode,
                max_retries=max_retries,
            )
        else:
            phi_chunk, a_field_chunk, _ = generate_channelized_fields(
                batch_size=chunk_batch_size,
                n=n,
                contrast=contrast,
                max_horizontal=channelized_config["max_horizontal"],
                max_vertical=channelized_config["max_vertical"],
                width_range=channelized_config["width_range"],
                min_width_cells=channelized_config["min_width_cells"],
                allow_disconnected=channelized_config["allow_disconnected"],
                device=device,
                channel_touch_boundary=channelized_config["touch_boundary"],
                channel_boundary_margin=channelized_config["boundary_margin"],
                channel_style=channelized_config["style"],
                channel_max_width_cells=channelized_config["max_width_cells"],
            )
            h = 1.0 / n
            x_1d_chunk = torch.linspace(h / 2, 1.0 - h / 2, n, device=device)
            Y_chunk, X_chunk = torch.meshgrid(x_1d_chunk, x_1d_chunk, indexing='ij')

        # 2. 初始化高精度求解器
        a_field_double = a_field_chunk.double()
        equation = CCFVM_ASM(
            a=a_field_double,
            k_1d=k_1d,
            build_local_solver=(snapshot_smoother == "asm"),
            subdomain_solver=subdomain_solver_config,
            jacobi_pcg_iters=jacobi_pcg_iters_config,
            vcycle_coarsest_threshold=vcycle_coarsest_threshold_config,
            vcycle_smooth_steps=vcycle_smooth_steps_config,
            vcycle_iter=vcycle_iter_config,
            vcycle_omega=vcycle_omega_config,
            vcycle_compile=vcycle_compile_config,
            phi=(phi_chunk.double() if transmissibility != "harmonic" else None),
            transmissibility=transmissibility,
        ).to(device)

        # 3. 核心：为同一个物理场生成多个不同的误差探针 (Error Family)
        probes_list = None
        if vectorized_probes:
            e_init = torch.randn(
                chunk_batch_size,
                num_probes,
                n,
                n,
                device=device,
                dtype=torch.float64,
            )

            with torch.no_grad():
                e_smooth_family_chunk = smooth_interface_snapshot_family(
                    equation,
                    e_init,
                    M_smooth=M_smooth,
                    omega=omega,
                    smoother=snapshot_smoother,
                    config=snapshot_config,
                )
        else:
            probes_list = []
            for _ in range(num_probes):
                e_init = torch.randn(chunk_batch_size, 1, n, n, device=device, dtype=torch.float64)

                with torch.no_grad():
                    e_smooth = smooth_interface_snapshot_family(
                        equation,
                        e_init,
                        M_smooth=M_smooth,
                        omega=omega,
                        smoother=snapshot_smoother,
                        config=snapshot_config,
                    )
                probes_list.append(e_smooth)

            # 将 probes 拼接在一起: shape -> (Batch, num_probes, n, n)
            e_smooth_family_chunk = torch.cat(probes_list, dim=1)

        e_family_chunk = (
            e_smooth_family_chunk.view(chunk_batch_size, num_probes, n * n)
            .transpose(1, 2)
            .contiguous()
        )

        # Long-lived dataset tensors are stored on CPU. Otherwise, for large
        # datasets, the list accumulation plus the final torch.cat can double
        # GPU memory usage and fail near the end of generation.
        phi_list.append(_detach_to_cpu(phi_chunk))
        a_field_list.append(_detach_to_cpu(a_field_chunk))
        e_family_list.append(_detach_to_cpu(e_family_chunk))

        if svd_basis_list is not None:
            svd_basis_chunk = _compute_truncated_left_svd_basis(
                e_family_chunk,
                rank=svd_rank,
            )
            svd_basis_list.append(_detach_to_cpu(svd_basis_chunk))
            del svd_basis_chunk

        if e_family_white_list is not None:
            e_family_white_chunk = _compute_whitened_error_family(
                e_family_chunk,
                rank=whiten_rank,
                metric=whiten_metric,
                op_solver=equation if whiten_metric == "a" else None,
            )
            e_family_white_list.append(_detach_to_cpu(e_family_white_chunk))
            del e_family_white_chunk

        del (
            phi_chunk,
            a_field_chunk,
            X_chunk,
            Y_chunk,
            a_field_double,
            equation,
            probes_list,
            e_init,
            e_smooth_family_chunk,
            e_family_chunk,
        )
        if not vectorized_probes:
            del e_smooth
        _empty_cuda_cache_if_needed(device)

        if (start_idx // chunk_size) % 5 == 0:
            print(f"    [Generator] Progress: {end_idx}/{batch_size} samples done.")

    # 汇总
    phi = torch.cat(phi_list, dim=0)
    a_field = torch.cat(a_field_list, dim=0)
    e_family = torch.cat(e_family_list, dim=0)

    h = 1.0 / n
    x_1d = torch.linspace(h/2, 1.0 - h/2, n, device="cpu")
    Y, X = torch.meshgrid(x_1d, x_1d, indexing='ij')
    coords = torch.stack([X.flatten(), Y.flatten()], dim=1)

    if svd_basis_list is None and e_family_white_list is None:
        return phi, a_field, coords, e_family, X, Y

    if svd_basis_list is not None:
        svd_basis = torch.cat(svd_basis_list, dim=0)
    if e_family_white_list is not None:
        e_family_white = torch.cat(e_family_white_list, dim=0)

    if svd_basis_list is not None and e_family_white_list is None:
        return phi, a_field, coords, e_family, X, Y, svd_basis
    if svd_basis_list is None and e_family_white_list is not None:
        return phi, a_field, coords, e_family, X, Y, e_family_white
    return phi, a_field, coords, e_family, X, Y, svd_basis, e_family_white


def generate_data(
    batch_size=100,
    n=63,
    M_smooth=2,
    contrast=1e4,
    device='cpu',
    chunk_size=20,
    num_probes=10,
    k_1d=10,
    omega=0.8,
    return_subspace=False,
    topology_mode='mixed',
    max_retries=100,
    svd_rank=None,
    whiten_rank=None,
    whiten_metric=None,
    vectorized_probes=True,
    snapshot_smoother="asm",
    snapshot_config=None,
    dataset_type="interface",
    channel_max_horizontal=5,
    channel_max_vertical=5,
    channel_width_range=(0.04, 0.10),
    channel_min_width_cells=3,
    channel_max_width_cells=None,
    channel_allow_disconnected=True,
    channel_touch_boundary=True,
    channel_boundary_margin=0.0,
    channel_style="legacy_full",
    transmissibility="harmonic",
):
    """
    Backward-compatible entry point used by the SVD study script.

    Returns:
        phi: (Batch, 1, n, n)
        a_field: (Batch, 1, n, n)
        coords: (n*n, 2)
        e_family: (Batch, n*n, num_probes)
        X, Y: shared meshgrid tensors with shape (n, n)
    """
    if return_subspace:
        raise NotImplementedError("generate_data(return_subspace=True) is not supported for interface data.")

    return generate_data_family(
        batch_size=batch_size,
        n=n,
        M_smooth=M_smooth,
        num_probes=num_probes,
        contrast=contrast,
        device=device,
        chunk_size=chunk_size,
        k_1d=k_1d,
        omega=omega,
        topology_mode=topology_mode,
        max_retries=max_retries,
        svd_rank=svd_rank,
        whiten_rank=whiten_rank,
        whiten_metric=whiten_metric,
        vectorized_probes=vectorized_probes,
        snapshot_smoother=snapshot_smoother,
        snapshot_config=snapshot_config,
        dataset_type=dataset_type,
        channel_max_horizontal=channel_max_horizontal,
        channel_max_vertical=channel_max_vertical,
        channel_width_range=channel_width_range,
        channel_min_width_cells=channel_min_width_cells,
        channel_max_width_cells=channel_max_width_cells,
        channel_allow_disconnected=channel_allow_disconnected,
        channel_touch_boundary=channel_touch_boundary,
        channel_boundary_margin=channel_boundary_margin,
        channel_style=channel_style,
        transmissibility=transmissibility,
    )


def generate_parameter_data(
    batch_size=100,
    n=63,
    contrast=1e4,
    device='cpu',
    chunk_size=20,
    topology_mode='mixed',
    max_retries=100,
    dataset_type="interface",
    channel_max_horizontal=5,
    channel_max_vertical=5,
    channel_width_range=(0.04, 0.10),
    channel_min_width_cells=3,
    channel_max_width_cells=None,
    channel_allow_disconnected=True,
    channel_touch_boundary=True,
    channel_boundary_margin=0.0,
    channel_style="legacy_full",
):
    """
    Generate parameter fields only, without building any error snapshots.
    """
    batch_size = _validate_positive_count("batch_size", batch_size)
    chunk_size = _validate_positive_count("chunk_size", chunk_size)
    dataset_type = _canonical_dataset_type(dataset_type)
    if dataset_type == "interface":
        _validate_topology_mode(topology_mode)
    channelized_config = _resolve_channelized_config({
        "channel_max_horizontal": channel_max_horizontal,
        "channel_max_vertical": channel_max_vertical,
        "channel_width_range": channel_width_range,
        "channel_min_width_cells": channel_min_width_cells,
        "channel_max_width_cells": channel_max_width_cells,
        "channel_allow_disconnected": channel_allow_disconnected,
        "channel_touch_boundary": channel_touch_boundary,
        "channel_boundary_margin": channel_boundary_margin,
        "channel_style": channel_style,
    })
    phi_list = []
    a_field_list = []

    for start_idx in range(0, batch_size, chunk_size):
        end_idx = min(start_idx + chunk_size, batch_size)
        chunk_batch_size = end_idx - start_idx

        if dataset_type == "interface":
            phi_chunk, a_field_chunk, _, _ = generate_random_interface_coeffs_with_topology(
                batch_size=chunk_batch_size,
                n=n,
                contrast=contrast,
                device=device,
                topology_mode=topology_mode,
                max_retries=max_retries,
            )
        else:
            phi_chunk, a_field_chunk, _ = generate_channelized_fields(
                batch_size=chunk_batch_size,
                n=n,
                contrast=contrast,
                max_horizontal=channelized_config["max_horizontal"],
                max_vertical=channelized_config["max_vertical"],
                width_range=channelized_config["width_range"],
                min_width_cells=channelized_config["min_width_cells"],
                allow_disconnected=channelized_config["allow_disconnected"],
                device=device,
                channel_touch_boundary=channelized_config["touch_boundary"],
                channel_boundary_margin=channelized_config["boundary_margin"],
                channel_style=channelized_config["style"],
                channel_max_width_cells=channelized_config["max_width_cells"],
            )

        phi_list.append(_detach_to_cpu(phi_chunk))
        a_field_list.append(_detach_to_cpu(a_field_chunk))
        del phi_chunk, a_field_chunk
        _empty_cuda_cache_if_needed(device)

    phi = torch.cat(phi_list, dim=0)
    a_field = torch.cat(a_field_list, dim=0)

    h = 1.0 / n
    x_1d = torch.linspace(h / 2, 1.0 - h / 2, n, device="cpu")
    Y, X = torch.meshgrid(x_1d, x_1d, indexing='ij')
    coords = torch.stack([X.flatten(), Y.flatten()], dim=1)

    return phi, a_field, coords, X, Y


# ==============================================================================
# 3. 本地缓存与数据加载器 (Caching & DataLoader API)
# ==============================================================================
def create_cached_family_data_loader(
    config,
    batch_size,
    n,
    M_smooth,
    num_samples,
    num_probes,
    contrast=1e4,
    split='train',
    topology_mode=None,
    max_retries=None,
):
    compute_device = config.get('device', 'cpu')
    logger = config.get('logger')
    cache_dir = config.get('dataset_cache_dir', './datasets/interface_family')
    os.makedirs(cache_dir, exist_ok=True)
    transmissibility = _canonical_transmissibility(
        config.get("transmissibility", "harmonic")
    )
    dataset_type = _canonical_dataset_type(config.get("dataset_type", "interface"))
    topology_mode = config.get('topology_mode', 'mixed') if topology_mode is None else topology_mode
    max_retries = config.get('max_topology_retries', 100) if max_retries is None else max_retries
    requested_svd_rank = _resolve_cached_svd_rank(config, n=n, num_probes=num_probes)
    requested_whiten_rank = _resolve_cached_capture_whiten_rank(
        config,
        n=n,
        num_probes=num_probes,
    )
    requested_whiten_metric = _resolve_cached_capture_whiten_metric(config)
    vectorized_probes = bool(config.get("vectorized_probes", True))
    snapshot_smoother = _snapshot_smoother_from_config(config)
    snapshot_omega = float(config.get("snapshot_smooth_omega", 0.8))
    channelized_config = (
        _resolve_channelized_config(config)
        if dataset_type == "channelized"
        else _resolve_channelized_config({})
    )
    if dataset_type == "interface":
        _validate_topology_mode(topology_mode)
    num_samples = _validate_positive_count("num_samples", num_samples)
    batch_size = _validate_positive_count("batch_size", batch_size)
    loader_batch_size = _resolve_loader_batch_size(num_samples, batch_size)

    # Adaptive whitening config
    whiten_rank_mode = _resolve_whiten_rank_mode(config)
    whiten_rank_min, whiten_rank_max, whiten_rank_floor = _resolve_adaptive_whiten_params(
        config, num_probes=num_probes
    )
    whiten_beta = _resolve_whiten_beta(config)
    if whiten_rank_mode == "adaptive" and config.get("whiten_rank") is not None:
        warnings.warn(
            "whiten_rank_mode='adaptive': the 'whiten_rank' config key is ignored. "
            "Use 'whiten_rank_min' / 'whiten_rank_max' to control the adaptive range.",
            UserWarning,
            stacklevel=2,
        )

    _log = logger.info if logger is not None else print
    _log(
        f"[WhitenConfig] mode={whiten_rank_mode!r} "
        f"rank={config.get('whiten_rank', 'N/A')} "
        f"min={whiten_rank_min} max={whiten_rank_max} "
        f"floor_relative={whiten_rank_floor:.1e} "
        f"metric={requested_whiten_metric!r} "
        f"beta={whiten_beta:.4f}"
    )

    smoothing_tag = _snapshot_smoothing_cache_tag(config)
    whiten_tag = _whiten_mode_cache_tag(config)
    parameter_tag = _dataset_parameter_cache_tag(config, dataset_type, contrast, topology_mode)
    transmissibility_tag = (
        "" if transmissibility == "harmonic" else f"_tx{transmissibility}"
    )
    filename = (
        f"dataset_{split}_N{num_samples}_n{n}_M{M_smooth}_q{num_probes}_"
        f"{parameter_tag}{transmissibility_tag}{smoothing_tag}{whiten_tag}.pt"
    )
    file_path = os.path.join(cache_dir, filename)

    if batch_size > num_samples:
        message = (
            f"[{split.capitalize()}] Requested batch_size={batch_size} exceeds num_samples={num_samples}; "
            f"using loader batch_size={loader_batch_size}."
        )
        if logger is not None:
            logger.warning(message)
        else:
            print(message)

    vectorized_message = (
        f"[DataGen] vectorized_probes={vectorized_probes}, num_probes={num_probes}, "
        f"snapshot_smoother={snapshot_smoother}, omega={snapshot_omega}"
    )
    if logger is not None:
        logger.info(vectorized_message)
    else:
        print(vectorized_message)

    if os.path.exists(file_path):
        print(f"[{split.capitalize()}] Loading cached dataset from {file_path}")
        dataset_dict = torch.load(file_path, map_location="cpu")
        cached_transmissibility = _canonical_transmissibility(
            dataset_dict.get("transmissibility", "harmonic")
        )
        if cached_transmissibility != transmissibility:
            raise ValueError(
                "cached dataset transmissibility mismatch: "
                f"requested={transmissibility!r}, cached={cached_transmissibility!r}, "
                f"path={file_path}"
            )
        if transmissibility != "harmonic" and dataset_dict.get("phi") is None:
            raise ValueError(
                f"interface-aware cached dataset does not contain phi: {file_path}"
            )
        dataset_dict, _ = _ensure_cached_svd_basis(
            dataset_dict,
            requested_rank=requested_svd_rank,
            chunk_size=config.get('chunk_size', 20),
            file_path=file_path,
            split=split,
            compute_device=compute_device,
        )
        dataset_dict, _ = _ensure_cached_whitened_family(
            dataset_dict,
            requested_rank=requested_whiten_rank,
            requested_metric=requested_whiten_metric,
            chunk_size=config.get('chunk_size', 20),
            k_1d=config.get('k_1d', 10),
            file_path=file_path,
            split=split,
            compute_device=compute_device,
            whiten_rank_mode=whiten_rank_mode,
            whiten_rank_min=whiten_rank_min,
            whiten_rank_max=whiten_rank_max,
            whiten_rank_floor_relative=whiten_rank_floor,
            whiten_beta=whiten_beta,
            transmissibility=transmissibility,
        )
        a_field = dataset_dict['a_field']
        phi = dataset_dict['phi']
        X_flat = dataset_dict['X_flat']
        Y_flat = dataset_dict['Y_flat']
        e_family = dataset_dict['e_family']
        svd_basis = dataset_dict['svd_basis']
        snapshot_smoothing_steps = dataset_dict.get('snapshot_smoothing_steps')
        if requested_whiten_metric is None:
            e_family_white = None
        else:
            whiten_field_name, _ = _whitened_family_field_names(requested_whiten_metric)
            e_family_white = dataset_dict.get(whiten_field_name)
    else:
        print(f"[{split.capitalize()}] Cache not found. Generating new dataset...")
        generated = generate_data_family(
            batch_size=num_samples,
            n=n,
            M_smooth=M_smooth,
            num_probes=num_probes,
            contrast=contrast,
            device=compute_device,
            chunk_size=config.get('chunk_size', 20),
            k_1d=config.get('k_1d', 10),
            omega=snapshot_omega,
            topology_mode=topology_mode,
            max_retries=max_retries,
            svd_rank=requested_svd_rank,
            whiten_rank=requested_whiten_rank,
            whiten_metric=requested_whiten_metric,
            vectorized_probes=vectorized_probes,
            snapshot_smoother=snapshot_smoother,
            snapshot_config=config,
            dataset_type=dataset_type,
            channel_max_horizontal=channelized_config["max_horizontal"],
            channel_max_vertical=channelized_config["max_vertical"],
            channel_width_range=channelized_config["width_range"],
            channel_min_width_cells=channelized_config["min_width_cells"],
            channel_max_width_cells=channelized_config["max_width_cells"],
            channel_allow_disconnected=channelized_config["allow_disconnected"],
            channel_touch_boundary=channelized_config["touch_boundary"],
            channel_boundary_margin=channelized_config["boundary_margin"],
            channel_style=channelized_config["style"],
            transmissibility=transmissibility,
        )
        if requested_whiten_rank is None:
            phi, a_field, coords, e_family, X, Y, svd_basis = generated
            e_family_white = None
        else:
            phi, a_field, coords, e_family, X, Y, svd_basis, e_family_white = generated

        X_flat = X.flatten().unsqueeze(0).unsqueeze(2).expand(num_samples, -1, -1).contiguous()
        Y_flat = Y.flatten().unsqueeze(0).unsqueeze(2).expand(num_samples, -1, -1).contiguous()

        dataset_dict = {
            'a_field': a_field.cpu(),
            'phi': phi.cpu(),
            'X_flat': X_flat.cpu(),
            'Y_flat': Y_flat.cpu(),
            'e_family': e_family.cpu(),
            'svd_basis': svd_basis.cpu(),
            'svd_rank': int(svd_basis.shape[-1]),
            'dataset_type': dataset_type,
            'topology_mode': topology_mode,
            'channelized_config': (
                _resolve_channelized_config(config) if dataset_type == "channelized" else None
            ),
            'vectorized_probes': bool(vectorized_probes),
            'snapshot_smoother': snapshot_smoother,
            'transmissibility': transmissibility,
            'contrast': float(contrast),
            'grid_size': int(n),
            'level_set_type': topology_mode if dataset_type == "interface" else "channelized_signed_field",
            'random_seed': int(config.get("seed", config.get("random_seed", 0))),
            'asm_parameters': {
                "k_1d": int(config.get("k_1d", 10)),
                "target_overlap": float(config.get("target_overlap", 0.3)),
                "subdomain_solver": str(config.get("subdomain_solver", "cholesky")),
                "jacobi_pcg_iters": int(config.get("jacobi_pcg_iters", 10)),
            },
            'snapshot_smoothing_steps': torch.full((num_samples, 1), int(M_smooth), dtype=torch.long),
            'snapshot_smoothing_omega': float(snapshot_omega),
            'snapshot_gmg_config': {
                "num_levels": config.get("snapshot_gmg_num_levels"),
                "min_coarse_size": int(config.get("snapshot_gmg_min_coarse_size", 4)),
                "smoother": str(config.get("snapshot_gmg_smoother", "jacobi")),
                "pre_smooth_steps": int(config.get("snapshot_gmg_pre_smooth_steps", 2)),
                "post_smooth_steps": int(config.get("snapshot_gmg_post_smooth_steps", 2)),
                "omega": float(config.get("snapshot_gmg_omega", 0.8)),
                "outer_omega": float(config.get("snapshot_gmg_outer_omega", snapshot_omega)),
                "coarse_solve": str(config.get("snapshot_gmg_coarse_solve", "direct")),
            } if snapshot_smoother == "gmg" else None,
        }
        if e_family_white is not None:
            whiten_field_name, whiten_rank_name = _whitened_family_field_names(requested_whiten_metric)
            if whiten_rank_mode == "adaptive":
                # Recompute adaptively over the full newly-generated e_family
                e_white_adp, k_per_sample, eigvals_ps = _compute_dataset_whitened_error_family_adaptive(
                    dataset_dict["e_family"],
                    metric=requested_whiten_metric,
                    k_max=whiten_rank_max,
                    min_rank=whiten_rank_min,
                    max_rank=whiten_rank_max,
                    floor_relative=whiten_rank_floor,
                    beta=whiten_beta,
                    a_field=dataset_dict.get("a_field"),
                    phi=dataset_dict.get("phi"),
                    transmissibility=transmissibility,
                    batch_chunk_size=config.get("chunk_size", 20),
                    k_1d=config.get("k_1d", 10),
                    compute_device=compute_device,
                )
                dataset_dict[whiten_field_name] = e_white_adp
                dataset_dict[whiten_rank_name] = whiten_rank_max
                dataset_dict[f"whiten_rank_per_sample_{requested_whiten_metric}"] = k_per_sample
                dataset_dict[f"whiten_eigvals_per_sample_{requested_whiten_metric}"] = eigvals_ps
                dataset_dict["whiten_rank_mode"] = "adaptive"
                dataset_dict["whiten_beta"] = whiten_beta
                dataset_dict["whiten_mode_signature"] = f"adaptive_beta{whiten_beta:.2f}"
                e_family_white = e_white_adp
            else:
                dataset_dict[whiten_field_name] = e_family_white.cpu()
                dataset_dict[whiten_rank_name] = int(requested_whiten_rank)
        torch.save(dataset_dict, file_path)
        print(f"[{split.capitalize()}] Dataset saved to {file_path}")
        snapshot_smoothing_steps = dataset_dict.get('snapshot_smoothing_steps')

    if (
        requested_whiten_rank is not None
        and e_family_white is not None
        and snapshot_smoothing_steps is not None
    ):
        dataset_tensors = (
            a_field,
            phi,
            X_flat,
            Y_flat,
            e_family,
            svd_basis,
            e_family_white,
            snapshot_smoothing_steps,
        )
    elif requested_whiten_rank is not None and e_family_white is not None:
        dataset_tensors = (a_field, phi, X_flat, Y_flat, e_family, svd_basis, e_family_white)
    elif snapshot_smoothing_steps is not None:
        dataset_tensors = (a_field, phi, X_flat, Y_flat, e_family, svd_basis, snapshot_smoothing_steps)
    else:
        dataset_tensors = (a_field, phi, X_flat, Y_flat, e_family, svd_basis)

    dataset_tensors, dataset_on_gpu = _maybe_preload_dataset_tensors_to_gpu(
        dataset_tensors,
        device=compute_device,
        enabled=bool(config.get('preload_dataset_to_gpu', False)),
        safety_margin_gb=float(config.get('preload_dataset_safety_margin_gb', 8.0)),
        logger=logger,
    )

    dataset = TensorDataset(*dataset_tensors)

    if dataset_on_gpu:
        num_workers = 0
        pin_memory = False
        loader_kwargs = {}
    else:
        num_workers = int(config.get('num_workers', config.get('dataloader_num_workers', 4)))
        pin_memory = bool(config.get('pin_memory', _is_cuda_device(compute_device)))
        loader_kwargs = {}
        if num_workers > 0:
            loader_kwargs['persistent_workers'] = bool(config.get('persistent_workers', split == 'train'))
            loader_kwargs['prefetch_factor'] = int(config.get('prefetch_factor', 2))

    loader = DataLoader(
        dataset,
        batch_size=loader_batch_size,
        shuffle=(split == 'train'),
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **loader_kwargs,
    )

    return loader, file_path


def prepare_interface_parameter_dataset(
    config,
    split,
    num_samples,
    n,
    contrast=1e4,
    topology_mode=None,
    max_retries=None,
):
    """用于 Periodic Test 和 Eval 生成评估参数基底"""
    compute_device = config.get('device', 'cpu')
    cache_dir = config.get('dataset_cache_dir', './datasets/interface_family')
    os.makedirs(cache_dir, exist_ok=True)
    dataset_type = _canonical_dataset_type(config.get("dataset_type", "interface"))
    topology_mode = config.get('topology_mode', 'mixed') if topology_mode is None else topology_mode
    max_retries = config.get('max_topology_retries', 100) if max_retries is None else max_retries
    if dataset_type == "interface":
        _validate_topology_mode(topology_mode)
    channelized_config = (
        _resolve_channelized_config(config) if dataset_type == "channelized" else _resolve_channelized_config({})
    )

    smoothing_tag = _snapshot_smoothing_cache_tag(config)
    parameter_tag = _dataset_parameter_cache_tag(config, dataset_type, contrast, topology_mode)
    filename = f"params_{split}_N{num_samples}_n{n}_{parameter_tag}{smoothing_tag}.pt"
    file_path = os.path.join(cache_dir, filename)

    if os.path.exists(file_path):
        print(f"[Eval Param] Loading cached parameters from {file_path}")
        payload = torch.load(file_path, map_location="cpu")
    else:
        print(f"[Eval Param] Cache not found. Generating...")
        # 评估参数集只需要几何/系数字段，避免额外构建求解器和误差快照。
        phi, a_field, coords, X, Y = generate_parameter_data(
            batch_size=num_samples,
            n=n,
            contrast=contrast,
            device=compute_device,
            chunk_size=config.get('chunk_size', 20),
            topology_mode=topology_mode,
            max_retries=max_retries,
            dataset_type=dataset_type,
            channel_max_horizontal=channelized_config["max_horizontal"],
            channel_max_vertical=channelized_config["max_vertical"],
            channel_width_range=channelized_config["width_range"],
            channel_min_width_cells=channelized_config["min_width_cells"],
            channel_max_width_cells=channelized_config["max_width_cells"],
            channel_allow_disconnected=channelized_config["allow_disconnected"],
            channel_touch_boundary=channelized_config["touch_boundary"],
            channel_boundary_margin=channelized_config["boundary_margin"],
            channel_style=channelized_config["style"],
        )

        payload = {
            'phi': phi.to(torch.float64),
            'a_field': a_field.to(torch.float64),
            'coords': coords.to(torch.float64),
            'X': X.to(torch.float64),
            'Y': Y.to(torch.float64),
            'dataset_type': dataset_type,
            'topology_mode': topology_mode,
            'channelized_config': (
                channelized_config if dataset_type == "channelized" else None
            ),
        }
        torch.save(payload, file_path)

    return payload, file_path
