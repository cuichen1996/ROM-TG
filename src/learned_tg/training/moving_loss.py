"""Numerical routines used by the paper; no experiment orchestration."""
from __future__ import annotations
import torch
from learned_tg.model.two_phase import MobilityFVMOperatorASM
from learned_tg.model.losses.subspace import SubspaceLoss
from learned_tg.model.basis_utils.smoothing import smooth_basis as _smooth_two_phase_basis
from learned_tg.training.checkpoint import get_model_core

def _cuda_memory_info(device_idx):
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
    except Exception:
        props = torch.cuda.get_device_properties(device_idx)
        total_bytes = int(props.total_memory)
        free_bytes = total_bytes
    return int(free_bytes), int(total_bytes)

def _unpack_two_phase_batch(batch, config=None):
    """
    Return:
        a_field, phi, X_flat, Y_flat,
        e_family, e_family_white_a, svd_basis, snapshot_steps

    Supported formats:

    Old 6-tensor:
        a, phi, X, Y, e_family, snapshot_steps
        a, phi, X, Y, e_family, svd_basis

    Old 7-tensor:
        a, phi, X, Y, e_family, svd_basis, snapshot_steps
        a, phi, X, Y, e_family, e_family_white_a, snapshot_steps

    New 8-tensor:
        a, phi, X, Y, e_family, e_family_white_a, svd_basis, snapshot_steps
    """
    loss_type = "capture_whiten"
    if config is not None:
        loss_type = str(config.get("two_phase_loss_type", "capture_whiten")).lower()

    e_family_white_a = None
    svd_basis = None
    snapshot_steps = None

    if len(batch) == 6:
        a_field, phi, X_flat, Y_flat, e_family, extra = batch

        if torch.is_tensor(extra) and extra.dim() >= 3:
            if loss_type == "svd":
                svd_basis = extra
            elif loss_type == "capture_whiten":
                # Ambiguous old format. Prefer treating it as cached E_white
                # when capture_whiten is active.
                e_family_white_a = extra
            else:
                svd_basis = extra
        else:
            snapshot_steps = extra

    elif len(batch) == 7:
        a_field, phi, X_flat, Y_flat, e_family, sixth, seventh = batch

        if torch.is_tensor(sixth) and sixth.dim() >= 3:
            if loss_type == "svd":
                svd_basis = sixth
            elif loss_type == "capture_whiten":
                e_family_white_a = sixth
            else:
                svd_basis = sixth
        else:
            raise ValueError(
                "Unexpected 7-tensor two-phase batch: the 6th tensor should be "
                "svd_basis or e_family_white_a."
            )

        snapshot_steps = seventh

    elif len(batch) == 8:
        (
            a_field,
            phi,
            X_flat,
            Y_flat,
            e_family,
            e_family_white_a,
            svd_basis,
            snapshot_steps,
        ) = batch

    else:
        raise ValueError(f"Unexpected two-phase batch with {len(batch)} tensors")

    return (
        a_field,
        phi,
        X_flat,
        Y_flat,
        e_family,
        e_family_white_a,
        svd_basis,
        snapshot_steps,
    )

def _to_device(item, device, dtype=None):
    if item is None:
        return None
    kwargs = {"device": device}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if isinstance(device, torch.device) and device.type == "cuda":
        kwargs["non_blocking"] = True
    return item.to(**kwargs)

def _tensor_device_name(item):
    if item is None:
        return "None"
    if not torch.is_tensor(item):
        return type(item).__name__
    return f"{item.device}/{item.dtype}/{tuple(item.shape)}"

def _model_parameter_device(model):
    try:
        param = next(get_model_core(model).parameters())
    except StopIteration:
        return "no-parameters"
    return f"{param.device}/{param.dtype}"

def _format_cuda_memory(device):
    if not (isinstance(device, torch.device) and device.type == "cuda"):
        return None
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    free_bytes, total_bytes = _cuda_memory_info(device.index if device.index is not None else torch.cuda.current_device())
    return (
        f"allocated={allocated:.2f} GiB | reserved={reserved:.2f} GiB | "
        f"peak_allocated={max_allocated:.2f} GiB | "
        f"driver_free={free_bytes / 1024 ** 3:.2f}/{total_bytes / 1024 ** 3:.2f} GiB"
    )

def _maybe_log_device_probe(config, model, tensors):
    if not bool(config.get("log_device_probe", True)):
        return
    if bool(config.get("_device_probe_logged", False)):
        return
    logger = config.get("logger")
    if logger is None:
        return
    tensor_summary = " | ".join(
        f"{name}={_tensor_device_name(value)}"
        for name, value in tensors.items()
    )
    memory_summary = _format_cuda_memory(config.get("device"))
    logger.info(
        "device probe | model_param={} | {}".format(
            _model_parameter_device(model),
            tensor_summary,
        )
    )
    if memory_summary is not None:
        logger.info(f"device probe | cuda memory after first forward: {memory_summary}")
    config["_device_probe_logged"] = True

def _build_operator(a_field, config, *, phi=None, build_local_solver=False):
    if a_field.dim() == 3:
        a_field = a_field.unsqueeze(1)

    n = int(a_field.shape[-1])
    ny = int(a_field.shape[-2])

    kwargs = dict(
        lam=a_field.to(torch.float64),
        hx=1.0 / float(n),
        hy=1.0 / float(ny),
        p_left=float(config.get("p_in", 1.0)),
        p_right=float(config.get("p_out", 0.0)),
        kx_sub=int(config.get("k_1d", 8)),
        ky_sub=int(config.get("k_1d", 8)),
        target_overlap=float(config.get("target_overlap", 0.3)),
        subdomain_solver=str(config.get("subdomain_solver", "cholesky")),
        jacobi_pcg_iters=int(config.get("jacobi_pcg_iters", 10)),
        phi=phi,
        transmissibility=str(config.get("transmissibility", "harmonic")),
    )

    try:
        return MobilityFVMOperatorASM(
            **kwargs,
            build_local_solver=bool(build_local_solver),
        )
    except TypeError:
        # Backward-compatible fallback for older model/two_phase.py.
        return MobilityFVMOperatorASM(**kwargs)

def _compute_loss(model, batch, config, device, return_stats=True):
    (
        a_field,
        phi,
        X_flat,
        Y_flat,
        e_family,
        e_family_white_a,
        svd_basis,
        snapshot_steps,
    ) = _unpack_two_phase_batch(batch, config=config)

    loss_type = str(config.get("two_phase_loss_type", "capture_whiten")).lower()

    a_field = _to_device(a_field, device)
    phi = _to_device(phi, device)
    X_flat = _to_device(X_flat, device)
    Y_flat = _to_device(Y_flat, device)

    e_family_for_loss = _to_device(e_family, device)
    e_family_white_for_loss = None
    svd_basis_for_loss = None

    if loss_type == "svd":
        svd_basis_for_loss = _to_device(svd_basis, device)
    elif loss_type == "capture_whiten":
        if e_family_white_a is not None:
            e_family_white_for_loss = _to_device(e_family_white_a, device)
        else:
            logger = config.get("logger")
            if logger is not None and not bool(config.get("_warned_no_cached_ewhite", False)):
                logger.warning(
                    "capture_whiten is recomputing A-whitening online because "
                    "e_family_white_a is missing from the batch. "
                    "This is slow. Regenerate or update cache with "
                    "cache_capture_whiten=True / compute_whitened_family=True."
                )
                config["_warned_no_cached_ewhite"] = True

    if snapshot_steps is not None:
        snapshot_steps = _to_device(snapshot_steps, device)

    model_core = get_model_core(model)
    active_dim = int(config.get("m_dim", model_core.m_dim))

    V = model_core.predict_basis(
        a_field,
        phi,
        X_flat,
        Y_flat,
        active_dim=active_dim,
    )

    capture_whiten = (loss_type == "capture_whiten")
    subspace_loss_type = "svd" if loss_type == "svd" else "capture"
    subspace_metric = "l2" if loss_type == "svd" else "a"
    whiten_rank = config.get("whiten_rank", active_dim)
    whiten_metric = str(config.get("whiten_metric", "a"))
    criterion = SubspaceLoss(
        loss_type=subspace_loss_type,
        metric=subspace_metric,
        capture_whiten=capture_whiten,
        capture_whiten_rank=whiten_rank if capture_whiten else None,
        capture_whiten_metric=whiten_metric if capture_whiten else None,
        log_extra_metrics=True,
        orth_reg=float(config.get("orth_reg", 1e-10)),
        eps=float(config.get("loss_reg", 1e-12)),
        dtype=torch.float64,
    )

    if loss_type == "svd":
        loss, stats = criterion(
            V_raw=V,
            e_family=e_family_for_loss,
            svd_basis=svd_basis_for_loss,
            return_stats=True,
        )
    else:
        op = _build_operator(
            a_field,
            config,
            phi=phi,
            build_local_solver=False,
        )
        basis_smooth_mode = str(config.get("basis_smooth_mode", "none"))
        basis_smooth_steps = int(config.get("basis_smooth_steps", 0))
        if basis_smooth_mode != "none" and basis_smooth_steps > 0:
            V = _smooth_two_phase_basis(
                op, V,
                mode=basis_smooth_mode,
                num_steps=basis_smooth_steps,
                omega=float(config.get("basis_smooth_omega", 0.6)),
            )
        loss, stats = criterion(
            V_raw=V,
            e_family=e_family_for_loss,
            e_family_white=e_family_white_for_loss,
            e_family_white_metric=whiten_metric if e_family_white_for_loss is not None else None,
            op_solver=op,
            return_stats=True,
        )

    if loss_type not in {"svd", "capture", "capture_whiten"}:
        raise ValueError(
            "two_phase_loss_type must be one of {'svd', 'capture', 'capture_whiten'}, "
            f"got {loss_type!r}"
        )

    if snapshot_steps is not None:
        stats["snapshot_smoothing_steps"] = (
            snapshot_steps.to(torch.float64).mean().detach()
        )

    _maybe_log_device_probe(
        config,
        model,
        {
            "a_field": a_field,
            "phi": phi,
            "X_flat": X_flat,
            "Y_flat": Y_flat,
            "V": V,
            "svd_basis": svd_basis if loss_type == "svd" else None,
            "e_family": e_family_for_loss,
            "e_family_white_a": e_family_white_for_loss,
            "loss": loss,
        },
    )

    if return_stats:
        return loss, stats
    return loss
