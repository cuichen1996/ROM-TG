"""Shared checkpoint and model-loading utilities.

Used by interface_train.py, two_phase_train.py, and two_phase_eval.py so that
evaluation scripts do not need to import training-entry-point modules.
"""

import numpy as np
import torch


def setup_seed(seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))


def get_model_core(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _extract_state_dict(checkpoint_state):
    if isinstance(checkpoint_state, dict) and "state_dict" in checkpoint_state:
        return checkpoint_state["state_dict"]
    return checkpoint_state


def _normalize_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid state_dict")
    if state_dict and all(isinstance(key, str) and key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def _summarize_state_dict_compatibility(model_state, checkpoint_state):
    missing_keys = [key for key in model_state.keys() if key not in checkpoint_state]
    unexpected_keys = [key for key in checkpoint_state.keys() if key not in model_state]
    shape_mismatches = []
    for key in model_state.keys():
        if key not in checkpoint_state:
            continue
        if model_state[key].shape != checkpoint_state[key].shape:
            shape_mismatches.append(
                f"{key}: model={tuple(model_state[key].shape)} ckpt={tuple(checkpoint_state[key].shape)}"
            )
    return missing_keys, unexpected_keys, shape_mismatches


def _log_state_dict_incompatibilities(
    logger,
    checkpoint_path,
    context,
    missing_keys,
    unexpected_keys,
    shape_mismatches,
    *,
    strict,
):
    if logger is None:
        return
    level = logger.error if strict else logger.warning
    severity = "ERROR" if strict else "WARNING"
    level(f"[{context}] {severity}: checkpoint compatibility issue detected for {checkpoint_path}")
    if missing_keys:
        level(f"[{context}] Missing keys ({len(missing_keys)}):")
        for key in missing_keys:
            level(f"  - {key}")
    if unexpected_keys:
        level(f"[{context}] Unexpected keys ({len(unexpected_keys)}):")
        for key in unexpected_keys:
            level(f"  - {key}")
    if shape_mismatches:
        level(f"[{context}] Shape mismatches ({len(shape_mismatches)}):")
        for msg in shape_mismatches:
            level(f"  - {msg}")


def _load_model_state(model, checkpoint_path, device, strict=True, logger=None, context="checkpoint"):
    checkpoint_state = torch.load(checkpoint_path, map_location=device)
    checkpoint_state = _normalize_state_dict_keys(_extract_state_dict(checkpoint_state))
    model_core = get_model_core(model)
    model_state = model_core.state_dict()
    missing_keys, unexpected_keys, shape_mismatches = _summarize_state_dict_compatibility(
        model_state,
        checkpoint_state,
    )

    has_incompatibility = bool(missing_keys or unexpected_keys or shape_mismatches)
    if has_incompatibility:
        _log_state_dict_incompatibilities(
            logger,
            checkpoint_path,
            context,
            missing_keys,
            unexpected_keys,
            shape_mismatches,
            strict=strict,
        )

    if shape_mismatches:
        raise RuntimeError(
            f"[{context}] Checkpoint shape mismatch for {checkpoint_path}: "
            f"{len(shape_mismatches)} parameter(s) differ in shape."
        )

    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            f"[{context}] Strict checkpoint load failed for {checkpoint_path}: "
            f"{len(missing_keys)} missing key(s), {len(unexpected_keys)} unexpected key(s). "
            "Use explicit warm-start mapping instead of resume/test loading when architectures differ."
        )

    incompatible = model_core.load_state_dict(checkpoint_state, strict=strict)
    if not strict and logger is not None:
        if incompatible.missing_keys or incompatible.unexpected_keys:
            logger.warning(
                f"[{context}] Non-strict checkpoint load completed with "
                f"{len(incompatible.missing_keys)} missing key(s) and "
                f"{len(incompatible.unexpected_keys)} unexpected key(s)."
            )
        else:
            logger.info(f"[{context}] Non-strict checkpoint load matched all keys.")
