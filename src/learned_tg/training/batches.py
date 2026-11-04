"""Numerical routines used by the paper; no experiment orchestration."""
from __future__ import annotations
import random
import torch

class MultiResolutionFamilyLoader:
    """Yield batches from multiple per-grid DataLoaders with weighted sampling.

    Each yielded batch comes from a single grid resolution. Different batches
    may come from different resolutions according to the configured weights.
    The total length equals the sum of all per-grid loader lengths, preserving
    the "one pass over all data" semantics for the outer epoch loop.
    """

    def __init__(self, loaders, weights, seed=None):
        if not loaders:
            raise ValueError("At least one loader is required")
        if len(loaders) != len(weights):
            raise ValueError(
                f"loaders ({len(loaders)}) and weights ({len(weights)}) length mismatch"
            )
        self._loaders = list(loaders)
        total = float(sum(weights))
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        self._weights = [float(w) / total for w in weights]
        self._rng = random.Random(seed)  # shared state across epochs; not reset per __iter__

    def __len__(self):
        return sum(len(loader) for loader in self._loaders)

    def __iter__(self):
        iters = [iter(loader) for loader in self._loaders]
        for _ in range(len(self)):
            idx = self._rng.choices(range(len(iters)), weights=self._weights)[0]
            try:
                yield next(iters[idx])
            except StopIteration:
                iters[idx] = iter(self._loaders[idx])
                yield next(iters[idx])

    @property
    def per_loader_grid_sizes(self):
        """Infer grid size n from the first tensor of each loader's dataset."""
        sizes = []
        for loader in self._loaders:
            ds = getattr(loader, 'dataset', None)
            n = None
            if ds is not None and len(ds) > 0:
                first = ds[0]
                if isinstance(first, (tuple, list)) and len(first) > 0:
                    t = first[0]
                    if torch.is_tensor(t) and t.dim() >= 2:
                        n = int(t.shape[-1])
            sizes.append(n)
        return tuple(sizes)

def _unpack_family_batch(batch, config=None):
    config_view = {} if config is None else config
    prefer_white_family = (
        str(config_view.get("capture_family", "raw")).lower() == "white"
        and str(config_view.get("loss_objective", "capture")).lower() == "capture"
    )
    if len(batch) == 5:
        a_field_batch, phi_field_batch, X_batch, Y_batch, e_family_batch = batch
        svd_basis_batch = None
        e_family_white_batch = None
        snapshot_smoothing_steps_batch = None
    elif len(batch) == 6:
        a_field_batch, phi_field_batch, X_batch, Y_batch, e_family_batch, extra_batch = batch
        if torch.is_tensor(extra_batch) and extra_batch.dim() >= 3:
            if prefer_white_family:
                svd_basis_batch = None
                e_family_white_batch = extra_batch
            else:
                svd_basis_batch = extra_batch
                e_family_white_batch = None
            snapshot_smoothing_steps_batch = None
        else:
            svd_basis_batch = None
            e_family_white_batch = None
            snapshot_smoothing_steps_batch = extra_batch
    elif len(batch) == 7:
        (
            a_field_batch,
            phi_field_batch,
            X_batch,
            Y_batch,
            e_family_batch,
            extra_batch_1,
            extra_batch_2,
        ) = batch
        if (
            torch.is_tensor(extra_batch_1) and extra_batch_1.dim() >= 3
            and torch.is_tensor(extra_batch_2) and extra_batch_2.dim() >= 3
        ):
            svd_basis_batch = extra_batch_1
            e_family_white_batch = extra_batch_2
            snapshot_smoothing_steps_batch = None
        elif torch.is_tensor(extra_batch_1) and extra_batch_1.dim() >= 3:
            if prefer_white_family:
                svd_basis_batch = None
                e_family_white_batch = extra_batch_1
            else:
                svd_basis_batch = extra_batch_1
                e_family_white_batch = None
            snapshot_smoothing_steps_batch = extra_batch_2
        else:
            svd_basis_batch = None
            e_family_white_batch = extra_batch_2 if (
                torch.is_tensor(extra_batch_2) and extra_batch_2.dim() >= 3
            ) else None
            snapshot_smoothing_steps_batch = extra_batch_1
    elif len(batch) == 8:
        (
            a_field_batch,
            phi_field_batch,
            X_batch,
            Y_batch,
            e_family_batch,
            svd_basis_batch,
            e_family_white_batch,
            snapshot_smoothing_steps_batch,
        ) = batch
    else:
        raise ValueError(f"Unexpected family batch size: {len(batch)} tensors")
    return (
        a_field_batch,
        phi_field_batch,
        X_batch,
        Y_batch,
        e_family_batch,
        svd_basis_batch,
        e_family_white_batch,
        snapshot_smoothing_steps_batch,
    )

def _move_family_batch_to_device(batch, device):
    moved = []
    for item in batch:
        if item is None:
            moved.append(None)
        else:
            moved.append(item.to(device))
    return tuple(moved)

def _unwrap_loss_output(loss_output):
    if isinstance(loss_output, tuple):
        loss, stats = loss_output
    else:
        loss, stats = loss_output, None

    if torch.is_tensor(loss) and loss.dim() > 0:
        loss = loss.mean()
    return loss, stats
