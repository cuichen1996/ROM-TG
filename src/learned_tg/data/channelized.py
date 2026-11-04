"""Channelized high-contrast Darcy coefficient fields.

This module generates axis-aligned channel unions directly on the requested
cell-centered grid.  The signed distance is recomputed for that grid instead of
being interpolated from another resolution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


def _require_ndimage():
    try:
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise ImportError(
            "generate_channelized_fields requires scipy.ndimage for signed "
            "distance computation. Install scipy or provide a project-local "
            "distance-transform implementation."
        ) from exc
    return ndimage


def _normalize_width_range(
    width_range: Sequence[float],
    *,
    n: int,
    min_width_cells: Optional[int],
    max_width_cells: Optional[int],
) -> Tuple[float, float, float]:
    if len(width_range) != 2:
        raise ValueError(f"width_range must contain two values, got {width_range!r}")
    lo, hi = sorted(float(v) for v in width_range)
    if lo <= 0.0 or hi <= 0.0:
        raise ValueError(f"channel widths must be positive, got {width_range!r}")
    min_width = 0.0
    if min_width_cells is not None:
        min_width = max(1, int(min_width_cells)) / float(n)
        lo = max(lo, min_width)
    max_width = None
    if max_width_cells is not None:
        max_width = max(1, int(max_width_cells)) / float(n)
        if min_width > max_width + 1e-15:
            raise ValueError(
                "min_width_cells cannot exceed channel_max_width_cells "
                f"({min_width_cells} > {max_width_cells})."
            )
        hi = min(hi, max_width)
    hi = max(hi, lo)
    hi = min(hi, 0.95)
    if lo >= 1.0:
        raise ValueError(
            f"min_width_cells={min_width_cells} is too large for n={n}; "
            f"effective physical width would be {lo:.4f}."
        )
    return lo, hi, min_width


def _canonical_channel_style(channel_style: str) -> str:
    key = str(channel_style or "paper_internal_thin").strip().lower()
    aliases = {
        "legacy": "legacy_full",
        "legacy_full": "legacy_full",
        "full": "legacy_full",
        "full_span": "legacy_full",
        "paper": "paper_internal_thin",
        "paper_internal": "paper_internal_thin",
        "paper_internal_thin": "paper_internal_thin",
    }
    if key not in aliases:
        raise ValueError(
            "channel_style must be one of {'paper_internal_thin', 'legacy_full'}, "
            f"got {channel_style!r}."
        )
    return aliases[key]


def _rand_uniform(
    low: float,
    high: float,
    *,
    generator: Optional[torch.Generator],
) -> float:
    if high <= low:
        return float(low)
    value = torch.rand((), generator=generator).item()
    return float(low + (high - low) * value)


def _rand_int(
    low: int,
    high_exclusive: int,
    *,
    generator: Optional[torch.Generator],
) -> int:
    if high_exclusive <= low:
        return int(low)
    return int(torch.randint(low, high_exclusive, (1,), generator=generator).item())


def _make_generator(
    *,
    seed: Optional[int],
    generator: Optional[torch.Generator],
) -> Optional[torch.Generator]:
    if seed is not None and generator is not None:
        raise ValueError("Pass either seed or generator, not both.")
    if generator is not None:
        return generator
    if seed is None:
        return None
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def _sample_channel_counts(
    *,
    max_horizontal: int,
    max_vertical: int,
    allow_disconnected: bool,
    generator: Optional[torch.Generator],
) -> Tuple[int, int]:
    if max_horizontal < 0 or max_vertical < 0:
        raise ValueError("max_horizontal and max_vertical must be nonnegative.")
    if max_horizontal == 0 and max_vertical == 0:
        raise ValueError("At least one of max_horizontal/max_vertical must be positive.")

    if not allow_disconnected:
        if max_horizontal > 0 and max_vertical > 0:
            return (
                _rand_int(1, max_horizontal + 1, generator=generator),
                _rand_int(1, max_vertical + 1, generator=generator),
            )
        if max_horizontal > 0:
            return 1, 0
        return 0, 1

    h_count = _rand_int(0, max_horizontal + 1, generator=generator)
    v_count = _rand_int(0, max_vertical + 1, generator=generator)
    if h_count == 0 and v_count == 0:
        if max_horizontal > 0 and max_vertical > 0:
            if _rand_uniform(0.0, 1.0, generator=generator) < 0.5:
                h_count = 1
            else:
                v_count = 1
        elif max_horizontal > 0:
            h_count = 1
        else:
            v_count = 1
    return h_count, v_count


def _connected_components(mask: np.ndarray) -> Tuple[bool, int]:
    ndimage = _require_ndimage()
    structure = np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        dtype=bool,
    )
    _, component_count = ndimage.label(mask, structure=structure)
    return bool(component_count <= 1 and mask.any()), int(component_count)


def _signed_distance(mask: np.ndarray, *, h: float) -> np.ndarray:
    ndimage = _require_ndimage()
    outside_distance = ndimage.distance_transform_edt(~mask, sampling=(h, h))
    inside_distance = ndimage.distance_transform_edt(mask, sampling=(h, h))
    return outside_distance - inside_distance


def _sample_interval_within_margin(
    *,
    margin: float,
    min_length: float,
    max_length: float,
    generator: Optional[torch.Generator],
) -> Tuple[float, float, float]:
    domain_lo = float(margin)
    domain_hi = float(1.0 - margin)
    span = max(0.0, domain_hi - domain_lo)
    if span <= 0.0:
        raise ValueError(f"channel_boundary_margin={margin} leaves no interior span.")
    lo_len = min(max(float(min_length), 0.0), span)
    hi_len = min(max(float(max_length), lo_len), span)
    length = _rand_uniform(lo_len, hi_len, generator=generator)
    start = _rand_uniform(domain_lo, domain_hi - length, generator=generator)
    end = start + length
    return float(start), float(end), float(length)


def _sample_strip_bounds(
    *,
    width: float,
    margin: float,
    generator: Optional[torch.Generator],
) -> Tuple[float, float, float]:
    lo = float(margin) + 0.5 * float(width)
    hi = 1.0 - float(margin) - 0.5 * float(width)
    if hi < lo:
        raise ValueError(
            "channel width and boundary margin leave no valid strip center: "
            f"width={width:.4f}, margin={margin:.4f}."
        )
    center = _rand_uniform(lo, hi, generator=generator)
    return float(center - width / 2.0), float(center + width / 2.0), float(center)


def _touches_boundary(channels: List[Dict[str, Any]], tol: float = 1e-12) -> bool:
    for channel in channels:
        if (
            float(channel["x0"]) <= tol
            or float(channel["y0"]) <= tol
            or float(channel["x1"]) >= 1.0 - tol
            or float(channel["y1"]) >= 1.0 - tol
        ):
            return True
    return False


def generate_channelized_fields(
    batch_size: int,
    n: int,
    contrast: float = 1e4,
    max_horizontal: int = 5,
    max_vertical: int = 5,
    width_range: Sequence[float] = (0.012, 0.035),
    min_width_cells: Optional[int] = None,
    allow_disconnected: bool = True,
    device: Union[torch.device, str] = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    channel_touch_boundary: bool = False,
    channel_boundary_margin: float = 0.05,
    channel_style: str = "paper_internal_thin",
    channel_max_width_cells: Optional[int] = 3,
    channel_min_width_cells: Optional[int] = None,
):
    """Generate channelized coefficient fields on a cell-centered unit square.

    Returns:
        phi: ``(batch_size, 1, n, n)`` signed distance, negative inside channels.
        a: ``(batch_size, 1, n, n)`` with values ``1`` and ``contrast``.
        meta: list of per-sample dictionaries describing channel geometry.
    """
    batch_size = int(batch_size)
    n = int(n)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if n <= 1:
        raise ValueError(f"n must be greater than 1, got {n}.")
    contrast = float(contrast)
    if contrast <= 0.0:
        raise ValueError(f"contrast must be positive, got {contrast}.")

    channel_style = _canonical_channel_style(channel_style)
    if channel_min_width_cells is not None:
        min_width_cells = int(channel_min_width_cells)
    if min_width_cells is None:
        min_width_cells = 2 if channel_style == "paper_internal_thin" else 3
    if channel_style == "legacy_full":
        channel_touch_boundary = True
        channel_boundary_margin = 0.0
        if channel_max_width_cells == 3:
            channel_max_width_cells = None

    max_horizontal = int(max_horizontal)
    max_vertical = int(max_vertical)
    allow_disconnected = bool(allow_disconnected)
    generator = _make_generator(seed=seed, generator=generator)
    width_lo, width_hi, min_width = _normalize_width_range(
        width_range,
        n=n,
        min_width_cells=int(min_width_cells),
        max_width_cells=channel_max_width_cells,
    )
    max_width = None if channel_max_width_cells is None else max(1, int(channel_max_width_cells)) / float(n)
    margin = 0.0 if channel_touch_boundary else float(channel_boundary_margin)
    if margin < 0.0 or margin >= 0.5:
        raise ValueError(f"channel_boundary_margin must satisfy 0 <= margin < 0.5, got {margin}.")

    h = 1.0 / float(n)
    centers = (np.arange(n, dtype=np.float64) + 0.5) * h
    x_grid, y_grid = np.meshgrid(centers, centers, indexing="xy")

    phi_samples: List[np.ndarray] = []
    a_samples: List[np.ndarray] = []
    meta: List[Dict[str, Any]] = []

    for sample_idx in range(batch_size):
        max_attempts = 256 if (channel_style == "paper_internal_thin" and not allow_disconnected) else 1
        accepted = False
        for attempt_idx in range(max_attempts):
            h_count, v_count = _sample_channel_counts(
                max_horizontal=max_horizontal,
                max_vertical=max_vertical,
                allow_disconnected=allow_disconnected,
                generator=generator,
            )
            mask = np.zeros((n, n), dtype=bool)
            channels: List[Dict[str, Any]] = []

            for _ in range(h_count):
                width = _rand_uniform(width_lo, width_hi, generator=generator)
                if channel_style == "legacy_full":
                    center = _rand_uniform(width / 2.0, 1.0 - width / 2.0, generator=generator)
                    x0, x1 = 0.0, 1.0
                    y0 = max(0.0, center - width / 2.0)
                    y1 = min(1.0, center + width / 2.0)
                    length = 1.0
                else:
                    y0, y1, center = _sample_strip_bounds(
                        width=width,
                        margin=margin,
                        generator=generator,
                    )
                    interior_span = 1.0 - 2.0 * margin
                    min_len = max(0.25 * interior_span, 6.0 * width)
                    max_len = interior_span
                    x0, x1, length = _sample_interval_within_margin(
                        margin=margin,
                        min_length=min_len,
                        max_length=max_len,
                        generator=generator,
                    )
                mask |= (x_grid >= x0) & (x_grid <= x1) & (y_grid >= y0) & (y_grid <= y1)
                channels.append(
                    {
                        "orientation": "horizontal",
                        "x0": float(x0),
                        "x1": float(x1),
                        "y0": float(y0),
                        "y1": float(y1),
                        "center": float(center),
                        "width": float(width),
                        "width_cells": float(width * n),
                        "length": float(length),
                    }
                )

            for _ in range(v_count):
                width = _rand_uniform(width_lo, width_hi, generator=generator)
                if channel_style == "legacy_full":
                    center = _rand_uniform(width / 2.0, 1.0 - width / 2.0, generator=generator)
                    x0 = max(0.0, center - width / 2.0)
                    x1 = min(1.0, center + width / 2.0)
                    y0, y1 = 0.0, 1.0
                    length = 1.0
                else:
                    x0, x1, center = _sample_strip_bounds(
                        width=width,
                        margin=margin,
                        generator=generator,
                    )
                    interior_span = 1.0 - 2.0 * margin
                    min_len = max(0.25 * interior_span, 6.0 * width)
                    max_len = interior_span
                    y0, y1, length = _sample_interval_within_margin(
                        margin=margin,
                        min_length=min_len,
                        max_length=max_len,
                        generator=generator,
                    )
                mask |= (x_grid >= x0) & (x_grid <= x1) & (y_grid >= y0) & (y_grid <= y1)
                channels.append(
                    {
                        "orientation": "vertical",
                        "x0": float(x0),
                        "x1": float(x1),
                        "y0": float(y0),
                        "y1": float(y1),
                        "center": float(center),
                        "width": float(width),
                        "width_cells": float(width * n),
                        "length": float(length),
                    }
                )

            connected, component_count = _connected_components(mask)
            if allow_disconnected or connected:
                accepted = True
                break

        if not accepted:
            raise RuntimeError(
                "Failed to sample a connected channelized field. "
                "Consider increasing max_horizontal/max_vertical or relaxing "
                "the channel geometry constraints."
            )

        phi = _signed_distance(mask, h=h).astype(np.float64, copy=False)
        a = np.where(mask, contrast, 1.0).astype(np.float64, copy=False)

        phi_samples.append(phi)
        a_samples.append(a)
        meta.append(
            {
                "sample_index": int(sample_idx),
                "num_horizontal": int(h_count),
                "num_vertical": int(v_count),
                "num_channels": int(h_count + v_count),
                "channels": channels,
                "connected": connected,
                "num_connected_components": int(component_count),
                "allow_disconnected": allow_disconnected,
                "channel_style": channel_style,
                "channel_touch_boundary": bool(channel_touch_boundary),
                "channel_boundary_margin": float(margin),
                "touches_boundary": bool(_touches_boundary(channels)),
                "width_range": (float(width_lo), float(width_hi)),
                "requested_width_range": tuple(float(v) for v in width_range),
                "min_width_cells": int(min_width_cells),
                "max_width_cells": None if channel_max_width_cells is None else int(channel_max_width_cells),
                "min_physical_width": float(min_width),
                "max_physical_width": None if max_width is None else float(max_width),
            }
        )

    phi_np = np.stack(phi_samples, axis=0)[:, None, :, :]
    a_np = np.stack(a_samples, axis=0)[:, None, :, :]
    phi_tensor = torch.as_tensor(phi_np, dtype=dtype, device=device)
    a_tensor = torch.as_tensor(a_np, dtype=dtype, device=device)
    return phi_tensor, a_tensor, meta
