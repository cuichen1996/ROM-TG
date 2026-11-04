"""Modified GenEO/local-spectral coarse spaces for cell-centred FVM.

The mathematical pencil is Spillane's thesis Definition 4.38, equation (4.25):

    A_j^N p = lambda D_j A_j^N D_j p.

``A_j^N`` is assembled from FVM face transmissibilities with natural boundary
conditions on artificial patch boundaries.  The reciprocal pencil is used only
inside the sparse eigensolver; every public eigenvalue and threshold uses the
thesis ``lambda`` convention in ascending order.

The fixed-rank provider in this module is deliberately called a compressed
local-spectral dictionary.  It is not a threshold-selected GenEO coarse space
and does not inherit the robust GenEO theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import warnings

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as spla
import torch


_SIDES = ("west", "east", "north", "south")
_ALL_DIRICHLET = {side: "dirichlet" for side in _SIDES}
_MOVING_INTERFACE_BOUNDARY = {
    "west": "dirichlet",
    "east": "dirichlet",
    "north": "neumann",
    "south": "neumann",
}
_TOPOLOGY_CACHE: Dict[Tuple[Any, ...], "LocalSpectralTopologyCache"] = {}


class ThresholdBracketError(RuntimeError):
    """Raised when the first finite eigenvalue above a threshold is unknown."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any]):
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


@dataclass(frozen=True)
class LocalSpectralGlobalData:
    """One CPU snapshot of all coefficient-dependent global FVM data."""

    east: np.ndarray
    west: np.ndarray
    north: np.ndarray
    south: np.ndarray
    multiplicity: np.ndarray
    n: int
    boundary_types: Dict[str, str]
    dtype: str
    source_device: str
    transfer_time: float
    full_grid_cpu_transfer_count: int


@dataclass(frozen=True)
class LocalSubdomainTopology:
    """Coefficient-independent topology and PoU data for one ASM patch."""

    subdomain: int
    patch_row: int
    patch_col: int
    row_start: int
    col_start: int
    height: int
    width: int
    global_indices: np.ndarray
    pou_weights: np.ndarray
    internal_i: np.ndarray
    internal_j: np.ndarray
    internal_axis: np.ndarray
    internal_global_rows: np.ndarray
    internal_global_cols: np.ndarray
    coo_rows: np.ndarray
    coo_cols: np.ndarray
    physical_local_indices: Dict[str, np.ndarray]
    physical_global_rows: Dict[str, np.ndarray]
    physical_global_cols: Dict[str, np.ndarray]
    touches_physical_boundary: Dict[str, bool]
    connected_components: int


@dataclass(frozen=True)
class LocalSpectralTopologyCache:
    """All patch topology for one resolved ASM partition."""

    key: Tuple[Any, ...]
    patches: Dict[int, LocalSubdomainTopology]
    active_subdomain_ids: Tuple[int, ...]
    padding_only_slots: Tuple[int, ...]
    build_time: float


@dataclass
class LocalSubdomain:
    """Numeric local pencil data for one active ASM patch."""

    subdomain: int
    patch_row: int
    patch_col: int
    row_start: int
    col_start: int
    height: int
    width: int
    global_indices: np.ndarray
    pou_weights: np.ndarray
    A_neumann: sp.csr_matrix
    B_pou_energy: sp.csr_matrix
    physical_boundary_faces: int
    physical_neumann_faces: int
    has_neumann_nullspace: bool
    boundary_types: Dict[str, str]
    touches_physical_boundary: Dict[str, bool]
    connected_components: int
    face_transmissibility_min: float
    face_transmissibility_median: float
    face_transmissibility_max: float
    A_norm_inf: float
    B_norm_inf: float
    A_symmetry_error: float
    B_symmetry_error: float
    numeric_assembly_time: float


@dataclass
class LocalMode:
    """One accepted local thesis-convention eigenmode."""

    subdomain: int
    local_mode: int
    eigenvalue: float
    reciprocal_eigenvalue: float
    vector: np.ndarray
    global_indices: np.ndarray
    pou_weights: np.ndarray
    is_neumann_null_mode: bool
    rayleigh_quotient: float
    eig_residual: float
    relative_rq_error: Optional[float]
    lambda_scale_floor: float
    zero_mode_nullspace_error: Optional[float]
    reciprocal_residual: Optional[float]

    @property
    def relative_residual(self) -> float:
        """Compatibility alias; this now is the direct-pencil residual."""

        return float(self.eig_residual)


def clear_local_spectral_topology_cache() -> None:
    """Clear the process-local coefficient-independent topology cache."""

    _TOPOLOGY_CACHE.clear()


def _normalize_boundary_types(
    boundary_types: Optional[Mapping[str, str]],
) -> Dict[str, str]:
    resolved = dict(_ALL_DIRICHLET)
    if boundary_types is not None:
        unknown = sorted(set(boundary_types) - set(_SIDES))
        if unknown:
            raise ValueError(f"unknown physical boundary side(s): {unknown}")
        resolved.update({str(k): str(v).lower() for k, v in boundary_types.items()})
    unsupported = {side: kind for side, kind in resolved.items() if kind not in {"dirichlet", "neumann"}}
    if unsupported:
        raise NotImplementedError(
            "local spectral FVM assembly supports only homogeneous Dirichlet/Neumann "
            f"face types; got {unsupported}"
        )
    return {side: resolved[side] for side in _SIDES}


def _sync_tensor_device(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def _copy_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Copy one tensor after the caller has established a transfer boundary."""

    return np.asarray(tensor.detach().cpu(), dtype=np.float64)


@dataclass(frozen=True)
class _LocalSpectralASMView:
    """Canonical square-grid view of an existing ASM operator.

    The view does not define a new Schwarz partition or local solver.  It only
    exposes the already-built partition and face transmissibilities in the
    names used by the cell-centred static-interface operator.
    """

    a: torch.Tensor
    se: torch.Tensor
    sw: torch.Tensor
    sn: torch.Tensor
    ss: torch.Tensor
    overlap_mask: torch.Tensor
    n: int
    s: int
    d: int
    k_1d: int
    K: int
    target_overlap: float
    build_local_solver: bool
    device: torch.device
    dtype: torch.dtype
    source_operator_type: str


def _as_local_spectral_asm_view(asm: Any) -> Any:
    """Return the canonical local-spectral view of a supported ASM operator."""

    canonical = (
        "a", "se", "sw", "sn", "ss", "overlap_mask", "n", "s", "d",
        "k_1d", "K", "target_overlap", "build_local_solver", "device", "dtype",
    )
    if all(hasattr(asm, name) for name in canonical):
        return asm

    cached = getattr(asm, "_modified_geneo_local_spectral_view", None)
    if cached is not None:
        return cached

    mobility = (
        "lam", "ae", "aw", "an", "as_", "diag", "overlap_mask", "nx", "ny",
        "sx", "sy", "starts_x", "starts_y", "kx_sub", "ky_sub", "K",
        "target_overlap", "build_local_solver", "device", "dtype",
    )
    missing = [name for name in mobility if not hasattr(asm, name)]
    if missing:
        raise TypeError(
            "modified GenEO requires CCFVM_ASM or MobilityFVMOperatorASM-compatible "
            f"data; missing attributes: {missing}"
        )
    if int(asm.nx) != int(asm.ny):
        raise NotImplementedError("modified GenEO currently requires a square mobility grid")
    if int(asm.sx) != int(asm.sy) or int(asm.kx_sub) != int(asm.ky_sub):
        raise NotImplementedError(
            "modified GenEO currently requires the same ASM patch geometry in x and y"
        )
    starts_x = [int(value) for value in asm.starts_x]
    starts_y = [int(value) for value in asm.starts_y]
    if starts_x != starts_y:
        raise NotImplementedError(
            "modified GenEO currently requires identical ASM starts in x and y"
        )
    if len(starts_x) != int(asm.kx_sub):
        raise RuntimeError("mobility ASM start count does not match kx_sub")
    if len(starts_x) == 1:
        stride = int(asm.sx)
        expected_starts = [0]
    else:
        stride = int(starts_x[1] - starts_x[0])
        expected_starts = [idx * stride for idx in range(len(starts_x))]
    if stride <= 0 or starts_x != expected_starts:
        raise NotImplementedError(
            "modified GenEO currently requires a uniform-stride mobility ASM partition"
        )

    # MobilityFVMOperatorASM stores positive neighbor coefficients and puts
    # west/east half-cell Dirichlet energy directly in ``diag``.  Recover those
    # two physical face values without changing the global operator.  The
    # north/south physical entries remain zero because those faces are Neumann.
    west = asm.aw.clone()
    east = asm.ae.clone()
    west[..., :, 0] = (
        asm.diag[..., :, 0]
        - asm.ae[..., :, 0]
        - asm.as_[..., :, 0]
        - asm.an[..., :, 0]
    )
    east[..., :, -1] = (
        asm.diag[..., :, -1]
        - asm.aw[..., :, -1]
        - asm.as_[..., :, -1]
        - asm.an[..., :, -1]
    )
    view = _LocalSpectralASMView(
        a=asm.lam,
        se=-east,
        sw=-west,
        # Array row zero is the north physical boundary.  For internal faces,
        # ``as_`` couples to row-1 and ``an`` couples to row+1.
        sn=-asm.as_,
        ss=-asm.an,
        overlap_mask=asm.overlap_mask,
        n=int(asm.nx),
        s=int(asm.sx),
        d=int(stride),
        k_1d=int(asm.kx_sub),
        K=int(asm.K),
        target_overlap=float(asm.target_overlap),
        build_local_solver=bool(asm.build_local_solver),
        device=asm.device,
        dtype=asm.dtype,
        source_operator_type=type(asm).__name__,
    )
    setattr(asm, "_modified_geneo_local_spectral_view", view)
    return view


def prepare_local_spectral_global_data(
    asm: Any,
    *,
    batch_index: int = 0,
    boundary_types: Optional[Mapping[str, str]] = None,
) -> LocalSpectralGlobalData:
    """Transfer the full-grid stencil and multiplicity exactly once per sample."""

    asm = _as_local_spectral_asm_view(asm)
    batch_size = int(asm.a.shape[0])
    if not (0 <= int(batch_index) < batch_size):
        raise IndexError(f"batch_index must lie in [0,{batch_size}), got {batch_index}")
    resolved_boundary = _normalize_boundary_types(boundary_types)
    source = asm.se[int(batch_index), 0]
    _sync_tensor_device(source)
    start = time.perf_counter()
    east = np.ascontiguousarray(-_copy_tensor_to_numpy(asm.se[int(batch_index), 0]))
    south = np.ascontiguousarray(-_copy_tensor_to_numpy(asm.ss[int(batch_index), 0]))
    west = np.ascontiguousarray(-_copy_tensor_to_numpy(asm.sw[int(batch_index), 0]))
    north = np.ascontiguousarray(-_copy_tensor_to_numpy(asm.sn[int(batch_index), 0]))
    multiplicity = np.ascontiguousarray(_copy_tensor_to_numpy(asm.overlap_mask[0, 0]))
    _sync_tensor_device(source)
    elapsed = time.perf_counter() - start
    expected_shape = (int(asm.n), int(asm.n))
    for name, value in {
        "east": east,
        "west": west,
        "north": north,
        "south": south,
        "multiplicity": multiplicity,
    }.items():
        if value.shape != expected_shape:
            raise RuntimeError(f"{name} has shape {value.shape}, expected {expected_shape}")
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"{name} contains non-finite values")
    if np.any(multiplicity <= 0.0):
        raise RuntimeError("ASM overlap multiplicity must be strictly positive")
    return LocalSpectralGlobalData(
        east=east,
        west=west,
        north=north,
        south=south,
        multiplicity=multiplicity,
        n=int(asm.n),
        boundary_types=resolved_boundary,
        dtype=str(source.dtype),
        source_device=str(source.device),
        transfer_time=float(elapsed),
        full_grid_cpu_transfer_count=1,
    )


def _patch_geometry(asm: Any, subdomain: int) -> Tuple[int, int, int, int, int, int]:
    asm = _as_local_spectral_asm_view(asm)
    if not bool(getattr(asm, "build_local_solver", False)):
        raise RuntimeError("modified GenEO requires the existing ASM partition")
    k_1d = int(asm.k_1d)
    K = int(asm.K)
    if not (0 <= int(subdomain) < K):
        raise IndexError(f"subdomain must lie in [0,{K}), got {subdomain}")
    patch_row, patch_col = divmod(int(subdomain), k_1d)
    row_start = patch_row * int(asm.d)
    col_start = patch_col * int(asm.d)
    row_stop = min(row_start + int(asm.s), int(asm.n))
    col_stop = min(col_start + int(asm.s), int(asm.n))
    height = max(0, row_stop - row_start)
    width = max(0, col_stop - col_start)
    return patch_row, patch_col, row_start, col_start, height, width


def _multiplicity_signature(multiplicity: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(multiplicity, dtype=np.float64)
    return hashlib.sha1(contiguous.view(np.uint8)).hexdigest()


def _topology_cache_key(
    asm: Any, global_data: LocalSpectralGlobalData
) -> Tuple[Any, ...]:
    asm = _as_local_spectral_asm_view(asm)
    return (
        "local_spectral_topology_v2",
        int(asm.n),
        int(asm.s),
        int(asm.d),
        int(asm.k_1d),
        int(asm.K),
        tuple((side, global_data.boundary_types[side]) for side in _SIDES),
        _multiplicity_signature(global_data.multiplicity),
    )


def _physical_face_arrays(
    *, row_start: int, col_start: int, height: int, width: int, n: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    local: Dict[str, np.ndarray] = {side: np.zeros(0, dtype=np.int64) for side in _SIDES}
    rows: Dict[str, np.ndarray] = {side: np.zeros(0, dtype=np.int64) for side in _SIDES}
    cols: Dict[str, np.ndarray] = {side: np.zeros(0, dtype=np.int64) for side in _SIDES}
    if col_start == 0:
        local["west"] = np.arange(0, height * width, width, dtype=np.int64)
        rows["west"] = np.arange(row_start, row_start + height, dtype=np.int64)
        cols["west"] = np.zeros(height, dtype=np.int64)
    if col_start + width == n:
        local["east"] = np.arange(width - 1, height * width, width, dtype=np.int64)
        rows["east"] = np.arange(row_start, row_start + height, dtype=np.int64)
        cols["east"] = np.full(height, n - 1, dtype=np.int64)
    if row_start == 0:
        local["north"] = np.arange(width, dtype=np.int64)
        rows["north"] = np.zeros(width, dtype=np.int64)
        cols["north"] = np.arange(col_start, col_start + width, dtype=np.int64)
    if row_start + height == n:
        local["south"] = np.arange((height - 1) * width, height * width, dtype=np.int64)
        rows["south"] = np.full(width, n - 1, dtype=np.int64)
        cols["south"] = np.arange(col_start, col_start + width, dtype=np.int64)
    return local, rows, cols


def _build_patch_topology(
    asm: Any,
    subdomain: int,
    global_data: LocalSpectralGlobalData,
) -> LocalSubdomainTopology:
    asm = _as_local_spectral_asm_view(asm)
    patch_row, patch_col, row_start, col_start, height, width = _patch_geometry(
        asm, subdomain
    )
    if height <= 0 or width <= 0:
        raise ValueError(f"subdomain {subdomain} is a padding-only slot")
    nloc = int(height * width)
    ids = np.arange(nloc, dtype=np.int64).reshape(height, width)
    east_i = ids[:, :-1].reshape(-1)
    east_j = ids[:, 1:].reshape(-1)
    south_i = ids[:-1, :].reshape(-1)
    south_j = ids[1:, :].reshape(-1)
    internal_i = np.concatenate((east_i, south_i)).astype(np.int64, copy=False)
    internal_j = np.concatenate((east_j, south_j)).astype(np.int64, copy=False)
    internal_axis = np.concatenate(
        (np.zeros(east_i.size, dtype=np.int8), np.ones(south_i.size, dtype=np.int8))
    )
    east_gr = np.repeat(np.arange(row_start, row_start + height), max(width - 1, 0))
    east_gc = np.tile(np.arange(col_start, col_start + max(width - 1, 0)), height)
    south_gr = np.repeat(np.arange(row_start, row_start + max(height - 1, 0)), width)
    south_gc = np.tile(np.arange(col_start, col_start + width), max(height - 1, 0))
    internal_global_rows = np.concatenate((east_gr, south_gr)).astype(np.int64, copy=False)
    internal_global_cols = np.concatenate((east_gc, south_gc)).astype(np.int64, copy=False)
    diagonal_ids = np.arange(nloc, dtype=np.int64)
    coo_rows = np.concatenate((internal_i, internal_j, diagonal_ids))
    coo_cols = np.concatenate((internal_j, internal_i, diagonal_ids))
    global_rows = np.repeat(np.arange(row_start, row_start + height), width)
    global_cols = np.tile(np.arange(col_start, col_start + width), height)
    global_indices = (global_rows * int(asm.n) + global_cols).astype(np.int64, copy=False)
    pou_weights = np.asarray(
        1.0 / global_data.multiplicity[global_rows, global_cols], dtype=np.float64
    )
    if not np.all(np.isfinite(pou_weights)) or np.any(pou_weights <= 0.0):
        raise RuntimeError(f"invalid ASM partition-of-unity weights in patch {subdomain}")
    physical_local, physical_rows, physical_cols = _physical_face_arrays(
        row_start=row_start,
        col_start=col_start,
        height=height,
        width=width,
        n=int(asm.n),
    )
    if nloc <= 1:
        component_count = nloc
    else:
        adjacency = sp.coo_matrix(
            (
                np.ones(2 * internal_i.size, dtype=np.int8),
                (
                    np.concatenate((internal_i, internal_j)),
                    np.concatenate((internal_j, internal_i)),
                ),
            ),
            shape=(nloc, nloc),
        ).tocsr()
        component_count = int(csgraph.connected_components(adjacency, directed=False)[0])
    return LocalSubdomainTopology(
        subdomain=int(subdomain),
        patch_row=int(patch_row),
        patch_col=int(patch_col),
        row_start=int(row_start),
        col_start=int(col_start),
        height=int(height),
        width=int(width),
        global_indices=global_indices,
        pou_weights=pou_weights,
        internal_i=internal_i,
        internal_j=internal_j,
        internal_axis=internal_axis,
        internal_global_rows=internal_global_rows,
        internal_global_cols=internal_global_cols,
        coo_rows=coo_rows,
        coo_cols=coo_cols,
        physical_local_indices=physical_local,
        physical_global_rows=physical_rows,
        physical_global_cols=physical_cols,
        touches_physical_boundary={side: bool(physical_local[side].size) for side in _SIDES},
        connected_components=component_count,
    )


def get_local_spectral_topology_cache(
    asm: Any,
    global_data: LocalSpectralGlobalData,
) -> Tuple[LocalSpectralTopologyCache, bool, float]:
    """Look up or build coefficient-independent patch topology."""

    asm = _as_local_spectral_asm_view(asm)
    lookup_start = time.perf_counter()
    key = _topology_cache_key(asm, global_data)
    cached = _TOPOLOGY_CACHE.get(key)
    lookup_time = time.perf_counter() - lookup_start
    if cached is not None:
        return cached, True, float(lookup_time)
    build_start = time.perf_counter()
    patches: Dict[int, LocalSubdomainTopology] = {}
    padding: List[int] = []
    for subdomain in range(int(asm.K)):
        geometry = _patch_geometry(asm, subdomain)
        if geometry[4] <= 0 or geometry[5] <= 0:
            padding.append(int(subdomain))
            continue
        patches[int(subdomain)] = _build_patch_topology(asm, subdomain, global_data)
    built = LocalSpectralTopologyCache(
        key=key,
        patches=patches,
        active_subdomain_ids=tuple(sorted(patches)),
        padding_only_slots=tuple(padding),
        build_time=float(time.perf_counter() - build_start),
    )
    _TOPOLOGY_CACHE[key] = built
    return built, False, float(lookup_time)


def _face_values(
    side: str,
    global_data: LocalSpectralGlobalData,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    return np.asarray(getattr(global_data, side)[rows, cols], dtype=np.float64)


def assemble_local_neumann_matrix(
    asm: Any,
    subdomain: int,
    *,
    batch_index: int = 0,
    boundary_types: Optional[Mapping[str, str]] = None,
    global_data: Optional[LocalSpectralGlobalData] = None,
    topology_cache: Optional[LocalSpectralTopologyCache] = None,
) -> LocalSubdomain:
    """Assemble one face-based local natural-boundary FVM pencil.

    Artificial patch-boundary faces never contribute.  Physical Dirichlet
    faces retain their half-cell energy; physical Neumann faces contribute no
    Dirichlet diagonal.  The high-level provider supports the validated static
    all-Dirichlet and moving-interface west/east-Dirichlet, north/south-Neumann
    configurations.
    """

    asm = _as_local_spectral_asm_view(asm)
    if global_data is None:
        global_data = prepare_local_spectral_global_data(
            asm, batch_index=batch_index, boundary_types=boundary_types
        )
    else:
        resolved = _normalize_boundary_types(boundary_types or global_data.boundary_types)
        if resolved != global_data.boundary_types:
            raise ValueError("boundary_types do not match LocalSpectralGlobalData")
    if topology_cache is None:
        topology_cache, _hit, _lookup = get_local_spectral_topology_cache(asm, global_data)
    topology = topology_cache.patches.get(int(subdomain))
    if topology is None:
        geometry = _patch_geometry(asm, int(subdomain))
        raise ValueError(
            f"ASM patch {subdomain} is a padding-only slot with no physical cells: "
            f"start=({geometry[2]},{geometry[3]}), n={asm.n}, s={asm.s}"
        )

    start = time.perf_counter()
    nloc = int(topology.height * topology.width)
    internal_values = np.empty(topology.internal_i.size, dtype=np.float64)
    east_mask = topology.internal_axis == 0
    south_mask = ~east_mask
    if np.any(east_mask):
        internal_values[east_mask] = global_data.east[
            topology.internal_global_rows[east_mask], topology.internal_global_cols[east_mask]
        ]
    if np.any(south_mask):
        internal_values[south_mask] = global_data.south[
            topology.internal_global_rows[south_mask], topology.internal_global_cols[south_mask]
        ]
    if not np.all(np.isfinite(internal_values)) or np.any(internal_values <= 0.0):
        bad = internal_values[~np.isfinite(internal_values) | (internal_values <= 0.0)]
        raise RuntimeError(
            f"non-positive local transmissibility in subdomain {subdomain}: "
            f"{bad[:4].tolist()}"
        )
    diagonal = np.bincount(
        np.concatenate((topology.internal_i, topology.internal_j)),
        weights=np.concatenate((internal_values, internal_values)),
        minlength=nloc,
    ).astype(np.float64, copy=False)
    physical_dirichlet_faces = 0
    physical_neumann_faces = 0
    for side in _SIDES:
        local_ids = topology.physical_local_indices[side]
        if local_ids.size == 0:
            continue
        if global_data.boundary_types[side] == "dirichlet":
            values = _face_values(
                side,
                global_data,
                topology.physical_global_rows[side],
                topology.physical_global_cols[side],
            )
            if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise RuntimeError(
                    f"invalid physical {side} Dirichlet transmissibility in patch {subdomain}"
                )
            np.add.at(diagonal, local_ids, values)
            physical_dirichlet_faces += int(local_ids.size)
        else:
            physical_neumann_faces += int(local_ids.size)
    coo_data = np.concatenate((-internal_values, -internal_values, diagonal))
    A_neumann = sp.coo_matrix(
        (coo_data, (topology.coo_rows, topology.coo_cols)),
        shape=(nloc, nloc),
        dtype=np.float64,
    ).tocsr()
    A_neumann.sum_duplicates()
    D = sp.diags(topology.pou_weights, format="csr")
    B_pou_energy = (D @ A_neumann @ D).tocsr()
    ones = np.ones(nloc, dtype=np.float64)
    norm_A = max(float(spla.norm(A_neumann, ord=np.inf)), np.finfo(np.float64).tiny)
    norm_B = max(float(spla.norm(B_pou_energy, ord=np.inf)), np.finfo(np.float64).tiny)
    symmetry_A = float(spla.norm(A_neumann - A_neumann.T, ord=np.inf) / norm_A)
    symmetry_B = float(spla.norm(B_pou_energy - B_pou_energy.T, ord=np.inf) / norm_B)
    row_sum_error = float(np.linalg.norm(A_neumann @ ones))
    null_tol = 128.0 * np.finfo(np.float64).eps * max(nloc, 1) * norm_A * np.linalg.norm(ones)
    has_neumann_nullspace = bool(
        physical_dirichlet_faces == 0 and row_sum_error <= null_tol
    )
    return LocalSubdomain(
        subdomain=int(subdomain),
        patch_row=int(topology.patch_row),
        patch_col=int(topology.patch_col),
        row_start=int(topology.row_start),
        col_start=int(topology.col_start),
        height=int(topology.height),
        width=int(topology.width),
        global_indices=topology.global_indices,
        pou_weights=topology.pou_weights,
        A_neumann=A_neumann,
        B_pou_energy=B_pou_energy,
        physical_boundary_faces=int(physical_dirichlet_faces),
        physical_neumann_faces=int(physical_neumann_faces),
        has_neumann_nullspace=has_neumann_nullspace,
        boundary_types=dict(global_data.boundary_types),
        touches_physical_boundary=dict(topology.touches_physical_boundary),
        connected_components=int(topology.connected_components),
        face_transmissibility_min=float(np.min(internal_values)) if internal_values.size else float("nan"),
        face_transmissibility_median=float(np.median(internal_values)) if internal_values.size else float("nan"),
        face_transmissibility_max=float(np.max(internal_values)) if internal_values.size else float("nan"),
        A_norm_inf=float(norm_A),
        B_norm_inf=float(norm_B),
        A_symmetry_error=symmetry_A,
        B_symmetry_error=symmetry_B,
        numeric_assembly_time=float(time.perf_counter() - start),
    )


def _constraint_basis(constraint: np.ndarray) -> sp.csr_matrix:
    """Return a sparse basis ``Z`` satisfying ``constraint.T @ Z == 0``."""

    constraint = np.asarray(constraint, dtype=np.float64).reshape(-1)
    n = int(constraint.size)
    if n <= 1:
        return sp.csr_matrix((n, 0), dtype=np.float64)
    pivot = int(np.argmax(np.abs(constraint)))
    pivot_value = float(constraint[pivot])
    scale = max(float(np.linalg.norm(constraint, ord=np.inf)), np.finfo(np.float64).tiny)
    if abs(pivot_value) <= 64.0 * np.finfo(np.float64).eps * scale:
        raise RuntimeError("cannot build a constraint basis from a zero vector")
    nonpivot = np.asarray([i for i in range(n) if i != pivot], dtype=np.int64)
    columns = np.arange(n - 1, dtype=np.int64)
    row = np.concatenate((nonpivot, np.full(n - 1, pivot, dtype=np.int64)))
    col = np.concatenate((columns, columns))
    values = np.concatenate(
        (np.ones(n - 1, dtype=np.float64), -constraint[nonpivot] / pivot_value)
    )
    return sp.coo_matrix((values, (row, col)), shape=(n, n - 1)).tocsr()


def _scaled_action_error(
    matrix: sp.csr_matrix,
    vector: np.ndarray,
    matrix_norm: Optional[float] = None,
) -> float:
    vector = np.asarray(vector, dtype=np.float64)
    action = np.asarray(matrix @ vector, dtype=np.float64)
    norm = float(matrix_norm) if matrix_norm is not None else float(spla.norm(matrix, ord=np.inf))
    scale = norm * float(np.linalg.norm(vector))
    eps_scale = np.finfo(np.float64).eps * max(scale, np.finfo(np.float64).tiny)
    return float(np.linalg.norm(action) / max(scale + eps_scale, np.finfo(np.float64).tiny))


def _detect_nullspace_structure(local: LocalSubdomain) -> Dict[str, Any]:
    nloc = int(local.A_neumann.shape[0])
    ones = np.ones(nloc, dtype=np.float64)
    inverse_D_ones = 1.0 / np.asarray(local.pou_weights, dtype=np.float64)
    d_mean = float(np.mean(local.pou_weights))
    d_constant_error = float(
        np.max(np.abs(local.pou_weights - d_mean))
        / max(abs(d_mean), np.finfo(np.float64).tiny)
    )
    a_ones_error = _scaled_action_error(local.A_neumann, ones, local.A_norm_inf)
    b_ones_error = _scaled_action_error(local.B_pou_energy, ones, local.B_norm_inf)
    a_bnull_error = _scaled_action_error(
        local.A_neumann, inverse_D_ones, local.A_norm_inf
    )
    b_bnull_error = _scaled_action_error(
        local.B_pou_energy, inverse_D_ones, local.B_norm_inf
    )
    tolerance = 512.0 * np.finfo(np.float64).eps * max(nloc, 1)
    common = bool(
        local.has_neumann_nullspace
        and d_constant_error <= tolerance
        and a_ones_error <= tolerance
        and b_ones_error <= tolerance
        and a_bnull_error <= tolerance
        and b_bnull_error <= tolerance
    )
    return {
        "common_nullspace": common,
        "D_constant_error": d_constant_error,
        "A_ones_relative_error": a_ones_error,
        "B_ones_relative_error": b_ones_error,
        "A_Bnull_relative_error": a_bnull_error,
        "B_Bnull_relative_error": b_bnull_error,
        "nullspace_detection_tolerance": tolerance,
    }


def _generalized_largest_modes(
    numerator: sp.csr_matrix,
    metric: sp.csr_matrix,
    *,
    k: int,
    dense_cutoff: int,
    eig_tol: float,
    maxiter: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Solve ``numerator x = mu metric x`` for the largest reciprocal modes."""

    dimension = int(numerator.shape[0])
    started = time.perf_counter()
    if dimension == 0 or k <= 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros((dimension, 0), dtype=np.float64),
            {
                "eigensolver": "none",
                "eigensolver_status": "converged",
                "eigensolver_message": "no reciprocal modes requested",
                "requested_reciprocal_modes": int(max(k, 0)),
                "returned_raw_reciprocal_modes": 0,
                "local_eigensolve_time": float(time.perf_counter() - started),
            },
        )
    k = min(int(k), dimension)
    equilibrated = False
    equilibration_min = 1.0
    equilibration_max = 1.0
    if dimension <= int(dense_cutoff) or k >= dimension:
        eigenvalues, eigenvectors = la.eigh(
            numerator.toarray(), metric.toarray(), check_finite=True, driver="gvd"
        )
        order = np.argsort(eigenvalues)[::-1][:k]
        values = np.asarray(eigenvalues[order], dtype=np.float64)
        vectors = np.asarray(eigenvectors[:, order], dtype=np.float64)
        status = "converged"
        solver = "dense_eigh"
        message = "full dense generalized eigensolve"
    else:
        metric_diagonal = np.asarray(metric.diagonal(), dtype=np.float64)
        if (
            metric_diagonal.size != dimension
            or not np.all(np.isfinite(metric_diagonal))
            or np.any(metric_diagonal <= 0.0)
        ):
            raise RuntimeError(
                "the reduced reciprocal GenEO metric must have a finite positive diagonal"
            )
        diagonal_reference = float(np.median(metric_diagonal))
        diagonal_reference = max(diagonal_reference, np.finfo(np.float64).tiny)
        equilibration = np.sqrt(diagonal_reference / metric_diagonal)
        equilibration_min = float(np.min(equilibration))
        equilibration_max = float(np.max(equilibration))
        scaling = sp.diags(equilibration, format="csr")
        numerator_solve = (scaling @ numerator @ scaling).tocsr()
        metric_solve = (scaling @ metric @ scaling).tocsr()
        equilibrated = True
        rng = np.random.default_rng(int(seed))
        v0 = rng.standard_normal(dimension)
        ncv = min(dimension, max(8 * int(k) + 20, 64))
        try:
            values, vectors = spla.eigsh(
                numerator_solve,
                M=metric_solve,
                k=int(k),
                ncv=int(ncv),
                which="LA",
                v0=v0,
                tol=float(eig_tol),
                maxiter=(int(maxiter) if maxiter is not None else max(2000, 12 * dimension)),
            )
            status = "converged"
            solver = "eigsh_largest_reciprocal"
            message = "ARPACK converged"
        except spla.ArpackNoConvergence as exc:
            values = (
                np.asarray(exc.eigenvalues, dtype=np.float64)
                if exc.eigenvalues is not None
                else np.zeros(0, dtype=np.float64)
            )
            vectors = (
                np.asarray(exc.eigenvectors, dtype=np.float64)
                if exc.eigenvectors is not None
                else np.zeros((dimension, 0), dtype=np.float64)
            )
            status = "partial"
            solver = "eigsh_partial_reciprocal"
            message = f"ARPACK returned {values.size}/{k} requested reciprocal modes"
        order = np.argsort(values)[::-1]
        values = np.asarray(values[order], dtype=np.float64)
        vectors = np.asarray(vectors[:, order], dtype=np.float64)
        vectors = np.asarray(equilibration[:, None] * vectors, dtype=np.float64)
    return values, vectors, {
        "eigensolver": solver,
        "eigensolver_status": status,
        "eigensolver_message": message,
        "requested_reciprocal_modes": int(k),
        "arpack_ncv": int(ncv) if equilibrated else None,
        "returned_raw_reciprocal_modes": int(values.size),
        "diagonal_congruence_equilibrated": bool(equilibrated),
        "equilibration_scale_min": float(equilibration_min),
        "equilibration_scale_max": float(equilibration_max),
        "local_eigensolve_time": float(time.perf_counter() - started),
    }


def _generalized_smallest_modes(
    numerator: sp.csr_matrix,
    metric: sp.csr_matrix,
    *,
    k: int,
    dense_cutoff: int,
    eig_tol: float,
    maxiter: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Solve ``numerator x = lambda metric x`` for its smallest modes.

    Singular directions, when present, must be removed by congruence before
    this helper is called, so both input matrices are positive definite.  The
    returned values are the thesis-convention ``lambda`` values, not
    reciprocals.
    """

    dimension = int(numerator.shape[0])
    started = time.perf_counter()
    if dimension == 0 or k <= 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros((dimension, 0), dtype=np.float64),
            {
                "eigensolver": "none",
                "eigensolver_status": "converged",
                "eigensolver_message": "no direct finite modes requested",
                "requested_direct_modes": int(max(k, 0)),
                "returned_raw_direct_modes": 0,
                "local_eigensolve_time": float(time.perf_counter() - started),
            },
        )
    k = min(int(k), dimension)
    equilibrated = False
    equilibration_min = 1.0
    equilibration_max = 1.0
    if dimension <= int(dense_cutoff) or k >= dimension:
        eigenvalues, eigenvectors = la.eigh(
            numerator.toarray(), metric.toarray(), check_finite=True, driver="gvd"
        )
        order = np.argsort(eigenvalues)[:k]
        values = np.asarray(eigenvalues[order], dtype=np.float64)
        vectors = np.asarray(eigenvectors[:, order], dtype=np.float64)
        status = "converged"
        solver = "dense_eigh_smallest_direct"
        message = "full dense generalized eigensolve for smallest finite lambdas"
        ncv: Optional[int] = None
    else:
        numerator_diagonal = np.asarray(numerator.diagonal(), dtype=np.float64)
        if (
            numerator_diagonal.size != dimension
            or not np.all(np.isfinite(numerator_diagonal))
            or np.any(numerator_diagonal <= 0.0)
        ):
            raise RuntimeError(
                "the reduced direct GenEO numerator must have a finite positive diagonal"
            )
        diagonal_reference = max(
            float(np.median(numerator_diagonal)), np.finfo(np.float64).tiny
        )
        equilibration = np.sqrt(diagonal_reference / numerator_diagonal)
        equilibration_min = float(np.min(equilibration))
        equilibration_max = float(np.max(equilibration))
        scaling = sp.diags(equilibration, format="csr")
        numerator_solve = (scaling @ numerator @ scaling).tocsr()
        metric_solve = (scaling @ metric @ scaling).tocsr()
        equilibrated = True
        rng = np.random.default_rng(int(seed))
        v0 = rng.standard_normal(dimension)
        ncv = min(dimension, max(8 * int(k) + 20, 64))
        try:
            values, vectors = spla.eigsh(
                numerator_solve,
                M=metric_solve,
                k=int(k),
                sigma=0.0,
                which="LM",
                v0=v0,
                ncv=int(ncv),
                tol=float(eig_tol),
                maxiter=(
                    int(maxiter) if maxiter is not None else max(2000, 12 * dimension)
                ),
            )
            status = "converged"
            solver = "eigsh_smallest_direct"
            message = "shift-invert ARPACK converged on the finite direct pencil"
        except spla.ArpackNoConvergence as exc:
            values = (
                np.asarray(exc.eigenvalues, dtype=np.float64)
                if exc.eigenvalues is not None
                else np.zeros(0, dtype=np.float64)
            )
            vectors = (
                np.asarray(exc.eigenvectors, dtype=np.float64)
                if exc.eigenvectors is not None
                else np.zeros((dimension, 0), dtype=np.float64)
            )
            status = "partial"
            solver = "eigsh_partial_smallest_direct"
            message = f"ARPACK returned {values.size}/{k} requested direct modes"
        order = np.argsort(values)
        values = np.asarray(values[order], dtype=np.float64)
        vectors = np.asarray(vectors[:, order], dtype=np.float64)
        vectors = np.asarray(equilibration[:, None] * vectors, dtype=np.float64)
    return values, vectors, {
        "eigensolver": solver,
        "eigensolver_status": status,
        "eigensolver_message": message,
        "requested_direct_modes": int(k),
        "arpack_ncv": int(ncv) if ncv is not None else None,
        "returned_raw_direct_modes": int(values.size),
        "diagonal_congruence_equilibrated": bool(equilibrated),
        "equilibration_scale_min": float(equilibration_min),
        "equilibration_scale_max": float(equilibration_max),
        "local_eigensolve_time": float(time.perf_counter() - started),
    }


def _apply_compact_q(
    qr_raw: np.ndarray,
    tau: np.ndarray,
    value: np.ndarray,
    *,
    transpose: bool,
) -> np.ndarray:
    """Apply the compact Householder Q from a raw LAPACK QR factorization."""

    output = np.asarray(value, dtype=np.float64).copy()
    squeeze = output.ndim == 1
    if squeeze:
        output = output[:, None]
    order = range(tau.size) if transpose else range(tau.size - 1, -1, -1)
    for column in order:
        tail = np.asarray(qr_raw[column + 1 :, column], dtype=np.float64)
        projection = output[column, :].copy()
        if tail.size:
            projection += tail @ output[column + 1 :, :]
        projection *= float(tau[column])
        output[column, :] -= projection
        if tail.size:
            output[column + 1 :, :] -= tail[:, None] * projection[None, :]
    return output[:, 0] if squeeze else output


def _generalized_smallest_constrained_modes(
    numerator: sp.csr_matrix,
    metric: sp.csr_matrix,
    constraints: np.ndarray,
    *,
    k: int,
    eig_tol: float,
    maxiter: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Smallest finite modes using an implicit constrained nullspace basis.

    The sparse KKT factorization avoids explicitly forming ``T.T @ A @ T``;
    pivot-elimination bases can introduce severe fill for large Neumann
    patches.  Householder reflectors provide an orthonormal implicit basis for
    the same constraint complement, so this is algebraically the same reduced
    direct pencil without a diagonal shift.
    """

    started = time.perf_counter()
    dimension = int(numerator.shape[0])
    constraints = np.asarray(constraints, dtype=np.float64)
    if constraints.ndim == 1:
        constraints = constraints[:, None]
    if constraints.shape[0] != dimension or constraints.shape[1] <= 0:
        raise ValueError("constraints must have shape (dimension, positive_count)")
    constraint_count = int(constraints.shape[1])
    reduced_dimension = dimension - constraint_count
    if reduced_dimension <= 0 or k <= 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros((dimension, 0), dtype=np.float64),
            {
                "eigensolver": "none",
                "eigensolver_status": "converged",
                "eigensolver_message": "no constrained direct modes requested",
                "requested_direct_modes": int(max(k, 0)),
                "returned_raw_direct_modes": 0,
                "constraint_count": constraint_count,
                "implicit_constraint_nullspace": True,
                "local_eigensolve_time": float(time.perf_counter() - started),
            },
        )
    k = min(int(k), reduced_dimension)

    numerator_diagonal = np.asarray(numerator.diagonal(), dtype=np.float64)
    if (
        numerator_diagonal.size != dimension
        or not np.all(np.isfinite(numerator_diagonal))
        or np.any(numerator_diagonal <= 0.0)
    ):
        raise RuntimeError(
            "the constrained direct GenEO numerator must have a finite positive diagonal"
        )
    diagonal_reference = max(
        float(np.median(numerator_diagonal)), np.finfo(np.float64).tiny
    )
    equilibration = np.sqrt(diagonal_reference / numerator_diagonal)
    scaling = sp.diags(equilibration, format="csr")
    numerator_solve = (scaling @ numerator @ scaling).tocsr()
    metric_solve = (scaling @ metric @ scaling).tocsr()
    scaled_constraints = equilibration[:, None] * constraints
    constraint_q, constraint_r = la.qr(
        scaled_constraints, mode="economic", check_finite=True
    )
    constraint_scale = max(
        float(np.linalg.norm(constraint_r, ord=np.inf)), np.finfo(np.float64).tiny
    )
    constraint_rank_floor = (
        128.0 * np.finfo(np.float64).eps * max(dimension, 1) * constraint_scale
    )
    if np.any(np.abs(np.diag(constraint_r)) <= constraint_rank_floor):
        raise RuntimeError("the GenEO finite-mode constraints are numerically dependent")
    (qr_raw, tau), _r = la.qr(constraint_q, mode="raw", check_finite=True)

    def expand(reduced: np.ndarray) -> np.ndarray:
        reduced = np.asarray(reduced, dtype=np.float64)
        full = np.zeros(dimension, dtype=np.float64)
        full[constraint_count:] = reduced
        return _apply_compact_q(qr_raw, tau, full, transpose=False)

    def reduce(full: np.ndarray) -> np.ndarray:
        rotated = _apply_compact_q(qr_raw, tau, full, transpose=True)
        return np.asarray(rotated[constraint_count:], dtype=np.float64)

    numerator_operator = spla.LinearOperator(
        (reduced_dimension, reduced_dimension),
        matvec=lambda value: reduce(numerator_solve @ expand(value)),
        dtype=np.float64,
    )
    metric_operator = spla.LinearOperator(
        (reduced_dimension, reduced_dimension),
        matvec=lambda value: reduce(metric_solve @ expand(value)),
        dtype=np.float64,
    )
    kkt = sp.bmat(
        [
            [numerator_solve.tocsc(), sp.csc_matrix(constraint_q)],
            [sp.csc_matrix(constraint_q.T), sp.csc_matrix((constraint_count, constraint_count))],
        ],
        format="csc",
    )
    factor_started = time.perf_counter()
    try:
        kkt_factor = spla.splu(kkt, permc_spec="COLAMD")
    except (RuntimeError, SystemError, MemoryError) as exc:
        raise RuntimeError(
            "constrained GenEO KKT factorization failed without forming T^T A T"
        ) from exc
    factor_time = time.perf_counter() - factor_started

    def inverse_action(reduced_rhs: np.ndarray) -> np.ndarray:
        full_rhs = expand(reduced_rhs)
        augmented_rhs = np.concatenate(
            (full_rhs, np.zeros(constraint_count, dtype=np.float64))
        )
        solution = np.asarray(kkt_factor.solve(augmented_rhs), dtype=np.float64)
        return reduce(solution[:dimension])

    inverse_operator = spla.LinearOperator(
        (reduced_dimension, reduced_dimension),
        matvec=inverse_action,
        dtype=np.float64,
    )
    rng = np.random.default_rng(int(seed))
    v0 = rng.standard_normal(reduced_dimension)
    ncv = min(reduced_dimension, max(8 * int(k) + 20, 64))
    try:
        values, reduced_vectors = spla.eigsh(
            numerator_operator,
            M=metric_operator,
            k=int(k),
            sigma=0.0,
            which="LM",
            OPinv=inverse_operator,
            v0=v0,
            ncv=int(ncv),
            tol=float(eig_tol),
            maxiter=(
                int(maxiter)
                if maxiter is not None
                else max(2000, 12 * reduced_dimension)
            ),
        )
        status = "converged"
        solver = "eigsh_smallest_direct_implicit_constraints"
        message = "shift-invert ARPACK converged with a constrained KKT inverse"
    except spla.ArpackNoConvergence as exc:
        values = (
            np.asarray(exc.eigenvalues, dtype=np.float64)
            if exc.eigenvalues is not None
            else np.zeros(0, dtype=np.float64)
        )
        reduced_vectors = (
            np.asarray(exc.eigenvectors, dtype=np.float64)
            if exc.eigenvectors is not None
            else np.zeros((reduced_dimension, 0), dtype=np.float64)
        )
        status = "partial"
        solver = "eigsh_partial_smallest_direct_implicit_constraints"
        message = f"ARPACK returned {values.size}/{k} constrained direct modes"
    order = np.argsort(values)
    values = np.asarray(values[order], dtype=np.float64)
    reduced_vectors = np.asarray(reduced_vectors[:, order], dtype=np.float64)
    scaled_full_vectors = np.column_stack(
        [expand(reduced_vectors[:, column]) for column in range(values.size)]
    ) if values.size else np.zeros((dimension, 0), dtype=np.float64)
    vectors = equilibration[:, None] * scaled_full_vectors
    return values, vectors, {
        "eigensolver": solver,
        "eigensolver_status": status,
        "eigensolver_message": message,
        "requested_direct_modes": int(k),
        "arpack_ncv": int(ncv),
        "returned_raw_direct_modes": int(values.size),
        "diagonal_congruence_equilibrated": True,
        "equilibration_scale_min": float(np.min(equilibration)),
        "equilibration_scale_max": float(np.max(equilibration)),
        "constraint_count": constraint_count,
        "implicit_constraint_nullspace": True,
        "constraint_rank_floor": float(constraint_rank_floor),
        "kkt_nnz": int(kkt.nnz),
        "kkt_L_nnz": int(kkt_factor.L.nnz),
        "kkt_U_nnz": int(kkt_factor.U.nnz),
        "kkt_factorization_time": float(factor_time),
        "local_eigensolve_time": float(time.perf_counter() - started),
    }


def _reciprocal_pair_residual(
    numerator: sp.csr_matrix,
    metric: sp.csr_matrix,
    vector: np.ndarray,
    reciprocal: float,
    *,
    numerator_norm: Optional[float] = None,
    metric_norm: Optional[float] = None,
) -> float:
    vector = np.asarray(vector, dtype=np.float64)
    Nv = np.asarray(numerator @ vector, dtype=np.float64)
    Mv = np.asarray(metric @ vector, dtype=np.float64)
    raw = float(np.linalg.norm(Nv - reciprocal * Mv))
    scale = float(np.linalg.norm(Nv)) + abs(float(reciprocal)) * float(np.linalg.norm(Mv))
    op_scale = (
        (
            float(numerator_norm)
            if numerator_norm is not None
            else float(spla.norm(numerator, ord=np.inf))
        )
        + abs(float(reciprocal))
        * (
            float(metric_norm)
            if metric_norm is not None
            else float(spla.norm(metric, ord=np.inf))
        )
    ) * float(np.linalg.norm(vector))
    eps = np.finfo(np.float64).eps * max(scale, op_scale, np.finfo(np.float64).tiny)
    return float(raw / max(scale + eps, np.finfo(np.float64).tiny))


def _mode_diagnostics(
    local: LocalSubdomain,
    vector: np.ndarray,
    eigenvalue: float,
    *,
    is_zero_mode: bool,
) -> Dict[str, Any]:
    p = np.asarray(vector, dtype=np.float64)
    Ap = np.asarray(local.A_neumann @ p, dtype=np.float64)
    Bp = np.asarray(local.B_pou_energy @ p, dtype=np.float64)
    numerator = float(p @ Ap)
    denominator = float(p @ Bp)
    norm_A = float(local.A_norm_inf)
    norm_B = float(local.B_norm_inf)
    lambda_operator_scale = norm_A / norm_B
    lambda_scale_floor = (
        64.0
        * np.finfo(np.float64).eps
        * max(lambda_operator_scale, abs(float(eigenvalue)), np.finfo(np.float64).tiny)
    )
    denominator_floor = (
        64.0
        * np.finfo(np.float64).eps
        * max(norm_B * float(np.linalg.norm(p)) ** 2, abs(numerator), np.finfo(np.float64).tiny)
    )
    rayleigh = numerator / denominator if abs(denominator) > denominator_floor else float("nan")
    residual_vector = Ap - float(eigenvalue) * Bp
    direct_scale = float(np.linalg.norm(Ap)) + abs(float(eigenvalue)) * float(np.linalg.norm(Bp))
    operator_vector_scale = (
        norm_A + abs(float(eigenvalue)) * norm_B
    ) * float(np.linalg.norm(p))
    eps = np.finfo(np.float64).eps * max(
        direct_scale, operator_vector_scale, np.finfo(np.float64).tiny
    )
    eig_residual = float(
        np.linalg.norm(residual_vector)
        / max(direct_scale + eps, np.finfo(np.float64).tiny)
    )
    if is_zero_mode:
        relative_rq_error: Optional[float] = None
        zero_error: Optional[float] = _scaled_action_error(
            local.A_neumann, p, local.A_norm_inf
        )
    else:
        relative_rq_error = (
            float(abs(rayleigh - float(eigenvalue)) / max(abs(float(eigenvalue)), lambda_scale_floor))
            if math.isfinite(rayleigh)
            else float("inf")
        )
        zero_error = None
    return {
        "rayleigh_quotient": float(rayleigh),
        "eig_residual": eig_residual,
        "relative_rq_error": relative_rq_error,
        "lambda_scale_floor": float(lambda_scale_floor),
        "zero_mode_nullspace_error": zero_error,
        "direct_residual_epsilon": float(eps),
    }


def _normalize_local_mode(local: LocalSubdomain, vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    energy = float(vector @ (local.B_pou_energy @ vector))
    energy_floor = (
        64.0
        * np.finfo(np.float64).eps
        * max(
            float(local.B_norm_inf) * float(np.linalg.norm(vector)) ** 2,
            np.finfo(np.float64).tiny,
        )
    )
    if math.isfinite(energy) and energy > energy_floor:
        vector = vector / math.sqrt(energy)
    else:
        norm = float(np.linalg.norm(vector))
        if norm <= np.finfo(np.float64).tiny:
            raise RuntimeError("local eigensolver returned a zero eigenvector")
        vector = vector / norm
    pivot = int(np.argmax(np.abs(vector)))
    if vector[pivot] < 0.0:
        vector = -vector
    return vector


def solve_modified_geneo_local_spectrum(
    local: LocalSubdomain,
    *,
    num_modes: int,
    dense_cutoff: int = 512,
    eig_tol: float = 1e-9,
    eigen_residual_tol: float = 1e-6,
    maxiter: Optional[int] = None,
    seed: int = 0,
) -> Tuple[List[LocalMode], Dict[str, Any]]:
    """Return the smallest accepted thesis-``lambda`` modes of one local pencil."""

    requested = max(0, int(num_modes))
    nloc = int(local.A_neumann.shape[0])
    if requested == 0 or nloc == 0:
        return [], {
            "eigensolver": "none",
            "eigensolver_status": "converged",
            "requested_modes": requested,
            "returned_raw_modes": 0,
            "accepted_finite_modes": 0,
        }
    validation_started = time.perf_counter()
    nullspace = _detect_nullspace_structure(local)
    common_nullspace = bool(nullspace["common_nullspace"])
    explicit_zero_count = int(local.has_neumann_nullspace)
    reduced_dimension = nloc - explicit_zero_count
    if not local.has_neumann_nullspace:
        positive_available = nloc
        infinite_mode_count = 0
    elif common_nullspace:
        positive_available = reduced_dimension
        infinite_mode_count = 0
    else:
        positive_available = max(0, reduced_dimension - 1)
        infinite_mode_count = 1
    finite_spectrum_dimension = explicit_zero_count + positive_available

    modes: List[LocalMode] = []
    transform: Optional[sp.csr_matrix] = None
    constraint_matrix: Optional[np.ndarray] = None
    # Once the relevant null directions have been removed, solve the original
    # thesis-convention pencil directly for its smallest finite lambdas.  This
    # avoids amplifying high-contrast eigenvector errors through reciprocal
    # conversion while leaving the mathematical pencil and selection unchanged.
    direct_finite_solve = True
    numerator = local.A_neumann
    metric = local.B_pou_energy
    double_deflated_direct_finite_solve = bool(
        local.has_neumann_nullspace and not common_nullspace
    )
    if local.has_neumann_nullspace:
        constant = _normalize_local_mode(local, np.ones(nloc, dtype=np.float64))
        mode_diag = _mode_diagnostics(local, constant, 0.0, is_zero_mode=True)
        modes.append(
            LocalMode(
                subdomain=int(local.subdomain),
                local_mode=0,
                eigenvalue=0.0,
                reciprocal_eigenvalue=float("inf"),
                vector=constant,
                global_indices=local.global_indices,
                pou_weights=local.pou_weights,
                is_neumann_null_mode=True,
                rayleigh_quotient=float(mode_diag["rayleigh_quotient"]),
                eig_residual=float(mode_diag["eig_residual"]),
                relative_rq_error=None,
                lambda_scale_floor=float(mode_diag["lambda_scale_floor"]),
                zero_mode_nullspace_error=float(mode_diag["zero_mode_nullspace_error"]),
                reciprocal_residual=None,
            )
        )
        if reduced_dimension > 0:
            if common_nullspace:
                constraint = np.ones(nloc, dtype=np.float64)
                constraint_matrix = constraint[:, None]
                if nloc <= int(dense_cutoff):
                    transform = _constraint_basis(constraint)
                    numerator = (transform.T @ local.A_neumann @ transform).tocsr()
                    metric = (transform.T @ local.B_pou_energy @ transform).tocsr()
            else:
                # Positive finite eigenvectors obey both identities
                #
                #   (B 1)^T p = 0,             from 1^T A p = 0,
                #   (A D^{-1}1)^T p = 0,       from (D^{-1}1)^T B p = 0.
                #
                # The first constraint removes the explicit lambda=0
                # constant mode and the second removes the independent B-null
                # direction (lambda=infinity).  The resulting direct pencil is
                # SPD--SPD and can be solved for its smallest thesis lambdas.
                first_constraint = np.asarray(
                    local.B_pou_energy @ np.ones(nloc, dtype=np.float64), dtype=np.float64
                )
                b_null = 1.0 / np.asarray(local.pou_weights, dtype=np.float64)
                second_full_constraint = np.asarray(
                    local.A_neumann @ b_null, dtype=np.float64
                ).reshape(-1)
                constraint_matrix = np.column_stack(
                    (first_constraint, second_full_constraint)
                )
                if nloc <= int(dense_cutoff):
                    first_transform = _constraint_basis(first_constraint)
                    second_constraint = np.asarray(
                        first_transform.T @ second_full_constraint, dtype=np.float64
                    ).reshape(-1)
                    second_transform = _constraint_basis(second_constraint)
                    transform = (first_transform @ second_transform).tocsr()
                    numerator = (transform.T @ local.A_neumann @ transform).tocsr()
                    metric = (transform.T @ local.B_pou_energy @ transform).tocsr()

    finite_requested = min(
        max(0, requested - explicit_zero_count), positive_available
    )
    implicit_constraint_solve = bool(
        constraint_matrix is not None and nloc > int(dense_cutoff)
    )
    if implicit_constraint_solve:
        raw_values, raw_vectors, solver_diag = _generalized_smallest_constrained_modes(
            local.A_neumann,
            local.B_pou_energy,
            constraint_matrix,
            k=finite_requested,
            eig_tol=float(eig_tol),
            maxiter=maxiter,
            seed=int(seed) + 7919 * int(local.subdomain),
        )
    elif direct_finite_solve:
        raw_values, raw_vectors, solver_diag = _generalized_smallest_modes(
            numerator,
            metric,
            k=finite_requested,
            dense_cutoff=int(dense_cutoff),
            eig_tol=float(eig_tol),
            maxiter=maxiter,
            seed=int(seed) + 7919 * int(local.subdomain),
        )
    else:
        raw_values, raw_vectors, solver_diag = _generalized_largest_modes(
            numerator,
            metric,
            k=finite_requested,
            dense_cutoff=int(dense_cutoff),
            eig_tol=float(eig_tol),
            maxiter=maxiter,
            seed=int(seed) + 7919 * int(local.subdomain),
        )
    rejected_nonfinite = 0
    rejected_negative = 0
    rejected_numerical_zero = 0
    rejected_bad_residual = 0
    filtered_modes: List[Dict[str, Any]] = []
    norm_metric = max(float(spla.norm(metric, ord=np.inf)), np.finfo(np.float64).tiny)
    norm_numerator = max(
        float(spla.norm(numerator, ord=np.inf)), np.finfo(np.float64).tiny
    )
    raw_operator_scale = norm_numerator / norm_metric
    reduced_dim = int(metric.shape[0])
    for raw_value, reduced_vector in zip(raw_values, raw_vectors.T):
        raw_value = float(raw_value)
        # A positive generalized eigenvalue is not a numerical zero merely
        # because another eigenvalue is much larger.  This floor is tied only
        # to float64 underflow and the current value itself; it cannot act as a
        # physical spectral truncation.
        machine_zero_floor = max(
            np.finfo(np.float64).tiny
            * max(raw_operator_scale, 1.0)
            * max(reduced_dim, 1),
            64.0 * np.finfo(np.float64).eps * abs(raw_value),
        )
        reason: Optional[str] = None
        if not math.isfinite(raw_value):
            rejected_nonfinite += 1
            reason = (
                "nonfinite_direct_eigenvalue"
                if direct_finite_solve
                else "nonfinite_reciprocal"
            )
        elif raw_value < -machine_zero_floor:
            rejected_negative += 1
            reason = (
                "negative_direct_eigenvalue"
                if direct_finite_solve
                else "negative_reciprocal"
            )
        elif abs(raw_value) <= machine_zero_floor:
            rejected_numerical_zero += 1
            reason = (
                "machine_precision_direct_eigenvalue_zero"
                if direct_finite_solve
                else "machine_precision_reciprocal_zero"
            )
        if reason is not None:
            filtered_modes.append(
                {
                    "raw_eigenvalue": raw_value,
                    "reason": reason,
                    "machine_zero_floor": float(machine_zero_floor),
                }
            )
            continue
        if direct_finite_solve:
            eigenvalue = raw_value
            reciprocal = float(1.0 / eigenvalue)
            reciprocal_residual = _reciprocal_pair_residual(
                metric,
                numerator,
                reduced_vector,
                reciprocal,
                numerator_norm=norm_metric,
                metric_norm=norm_numerator,
            )
        else:
            reciprocal = raw_value
            eigenvalue = float(1.0 / reciprocal)
            reciprocal_residual = _reciprocal_pair_residual(
                numerator,
                metric,
                reduced_vector,
                reciprocal,
                numerator_norm=norm_numerator,
                metric_norm=norm_metric,
            )
        vector = (
            np.asarray(transform @ reduced_vector, dtype=np.float64)
            if transform is not None
            else np.asarray(reduced_vector, dtype=np.float64)
        )
        vector = _normalize_local_mode(local, vector)
        mode_diag = _mode_diagnostics(local, vector, eigenvalue, is_zero_mode=False)
        if (
            not math.isfinite(float(mode_diag["eig_residual"]))
            or float(mode_diag["eig_residual"]) > float(eigen_residual_tol)
        ):
            rejected_bad_residual += 1
            filtered_modes.append(
                {
                    "reciprocal_eigenvalue": reciprocal,
                    "eigenvalue": eigenvalue,
                    "reason": "bad_direct_pencil_residual",
                    "eig_residual": float(mode_diag["eig_residual"]),
                    "residual_tolerance": float(eigen_residual_tol),
                    "reciprocal_residual": float(reciprocal_residual),
                }
            )
            continue
        modes.append(
            LocalMode(
                subdomain=int(local.subdomain),
                local_mode=len(modes),
                eigenvalue=eigenvalue,
                reciprocal_eigenvalue=reciprocal,
                vector=vector,
                global_indices=local.global_indices,
                pou_weights=local.pou_weights,
                is_neumann_null_mode=False,
                rayleigh_quotient=float(mode_diag["rayleigh_quotient"]),
                eig_residual=float(mode_diag["eig_residual"]),
                relative_rq_error=float(mode_diag["relative_rq_error"]),
                lambda_scale_floor=float(mode_diag["lambda_scale_floor"]),
                zero_mode_nullspace_error=None,
                reciprocal_residual=float(reciprocal_residual),
            )
        )
    modes.sort(key=lambda item: (float(item.eigenvalue), int(item.local_mode)))
    for local_mode, mode in enumerate(modes):
        mode.local_mode = int(local_mode)
    validation_time = time.perf_counter() - validation_started - float(
        solver_diag["local_eigensolve_time"]
    )
    rejected_nonpositive = rejected_negative + rejected_numerical_zero
    accepted_finite = len(modes)
    finite_complete = bool(
        solver_diag["eigensolver_status"] == "converged"
        and accepted_finite >= min(requested, finite_spectrum_dimension)
        and requested >= finite_spectrum_dimension
        and rejected_nonfinite == 0
        and rejected_nonpositive == 0
        and rejected_bad_residual == 0
    )
    positive_modes = [mode for mode in modes if not mode.is_neumann_null_mode]
    diagnostics: Dict[str, Any] = {
        "subdomain": int(local.subdomain),
        "patch_row": int(local.patch_row),
        "patch_col": int(local.patch_col),
        "local_shape": [int(local.height), int(local.width)],
        "local_dofs": nloc,
        "physical_boundary_faces": int(local.physical_boundary_faces),
        "physical_neumann_faces": int(local.physical_neumann_faces),
        "touches_physical_boundary": dict(local.touches_physical_boundary),
        "connected_components": int(local.connected_components),
        "local_A_symmetry_error": float(local.A_symmetry_error),
        "local_B_symmetry_error": float(local.B_symmetry_error),
        "face_transmissibility_min": float(local.face_transmissibility_min),
        "face_transmissibility_median": float(local.face_transmissibility_median),
        "face_transmissibility_max": float(local.face_transmissibility_max),
        "face_transmissibility_contrast": float(
            local.face_transmissibility_max
            / max(local.face_transmissibility_min, np.finfo(np.float64).tiny)
        ),
        "has_neumann_nullspace": bool(local.has_neumann_nullspace),
        **nullspace,
        "reduced_dimension": int(reduced_dimension),
        "available_positive_finite_modes": int(positive_available),
        "finite_spectrum_dimension": int(finite_spectrum_dimension),
        "infinite_mode_count": int(infinite_mode_count),
        "direct_finite_solve": bool(direct_finite_solve),
        "double_deflated_direct_finite_solve": bool(
            double_deflated_direct_finite_solve
        ),
        "implicit_constraint_solve": bool(implicit_constraint_solve),
        "requested_modes": int(requested),
        "returned_raw_modes": int(explicit_zero_count + raw_values.size),
        "accepted_finite_modes": int(accepted_finite),
        "returned_modes": int(len(modes)),
        "rejected_nonfinite_modes": int(rejected_nonfinite),
        "rejected_nonpositive_modes": int(rejected_nonpositive),
        "rejected_negative_modes": int(rejected_negative),
        "rejected_numerical_zero_modes": int(rejected_numerical_zero),
        "rejected_bad_residual_modes": int(rejected_bad_residual),
        "filtered_raw_modes": filtered_modes,
        "filtered_reciprocal_modes": filtered_modes,
        "finite_spectrum_complete": finite_complete,
        "eigenvalues": [float(mode.eigenvalue) for mode in modes],
        "reciprocal_eigenvalues": [float(mode.reciprocal_eigenvalue) for mode in modes],
        "eig_residuals": [float(mode.eig_residual) for mode in modes],
        "relative_rq_errors_positive_finite": [
            float(mode.relative_rq_error) for mode in positive_modes if mode.relative_rq_error is not None
        ],
        "zero_mode_nullspace_errors": [
            float(mode.zero_mode_nullspace_error)
            for mode in modes
            if mode.zero_mode_nullspace_error is not None
        ],
        "max_direct_eig_residual": max(
            (float(mode.eig_residual) for mode in positive_modes), default=0.0
        ),
        "max_relative_rq_error_positive_finite": max(
            (float(mode.relative_rq_error) for mode in positive_modes if mode.relative_rq_error is not None),
            default=0.0,
        ),
        "max_zero_mode_nullspace_error": max(
            (float(mode.zero_mode_nullspace_error) for mode in modes if mode.zero_mode_nullspace_error is not None),
            default=0.0,
        ),
        "local_mode_validation_time": float(max(validation_time, 0.0)),
        "convention": "thesis_4.25_small_lambda",
        "residual_definition": "||Ap-lambda*Bp||/(||Ap||+|lambda|*||Bp||+scaled_float64_eps)",
        **solver_diag,
    }
    return modes[:requested], diagnostics


def dense_modified_geneo_finite_spectrum(local: LocalSubdomain) -> np.ndarray:
    """Dense small-problem reference for all reported finite thesis lambdas."""

    nloc = int(local.A_neumann.shape[0])
    nullspace = _detect_nullspace_structure(local)
    values: List[float] = []
    if not local.has_neumann_nullspace:
        dense = la.eigh(
            local.A_neumann.toarray(), local.B_pou_energy.toarray(), check_finite=True
        )[0]
        values.extend(float(value) for value in dense if math.isfinite(float(value)) and value > 0.0)
    else:
        values.append(0.0)
        ones = np.ones(nloc, dtype=np.float64)
        if bool(nullspace["common_nullspace"]):
            transform = _constraint_basis(ones)
            Ared = np.asarray((transform.T @ local.A_neumann @ transform).toarray())
            Bred = np.asarray((transform.T @ local.B_pou_energy @ transform).toarray())
            reciprocal = la.eigh(Bred, Ared, check_finite=True)[0]
            scale = max(
                float(np.linalg.norm(Bred, ord=np.inf))
                / max(float(np.linalg.norm(Ared, ord=np.inf)), np.finfo(np.float64).tiny),
                np.finfo(np.float64).tiny,
            )
            floor = 64.0 * np.finfo(np.float64).eps * max(nloc, 1) * scale
            values.extend(float(1.0 / mu) for mu in reciprocal if mu > floor)
        else:
            first = _constraint_basis(
                np.asarray(local.B_pou_energy @ ones, dtype=np.float64)
            )
            b_null = 1.0 / np.asarray(local.pou_weights, dtype=np.float64)
            second = _constraint_basis(
                np.asarray(
                    first.T @ (local.A_neumann @ b_null), dtype=np.float64
                ).reshape(-1)
            )
            transform = (first @ second).tocsr()
            Ared = np.asarray((transform.T @ local.A_neumann @ transform).toarray())
            Bred = np.asarray((transform.T @ local.B_pou_energy @ transform).toarray())
            direct = la.eigh(Ared, Bred, check_finite=True)[0]
            values.extend(
                float(value)
                for value in direct
                if math.isfinite(float(value)) and value > 0.0
            )
    return np.asarray(sorted(values), dtype=np.float64)


def _assemble_and_compress(
    candidates: Sequence[LocalMode],
    *,
    num_global_dofs: int,
    target_dimension: Optional[int],
    independence_tol: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """PoU-extend candidates and either preserve their span or SVD-compress it."""

    extension_start = time.perf_counter()
    if not candidates:
        return np.zeros((int(num_global_dofs), 0), dtype=np.float64), {
            "candidate_count": 0,
            "candidate_rank": 0,
            "dropped_dependent": 0,
            "gram_eigenvalues": [],
            "pou_extension_time": float(time.perf_counter() - extension_start),
            "candidate_compression_time": 0.0,
            "global_independent_span_time": 0.0,
        }
    data_parts: List[np.ndarray] = []
    index_parts: List[np.ndarray] = []
    indptr = [0]
    for mode in candidates:
        values = np.asarray(mode.pou_weights * mode.vector, dtype=np.float64)
        norm = float(np.linalg.norm(values))
        if not math.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
            continue
        data_parts.append(values / norm)
        index_parts.append(np.asarray(mode.global_indices, dtype=np.int64))
        indptr.append(indptr[-1] + int(values.size))
    valid_candidates = len(data_parts)
    if valid_candidates == 0:
        return np.zeros((int(num_global_dofs), 0), dtype=np.float64), {
            "candidate_count": len(candidates),
            "candidate_rank": 0,
            "dropped_dependent": len(candidates),
            "gram_eigenvalues": [],
            "pou_extension_time": float(time.perf_counter() - extension_start),
            "candidate_compression_time": 0.0,
            "global_independent_span_time": 0.0,
        }
    sparse_pool = sp.csc_matrix(
        (
            np.concatenate(data_parts),
            np.concatenate(index_parts),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(int(num_global_dofs), int(valid_candidates)),
        dtype=np.float64,
    )
    extension_time = time.perf_counter() - extension_start
    span_start = time.perf_counter()
    gram = np.asarray((sparse_pool.T @ sparse_pool).toarray(), dtype=np.float64)
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)
    scale = max(float(eigenvalues[0]) if eigenvalues.size else 0.0, np.finfo(np.float64).tiny)
    rank = int(np.sum(eigenvalues > float(independence_tol) * scale))
    output_dimension = rank if target_dimension is None else min(int(target_dimension), rank)
    if output_dimension == 0:
        basis = np.zeros((int(num_global_dofs), 0), dtype=np.float64)
    else:
        transform = eigenvectors[:, :output_dimension] * eigenvalues[:output_dimension].clip(
            min=np.finfo(np.float64).tiny
        ) ** -0.5
        basis = np.asarray(sparse_pool @ transform, dtype=np.float64)
    span_time = time.perf_counter() - span_start
    return basis, {
        "candidate_count": int(len(candidates)),
        "valid_candidate_count": int(valid_candidates),
        "candidate_rank": int(rank),
        "dropped_dependent": int(len(candidates) - rank),
        "output_dimension": int(output_dimension),
        "gram_eigenvalues": [float(value) for value in eigenvalues],
        "compression": (
            "full_independent_span"
            if target_dimension is None
            else "leading_euclidean_svd_of_normalized_pou_pool"
        ),
        "pou_extension_time": float(extension_time),
        "candidate_compression_time": float(span_time if target_dimension is not None else 0.0),
        "global_independent_span_time": float(span_time if target_dimension is None else 0.0),
    }


def _geometric_threshold(asm: Any) -> float:
    """Return the structured-grid geometric heuristic ``(s-d)/s``."""

    asm = _as_local_spectral_asm_view(asm)
    overlap_cells = max(int(asm.s) - int(asm.d), 1)
    return float(overlap_cells / max(int(asm.s), 1))


def _threshold_search_local(
    local: LocalSubdomain,
    *,
    threshold: float,
    initial_modes: int,
    max_local_modes: int,
    dense_cutoff: int,
    eig_tol: float,
    eigen_residual_tol: float,
    seed: int,
    grid: int,
    sample_id: Optional[int],
) -> Tuple[List[LocalMode], Dict[str, Any]]:
    requested = max(1, min(int(initial_modes), int(max_local_modes), local.A_neumann.shape[0]))
    last_modes: List[LocalMode] = []
    last_diag: Dict[str, Any] = {}
    while True:
        modes, diagnostics = solve_modified_geneo_local_spectrum(
            local,
            num_modes=requested,
            dense_cutoff=int(dense_cutoff),
            eig_tol=float(eig_tol),
            eigen_residual_tol=float(eigen_residual_tol),
            seed=int(seed),
        )
        last_modes, last_diag = modes, diagnostics
        chosen = [mode for mode in modes if float(mode.eigenvalue) <= float(threshold)]
        unselected = [mode for mode in modes if float(mode.eigenvalue) > float(threshold)]
        largest_selected = (
            max(float(mode.eigenvalue) for mode in chosen) if chosen else float("-inf")
        )
        smallest_unselected = (
            min(float(mode.eigenvalue) for mode in unselected) if unselected else float("inf")
        )
        solver_failure = diagnostics.get("eigensolver_status") != "converged"
        numerical_filter = any(
            int(diagnostics.get(key, 0)) > 0
            for key in (
                "rejected_nonfinite_modes",
                "rejected_nonpositive_modes",
                "rejected_bad_residual_modes",
            )
        )
        true_bracket = bool(unselected and smallest_unselected > float(threshold))
        exhausted = bool(diagnostics.get("finite_spectrum_complete", False))
        bracketed = bool((true_bracket or exhausted) and not solver_failure and not numerical_filter)
        threshold_diag = {
            **diagnostics,
            "threshold": float(threshold),
            "selected_count": int(len(chosen)),
            "largest_selected_lambda": float(largest_selected),
            "smallest_unselected_lambda": float(smallest_unselected),
            "threshold_margin_below": float(threshold - largest_selected),
            "threshold_margin_above": float(smallest_unselected - threshold),
            "threshold_bracketed": bracketed,
            "stopped_by_true_bracket": bool(bracketed and true_bracket),
            "stopped_by_available_dimension": bool(bracketed and exhausted and not true_bracket),
            "stopped_by_max_local_modes": False,
            "stopped_by_eigensolver_failure": bool(solver_failure),
            "stopped_by_numerical_filter": bool(numerical_filter),
            "search_requested_modes": int(requested),
        }
        if bracketed:
            return chosen, threshold_diag
        if solver_failure or numerical_filter:
            last_diag = threshold_diag
            break
        finite_dimension = int(diagnostics.get("finite_spectrum_dimension", local.A_neumann.shape[0]))
        next_requested = min(
            max(requested + 1, 2 * requested), int(max_local_modes), finite_dimension
        )
        if next_requested <= requested:
            threshold_diag["stopped_by_max_local_modes"] = bool(
                requested >= int(max_local_modes) and requested < finite_dimension
            )
            threshold_diag["stopped_by_available_dimension"] = bool(
                requested >= finite_dimension and not exhausted
            )
            last_diag = threshold_diag
            break
        requested = next_requested
    context = (
        f"grid={grid}, sample={sample_id if sample_id is not None else 'unknown'}, "
        f"subdomain={local.subdomain}, threshold={threshold:.16g}, "
        f"requested={last_diag.get('search_requested_modes', requested)}, "
        f"returned_raw={last_diag.get('returned_raw_modes')}, "
        f"accepted={last_diag.get('accepted_finite_modes')}, "
        f"solver={last_diag.get('eigensolver_status')}"
    )
    raise ThresholdBracketError(
        "modified GenEO threshold is not strictly bracketed; "
        f"{context}. Increase max_local_modes or inspect the eigensolver diagnostics.",
        last_diag,
    )


def build_modified_geneo_coarse_space(
    asm: Any,
    *,
    selection: str,
    rank: Optional[int] = None,
    budget: Optional[int] = None,
    threshold: Optional[float] = None,
    dictionary_modes_per_patch: Optional[int] = None,
    budget_search_modes: int = 2,
    threshold_initial_modes: int = 8,
    max_local_modes: int = 64,
    dense_cutoff: int = 512,
    eig_tol: float = 1e-9,
    eigen_residual_tol: float = 1e-6,
    independence_tol: float = 1e-10,
    diagnostic_spectrum_modes: int = 0,
    boundary_types: Optional[Mapping[str, str]] = None,
    batch_index: int = 0,
    sample_id: Optional[int] = None,
    seed: int = 0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Build a threshold GenEO span or a compressed local-spectral dictionary."""

    total_started = time.perf_counter()
    source_operator_type = type(asm).__name__
    asm = _as_local_spectral_asm_view(asm)
    if int(asm.a.shape[0]) != 1:
        raise ValueError(
            "local spectral provider expects one coefficient sample per operator; "
            f"got batch={asm.a.shape[0]}"
        )
    resolved_boundary = _normalize_boundary_types(boundary_types)
    if (
        resolved_boundary != _ALL_DIRICHLET
        and resolved_boundary != _MOVING_INTERFACE_BOUNDARY
    ):
        raise NotImplementedError(
            "The validated modified GenEO provider supports only the static all-Dirichlet "
            "configuration or the moving-interface configuration with west/east "
            "Dirichlet and north/south Neumann boundaries."
        )
    if (
        source_operator_type == "MobilityFVMOperatorASM"
        and resolved_boundary != _MOVING_INTERFACE_BOUNDARY
    ):
        raise ValueError(
            "MobilityFVMOperatorASM must use west/east Dirichlet and north/south "
            "Neumann boundary_types in modified GenEO."
        )
    selection_key = str(selection).strip().lower().replace("-", "_")
    legacy_alias = selection_key in {"budget", "modified_geneo_budget"}
    if legacy_alias:
        warnings.warn(
            f"selection={selection!r} is a legacy alias; use "
            "'compressed_local_spectral_dictionary'. Results use the new non-GenEO label.",
            FutureWarning,
            stacklevel=2,
        )
        selection_key = "compressed_local_spectral_dictionary"
    if selection_key in {"compressed", "compressed_local_spectral"}:
        selection_key = "compressed_local_spectral_dictionary"
    if selection_key in {"threshold", "threshold_modified_geneo"}:
        selection_key = "threshold_modified_geneo"
    if selection_key not in {
        "compressed_local_spectral_dictionary",
        "threshold_modified_geneo",
    }:
        raise ValueError(
            "selection must be 'compressed_local_spectral_dictionary' or "
            "'threshold_modified_geneo'"
        )
    if rank is not None and budget is not None and int(rank) != int(budget):
        raise ValueError("rank and legacy budget disagree")
    target_rank = int(rank if rank is not None else budget) if (rank is not None or budget is not None) else None
    if selection_key == "compressed_local_spectral_dictionary":
        if target_rank is None or target_rank <= 0:
            raise ValueError("compressed local spectral selection requires a positive rank")
        modes_per_patch = int(
            dictionary_modes_per_patch
            if dictionary_modes_per_patch is not None
            else budget_search_modes
        )
        if modes_per_patch <= 0:
            raise ValueError("dictionary_modes_per_patch must be positive")
    else:
        threshold = _geometric_threshold(asm) if threshold is None else float(threshold)
        if not math.isfinite(float(threshold)) or float(threshold) < 0.0:
            raise ValueError("threshold must be finite and non-negative")

    global_data = prepare_local_spectral_global_data(
        asm, batch_index=int(batch_index), boundary_types=resolved_boundary
    )
    topology_cache, topology_hit, topology_lookup_time = get_local_spectral_topology_cache(
        asm, global_data
    )
    all_modes: List[LocalMode] = []
    local_diagnostics: List[Dict[str, Any]] = []
    numeric_assembly_time = 0.0
    local_eigensolve_time = 0.0
    local_validation_time = 0.0
    patch_features: List[Dict[str, Any]] = []
    for subdomain in topology_cache.active_subdomain_ids:
        local = assemble_local_neumann_matrix(
            asm,
            subdomain,
            global_data=global_data,
            topology_cache=topology_cache,
        )
        numeric_assembly_time += float(local.numeric_assembly_time)
        patch_features.append(
            {
                "subdomain": int(subdomain),
                "is_boundary": bool(any(local.touches_physical_boundary.values())),
                "face_contrast": float(
                    local.face_transmissibility_max
                    / max(local.face_transmissibility_min, np.finfo(np.float64).tiny)
                ),
                "face_median": float(local.face_transmissibility_median),
            }
        )
        try:
            if selection_key == "threshold_modified_geneo":
                chosen, diagnostics = _threshold_search_local(
                    local,
                    threshold=float(threshold),
                    initial_modes=int(threshold_initial_modes),
                    max_local_modes=int(max_local_modes),
                    dense_cutoff=int(dense_cutoff),
                    eig_tol=float(eig_tol),
                    eigen_residual_tol=float(eigen_residual_tol),
                    seed=int(seed),
                    grid=int(asm.n),
                    sample_id=sample_id,
                )
                all_modes.extend(chosen)
            else:
                requested = min(int(modes_per_patch), int(local.A_neumann.shape[0]))
                chosen, diagnostics = solve_modified_geneo_local_spectrum(
                    local,
                    num_modes=requested,
                    dense_cutoff=int(dense_cutoff),
                    eig_tol=float(eig_tol),
                    eigen_residual_tol=float(eigen_residual_tol),
                    seed=int(seed),
                )
        except ThresholdBracketError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"local spectral solve failed for grid={asm.n}, sample={sample_id}, "
                f"subdomain={subdomain}: {type(exc).__name__}: {exc}"
            ) from exc
        if selection_key != "threshold_modified_geneo":
            if diagnostics.get("eigensolver_status") != "converged":
                raise RuntimeError(
                    f"local eigensolver failed for grid={asm.n}, sample={sample_id}, "
                    f"subdomain={subdomain}: {diagnostics.get('eigensolver_message')}"
                )
            if int(diagnostics.get("rejected_bad_residual_modes", 0)) > 0:
                raise RuntimeError(
                    f"local direct-pencil residual rejection for grid={asm.n}, "
                    f"sample={sample_id}, subdomain={subdomain}"
                )
            all_modes.extend(chosen)
        local_eigensolve_time += float(diagnostics.get("local_eigensolve_time", 0.0))
        local_validation_time += float(diagnostics.get("local_mode_validation_time", 0.0))
        local_diagnostics.append(diagnostics)

    all_modes.sort(
        key=lambda item: (float(item.eigenvalue), int(item.subdomain), int(item.local_mode))
    )
    basis_numpy, assembly_diagnostics = _assemble_and_compress(
        all_modes,
        num_global_dofs=int(asm.n) * int(asm.n),
        target_dimension=(target_rank if selection_key == "compressed_local_spectral_dictionary" else None),
        independence_tol=float(independence_tol),
    )
    if selection_key == "compressed_local_spectral_dictionary" and basis_numpy.shape[1] != int(target_rank):
        raise RuntimeError(
            f"rank={target_rank} requested but only {basis_numpy.shape[1]} independent "
            f"global modes were found from {len(all_modes)} candidates; increase "
            "dictionary_modes_per_patch"
        )
    if basis_numpy.shape[1] == 0:
        raise RuntimeError("local spectral selection produced an empty global coarse space")
    device_transfer_start = time.perf_counter()
    basis = torch.from_numpy(basis_numpy).to(device=asm.device, dtype=torch.float64)
    _sync_tensor_device(basis)
    basis_device_transfer_time = time.perf_counter() - device_transfer_start
    basis = basis.unsqueeze(0).contiguous()
    selected_by_subdomain: Dict[int, int] = {}
    for mode in all_modes:
        selected_by_subdomain[mode.subdomain] = selected_by_subdomain.get(mode.subdomain, 0) + 1
    is_threshold = selection_key == "threshold_modified_geneo"
    pou_sum = np.zeros(int(asm.n) * int(asm.n), dtype=np.float64)
    for topology in topology_cache.patches.values():
        pou_sum[topology.global_indices] += topology.pou_weights
    pou_error = float(np.linalg.norm(pou_sum - 1.0, ord=np.inf))
    total_basis_setup_time = time.perf_counter() - total_started
    topology_build_time = 0.0 if topology_hit else float(topology_cache.build_time)
    stage_timing = {
        "global_stencil_transfer_time": float(global_data.transfer_time),
        "topology_cache_build_time": topology_build_time,
        "topology_cache_lookup_time": float(topology_lookup_time),
        "local_neumann_numeric_assembly_time": float(numeric_assembly_time),
        "local_eigensolve_time": float(local_eigensolve_time),
        "local_mode_validation_time": float(local_validation_time),
        "pou_extension_time": float(assembly_diagnostics["pou_extension_time"]),
        "candidate_compression_time": float(assembly_diagnostics["candidate_compression_time"]),
        "global_independent_span_time": float(assembly_diagnostics["global_independent_span_time"]),
        "basis_device_transfer_time": float(basis_device_transfer_time),
        "global_A_orthonormalization_time": 0.0,
        "Galerkin_assembly_time": 0.0,
        "total_basis_setup_time": float(total_basis_setup_time),
    }
    representative_local_spectra: List[Dict[str, Any]] = []
    diagnostic_spectra_started = time.perf_counter()
    if int(diagnostic_spectrum_modes) > 0 and patch_features:
        interior = [item for item in patch_features if not item["is_boundary"]]
        boundary = [item for item in patch_features if item["is_boundary"]]
        representatives: List[Tuple[str, Dict[str, Any]]] = []
        if interior:
            representatives.extend(
                [
                    ("interior_not_crossing_interface", min(interior, key=lambda x: x["face_contrast"])),
                    ("interior_crossing_interface", max(interior, key=lambda x: x["face_contrast"])),
                    ("high_coefficient_region", max(interior, key=lambda x: x["face_median"])),
                    ("low_coefficient_region", min(interior, key=lambda x: x["face_median"])),
                ]
            )
        if boundary:
            representatives.append(
                ("physical_boundary_patch", max(boundary, key=lambda x: x["face_contrast"]))
            )
        solved_subdomains: Dict[int, Tuple[List[LocalMode], Dict[str, Any]]] = {}
        for patch_type, feature in representatives:
            subdomain = int(feature["subdomain"])
            if subdomain not in solved_subdomains:
                local = assemble_local_neumann_matrix(
                    asm,
                    subdomain,
                    global_data=global_data,
                    topology_cache=topology_cache,
                )
                modes, local_diag = solve_modified_geneo_local_spectrum(
                    local,
                    num_modes=min(int(diagnostic_spectrum_modes), int(local.A_neumann.shape[0])),
                    dense_cutoff=int(dense_cutoff),
                    eig_tol=float(eig_tol),
                    eigen_residual_tol=float(eigen_residual_tol),
                    seed=int(seed),
                )
                solved_subdomains[subdomain] = (modes, local_diag)
            modes, local_diag = solved_subdomains[subdomain]
            representative_local_spectra.append(
                {
                    "patch_type": patch_type,
                    "subdomain": subdomain,
                    "face_transmissibility_contrast": float(feature["face_contrast"]),
                    "face_transmissibility_median": float(feature["face_median"]),
                    "eigenvalues": [float(mode.eigenvalue) for mode in modes],
                    "common_nullspace": bool(local_diag.get("common_nullspace", False)),
                    "has_neumann_nullspace": bool(local_diag.get("has_neumann_nullspace", False)),
                    "eigensolver": str(local_diag.get("eigensolver")),
                    "max_direct_eig_residual": float(local_diag.get("max_direct_eig_residual", 0.0)),
                }
            )
    diagnostic_local_spectra_time = time.perf_counter() - diagnostic_spectra_started
    diagnostics: Dict[str, Any] = {
        "provider": (
            "threshold_modified_geneo_whole_subdomain"
            if is_threshold
            else "compressed_local_spectral_dictionary"
        ),
        "method_label": (
            "threshold-modified-GenEO-TG-FGCR"
            if is_threshold
            else f"compressed-local-spectral-{target_rank}-TG-FGCR"
        ),
        "reference_formula": "Spillane_thesis_Definition_4.38_equation_4.25",
        "selection": selection_key,
        "legacy_selection_alias_used": bool(legacy_alias),
        "rank": int(target_rank) if target_rank is not None else None,
        "threshold": float(threshold) if threshold is not None else None,
        "geometric_heuristic_threshold": float(_geometric_threshold(asm)),
        "eigenvalue_convention": "lambda=A_N_energy/PoU_A_N_energy; ascending",
        "reciprocal_solver_convention": "mu=1/lambda; descending; output converted to lambda",
        "is_standard_geneo": bool(is_threshold),
        "is_standard_geneo_coarse_space": bool(is_threshold),
        "embedded_in_common_tg_wrapper": True,
        "is_standard_full_geneo_preconditioner": False,
        "num_subdomains": int(len(topology_cache.active_subdomain_ids)),
        "active_subdomain_ids": list(topology_cache.active_subdomain_ids),
        "asm_patch_slots": int(asm.K),
        "skipped_padding_only_patch_slots": int(len(topology_cache.padding_only_slots)),
        "padding_only_slots": list(topology_cache.padding_only_slots),
        "patch_size": int(asm.s),
        "patch_stride": int(asm.d),
        "target_overlap": float(asm.target_overlap),
        "boundary_types": dict(resolved_boundary),
        "source_operator_type": source_operator_type,
        "formal_validation_scope": (
            "static_all_dirichlet_interface"
            if resolved_boundary == _ALL_DIRICHLET
            else "moving_interface_west_east_dirichlet_north_south_neumann"
        ),
        "raw_selected_local_modes": int(len(all_modes)),
        "effective_global_dimension": int(basis.shape[-1]),
        "final_A_rank": None,
        "rejected_dependent_modes": int(assembly_diagnostics["dropped_dependent"]),
        "global_assembly": assembly_diagnostics,
        "full_grid_cpu_transfer_count": int(global_data.full_grid_cpu_transfer_count),
        "source_device": global_data.source_device,
        "source_dtype": global_data.dtype,
        "topology_cache_hit": bool(topology_hit),
        "topology_cache_key": repr(topology_cache.key),
        "stage_timing": stage_timing,
        "diagnostic_local_spectra_time_excluded_from_basis_setup": float(
            diagnostic_local_spectra_time
        ),
        "representative_local_spectra": representative_local_spectra,
        "candidate_local_modes": [
            {
                "subdomain": int(mode.subdomain),
                "local_mode": int(mode.local_mode),
                "eigenvalue": float(mode.eigenvalue),
                "is_neumann_null_mode": bool(mode.is_neumann_null_mode),
                "eig_residual": float(mode.eig_residual),
                "relative_rq_error": (
                    float(mode.relative_rq_error) if mode.relative_rq_error is not None else None
                ),
            }
            for mode in all_modes
        ],
        "selected_modes_by_subdomain": {
            str(key): int(value) for key, value in sorted(selected_by_subdomain.items())
        },
        "max_selected_eigenvalue": max(
            (float(mode.eigenvalue) for mode in all_modes), default=float("nan")
        ),
        "max_selected_direct_eig_residual": max(
            (float(mode.eig_residual) for mode in all_modes if not mode.is_neumann_null_mode),
            default=0.0,
        ),
        "max_selected_zero_mode_nullspace_error": max(
            (
                float(mode.zero_mode_nullspace_error)
                for mode in all_modes
                if mode.zero_mode_nullspace_error is not None
            ),
            default=0.0,
        ),
        "partition_of_unity_error": pou_error,
        "max_local_A_symmetry_error": max(
            (float(item.get("local_A_symmetry_error", 0.0)) for item in local_diagnostics),
            default=0.0,
        ),
        "max_local_B_symmetry_error": max(
            (float(item.get("local_B_symmetry_error", 0.0)) for item in local_diagnostics),
            default=0.0,
        ),
        "num_selected_neumann_null_modes": int(
            sum(bool(mode.is_neumann_null_mode) for mode in all_modes)
        ),
        "local_spectra": local_diagnostics,
        "threshold_complete": bool(
            is_threshold
            and all(bool(item.get("threshold_bracketed", False)) for item in local_diagnostics)
        ),
        "compressed_dictionary_is_not_standard_geneo": not is_threshold,
        "compression_metric": "euclidean" if not is_threshold else "none_span_preserving",
        "pou_identity": "sum_j R_j^T D_j R_j = I from CCFVM_ASM.overlap_mask",
        "local_matrix": "face_assembled_neumann_with_explicit_physical_boundary_types",
        "residual_definition": "direct_original_pencil_relative_residual",
    }
    return basis, diagnostics


__all__ = [
    "LocalMode",
    "LocalSpectralGlobalData",
    "LocalSpectralTopologyCache",
    "LocalSubdomain",
    "LocalSubdomainTopology",
    "ThresholdBracketError",
    "assemble_local_neumann_matrix",
    "build_modified_geneo_coarse_space",
    "clear_local_spectral_topology_cache",
    "dense_modified_geneo_finite_spectrum",
    "get_local_spectral_topology_cache",
    "prepare_local_spectral_global_data",
    "solve_modified_geneo_local_spectrum",
]
