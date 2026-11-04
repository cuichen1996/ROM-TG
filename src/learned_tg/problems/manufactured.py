"""Explicitly assembled cell-centred FVM for jump-coefficient diffusion.

The matrix uses the scale of an integrated finite-volume balance: it does not
contain a factor ``1 / h**2``.  Consequently, sampled continuous source values
must be multiplied by ``h**2`` before solving.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ManufacturedCase:
    """Cell-centred samples for one manufactured interface problem."""

    name: str
    n: int
    h: float
    x: FloatArray
    y: FloatArray
    coefficient: FloatArray
    exact_solution: FloatArray
    rhs: FloatArray
    exact_source: FloatArray
    level_set: FloatArray


def make_grid(n: int) -> tuple[FloatArray, FloatArray, float]:
    """Return the two-dimensional cell-centre grid and mesh width."""

    if n < 2:
        raise ValueError("n must be at least 2")
    h = 1.0 / n
    coordinates = (np.arange(n, dtype=np.float64) + 0.5) * h
    x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
    return x, y, h


def manufactured_case(
    case: str,
    n: int,
    kappa: float = 10.0,
    radius: float = 0.23,
) -> ManufacturedCase:
    """Construct the aligned or circular manufactured interface problem."""

    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    if not 0.0 < radius < 0.5:
        raise ValueError("radius must lie in (0, 0.5)")

    x, y, h = make_grid(n)
    q = np.sin(math.pi * x) * np.sin(math.pi * y)
    qx = math.pi * np.cos(math.pi * x) * np.sin(math.pi * y)
    qy = math.pi * np.sin(math.pi * x) * np.cos(math.pi * y)

    if case == "aligned":
        if n % 2:
            raise ValueError("the aligned-interface case requires even n")
        level_set = x - 0.5
        phix = np.ones_like(x)
        phiy = np.zeros_like(x)
        laplacian_phi = np.zeros_like(x)
    elif case == "circle":
        level_set = (x - 0.5) ** 2 + (y - 0.5) ** 2 - radius**2
        phix = 2.0 * (x - 0.5)
        phiy = 2.0 * (y - 0.5)
        laplacian_phi = np.full_like(x, 4.0)
    else:
        raise ValueError(f"unknown case: {case!r}")

    coefficient = np.where(level_set < 0.0, kappa, 1.0)
    exact_solution = q * level_set / coefficient
    gradq_dot_gradphi = qx * phix + qy * phiy
    exact_source = (
        2.0 * math.pi**2 * level_set * q
        - 2.0 * gradq_dot_gradphi
        - q * laplacian_phi
    )
    rhs = h**2 * exact_source

    return ManufacturedCase(
        name=case,
        n=n,
        h=h,
        x=x,
        y=y,
        coefficient=coefficient,
        exact_solution=exact_solution,
        rhs=rhs,
        exact_source=exact_source,
        level_set=level_set,
    )


def _harmonic_mean(left: FloatArray, right: FloatArray) -> FloatArray:
    return 2.0 * left * right / (left + right)


def _face_transmissibility(
    a_left: FloatArray,
    a_right: FloatArray,
    phi_left: FloatArray | None = None,
    phi_right: FloatArray | None = None,
    mode: str = "harmonic",
) -> FloatArray:
    """Return positive transmissibilities for one set of interior faces.

    ``interface`` and ``interface_aware`` are aliases.  On a sign-changing
    face, they use the level-set intersection position along the line between
    the two cell centres.  Faces that do not cross the interface retain the
    ordinary harmonic mean.
    """

    left = np.asarray(a_left, dtype=np.float64)
    right = np.asarray(a_right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("left and right coefficient arrays must have the same shape")
    if (
        not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
        or np.any(left <= 0.0)
        or np.any(right <= 0.0)
    ):
        raise ValueError("face coefficients must be finite and positive")
    if mode not in {"harmonic", "interface", "interface_aware"}:
        raise ValueError(
            "transmissibility must be 'harmonic', 'interface', or "
            "'interface_aware'"
        )

    harmonic = _harmonic_mean(left, right)
    if mode == "harmonic":
        transmissibility = harmonic
    else:
        if phi_left is None or phi_right is None:
            raise ValueError(f"level-set values are required for mode={mode!r}")
        left_phi = np.asarray(phi_left, dtype=np.float64)
        right_phi = np.asarray(phi_right, dtype=np.float64)
        if left_phi.shape != left.shape or right_phi.shape != right.shape:
            raise ValueError(
                "left and right level-set arrays must match the coefficient shapes"
            )
        if not np.all(np.isfinite(left_phi)) or not np.all(np.isfinite(right_phi)):
            raise ValueError("face level-set values must be finite")

        crossing = (left_phi * right_phi) < 0.0
        eps = np.finfo(np.float64).eps
        distance_sum = np.maximum(np.abs(left_phi) + np.abs(right_phi), eps)
        theta = np.abs(left_phi) / distance_sum
        interface_denominator = theta / left + (1.0 - theta) / right
        interface_value = 1.0 / np.maximum(interface_denominator, eps)
        transmissibility = np.where(crossing, interface_value, harmonic)

    if not np.all(np.isfinite(transmissibility)) or np.any(transmissibility <= 0.0):
        raise RuntimeError("face transmissibilities must be finite and positive")
    return np.asarray(transmissibility, dtype=np.float64)


def assemble_ccfvm_matrix(
    coefficient: FloatArray,
    level_set: FloatArray | None = None,
    transmissibility: str = "harmonic",
) -> sparse.csr_matrix:
    """Explicitly assemble the unscaled five-point CCFVM matrix.

    Interior faces use either the ordinary harmonic mean or a level-set-aware
    intersection distance.  A homogeneous Dirichlet boundary face always
    contributes ``2*a`` because its cell-centre-to-boundary distance is
    ``h/2``; interface-aware corrections are not applied at the boundary.
    """

    coefficient = np.asarray(coefficient, dtype=np.float64)
    if coefficient.ndim != 2 or coefficient.shape[0] != coefficient.shape[1]:
        raise ValueError("coefficient must be a square two-dimensional array")
    if not np.all(np.isfinite(coefficient)) or np.any(coefficient <= 0.0):
        raise ValueError("coefficient entries must be finite and positive")
    if transmissibility not in {"harmonic", "interface", "interface_aware"}:
        raise ValueError(
            "transmissibility must be 'harmonic', 'interface', or "
            "'interface_aware'"
        )
    if level_set is None:
        if transmissibility != "harmonic":
            raise ValueError(
                f"level_set is required for transmissibility={transmissibility!r}"
            )
    else:
        level_set = np.asarray(level_set, dtype=np.float64)
        if level_set.shape != coefficient.shape:
            raise ValueError("level_set and coefficient must have the same shape")
        if not np.all(np.isfinite(level_set)):
            raise ValueError("level_set entries must be finite")

    n = coefficient.shape[0]
    cell = np.arange(n * n, dtype=np.int64).reshape(n, n)

    left = cell[:, :-1].ravel()
    right = cell[:, 1:].ravel()
    tx = _face_transmissibility(
        coefficient[:, :-1],
        coefficient[:, 1:],
        level_set[:, :-1] if level_set is not None else None,
        level_set[:, 1:] if level_set is not None else None,
        mode=transmissibility,
    ).ravel()

    lower = cell[:-1, :].ravel()
    upper = cell[1:, :].ravel()
    ty = _face_transmissibility(
        coefficient[:-1, :],
        coefficient[1:, :],
        level_set[:-1, :] if level_set is not None else None,
        level_set[1:, :] if level_set is not None else None,
        mode=transmissibility,
    ).ravel()

    boundary_diagonal = np.zeros(n * n, dtype=np.float64)
    boundary_diagonal[cell[:, 0]] += 2.0 * coefficient[:, 0]
    boundary_diagonal[cell[:, -1]] += 2.0 * coefficient[:, -1]
    boundary_diagonal[cell[0, :]] += 2.0 * coefficient[0, :]
    boundary_diagonal[cell[-1, :]] += 2.0 * coefficient[-1, :]
    boundary_cells = np.flatnonzero(boundary_diagonal)

    rows = np.concatenate(
        (left, right, left, right, lower, upper, lower, upper, boundary_cells)
    )
    cols = np.concatenate(
        (left, right, right, left, lower, upper, upper, lower, boundary_cells)
    )
    data = np.concatenate(
        (tx, tx, -tx, -tx, ty, ty, -ty, -ty, boundary_diagonal[boundary_cells])
    )
    matrix = sparse.coo_matrix(
        (data, (rows, cols)), shape=(n * n, n * n), dtype=np.float64
    ).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def solve_system(
    matrix: sparse.spmatrix,
    rhs: FloatArray,
) -> tuple[FloatArray, float, float]:
    """Solve the assembled sparse system directly with SuperLU.

    Returns ``(solution, relative_residual, solve_seconds)``.  The minimum
    degree ordering limits fill-in for these symmetric five-point matrices.
    """

    rhs_vector = np.asarray(rhs, dtype=np.float64).reshape(-1)
    matrix_csc = sparse.csc_matrix(matrix)
    start = time.perf_counter()
    solution = sparse_linalg.spsolve(
        matrix_csc,
        rhs_vector,
        permc_spec="MMD_AT_PLUS_A",
        use_umfpack=False,
    )
    solve_seconds = time.perf_counter() - start

    if not np.all(np.isfinite(solution)):
        raise RuntimeError("the sparse direct solver returned non-finite values")
    residual = matrix_csc @ solution - rhs_vector
    rhs_norm = np.linalg.norm(rhs_vector)
    relative_residual = np.linalg.norm(residual) / max(rhs_norm, np.finfo(float).tiny)
    return solution, float(relative_residual), float(solve_seconds)


def compute_errors(
    matrix: sparse.spmatrix,
    numerical_solution: FloatArray,
    exact_solution: FloatArray,
    h: float,
) -> dict[str, float]:
    """Compute discrete L2, Linf, and finite-volume energy errors."""

    error = np.asarray(numerical_solution).reshape(-1) - np.asarray(
        exact_solution
    ).reshape(-1)
    matrix_error = matrix @ error
    energy_squared = float(error @ matrix_error)
    roundoff_scale = np.finfo(float).eps * max(float(np.linalg.norm(error) ** 2), 1.0)
    if energy_squared < -100.0 * roundoff_scale:
        raise RuntimeError(f"negative discrete energy: {energy_squared:.6e}")
    return {
        "L2": float(h * np.linalg.norm(error)),
        "Linf": float(np.max(np.abs(error))),
        "A": float(math.sqrt(max(energy_squared, 0.0))),
    }


def observed_orders(errors: Iterable[float]) -> list[float]:
    """Return base-two rates, with NaN for the first grid."""

    values = [float(value) for value in errors]
    orders = [math.nan]
    for old, new in zip(values, values[1:]):
        orders.append(math.log(old / new) / math.log(2.0))
    return orders


def run_convergence_study(
    n_values: Iterable[int],
    cases: Iterable[str] = ("aligned", "circle"),
    kappa: float = 10.0,
    radius: float = 0.23,
    transmissibility: str = "harmonic",
    keep_error_fields_at: Iterable[int] = (),
) -> tuple[list[dict[str, float | int | str]], dict[tuple[str, int], FloatArray]]:
    """Run all requested grids with explicit assembly and direct solves."""

    grids = [int(n) for n in n_values]
    if not grids or grids != sorted(set(grids)):
        raise ValueError("n_values must be a non-empty, strictly increasing sequence")
    kept_grids = set(int(n) for n in keep_error_fields_at)

    rows: list[dict[str, float | int | str]] = []
    error_fields: dict[tuple[str, int], FloatArray] = {}
    for case_name in cases:
        case_rows: list[dict[str, float | int | str]] = []
        for n in grids:
            problem = manufactured_case(case_name, n, kappa=kappa, radius=radius)
            assembly_start = time.perf_counter()
            matrix = assemble_ccfvm_matrix(
                problem.coefficient,
                level_set=problem.level_set,
                transmissibility=transmissibility,
            )
            assembly_seconds = time.perf_counter() - assembly_start
            solution, relative_residual, solve_seconds = solve_system(matrix, problem.rhs)
            if relative_residual > 1.0e-11:
                raise RuntimeError(
                    f"relative residual {relative_residual:.3e} exceeds 1e-11 "
                    f"for case={case_name}, n={n}"
                )
            errors = compute_errors(matrix, solution, problem.exact_solution, problem.h)
            row: dict[str, float | int | str] = {
                "case": case_name,
                "transmissibility": transmissibility,
                "n": n,
                "h": problem.h,
                "unknowns": n * n,
                "nnz": matrix.nnz,
                "L2_error": errors["L2"],
                "Linf_error": errors["Linf"],
                "A_error": errors["A"],
                "relative_residual": relative_residual,
                "assembly_seconds": assembly_seconds,
                "solve_seconds": solve_seconds,
            }
            case_rows.append(row)
            if n in kept_grids:
                error_fields[(case_name, n)] = (
                    solution.reshape(n, n) - problem.exact_solution
                )

        for metric in ("L2", "Linf", "A"):
            rates = observed_orders(row[f"{metric}_error"] for row in case_rows)
            for row, rate in zip(case_rows, rates):
                row[f"{metric}_order"] = rate
        rows.extend(case_rows)

    return rows, error_fields
