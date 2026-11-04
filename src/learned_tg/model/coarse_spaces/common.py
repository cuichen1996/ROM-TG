"""Common sparse coarse-space utilities.

The routines here are intentionally NumPy/SciPy based.  They are used by
baseline experiments that compare hand-built coarse spaces without training a
network and without densifying the large stiffness matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class AOrthonormalizationDiagnostics:
    """Diagnostics returned by :func:`a_orthonormalize`."""

    input_dim: int
    rank: int
    dropped_modes: int
    condition_number: float
    eig_threshold: float
    min_eig: float
    max_eig: float
    num_nonpositive: int
    eigvals: Tuple[float, ...]
    kept_eigvals: Tuple[float, ...]
    gram_error: float

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "input_dim": int(self.input_dim),
            "rank": int(self.rank),
            "dropped_modes": int(self.dropped_modes),
            "condition_number": float(self.condition_number),
            "eig_threshold": float(self.eig_threshold),
            "min_eig": float(self.min_eig),
            "max_eig": float(self.max_eig),
            "num_nonpositive": int(self.num_nonpositive),
            "eigvals": [float(v) for v in self.eigvals],
            "kept_eigvals": [float(v) for v in self.kept_eigvals],
            "gram_error": float(self.gram_error),
        }


def _as_2d_float_array(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype=np.float64)
    if V.ndim != 2:
        raise ValueError(f"V must have shape (N, m), got {V.shape}")
    return np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)


def _matmat(A: Any, X: np.ndarray) -> np.ndarray:
    """Apply ``A`` to dense columns ``X`` without densifying sparse matrices."""
    if sp.issparse(A):
        return np.asarray(A @ X, dtype=np.float64)
    if hasattr(A, "matmat"):
        return np.asarray(A.matmat(X), dtype=np.float64)
    if hasattr(A, "dot"):
        return np.asarray(A.dot(X), dtype=np.float64)
    return np.asarray(A @ X, dtype=np.float64)


def coarse_gram(V: np.ndarray, A: Any) -> np.ndarray:
    """Return the small dense Gram matrix ``V.T @ A @ V``."""
    V = _as_2d_float_array(V)
    AV = _matmat(A, V)
    G = V.T @ AV
    return 0.5 * (G + G.T)


def a_orthonormalize(
    V: np.ndarray,
    A: Any,
    tol: float = 1e-10,
    jitter: float = 1e-12,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """A-orthonormalize candidate basis columns.

    Parameters
    ----------
    V:
        Dense candidate basis with shape ``(N, m)``.
    A:
        SPD matrix or matrix-like object.  SciPy sparse matrices are supported
        directly; ``A`` is never converted to dense.
    tol:
        Relative eigenvalue threshold used to discard linearly dependent or
        nearly dependent directions in the small Gram matrix.
    jitter:
        Small absolute floor used only in the eigenvalue threshold.  It is not
        added to the large sparse matrix and does not keep zero modes alive.

    Returns
    -------
    V_orth, diagnostics:
        ``V_orth`` has shape ``(N, rank)`` and satisfies
        ``V_orth.T @ A @ V_orth ~= I``.  The diagnostics include the numerical
        rank, condition number, discarded modes, and Gram spectrum.
    """
    V = _as_2d_float_array(V)
    num_nodes, input_dim = V.shape
    if input_dim == 0:
        diagnostics = AOrthonormalizationDiagnostics(
            input_dim=0,
            rank=0,
            dropped_modes=0,
            condition_number=float("inf"),
            eig_threshold=float(max(tol, jitter)),
            min_eig=0.0,
            max_eig=0.0,
            num_nonpositive=0,
            eigvals=(),
            kept_eigvals=(),
            gram_error=0.0,
        ).to_dict()
        return np.zeros((num_nodes, 0), dtype=np.float64), diagnostics

    G = coarse_gram(V, A)
    try:
        eigvals, eigvecs = np.linalg.eigh(G)
    except np.linalg.LinAlgError:
        G = G + float(jitter) * np.eye(input_dim, dtype=np.float64)
        eigvals, eigvecs = np.linalg.eigh(G)

    eigvals = np.asarray(eigvals, dtype=np.float64)
    scale = max(float(np.max(np.abs(eigvals))) if eigvals.size else 0.0, 1.0)
    eig_threshold = max(float(jitter), float(tol) * scale)
    keep = eigvals > eig_threshold
    kept_eigvals_asc = eigvals[keep]

    if kept_eigvals_asc.size == 0:
        diagnostics = AOrthonormalizationDiagnostics(
            input_dim=input_dim,
            rank=0,
            dropped_modes=input_dim,
            condition_number=float("inf"),
            eig_threshold=eig_threshold,
            min_eig=float(np.min(eigvals)),
            max_eig=float(np.max(eigvals)),
            num_nonpositive=int(np.sum(eigvals <= eig_threshold)),
            eigvals=tuple(float(v) for v in eigvals),
            kept_eigvals=(),
            gram_error=0.0,
        ).to_dict()
        return np.zeros((num_nodes, 0), dtype=np.float64), diagnostics

    # Order columns by descending retained Gram energy for deterministic
    # truncation in hybrid spaces.
    keep_indices = np.flatnonzero(keep)
    order = np.argsort(eigvals[keep_indices])[::-1]
    keep_indices = keep_indices[order]
    kept_eigvals = eigvals[keep_indices]
    transform = eigvecs[:, keep_indices] @ np.diag(1.0 / np.sqrt(kept_eigvals))
    V_orth = V @ transform
    V_orth = np.asarray(V_orth, dtype=np.float64)

    G_orth = coarse_gram(V_orth, A)
    eye = np.eye(V_orth.shape[1], dtype=np.float64)
    gram_error = float(np.linalg.norm(G_orth - eye, ord=np.inf)) if V_orth.shape[1] else 0.0
    condition_number = (
        float(np.max(kept_eigvals) / np.min(kept_eigvals))
        if kept_eigvals.size
        else float("inf")
    )
    diagnostics = AOrthonormalizationDiagnostics(
        input_dim=input_dim,
        rank=int(V_orth.shape[1]),
        dropped_modes=int(input_dim - V_orth.shape[1]),
        condition_number=condition_number,
        eig_threshold=eig_threshold,
        min_eig=float(np.min(eigvals)),
        max_eig=float(np.max(eigvals)),
        num_nonpositive=int(np.sum(eigvals <= eig_threshold)),
        eigvals=tuple(float(v) for v in eigvals),
        kept_eigvals=tuple(float(v) for v in kept_eigvals),
        gram_error=gram_error,
    ).to_dict()
    return V_orth, diagnostics


def unpack_builder_result(result: Any) -> tuple[np.ndarray, Dict[str, Any]]:
    """Normalise builder outputs to ``(V, diagnostics)``."""
    if isinstance(result, tuple) and len(result) == 2:
        V, diagnostics = result
        return np.asarray(V, dtype=np.float64), dict(diagnostics)
    return np.asarray(result, dtype=np.float64), {}
