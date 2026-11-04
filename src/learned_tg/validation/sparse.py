"""Numerical routines used by the paper; no experiment orchestration."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp

def assemble_fvm_matrix(asm) -> sp.csc_matrix:
    """Assemble the exact sparse matrix represented by ``CCFVM_ASM.forward``."""
    n = int(asm.n)
    size = n * n
    grid = np.arange(size, dtype=np.int64).reshape(n, n)

    diagonal = asm.sc.detach().cpu().numpy().reshape(-1)
    west = asm.sw.detach().cpu().numpy().reshape(n, n)[:, 1:].reshape(-1)
    east = asm.se.detach().cpu().numpy().reshape(n, n)[:, :-1].reshape(-1)
    north = asm.sn.detach().cpu().numpy().reshape(n, n)[1:, :].reshape(-1)
    south = asm.ss.detach().cpu().numpy().reshape(n, n)[:-1, :].reshape(-1)

    rows = np.concatenate(
        (
            np.arange(size, dtype=np.int64),
            grid[:, 1:].reshape(-1),
            grid[:, :-1].reshape(-1),
            grid[1:, :].reshape(-1),
            grid[:-1, :].reshape(-1),
        )
    )
    columns = np.concatenate(
        (
            np.arange(size, dtype=np.int64),
            grid[:, :-1].reshape(-1),
            grid[:, 1:].reshape(-1),
            grid[:-1, :].reshape(-1),
            grid[1:, :].reshape(-1),
        )
    )
    values = np.concatenate((diagonal, west, east, north, south))
    return sp.coo_matrix(
        (values, (rows, columns)),
        shape=(size, size),
        dtype=np.float64,
    ).tocsc()
