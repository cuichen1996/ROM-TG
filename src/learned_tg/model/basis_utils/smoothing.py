"""
Operator-agnostic basis smoothing.

Works with any operator that satisfies the SupportsApplyA protocol:
  - op.apply_A(X: (B,N,K)) -> (B,N,K)
  - op.diag_for_jacobi()   -> (B,N,1)
  - op.grid_shape()        -> (ny, nx)
  - op.asm_smooth(x4d, f4d, M_smooth, omega) -> 4d tensor
"""

import torch


def smooth_basis(op, V: torch.Tensor, mode: str, num_steps: int, omega: float) -> torch.Tensor:
    """
    Post-process a network-predicted basis V by applying num_steps smoothing
    sweeps using the given operator op.

    Parameters
    ----------
    op : SupportsApplyA
        FVM operator with apply_A, diag_for_jacobi, grid_shape, asm_smooth.
    V : Tensor, shape (B, N, m)
    mode : "none" | "jacobi" | "asm"
    num_steps : int
    omega : float

    Returns
    -------
    Tensor, shape (B, N, m)
    """
    V = torch.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0).to(
        device=op.device, dtype=op.dtype
    )
    mode = str(mode).lower()
    if mode == "none" or int(num_steps) <= 0:
        return V

    if V.dim() != 3:
        raise ValueError(f"V must have shape (B, N, m), got {tuple(V.shape)}")

    B, N, m = V.shape

    if mode == "jacobi":
        diag = op.diag_for_jacobi()  # (B, N, 1) or broadcastable
        V_smooth = V.clone()
        for _ in range(int(num_steps)):
            AV = op.apply_A(V_smooth)
            V_smooth = V_smooth - float(omega) * (AV / diag)
        return V_smooth

    if mode == "asm":
        ny, nx = op.grid_shape()
        V_img = V.transpose(1, 2).reshape(B, m, ny, nx)
        f_zero = torch.zeros_like(V_img)
        # asm_smooth signatures differ between operators; try both kwarg names.
        try:
            V_smooth_img = op.asm_smooth(V_img, f_zero, M_smooth=int(num_steps), omega=float(omega))
        except TypeError:
            V_smooth_img = op.asm_smooth(V_img, f_zero, steps=int(num_steps), omega=float(omega))
        # asm_smooth may return 4D or the original shape; normalise to (B, m, N).
        V_smooth_img = V_smooth_img.reshape(B, m, ny, nx)
        V_smooth = V_smooth_img.reshape(B, m, N).transpose(1, 2)
        return V_smooth

    raise ValueError(f"Unknown basis smoothing mode: {mode!r}")
