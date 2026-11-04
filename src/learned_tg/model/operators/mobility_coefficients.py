"""
Shared coefficient-builder for the two-phase cell-centered pressure FVM
operator.  Used by both MobilityFVMOperatorASM (model/two_phase.py) and
_LightweightMobilityFVMOperator (data/gen_data_two_phase.py).
"""

import torch


def harmonic_mean(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:
    return 2.0 * a * b / (a + b + eps)


def _normalize_phi(phi: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    if phi.dim() == 2:
        phi = phi[None, None, :, :]
    elif phi.dim() == 3:
        phi = phi[:, None, :, :]
    elif phi.dim() != 4:
        raise ValueError(
            "phi must have dim 2, 3, or 4, "
            f"got shape={tuple(phi.shape)}"
        )
    if tuple(phi.shape[-2:]) != tuple(lam.shape[-2:]):
        raise ValueError(
            "phi spatial shape must match lam: "
            f"phi={tuple(phi.shape)}, lam={tuple(lam.shape)}"
        )
    if phi.shape[0] == 1 and lam.shape[0] > 1:
        phi = phi.expand(lam.shape[0], -1, -1, -1)
    elif phi.shape[0] != lam.shape[0]:
        raise ValueError(
            "phi batch size must match lam: "
            f"phi={phi.shape[0]}, lam={lam.shape[0]}"
        )
    return phi.to(device=lam.device, dtype=lam.dtype)


def face_transmissibility(
    a0: torch.Tensor,
    a1: torch.Tensor,
    phi0: torch.Tensor = None,
    phi1: torch.Tensor = None,
    mode: str = "harmonic",
) -> torch.Tensor:
    """Return harmonic or level-set interface-aware face mobility."""
    mode = str(mode).lower()
    if mode == "interface":
        mode = "interface_aware"
    if mode not in {"harmonic", "interface_aware"}:
        raise ValueError(
            "transmissibility must be 'harmonic', 'interface', or "
            f"'interface_aware', got {mode!r}"
        )

    harmonic = harmonic_mean(a0, a1)
    if mode == "harmonic":
        return harmonic
    if phi0 is None or phi1 is None:
        raise ValueError("phi is required for interface-aware transmissibility")

    crossing = phi0 * phi1 < 0
    abs0 = phi0.abs()
    abs1 = phi1.abs()
    theta = abs0 / (abs0 + abs1).clamp_min(torch.finfo(a0.dtype).tiny)
    interface_value = 1.0 / (
        theta / a0.clamp_min(torch.finfo(a0.dtype).tiny)
        + (1.0 - theta) / a1.clamp_min(torch.finfo(a1.dtype).tiny)
    )
    return torch.where(crossing, interface_value, harmonic)


def build_mobility_coefficients(
    lam,
    hx,
    hy,
    p_left=1.0,
    p_right=0.0,
    *,
    phi=None,
    transmissibility="harmonic",
):
    """
    Compute FVM stencil coefficients for  -div(lam grad p) = q  on a
    rectangular grid with Dirichlet BCs p=p_left at x=0, p=p_right at x=Lx,
    and zero-flux Neumann at y=0,Ly.

    Parameters
    ----------
    lam : Tensor, shape (B, 1, ny, nx), dtype float64
    hx, hy : cell widths
    p_left, p_right : Dirichlet boundary values

    Returns
    -------
    dict with keys: aw, ae, as_, an, diag, bc_rhs, lam_x, lam_y
    """
    hx2 = float(hx) ** 2
    hy2 = float(hy) ** 2

    aw = torch.zeros_like(lam)
    ae = torch.zeros_like(lam)
    as_ = torch.zeros_like(lam)
    an = torch.zeros_like(lam)

    mode = str(transmissibility).lower()
    if mode == "interface":
        mode = "interface_aware"
    phi4 = None
    if mode != "harmonic":
        if phi is None:
            raise ValueError("phi is required for interface-aware transmissibility")
        phi4 = _normalize_phi(phi, lam)

    lam_x = face_transmissibility(
        lam[..., :, :-1],
        lam[..., :, 1:],
        None if phi4 is None else phi4[..., :, :-1],
        None if phi4 is None else phi4[..., :, 1:],
        mode=mode,
    )
    ae[..., :, :-1] = lam_x / hx2
    aw[..., :, 1:] = lam_x / hx2

    lam_y = face_transmissibility(
        lam[..., :-1, :],
        lam[..., 1:, :],
        None if phi4 is None else phi4[..., :-1, :],
        None if phi4 is None else phi4[..., 1:, :],
        mode=mode,
    )
    an[..., :-1, :] = lam_y / hy2
    as_[..., 1:, :] = lam_y / hy2

    diag = aw + ae + as_ + an
    bc_rhs = torch.zeros_like(lam)

    c_left = 2.0 * lam[..., :, 0] / hx2
    c_right = 2.0 * lam[..., :, -1] / hx2
    diag[..., :, 0] += c_left
    diag[..., :, -1] += c_right
    bc_rhs[..., :, 0] += c_left * float(p_left)
    bc_rhs[..., :, -1] += c_right * float(p_right)

    return dict(aw=aw, ae=ae, as_=as_, an=an, diag=diag,
                bc_rhs=bc_rhs, lam_x=lam_x, lam_y=lam_y)
