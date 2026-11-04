"""Numerical routines used by the paper; no experiment orchestration."""
from __future__ import annotations
import math
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from learned_tg.model.interface import CCFVM_ASM

def smooth_heaviside(phi: torch.Tensor, eps: float) -> torch.Tensor:
    eps = float(eps)
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    H = torch.zeros_like(phi)
    H = torch.where(phi > eps, torch.ones_like(H), H)
    H = torch.where(phi < -eps, torch.zeros_like(H), H)
    mask = torch.abs(phi) <= eps
    s = phi / eps
    H_mid = 0.5 * (1.0 + s + torch.sin(math.pi * s) / math.pi)
    return torch.where(mask, H_mid, H)

def _ensure_4d_field(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dim() == 2:
        tensor = tensor[None, None, :, :]
    elif tensor.dim() == 3:
        tensor = tensor[:, None, :, :]
    elif tensor.dim() != 4:
        raise ValueError(
            f"{name} must have shape (n,n), (B,n,n), or (B,1,n,n); got {tuple(tensor.shape)}"
        )
    if tensor.shape[1] != 1:
        raise ValueError(f"{name} must have one channel; got {tuple(tensor.shape)}")
    return tensor.contiguous()

def _smooth_interface_coefficient_from_phi(
    phi: torch.Tensor,
    a_hard: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    phi = _ensure_4d_field(phi, "phi")
    a_hard = _ensure_4d_field(a_hard, "a_hard")
    if phi.shape != a_hard.shape:
        raise ValueError(f"phi/a_hard shape mismatch: {tuple(phi.shape)} vs {tuple(a_hard.shape)}")

    H = smooth_heaviside(phi, eps=eps)
    result = torch.empty_like(a_hard)
    for b in range(phi.shape[0]):
        inside = phi[b : b + 1] < 0.0
        outside = ~inside
        if not bool(inside.any().item()) or not bool(outside.any().item()):
            warnings.warn(
                f"sample {b}: phi has an empty inside/outside region; using hard coefficient means"
            )
        inside_values = a_hard[b : b + 1][inside]
        outside_values = a_hard[b : b + 1][outside]
        a_inside = inside_values.mean() if inside_values.numel() else a_hard[b : b + 1].mean()
        a_outside = outside_values.mean() if outside_values.numel() else a_hard[b : b + 1].mean()
        result[b : b + 1] = a_inside * (1.0 - H[b : b + 1]) + a_outside * H[b : b + 1]
    return result

def _apply_A_interface(op_solver: CCFVM_ASM, X: torch.Tensor) -> torch.Tensor:
    if X.dim() != 3:
        raise ValueError(f"interface A apply expects X shape (B,N,K), got {tuple(X.shape)}")
    B, N, K = X.shape
    n = int(round(math.sqrt(N)))
    if n * n != N:
        raise ValueError(f"interface A apply expected square node count, got N={N}")
    X_img = X.transpose(1, 2).reshape(B, K, n, n)
    AX_img = op_solver.forward(X_img)
    return torch.nan_to_num(
        AX_img.reshape(B, K, N).transpose(1, 2).contiguous(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

@torch.no_grad()
def build_oracle_basis(
    E: torch.Tensor,
    op_apply_A: Callable[[torch.Tensor], torch.Tensor],
    basis_type: str,
    rank: int,
    reg: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Build a per-sample oracle L2-POD or A-POD basis.

    Parameters
    ----------
    E:
        Snapshot tensor with shape (B, N, q).
    op_apply_A:
        Callable that maps (B, N, k) to A @ X with the same shape.
    basis_type:
        Either "l2_pod" or "a_pod".
    rank:
        Requested basis rank.  It is clipped to min(N, q).
    reg:
        Eigenvalue and coarse regularization floor.
    """
    if E.dim() != 3:
        raise ValueError(f"E must have shape (B,N,q), got {tuple(E.shape)}")
    B, N, q = E.shape
    rank_eff = max(1, min(int(rank), int(N), int(q)))
    E = torch.nan_to_num(E.to(torch.float64), nan=0.0, posinf=0.0, neginf=0.0)
    basis_type = str(basis_type).lower()

    if basis_type == "l2_pod":
        U, S, _ = torch.linalg.svd(E, full_matrices=False)
        Q = U[:, :, :rank_eff].contiguous()
        AQ = op_apply_A(Q)
        energy = (S * S).clamp_min(0.0)
        total = energy.sum(dim=-1).clamp_min(float(reg))
        capture = energy[:, :rank_eff].sum(dim=-1) / total
        q_a_orth_error = _q_a_orthogonality_error(Q, AQ)
        return Q, AQ, {
            "basis_type": "l2_pod",
            "rank_eff": int(rank_eff),
            "singular_values_l2": S.detach().cpu(),
            "l2_capture_ratio_rank_m": capture.detach().cpu(),
            "l2_tail_energy_ratio": (1.0 - capture).detach().cpu(),
            "capture_ratio_rank_m": capture.detach().cpu(),
            "tail_energy_ratio": (1.0 - capture).detach().cpu(),
            "q_a_orth_error": q_a_orth_error.detach().cpu(),
        }

    if basis_type != "a_pod":
        raise ValueError(f"basis_type must be 'l2_pod' or 'a_pod', got {basis_type!r}")

    AE = op_apply_A(E)
    G = torch.matmul(E.transpose(1, 2), AE)
    G = 0.5 * (G + G.transpose(1, 2))
    eigvals, eigvecs = torch.linalg.eigh(G)
    order = torch.argsort(eigvals, dim=-1, descending=True)
    eigvals_desc = torch.gather(eigvals, dim=-1, index=order)
    eigvecs_desc = torch.gather(
        eigvecs,
        dim=-1,
        index=order.unsqueeze(1).expand(B, q, q),
    )

    min_eig = eigvals.amin(dim=-1)
    max_eig = eigvals.abs().amax(dim=-1).clamp_min(float(reg))
    large_negative = bool((min_eig < -1e-8 * max_eig).any().item())
    if large_negative:
        warnings.warn(
            "A-POD snapshot Gram has a negative eigenvalue beyond numerical tolerance: "
            f"min={min_eig.detach().cpu().tolist()}"
        )

    eigvals_m = eigvals_desc[:, :rank_eff].clamp_min(float(reg))
    W_m = eigvecs_desc[:, :, :rank_eff]
    Q = torch.matmul(E, W_m) * eigvals_m.rsqrt().unsqueeze(1)
    Q = torch.nan_to_num(Q.contiguous(), nan=0.0, posinf=0.0, neginf=0.0)
    AQ = op_apply_A(Q)
    q_a_orth_error = _q_a_orthogonality_error(Q, AQ)

    eigvals_pos = eigvals_desc.clamp_min(0.0)
    total = eigvals_pos.sum(dim=-1).clamp_min(float(reg))
    capture = eigvals_pos[:, :rank_eff].sum(dim=-1) / total
    return Q, AQ, {
        "basis_type": "a_pod",
        "rank_eff": int(rank_eff),
        "energy_eigenvalues": eigvals_desc.detach().cpu(),
        "a_capture_ratio_rank_m": capture.detach().cpu(),
        "a_tail_energy_ratio": (1.0 - capture).detach().cpu(),
        "capture_ratio_rank_m": capture.detach().cpu(),
        "tail_energy_ratio": (1.0 - capture).detach().cpu(),
        "q_a_orth_error": q_a_orth_error.detach().cpu(),
        "min_energy_eigenvalue": min_eig.detach().cpu(),
        "large_negative_energy_eigenvalue": large_negative,
    }

def _q_a_orthogonality_error(Q: torch.Tensor, AQ: torch.Tensor) -> torch.Tensor:
    if Q.dim() != 3 or AQ.shape != Q.shape:
        raise ValueError(f"Q/AQ shape mismatch: {tuple(Q.shape)} vs {tuple(AQ.shape)}")
    basis_dim = int(Q.shape[-1])
    eye = torch.eye(basis_dim, device=Q.device, dtype=Q.dtype).unsqueeze(0)
    gram = torch.matmul(Q.transpose(1, 2), AQ)
    gram = 0.5 * (gram + gram.transpose(1, 2))
    return torch.linalg.matrix_norm(gram - eye, ord="fro", dim=(-2, -1))

def _make_prepared_coarse_basis(
    Q: torch.Tensor,
    AQ: torch.Tensor,
    reg: float = 1e-10,
    coarse_basis_mode: str = "raw",
    timing_diagnostics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    if Q.dim() != 3 or AQ.shape != Q.shape:
        raise ValueError(f"Q/AQ must both have shape (B,N,m), got {tuple(Q.shape)} and {tuple(AQ.shape)}")
    B, N, m = Q.shape

    def sync() -> None:
        if Q.is_cuda:
            torch.cuda.synchronize(Q.device)

    sync()
    prepare_started = time.perf_counter()
    galerkin_started = prepare_started
    A_rom = torch.matmul(Q.transpose(1, 2), AQ)
    A_rom = 0.5 * (A_rom + A_rom.transpose(1, 2))
    sync()
    galerkin_time = time.perf_counter() - galerkin_started
    a_orthonormalization_time = 0.0
    input_A_numerical_rank = int(m)
    input_A_rank_floor = 0.0
    input_A_eigenvalues: List[float] = []
    mode = str(coarse_basis_mode).lower().replace("-", "_")
    if mode in {"a_orthonormal", "a_orth", "orthonormal"}:
        orth_started = time.perf_counter()
        eigvals, eigvecs = torch.linalg.eigh(A_rom)
        order = torch.argsort(eigvals, dim=-1, descending=True)
        eigvals = torch.gather(eigvals, dim=-1, index=order)
        eigvecs = torch.gather(
            eigvecs,
            dim=-1,
            index=order.unsqueeze(1).expand(B, m, m),
        )
        input_A_eigenvalues = [
            float(value)
            for value in eigvals.detach().cpu().reshape(-1).tolist()
        ]
        input_A_max = max((abs(value) for value in input_A_eigenvalues), default=0.0)
        input_A_rank_floor = float(
            torch.finfo(Q.dtype).eps * max(m, 1) * max(input_A_max, torch.finfo(Q.dtype).tiny)
        )
        input_A_numerical_rank = int(
            sum(value > input_A_rank_floor for value in input_A_eigenvalues)
        )
        min_eig = eigvals.amin(dim=-1)
        max_eig = eigvals.abs().amax(dim=-1).clamp_min(float(reg))
        if bool((min_eig < -1e-8 * max_eig).any().item()):
            warnings.warn(
                "coarse-basis A-orthonormalization saw a negative eigenvalue: "
                f"min={min_eig.detach().cpu().tolist()}"
            )
        scale = eigvals.clamp_min(float(reg)).rsqrt().unsqueeze(1)
        Q = torch.matmul(Q, eigvecs) * scale
        AQ = torch.matmul(AQ, eigvecs) * scale
        Q = torch.nan_to_num(Q.contiguous(), nan=0.0, posinf=0.0, neginf=0.0)
        AQ = torch.nan_to_num(AQ.contiguous(), nan=0.0, posinf=0.0, neginf=0.0)
        sync()
        a_orthonormalization_time = time.perf_counter() - orth_started
        galerkin_started = time.perf_counter()
        A_rom = torch.matmul(Q.transpose(1, 2), AQ)
        A_rom = 0.5 * (A_rom + A_rom.transpose(1, 2))
        sync()
        galerkin_time += time.perf_counter() - galerkin_started
    elif mode not in {"raw", "galerkin", "raw_galerkin"}:
        raise ValueError(
            "coarse_basis_mode must be one of {'raw', 'a_orthonormal'}, "
            f"got {coarse_basis_mode!r}"
        )
    galerkin_started = time.perf_counter()
    eye = torch.eye(m, device=Q.device, dtype=Q.dtype).unsqueeze(0)
    A_rom_reg = A_rom + float(reg) * eye
    coarse_condition_number = torch.linalg.cond(A_rom_reg).detach().cpu()
    try:
        coarse_factor = torch.linalg.cholesky(A_rom_reg)
        coarse_solver = "cholesky"
    except RuntimeError:
        coarse_factor = torch.linalg.pinv(A_rom_reg)
        coarse_solver = "pinv"
    sync()
    galerkin_time += time.perf_counter() - galerkin_started
    if timing_diagnostics is not None:
        timing_diagnostics.update(
            {
                "global_A_orthonormalization_time": float(a_orthonormalization_time),
                "Galerkin_assembly_time": float(galerkin_time),
                "total_prepare_coarse_basis_time": float(
                    time.perf_counter() - prepare_started
                ),
                "input_A_numerical_rank": int(input_A_numerical_rank),
                "input_A_rank_floor": float(input_A_rank_floor),
                "input_A_eigenvalues": input_A_eigenvalues,
            }
        )
    return {
        "Q": Q.contiguous(),
        "AQ": AQ.contiguous(),
        "A_rom": A_rom.contiguous(),
        "A_rom_reg": A_rom_reg.contiguous(),
        "coarse_solver": coarse_solver,
        "coarse_factor": coarse_factor.contiguous(),
        "coarse_condition_number": coarse_condition_number,
        "basis_dim": int(m),
        "num_nodes": int(N),
        "batch_size": int(B),
        "orth_reg": float(reg),
        "basis_smooth_mode": "none",
        "basis_smooth_steps": 0,
        "basis_smooth_omega": 0.0,
        "coarse_basis_mode": mode,
        "orthonormalize_metric": "a" if mode in {"a_orthonormal", "a_orth", "orthonormal"} else "raw",
    }
