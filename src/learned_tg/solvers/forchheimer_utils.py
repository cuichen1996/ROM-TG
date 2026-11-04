"""Numerical routines used by the paper; no experiment orchestration."""
from __future__ import annotations
import time
from typing import Any, Dict, List, Tuple
import torch
from learned_tg.data.channelized import generate_channelized_fields
from learned_tg.model.interface import JumpingCoefficient, Solver

def _coords(n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h = 1.0 / float(n)
    x1d = torch.linspace(h / 2.0, 1.0 - h / 2.0, int(n), device=device)
    yy, xx = torch.meshgrid(x1d, x1d, indexing="ij")
    coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    x_net = coords[:, 0:1].unsqueeze(0).to(torch.float32)
    y_net = coords[:, 1:2].unsqueeze(0).to(torch.float32)
    return coords, x_net, y_net

def _rhs(n: int, source: float, device: torch.device) -> torch.Tensor:
    h = 1.0 / float(n)
    return torch.full((1, 1, n, n), float(source) * h * h, dtype=torch.float64, device=device)

def _condition_number(matrix: torch.Tensor) -> float:
    eigvals = torch.linalg.eigvalsh(0.5 * (matrix + matrix.transpose(1, 2)))
    amin = eigvals.abs().amin(dim=-1).clamp_min(1e-300)
    amax = eigvals.abs().amax(dim=-1)
    return float((amax / amin).mean().detach().cpu().item())

def _make_operator(
    coeff: torch.Tensor,
    phi: torch.Tensor,
    *,
    n: int,
    k_1d: int,
    config: Dict[str, Any],
) -> JumpingCoefficient:
    return JumpingCoefficient(
        coeff,
        n=int(n),
        k_1d=int(k_1d),
        coarse_space_append_island_basis=bool(config.get("coarse_space_append_island_basis", False)),
        coarse_space_island_basis_threshold=float(config.get("coarse_space_island_basis_threshold", 500.0)),
        subdomain_solver=str(config.get("subdomain_solver", "cholesky")),
        jacobi_pcg_iters=int(config.get("jacobi_pcg_iters", 10)),
        vcycle_coarsest_threshold=int(config.get("vcycle_coarsest_threshold", 24)),
        vcycle_smooth_steps=int(config.get("vcycle_smooth_steps", 2)),
        vcycle_iter=int(config.get("vcycle_iter", 2)),
        vcycle_omega=float(config.get("vcycle_omega", 2.0 / 3.0)),
        vcycle_compile=bool(config.get("vcycle_compile", True)),
        phi=phi,
        transmissibility=str(config.get("transmissibility", "harmonic")),
    )

def _predict_raw_basis(
    model: Solver,
    coeff: torch.Tensor,
    phi: torch.Tensor,
    x_net: torch.Tensor,
    y_net: torch.Tensor,
    *,
    m_dim: int,
) -> Tuple[torch.Tensor, float]:
    tic = time.perf_counter()
    with torch.no_grad():
        V = model.predict_basis(
            coeff.float(),
            phi.float(),
            x_net,
            y_net,
            active_dim=int(m_dim),
        ).to(torch.float64)
    return V, time.perf_counter() - tic

def _prepare_raw_basis(
    op: JumpingCoefficient,
    raw_basis: torch.Tensor,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    with torch.no_grad():
        return op.prepare_coarse_basis(
            raw_basis,
            basis_smooth_mode=str(config.get("basis_smooth_mode_solve", "asm")),
            basis_smooth_steps=int(config.get("basis_smooth_steps_solve", 0)),
            basis_smooth_omega=float(config.get("basis_smooth_omega_solve", 0.6)),
            orthonormalize_metric=str(config.get("coarse_basis_orth_metric", "a")),
            orth_reg=float(config.get("basis_orth_reg", config.get("loss_reg", 1e-8))),
        )

def _sample_connected_channelized(
    *,
    n: int,
    samples: int,
    contrast: float,
    config: Dict[str, Any],
    seed: int,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
    out: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]] = []
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    attempts = 0
    max_attempts = max(200, int(samples) * 200)
    while len(out) < int(samples) and attempts < max_attempts:
        attempts += 1
        phi, a_field, meta_list = generate_channelized_fields(
            batch_size=1,
            n=int(n),
            contrast=float(contrast),
            max_horizontal=int(config.get("channel_max_horizontal", 5)),
            max_vertical=int(config.get("channel_max_vertical", 5)),
            width_range=config.get("channel_width_range", (0.012, 0.035)),
            min_width_cells=int(config.get("channel_min_width_cells", 2)),
            channel_min_width_cells=config.get("channel_min_width_cells", 2),
            channel_max_width_cells=config.get("channel_max_width_cells", 3),
            channel_touch_boundary=bool(config.get("channel_touch_boundary", False)),
            channel_boundary_margin=float(config.get("channel_boundary_margin", 0.05)),
            channel_style=str(config.get("channel_style", "paper_internal_thin")),
            allow_disconnected=False,
            device="cpu",
            dtype=torch.float64,
            generator=generator,
        )
        meta = dict(meta_list[0])
        if bool(meta.get("connected", False)):
            meta["attempts_before_accept"] = int(attempts)
            out.append((phi.to(device), a_field.to(device), meta))
    if len(out) < int(samples):
        raise RuntimeError(f"Only generated {len(out)} connected samples after {attempts} attempts.")
    return out
