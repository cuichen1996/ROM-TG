"""Numerical routines used by the paper; no experiment orchestration."""
from __future__ import annotations
import random
from typing import Sequence
import numpy as np
import torch
from learned_tg.data.gen_data_interface import SUPPORTED_TOPOLOGY_MODES, check_topology, generate_parameter_data
from learned_tg.model.interface import JumpingCoefficient

DEFAULT_SMOOTHING_SPECS = (
    {"label": "none", "mode": "none", "steps": 0, "omega": 1.0},
    {"label": "jacobi-1", "mode": "jacobi", "steps": 1, "omega": 0.8},
    {"label": "jacobi-4", "mode": "jacobi", "steps": 4, "omega": 0.8},
    {"label": "asm-1", "mode": "asm", "steps": 1, "omega": 0.8},
    {"label": "asm-2", "mode": "asm", "steps": 2, "omega": 0.8},
    {"label": "asm-4", "mode": "asm", "steps": 4, "omega": 0.8},
)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def zero_dirichlet_boundary(field: torch.Tensor) -> torch.Tensor:
    field = field.clone()
    field[..., 0, :] = 0.0
    field[..., -1, :] = 0.0
    field[..., :, 0] = 0.0
    field[..., :, -1] = 0.0
    return field

def create_operator(sample: dict, k_1d: int = 10) -> JumpingCoefficient:
    transmissibility = str(sample.get("transmissibility", "harmonic"))
    return JumpingCoefficient(
        sample["a_field"],
        n=sample["n"],
        k_1d=k_1d,
        phi=sample.get("phi") if transmissibility.lower() != "harmonic" else None,
        transmissibility=transmissibility,
    )

def generate_single_parameter_sample(
    *,
    n: int = 63,
    contrast: float = 1e4,
    device: str | torch.device = "cpu",
    topology_mode: str = "mixed",
    max_retries: int = 100,
) -> dict:
    if topology_mode not in SUPPORTED_TOPOLOGY_MODES:
        raise ValueError(
            f"Unsupported topology_mode={topology_mode!r}. "
            f"Expected one of {sorted(SUPPORTED_TOPOLOGY_MODES)}."
        )

    phi, a_field, coords, X, Y = generate_parameter_data(
        batch_size=1,
        n=n,
        contrast=contrast,
        device=device,
        topology_mode=topology_mode,
        max_retries=max_retries,
    )

    actual_topology = check_topology(a_field[0])
    return {
        "phi": phi.to(torch.float64),
        "a_field": a_field.to(torch.float64),
        "coords": coords.to(torch.float64),
        "X": X.to(torch.float64),
        "Y": Y.to(torch.float64),
        "n": int(n),
        "contrast": float(contrast),
        "requested_topology": topology_mode,
        "actual_topology": actual_topology,
    }

def energy_norm(operator: JumpingCoefficient, field: torch.Tensor) -> float:
    field = field.to(torch.float64)
    A_field = operator.forward(field)
    flat = field.reshape(field.shape[0], -1)
    A_flat = A_field.reshape(A_field.shape[0], -1)
    energy_sq = (flat * A_flat).sum(dim=1).clamp_min(0.0)
    return float(torch.sqrt(energy_sq).mean().item())

def l2_norm(field: torch.Tensor) -> float:
    flat = field.reshape(field.shape[0], -1)
    return float(torch.linalg.norm(flat, dim=1).mean().item())

@torch.no_grad()
def collect_smoothing_results(
    sample: dict,
    smoothing_specs: Sequence[dict] = DEFAULT_SMOOTHING_SPECS,
    *,
    seed: int = 0,
    k_1d: int = 10,
) -> dict:
    if not smoothing_specs:
        raise ValueError("smoothing_specs must be non-empty.")

    seed_everything(seed)
    operator = create_operator(sample, k_1d=k_1d)
    n = sample["n"]

    e_init = torch.randn(1, 1, n, n, device=sample["a_field"].device, dtype=torch.float64)
    e_init = zero_dirichlet_boundary(e_init)
    f_zero = torch.zeros_like(e_init)

    results = []
    for spec in smoothing_specs:
        mode = str(spec["mode"]).lower()
        steps = int(spec.get("steps", 0))
        omega = float(spec.get("omega", 0.8))
        label = str(spec.get("label", f"{mode}-{steps}"))

        if mode == "none":
            error = e_init.clone()
        else:
            error = operator.solution_smooth(
                e_init,
                f_zero,
                mode=mode,
                M_smooth=steps,
                omega=omega,
            )

        results.append(
            {
                "label": label,
                "mode": mode,
                "steps": steps,
                "omega": omega,
                "error": error.detach().clone(),
                "l2_norm": l2_norm(error),
                "energy_norm": energy_norm(operator, error),
            }
        )

    return {
        "sample": sample,
        "operator": operator,
        "initial_error": e_init,
        "results": results,
    }
