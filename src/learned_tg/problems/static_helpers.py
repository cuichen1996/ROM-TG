"""Numerical routines used by the paper; no experiment orchestration."""
from __future__ import annotations
from learned_tg.coarse.pod import _apply_A_interface
import argparse
import time
import traceback
from typing import Any, Dict, List, Tuple
import torch
from learned_tg.data.gen_data_interface import check_topology, generate_random_interface_coeffs_with_topology, smooth_interface_snapshot_family
from learned_tg.data.gen_data_two_phase import smooth_two_phase_snapshot_family
from learned_tg.coarse.pod import _ensure_4d_field, _smooth_interface_coefficient_from_phi
from learned_tg.model.interface import JumpingCoefficient

def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)

def _time_call(device: torch.device, fn):
    _sync_if_needed(device)
    tic = time.perf_counter()
    out = fn()
    _sync_if_needed(device)
    return out, time.perf_counter() - tic

def _safe_k_1d(n: int, requested: int) -> int:
    n = int(n)
    requested = int(requested)
    return max(1, min(requested, max(1, n - 1)))

def _gmg_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "num_levels": args.gmg_num_levels,
        "min_coarse_size": int(args.gmg_min_coarse_size),
        "smoother": str(args.gmg_smoother),
        "pre_smooth_steps": int(args.gmg_pre_smooth_steps),
        "post_smooth_steps": int(args.gmg_post_smooth_steps),
        "omega": float(args.omega),
        "coarse_solve": str(args.gmg_coarse_solve),
        "snapshot_gmg_num_levels": args.gmg_num_levels,
        "snapshot_gmg_min_coarse_size": int(args.gmg_min_coarse_size),
        "snapshot_gmg_smoother": str(args.gmg_smoother),
        "snapshot_gmg_pre_smooth_steps": int(args.gmg_pre_smooth_steps),
        "snapshot_gmg_post_smooth_steps": int(args.gmg_post_smooth_steps),
        "snapshot_gmg_omega": float(args.omega),
        "snapshot_gmg_outer_omega": float(args.omega),
        "snapshot_gmg_coarse_solve": str(args.gmg_coarse_solve),
    }

def _make_interface_operator(sample: Dict[str, Any], args: argparse.Namespace) -> JumpingCoefficient:
    n = int(sample["n"])
    transmissibility = str(getattr(args, "transmissibility", "harmonic"))
    return JumpingCoefficient(
        sample["a_field"].to(device=torch.device(args.device), dtype=torch.float64),
        n=n,
        k_1d=_safe_k_1d(n, int(args.k_1d)),
        subdomain_solver=str(args.subdomain_solver),
        vcycle_iter=int(args.vcycle_iter),
        vcycle_smooth_steps=int(args.vcycle_smooth_steps),
        vcycle_omega=float(args.vcycle_omega),
        vcycle_compile=bool(getattr(args, "vcycle_compile", True)),
        phi=(
            sample.get("phi")
            if transmissibility.lower() != "harmonic"
            else None
        ),
        transmissibility=transmissibility,
    )

def generate_interface_samples_for_grid(
    n: int,
    config: argparse.Namespace,
    num_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Generate deterministic interface samples for one grid size."""
    device = torch.device(config.device)
    dtype = torch.float64
    samples: List[Dict[str, Any]] = []

    for sample_id in range(int(num_samples)):
        seed_sample = int(seed) + 1000 * int(n) + int(sample_id)
        try:
            fork_devices = [device.index if device.index is not None else 0] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(seed_sample)
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_sample)
                phi, a_field, X, Y = generate_random_interface_coeffs_with_topology(
                    batch_size=1,
                    n=int(n),
                    contrast=float(config.contrast),
                    device=device,
                    topology_mode=str(config.topology_mode),
                    max_retries=int(config.max_shape_retries),
                )

            phi = _ensure_4d_field(phi, "phi").to(device=device, dtype=dtype)
            a_hard = _ensure_4d_field(a_field, "a_field").to(device=device, dtype=dtype)
            if str(config.interface_coeff_mode).lower() == "smooth":
                eps = float(config.interface_eps_factor) / float(n)
                a_used = _smooth_interface_coefficient_from_phi(phi, a_hard, eps=eps)
            else:
                a_used = a_hard

            h = 1.0 / float(n + 1)
            b = torch.full((1, 1, int(n), int(n)), h ** 2, device=device, dtype=dtype)
            samples.append(
                {
                    "failed": False,
                    "problem": "interface",
                    "n": int(n),
                    "sample_id": int(sample_id),
                    "seed": int(seed_sample),
                    "phi": phi,
                    "a_field": a_used,
                    "a_hard": a_hard,
                    "b": b,
                    "rhs": b,
                    "X": X.detach().cpu().to(dtype),
                    "Y": Y.detach().cpu().to(dtype),
                    "topology_or_shape": check_topology(a_used[0, 0].detach().cpu()),
                    "coeff_mode": str(config.interface_coeff_mode).lower(),
                    "contrast": float(a_used.max().item() / max(a_used.min().item(), 1e-300)),
                }
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics should continue.
            traceback.print_exc()
            samples.append(
                {
                    "failed": True,
                    "problem": "interface",
                    "n": int(n),
                    "sample_id": int(sample_id),
                    "seed": int(seed_sample),
                    "notes": f"failed: sample generation failed: {exc}",
                }
            )
    return samples

@torch.no_grad()
def _generate_smoothed_snapshots(
    *,
    problem: str,
    operator: Any,
    n: int,
    sample_seed: int,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, float]:
    device = torch.device(args.device)

    def work() -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(sample_seed) + 7919)
        e0 = torch.randn(
            1,
            int(args.num_probes),
            int(n),
            int(n),
            generator=generator,
            dtype=torch.float64,
        ).to(device=device)
        if problem == "two_phase":
            e_smooth = smooth_two_phase_snapshot_family(
                operator,
                e0,
                M_smooth=int(args.snapshot_smooth_steps),
                omega=float(args.omega),
                smoother=str(args.snapshot_smoother),
                config=_gmg_config(args),
            )
        elif problem == "interface":
            e_smooth = smooth_interface_snapshot_family(
                operator.asm,
                e0,
                M_smooth=int(args.snapshot_smooth_steps),
                omega=float(args.omega),
                smoother=str(args.snapshot_smoother),
                config=_gmg_config(args),
            )
        else:
            raise ValueError(f"Unknown problem={problem!r}")
        return e_smooth.reshape(1, int(args.num_probes), int(n) * int(n)).transpose(1, 2).contiguous()

    E, elapsed = _time_call(device, work)
    if E.dim() != 3 or E.shape != (1, int(n) * int(n), int(args.num_probes)):
        raise ValueError(f"snapshot E shape mismatch: got {tuple(E.shape)}")
    return E, float(elapsed)
