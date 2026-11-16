"""Split-pressure Picard trajectories with independent CPU acceptance."""
from __future__ import annotations

import numpy as np
import torch

from learned_tg.runtime import initialize, load_model, new_output, read_config
from learned_tg.solvers.forchheimer_utils import _coords, _rhs, _sample_connected_channelized, _predict_raw_basis
from learned_tg.solvers.geneo import GenEOProvider
from learned_tg.solvers.picard import PicardSettings, solve_picard_split, sha, tensor_sha, write_json
from learned_tg.solvers.static import dump, record_protocol
from learned_tg.validation.split import check_step
from learned_tg.model.forchheimer_split import mobility_from_split


class LearnedProvider:
    preparation = {"coarse_basis_orth_metric": "a"}
    signature = {"method": "learned", "preparation": "a", "network_precision": "float32"}

    def __init__(self, model, phi, x, y, dimension):
        self.model, self.phi, self.x, self.y, self.dimension = model, phi, x, y, dimension

    def __call__(self, operator, step, mobility):
        raw, _ = _predict_raw_basis(self.model, mobility, self.phi, self.x, self.y, m_dim=self.dimension)
        return raw, {"method": "learned", "span_reused": False}


@torch.no_grad()
def run(args):
    config = read_config(args.config)
    device = initialize(config.get("evaluation_seed", 20260720), args.device, args.threads)
    output = new_output(args.output, args.resume)
    n = args.grid or int(config.get("evaluation_grid", 1023))
    methods = args.methods.split(",")
    if not set(methods) <= {"learned", "rebuild", "frozen"}:
        raise ValueError("Methods must be learned, rebuild, or frozen.")
    model = None
    checkpoint_digest = "not_used_by_spectral_methods"
    if "learned" in methods:
        if not args.checkpoint:
            raise ValueError("The learned method requires --checkpoint.")
        model = load_model(args.checkpoint, config, device)
        checkpoint_digest = sha(args.checkpoint)
    config.update(device=str(device), subdomain_solver="cholesky", basis_orth_reg=1e-8,
                  coarse_basis_orth_metric="a", basis_smooth_mode_solve="none", basis_smooth_steps_solve=0,
                  pre_smooth_mode="asm", post_smooth_mode="asm", pre_smooth_steps=1,
                  post_smooth_steps=1, pre_smooth_omega=0.8, post_smooth_omega=0.8,
                  k_1d=args.subdomains or config.get("k_1d", 10), vcycle_compile=False)
    fields = _sample_connected_channelized(n=n, samples=args.samples, contrast=1e6, config=config,
                                           seed=config.get("evaluation_seed", 20260720), device="cpu")
    _, x, y = _coords(n, device)
    protocol = dict(solver="FGMRES_split_compensated", restart=50, linear_tol=1e-6,
                    safety=0.5, max_linear_updates=300, coarse_reg=1e-8,
                    max_picard_steps=args.max_picard_steps, threshold=0.30, max_local_modes=64,
                    timing_eligible=False, grid=n, samples=args.samples, methods=methods,
                    subdomains=config["k_1d"])
    record_protocol(output / "protocol.json", protocol, args.resume)
    rows, representative = [], []
    for field, (phi, permeability, metadata) in enumerate(fields):
        cpu = dict(phi=phi, kappa0=permeability, rhs=_rhs(n, 10.0, "cpu"))
        inputs = {key: value.to(device) for key, value in cpu.items()}
        dump(output / f"field{field}.json", dict(geometry=metadata,
             tensor_hashes={key: tensor_sha(value) for key, value in cpu.items()}))
        for alpha in [float(value) for value in args.alphas.split(",")]:
            for method in methods:
                case = output / f"{method}_field{field}_alpha{alpha:g}"
                provider = (LearnedProvider(model, inputs["phi"], x, y, int(config.get("m_dim", 30)))
                            if method == "learned" else
                            GenEOProvider(method, field, output / "spectra" / case.name))

                def independent(path):
                    result = check_step(path, cpu, alpha)
                    write_json(path / "status.json", dict(status="accepted",
                               independent_validation_sha256=sha(path / "independent_validation.json")))
                    return result

                _, summary, terminal, _ = solve_picard_split(model=model, **inputs, x_net=x, y_net=y,
                    config=config, out_dir=case, sample=field,
                    settings=PicardSettings(alpha=alpha, max_steps=args.max_picard_steps, full_trajectory=True),
                    checkpoint_sha256=checkpoint_digest, basis_provider=provider,
                    independent_checker=independent, protocol_signature=protocol,
                    save_bases=args.save_bases, allow_existing=args.resume)
                rows.append(dict(summary, method=method, artifact_dir=case.name))
                dump(output / "summary.json", dict(results=rows, timing_eligible=False))
                if field == 0 and method == "learned" and summary["nonlinear_converged"]:
                    mobility = mobility_from_split(inputs["kappa0"], terminal, alpha=alpha).mobility
                    representative.append(dict(alpha=alpha, phi=phi.numpy().squeeze(),
                        kappa0=permeability.numpy().squeeze(), high=terminal.high.cpu().numpy().squeeze(),
                        low=terminal.low.cpu().numpy().squeeze(), mobility=mobility.cpu().numpy().squeeze()))
    if representative:
        np.savez_compressed(output / "representative_fields.npz", phi=representative[0]["phi"],
            kappa0=representative[0]["kappa0"], alphas=np.array([r["alpha"] for r in representative]),
            pressure_high=np.stack([r["high"] for r in representative]),
            pressure_low=np.stack([r["low"] for r in representative]),
            mobility=np.stack([r["mobility"] for r in representative]))
    return rows
