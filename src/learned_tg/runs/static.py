"""Static interface benchmarks with independent residual acceptance."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from learned_tg.baselines.deeponet_tb.adapter import (
    evaluate_trunk_matrix, build_tb_basis_with_qr_info, prepare_tb_coarse_basis,
)
from learned_tg.coarse.baselines import _compress_global_bases, _predict_learned_basis
from learned_tg.coarse.pod import build_oracle_basis
from learned_tg.coarse.spectral import _generate_canonical_interface_samples, _build_local_spectral
from learned_tg.coarse.tb import _load_tb_model, _reconstruct_svd_selection
from learned_tg.problems.static_helpers import _make_interface_operator, _generate_smoothed_snapshots
from learned_tg.runtime import initialize, load_model, new_output, read_config
from learned_tg.solvers.static import dump, prepare, sha, solve, thash
from learned_tg.validation.sparse import assemble_fvm_matrix

METHODS = {"asm": "ASM", "pod": "POD-ref", "learned": "Learned-white", "raw": "Learned-raw",
           "global-pod": "Global-POD", "tb": "TB-SVD", "geneo": "GenEO"}


def arguments(config, device, samples=None, subdomains=None):
    config = dict(config)
    config.update(device=str(device), sample_generation_device="cpu", subdomain_solver="cholesky",
                  pre_smooth_steps=1, post_smooth_steps=1, omega=0.8, vcycle_compile=False)
    if samples is not None:
        config["num_samples"] = samples
    if subdomains is not None:
        config["k_1d"] = subdomains
    return argparse.Namespace(**config)


class Models:
    def __init__(self, args, config):
        self.args, self.config, self.cache = args, config, {}
        self.global_bases = {}

    def learned(self, label):
        if label not in self.cache:
            raw = label == "Learned-raw"
            path = self.args.raw_checkpoint if raw else self.args.checkpoint
            cfg_path = self.args.raw_model_config if raw else self.args.model_config
            if not path or not cfg_path:
                raise ValueError(f"{label} needs a checkpoint and its model configuration.")
            self.cache[label] = load_model(path, read_config(cfg_path), self.args.device)
        return self.cache[label]


def snapshots(operator, sample, cfg):
    return _generate_smoothed_snapshots(problem="interface", operator=operator,
        n=sample["n"], sample_seed=sample["seed"], args=cfg)[0]


def global_basis(n, cfg, output):
    """Build the fixed 50-parameter, 15-probe Global-POD training pool."""
    pool = []
    local = argparse.Namespace(**vars(cfg))
    local.num_samples = 1
    local.num_probes = 15
    for seed in range(91234, 91284):
        local.seed = seed - 1000 * n
        sample = _generate_canonical_interface_samples(n, local)[0]
        if sample.get("failed"):
            raise RuntimeError(f"Global-POD parameter generation failed: {seed}")
        op = _make_interface_operator(sample, local)
        E = snapshots(op, sample, local)
        raw, _, _ = build_oracle_basis(E, op.asm.apply_A, basis_type="a_pod", rank=15, reg=1e-12)
        pool.append(raw.cpu())
    bases, _, rank = _compress_global_bases([b.to(cfg.device) for b in pool], [30], torch.device(cfg.device))
    payload = {"bases": {"global-POD-n30": bases[30].cpu()}, "pool_rank": rank,
               "parameter_seeds": list(range(91234, 91284)), "num_probes": 15}
    torch.save(payload, output)
    return bases[30]


def coarse_basis(method, operator, sample, cfg, models, dest, E=None):
    if method == "ASM":
        return None
    if method == "POD-ref":
        E = snapshots(operator, sample, cfg) if E is None else E
        raw, _, diagnostics = build_oracle_basis(E, operator.asm.apply_A, "a_pod", rank=15, reg=1e-12)
        dump(dest / "snapshots.json", diagnostics)
        prepared, _ = prepare(operator, raw, expected=15)
    elif method.startswith("Learned"):
        raw, _ = _predict_learned_basis(models.learned(method), sample, cfg)
        prepared, _ = prepare(operator, raw, expected=15)
    elif method == "Global-POD":
        if sample["n"] in models.global_bases:
            raw = models.global_bases[sample["n"]]
        elif models.args.global_cache:
            path = Path(models.args.global_cache.format(n=sample["n"]))
            payload = torch.load(path, map_location="cpu", weights_only=False)
            raw = payload["bases"]["global-POD-n30"].to(cfg.device, dtype=torch.float64)
        else:
            raw = global_basis(sample["n"], cfg, dest / "global_basis.pt")
        models.global_bases[sample["n"]] = raw
        prepared, _ = prepare(operator, raw, expected=30)
    elif method == "GenEO":
        raw, diagnostics = _build_local_spectral(operator, cfg, selection="threshold_modified_geneo",
                                                sample_id=sample["sample_id"], threshold=0.30)
        if not diagnostics["threshold_complete"]:
            raise RuntimeError("GenEO threshold was not bracketed.")
        dump(dest / "spectrum.json", diagnostics)
        prepared, _ = prepare(operator, raw, mode="raw")
    elif method == "TB-SVD":
        if not models.args.tb_checkpoint or not models.args.tb_metadata:
            raise ValueError("TB-SVD needs --tb-checkpoint and --tb-metadata.")
        meta = read_config(models.args.tb_metadata)
        model = _load_tb_model(Path(models.args.tb_checkpoint), meta, torch.device(cfg.device))
        _, selection = _reconstruct_svd_selection(meta)
        trunk = evaluate_trunk_matrix(model, sample["n"], torch.device(cfg.device))
        raw, _ = build_tb_basis_with_qr_info(trunk, selection, torch.device(cfg.device), qr_epsilon=1e-10)
        if raw.shape[-1] != 15:
            raise RuntimeError("The TB-SVD coarse space must have rank 15.")
        prepared = prepare_tb_coarse_basis(operator, raw, coarse_basis_mode="a_orthonormal")
    else:
        raise ValueError(method)
    torch.save(raw.detach().cpu(), dest / "raw_basis.pt")
    torch.save(prepared["Q"].detach().cpu(), dest / "prepared_basis.pt")
    return prepared


def system_record(operator, sample, dest):
    import numpy as np
    dest.mkdir(parents=True, exist_ok=True)
    arrays = {k: getattr(operator.asm, k).cpu().numpy() for k in ["sw", "se", "sn", "ss", "sc"]}
    arrays.update(rhs=sample["b"].cpu().numpy(), phi=sample["phi"].cpu().numpy(), coefficient=sample["a_field"].cpu().numpy())
    np.savez_compressed(dest / "system.npz", **arrays)
    return dict(n=sample["n"], sample_id=sample["sample_id"], sample_seed=sample["seed"],
                field_sha256=sample["sample_field_sha256"], rhs_sha256=thash(sample["b"]),
                linear_tol=1e-6, max_updates=200, eps_local=1e-6, eps_coarse=1e-10)


@torch.no_grad()
def run(args):
    config = read_config(args.config)
    initialize(config.get("seed", 1234), args.device, args.threads)
    output = new_output(args.output)
    cfg = arguments(config, args.device, args.samples, args.subdomains)
    models = Models(args, config)
    names = list(METHODS) if args.methods == "all" else args.methods.split(",")
    methods = [METHODS[n] for n in names]
    for method in methods:
        if method.startswith("Learned"):
            models.learned(method)
    dump(output / "protocol.json", dict(configuration=config, device=args.device,
        checkpoint_sha256=sha(args.checkpoint) if args.checkpoint else None,
        raw_checkpoint_sha256=sha(args.raw_checkpoint) if args.raw_checkpoint else None,
        solver="FGMRES(50)", timing_eligible=False))
    rows = []
    for n in [int(v) for v in args.grids.split(",")]:
        for sample in _generate_canonical_interface_samples(n, cfg):
            if sample.get("failed"):
                rows.append(dict(n=n, status="field_generation_failure", accepted=False))
                continue
            dest = output / f"n{n}" / f"sample{sample['sample_id']}"
            operator = _make_interface_operator(sample, cfg)
            base = system_record(operator, sample, dest)
            matrix = assemble_fvm_matrix(operator.asm).tocsr()
            for method in methods:
                folder = dest / method
                basis_dir = folder / "basis"
                basis_dir.mkdir(parents=True)
                solvers = ["FGMRES"]
                if args.include_tg and method in {"Learned-white", "Learned-raw"}:
                    solvers.append("TG")
                try:
                    prepared = coarse_basis(method, operator, sample, cfg, models, basis_dir)
                except (ValueError, RuntimeError) as error:
                    if isinstance(error, torch.cuda.OutOfMemoryError):
                        raise
                    for solver in solvers:
                        row = dict(base, method=method, solver=solver, status="construction_failure",
                                   accepted=False, detail=str(error), timing_eligible=False)
                        dump(folder / solver / "metrics.json", row)
                        rows.append(row)
                    continue
                for solver in solvers:
                    rows.append(solve(operator, matrix, sample["b"], prepared, method, solver, folder / solver, base))
    dump(output / "summary.json", dict(accepted=sum(r["accepted"] for r in rows), results=rows,
                                       timing_eligible=False))
    return rows
