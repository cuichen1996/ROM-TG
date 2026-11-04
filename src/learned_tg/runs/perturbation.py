"""Recompute the same-grid diagnostics after a matching static comparison."""
from __future__ import annotations

import torch
from learned_tg.runtime import initialize, new_output, read_config, load_model
from learned_tg.runs.static import arguments
from learned_tg.coarse.spectral import _generate_canonical_interface_samples
from learned_tg.validation.perturbation import diagnose
from learned_tg.solvers.static import dump


@torch.no_grad()
def run(args):
    cfg = read_config(args.config)
    initialize(cfg["seed"], args.device, args.threads)
    output = new_output(args.output)
    config = arguments(cfg, args.device, args.samples)
    model = load_model(args.checkpoint, read_config(args.model_config), args.device)
    stored = {}
    for row in read_config(args.static_results)["results"]:
        if row.get("solver") != "FGMRES" or row.get("method") not in {"POD-ref", "Learned-white"}:
            continue
        if not row.get("accepted"):
            raise ValueError("The diagnostics require independently accepted paired solves.")
        stored.setdefault((row["n"], row["sample_id"]), []).append(dict(row,
            iterations=row["updates"], sample_field_sha256=row["field_sha256"]))
    result = []
    for n in map(int, args.grids.split(",")):
        for sample in _generate_canonical_interface_samples(n, config):
            pair = stored.get((n, sample["sample_id"]), [])
            if {r["method"] for r in pair} != {"POD-ref", "Learned-white"}:
                raise ValueError(f"Missing matched static results: n={n}, sample={sample['sample_id']}")
            result.append(diagnose(n, sample, config, model, stored, output / f"n{n}_sample{sample['sample_id']}"))
    dump(output / "summary.json", dict(results=result, timing_eligible=False))
