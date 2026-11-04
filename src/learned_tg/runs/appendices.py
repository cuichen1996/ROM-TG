"""Manufactured-interface convergence and deterministic error smoothing."""
from __future__ import annotations
import csv

from learned_tg.runtime import initialize, new_output, read_config
from learned_tg.problems.manufactured import run_convergence_study
from learned_tg.problems.smoothing import seed_everything, generate_single_parameter_sample, collect_smoothing_results
from learned_tg.plotting.smoothing import _write_payload, _load_payload, plot_experiment
from learned_tg.solvers.static import dump


def run(args):
    cfg = read_config(args.config)
    initialize(device=args.device, threads=args.threads)
    output = new_output(args.output)
    if args.kind == "circle":
        rows = []
        for mode in ["harmonic", "interface_aware"]:
            result, _ = run_convergence_study(cfg["grids"], cases=["circle"], kappa=cfg["contrast"],
                                             radius=cfg["radius"], transmissibility=mode)
            # Runtime seconds are not controlled timing measurements.
            rows.extend({k: v for k, v in row.items() if not k.endswith("seconds")} for row in result)
        with (output / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        dump(output / "summary.json", dict(results=rows, timing_eligible=False))
    else:
        seed_everything(cfg["sample_seed"])
        sample = generate_single_parameter_sample(n=cfg["n"], contrast=cfg["contrast"], device="cpu",
                                                 topology_mode="island", max_retries=cfg["max_retries"])
        sample["transmissibility"] = "interface_aware"
        specs = [dict(label=f"asm-{s}", mode="none" if s == 0 else "asm", steps=s, omega=cfg["omega"])
                 for s in cfg["smoothing_steps"]]
        data = collect_smoothing_results(sample, specs, seed=cfg["error_seed"], k_1d=cfg["k_1d"])
        _write_payload(data, output / "smoothing.npz")
        plot_experiment(_load_payload(output / "smoothing.npz"), output / "smoothing.png")
