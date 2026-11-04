"""Evaluate epoch checkpoints on the independent Figure 4 validation cohort."""
from __future__ import annotations
import csv
from types import SimpleNamespace

import numpy as np
import torch

from learned_tg.runtime import initialize, load_model, new_output, read_config
from learned_tg.runs.static import arguments, coarse_basis, system_record
from learned_tg.coarse.spectral import _generate_canonical_interface_samples
from learned_tg.problems.static_helpers import _make_interface_operator
from learned_tg.solvers.static import dump, solve
from learned_tg.validation.sparse import assemble_fvm_matrix


@torch.no_grad()
def run(args):
    cfg = read_config(args.config)
    initialize(20261101, args.device, args.threads)
    output = new_output(args.output)
    cfg.update(seed=20261101)
    config = arguments(cfg, args.device, args.samples)
    samples = _generate_canonical_interface_samples(63, config)
    columns = ["label", "epoch", "solver", "attempted", "accepted", "constructed", "mean_success_updates",
               "mean_consumed_updates", "maximum_accepted_cpu_dd_residual"]
    records = []
    with (output / "validation_history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for epoch in range(args.first_epoch, args.last_epoch + 1, args.epoch_step):
            for label in ["white", "raw"]:
                path = getattr(args, f"{label}_checkpoints").format(epoch=epoch)
                model = load_model(path, read_config(getattr(args, f"{label}_config")), args.device)
                for sample in samples:
                    dest = output / label / f"epoch{epoch:04d}" / f"sample{sample['sample_id']}"
                    op = _make_interface_operator(sample, config)
                    base = system_record(op, sample, dest)
                    matrix = assemble_fvm_matrix(op.asm).tocsr()
                    basis = dest / "basis"; basis.mkdir()
                    models = SimpleNamespace(learned=lambda _: model)
                    try:
                        prepared = coarse_basis("Learned-white", op, sample, config, models, basis)
                    except (ValueError, RuntimeError) as error:
                        if isinstance(error, torch.cuda.OutOfMemoryError):
                            raise
                        for solver in ["TG", "FGMRES"]:
                            records.append(dict(label=label, epoch=epoch, solver=solver, constructed=False,
                                                accepted=False, updates=0, error=str(error)))
                        continue
                    for solver in ["TG", "FGMRES"]:
                        row = solve(op, matrix, sample["b"], prepared, f"Learned-{label}", solver, dest / solver, base)
                        records.append(dict(row, label=label, epoch=epoch, constructed=True))
                for solver in ["TG", "FGMRES"]:
                    group = [r for r in records if (r["label"], r["epoch"], r["solver"]) == (label, epoch, solver)]
                    accepted = [r for r in group if r["accepted"]]
                    writer.writerow(dict(label=label, epoch=epoch, solver=solver, attempted=len(group),
                        accepted=len(accepted), constructed=sum(r["constructed"] for r in group),
                        mean_success_updates=np.mean([r["updates"] for r in accepted]) if accepted else "",
                        mean_consumed_updates=np.mean([r["updates"] for r in group if r["constructed"]]) if any(r["constructed"] for r in group) else "",
                        maximum_accepted_cpu_dd_residual=max((r["independent"]["cpu_dd_relative_residual"] for r in accepted), default="")))
                    stream.flush()
    dump(output / "summary.json", dict(results=records, validation_seed=20261101, timing_eligible=False))
