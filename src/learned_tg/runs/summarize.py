"""Create table-ready statistics while retaining failed-case counts."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def run(args):
    source = Path(args.input)
    data = json.loads(source.read_text())
    rows = data["results"]
    groups = defaultdict(list)
    output = []
    if args.kind == "forchheimer":
        for row in rows:
            groups[(row["alpha"], row["method"])].append(row)
        for (alpha, method), group in sorted(groups.items()):
            accepted = [r for r in group if r["nonlinear_converged"] and r["all_linear_accepted"]]
            picard = [r["accepted_updates"] for r in accepted]
            linear = [r["total_fgmres_updates"] / r["accepted_updates"] for r in accepted]
            output.append(dict(alpha=alpha, method=method, attempted=len(group), accepted=len(accepted),
                picard_mean=float(np.mean(picard)) if picard else "", picard_std=float(np.std(picard)) if picard else "",
                fgmres_per_trajectory_mean=float(np.mean(linear)) if linear else "",
                fgmres_per_trajectory_std=float(np.std(linear)) if linear else ""))
    else:
        keys = ["method"] if args.kind == "moving" else ["n", "method", "solver", "nu_snap", "q"]
        for row in rows:
            groups[tuple(row.get(k, "") for k in keys)].append(row)
        for key, group in sorted(groups.items()):
            accepted = [r for r in group if r["accepted"]]
            counts = [r["updates"] for r in accepted]
            output.append(dict(zip(keys, key), attempted=len(group), accepted=len(accepted), failed=len(group) - len(accepted),
                mean=float(np.mean(counts)) if counts else "", population_std=float(np.std(counts)) if counts else ""))
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
