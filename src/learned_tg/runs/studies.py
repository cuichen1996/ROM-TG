"""Fixed q=200 example and nested-prefix snapshot ablation."""
from __future__ import annotations

from types import SimpleNamespace
import torch

from learned_tg.runs.static import arguments, coarse_basis, snapshots, system_record
from learned_tg.coarse.spectral import _generate_canonical_interface_samples
from learned_tg.problems.static_helpers import _make_interface_operator
from learned_tg.runtime import initialize, new_output, read_config
from learned_tg.solvers.static import dump, solve, thash
from learned_tg.validation.sparse import assemble_fvm_matrix


@torch.no_grad()
def run(args):
    config = read_config(args.config)
    initialize(config.get("seed", 1234), args.device, args.threads)
    output = new_output(args.output)
    cfg = arguments(config, args.device, 1 if args.kind == "q200" else args.samples, args.subdomains)
    rows = []
    for sample in _generate_canonical_interface_samples(63, cfg):
        op = _make_interface_operator(sample, cfg)
        dest = output if args.kind == "q200" else output / f"sample{sample['sample_id']}"
        base = system_record(op, sample, dest)
        A = assemble_fvm_matrix(op.asm).tocsr()
        if args.kind == "q200":
            cfg.num_probes, cfg.snapshot_smooth_steps = 200, 50
            E = snapshots(op, sample, cfg)
            AE = op.asm.apply_A(E)
            dump(dest / "spectral.json", dict(energy_eigenvalues=torch.linalg.eigvalsh((E.mT @ AE + AE.mT @ E) / 2).flip(-1),
                singular_values=torch.linalg.svdvals(E), snapshot_sha256=thash(E), sample_seed=sample["seed"],
                probe_seed=sample["seed"] + 7919, q=200, nu_snap=50))
            basis = dest / "basis"; basis.mkdir()
            prepared = coarse_basis("POD-ref", op, sample, cfg, None, basis, E)
            for method, solver in [("ASM", "FGMRES"), ("POD-ref", "TG"), ("POD-ref", "FGMRES")]:
                rows.append(solve(op, A, sample["b"], None if method == "ASM" else prepared, method, solver,
                                  dest / f"{method}_{solver}", dict(base, record_true_curve=True)))
        else:
            generator = torch.Generator(device="cpu").manual_seed(sample["seed"] + 7919)
            probes = torch.cat([torch.randn((1, 20, 63, 63), generator=generator, dtype=torch.float64),
                                torch.randn((1, 55, 63, 63), generator=generator, dtype=torch.float64)], dim=1).to(args.device)
            dump(dest / "probes.json", dict(seed=sample["seed"] + 7919, sha256=thash(probes),
                                          rule="original 20 then 55; all q values use nested prefixes"))
            for m, q in [(m, 20) for m in [5, 10, 20, 30, 50, 75]] + [(50, q) for q in [15, 30, 40, 50, 75]]:
                case = dest / f"M{m}_q{q}"
                basis = case / "basis"; basis.mkdir(parents=True)
                error = op.asm.asm_smooth(probes[:, :q].clone(), torch.zeros_like(probes[:, :q]), M_smooth=m, omega=0.8)
                E = error.reshape(1, q, 63 * 63).transpose(1, 2).contiguous()
                cfg.num_probes, cfg.snapshot_smooth_steps = q, m
                prepared = coarse_basis("POD-ref", op, sample, cfg, None, basis, E)
                for solver in ["TG", "FGMRES"]:
                    rows.append(solve(op, A, sample["b"], prepared, "POD-ref", solver, case / solver,
                                      dict(base, nu_snap=m, q=q)))
    dump(output / "summary.json", dict(results=rows, timing_eligible=False))
    return rows
