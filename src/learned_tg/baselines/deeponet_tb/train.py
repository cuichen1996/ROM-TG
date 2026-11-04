"""Train the solution-DeepONet for the trunk-basis (TB) baseline.

Generates certified solution labels on the static random-interface benchmark,
trains the coordinate-only-trunk DeepONet against the relative L2 solution error,
extracts the retained SVD trunk basis on the offline selection grid, and writes
the checkpoint + metadata + cost files to a fresh output dir.

This is an independent implementation of the DeepONet trunk-basis method based on
Kopaničáková and Karniadakis (2025).  No official implementation was available.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from learned_tg.baselines.deeponet_tb.basis_selection import select_trunk_basis
from learned_tg.baselines.deeponet_tb.labels import generate_certified_labels
from learned_tg.baselines.deeponet_tb.solution_deeponet import SolutionDeepONet, cell_center_coords, train_solution_deeponet
from learned_tg.problems.static_helpers import generate_interface_samples_for_grid


def _stack_fields(samples: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a_list, phi_list, coords_list = [], [], []
    for s in samples:
        a_list.append(torch.as_tensor(s["a_field"], dtype=torch.float32).reshape(1, 1, int(s["n"]), int(s["n"])))
        phi_list.append(torch.as_tensor(s["phi"], dtype=torch.float32).reshape(1, 1, int(s["n"]), int(s["n"])))
        coords_list.append(torch.stack([torch.as_tensor(s["X"]).reshape(-1), torch.as_tensor(s["Y"]).reshape(-1)], dim=1))
    a = torch.cat(a_list, dim=0)
    phi = torch.cat(phi_list, dim=0)
    coords = coords_list[0]  # all samples share the same grid coords
    return a, phi, coords


def _batches(a: torch.Tensor, phi: torch.Tensor, u: torch.Tensor, batch_size: int):
    n = a.shape[0]
    for i in range(0, n, batch_size):
        yield a[i:i + batch_size], phi[i:i + batch_size], u[i:i + batch_size]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n", type=int, default=63)
    p.add_argument("--contrast", type=float, default=1e4)
    p.add_argument("--topology-mode", default="island")
    p.add_argument("--transmissibility", default="interface_aware")
    p.add_argument("--k-1d", type=int, default=10)
    p.add_argument("--num-train-samples", type=int, default=5000)
    p.add_argument("--num-val-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--val-seed", type=int, default=424242)
    p.add_argument("--label-tol", type=float, default=1e-8)
    p.add_argument("--label-max-iter", type=int, default=500)
    # model / training
    p.add_argument("--trunk", choices=["paper_fixed", "fourier_fixed"], default="paper_fixed")
    p.add_argument("--p", type=int, default=128)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--trunk-width", type=int, default=256)
    p.add_argument("--trunk-depth", type=int, default=3)
    p.add_argument("--lr-schedule", action="store_true")
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--patience", type=int, default=10000)
    p.add_argument("--label-cache", default=None, help="optional path to cache generated labels")
    p.add_argument(
        "--checkpoint-policy",
        choices=["best_val", "min_train"],
        default="best_val",
        help="checkpoint whose trunk is used to construct tb_metadata.json",
    )
    # selection
    p.add_argument("--m", type=int, default=15)
    p.add_argument("--qr-epsilon", type=float, default=1e-10)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not np.isfinite(args.qr_epsilon) or float(args.qr_epsilon) < 0.0:
        raise ValueError(f"qr epsilon must be finite and non-negative, got {args.qr_epsilon}")
    from learned_tg.runtime import new_output
    out_dir = new_output(args.output_dir)
    device = torch.device(args.device)

    # Operator/solver namespace for label generation (cholesky local solver is
    # exact on subdomains and used only for offline high-accuracy labels; the
    # local solves here are used only for offline label generation).
    op_args = argparse.Namespace(
        device=args.device,
        k_1d=args.k_1d,
        subdomain_solver="cholesky",
        transmissibility=args.transmissibility,
        contrast=args.contrast,
        topology_mode=args.topology_mode,
        interface_coeff_mode="hard",
        interface_eps_factor=1.0,
        max_shape_retries=200,
        n=args.n,
        pre_smooth_steps=1,
        post_smooth_steps=1,
        omega=0.8,
        fcg_m_max=50,
        seed=args.seed,
        num_probes=50,
        snapshot_smooth_steps=50,
        snapshot_smoother="asm",
        gmg_num_levels=None,
        gmg_min_coarse_size=4,
        gmg_smoother="jacobi",
        gmg_pre_smooth_steps=2,
        gmg_post_smooth_steps=2,
        gmg_omega=0.8,
        gmg_coarse_solve="direct",
    )

    cache_path = Path(args.label_cache) if args.label_cache else None
    if cache_path and cache_path.exists():
        print(f"[train] loading cached labels from {cache_path}", flush=True)
        cache = torch.load(str(cache_path), map_location="cpu", weights_only=True)
        a_tr, phi_tr, train_u = cache["a_tr"], cache["phi_tr"], cache["train_u"]
        a_val, phi_val, val_u = cache["a_val"], cache["phi_val"], cache["val_u"]
        gen_time, label_time = 0.0, 0.0
        train_infos, val_infos = None, None
    else:
        t0 = time.time()
        print(f"[train] generating {args.num_train_samples} train samples (n={args.n}, island, contrast={args.contrast:g})", flush=True)
        train_samples = generate_interface_samples_for_grid(
            args.n, op_args, args.num_train_samples, args.seed
        )
        val_samples = generate_interface_samples_for_grid(
            args.n, op_args, args.num_val_samples, args.val_seed
        )
        gen_time = time.time() - t0

        print("[train] generating certified solution labels", flush=True)
        t0 = time.time()
        train_u, train_infos = generate_certified_labels(
            train_samples, op_args, tol=args.label_tol, max_iter=args.label_max_iter, device=device
        )
        val_u, val_infos = generate_certified_labels(
            val_samples, op_args, tol=args.label_tol, max_iter=args.label_max_iter, device=device
        )
        label_time = time.time() - t0

        # Certification summary
        all_infos = train_infos + val_infos
        failed = [i for i in all_infos if not i.converged]
        if failed:
            print(f"[train] WARNING: {len(failed)} labels failed certification; excluding them", flush=True)
        print(
            f"[train] label gen: train={args.num_train_samples} val={args.num_val_samples} "
            f"failed={len(failed)} time={label_time:.1f}s", flush=True
        )

        # Filter failed labels (never silently include them)
        def _filter(samples, u, infos):
            ok = [i for i in range(len(samples)) if infos[i].converged and torch.isfinite(u[i]).all()]
            return [samples[i] for i in ok], u[ok], [infos[i] for i in ok]

        train_samples, train_u, _ = _filter(train_samples, train_u, train_infos)
        val_samples, val_u, _ = _filter(val_samples, val_u, val_infos)

        a_tr, phi_tr, coords_tr = _stack_fields(train_samples)
        a_val, phi_val, _ = _stack_fields(val_samples)

        if cache_path:
            torch.save(
                {"a_tr": a_tr, "phi_tr": phi_tr, "train_u": train_u,
                 "a_val": a_val, "phi_val": phi_val, "val_u": val_u},
                str(cache_path),
            )

    train_batches = list(_batches(a_tr, phi_tr, train_u, args.batch_size))
    val_batches = list(_batches(a_val, phi_val, val_u, args.batch_size))

    model = SolutionDeepONet(
        p=args.p,
        trunk=args.trunk,
        branch_dropout=args.dropout,
        trunk_hidden=args.trunk_width,
        trunk_depth=args.trunk_depth,
    )
    coords = cell_center_coords(args.n, device=torch.device("cpu"), dtype=torch.float32)

    print(
        f"[train] training SolutionDeepONet (p={args.p}, trunk={args.trunk}, "
        f"wd={args.weight_decay}, dropout={args.dropout}, width={args.trunk_width}, "
        f"schedule={args.lr_schedule})",
        flush=True,
    )
    t0 = time.time()
    result = train_solution_deeponet(
        model,
        coords,
        train_batches,
        val_batches,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_scheduler=args.lr_schedule,
        patience=args.patience,
        device=device,
        seed=args.seed,
    )
    train_time = time.time() - t0
    print(
        f"[train] done: best_val={result.best_val_loss:.6e} @ epoch {result.best_epoch}; "
        f"min_train={result.min_train_loss:.6e} @ epoch {result.min_train_epoch}",
        flush=True,
    )

    if result.best_state is None or result.min_train_state is None:
        raise RuntimeError("training did not produce both best-validation and minimum-training states")
    torch.save(result.best_state, out_dir / "checkpoint_best.pt")
    torch.save(result.min_train_state, out_dir / "checkpoint_min_train.pt")
    selected_state = (
        result.best_state if args.checkpoint_policy == "best_val" else result.min_train_state
    )
    selected_checkpoint = (
        "checkpoint_best.pt"
        if args.checkpoint_policy == "best_val"
        else "checkpoint_min_train.pt"
    )
    model.load_state_dict(selected_state)

    # Retained TB-SVD selection on the offline selection grid (n=63).
    print(
        f"[train] extracting TB-SVD trunk basis from {selected_checkpoint}",
        flush=True,
    )
    model.eval().to(device)
    with torch.no_grad():
        T_sel = model.trunk_matrix(
            cell_center_coords(args.n, device=device, dtype=torch.float32)
        ).detach().cpu().to(torch.float64).numpy()
    h = 1.0 / float(args.n)
    selection = select_trunk_basis(T_sel, args.m, cell_volume=h * h)

    meta = {
        "description": (
            "Independent implementation of the DeepONet trunk-basis method based on "
            "Kopaničáková and Karniadakis (2025). No official implementation was available."
        ),
        "p": args.p,
        "trunk": args.trunk,
        "branch_dropout": args.dropout,
        "trunk_width": args.trunk_width,
        "trunk_depth": args.trunk_depth,
        "checkpoint_policy": args.checkpoint_policy,
        "checkpoint_file": selected_checkpoint,
        "m": args.m,
        "qr_epsilon": args.qr_epsilon,
        "selection_grid_n": args.n,
        "selection_cell_volume": h * h,
        "selections": [{"method": "svd", "m": selection.m, "W": selection.W.tolist()}],
        "train_split": {"num_train": args.num_train_samples, "seed": args.seed, "num_kept": int(a_tr.shape[0])},
        "val_split": {"num_val": args.num_val_samples, "seed": args.val_seed, "num_kept": int(a_val.shape[0])},
        "best_val_loss": result.best_val_loss,
        "best_epoch": result.best_epoch,
        "min_train_loss": result.min_train_loss,
        "min_train_epoch": result.min_train_epoch,
        "label_tol": args.label_tol,
    }

    with (out_dir / "tb_metadata.json").open("w") as fh:
        json.dump(meta, fh, indent=2)

    # config.yaml (used config)
    with (out_dir / "config.yaml").open("w") as fh:
        fh.write("# DeepONet trunk-basis (TB) training configuration\n")
        for k, v in sorted(vars(args).items()):
            fh.write(f"{k}: {v}\n")

    with (out_dir / "environment.txt").open("w") as fh:
        fh.write(f"python: {sys.version}\nplatform: {platform.platform()}\n")
        fh.write(f"torch: {torch.__version__}\ncuda_available: {torch.cuda.is_available()}\n")


    # train history
    with (out_dir / "train_history.csv").open("w") as fh:
        fh.write("epoch,train_loss,val_loss\n")
        for e, (tl, vl) in enumerate(zip(result.train_loss, result.val_loss)):
            fh.write(f"{e},{tl},{vl}\n")

    # label metrics (only available when labels were generated in this run)
    if train_infos is not None:
        with (out_dir / "label_generation_metrics.csv").open("w") as fh:
            fh.write("split,sample_id,converged,iterations,true_relative_residual,wall_time,retry\n")
            for split, infos in (("train", train_infos), ("val", val_infos)):
                for info in infos:
                    fh.write(
                        f"{split},{info.sample_id},{info.converged},{info.iterations},"
                        f"{info.true_relative_residual},{info.wall_time},{info.retry}\n"
                    )

    # offline costs
    with (out_dir / "offline_costs.csv").open("w") as fh:
        n_params = sum(p.numel() for p in model.parameters())
        fh.write("stage,seconds\n")
        fh.write(f"sample_generation,{gen_time}\n")
        fh.write(f"label_generation,{label_time}\n")
        fh.write(f"training,{train_time}\n")
        fh.write(f"total_offline,{gen_time + label_time + train_time}\n")
        fh.write(f"model_parameter_count,{n_params}\n")

    print(f"[train] wrote outputs to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
