"""Dataset generation and paired-loss model training with explicit inputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from learned_tg.data.gen_data_interface import create_cached_family_data_loader
from learned_tg.data.gen_data_two_phase import create_cached_two_phase_family_data_loader
from learned_tg.model.interface import Solver
from learned_tg.model.two_phase import TwoPhaseSolver
from learned_tg.runtime import initialize, new_output, read_config
from learned_tg.solvers.static import dump
from learned_tg.training.batches import (
    MultiResolutionFamilyLoader, _unpack_family_batch, _move_family_batch_to_device, _unwrap_loss_output,
)
from learned_tg.training.checkpoint import _load_model_state
from learned_tg.training.moving_loss import _compute_loss


def make_loader(config, device, output, grid_override=None, sample_override=None):
    config = dict(config, device=str(device), dataset_cache_dir=str(output / "datasets"),
                  preload_dataset_to_gpu=False, num_workers=0, dataloader_num_workers=0,
                  pin_memory=device.type == "cuda", persistent_workers=False)
    grids = [grid_override] if grid_override else config.get("train_grid_sizes") or [config.get("n", 63)]
    counts = ([sample_override] * len(grids) if sample_override else
              config.get("train_samples_per_grid") if config.get("train_grid_sizes") and not grid_override else
              [config.get("num_train_samples", 5000)])
    if len(counts) != len(grids):
        raise ValueError("Each training grid needs its own sample count.")
    loaders = []
    for grid, samples in zip(grids, counts):
        if config.get("problem") == "moving":
            loader, _ = create_cached_two_phase_family_data_loader(config=dict(config, n=grid, ny=grid), batch_size=config["batch_size"],
                n=grid, M_smooth=config["M_smooth"], num_samples=samples, num_probes=config["num_probes"], split="train")
        else:
            loader, _ = create_cached_family_data_loader(config=config, batch_size=config["batch_size"],
                n=grid, M_smooth=config["M_smooth"], num_samples=samples, num_probes=config["num_probes"],
                contrast=config["contrast"], split="train")
        loaders.append(loader)
    weights = config.get("train_grid_weights") or [1.0] * len(loaders)
    if len(weights) != len(loaders):
        weights = [1.0] * len(loaders)
    return loaders[0] if len(loaders) == 1 else MultiResolutionFamilyLoader(loaders, weights, seed=config["seed"])


def run(args):
    config = read_config(args.config)
    if config.get("requires_initial_checkpoint") and not args.initialize_from and not args.from_scratch:
        raise ValueError("This stored training recipe uses pretrained initialization. Pass --initialize-from, "
                         "or use --from-scratch for a new model (not the reported checkpoint).")
    device = initialize(config.get("seed", 1234), args.device, args.threads, dtype=torch.float32)
    output = new_output(args.output)
    moving = config.get("problem") == "moving"
    config.update(device=str(device), dataset_cache_dir=str(output / "datasets"), load_checkpoint=False,
                  subdomain_solver="cholesky", snapshot_smoother="asm", vcycle_compile=False)
    if args.grid:
        config["n"] = args.grid
        if moving:
            config["ny"] = args.grid
    model = (TwoPhaseSolver if moving else Solver)(config).to(device)
    if args.initialize_from:
        _load_model_state(model, args.initialize_from, device, strict=True, context="training initialization")
    loader = make_loader(config, device, output, args.grid, args.samples)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config.get("step_size", 50)),
                                               gamma=float(config.get("gamma", 0.5)))
    epochs = args.epochs or int(config.get("epoch", 500))
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    dump(output / "config.json", config)
    with (output / "training.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "loss", "learning_rate"])
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            model.train()
            total, batches = 0.0, 0
            for batch in loader:
                optimizer.zero_grad()
                if moving:
                    result = _compute_loss(model, batch, config, device, return_stats=True)
                else:
                    a, phi, x, y, family, svd, white, steps = _move_family_batch_to_device(
                        _unpack_family_batch(batch, config=config), device)
                    result = model(a, phi, x, y, family, epoch, svd_basis=svd,
                                   e_family_white=white, e_family_white_metric=config.get("whiten_metric", "a"),
                                   return_stats=True)
                loss, _ = _unwrap_loss_output(result)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Training produced a non-finite loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 10.0)))
                optimizer.step()
                total += float(loss.detach())
                batches += 1
            if config.get("use_scheduler", True):
                scheduler.step()
            for group in optimizer.param_groups:
                group["lr"] = max(group["lr"], float(config.get("min_lr", 1e-4)))
            record = dict(epoch=epoch, loss=total / batches, learning_rate=optimizer.param_groups[0]["lr"])
            writer.writerow(record)
            stream.flush()
            print(json.dumps(record), flush=True)
            if epoch % int(config.get("ckpt_freq", 10)) == 0 or epoch == epochs:
                state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                torch.save(state, checkpoints / f"epoch{epoch:04d}.pth")
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, output / "model_last.pth")
