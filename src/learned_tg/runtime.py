"""Portable configuration, model loading, and reproducible run setup."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from learned_tg.model.interface import Solver
from learned_tg.model.two_phase import TwoPhaseSolver
from learned_tg.training.checkpoint import _load_model_state


def read_config(path):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def initialize(seed=1234, device="cpu", threads=4, dtype=torch.float64):
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; select --device cpu.")
    torch.set_num_threads(threads)
    torch.set_default_dtype(dtype)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if target.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return target


def new_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_model(checkpoint, config, device="cpu", moving=False):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
    config = dict(config, device=str(device), use_cuda=torch.device(device).type == "cuda")
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        model = (TwoPhaseSolver if moving else Solver)(config).to(device)
    finally:
        torch.set_default_dtype(old_dtype)
    _load_model_state(model, str(checkpoint), device, strict=True, context="evaluation")
    model.eval()
    if next(model.parameters()).dtype != torch.float32:
        raise TypeError("The published networks use float32 parameters.")
    return model
