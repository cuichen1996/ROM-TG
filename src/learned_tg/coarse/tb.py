"""Numerical routines used by the paper; no experiment orchestration."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import torch
from learned_tg.baselines.deeponet_tb.basis_selection import SelectionResult
from learned_tg.baselines.deeponet_tb.solution_deeponet import SolutionDeepONet

METHOD_TB_SVD = "TB-SVD"

def _load_tb_model(checkpoint: Path, meta: Dict[str, Any], device: torch.device) -> SolutionDeepONet:
    model = SolutionDeepONet(
        p=int(meta.get("p", 128)),
        trunk=str(meta.get("trunk", "paper_fixed")),
        branch_dropout=float(meta.get("branch_dropout", 0.0)),
        trunk_hidden=int(meta.get("trunk_width", 256)),
        trunk_depth=int(meta.get("trunk_depth", 3)),
    ).float()
    state = torch.load(str(checkpoint), map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def _reconstruct_svd_selection(meta: Dict[str, Any]) -> Tuple[str, SelectionResult]:
    """Load the sole retained SVD selection, including from legacy metadata."""
    svd_entries = [s for s in meta["selections"] if s.get("method") == "svd"]
    if len(svd_entries) != 1:
        raise ValueError(
            "TB metadata must contain exactly one SVD selection, "
            f"found {len(svd_entries)}"
        )
    entry = svd_entries[0]
    selection = SelectionResult(
        method="svd",
        W=np.asarray(entry["W"]),
        m=int(entry["m"]),
    )
    return METHOD_TB_SVD, selection
