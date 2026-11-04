"""Solution-trained DeepONet for the trunk-basis (TB) baseline.

This module defines an *independent* solution-DeepONet

    u_hat(mu; x) = sum_{j=1}^{p} B_j(mu) * T_j(x),   p = 128,

trained against the relative discrete L2 solution error (eq. 2.10 of
Kopaničáková & Karniadakis, 2025).  The branch consumes the parameter fields
``[log10(a), phi]``; the trunk consumes *coordinates only*, so the extracted
trunk span ``Ran(T_h)`` is independent of the parameter ``mu``.

Implementation choices (not specified by the paper) are documented inline:
  * branch architecture mirrors the project's existing encoder pattern;
  * trunk activation/widths follow the paper's JumpDiff trunk (tanh MLP);
  * single-precision training (the project's learned model is also float32).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Coordinate helpers
# --------------------------------------------------------------------------- #
def cell_center_coords(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return cell-center coordinates (N, 2) for an n x n uniform grid.

    Matches ``data/gen_data_interface.py``: h = 1/n and cell centers at
    (i + 0.5)/n.  Column order is (x, y); rows follow row-major (i, j).
    """
    h = 1.0 / float(n)
    x_1d = torch.linspace(h / 2.0, 1.0 - h / 2.0, n, device=device, dtype=dtype)
    Y, X = torch.meshgrid(x_1d, x_1d, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)  # (N, 2)


# --------------------------------------------------------------------------- #
# Trunk networks (coordinate-only)
# --------------------------------------------------------------------------- #
class TrunkPaperFixed(nn.Module):
    """Coordinate-only tanh MLP trunk following the paper's JumpDiff trunk.

    Architecture 2 -> 256 -> 256 -> 256 -> 128 (tanh).  No Fourier features, no
    boundary mask, no parameter input.  Cell-centered FVM has no Dirichlet
    boundary DOFs, so no nodal zero-boundary mask is applied.
    """

    def __init__(self, p: int = 128, hidden: int = 256, depth: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = 2
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.Tanh())
            in_dim = hidden
        layers.append(nn.Linear(hidden, p))
        self.net = nn.Sequential(*layers)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (N, 2) -> (N, p)
        return self.net(coords)


class TrunkFourierFixed(nn.Module):
    """Enhanced coordinate-only trunk with Fourier features (still mu-independent).

    This is *not* the paper-style trunk; it is an implementation choice used only
    as an optional enhanced baseline (``TB-Fourier-fixed``).
    """

    def __init__(self, p: int = 128, hidden: int = 256, depth: int = 3, freqs: int = 6):
        super().__init__()
        self.freqs = freqs
        input_dim = 2 + 4 * freqs  # sin/cos of 2^k * pi * x and * y
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.GELU())
            in_dim = hidden
        layers.append(nn.Linear(hidden, p))
        self.net = nn.Sequential(*layers)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        features = [x, y]
        if self.freqs > 0:
            f = (2.0 ** torch.arange(self.freqs, device=coords.device, dtype=coords.dtype) * math.pi).view(1, -1)
            xp = x * f
            yp = y * f
            features.extend([torch.sin(xp), torch.cos(xp), torch.sin(yp), torch.cos(yp)])
        return self.net(torch.cat(features, dim=1))


# --------------------------------------------------------------------------- #
# Branch network (parameter encoder)
# --------------------------------------------------------------------------- #
class BranchEncoder(nn.Module):
    """CNN encoder over ``[log10(a), phi]`` -> p-dimensional branch coefficients.

    Implementation choice: mirrors the input encoding of the project's existing
    branch encoder (``log10(a).clamp(min)`` concatenated with ``phi``, resized to
    a canonical 64x64 image), then applies a small convolutional stack and an MLP
    head to a p-dimensional latent.
    """

    def __init__(
        self,
        p: int = 128,
        channels: Sequence[int] = (32, 64, 128),
        latent_dim: int = 128,
        canonical_input_size: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.canonical_input_size = canonical_input_size
        blocks: List[nn.Module] = []
        in_ch = 2
        for out_ch in channels:
            blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False))
            blocks.append(nn.GroupNorm(1, out_ch))
            blocks.append(nn.GELU())
            blocks.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False))
            blocks.append(nn.GroupNorm(1, out_ch))
            blocks.append(nn.GELU())
            if dropout > 0:
                blocks.append(nn.Dropout2d(dropout))
            in_ch = out_ch
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        final_ch = int(channels[-1])
        head_blocks: List[nn.Module] = [nn.Linear(final_ch, latent_dim), nn.GELU()]
        if dropout > 0:
            head_blocks.append(nn.Dropout(dropout))
        head_blocks.append(nn.Linear(latent_dim, p))
        self.head = nn.Sequential(*head_blocks)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in [*self.conv, *self.head]:
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, a_field: torch.Tensor, phi_field: torch.Tensor) -> torch.Tensor:
        # a_field, phi_field: (B, 1, n, n) or (B, n, n)
        a_img = self._as_4d(a_field)
        phi_img = self._as_4d(phi_field)
        log_a = torch.log10(a_img.clamp(min=1e-12))
        inputs = torch.cat([log_a, phi_img], dim=1)  # (B, 2, n, n)
        inputs = F.interpolate(
            inputs,
            size=(self.canonical_input_size, self.canonical_input_size),
            mode="bilinear",
            align_corners=False,
        )
        h = self.pool(self.conv(inputs)).flatten(1)
        return self.head(h)  # (B, p)

    @staticmethod
    def _as_4d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x.unsqueeze(1)
        if x.dim() == 4 and x.shape[1] == 1:
            return x
        if x.dim() == 2:
            n = int(round(math.sqrt(x.shape[-1])))
            return x.reshape(x.shape[0], 1, n, n)
        raise ValueError(f"unexpected field shape {tuple(x.shape)}")


# --------------------------------------------------------------------------- #
# Solution DeepONet
# --------------------------------------------------------------------------- #
class SolutionDeepONet(nn.Module):
    """DeepONet approximating the PDE solution u(mu; x) = B(mu)^T T(x)."""

    def __init__(
        self,
        p: int = 128,
        trunk: str = "paper_fixed",
        branch_channels: Sequence[int] = (32, 64, 128),
        branch_latent_dim: int = 128,
        branch_dropout: float = 0.0,
        trunk_hidden: int = 256,
        trunk_depth: int = 3,
        trunk_freqs: int = 6,
    ):
        super().__init__()
        self.p = int(p)
        if trunk == "paper_fixed":
            self.trunk = TrunkPaperFixed(p=p, hidden=trunk_hidden, depth=trunk_depth)
        elif trunk == "fourier_fixed":
            self.trunk = TrunkFourierFixed(p=p, hidden=trunk_hidden, depth=trunk_depth, freqs=trunk_freqs)
        else:
            raise ValueError(f"unknown trunk {trunk!r}")
        self.branch = BranchEncoder(
            p=p, channels=branch_channels, latent_dim=branch_latent_dim, dropout=branch_dropout
        )

    def trunk_matrix(self, coords: torch.Tensor) -> torch.Tensor:
        """Evaluate the (coordinate-only) trunk: (N, 2) -> (N, p)."""
        return self.trunk(coords)

    def forward(
        self,
        a_field: torch.Tensor,
        phi_field: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        B = self.branch(a_field, phi_field)          # (B, p)
        T = self.trunk_matrix(coords)                # (N, p)
        return torch.matmul(B, T.transpose(0, 1))    # (B, N)


def relative_l2_solution_loss(
    u_hat: torch.Tensor,
    u_h: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Relative discrete L2 solution error (paper eq. 2.10).

    The cell-centered FVM has no mass matrix; a uniform cell volume (h^2)
    weights every DOF equally and cancels in the ratio, so a plain Euclidean
    norm is used.
    """
    num = torch.sum((u_hat - u_h) ** 2, dim=1)
    den = torch.sum(u_h ** 2, dim=1).clamp_min(float(eps))
    return torch.mean(num / den)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainingResult:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = -1
    best_state: Optional[dict] = None
    min_train_loss: float = float("inf")
    min_train_epoch: int = -1
    min_train_state: Optional[dict] = None


def train_solution_deeponet(
    model: SolutionDeepONet,
    coords: torch.Tensor,
    train_batches: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    val_batches: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    epochs: int = 500,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    use_scheduler: bool = False,
    patience: int = 10000,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
    callback: Optional[Callable[[int, float, float], None]] = None,
) -> TrainingResult:
    """Train a SolutionDeepONet against relative L2 solution labels.

    Each batch is a tuple (a_field, phi_field, u_h) with a_field/phi_field of
    shape (B, n, n) or (B, 1, n, n) and u_h of shape (B, N).  The trunk is
    evaluated on the shared ``coords``.  Its *input* is parameter-independent,
    but its weights change during training, so validation always recomputes the
    current trunk matrix after the epoch's optimizer updates.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs))
        if use_scheduler
        else None
    )
    coords = coords.to(device)

    result = TrainingResult()
    best_val = float("inf")
    best_epoch = -1
    best_state: Optional[dict] = None
    min_train = float("inf")
    min_train_epoch = -1
    min_train_state: Optional[dict] = None
    no_improve = 0

    def _loss(batches: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> float:
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            T_eval = model.trunk_matrix(coords)  # current post-update trunk, (N, p)
            for a, phi, u_h in batches:
                a = a.to(device)
                phi = phi.to(device)
                u_h = u_h.to(device)
                B = model.branch(a, phi)
                u_hat = torch.matmul(B, T_eval.transpose(0, 1))
                total += float(relative_l2_solution_loss(u_hat, u_h).item()) * a.shape[0]
                count += int(a.shape[0])
        return total / max(count, 1)

    for epoch in range(int(epochs)):
        model.train()
        running = 0.0
        count = 0
        for a, phi, u_h in train_batches:
            a = a.to(device)
            phi = phi.to(device)
            u_h = u_h.to(device)
            optimizer.zero_grad()
            u_hat = model(a, phi, coords)
            loss = relative_l2_solution_loss(u_hat, u_h)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * a.shape[0]
            count += int(a.shape[0])
        if scheduler is not None:
            scheduler.step()
        train_loss = running / max(count, 1)
        if train_loss < min_train:
            min_train = float(train_loss)
            min_train_epoch = epoch
            min_train_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        val_loss = _loss(val_batches) if val_batches else float("nan")
        result.train_loss.append(train_loss)
        result.val_loss.append(val_loss)

        if val_batches and val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        elif val_batches:
            no_improve += 1
            if no_improve >= int(patience):
                break

        if callback is not None:
            callback(epoch, train_loss, val_loss)

    result.best_val_loss = best_val
    result.best_epoch = best_epoch
    result.best_state = best_state
    result.min_train_loss = min_train
    result.min_train_epoch = min_train_epoch
    result.min_train_state = min_train_state
    if best_state is not None:
        model.load_state_dict(best_state)
    return result
