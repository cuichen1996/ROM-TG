import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from learned_tg.model.losses import primitives as coarse_loss
from learned_tg.model.losses.subspace import SubspaceLoss
from learned_tg.model.losses.base import normalize_snapshot_family
from learned_tg.model.deeponet import InterfaceDeepONet
from learned_tg.model.fno import InterfaceFNO
from learned_tg.model.basis_utils.smoothing import smooth_basis as _smooth_basis_impl
from learned_tg.model.operators.two_phase_fvm import MobilityFVMOperatorASM, get_fixed_subdomains_params_1d, _sync_if_needed, _timed, CGInfo
from learned_tg.model.operators.two_phase_gmg import StandardGMGHierarchy, gmg_down, gmg_up


@dataclass
class PressureSolveInfo:
    pressure_method: str
    num_iter: int
    rel_res: float
    converged: bool
    residual_history: List[float]
    operator_setup_time: float = 0.0
    preconditioner_setup_time: float = 0.0
    setup_time: float = 0.0
    solve_time: float = 0.0
    basis_dim: int = 0
    fallback_used: bool = False
    tg_iter: int = 0
    pcg_iter: int = 0
    preconditioner: str = "asm"
    inference_time: float = 0.0
    oracle_snapshot_setup_time: float = 0.0
    total_with_inference_time: float = 0.0
    nn_coeff_mode: str = "mobility"
    coarse_solver: str = "none"

    def __post_init__(self) -> None:
        self.operator_setup_time = float(self.operator_setup_time)
        self.preconditioner_setup_time = float(self.preconditioner_setup_time)
        self.inference_time = float(self.inference_time)
        self.solve_time = float(self.solve_time)
        self.oracle_snapshot_setup_time = float(self.oracle_snapshot_setup_time)
        self.setup_time = float(self.operator_setup_time + self.preconditioner_setup_time)
        self.total_with_inference_time = float(
            self.setup_time
            + self.inference_time
            + self.solve_time
            + self.oracle_snapshot_setup_time
        )

    @classmethod
    def from_pcg(
        cls,
        cg_info: CGInfo,
        *,
        solve_time: float,
        operator_setup_time: float = 0.0,
        preconditioner_setup_time: float = 0.0,
        preconditioner: str = "asm",
        nn_coeff_mode: str = "mobility",
        coarse_solver: str = "none",
    ) -> "PressureSolveInfo":
        return cls(
            pressure_method="pcg",
            num_iter=int(cg_info.num_iter),
            rel_res=float(cg_info.rel_res),
            converged=bool(cg_info.converged),
            residual_history=[float(x) for x in cg_info.residual_history],
            operator_setup_time=float(operator_setup_time),
            preconditioner_setup_time=float(preconditioner_setup_time),
            solve_time=float(solve_time),
            basis_dim=0,
            fallback_used=False,
            tg_iter=0,
            pcg_iter=int(cg_info.num_iter),
            preconditioner=str(preconditioner),
            inference_time=0.0,
            nn_coeff_mode=str(nn_coeff_mode),
            coarse_solver=str(coarse_solver),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "pressure_method": self.pressure_method,
            "num_iter": int(self.num_iter),
            "rel_res": float(self.rel_res),
            "converged": bool(self.converged),
            "residual_history": [float(x) for x in self.residual_history],
            "operator_setup_time": float(self.operator_setup_time),
            "preconditioner_setup_time": float(self.preconditioner_setup_time),
            "setup_time": float(self.setup_time),
            "solve_time": float(self.solve_time),
            "basis_dim": int(self.basis_dim),
            "fallback_used": bool(self.fallback_used),
            "tg_iter": int(self.tg_iter),
            "pcg_iter": int(self.pcg_iter),
            "preconditioner": self.preconditioner,
            "inference_time": float(self.inference_time),
            "oracle_snapshot_setup_time": float(self.oracle_snapshot_setup_time),
            "total_with_inference_time": float(self.total_with_inference_time),
            "total_time": float(self.total_with_inference_time),
            "nn_coeff_mode": self.nn_coeff_mode,
            "coarse_solver": self.coarse_solver,
        }


def _symmetrize_matrix(M: torch.Tensor) -> torch.Tensor:
    return coarse_loss.symmetrize_matrix(M)


def _expand_batch_dim(tensor: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1 and batch_size > 1:
        return tensor.expand(batch_size, *tensor.shape[1:])
    raise ValueError(
        f"{name} batch mismatch: expected batch_size={batch_size}, got {tensor.shape[0]}"
    )


def _call_asm_smooth(
    op: "MobilityFVMOperatorASM",
    x: torch.Tensor,
    f: torch.Tensor,
    num_steps: int,
    omega: float,
) -> torch.Tensor:
    if int(num_steps) <= 0:
        return x.to(device=op.device, dtype=torch.float64)
    try:
        return op.asm_smooth(x, f, steps=int(num_steps), omega=float(omega))
    except TypeError:
        return op.asm_smooth(x, f, M_smooth=int(num_steps), omega=float(omega))


def _two_phase_smoother_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode in {"asm_avg", "ras"}:
        return "asm"
    if mode in {"none", "jacobi", "asm", "gmg"}:
        return mode
    raise ValueError(
        "two-phase smoother must be one of {'none', 'asm', 'jacobi', 'gmg'}, "
        f"got {mode!r}"
    )


def _gmg_hierarchy_from_config(
    op: "MobilityFVMOperatorASM",
    config: Optional[Dict[str, object]] = None,
) -> "StandardGMGHierarchy":
    config = {} if config is None else dict(config)
    return StandardGMGHierarchy(
        op,
        num_levels=config.get("num_levels", config.get("gmg_num_levels")),
        min_coarse_size=int(config.get("min_coarse_size", config.get("gmg_min_coarse_size", 4))),
        smoother=str(config.get("smoother", config.get("gmg_smoother", "jacobi"))),
        pre_smooth_steps=int(config.get("pre_smooth_steps", config.get("gmg_pre_smooth_steps", 2))),
        post_smooth_steps=int(config.get("post_smooth_steps", config.get("gmg_post_smooth_steps", 2))),
        omega=float(config.get("omega", config.get("gmg_omega", 0.8))),
        coarse_solve=str(config.get("coarse_solve", config.get("gmg_coarse_solve", "direct"))),
    )


def _call_two_phase_smooth(
    op: "MobilityFVMOperatorASM",
    x: torch.Tensor,
    b: torch.Tensor,
    num_steps: int,
    omega: float,
    mode: str = "asm",
    *,
    gmg_hierarchy: Optional["StandardGMGHierarchy"] = None,
) -> torch.Tensor:
    mode = _two_phase_smoother_mode(mode)
    if int(num_steps) <= 0 or mode == "none":
        return x.to(device=op.device, dtype=torch.float64)
    if mode == "asm":
        return _call_asm_smooth(
            op,
            x,
            b,
            num_steps=int(num_steps),
            omega=float(omega),
        )

    x4, _ = op._to_4d(x)
    b4, _ = op._to_4d(b)
    if mode == "jacobi":
        for _ in range(int(num_steps)):
            r = b4 - op.forward(x4)
            x4 = x4 + float(omega) * op.apply_jacobi(r)
        return x4

    if gmg_hierarchy is None:
        raise ValueError(
            "two-grid smoother='gmg' requires a pre-built StandardGMGHierarchy. "
            "Construct it once before iteration and pass it through gmg_hierarchy."
        )
    for _ in range(int(num_steps)):
        r = b4 - op.forward(x4)
        z = gmg_hierarchy.apply(r)
        z4, _ = op._to_4d(z)
        x4 = x4 + float(omega) * z4
    return x4


# ============================================================
# Learned two-grid utilities for main.tex "重复求解场景下的进一步验证"
# ============================================================
def _apply_A_with_operator(op: "MobilityFVMOperatorASM", X: torch.Tensor) -> torch.Tensor:
    """
    Apply a matrix-free two-phase operator column-wise.

    Parameters
    ----------
    X:
        Shape (B, N, m) or (B, N, K), with N = op.nx * op.ny.
    """
    X = X.to(
        device=op.device,
        dtype=torch.float64,
    )
    if X.dim() != 3:
        raise ValueError(f"X must have shape (B, N, m), got {tuple(X.shape)}")

    batch_size, num_nodes, num_cols = X.shape
    expected_nodes = int(op.nx * op.ny)
    if num_nodes != expected_nodes:
        raise ValueError(
            f"X node count mismatch: expected N={expected_nodes}, got {num_nodes}"
        )

    X_img = X.transpose(1, 2).reshape(batch_size, num_cols, op.ny, op.nx)
    AX_img = op.forward(X_img)
    AX = AX_img.reshape(batch_size, num_cols, num_nodes).transpose(1, 2)
    return AX


def _smooth_basis_with_operator(
    op: "MobilityFVMOperatorASM",
    V: torch.Tensor,
    mode: str,
    num_steps: int,
    omega: float,
) -> torch.Tensor:
    """Thin wrapper around model.basis_utils.smoothing.smooth_basis."""
    return _smooth_basis_impl(op, V, mode=mode, num_steps=num_steps, omega=omega)


def _a_orthonormalize_from_operator(
    op: "MobilityFVMOperatorASM",
    V: torch.Tensor,
    reg: float = 1e-10,
) -> Dict[str, object]:
    V = V.to(
        device=op.device,
        dtype=torch.float64,
    )
    if V.dim() != 3:
        raise ValueError(f"V must have shape (B, N, m), got {tuple(V.shape)}")

    batch_size, num_nodes, basis_dim = V.shape
    AV = _apply_A_with_operator(op, V)
    Q, AQ, A_rom = coarse_loss.a_orthonormalize_from_AV(
        V,
        AV,
        reg=float(reg),
    )
    eye = torch.eye(basis_dim, device=Q.device, dtype=Q.dtype).unsqueeze(0)
    A_rom_reg = A_rom + float(reg) * eye

    try:
        coarse_factor = torch.linalg.cholesky(A_rom_reg)
        coarse_solver = "cholesky"
    except RuntimeError:
        coarse_factor = torch.linalg.pinv(A_rom_reg)
        coarse_solver = "pinv"

    return {
        "Q": Q,
        "AQ": AQ,
        "A_rom": A_rom,
        "A_rom_reg": A_rom_reg,
        "coarse_solver": coarse_solver,
        "coarse_factor": coarse_factor,
        "basis_dim": int(basis_dim),
        "num_nodes": int(num_nodes),
        "batch_size": int(batch_size),
        "orth_reg": float(reg),
        "coarse_basis_mode": "a_orthonormal",
    }


def _raw_galerkin_from_operator(
    op: "MobilityFVMOperatorASM",
    V: torch.Tensor,
    reg: float = 1e-10,
) -> Dict[str, object]:
    V = V.to(
        device=op.device,
        dtype=torch.float64,
    )
    if V.dim() != 3:
        raise ValueError(f"V must have shape (B, N, m), got {tuple(V.shape)}")

    batch_size, num_nodes, basis_dim = V.shape
    AV = _apply_A_with_operator(op, V)
    A_rom = _symmetrize_matrix(torch.matmul(V.transpose(1, 2), AV))
    eye = torch.eye(basis_dim, device=V.device, dtype=V.dtype).unsqueeze(0)
    A_rom_reg = A_rom + float(reg) * eye

    try:
        coarse_factor = torch.linalg.cholesky(A_rom_reg)
        coarse_solver = "cholesky"
    except RuntimeError:
        coarse_factor = torch.linalg.pinv(A_rom_reg)
        coarse_solver = "pinv"

    return {
        "Q": V,
        "AQ": AV,
        "A_rom": A_rom,
        "A_rom_reg": A_rom_reg,
        "coarse_solver": coarse_solver,
        "coarse_factor": coarse_factor,
        "basis_dim": int(basis_dim),
        "num_nodes": int(num_nodes),
        "batch_size": int(batch_size),
        "orth_reg": float(reg),
        "coarse_basis_mode": "raw",
    }


def prepare_two_phase_coarse_basis(
    op: "MobilityFVMOperatorASM",
    V: torch.Tensor,
    basis_smooth_mode: str = "none",
    basis_smooth_steps: int = 0,
    basis_smooth_omega: float = 0.6,
    orth_reg: float = 1e-10,
    coarse_basis_mode: str = "a_orthonormal",
) -> Dict[str, object]:
    V = torch.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0).to(
        device=op.device,
        dtype=torch.float64,
    )
    V_tilde = _smooth_basis_with_operator(
        op,
        V,
        mode=basis_smooth_mode,
        num_steps=int(basis_smooth_steps),
        omega=float(basis_smooth_omega),
    )
    mode = str(coarse_basis_mode).lower().replace("-", "_")
    if mode in {"raw", "galerkin", "raw_galerkin"}:
        prepared = _raw_galerkin_from_operator(op, V_tilde, reg=orth_reg)
    elif mode in {"a_orthonormal", "a_orth", "orthonormal", "online_a_orthonormal"}:
        prepared = _a_orthonormalize_from_operator(op, V_tilde, reg=orth_reg)
    else:
        raise ValueError(
            "coarse_basis_mode must be one of {'raw', 'a_orthonormal'}, "
            f"got {coarse_basis_mode!r}"
        )
    prepared.update({
        "basis_smooth_mode": str(basis_smooth_mode),
        "basis_smooth_steps": int(basis_smooth_steps),
        "basis_smooth_omega": float(basis_smooth_omega),
        "coarse_basis_mode": str(prepared.get("coarse_basis_mode", mode)),
    })
    return prepared


class TwoPhaseSolver(nn.Module):
    """
    Learned coarse-basis model for the two-phase closed-droplet pressure operator.

    This intentionally does not inherit from model.interface.Solver and never
    builds the static interface operator. Training losses are evaluated against
    MobilityFVMOperatorASM.
    """

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.problem_type = str(config.get("problem_type", "two_phase_closed_droplet"))
        self.m_dim = int(config.get("m_dim", 20))
        self.V_net_type = str(config.get("V_net", "fno")).lower()
        self.basis_mode = str(config.get("basis_mode", "direct")).lower()
        if self.basis_mode != "direct":
            raise ValueError("TwoPhaseSolver currently supports basis_mode='direct' only")
        self.two_phase_loss_type = str(config.get("two_phase_loss_type", "capture_whiten")).lower()
        self.enforce_zero_boundary = bool(config.get("enforce_zero_boundary", False))
        self.k_1d = int(config.get("k_1d", 10))
        self.transmissibility = str(config.get("transmissibility", "harmonic")).lower()
        self.target_overlap = float(config.get("target_overlap", 0.3))
        self.p_in = float(config.get("p_in", 1.0))
        self.p_out = float(config.get("p_out", 0.0))
        self.basis_smooth_mode = str(config.get("basis_smooth_mode", "none"))
        self.basis_smooth_steps = int(config.get("basis_smooth_steps", 0))
        self.basis_smooth_omega = float(config.get("basis_smooth_omega", 0.6))
        self.orth_reg = float(config.get("orth_reg", config.get("basis_orth_reg", 1e-10)))
        self.loss_reduction = str(config.get("loss_reduction", "per_sample_mean"))
        self.whiten_rank = config.get("whiten_rank", self.m_dim)
        self.whiten_reg = float(config.get("whiten_reg", 1e-10))
        self.whiten_metric = str(config.get("whiten_metric", "a"))

        deeponet_branch_latent_dim = config.get(
            "branch_latent_dim",
            config.get("branch_hidden_dim", 256),
        )
        deeponet_common_kwargs = dict(
            num_fractures=config.get("num_fractures", 1),
            m_dim=self.m_dim,
            p_dim=config.get("p_dim", 64),
            activation=config.get("activation", "gelu"),
            trunk_activation=config.get("trunk_activation"),
            branch_activation=config.get("branch_activation"),
            head_activation=config.get("head_activation"),
            trunk_width=config.get("trunk_width", 256),
            trunk_depth=config.get("trunk_depth", 4),
            trunk_num_fourier_freqs=config.get("trunk_num_fourier_freqs", 6),
            trunk_use_film=config.get("trunk_use_film", True),
            trunk_use_layernorm=config.get("trunk_use_layernorm", True),
            trunk_dropout=config.get("trunk_dropout", 0.0),
            trunk_use_grad_phi=config.get("trunk_use_grad_phi", False),
            branch_channels=config.get("branch_channels", [32, 64, 128, 128]),
            branch_blocks_per_stage=config.get("branch_blocks_per_stage", 2),
            branch_latent_dim=deeponet_branch_latent_dim,
            branch_norm_type=config.get("branch_norm_type", "group"),
            branch_use_se=config.get("branch_use_se", True),
            head_hidden_dim=config.get("head_hidden_dim", 256),
            head_depth=config.get("head_depth", 3),
            column_embedding_dim=config.get("column_embedding_dim", 32),
            enforce_zero_boundary=self.enforce_zero_boundary,
        )

        if self.V_net_type == "fno":
            self.V_net = InterfaceFNO(
                num_fractures=config.get("num_fractures", 1),
                m_dim=self.m_dim,
                modes=config.get("modes", 12),
                width=config.get("width", 32),
                activation=config.get("activation", "gelu"),
                enforce_zero_boundary=self.enforce_zero_boundary,
            )
        elif self.V_net_type == "deeponet":
            self.V_net = InterfaceDeepONet(
                use_column_embedding=config.get("use_column_embedding", False),
                **deeponet_common_kwargs,
            )
        else:
            raise ValueError(f"Unknown V_net type: {self.V_net_type!r}. Expected 'fno' or 'deeponet'.")

        capture_whiten = (self.two_phase_loss_type == "capture_whiten")
        subspace_loss_type = "svd" if self.two_phase_loss_type == "svd" else "capture"
        subspace_metric = "l2" if self.two_phase_loss_type == "svd" else "a"
        self.criterion = SubspaceLoss(
            loss_type=subspace_loss_type,
            metric=subspace_metric,
            capture_whiten=capture_whiten,
            capture_whiten_rank=self.whiten_rank if capture_whiten else None,
            capture_whiten_metric=self.whiten_metric if capture_whiten else None,
            log_extra_metrics=True,
            orth_reg=self.orth_reg,
            eps=float(self.config.get("loss_reg", 1e-12)),
            dtype=torch.float64,
        )

    def _align_model_inputs(self, a_field, phi_field):
        if a_field.dim() == 3:
            a_field = a_field.unsqueeze(1)
        if phi_field.dim() == 3:
            phi_field = phi_field.unsqueeze(1)
        return a_field, phi_field

    def _prepare_network_inputs(self, a_field, phi_field, X, Y):
        a_field, phi_field = self._align_model_inputs(a_field, phi_field)
        return (
            a_field,
            phi_field,
            a_field.to(torch.float32),
            phi_field.to(torch.float32),
            X.to(torch.float32),
            Y.to(torch.float32),
        )

    def _predict_raw_basis(self, a_field, phi_field, X, Y, active_dim=None):
        a_field, phi_field, a_net, phi_net, X_net, Y_net = self._prepare_network_inputs(
            a_field,
            phi_field,
            X,
            Y,
        )
        V = self.V_net(a_net, X_net, Y_net, phi_net)
        if active_dim is not None:
            active_dim = int(active_dim)
            if active_dim < 1 or active_dim > self.m_dim:
                raise ValueError(
                    f"active_dim must satisfy 1 <= active_dim <= {self.m_dim}, got {active_dim}"
                )
            V = V[:, :, :active_dim]
        return a_field, phi_field, torch.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float64)

    def predict_basis(self, a_field, phi_field, X, Y, active_dim=None, **_):
        _, _, V = self._predict_raw_basis(
            a_field,
            phi_field,
            X,
            Y,
            active_dim=active_dim,
        )
        return V

    def _build_operator_from_lambda(
        self,
        a_field,
        phi_field,
        *,
        build_local_solver: bool = False,
    ):
        lam, phi = self._align_model_inputs(a_field, phi_field)
        n = int(lam.shape[-1])
        ny = int(lam.shape[-2])

        return MobilityFVMOperatorASM(
            lam=lam,
            hx=1.0 / float(n),
            hy=1.0 / float(ny),
            p_left=self.p_in,
            p_right=self.p_out,
            kx_sub=self.k_1d,
            ky_sub=self.k_1d,
            target_overlap=self.target_overlap,
            build_local_solver=bool(build_local_solver),
            phi=phi,
            transmissibility=self.transmissibility,
        )

    def _loss_kwargs(self):
        return {
            "basis_smooth_mode": self.basis_smooth_mode,
            "basis_smooth_steps": self.basis_smooth_steps,
            "basis_smooth_omega": self.basis_smooth_omega,
            "orth_reg": self.orth_reg,
            "loss_reduction": self.loss_reduction,
            "whiten_rank": self.whiten_rank,
            "whiten_reg": self.whiten_reg,
            "whiten_metric": self.whiten_metric,
            "eps": float(self.config.get("loss_reg", 1e-12)),
        }

    def forward(self, *args, op=None, svd_basis=None, return_stats=False, **kwargs):
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            batch = list(args[0])
            if len(batch) == 6:
                a_field, phi_field, X, Y, e_family, extra = batch
                if torch.is_tensor(extra) and extra.dim() >= 3:
                    svd_basis = extra
            elif len(batch) >= 7:
                a_field, phi_field, X, Y, e_family, svd_basis, *_ = batch
            else:
                raise ValueError(f"Unexpected two-phase batch length: {len(batch)}")
        elif len(args) >= 5:
            a_field, phi_field, X, Y, e_family = args[:5]
            if len(args) >= 6:
                svd_basis = args[5]
        else:
            a_field = kwargs["a_field"]
            phi_field = kwargs["phi_field"]
            X = kwargs["X_flat"]
            Y = kwargs["Y_flat"]
            e_family = kwargs["e_family"]
            svd_basis = kwargs.get("svd_basis", svd_basis)

        active_dim = kwargs.get("active_dim", self.m_dim)
        V = self.predict_basis(a_field, phi_field, X, Y, active_dim=active_dim)
        if op is None and self.two_phase_loss_type != "svd":
            op = self._build_operator_from_lambda(a_field, phi_field)
        if (
            op is not None
            and self.basis_smooth_mode != "none"
            and self.basis_smooth_steps > 0
        ):
            V = _smooth_basis_impl(
                op, V,
                mode=self.basis_smooth_mode,
                num_steps=self.basis_smooth_steps,
                omega=self.basis_smooth_omega,
            )
        return self.criterion(
            V_raw=V,
            e_family=e_family,
            op_solver=op,
            svd_basis=svd_basis,
            return_stats=return_stats,
        )


def _validate_prepared_two_phase_coarse_basis(
    op: "MobilityFVMOperatorASM",
    prepared_coarse_basis: Dict[str, object],
    x: torch.Tensor,
) -> None:
    required_keys = {
        "Q",
        "AQ",
        "A_rom",
        "A_rom_reg",
        "coarse_solver",
        "coarse_factor",
        "basis_dim",
        "num_nodes",
        "batch_size",
    }
    missing = sorted(required_keys.difference(prepared_coarse_basis.keys()))
    if missing:
        raise KeyError(f"prepared_coarse_basis is missing keys: {missing}")

    Q = prepared_coarse_basis["Q"]
    if not torch.is_tensor(Q):
        raise TypeError("prepared_coarse_basis['Q'] must be a tensor")
    expected_nodes = int(op.nx * op.ny)
    if int(prepared_coarse_basis["num_nodes"]) != expected_nodes:
        raise ValueError(
            "prepared coarse basis node count does not match operator: "
            f"{prepared_coarse_basis['num_nodes']} vs {expected_nodes}"
        )
    batch_size = int(x.shape[0])
    prepared_batch = int(prepared_coarse_basis["batch_size"])
    if prepared_batch != batch_size:
        raise ValueError(
            "prepared coarse basis batch size mismatch: "
            f"{prepared_batch} vs {batch_size}"
        )
    if Q.shape != (
        prepared_batch,
        int(prepared_coarse_basis["num_nodes"]),
        int(prepared_coarse_basis["basis_dim"]),
    ):
        raise ValueError(
            "prepared_coarse_basis['Q'] shape is inconsistent with its metadata: "
            f"{tuple(Q.shape)}"
        )


def _solve_prepared_two_phase_coarse_system(
    prepared_coarse_basis: Dict[str, object],
    rhs_rom: torch.Tensor,
) -> torch.Tensor:
    coarse_solver = str(prepared_coarse_basis["coarse_solver"])
    coarse_factor = prepared_coarse_basis["coarse_factor"]
    if coarse_solver == "cholesky":
        return torch.cholesky_solve(rhs_rom, coarse_factor)
    if coarse_solver == "pinv":
        return torch.matmul(coarse_factor, rhs_rom)
    raise ValueError(f"Unknown coarse solver type: {coarse_solver!r}")


def two_phase_two_grid_sweep(
    op: "MobilityFVMOperatorASM",
    x: torch.Tensor,
    b: torch.Tensor,
    prepared_coarse_basis: Dict[str, object],
    pre_smooth_steps: int = 1,
    pre_smooth_omega: float = 0.8,
    post_smooth_steps: int = 0,
    post_smooth_omega: float = 0.8,
    smoother: str = "asm",
    pre_smoother: Optional[str] = None,
    post_smoother: Optional[str] = None,
    gmg_hierarchy: Optional["StandardGMGHierarchy"] = None,
) -> torch.Tensor:
    x = x.to(device=op.device, dtype=torch.float64)
    b = b.to(device=op.device, dtype=torch.float64)
    _validate_prepared_two_phase_coarse_basis(op, prepared_coarse_basis, x)

    pre_mode = _two_phase_smoother_mode(pre_smoother if pre_smoother is not None else smoother)
    post_mode = _two_phase_smoother_mode(post_smoother if post_smoother is not None else smoother)

    if int(pre_smooth_steps) > 0:
        x = _call_two_phase_smooth(
            op,
            x,
            b,
            num_steps=int(pre_smooth_steps),
            omega=float(pre_smooth_omega),
            mode=pre_mode,
            gmg_hierarchy=gmg_hierarchy,
        )

    batch_size = int(x.shape[0])
    r = b - op.forward(x)
    r_vec = r.reshape(batch_size, op.nx * op.ny, 1)
    Q = prepared_coarse_basis["Q"]
    rhs_rom = torch.matmul(Q.transpose(1, 2), r_vec)
    y = _solve_prepared_two_phase_coarse_system(prepared_coarse_basis, rhs_rom)
    correction = torch.matmul(Q, y).reshape(batch_size, 1, op.ny, op.nx)
    x = x + correction

    if int(post_smooth_steps) > 0:
        x = _call_two_phase_smooth(
            op,
            x,
            b,
            num_steps=int(post_smooth_steps),
            omega=float(post_smooth_omega),
            mode=post_mode,
            gmg_hierarchy=gmg_hierarchy,
        )
    return x


@torch.no_grad()
def solve_pressure_learned_two_grid(
    op: "MobilityFVMOperatorASM",
    b: torch.Tensor,
    V: Optional[torch.Tensor],
    x0: Optional[torch.Tensor] = None,
    tol: float = 1e-8,
    max_iter: int = 100,
    pre_smooth_steps: int = 1,
    pre_smooth_omega: float = 0.8,
    post_smooth_steps: int = 0,
    post_smooth_omega: float = 0.8,
    smoother: str = "asm",
    pre_smoother: Optional[str] = None,
    post_smoother: Optional[str] = None,
    basis_smooth_mode: str = "none",
    basis_smooth_steps: int = 0,
    basis_smooth_omega: float = 0.6,
    orth_reg: float = 1e-10,
    coarse_basis_mode: str = "a_orthonormal",
    fallback_to_pcg: bool = True,
    fallback_pcg_maxiter: int = 500,
    gmg_hierarchy: Optional["StandardGMGHierarchy"] = None,
    gmg_config: Optional[Dict[str, object]] = None,
    prepared_coarse_basis: Optional[Dict[str, object]] = None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    b4, _ = op._to_4d(b)
    x = torch.zeros_like(b4) if x0 is None else op._to_4d(x0)[0]
    pre_mode = _two_phase_smoother_mode(pre_smoother if pre_smoother is not None else smoother)
    post_mode = _two_phase_smoother_mode(post_smoother if post_smoother is not None else smoother)

    def _prepare_two_grid_state():
        if prepared_coarse_basis is None:
            if V is None:
                raise ValueError("V must be provided when prepared_coarse_basis is not reused")
            prepared = prepare_two_phase_coarse_basis(
                op,
                V,
                basis_smooth_mode=basis_smooth_mode,
                basis_smooth_steps=basis_smooth_steps,
                basis_smooth_omega=basis_smooth_omega,
                orth_reg=orth_reg,
                coarse_basis_mode=coarse_basis_mode,
            )
        else:
            prepared = prepared_coarse_basis
        hierarchy = gmg_hierarchy
        if hierarchy is None and (pre_mode == "gmg" or post_mode == "gmg"):
            hierarchy = _gmg_hierarchy_from_config(op, gmg_config)
        return prepared, hierarchy

    (prepared_coarse_basis, gmg_hierarchy), setup_time = _timed(
        op.device,
        _prepare_two_grid_state,
    )

    batch_size = int(b4.shape[0])
    bnorm = torch.linalg.norm(b4.reshape(batch_size, -1), dim=1).clamp_min(1e-30)
    residual_history: List[float] = []
    tg_iter = 0
    pcg_iter = 0
    fallback_used = False
    converged = False

    r0 = b4 - op.forward(x)
    rel0 = torch.max(torch.linalg.norm(r0.reshape(batch_size, -1), dim=1) / bnorm).item()
    if rel0 < tol:
        converged = True
        rel_res = float(rel0)
    else:
        _sync_if_needed(op.device)
        solve_tic = time.perf_counter()
        for tg_iter in range(1, int(max_iter) + 1):
            x = two_phase_two_grid_sweep(
                op,
                x,
                b4,
                prepared_coarse_basis,
                pre_smooth_steps=pre_smooth_steps,
                pre_smooth_omega=pre_smooth_omega,
                post_smooth_steps=post_smooth_steps,
                post_smooth_omega=post_smooth_omega,
                smoother=smoother,
                pre_smoother=pre_mode,
                post_smoother=post_mode,
                gmg_hierarchy=gmg_hierarchy,
            )
            r = b4 - op.forward(x)
            rel = torch.max(torch.linalg.norm(r.reshape(batch_size, -1), dim=1) / bnorm).item()
            residual_history.append(float(rel))
            if rel < tol:
                converged = True
                break
        _sync_if_needed(op.device)
        solve_time = time.perf_counter() - solve_tic
        rel_res = residual_history[-1] if residual_history else float(rel0)

        if not converged and bool(fallback_to_pcg):
            fallback_used = True
            _sync_if_needed(op.device)
            pcg_tic = time.perf_counter()
            x, pcg_info = op.pcg(
                b=b4,
                x0=x,
                tol=tol,
                maxiter=int(fallback_pcg_maxiter),
                preconditioner="asm",
            )
            _sync_if_needed(op.device)
            solve_time += time.perf_counter() - pcg_tic
            pcg_iter = int(pcg_info.num_iter)
            residual_history.extend([float(v) for v in pcg_info.residual_history])
            converged = bool(pcg_info.converged)
            rel_res = float(pcg_info.rel_res)
    if rel0 < tol:
        solve_time = 0.0
    total_iter = int(tg_iter + pcg_iter)

    return x, {
        "num_iter": total_iter,
        "rel_res": float(rel_res),
        "converged": bool(converged),
        "residual_history": residual_history,
        "setup_time": float(setup_time),
        "solve_time": float(solve_time),
        "basis_dim": int(prepared_coarse_basis["basis_dim"]),
        "fallback_used": bool(fallback_used),
        "tg_iter": int(tg_iter),
        "pcg_iter": int(pcg_iter),
        "coarse_solver": str(prepared_coarse_basis["coarse_solver"]),
        "coarse_basis_mode": str(prepared_coarse_basis.get("coarse_basis_mode", coarse_basis_mode)),
        "smoother": str(smoother),
        "pre_smoother": str(pre_mode),
        "post_smoother": str(post_mode),
        "prepared_coarse_basis": prepared_coarse_basis,
    }


class TwoPhaseClosedDropletFVM:
    """
    Two-phase Darcy--Hele-Shaw closed-droplet solver without surface tension and
    without radial injection.

    Model:
        - div(lambda(phi) grad p) = 0,
        u = - lambda(phi) grad p,
        phi_t + u · grad phi = 0.

    Phase convention:
        phi > 0: background phase Omega_1, mobility lambda1
        phi < 0: closed droplet/inclusion Omega_2, mobility lambda2

    Pressure boundary conditions:
        x = 0:  p = p_in
        x = Lx: p = p_out
        y = 0,Ly: no-flow.

    Level-set boundary treatment:
        inflow boundary is assigned a positive background value; outflow and walls use Neumann copying.
    """

    def __init__(
        self,
        nx: int = 128,
        ny: int = 128,
        Lx: float = 1.0,
        Ly: float = 1.0,
        mu_background: float = 1.0,
        mu_droplet: float = 50.0,
        permeability: float = 1.0,
        gap_width: Optional[float] = None,
        p_in: float = 1.0,
        p_out: float = 0.0,
        heaviside_eps: Optional[float] = None,
        phi_background_value: Optional[float] = None,
        enforce_inflow_phi_bc: bool = True,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        kx_sub: int = 8,
        ky_sub: int = 8,
        target_overlap: float = 0.3,
        subdomain_solver: str = "cholesky",
        jacobi_pcg_iters: int = 10,
        cfl: float = 0.3,
        reinit_every: int = 0,
        reinit_steps: int = 5,
        transmissibility: str = "harmonic",
    ):
        self.nx = int(nx)
        self.ny = int(ny)
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.hx = self.Lx / self.nx
        self.hy = self.Ly / self.ny

        self.mu_background = float(mu_background)
        self.mu_droplet = float(mu_droplet)
        self.permeability = float(permeability)
        self.gap_width = gap_width

        if gap_width is None:
            self.lambda1 = self.permeability / self.mu_background
            self.lambda2 = self.permeability / self.mu_droplet
        else:
            self.lambda1 = gap_width ** 2 / (12.0 * self.mu_background)
            self.lambda2 = gap_width ** 2 / (12.0 * self.mu_droplet)

        self.p_in = float(p_in)
        self.p_out = float(p_out)
        self.device = torch.device(device)
        self.dtype = dtype

        self.heaviside_eps = float(heaviside_eps) if heaviside_eps is not None else 1.5 * max(self.hx, self.hy)
        self.phi_background_value = float(phi_background_value) if phi_background_value is not None else max(self.Lx, self.Ly)
        self.enforce_inflow_phi_bc = bool(enforce_inflow_phi_bc)

        self.kx_sub = int(kx_sub)
        self.ky_sub = int(ky_sub)
        self.target_overlap = float(target_overlap)
        self.subdomain_solver = str(subdomain_solver).lower()
        self.jacobi_pcg_iters = int(jacobi_pcg_iters)
        self.transmissibility = str(transmissibility).lower()
        self.cfl = float(cfl)
        self.reinit_every = int(reinit_every)
        self.reinit_steps = int(reinit_steps)
        self.step_id = 0

        self._build_grid()

    def _build_grid(self) -> None:
        x = (torch.arange(self.nx, device=self.device, dtype=self.dtype) + 0.5) * self.hx
        y = (torch.arange(self.ny, device=self.device, dtype=self.dtype) + 0.5) * self.hy
        Y, X = torch.meshgrid(y, x, indexing="ij")
        self.X = X[None, None, :, :]
        self.Y = Y[None, None, :, :]

    def initial_droplet_phi(
        self,
        shape: str = "circle",
        xc: Optional[float] = None,
        yc: Optional[float] = None,
        R0: float = 0.18,
        ellipse_a: float = 0.24,
        ellipse_b: float = 0.14,
        star_k: int = 5,
        alpha: float = 0.20,
        rotation: float = 0.0,
    ) -> torch.Tensor:
        """
        Return phi0 with phi < 0 inside the closed droplet and phi > 0 outside.

        shape choices: circle, ellipse, star.
        star means a smooth k-fold flower/star: R(theta)=R0*(1+alpha*cos(k theta)).
        """
        if xc is None:
            xc = 0.5 * self.Lx
        if yc is None:
            yc = 0.5 * self.Ly

        Xc = self.X - float(xc)
        Yc = self.Y - float(yc)
        r = torch.sqrt(Xc ** 2 + Yc ** 2 + 1e-30)
        theta = torch.atan2(Yc, Xc) - float(rotation)

        shape = shape.lower()
        if shape == "circle":
            phi = r - float(R0)
        elif shape == "ellipse":
            a = float(ellipse_a)
            b = float(ellipse_b)
            rho = torch.sqrt((Xc / a) ** 2 + (Yc / b) ** 2 + 1e-30)
            phi = min(a, b) * (rho - 1.0)
        elif shape == "star":
            if not (0.0 <= alpha < 1.0):
                raise ValueError("For star droplet, alpha must satisfy 0 <= alpha < 1.")
            R_theta = float(R0) * (1.0 + float(alpha) * torch.cos(float(star_k) * theta))
            phi = r - R_theta
        else:
            raise ValueError("shape must be 'circle', 'ellipse', or 'star'.")

        return phi.clone()

    # Backward-compatible alias.
    def initial_phi(self, mode: str = "circle", **kwargs) -> torch.Tensor:
        return self.initial_droplet_phi(shape=mode, **kwargs)

    def smooth_heaviside(self, phi: torch.Tensor) -> torch.Tensor:
        eps = self.heaviside_eps
        H = torch.zeros_like(phi)
        H = torch.where(phi > eps, torch.ones_like(phi), H)
        H = torch.where(phi < -eps, torch.zeros_like(phi), H)
        mask = torch.abs(phi) <= eps
        s = phi / eps
        H_mid = 0.5 * (1.0 + s + torch.sin(math.pi * s) / math.pi)
        H = torch.where(mask, H_mid, H)
        return H

    def mobility(self, phi: torch.Tensor) -> torch.Tensor:
        if phi.dim() == 2:
            phi = phi[None, None, :, :]
        elif phi.dim() == 3:
            phi = phi[:, None, :, :]
        phi = phi.to(device=self.device, dtype=self.dtype)
        H = self.smooth_heaviside(phi)
        # phi>0: background lambda1; phi<0: droplet lambda2.
        return self.lambda1 * H + self.lambda2 * (1.0 - H)

    def build_operator(
        self,
        phi: torch.Tensor,
        *,
        build_local_solver: bool = True,
    ) -> MobilityFVMOperatorASM:
        return MobilityFVMOperatorASM(
            lam=self.mobility(phi),
            hx=self.hx,
            hy=self.hy,
            p_left=self.p_in,
            p_right=self.p_out,
            kx_sub=self.kx_sub,
            ky_sub=self.ky_sub,
            target_overlap=self.target_overlap,
            build_local_solver=bool(build_local_solver),
            subdomain_solver=self.subdomain_solver,
            jacobi_pcg_iters=self.jacobi_pcg_iters,
            phi=phi,
            transmissibility=self.transmissibility,
        )

    @staticmethod
    def _preconditioner_needs_local_solver(preconditioner: str) -> bool:
        return str(preconditioner).lower() in {"asm", "asm_avg", "ras"}

    @staticmethod
    def _tg_config_needs_local_solver(
        tg_config: Optional[dict],
        *,
        fallback_to_pcg: bool = False,
    ) -> bool:
        tg_config = {} if tg_config is None else dict(tg_config)

        basis_smooth_mode = str(tg_config.get("basis_smooth_mode", "none")).lower()
        basis_smooth_steps = int(tg_config.get("basis_smooth_steps", 0))
        if basis_smooth_mode == "asm" and basis_smooth_steps > 0:
            return True

        smoother = tg_config.get("smoother", tg_config.get("smooth_mode", "asm"))
        pre_smoother = tg_config.get(
            "pre_smoother",
            tg_config.get("pre_smooth_mode", None),
        )
        post_smoother = tg_config.get(
            "post_smoother",
            tg_config.get("post_smooth_mode", None),
        )
        pre_mode = _two_phase_smoother_mode(
            pre_smoother if pre_smoother is not None else smoother
        )
        post_mode = _two_phase_smoother_mode(
            post_smoother if post_smoother is not None else smoother
        )
        pre_steps = int(tg_config.get("pre_smooth_steps", 1))
        post_steps = int(tg_config.get("post_smooth_steps", 0))
        if (pre_steps > 0 and pre_mode == "asm") or (post_steps > 0 and post_mode == "asm"):
            return True

        return bool(fallback_to_pcg)

    def _pressure_solve_needs_local_solver(
        self,
        *,
        method: str,
        preconditioner: str,
        tg_config: Optional[dict],
    ) -> bool:
        method = str(method).lower()
        preconditioner = str(preconditioner).lower()
        if method == "pcg":
            return self._preconditioner_needs_local_solver(preconditioner)
        if method == "fcg":
            if preconditioner in {"tg", "learned_tg"}:
                return self._tg_config_needs_local_solver(tg_config)
            return self._preconditioner_needs_local_solver(preconditioner)
        if method == "fcg_tg":
            return self._tg_config_needs_local_solver(tg_config)
        if method == "learned_tg":
            fallback_to_pcg = bool((tg_config or {}).get("fallback_to_pcg", False))
            return self._tg_config_needs_local_solver(
                tg_config,
                fallback_to_pcg=fallback_to_pcg,
            )
        if method == "learned_tg_then_pcg":
            return self._tg_config_needs_local_solver(tg_config, fallback_to_pcg=True)
        return False

    def _prepare_predict_coord_tensor(
        self,
        coord: torch.Tensor,
        batch_size: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        coord = torch.as_tensor(coord, device=device, dtype=torch.float32)
        if coord.dim() == 2:
            if coord.shape == (self.ny, self.nx):
                coord = coord.reshape(1, -1, 1)
            elif coord.shape == (self.nx * self.ny, 1):
                coord = coord.unsqueeze(0)
            else:
                raise ValueError(
                    f"{name} coordinate has unsupported shape {tuple(coord.shape)}"
                )
        elif coord.dim() == 3:
            if coord.shape[-2:] == (self.ny, self.nx):
                coord = coord.reshape(coord.shape[0], -1, 1)
            elif coord.shape[1] == self.nx * self.ny and coord.shape[2] == 1:
                pass
            else:
                raise ValueError(
                    f"{name} coordinate has unsupported shape {tuple(coord.shape)}"
                )
        elif coord.dim() == 4:
            if coord.shape[1] != 1 or coord.shape[-2:] != (self.ny, self.nx):
                raise ValueError(
                    f"{name} coordinate has unsupported shape {tuple(coord.shape)}"
                )
            coord = coord.reshape(coord.shape[0], -1, 1)
        else:
            raise ValueError(
                f"{name} coordinate must have 2, 3, or 4 dims, got {tuple(coord.shape)}"
            )

        coord = _expand_batch_dim(coord, batch_size, f"{name} coordinate")
        return coord.contiguous()

    def _resolve_predict_coords(
        self,
        coords,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if coords is None:
            return (
                self._prepare_predict_coord_tensor(self.X, batch_size, device, "x"),
                self._prepare_predict_coord_tensor(self.Y, batch_size, device, "y"),
            )

        if isinstance(coords, dict):
            if "x_coords" in coords and "y_coords" in coords:
                x_coord = coords["x_coords"]
                y_coord = coords["y_coords"]
            elif "X" in coords and "Y" in coords:
                x_coord = coords["X"]
                y_coord = coords["Y"]
            elif "coords" in coords:
                coords = coords["coords"]
                x_coord = None
                y_coord = None
            else:
                raise ValueError(
                    "coords dict must contain {'x_coords','y_coords'}, {'X','Y'}, or {'coords'}"
                )
            if x_coord is not None and y_coord is not None:
                return (
                    self._prepare_predict_coord_tensor(x_coord, batch_size, device, "x"),
                    self._prepare_predict_coord_tensor(y_coord, batch_size, device, "y"),
                )

        if isinstance(coords, (tuple, list)) and len(coords) == 2:
            return (
                self._prepare_predict_coord_tensor(coords[0], batch_size, device, "x"),
                self._prepare_predict_coord_tensor(coords[1], batch_size, device, "y"),
            )

        coords_tensor = torch.as_tensor(coords, device=device, dtype=torch.float32)
        if coords_tensor.dim() == 2 and coords_tensor.shape[-1] == 2:
            x_coord = coords_tensor[:, 0:1].unsqueeze(0)
            y_coord = coords_tensor[:, 1:2].unsqueeze(0)
        elif coords_tensor.dim() == 3 and coords_tensor.shape[-1] == 2:
            x_coord = coords_tensor[..., 0:1]
            y_coord = coords_tensor[..., 1:2]
        else:
            raise ValueError(
                "coords must be one of: (N,2), (B,N,2), {'x_coords','y_coords'}, or {'X','Y'}"
            )

        return (
            self._prepare_predict_coord_tensor(x_coord, batch_size, device, "x"),
            self._prepare_predict_coord_tensor(y_coord, batch_size, device, "y"),
        )

    def _build_nn_input_coefficient(
        self,
        lam: torch.Tensor,
        nn_coeff_mode: str = "mobility",
    ) -> torch.Tensor:
        nn_coeff_mode = str(nn_coeff_mode).lower()
        lam = lam.to(device=lam.device, dtype=torch.float64)
        lam_min = lam.amin(dim=(1, 2, 3), keepdim=True).clamp_min(1e-30)
        inv_lam = 1.0 / lam.clamp_min(1e-30)
        inv_lam_min = inv_lam.amin(dim=(1, 2, 3), keepdim=True).clamp_min(1e-30)

        if nn_coeff_mode == "mobility":
            return lam
        if nn_coeff_mode == "inverse_mobility":
            return inv_lam
        if nn_coeff_mode == "normalized_mobility":
            return lam / lam_min
        if nn_coeff_mode == "normalized_inverse_mobility":
            return inv_lam / inv_lam_min
        raise ValueError(
            "nn_coeff_mode must be one of "
            "{'mobility', 'inverse_mobility', 'normalized_mobility', 'normalized_inverse_mobility'}"
        )

    @torch.no_grad()
    def solve_pressure(
        self,
        phi: torch.Tensor,
        method: str = "pcg",
        learned_model=None,
        coords=None,
        basis_dim: Optional[int] = None,
        nn_coeff_mode: str = "mobility",
        tg_config: Optional[dict] = None,
        p0: Optional[torch.Tensor] = None,
        tol: float = 1e-8,
        maxiter: int = 500,
        preconditioner: str = "asm",
        gmg_config: Optional[dict] = None,
        prepared_coarse_basis: Optional[Dict[str, object]] = None,
    ) -> Tuple[torch.Tensor, MobilityFVMOperatorASM, PressureSolveInfo]:
        method = str(method).lower()
        valid_methods = {"pcg", "learned_tg", "learned_tg_then_pcg", "fcg", "fcg_tg", "fcg_gmg"}
        if method not in valid_methods:
            raise ValueError(
                "method must be one of "
                "{'pcg', 'learned_tg', 'learned_tg_then_pcg', 'fcg', 'fcg_tg', 'fcg_gmg'}"
            )

        tg_config = {} if tg_config is None else dict(tg_config)
        gmg_config = {} if gmg_config is None else dict(gmg_config)
        preconditioner_kind = str(preconditioner).lower()
        op, operator_setup_time = _timed(
            self.device,
            lambda: self.build_operator(
                phi,
                build_local_solver=self._pressure_solve_needs_local_solver(
                    method=method,
                    preconditioner=preconditioner_kind,
                    tg_config=tg_config,
                ),
            ),
        )
        b = op.rhs(q=None)  # closed-droplet model: no radial injection, no volume source.

        if method == "pcg":
            preconditioner_setup_time = 0.0
            gmg = None
            coarse_solver = "none"
            if preconditioner_kind == "gmg":
                gmg, preconditioner_setup_time = _timed(
                    op.device,
                    lambda: StandardGMGHierarchy(
                        op,
                        num_levels=gmg_config.get("num_levels"),
                        min_coarse_size=int(gmg_config.get("min_coarse_size", 4)),
                        smoother=str(gmg_config.get("smoother", "jacobi")),
                        pre_smooth_steps=int(gmg_config.get("pre_smooth_steps", 2)),
                        post_smooth_steps=int(gmg_config.get("post_smooth_steps", 2)),
                        omega=float(gmg_config.get("omega", 0.8)),
                        coarse_solve=str(gmg_config.get("coarse_solve", "direct")),
                    ),
                )
                coarse_solver = "direct" if gmg.coarse_direct_matrix is not None else "jacobi"
            (p, cg_info), solve_time = _timed(
                op.device,
                lambda: op.pcg(
                    b=b,
                    x0=p0,
                    tol=tol,
                    maxiter=maxiter,
                    preconditioner=preconditioner_kind,
                    gmg_hierarchy=gmg,
                    gmg_config=gmg_config,
                ),
            )
            info = PressureSolveInfo.from_pcg(
                cg_info,
                solve_time=solve_time,
                operator_setup_time=operator_setup_time,
                preconditioner_setup_time=preconditioner_setup_time,
                preconditioner=preconditioner_kind,
                nn_coeff_mode=nn_coeff_mode,
                coarse_solver=coarse_solver,
            )
            return p, op, info

        if method == "fcg_gmg" or (method == "fcg" and preconditioner_kind == "gmg"):
            gmg, preconditioner_setup_time = _timed(
                op.device,
                lambda: StandardGMGHierarchy(
                    op,
                    num_levels=gmg_config.get("num_levels"),
                    min_coarse_size=int(gmg_config.get("min_coarse_size", 4)),
                    smoother=str(gmg_config.get("smoother", "jacobi")),
                    pre_smooth_steps=int(gmg_config.get("pre_smooth_steps", 2)),
                    post_smooth_steps=int(gmg_config.get("post_smooth_steps", 2)),
                    omega=float(gmg_config.get("omega", 0.8)),
                    coarse_solve=str(gmg_config.get("coarse_solve", "direct")),
                ),
            )
            (p, fcg_info), solve_time = _timed(
                op.device,
                lambda: op.fcg(
                    b=b,
                    x0=p0,
                    tol=float(gmg_config.get("tol", tol)),
                    maxiter=int(gmg_config.get("maxiter", gmg_config.get("max_iter", maxiter))),
                    preconditioner="gmg",
                    m_max=gmg_config.get("m_max"),
                    gmg_hierarchy=gmg,
                    gmg_config=gmg_config,
                ),
            )
            coarse_solver = "direct" if gmg.coarse_direct_matrix is not None else "jacobi"
            info = PressureSolveInfo(
                pressure_method="fcg_gmg",
                num_iter=int(fcg_info.num_iter),
                rel_res=float(fcg_info.rel_res),
                converged=bool(fcg_info.converged),
                residual_history=[float(v) for v in fcg_info.residual_history],
                operator_setup_time=float(operator_setup_time),
                preconditioner_setup_time=float(preconditioner_setup_time),
                solve_time=float(solve_time),
                basis_dim=0,
                fallback_used=False,
                tg_iter=0,
                pcg_iter=int(fcg_info.num_iter),
                preconditioner="gmg",
                inference_time=0.0,
                nn_coeff_mode=str(nn_coeff_mode),
                coarse_solver=coarse_solver,
            )
            return p, op, info

        if method == "fcg" and preconditioner_kind not in {"tg", "learned_tg", "gmg"}:
            (p, fcg_info), solve_time = _timed(
                op.device,
                lambda: op.fcg(
                    b=b,
                    x0=p0,
                    tol=float(tg_config.get("tol", tol)),
                    maxiter=int(tg_config.get("maxiter", tg_config.get("max_iter", maxiter))),
                    preconditioner=preconditioner_kind,
                    m_max=tg_config.get("m_max"),
                    tg_config=tg_config,
                ),
            )
            info = PressureSolveInfo(
                pressure_method="fcg",
                num_iter=int(fcg_info.num_iter),
                rel_res=float(fcg_info.rel_res),
                converged=bool(fcg_info.converged),
                residual_history=[float(v) for v in fcg_info.residual_history],
                operator_setup_time=float(operator_setup_time),
                preconditioner_setup_time=0.0,
                solve_time=float(solve_time),
                basis_dim=0,
                fallback_used=False,
                tg_iter=0,
                pcg_iter=int(fcg_info.num_iter),
                preconditioner=preconditioner_kind,
                inference_time=0.0,
                nn_coeff_mode=str(nn_coeff_mode),
                coarse_solver="none",
            )
            return p, op, info

        if learned_model is None and prepared_coarse_basis is None:
            raise ValueError(
                "learned_model must be provided for learned_tg, learned_tg_then_pcg, "
                "fcg_tg, or fcg with preconditioner='tg'."
            )

        phi_field = self._ensure_phi_4d(phi)
        batch_size = int(phi_field.shape[0])
        reused_prepared_basis = prepared_coarse_basis is not None
        V = None
        inference_time = 0.0
        if prepared_coarse_basis is None:
            model_core = learned_model.module if isinstance(learned_model, nn.DataParallel) else learned_model
            try:
                model_device = next(model_core.parameters()).device
            except StopIteration as exc:
                raise RuntimeError("learned_model has no parameters; cannot infer device") from exc

            active_basis_dim = int(basis_dim if basis_dim is not None else getattr(model_core, "m_dim"))
            x_coords, y_coords = self._resolve_predict_coords(coords, batch_size, model_device)
            a_field_for_nn = self._build_nn_input_coefficient(op.lam, nn_coeff_mode).to(
                device=model_device,
                dtype=torch.float32,
            )
            phi_field_for_nn = phi_field.to(device=model_device, dtype=torch.float32)

            V, inference_time = _timed(
                model_device,
                lambda: model_core.predict_basis(
                    a_field_for_nn,
                    phi_field_for_nn,
                    x_coords,
                    y_coords,
                    active_dim=active_basis_dim,
                ),
            )
            V = V.to(device=op.device, dtype=torch.float64)

        if method == "fcg_tg" or (method == "fcg" and preconditioner_kind in {"tg", "learned_tg"}):
            tg_smoother = tg_config.get("smoother", tg_config.get("smooth_mode", "asm"))
            tg_pre_smoother = tg_config.get(
                "pre_smoother",
                tg_config.get("pre_smooth_mode", None),
            )
            tg_post_smoother = tg_config.get(
                "post_smoother",
                tg_config.get("post_smooth_mode", None),
            )
            tg_pre_mode = _two_phase_smoother_mode(
                tg_pre_smoother if tg_pre_smoother is not None else tg_smoother
            )
            tg_post_mode = _two_phase_smoother_mode(
                tg_post_smoother if tg_post_smoother is not None else tg_smoother
            )

            def _prepare_tg_preconditioner():
                if prepared_coarse_basis is None:
                    if V is None:
                        raise ValueError("V must be available when TG coarse basis is rebuilt")
                    prepared = prepare_two_phase_coarse_basis(
                        op,
                        V,
                        basis_smooth_mode=str(tg_config.get("basis_smooth_mode", "none")),
                        basis_smooth_steps=int(tg_config.get("basis_smooth_steps", 0)),
                        basis_smooth_omega=float(tg_config.get("basis_smooth_omega", 0.6)),
                        orth_reg=float(tg_config.get("orth_reg", 1e-10)),
                        coarse_basis_mode=str(tg_config.get("coarse_basis_mode", "a_orthonormal")),
                    )
                else:
                    prepared = prepared_coarse_basis
                hierarchy = None
                if tg_pre_mode == "gmg" or tg_post_mode == "gmg":
                    hierarchy = _gmg_hierarchy_from_config(
                        op,
                        tg_config.get("gmg_config", gmg_config),
                    )
                return prepared, hierarchy

            (prepared_coarse_basis, tg_gmg_hierarchy), preconditioner_setup_time = _timed(
                op.device,
                _prepare_tg_preconditioner,
            )
            (p, fcg_info), solve_time = _timed(
                op.device,
                lambda: op.fcg(
                    b=b,
                    x0=p0,
                    tol=float(tg_config.get("tol", tol)),
                    maxiter=int(tg_config.get("maxiter", tg_config.get("max_iter", maxiter))),
                    preconditioner="tg",
                    m_max=tg_config.get("m_max"),
                    prepared_coarse_basis=prepared_coarse_basis,
                    tg_config=tg_config,
                    gmg_hierarchy=tg_gmg_hierarchy,
                ),
            )
            info = PressureSolveInfo(
                pressure_method="fcg_tg",
                num_iter=int(fcg_info.num_iter),
                rel_res=float(fcg_info.rel_res),
                converged=bool(fcg_info.converged),
                residual_history=[float(v) for v in fcg_info.residual_history],
                operator_setup_time=float(operator_setup_time),
                preconditioner_setup_time=float(preconditioner_setup_time),
                solve_time=float(solve_time),
                basis_dim=int(prepared_coarse_basis["basis_dim"]),
                fallback_used=False,
                tg_iter=0,
                pcg_iter=int(fcg_info.num_iter),
                preconditioner="tg",
                inference_time=float(inference_time),
                nn_coeff_mode=str(nn_coeff_mode),
                coarse_solver=str(prepared_coarse_basis["coarse_solver"]),
            )
            info.prepared_coarse_basis = prepared_coarse_basis
            info.basis_reused = bool(reused_prepared_basis)
            return p, op, info

        fallback_to_pcg = bool(tg_config.get("fallback_to_pcg", method == "learned_tg_then_pcg"))
        if method == "learned_tg_then_pcg":
            fallback_to_pcg = True

        p, learned_info = solve_pressure_learned_two_grid(
            op,
            b,
            V,
            x0=p0,
            tol=float(tg_config.get("tol", tol)),
            max_iter=int(tg_config.get("max_iter", maxiter)),
            pre_smooth_steps=int(tg_config.get("pre_smooth_steps", 1)),
            pre_smooth_omega=float(tg_config.get("pre_smooth_omega", 0.8)),
            post_smooth_steps=int(tg_config.get("post_smooth_steps", 0)),
            post_smooth_omega=float(tg_config.get("post_smooth_omega", 0.8)),
            smoother=str(tg_config.get("smoother", tg_config.get("smooth_mode", "asm"))),
            pre_smoother=tg_config.get("pre_smoother", tg_config.get("pre_smooth_mode", None)),
            post_smoother=tg_config.get("post_smoother", tg_config.get("post_smooth_mode", None)),
            basis_smooth_mode=str(tg_config.get("basis_smooth_mode", "none")),
            basis_smooth_steps=int(tg_config.get("basis_smooth_steps", 0)),
            basis_smooth_omega=float(tg_config.get("basis_smooth_omega", 0.6)),
            orth_reg=float(tg_config.get("orth_reg", 1e-10)),
            coarse_basis_mode=str(tg_config.get("coarse_basis_mode", "a_orthonormal")),
            fallback_to_pcg=fallback_to_pcg,
            fallback_pcg_maxiter=int(tg_config.get("fallback_pcg_maxiter", maxiter)),
            gmg_config=tg_config.get("gmg_config", gmg_config),
            prepared_coarse_basis=prepared_coarse_basis,
        )
        preconditioner_setup_time = float(learned_info["setup_time"])
        solve_time = float(learned_info["solve_time"])

        info = PressureSolveInfo(
            pressure_method=method,
            num_iter=int(learned_info["num_iter"]),
            rel_res=float(learned_info["rel_res"]),
            converged=bool(learned_info["converged"]),
            residual_history=[float(v) for v in learned_info["residual_history"]],
            operator_setup_time=float(operator_setup_time),
            preconditioner_setup_time=preconditioner_setup_time,
            solve_time=solve_time,
            basis_dim=int(learned_info["basis_dim"]),
            fallback_used=bool(learned_info["fallback_used"]),
            tg_iter=int(learned_info["tg_iter"]),
            pcg_iter=int(learned_info["pcg_iter"]),
            preconditioner=str(learned_info.get("pre_smoother", "asm")),
            inference_time=float(inference_time),
            nn_coeff_mode=str(nn_coeff_mode),
            coarse_solver=str(learned_info["coarse_solver"]),
        )
        info.prepared_coarse_basis = learned_info.get("prepared_coarse_basis")
        info.basis_reused = bool(reused_prepared_basis)
        return p, op, info

    def compute_velocity(self, op: MobilityFVMOperatorASM, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p4, _ = op._to_4d(p)
        B = p4.shape[0]
        u_face = torch.zeros(B, 1, self.ny, self.nx + 1, device=self.device, dtype=self.dtype)
        v_face = torch.zeros(B, 1, self.ny + 1, self.nx, device=self.device, dtype=self.dtype)

        lam = op.lam
        u_face[..., :, 1:self.nx] = -op.lam_x * (p4[..., :, 1:] - p4[..., :, :-1]) / self.hx
        u_face[..., :, 0] = -lam[..., :, 0] * (p4[..., :, 0] - self.p_in) / (0.5 * self.hx)
        u_face[..., :, self.nx] = -lam[..., :, -1] * (self.p_out - p4[..., :, -1]) / (0.5 * self.hx)
        v_face[..., 1:self.ny, :] = -op.lam_y * (p4[..., 1:, :] - p4[..., :-1, :]) / self.hy

        u_center = 0.5 * (u_face[..., :, :-1] + u_face[..., :, 1:])
        v_center = 0.5 * (v_face[..., :-1, :] + v_face[..., 1:, :])
        return u_center, v_center, u_face, v_face

    def compute_dt(self, u: torch.Tensor, v: torch.Tensor, dt_max: Optional[float] = None) -> float:
        umax = torch.max(torch.abs(u)).item()
        vmax = torch.max(torch.abs(v)).item()
        eps = 1e-14
        dt = self.cfl * min(self.hx / max(umax, eps), self.hy / max(vmax, eps))
        if dt_max is not None:
            dt = min(dt, float(dt_max))
        return dt

    def _ensure_phi_4d(self, phi: torch.Tensor) -> torch.Tensor:
        if phi.dim() == 2:
            phi = phi[None, None, :, :]
        elif phi.dim() == 3:
            phi = phi[:, None, :, :]
        elif phi.dim() != 4:
            raise ValueError(f"phi must have dim 2,3,4; got shape={tuple(phi.shape)}")
        return phi.to(device=self.device, dtype=self.dtype)

    def pad_phi_for_advection(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Ghost-cell padding for level-set advection.

        The physical closed-droplet benchmark has a background phase entering through the inflow boundary.
        For p_in > p_out, inflow is on the left and phi_ghost is set positive there.
        For p_in < p_out, inflow is on the right and phi_ghost is set positive there.
        Walls and outflow use Neumann copying.
        """
        phi = self._ensure_phi_4d(phi)
        phi_pad = F.pad(phi, (1, 1, 1, 1), mode="replicate")
        if self.enforce_inflow_phi_bc:
            bg = torch.tensor(self.phi_background_value, device=self.device, dtype=self.dtype)
            if self.p_in > self.p_out:
                phi_pad[..., 1:-1, 0] = bg
            elif self.p_in < self.p_out:
                phi_pad[..., 1:-1, -1] = bg
        return phi_pad

    def levelset_rhs_upwind(self, phi: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        phi = self._ensure_phi_4d(phi)
        phi_pad = self.pad_phi_for_advection(phi)
        c = phi_pad[..., 1:-1, 1:-1]

        dxb = (c - phi_pad[..., 1:-1, :-2]) / self.hx
        dxf = (phi_pad[..., 1:-1, 2:] - c) / self.hx
        dyb = (c - phi_pad[..., :-2, 1:-1]) / self.hy
        dyf = (phi_pad[..., 2:, 1:-1] - c) / self.hy

        ux_pos = torch.clamp(u, min=0.0)
        ux_neg = torch.clamp(u, max=0.0)
        vy_pos = torch.clamp(v, min=0.0)
        vy_neg = torch.clamp(v, max=0.0)
        return ux_pos * dxb + ux_neg * dxf + vy_pos * dyb + vy_neg * dyf

    def advect_phi_rk2(self, phi: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor:
        phi = self._ensure_phi_4d(phi)
        L0 = self.levelset_rhs_upwind(phi, u, v)
        phi1 = phi - float(dt) * L0
        L1 = self.levelset_rhs_upwind(phi1, u, v)
        return 0.5 * phi + 0.5 * (phi1 - float(dt) * L1)

    def reinitialize_sussman(self, phi: torch.Tensor, steps: int = 5, dtau: Optional[float] = None) -> torch.Tensor:
        phi = self._ensure_phi_4d(phi)
        if dtau is None:
            dtau = 0.3 * min(self.hx, self.hy)

        phi0 = phi.clone()
        psi = phi.clone()
        eps = min(self.hx, self.hy)
        sgn = phi0 / torch.sqrt(phi0 ** 2 + eps ** 2)

        for _ in range(steps):
            psi_pad = F.pad(psi, (1, 1, 1, 1), mode="replicate")
            c = psi_pad[..., 1:-1, 1:-1]
            a = (c - psi_pad[..., 1:-1, :-2]) / self.hx
            b = (psi_pad[..., 1:-1, 2:] - c) / self.hx
            cc = (c - psi_pad[..., :-2, 1:-1]) / self.hy
            d = (psi_pad[..., 2:, 1:-1] - c) / self.hy

            grad_pos = torch.sqrt(
                torch.clamp(a, min=0.0) ** 2
                + torch.clamp(b, max=0.0) ** 2
                + torch.clamp(cc, min=0.0) ** 2
                + torch.clamp(d, max=0.0) ** 2
                + 1e-30
            )
            grad_neg = torch.sqrt(
                torch.clamp(a, max=0.0) ** 2
                + torch.clamp(b, min=0.0) ** 2
                + torch.clamp(cc, max=0.0) ** 2
                + torch.clamp(d, min=0.0) ** 2
                + 1e-30
            )
            grad = torch.where(sgn >= 0.0, grad_pos, grad_neg)
            psi = psi - float(dtau) * sgn * (grad - 1.0)
        return psi

    def droplet_area(self, phi: torch.Tensor) -> torch.Tensor:
        phi = self._ensure_phi_4d(phi)
        H = self.smooth_heaviside(phi)
        chi_droplet = 1.0 - H
        return torch.sum(chi_droplet, dim=(1, 2, 3)) * self.hx * self.hy

    def droplet_centroid(self, phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        phi = self._ensure_phi_4d(phi)
        H = self.smooth_heaviside(phi)
        chi = 1.0 - H
        mass = torch.sum(chi, dim=(1, 2, 3)).clamp_min(1e-30)
        cx = torch.sum(chi * self.X, dim=(1, 2, 3)) / mass
        cy = torch.sum(chi * self.Y, dim=(1, 2, 3)) / mass
        return cx, cy

    @torch.no_grad()
    def step(
        self,
        phi: torch.Tensor,
        p0: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
        dt_max: Optional[float] = None,
        tol: float = 1e-8,
        maxiter: int = 500,
        preconditioner: str = "asm",
        pressure_method: str = "pcg",
        learned_model=None,
        coords=None,
        basis_dim: Optional[int] = None,
        nn_coeff_mode: str = "mobility",
        tg_config: Optional[dict] = None,
        gmg_config: Optional[dict] = None,
    ) -> Dict[str, object]:
        phi = self._ensure_phi_4d(phi)
        p, op, info = self.solve_pressure(
            phi=phi,
            method=pressure_method,
            learned_model=learned_model,
            coords=coords,
            basis_dim=basis_dim,
            nn_coeff_mode=nn_coeff_mode,
            tg_config=tg_config,
            p0=p0,
            tol=tol,
            maxiter=maxiter,
            preconditioner=preconditioner,
            gmg_config=gmg_config,
        )
        u, v, u_face, v_face = self.compute_velocity(op, p)

        if dt is None:
            dt = self.compute_dt(u, v, dt_max=dt_max)
        else:
            dt = float(dt)

        phi_new = self.advect_phi_rk2(phi, u, v, dt)
        self.step_id += 1
        if self.reinit_every > 0 and self.step_id % self.reinit_every == 0:
            phi_new = self.reinitialize_sussman(phi_new, steps=self.reinit_steps)

        area = self.droplet_area(phi_new)
        cx, cy = self.droplet_centroid(phi_new)

        return {
            "phi": phi_new,
            "p": p,
            "u": u,
            "v": v,
            "u_face": u_face,
            "v_face": v_face,
            "lambda": op.lam,
            "dt": torch.tensor(dt, device=self.device, dtype=self.dtype),
            "pressure_method": info.pressure_method,
            "pressure_iter": torch.tensor(info.num_iter, device=self.device),
            "pressure_rel_res": torch.tensor(info.rel_res, device=self.device, dtype=self.dtype),
            "pressure_converged": torch.tensor(info.converged, device=self.device),
            "tg_iter": torch.tensor(info.tg_iter, device=self.device),
            "pcg_iter": torch.tensor(info.pcg_iter, device=self.device),
            "fallback_used": torch.tensor(info.fallback_used, device=self.device),
            "basis_dim": torch.tensor(info.basis_dim, device=self.device),
            "operator_setup_time": torch.tensor(
                info.operator_setup_time,
                device=self.device,
                dtype=self.dtype,
            ),
            "preconditioner_setup_time": torch.tensor(
                info.preconditioner_setup_time,
                device=self.device,
                dtype=self.dtype,
            ),
            "setup_time": torch.tensor(info.setup_time, device=self.device, dtype=self.dtype),
            "solve_time": torch.tensor(info.solve_time, device=self.device, dtype=self.dtype),
            "inference_time": torch.tensor(info.inference_time, device=self.device, dtype=self.dtype),
            "oracle_snapshot_setup_time": torch.tensor(
                info.oracle_snapshot_setup_time,
                device=self.device,
                dtype=self.dtype,
            ),
            "total_with_inference_time": torch.tensor(
                info.total_with_inference_time,
                device=self.device,
                dtype=self.dtype,
            ),
            "total_time": torch.tensor(info.total_with_inference_time, device=self.device, dtype=self.dtype),
            "cg_iter": torch.tensor(info.num_iter, device=self.device),
            "cg_rel_res": torch.tensor(info.rel_res, device=self.device, dtype=self.dtype),
            "cg_converged": torch.tensor(info.converged, device=self.device),
            "area": area,
            "centroid_x": cx,
            "centroid_y": cy,
        }

    @torch.no_grad()
    def run(
        self,
        nsteps: int,
        phi0: Optional[torch.Tensor] = None,
        droplet_shape: str = "circle",
        droplet_kwargs: Optional[dict] = None,
        dt_fixed: Optional[float] = None,
        dt_max: Optional[float] = None,
        tol: float = 1e-8,
        maxiter: int = 500,
        preconditioner: str = "asm",
        pressure_method: str = "pcg",
        learned_model=None,
        coords=None,
        basis_dim: Optional[int] = None,
        nn_coeff_mode: str = "mobility",
        tg_config: Optional[dict] = None,
        store_every: int = 1,
        gmg_config: Optional[dict] = None,
    ) -> Dict[str, list]:
        if droplet_kwargs is None:
            droplet_kwargs = {}

        if phi0 is None:
            phi = self.initial_droplet_phi(shape=droplet_shape, **droplet_kwargs)
        else:
            phi = self._ensure_phi_4d(phi0)

        snapshots: Dict[str, list] = {
            "phi": [],
            "p": [],
            "lambda": [],
            "u": [],
            "v": [],
            "pressure_method": [],
            "pressure_iter": [],
            "pressure_rel_res": [],
            "pressure_converged": [],
            "tg_iter": [],
            "pcg_iter": [],
            "fallback_used": [],
            "basis_dim": [],
            "operator_setup_time": [],
            "preconditioner_setup_time": [],
            "setup_time": [],
            "solve_time": [],
            "inference_time": [],
            "oracle_snapshot_setup_time": [],
            "total_with_inference_time": [],
            "total_time": [],
            "cg_iter": [],
            "cg_rel_res": [],
            "cg_converged": [],
            "dt": [],
            "t": [],
            "area": [],
            "centroid_x": [],
            "centroid_y": [],
        }

        p0 = None
        t = 0.0
        self.step_id = 0

        for step_id in range(int(nsteps)):
            out = self.step(
                phi=phi,
                p0=p0,
                dt=dt_fixed,
                dt_max=dt_max,
                tol=tol,
                maxiter=maxiter,
                preconditioner=preconditioner,
                pressure_method=pressure_method,
                learned_model=learned_model,
                coords=coords,
                basis_dim=basis_dim,
                nn_coeff_mode=nn_coeff_mode,
                tg_config=tg_config,
                gmg_config=gmg_config,
            )
            phi = out["phi"]
            p0 = out["p"]
            t += float(out["dt"].item())

            if step_id % store_every == 0:
                snapshots["phi"].append(out["phi"].detach().cpu())
                snapshots["p"].append(out["p"].detach().cpu())
                snapshots["lambda"].append(out["lambda"].detach().cpu())
                snapshots["u"].append(out["u"].detach().cpu())
                snapshots["v"].append(out["v"].detach().cpu())
                snapshots["pressure_method"].append(out["pressure_method"])
                snapshots["pressure_iter"].append(int(out["pressure_iter"].item()))
                snapshots["pressure_rel_res"].append(float(out["pressure_rel_res"].item()))
                snapshots["pressure_converged"].append(bool(out["pressure_converged"].item()))
                snapshots["tg_iter"].append(int(out["tg_iter"].item()))
                snapshots["pcg_iter"].append(int(out["pcg_iter"].item()))
                snapshots["fallback_used"].append(bool(out["fallback_used"].item()))
                snapshots["basis_dim"].append(int(out["basis_dim"].item()))
                snapshots["operator_setup_time"].append(float(out["operator_setup_time"].item()))
                snapshots["preconditioner_setup_time"].append(float(out["preconditioner_setup_time"].item()))
                snapshots["setup_time"].append(float(out["setup_time"].item()))
                snapshots["solve_time"].append(float(out["solve_time"].item()))
                snapshots["inference_time"].append(float(out["inference_time"].item()))
                snapshots["oracle_snapshot_setup_time"].append(float(out["oracle_snapshot_setup_time"].item()))
                snapshots["total_with_inference_time"].append(
                    float(out["total_with_inference_time"].item())
                )
                snapshots["total_time"].append(float(out["total_time"].item()))
                snapshots["cg_iter"].append(int(out["cg_iter"].item()))
                snapshots["cg_rel_res"].append(float(out["cg_rel_res"].item()))
                snapshots["cg_converged"].append(bool(out["cg_converged"].item()))
                snapshots["dt"].append(float(out["dt"].item()))
                snapshots["t"].append(t)
                snapshots["area"].append(float(out["area"][0].item()))
                snapshots["centroid_x"].append(float(out["centroid_x"][0].item()))
                snapshots["centroid_y"].append(float(out["centroid_y"][0].item()))

        return snapshots
