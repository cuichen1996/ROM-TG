"""Generate one trajectory and compare coarse spaces on its fixed operators.

The FGMRES adapter is a portable entry for the pending full moving-interface
remeasurement. Historical FGCR table values are not emitted by this program.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from learned_tg.model.coarse_spaces.repeated_solve import build_threshold_geneo_raw_basis
from learned_tg.model.krylov import fgmres
from learned_tg.model.two_phase import (
    TwoPhaseClosedDropletFVM, PressureSolveInfo, prepare_two_phase_coarse_basis,
    two_phase_two_grid_sweep,
)
from learned_tg.runtime import initialize, load_model, new_output, read_config
from learned_tg.solvers.static import dump, thash

BOUNDARIES = dict(west="dirichlet", east="dirichlet", north="neumann", south="neumann")


def sparse_matrix(op):
    """Assemble the homogeneous pressure operator, including boundary diagonal."""
    n, m = op.ny, op.nx
    ids = np.arange(n * m).reshape(n, m)
    rows, cols, values = [ids.ravel()], [ids.ravel()], [op.diag.cpu().numpy().ravel()]
    for name, here, there in [("aw", ids[:, 1:], ids[:, :-1]), ("ae", ids[:, :-1], ids[:, 1:]),
                              ("as_", ids[1:, :], ids[:-1, :]), ("an", ids[:-1, :], ids[1:, :])]:
        value = getattr(op, name).cpu().numpy().reshape(n, m)
        block = value[:, 1:] if name == "aw" else value[:, :-1] if name == "ae" else value[1:, :] if name == "as_" else value[:-1, :]
        rows.append(here.ravel()); cols.append(there.ravel()); values.append(-block.ravel())
    return sp.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n * m, n * m)).tocsr()


class FGMRESFlow(TwoPhaseClosedDropletFVM):
    """Same FVM/transport model with a fixed TG map per FGMRES pressure solve."""

    def __init__(self, config, device="cpu", n=None, subdomains=None):
        n = n or config.get("evaluation_grid", 1023)
        k = subdomains or config.get("k_1d", 10)
        super().__init__(nx=n, ny=n, device=str(device), kx_sub=k, ky_sub=k,
                         mu_background=config.get("mu_background", 1.0), mu_droplet=config.get("mu_droplet", 1e-4),
                         p_in=1.0, p_out=0.0, permeability=1.0, cfl=config.get("cfl", 0.3),
                         reinit_every=config.get("reinit_every", 0), target_overlap=0.3,
                         subdomain_solver="cholesky", transmissibility="interface_aware")
        self.config = config
        self.last_verification = None
        self.shared_operator = None  # Set by the driver while several solves use one state.

    @torch.no_grad()
    def solve_pressure(self, phi, method="fgmres", learned_model=None, coords=None,
                       basis_dim=None, nn_coeff_mode="mobility", tg_config=None, p0=None,
                       tol=1e-6, maxiter=200, preconditioner="asm", gmg_config=None,
                       prepared_coarse_basis=None):
        if method != "fgmres":
            raise ValueError("This driver uses right-preconditioned FGMRES only.")
        op = self.shared_operator
        if op is None:
            op = self.build_operator(phi, build_local_solver=True)
        b = op.rhs(q=None)  # Boundary contribution is part of the effective RHS.
        prepared = prepared_coarse_basis
        if prepared is None and learned_model is not None:
            x, y = self._resolve_predict_coords(coords, 1, self.device)
            raw = learned_model.predict_basis(op.lam.float(), phi.float(), x, y,
                                             active_dim=basis_dim or learned_model.m_dim).double()
            prepared = prepare_two_phase_coarse_basis(op, raw, orth_reg=self.config.get("coarse_reg", 1e-10))
        if prepared is None:
            B = lambda r: op.asm_smooth(torch.zeros_like(r), r, steps=1, omega=0.8)
        else:
            B = lambda r: two_phase_two_grid_sweep(op, torch.zeros_like(r), r, prepared,
                    pre_smooth_steps=1, post_smooth_steps=1, pre_smooth_omega=0.8, post_smooth_omega=0.8)
        answer = fgmres(op.forward, B, b, p0, tol=0.5 * tol, maxiter=maxiter, restart=50)
        matrix = sparse_matrix(op)
        bc, xc = b.cpu().numpy().ravel(), answer.x.cpu().numpy().ravel()
        rho = float(np.linalg.norm(bc - matrix @ xc) / np.linalg.norm(bc))
        accepted = answer.info["converged"] and np.isfinite(rho) and rho <= tol
        self.last_verification = dict(accepted=bool(accepted), cpu_sparse_relative_residual=rho,
                                      effective_rhs_sha256=thash(b), solver_info=answer.info)
        info = PressureSolveInfo(pressure_method="fgmres", num_iter=answer.info["updates"],
            rel_res=rho, converged=bool(accepted), residual_history=answer.info["history"],
            basis_dim=0 if prepared is None else int(prepared["basis_dim"]), preconditioner="asm" if prepared is None else "tg")
        return answer.x, op, info


@torch.no_grad()
def run(args):
    cfg = read_config(args.config)
    device = initialize(cfg.get("seed", 1234), args.device, args.threads)
    output = new_output(args.output)
    if not args.checkpoint:
        raise ValueError("Trajectory generation requires the moving-interface --checkpoint.")
    model = load_model(args.checkpoint, cfg, device, moving=True)
    flow = FGMRESFlow(cfg, device, args.grid, args.subdomains)
    phi = flow.initial_droplet_phi(shape="star", R0=0.18, alpha=0.2, star_k=5)
    steps = args.steps or cfg.get("transport_steps", 1000)
    selected = set(np.linspace(0, steps - 1, min(steps, cfg.get("sampled_operators", 80)), dtype=int).tolist())
    previous, frozen, records, pressure = {}, None, [], None
    states = output / "operators"
    states.mkdir()
    for step in range(steps):
        # Every comparison uses this same state; none changes the transport path.
        if step in selected:
            torch.save(dict(phi=phi.cpu(), step=step), states / f"step{step:04d}.pt")
            # One operator (with its local factors) serves all comparison solves and
            # this step's trajectory solve; it is a function of phi alone.
            op = flow.shared_operator = flow.build_operator(flow._ensure_phi_4d(phi))
            for method in args.methods.split(","):
                if method not in {"learned", "rebuild", "frozen"}:
                    raise ValueError(method)
                prepared = None
                if method != "learned":
                    if method != "frozen" or frozen is None:
                        raw, spectrum = build_threshold_geneo_raw_basis(op, threshold=0.30,
                            boundary_types=BOUNDARIES, sample_id=step, max_local_modes=cfg.get("max_local_modes", 64))
                        dump(output / f"{method}_step{step:04d}_spectrum.json", spectrum)
                        if method == "frozen":
                            frozen = raw
                    else:
                        raw = frozen
                    prepared = prepare_two_phase_coarse_basis(op, raw, orth_reg=cfg.get("coarse_reg", 1e-10), coarse_basis_mode="raw")
                value, _, info = flow.solve_pressure(phi, learned_model=model if method == "learned" else None,
                    prepared_coarse_basis=prepared, p0=previous.get(method), tol=cfg.get("tol", 1e-6),
                    maxiter=cfg.get("max_iter", 200))
                previous[method] = value
                records.append(dict(step=step, method=method, dimension=info.basis_dim,
                    updates=info.num_iter, **flow.last_verification))
                dump(output / "comparison.json", dict(results=records, timing_eligible=False))
        try:
            result = flow.step(phi, p0=pressure, pressure_method="fgmres", learned_model=model,
                               tol=cfg.get("tol", 1e-6), maxiter=cfg.get("max_iter", 200))
        finally:
            flow.shared_operator = None
        if not bool(result["pressure_converged"]):
            raise RuntimeError(f"Trajectory pressure solve failed at step {step}; transport was not accepted.")
        if step in selected:
            np.savez_compressed(output / f"state{step:04d}.npz", phi=phi.cpu().numpy().squeeze(),
                mobility=result["lambda"].cpu().numpy().squeeze(), pressure=result["p"].cpu().numpy().squeeze())
        phi, pressure = result["phi"], result["p"]
    torch.save(dict(phi=phi.cpu(), pressure=pressure.cpu()), output / "terminal_state.pt")
    dump(output / "protocol.json", dict(grid=flow.nx, steps=steps, sampled_steps=sorted(selected),
         methods=args.methods, solver="FGMRES(50)", timing_eligible=False))
    return records
