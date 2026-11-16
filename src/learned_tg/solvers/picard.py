"""Numerical routines used by the paper; no experiment orchestration."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import time
import types
import gc
import numpy as np
import torch
from learned_tg.model.operators.ccfvm_asm import CCFVM_ASM
from learned_tg.model.split_pressure import SplitPressure, StoredFVMResidual, RepresentationError
from learned_tg.model.forchheimer_split import mobility_from_split, relative_pressure_change, nonlinear_converged, save_split_checkpoint, load_split_checkpoint
from learned_tg.solvers.forchheimer_utils import _make_operator, _predict_raw_basis, _prepare_raw_basis, _condition_number

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tensor_sha(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

@dataclass(frozen=True)
class PicardSettings:
    alpha: float=0.
    coarse_reg: float=1e-8
    beta: float=1e-4
    mu: float=1.
    rho: float=1.
    linear_tol: float=1e-6
    update_tol: float=1e-6
    nonlinear_tol: float=1e-6
    linear_max_iter: int=300
    restart: int=50
    safety: float=.5
    max_steps: int=5
    full_trajectory: bool=False

def validate_step_limit(max_steps,full_trajectory=False):
    limit=100 if full_trajectory else 5
    if not 1<=max_steps<=limit:
        raise ValueError(f'set Picard updates between 1 and {limit}; full trajectories require explicit opt-in')

def latest_checkpoint(directory):
    """Newest accepted-step checkpoint of a trajectory directory, or None."""
    found=sorted(Path(directory).glob('checkpoint_*.pt'),key=lambda p:int(p.stem.split('_')[1]))
    return found[-1] if found else None

def stored_coefficients(asm,prefix):
    return {prefix+k:getattr(asm,k).detach().cpu().numpy() for k in ['sw','se','sn','ss','sc']}

@torch.no_grad()
def solve_picard_split(*,model,kappa0,phi,rhs,x_net,y_net,config,out_dir,
                       sample,settings,checkpoint_sha256,paired_reg=None,
                       resume=None,linear_solver=None,predictor=None,case_kind='primary',
                       basis_provider=None,independent_checker=None,protocol_signature=None,
                       save_bases=False,allow_existing=False):
    """Run one Picard trajectory.

    ``save_bases`` also stores the raw and prepared coarse bases of every step
    (their hashes are always recorded). ``allow_existing`` continues ``out_dir``
    in place: a finished trajectory is returned as stored, an interrupted one
    restarts from its newest checkpoint and reruns only the unfinished step.
    """
    validate_step_limit(settings.max_steps,settings.full_trajectory)
    if settings.coarse_reg<=0:raise ValueError('coarse regularization must be positive')
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=allow_existing)
    n=int(rhs.shape[-1]);k_1d=int(config.get('k_1d',10));dimension=int(config.get('m_dim',30))
    signature=dict(field=int(sample),alpha=float(settings.alpha),coarse_reg=float(settings.coarse_reg),
        beta=float(settings.beta),mu=float(settings.mu),rho=float(settings.rho),
        linear_tol=settings.linear_tol,update_tol=settings.update_tol,nonlinear_tol=settings.nonlinear_tol,
        linear_max_iter=settings.linear_max_iter,restart=settings.restart,safety=settings.safety,
        n=n,checkpoint_sha256=checkpoint_sha256,kappa_sha256=tensor_sha(kappa0),
        phi_sha256=tensor_sha(phi),rhs_sha256=tensor_sha(rhs))
    if protocol_signature is not None:signature['protocol_signature']=protocol_signature
    if basis_provider is not None:signature['basis_provider']=basis_provider.signature
    if allow_existing and (out/'summary.json').exists():
        stored=json.loads((out/'resolved_config.json').read_text())['signature']
        if stored!=json.loads(json.dumps(signature)):
            raise ValueError(f'--resume found a trajectory with different settings in {out}')
        state=SplitPressure.zeros_like(rhs);history=[];last=latest_checkpoint(out)
        if last is not None:
            state,metadata=load_split_checkpoint(last,device=rhs.device,expected=signature);history=list(metadata['history'])
        return history,json.loads((out/'summary.json').read_text()),state,[]
    if allow_existing and resume is None:resume=latest_checkpoint(out)
    write_json(out/'resolved_config.json',dict(signature=signature,settings=asdict(settings),
        checkpoint_config=config,case_kind=case_kind,paired_reg=paired_reg,
        representation='two_float64_components',solver='FGMRES_split_compensated'))
    torch.save(dict(kappa0=kappa0.cpu(),phi=phi.cpu(),rhs=rhs.cpu()),out/'inputs.pt')
    state=SplitPressure.zeros_like(rhs);history=[];start_step=0
    if resume is not None:
        state,metadata=load_split_checkpoint(resume,device=rhs.device,expected=signature)
        start_step=int(metadata['step']);history=list(metadata['history'])
        if start_step>settings.max_steps:raise ValueError('checkpoint is beyond the bounded target')
    predictor=predictor or _predict_raw_basis
    new_rows=[];shadows=[];failure=None
    parameters=dict(alpha=settings.alpha,beta=settings.beta,mu=settings.mu,rho=settings.rho,p0=0.)
    already_converged=bool(history and all(r['linear_accepted'] for r in history) and nonlinear_converged(
        history[-1]['relative_pressure_change'],history[-1]['nonlinear_residual'],
        update_tol=settings.update_tol,residual_tol=settings.nonlinear_tol))
    if 0<start_step<settings.max_steps and not already_converged and hasattr(basis_provider,'restore'):
        # Every trajectory starts from zero pressure, so its first operator can be rebuilt.
        initial=mobility_from_split(kappa0,SplitPressure.zeros_like(rhs),**parameters)
        basis_provider.restore(_make_operator(initial.mobility,phi,n=n,k_1d=k_1d,config=config),history)
    for step in ([] if already_converged else range(start_step+1,settings.max_steps+1)):
        before=state.clone()
        try:mobility=mobility_from_split(kappa0,before,**parameters)
        except RepresentationError as error:
            failure=dict(step=step,status='mobility_failure',detail=str(error));break
        op=_make_operator(mobility.mobility,phi,n=n,k_1d=k_1d,config=config)
        if op.asm.subdomain_solver!='cholesky' or op.asm.cholesky_backend!='blocktri':
            raise ValueError('split Forchheimer validation requires the existing blocktri Cholesky')
        provider_metadata={}
        if basis_provider is None:
            raw,_=predictor(model,mobility.mobility,phi,x_net,y_net,m_dim=dimension)
        else:
            raw,provider_metadata=basis_provider(op,step,mobility.mobility)
        raw_sha=tensor_sha(raw);frozen_hash={k:tensor_sha(getattr(op.asm,k)) for k in ['sw','se','sn','ss','sc']}
        base_forward=op.asm.forward;counters=[0,0]
        def counted(self,x):
            counters[0]+=1;counters[1]+=int(x.shape[0]*x.shape[1]);return base_forward(x)
        op.asm.forward=types.MethodType(counted,op.asm)

        def attempt(reg,kind):
            dest=out/f'step_{step:02d}' if kind==case_kind else out/f'paired_reg{reg:g}'/f'step_{step:02d}'
            dest.mkdir(parents=True,exist_ok=allow_existing)
            counters[:]=[0,0]
            preparation=dict(config,basis_orth_reg=float(reg))
            if basis_provider is not None:preparation.update(basis_provider.preparation)
            prepared=_prepare_raw_basis(op,raw,preparation)
            assert prepared['orth_reg']==reg and tensor_sha(raw)==raw_sha
            setup_actions=counters[1];counters[:]=[0,0]
            evaluator=StoredFVMResidual(op.asm)
            warm_residual=evaluator.residual(rhs,before)
            initial_relative=float(torch.linalg.vector_norm(warm_residual)/torch.linalg.vector_norm(rhs))
            warm_high=tensor_sha(before.high);warm_low=tensor_sha(before.low)
            if rhs.device.type=='cuda':
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();resident=torch.cuda.memory_allocated()
            else:resident=0
            tic=time.perf_counter()
            result=(linear_solver(op,before,rhs,prepared,settings) if linear_solver is not None else
                op.fgmres_split_solve(before,rhs,prepared,max_iter=settings.linear_max_iter,
                    m_max=settings.restart,tol=settings.linear_tol,safety=settings.safety))
            verified=evaluator.residual(rhs,result.pressure)
            linear_relative=float(torch.linalg.vector_norm(verified)/torch.linalg.vector_norm(rhs))
            accepted=bool(result.info['converged'] and np.isfinite(linear_relative) and linear_relative<settings.linear_tol)
            if rhs.device.type=='cuda':torch.cuda.synchronize()
            elapsed=time.perf_counter()-tic
            assert tensor_sha(before.high)==warm_high and tensor_sha(before.low)==warm_low
            row=dict(field=int(sample),alpha=settings.alpha,picard_step=step,coarse_reg=reg,
                kind=kind,solver='FGMRES_split_compensated',fgmres_updates=result.info['updates'],
                linear_true_residual=linear_relative,initial_split_residual=initial_relative,
                linear_accepted=accepted,status='linear_accepted' if accepted else 'linear_failure',
                nonlinear_converged=False,coarse_dimension=int(prepared['basis_dim']),
                coarse_condition=_condition_number(prepared['A_rom_reg']),
                raw_span_sha256=raw_sha,Q_sha256=tensor_sha(prepared['Q']),
                coarse_Gram_sha256=tensor_sha(prepared['A_rom_reg']),
                frozen_mobility_sha256=tensor_sha(mobility.mobility),frozen_coefficient_hashes=frozen_hash,
                lambda_min=float(mobility.mobility.min()),lambda_max=float(mobility.mobility.max()),
                lambda_contrast=float(mobility.mobility.max()/mobility.mobility.min()),
                effective_kappa_min=float(mobility.permeability.min()),effective_kappa_max=float(mobility.permeability.max()),
                warm_high_sha256=warm_high,warm_low_sha256=warm_low,
                warm_low_norm=float(torch.linalg.vector_norm(before.low)),linear_info=result.info,
                ordinary_stencil_calls=counters[0],ordinary_vector_actions=counters[1],
                basis_preparation_vector_actions=setup_actions,
                external_accurate_residual_evaluations=evaluator.evaluations,
                solve_seconds_shared_gpu=elapsed,
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if rhs.device.type=='cuda' else 0,
                peak_incremental_bytes=torch.cuda.max_memory_allocated()-resident if rhs.device.type=='cuda' else 0)
            row.update(provider_metadata)
            row.update(actual_coarse_regularization=prepared['orth_reg'],
                prepared_basis_rule=prepared['orthonormalize_metric'],
                coarse_solver=prepared['coarse_solver'],coarse_factor_sha256=tensor_sha(prepared['coarse_factor']),
                coarse_min_eigenvalue=float(torch.linalg.eigvalsh(prepared['A_rom_reg']).min()))
            row['total_linear_matrix_action_equivalents']=counters[1]+result.info.get('correction_matrix_applications',0)+2*(evaluator.evaluations+result.info.get('accurate_residual_evaluations',0))
            torch.save(before.export(),dest/'warm_pressure.pt')
            torch.save(result.pressure.export(),dest/'pressure.pt')
            torch.save({k:prepared[k].cpu() for k in ['A_rom','A_rom_reg','coarse_factor']},dest/'coarse_small_matrices.pt')
            if basis_provider is not None and save_bases:
                torch.save(raw.cpu(),dest/'raw_span.pt')
                torch.save(prepared['Q'].cpu(),dest/'prepared_basis.pt')
            arrays=stored_coefficients(op.asm,'linear_')
            arrays.update(frozen_mobility=mobility.mobility.cpu().numpy(),
                frozen_permeability=mobility.permeability.cpu().numpy())
            if accepted:
                updated=mobility_from_split(kappa0,result.pressure,**parameters)
                nonlinear_op=CCFVM_ASM(updated.mobility,build_local_solver=False,phi=phi,
                    transmissibility=str(config.get('transmissibility','interface_aware')))
                nonlinear_evaluator=StoredFVMResidual(nonlinear_op)
                nonlinear=nonlinear_evaluator.residual(rhs,result.pressure)
                nonlinear_relative=float(torch.linalg.vector_norm(nonlinear)/torch.linalg.vector_norm(rhs))
                change=relative_pressure_change(result.pressure,before)
                row.update(nonlinear_residual=nonlinear_relative,relative_pressure_change=change,
                    nonlinear_converged=nonlinear_converged(change,nonlinear_relative,
                        update_tol=settings.update_tol,residual_tol=settings.nonlinear_tol),
                    updated_mobility_sha256=tensor_sha(updated.mobility),
                    updated_lambda_min=float(updated.mobility.min()),updated_lambda_max=float(updated.mobility.max()),
                    updated_lambda_contrast=float(updated.mobility.max()/updated.mobility.min()),
                    nonlinear_accurate_residual_evaluations=nonlinear_evaluator.evaluations,
                    pressure_low_norm=float(torch.linalg.vector_norm(result.pressure.low)),
                    pressure_low_nonzero=int(torch.count_nonzero(result.pressure.low)))
                arrays.update(stored_coefficients(nonlinear_op,'nonlinear_'))
                arrays.update(updated_mobility=updated.mobility.cpu().numpy(),
                    updated_permeability=updated.permeability.cpu().numpy(),
                    updated_gradient_magnitude=updated.gradient_magnitude.cpu().numpy())
                del nonlinear_op,nonlinear_evaluator,nonlinear,updated
            temp=dest/'stored_systems.npz.tmp'
            with temp.open('wb') as f:np.savez_compressed(f,**arrays)
            temp.replace(dest/'stored_systems.npz')
            row['artifact_dir']=str(dest)
            write_json(dest/'result.json',row)
            if accepted and independent_checker is not None:
                independent_checker(dest)
            return result.pressure,row

        try:
            candidate,row=attempt(settings.coarse_reg,case_kind)
            if paired_reg is not None:
                _,shadow=attempt(float(paired_reg),'paired_sensitivity')
                assert shadow['raw_span_sha256']==row['raw_span_sha256'] and shadow['frozen_coefficient_hashes']==row['frozen_coefficient_hashes']
                shadows.append(shadow)
            if not row['linear_accepted']:
                failure=dict(step=step,status='linear_failure',detail=row['linear_info']['status']);break
            state=candidate;new_rows.append(row);history.append(row)
            metadata=dict(signature,step=step,history=history,checkpoint_config=config,
                          settings=asdict(settings),case_kind=case_kind)
            save_split_checkpoint(out/f'checkpoint_{step:02d}.pt',state,metadata)
            print(json.dumps({k:row[k] for k in ['field','alpha','picard_step','coarse_reg','fgmres_updates','linear_true_residual','nonlinear_residual','relative_pressure_change','status']}),flush=True)
            if row['nonlinear_converged']:break
        except RepresentationError as error:
            failure=dict(step=step,status='representation_failure',detail=str(error));break
        finally:
            op.asm.forward=base_forward
            del op,base_forward,counted,attempt,raw
            gc.collect()
            if rhs.device.type=='cuda':torch.cuda.empty_cache()
    all_ok=bool(history and failure is None and all(r['linear_accepted'] for r in history))
    final_nonlinear=bool(all_ok and nonlinear_converged(history[-1]['relative_pressure_change'],history[-1]['nonlinear_residual'],
        update_tol=settings.update_tol,residual_tol=settings.nonlinear_tol))
    summary=dict(field=sample,alpha=settings.alpha,coarse_reg=settings.coarse_reg,case_kind=case_kind,
        accepted_updates=len(history),new_updates=len(new_rows),start_step=start_step,
        nonlinear_converged=final_nonlinear,all_linear_accepted=all_ok,failure=failure,
        status=failure['status'] if failure else 'nonlinear_converged' if final_nonlinear else
            'picard_iteration_limit' if settings.full_trajectory else 'bounded_step_limit',
        total_fgmres_updates=sum(r['fgmres_updates'] for r in history),
        newly_computed_fgmres_updates=sum(r['fgmres_updates'] for r in new_rows),
        maximum_linear_residual=max((r['linear_true_residual'] for r in history),default=None))
    write_json(out/'summary.json',summary)
    return history,summary,state,shadows
