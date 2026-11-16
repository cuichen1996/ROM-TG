"""Numerical routines used by the paper; no experiment orchestration."""
from __future__ import annotations
import hashlib, json, math, types
from pathlib import Path
import numpy as np
import scipy.linalg as la
import torch
from learned_tg.coarse.pod import _make_prepared_coarse_basis
from learned_tg.model.krylov import fgmres
from learned_tg.validation.residuals import evaluate
from learned_tg.validation.split import norm

PROJECT = Path(__file__).resolve().parents[1]

def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    if isinstance(x,torch.Tensor):return safe(x.detach().cpu().tolist())
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,np.generic):return safe(x.item())
    if isinstance(x,Path):return str(x)
    if isinstance(x,float) and not math.isfinite(x):return {'nonfinite_float':str(x)}
    return x

def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(safe(data),indent=2,allow_nan=False)+'\n');tmp.replace(path)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def thash(t):
    # Hash the tensor's bytes in place; ``tobytes`` would copy a multi-GB basis first.
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().reshape(-1).data).hexdigest()

def record_protocol(path,protocol,resume=False):
    """Write the run protocol, or on resume require that it is unchanged."""
    path=Path(path);protocol=safe(protocol)
    if resume and path.exists():
        previous=json.loads(path.read_text())
        changed=sorted(k for k in set(previous)|set(protocol) if previous.get(k)!=protocol.get(k))
        if changed:raise ValueError(f'--resume needs the original run settings; these differ from {path}: {changed}')
    else:dump(path,protocol)

def cached_metrics(dest,base):
    """Return a finished case's stored row, after checking it is the same linear system."""
    path=Path(dest)/'metrics.json'
    if not path.exists():return None
    row=json.loads(path.read_text())
    for key in ['n','sample_id','field_sha256','rhs_sha256']:
        if row.get(key)!=base[key]:raise ValueError(f'--resume found a result for a different system in {path}: {key}')
    return row

def prepare(op,raw,mode='a_orthonormal',expected=None):
    assert bool(torch.isfinite(raw).all())
    AQ=op.asm.apply_A(raw);assert bool(torch.isfinite(AQ).all())
    gram=(raw.transpose(1,2)@AQ);vals=torch.linalg.eigvalsh((gram+gram.mT)/2)
    floor=torch.finfo(raw.dtype).eps*raw.shape[-1]*vals.abs().max()
    rank=int((vals>floor).sum())
    if expected is not None and rank!=expected:raise ValueError(f'construction_failure: reliable energy rank {rank} != {expected}')
    prepared=_make_prepared_coarse_basis(raw,AQ,reg=1e-10,coarse_basis_mode=mode)
    assert prepared['orth_reg']==1e-10
    return prepared,dict(input_energy_rank=rank,input_energy_rank_floor=float(floor))

def independent(A,op,b,x):
    bc=b.cpu().numpy().ravel();xc=x.cpu().numpy().ravel();bn=la.norm(bc)
    sparse=float(la.norm(bc-A@xc)/bn)
    # CPU expansion check evaluates the SAME stored coefficient matrix.
    cs=[getattr(op.asm,k).cpu().numpy().squeeze() for k in ['sw','se','sn','ss','sc']]
    hh=x.cpu().numpy().squeeze();rhs=b.cpu().numpy().squeeze()
    h,l=evaluate(cs,hh,np.zeros_like(hh),rhs);dd=norm(h+l)/norm(rhs)
    return dict(cpu_sparse_relative_residual=sparse,cpu_dd_relative_residual=dd,
        accepted=bool(np.isfinite(sparse) and np.isfinite(dd) and sparse<=1e-6 and dd<=1e-6))

def tg(op,b,prepared,x):
    return op.two_grid(x,b,prepared_coarse_basis=prepared,M_smooth=1,omega=.8,
        pre_smooth_mode='asm',pre_smooth_steps=1,pre_smooth_omega=.8,post_smooth_mode='asm',post_smooth_steps=1,post_smooth_omega=.8,
        basis_smooth_mode='none',basis_smooth_steps=0,basis_smooth_omega=0.,coarse_basis_orth_metric='a',orth_reg=1e-10)

@torch.no_grad()
def solve(op,A,b,prepared,method,solver,dest,base,resume=False):
    if (dest/'metrics.json').exists():return json.loads((dest/'metrics.json').read_text())
    # A directory without metrics.json is an interrupted case; resume reruns it in place.
    dest.mkdir(parents=True,exist_ok=resume);n=b.shape[-1];zero=torch.zeros_like(b)
    B=(lambda r:op.solution_smooth(torch.zeros_like(r),r,mode='asm',M_smooth=1,omega=.8)) if method=='ASM' else (lambda r:tg(op,r,prepared,torch.zeros_like(r)))
    raw_forward=op.asm.forward;actions=[0,0]
    def counted(self,x):actions[0]+=1;actions[1]+=int(x.shape[0]*x.shape[1]);return raw_forward(x)
    op.asm.forward=types.MethodType(counted,op.asm)
    try:
        true_curve=[];observer=None
        if solver=='FGMRES' and base.get('record_true_curve',False):
            state={'cycle':None,'xbase':zero.clone(),'last':zero.clone(),'Z':[],'H':torch.zeros((51,50),dtype=torch.float64)}
            true_curve=[dict(j=0,cpu_sparse_relative_residual=1.)]
            def observer(event):
                if state['cycle']!=event['cycle']:
                    state.update(cycle=event['cycle'],xbase=state['last'].clone(),Z=[],H=torch.zeros((51,50),dtype=torch.float64))
                j=event['column'];state['Z'].append(event['z']);state['H'][:j+2,j]=event['H_column']
                beta=float(torch.linalg.vector_norm(b-op.forward(state['xbase'])))
                rhs=torch.zeros(j+2,dtype=torch.float64);rhs[0]=beta
                y=torch.linalg.lstsq(state['H'][:j+2,:j+1],rhs,driver='gelsd').solution
                dx=torch.zeros_like(b)
                for yi,zi in zip(y,state['Z']):dx=dx+float(yi)*zi
                state['last']=state['xbase']+dx
                cpu=b.cpu().numpy().ravel()-A@state['last'].cpu().numpy().ravel()
                true_curve.append(dict(j=event['j']+1,cpu_sparse_relative_residual=float(la.norm(cpu)/la.norm(b.cpu().numpy().ravel()))))
        if solver=='FGMRES':
            answer=fgmres(op.forward,B,b,zero,tol=1e-6,maxiter=200,restart=50,observer=observer)
            x=answer.x;info=answer.info;info['algorithm']='right_FGMRES_50_two_pass'
            if observer is not None:assert torch.allclose(x,state['last'],rtol=1e-13,atol=1e-15)
        else:
            x=zero.clone();history=[1.];info=dict(updates=0,algorithm='standalone_TG',status='iteration_limit',restarts=0,
                preconditioner_applications=0,matrix_vector_applications=0,true_residual_evaluations=1)
            for j in range(1,201):
                x=tg(op,b,prepared,x);rho=float(torch.linalg.vector_norm(b-op.forward(x))/torch.linalg.vector_norm(b));history.append(rho)
                info.update(updates=j,preconditioner_applications=j,true_residual_evaluations=j+1)
                if not bool(torch.isfinite(x).all()):info['status']='nonfinite';break
                if rho<=1e-6 and independent(A,op,b,x)['accepted']:info['status']='converged';break
            info.update(history=history,trace=[dict(j=j,true_relative_residual=v) for j,v in enumerate(history)],converged=info['status']=='converged')
        counters=dict(ordinary_stencil_calls=actions[0],ordinary_vector_actions=actions[1])
    finally:op.asm.forward=raw_forward
    check=independent(A,op,b,x)
    accepted=bool(info.get('converged') and check['accepted'])
    status='accepted' if accepted else 'independent_rejection' if info.get('converged') else info['status']
    torch.save(x.cpu(),dest/'solution.pt');dump(dest/'residual_history.json',dict(initial_update=0,inner_estimates=info['history'],true_checkpoints=info.get('trace',[]),independent_prefix_true_curve=true_curve,accepted_endpoint=check))
    row=dict(base,method=method,solver=solver,dimension=0 if prepared is None else int(prepared['basis_dim']),
        updates=info['updates'],status=status,accepted=accepted,independent=check,solver_info=info,**counters,
        TG_applications=0 if method=='ASM' else info.get('preconditioner_applications',info['updates']),
        ASM_applications=(1 if method=='ASM' else 2)*info.get('preconditioner_applications',info['updates']),
        solution_sha256=thash(x),kernel_sha256=sha(PROJECT/'model/krylov.py'),timing_eligible=False)
    dump(dest/'metrics.json',row);dump(dest/'status.json',dict(status=status,independent_checked=True,metrics_sha256=sha(dest/'metrics.json')))
    print(json.dumps({k:row[k] for k in ['n','sample_id','method','solver','updates','status']}),flush=True)
    return row
