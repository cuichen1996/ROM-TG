"""Representation-aware Forchheimer quantities and checkpoint interchange.

The original central/one-sided gradient stencils and constitutive law are
preserved. Pressure cancellation is evaluated before rounding gradients;
mobilities and assembled FVM coefficients remain float64.
"""
from dataclasses import dataclass
from pathlib import Path
import torch

from learned_tg.model.split_pressure import SplitPressure, RepresentationError, finite64, two_prod, expansion_add


def scaled_value(high,low,scale):
    h,l=two_prod(high,float(scale));hh,ll=two_prod(low,float(scale))
    h,l=expansion_add(h,l,hh,ll)
    return h+l


def split_gradient(state,h):
    if not isinstance(state,SplitPressure):raise TypeError('split pressure required')
    ph,pl=state.high,state.low;gx=torch.zeros_like(ph);gy=torch.zeros_like(ph)
    if ph.shape[-1]>1:
        dh,dl=expansion_add(ph[...,2:],pl[...,2:],-ph[...,:-2],-pl[...,:-2])
        gx[...,1:-1]=scaled_value(dh,dl,1/(2*h))
        gx[...,0]=scaled_value(ph[...,0],pl[...,0],1/(.5*h))
        gx[...,-1]=scaled_value(ph[...,-1],pl[...,-1],-1/(.5*h))
    if ph.shape[-2]>1:
        dh,dl=expansion_add(ph[...,2:,:],pl[...,2:,:],-ph[...,:-2,:],-pl[...,:-2,:])
        gy[...,1:-1,:]=scaled_value(dh,dl,1/(2*h))
        gy[...,0,:]=scaled_value(ph[...,0,:],pl[...,0,:],1/(.5*h))
        gy[...,-1,:]=scaled_value(ph[...,-1,:],pl[...,-1,:],-1/(.5*h))
    finite64(gx,'pressure gradient x');finite64(gy,'pressure gradient y')
    return gx,gy


def effective_permeability(kappa,state,*,alpha,p0=0.):
    finite64(kappa,'base permeability')
    if not isinstance(state,SplitPressure):raise TypeError('split pressure required')
    if alpha==0:
        effective=kappa
    else:
        ph,pl=expansion_add(state.high,state.low,torch.full_like(state.high,-p0),torch.zeros_like(state.low))
        eh,el=two_prod(ph,float(alpha));lh,ll=two_prod(pl,float(alpha))
        eh,el=expansion_add(eh,el,lh,ll)
        base=torch.exp(eh)
        effective=kappa*(base+base*torch.expm1(el))
    finite64(effective,'effective permeability')
    if not bool((effective>0).all()):raise RepresentationError('nonpositive permeability')
    return effective


@dataclass
class MobilityState:
    permeability: torch.Tensor
    gradient_magnitude: torch.Tensor
    mobility: torch.Tensor


def mobility_from_split(kappa,state,*,alpha,beta=1e-4,mu=1.,rho=1.,p0=0.):
    gx,gy=split_gradient(state,1/state.high.shape[-1])
    magnitude=torch.sqrt(gx*gx+gy*gy+1e-300)
    effective=effective_permeability(kappa,state,alpha=alpha,p0=p0)
    inv_k=float(mu)/effective.clamp_min(1e-300)
    G=.5*(inv_k+torch.sqrt(inv_k*inv_k+4.*float(beta)*float(rho)*magnitude))
    mobility=1./G.clamp_min(1e-300)
    finite64(mobility,'mobility')
    if not bool((mobility>0).all()):raise RepresentationError('nonpositive mobility')
    return MobilityState(effective,magnitude,mobility)


def split_mobility(kappa,state,**kwargs):
    value=mobility_from_split(kappa,state,**kwargs)
    return value.mobility,value.gradient_magnitude


def relative_pressure_change(new,old):
    h,l=expansion_add(new.high,new.low,-old.high,-old.low)
    return SplitPressure(h,l).norm()/max(new.norm(),1e-30)


def nonlinear_converged(relative_update,nonlinear_residual,*,update_tol=1e-6,residual_tol=1e-6,all_linear_accepted=True):
    return bool(all_linear_accepted and relative_update<=update_tol and nonlinear_residual<=residual_tol)


def resolve_coarse_regularization(config,explicit=None):
    if explicit is not None:return float(explicit),'explicit'
    if 'basis_orth_reg' in config:return float(config['basis_orth_reg']),'basis_orth_reg'
    return float(config.get('loss_reg',1e-8)),'historical_loss_reg_fallback'


def save_split_checkpoint(path,state,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    payload=dict(format='split-pressure-picard-v1',pressure=state.export(),metadata=metadata)
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(payload,temp);temp.replace(path)


def load_split_checkpoint(path,*,device,expected=None):
    payload=torch.load(path,map_location='cpu')
    if payload.get('format')!='split-pressure-picard-v1':raise ValueError('unsupported pressure checkpoint')
    for key,value in (expected or {}).items():
        if payload['metadata'].get(key)!=value:raise ValueError(f'checkpoint mismatch: {key}')
    h,l=payload['pressure']['high'],payload['pressure']['low']
    finite64(h,'checkpoint high');finite64(l,'checkpoint low')
    return SplitPressure(h.to(device),l.to(device)),payload['metadata']
