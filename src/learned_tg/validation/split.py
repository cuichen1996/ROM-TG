"""Numerical routines used by the paper; no experiment orchestration."""
from __future__ import annotations
from decimal import Decimal, localcontext
import json
import numpy as np
import torch
from learned_tg.validation.residuals import evaluate, decimal_check, add
from learned_tg.validation.physics import decimal_quantities
from learned_tg.io import dump, sha

KEYS=['sw','se','sn','ss','sc']

def norm(x):return float(np.sqrt(np.sum(np.asarray(x,dtype=np.longdouble)**2,dtype=np.longdouble)))

def pressure(path):
    d=torch.load(path,map_location='cpu');return [d[k].numpy().squeeze() for k in ['high','low']]

def gradients(h,l):
    n=h.shape[-1];gx=np.zeros_like(h);gy=np.zeros_like(h)
    def difference(a,b,c,d):
        hi,lo=add(a,b,-c,-d);return hi.astype(np.longdouble)+lo.astype(np.longdouble)
    scale=np.longdouble(float(1/(2*(1/n))));boundary=np.longdouble(float(1/(.5*(1/n))))
    gx[:,1:-1]=difference(h[:,2:],l[:,2:],h[:,:-2],l[:,:-2])*scale
    gy[1:-1,:]=difference(h[2:,:],l[2:,:],h[:-2,:],l[:-2,:])*scale
    gx[:,0]=(h[:,0].astype(np.longdouble)+l[:,0])*boundary
    gx[:,-1]=-(h[:,-1].astype(np.longdouble)+l[:,-1])*boundary
    gy[0,:]=(h[0,:].astype(np.longdouble)+l[0,:])*boundary
    gy[-1,:]=-(h[-1,:].astype(np.longdouble)+l[-1,:])*boundary
    return gx,gy

def quantities(h,l,k,alpha):
    gx,gy=gradients(h,l)
    # Constitutive evaluation independently uses extended CPU scalars.
    magnitude=np.sqrt(gx.astype(np.longdouble)**2+gy.astype(np.longdouble)**2+np.longdouble(1e-300))
    permeability=k.astype(np.longdouble)*np.exp(np.longdouble(alpha)*(h.astype(np.longdouble)+l.astype(np.longdouble)))
    inv=1/permeability;mobility=2/(inv+np.sqrt(inv**2+4*np.longdouble(1e-4)*magnitude))
    return magnitude,permeability,mobility

def assemble(m,phi):
    def face(a,b,p,q):
        harmonic=2*a*b/(a+b);theta=np.abs(p)/np.maximum(np.abs(p)+np.abs(q),np.finfo(float).eps)
        return np.where(p*q<0,1/np.maximum(theta/a+(1-theta)/b,np.finfo(float).eps),harmonic)
    tx=face(m[:,:-1],m[:,1:],phi[:,:-1],phi[:,1:]);ty=face(m[:-1,:],m[1:,:],phi[:-1,:],phi[1:,:])
    w=-np.concatenate([m[:,:1],tx],axis=1);e=-np.concatenate([tx,m[:,-1:]],axis=1)
    n=-np.concatenate([m[:1,:],ty],axis=0);s=-np.concatenate([ty,m[-1:,:]],axis=0)
    w[:,0]*=2;e[:,-1]*=2;n[0,:]*=2;s[-1,:]*=2
    return [w,e,n,s,-(w+e+n+s)]

def selected_cells(k):
    n=k.shape[-1];high=k>1;adj=np.zeros_like(high)
    adj[:,1:]|=high[:,1:]!=high[:,:-1];adj[:,:-1]|=high[:,1:]!=high[:,:-1]
    adj[1:,:]|=high[1:,:]!=high[:-1,:];adj[:-1,:]|=high[1:,:]!=high[:-1,:]
    boundary=np.zeros_like(high);boundary[[0,-1],:]=True;boundary[:,[0,-1]]=True
    rng=np.random.default_rng(20261031)
    masks=dict(interior=~boundary,boundary=boundary,high_permeability=high,low_permeability=~high,interface_adjacent=adj)
    return {name:sorted(rng.choice(np.flatnonzero(mask),size=min(32,int(mask.sum())),replace=False).tolist()) for name,mask in masks.items()}

def high_precision(h,l,k,alpha,gradient,mobility,selected):
    ch=h+l;cg,_,cm=quantities(ch,np.zeros_like(l),k,alpha);rows={}
    # Collapsed comparisons below reproduce rounded float64 subtractions.
    n=h.shape[-1];gx=np.zeros_like(h);gy=np.zeros_like(h);hh=1/n
    gx[:,1:-1]=(ch[:,2:]-ch[:,:-2])/(2*hh);gy[1:-1,:]=(ch[2:,:]-ch[:-2,:])/(2*hh)
    gx[:,0]=ch[:,0]/(.5*hh);gx[:,-1]=-ch[:,-1]/(.5*hh)
    gy[0,:]=ch[0,:]/(.5*hh);gy[-1,:]=-ch[-1,:]/(.5*hh)
    cg=np.sqrt(gx*gx+gy*gy+1e-300);ek=k*np.exp(alpha*ch);inv=1/ek
    cm=1/(.5*(inv+np.sqrt(inv*inv+4e-4*cg)))
    with localcontext() as ctx:
        ctx.prec=100;D=lambda x:Decimal.from_float(float(x))
        for category,indices in selected.items():
            errors={key:[] for key in ['split_gradient','collapsed_gradient','split_mobility','collapsed_mobility']}
            for index in indices:
                i,j=divmod(index,n);g,m=decimal_quantities(h,l,k,i,j,alpha)
                for key,value,reference in [('split_gradient',gradient[i,j],g),('collapsed_gradient',cg[i,j],g),
                    ('split_mobility',mobility[i,j],m),('collapsed_mobility',cm[i,j],m)]:
                    errors[key].append(float(abs(D(value)-reference)/reference))
            rows[category]=dict(cells=len(indices),**{key+'_max_relative_error':max(value,default=0.) for key,value in errors.items()})
    assert max(r['split_gradient_max_relative_error'] for r in rows.values())<3e-15
    assert max(r['split_mobility_max_relative_error'] for r in rows.values())<3e-15
    return rows

def conservation(cs,h,l,rhs,rh,rl):
    w,e,n,s,c=cs
    assert all(np.isfinite(x).all() for x in cs)
    assert (c>0).all() and all((x<0).all() for x in cs[:4])
    mismatch=max(float(np.max(np.abs(w[:,1:]-e[:,:-1]))),float(np.max(np.abs(n[1:,:]-s[:-1,:]))))
    assert mismatch==0
    # Stored row defects are retained, including the roundoff in the diagonal.
    row=c.astype(np.longdouble)
    for a in cs[:4]:row=row+a.astype(np.longdouble)
    ph=h.astype(np.longdouble)+l.astype(np.longdouble)
    boundary=-(np.sum(w[:,0].astype(np.longdouble)*ph[:,0])+np.sum(e[:,-1].astype(np.longdouble)*ph[:,-1])+
        np.sum(n[0,:].astype(np.longdouble)*ph[0,:])+np.sum(s[-1,:].astype(np.longdouble)*ph[-1,:]))
    defect=np.sum(row*ph,dtype=np.longdouble)
    total_source=np.sum(rhs,dtype=np.longdouble);total_residual=np.sum(rh.astype(np.longdouble)+rl,dtype=np.longdouble)
    balance=abs(total_source-boundary-defect-total_residual)
    assert balance<1e-10*max(abs(total_source),1)
    return dict(shared_face_mismatch=mismatch,source=float(total_source),boundary_outflow=float(boundary),
        stored_row_defect_contribution=float(defect),signed_residual_sum=float(total_residual),balance_error=float(balance),
        maximum_absolute_interior_row_defect=float(np.max(np.abs(row[1:-1,1:-1]))))

def check_step(path,inputs,alpha):
    cached=path/'independent_validation.json'
    if cached.exists():
        previous=json.loads(cached.read_text())
        if (previous.get('validator_sha256')==sha(__file__) and previous['result_sha256']==sha(path/'result.json')
            and previous['pressure_sha256']==sha(path/'pressure.pt') and previous['systems_sha256']==sha(path/'stored_systems.npz')):
            assert previous['independent_linear_residual']<1e-6
            return previous
    row=json.loads((path/'result.json').read_text());d=np.load(path/'stored_systems.npz')
    h,l=pressure(path/'pressure.pt');wh,wl=pressure(path/'warm_pressure.pt')
    b=inputs['rhs'].numpy().squeeze();k=inputs['kappa0'].numpy().squeeze();phi=inputs['phi'].numpy().squeeze();bn=norm(b)
    frozen=[d['linear_'+key].squeeze() for key in KEYS];nonlinear=[d['nonlinear_'+key].squeeze() for key in KEYS]
    rh,rl=evaluate(frozen,h,l,b);nrh,nrl=evaluate(nonlinear,h,l,b)
    linear=norm(rh+rl)/bn;nl=norm(nrh+nrl)/bn
    wrh,wrl=evaluate(frozen,wh,wl,b);warm=norm(wrh+wrl)/bn
    assert abs(warm-row['initial_split_residual'])<1e-12*max(1,warm)
    if row['picard_step']>1:assert np.count_nonzero(wl)>0
    assert row['linear_accepted'] and linear<1e-6
    assert abs(linear-row['linear_true_residual'])<1e-15 and abs(nl-row['nonlinear_residual'])<1e-12*max(1,nl)
    decimal_linear=decimal_check(frozen,h,l,b,rh,rl);decimal_nonlinear=decimal_check(nonlinear,h,l,b,nrh,nrl)
    changeh,changel=add(h,l,-wh,-wl);change=norm(changeh.astype(np.longdouble)+changel)/norm(h.astype(np.longdouble)+l)
    assert abs(change-row['relative_pressure_change'])<5e-15*max(1,change)
    selected=selected_cells(k);physical={}
    for label,hh,ll,stored_m,stored_k in [
        ('frozen',wh,wl,d['frozen_mobility'].squeeze(),d['frozen_permeability'].squeeze()),
        ('updated',h,l,d['updated_mobility'].squeeze(),d['updated_permeability'].squeeze())]:
        g,ek,m=quantities(hh,ll,k,alpha)
        merr=float(np.max(np.abs(stored_m.astype(np.longdouble)-m)/m))
        kerr=float(np.max(np.abs(stored_k.astype(np.longdouble)-ek)/ek))
        assert merr<5e-14 and kerr<3e-15,(path,label,merr,kerr)
        assert np.isfinite(stored_m).all() and (stored_m>0).all()
        if alpha==0:assert np.array_equal(stored_k,k)
        rebuilt=assemble(stored_m,phi);cs=frozen if label=='frozen' else nonlinear
        coeff_error=max(float(np.max(np.abs(a-b)/np.maximum(np.abs(b),1e-300))) for a,b in zip(rebuilt,cs))
        assert coeff_error<1e-15,(path,label,coeff_error)
        physical[label]=dict(full_field_mobility_max_relative_error=merr,full_field_permeability_max_relative_error=kerr,
            independent_FVM_reassembly_max_relative_error=coeff_error,finite_positive=True)
    precision=high_precision(h,l,k,alpha,d['updated_gradient_magnitude'].squeeze(),d['updated_mobility'].squeeze(),selected)
    collapsed=h+l;crh,crl=evaluate(frozen,collapsed,np.zeros_like(l),b)
    low_ratio=float(np.max(np.abs(l)/np.spacing(np.abs(h))))
    assert low_ratio<=.500001 and np.count_nonzero(l)>0
    result=dict(path=str(path),field=row['field'],alpha=alpha,step=row['picard_step'],kind=row['kind'],
        independent_linear_residual=linear,independent_nonlinear_residual=nl,independent_pressure_change=change,
        independent_initial_split_residual=warm,
        collapsed_linear_residual=norm(crh+crl)/bn,low_nonzero=int(np.count_nonzero(l)),low_over_ULP_max=low_ratio,
        decimal_linear=decimal_linear,decimal_nonlinear=decimal_nonlinear,constitutive=physical,high_precision=precision,
        conservation_linear=conservation(frozen,h,l,b,rh,rl),conservation_nonlinear=conservation(nonlinear,h,l,b,nrh,nrl),
        result_sha256=sha(path/'result.json'),pressure_sha256=sha(path/'pressure.pt'),systems_sha256=sha(path/'stored_systems.npz'),validator_sha256=sha(__file__))
    dump(path/'independent_validation.json',result)
    print(json.dumps({k:result[k] for k in ['field','alpha','step','kind','independent_linear_residual','independent_nonlinear_residual']}),flush=True)
    return result
