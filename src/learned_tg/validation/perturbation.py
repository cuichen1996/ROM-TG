"""Numerical routines used by the paper; no experiment orchestration."""
from __future__ import annotations
import gc
import json
import math
from pathlib import Path
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import torch
from learned_tg.validation.sparse import assemble_fvm_matrix
from learned_tg.problems.static_helpers import _generate_smoothed_snapshots, _make_interface_operator
from learned_tg.coarse.pod import _apply_A_interface
from learned_tg.coarse.pod import build_oracle_basis, _make_prepared_coarse_basis
from learned_tg.coarse.spectral import _prepare_raw_basis_staged
from learned_tg.coarse.baselines import _predict_learned_basis
from learned_tg.model.operators.ccfvm_asm import get_fixed_subdomains_params

def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")

def psqrt(G):
    v,U=la.eigh((G+G.T)/2)
    assert v[0]>-1e-8*max(v[-1],1), (v[0],v[-1])
    return (U*np.sqrt(np.maximum(v,0)))@U.T

def lowrank_norm(Fgram,C,Ggram):
    return float(la.svdvals(psqrt(Fgram)@C@psqrt(Ggram))[0])

class SparseASM:
    def __init__(self,A,n):
        self.A,self.n=A,n
        s,d,_=get_fixed_subdomains_params(n,10,.3)
        idx=np.arange(n*n).reshape(n,n)
        self.W=np.zeros(n*n); self.locals=[]
        for i in range(10):
            for j in range(10):
                ids=idx[i*d:min(i*d+s,n),j*d:min(j*d+s,n)].ravel()
                if not ids.size: continue
                Ak=A[ids][:,ids]+1e-6*sp.eye(ids.size,format="csc")
                self.locals.append((ids,sla.splu(Ak.tocsc(),permc_spec="MMD_AT_PLUS_A",diag_pivot_thresh=0,options={"SymmetricMode":True})))
                self.W[ids]+=1

    def sum(self,r,transpose=False):
        z=np.zeros_like(r)
        for ids,lu in self.locals:
            z[ids]+=lu.solve(r[ids],trans="T" if transpose else "N")
        return z

    def D(self,x):
        z=self.sum(x)
        return .8*z/(self.W if x.ndim==1 else self.W[:,None])

    def S(self,x): return x-self.D(self.A@x)

    def ST(self,x):
        return x-.8*(self.A@self.sum(x/(self.W if x.ndim==1 else self.W[:,None]),True))

    def B(self,r,Q,C):
        x=self.D(r)
        x=x+Q@(C@(Q.T@(r-self.A@x)))
        return x+self.D(r-self.A@x)

def polynomial_diagnostic(op,Q,C,b,delta,M):
    # The production recurrence in exact-arithmetic form; retain polynomial
    # coefficients only to assess the theorem's pessimistic coefficient bound.
    r=b.copy(); r0=la.norm(r); rp=np.array([1.])
    ps=[]; qs=[]; polys=[]; rows=[]
    T=lambda x:op.A@op.B(x,Q,C)
    for it in range(1,46):
        z=op.B(r,Q,C); q=op.A@z
        qp=np.r_[0.,rp]
        for _ in range(2):
            for pk,qk,ck in zip(ps,qs,polys):
                beta=q@qk/(qk@qk)
                z-=beta*pk; q-=beta*qk
                qp[:len(ck)]-=beta*ck
        if q@q<1e-28: break
        alpha=(r@q)/(q@q)
        r-=alpha*q
        rp=np.pad(rp,(0,len(qp)-len(rp)))-alpha*qp
        ps.append(z); qs.append(q); polys.append(qp)
        ratio=la.norm(r)/r0
        if it in [5,10,15,20] or ratio<1e-6:
            coeff_bound=sum(abs(float(rp[k]))*k*M**(k-1) for k in range(1,len(rp)))
            rows.append(dict(m=it,reference_relative_residual=ratio,
                coefficient_sensitivity_bound=coeff_bound,
                dual_energy_additive_bound=delta*coeff_bound,
                monomial_coefficient_l1=float(sum(abs(rp)))))
        if ratio<1e-6: break
    return rows

def diagnose(n,sample,args,model,stored,out):
    tic=time.perf_counter()
    key=n,int(sample["sample_id"])
    records=stored[key]
    assert all(r["sample_field_sha256"]==sample["sample_field_sha256"] for r in records)
    op=_make_interface_operator(sample,args)
    E,_=_generate_smoothed_snapshots(problem="interface",operator=op,n=n,sample_seed=int(sample["seed"]),args=args)
    ref,refA,_=build_oracle_basis(E,lambda X:_apply_A_interface(op.asm,X),basis_type="a_pod",rank=15,reg=1e-12)
    pref=_make_prepared_coarse_basis(ref,refA,reg=1e-10,coarse_basis_mode="a_orthonormal")
    raw,_=_predict_learned_basis(model,sample,args)
    plearn,_=_prepare_raw_basis_staged(raw,op,args)
    A=assemble_fvm_matrix(op.asm).tocsr()
    sr=SparseASM(A,n)
    lu=sla.splu(A.tocsc(),permc_spec="MMD_AT_PLUS_A",diag_pivot_thresh=0,options={"SymmetricMode":True})
    Vr=pref["Q"][0].detach().cpu().numpy(); Vt=plearn["Q"][0].detach().cpu().numpy()
    Gr=Vr.T@(A@Vr); Gt=Vt.T@(A@Vt)
    Qr=Vr@la.inv(la.cholesky(Gr,lower=True).T)
    Qt=Vt@la.inv(la.cholesky(Gt,lower=True).T)
    cos=la.svdvals(Qt.T@(A@Qr))
    sinsq=np.maximum(0,1-np.minimum(cos,1)**2)
    delta=float(np.sqrt(sinsq.max())); loss=float(sinsq.mean())
    W=np.column_stack([Vt,Vr]); AW=A@W
    RTAW=sr.ST(AW); inv=lu.solve(RTAW)
    Gright=RTAW.T@inv; LW=sr.S(W)
    Gleft=LW.T@(A@LW); Gbase=W.T@AW
    C=la.block_diag(la.inv(Gt),-la.inv(Gr))
    Creg=la.block_diag(la.inv(Gt+1e-10*np.eye(15)),-la.inv(Gr+1e-10*np.eye(15)))
    exact=lowrank_norm(Gleft,C,Gright)
    implemented=lowrank_norm(Gleft,Creg,Gright)
    online=lowrank_norm(Gbase,C,Gright)
    gg=sla.LinearOperator(A.shape,matvec=lambda x:sr.ST(A@sr.S(x)),dtype=float)
    ai=sla.LinearOperator(A.shape,matvec=lu.solve,dtype=float)
    vals,X=sla.eigsh(gg,k=1,M=A,Minv=ai,which="LA",tol=1e-8,v0=np.random.default_rng(6030).normal(size=n*n))
    normS=float(np.sqrt(vals[0])); bound=normS**2*delta
    assert exact<=bound*(1+1e-6)
    assert exact<=normS*online*(1+1e-6)
    rng=np.random.default_rng(2030); x=rng.normal(size=n*n)
    tx=torch.from_numpy(x).to(args.device).reshape(1,1,n,n)
    tz=op.asm.apply_asm(tx).cpu().numpy().ravel()
    torch_err=float(la.norm(.8*tz-sr.D(x))/la.norm(.8*tz))
    residual=sr.ST(A@sr.S(X[:,0]))-vals[0]*(A@X[:,0])
    norm_res=float(np.sqrt(max(0.,residual@lu.solve(residual)))/vals[0])
    # Independent matrix-free norm check on the smallest grid.
    check=None
    if n==63:
        def Pdiff(y): return W@(C@(W.T@(A@y)))
        def action(y): return sr.S(Pdiff(sr.S(y)))
        def adj(y): return sr.ST(A@W@(C@(W.T@sr.ST(y))))
        mm=sla.LinearOperator(A.shape,matvec=lambda y:adj(A@action(y)),dtype=float)
        ev=sla.eigsh(mm,k=1,M=A,Minv=ai,which="LA",tol=1e-9,return_eigenvectors=False)
        check=float(np.sqrt(ev[0]))
        assert abs(check-exact)<1e-6*max(1.,exact)
    count={r["method"]:int(r["iterations"]) for r in records}
    tau=1e-10/(min(la.eigvalsh(Gt)[0],la.eigvalsh(Gr)[0])+1e-10)
    data=dict(n=n,sample_id=key[1],sample_seed=int(sample["seed"]),field_sha256=sample["sample_field_sha256"],
        equal_rank=15,target_grid_whitened_loss=loss,delta=delta,Delta=exact,
        implemented_regularized_Delta=implemented,online_defect=online,
        product_bound=bound,online_bound=normS*online,pessimism_ratio=bound/exact,
        smoother_norm=normS,smoother_norm_energy_residual=norm_res,
        coarse_regularization_gap_bound=2*normS**2*tau,
        scipy_torch_Minv_relative=torch_err,independent_norm_check=check,
        Gt_eigenvalues=la.eigvalsh(Gt).tolist(),Gr_eigenvalues=la.eigvalsh(Gr).tolist(),
        stored_counts=count,reference_scope="paired within grid; geometry seed depends on N",
        polynomial=polynomial_diagnostic(sr,Vr,la.inv(Gr+1e-10*np.eye(15)),sample["b"].cpu().numpy().ravel(),implemented,1+normS**2),
        sqrt_kappa_conversion_upper_bound=100/math.sin(math.pi/(2*n)),
        elapsed_seconds=time.perf_counter()-tic)
    out.mkdir(parents=True,exist_ok=False)
    dump(out/"result.json",data)
    np.savez(out/"lowrank_grams.npz",Gleft=Gleft,Gright=Gright,Gbase=Gbase,C=C,Creg=Creg)
    print(json.dumps(data),flush=True)
    del op,E,ref,refA,raw,pref,plearn,tx,tz
    gc.collect(); torch.cuda.empty_cache()
    return data
