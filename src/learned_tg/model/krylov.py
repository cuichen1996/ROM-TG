"""Float64 right-preconditioned Krylov solvers with verified termination.

Every returned convergence flag is based on a final explicitly recomputed
true residual. ``updates`` counts Krylov steps, never residual evaluations.
No global solve, basis modification or higher precision is used.
"""
from dataclasses import dataclass
import math
import time

import torch


def norm(v):return float(torch.linalg.vector_norm(v).item())
def dot(v,w):return float(torch.sum(v*w).item())


@dataclass
class KrylovResult:
    x: torch.Tensor
    info: dict


class KrylovHistory(list):
    """List-compatible history carrying unambiguous iteration/status metadata."""
    def __init__(self,info):
        super().__init__(info['history'])
        if self:self[-1]=info['true_final_residual']
        self.solver_info=info
        self.num_updates=info['updates']


def solve_batched(A,B,b,x0=None,*,method='fgmres',tol=1e-6,maxiter=200,restart=50,diagnostics=False):
    """Independent RHS solves; batch>1 uses an explicit serial fallback.

    The operator/preconditioner must act independently on batch entries, as
    the FVM and prepared TG operators do. Embedding in a zero full batch keeps
    every entry's coefficient field/coarse basis correct. Primary experiments
    use batch=1 and incur no embedding or extra batched work.
    """
    if method not in ['fgmres','fgcr']:raise ValueError('method must be fgmres or fgcr')
    solve=fgmres if method=='fgmres' else repaired_gcr
    opts=dict(tol=tol,maxiter=maxiter,restart=restart,diagnostics=diagnostics)
    if method=='fgcr':opts['variant']='recompute'
    count=b.shape[0]
    if count==1:
        result=solve(A,B,b,x0,**opts)
        result.info.update(algorithm=method,batch_mode='single_rhs',batch_size=1)
        return result
    results=[]
    for i in range(count):
        def embed(fun,value):
            full=torch.zeros_like(b,dtype=torch.float64);full[i:i+1]=value
            return fun(full)[i:i+1]
        result=solve(lambda v:embed(A,v),lambda v:embed(B,v),b[i:i+1],None if x0 is None else x0[i:i+1],**opts)
        results.append(result)
    x=torch.cat([r.x for r in results],dim=0);steps=max(r.info['updates'] for r in results)
    hist=[max(r.info['history'][min(j,len(r.info['history'])-1)] for r in results) for j in range(steps+1)]
    info=dict(algorithm=method,batch_mode='independent_serial',batch_size=count,updates=steps,
        per_sample=[r.info for r in results],history=hist,
        converged=all(r.info['converged'] for r in results),
        status='converged' if all(r.info['converged'] for r in results) else 'batch_nonconvergence',
        true_final_residual=max(r.info['true_final_residual'] for r in results),
        recursive_final_residual=max(r.info['recursive_final_residual'] for r in results),
        residual_gap=max(r.info['residual_gap'] for r in results))
    for key in ['preconditioner_applications','matrix_vector_applications','true_residual_evaluations',
                'diagnostic_matvecs','restarts','false_convergence_recoveries','solution_updates']:
        info[key]=sum(r.info[key] for r in results)
    # Recheck the assembled batch too; a coupled batch callback cannot silently
    # pass by converging only its isolated diagonal blocks.
    residual=b.double()-A(x);info['matrix_vector_applications']+=1;info['true_residual_evaluations']+=1
    norms=torch.linalg.vector_norm(residual.reshape(count,-1),dim=1)
    den=torch.linalg.vector_norm(b.double().reshape(count,-1),dim=1);den=torch.where(den>0,den,torch.ones_like(den))
    rel=norms/den
    info['true_final_residual']=float(rel.max());info['converged']=bool(torch.isfinite(rel).all() and (rel<=tol).all())
    info['status']='converged' if info['converged'] else 'batch_nonconvergence'
    return KrylovResult(x,info)


class Context:
    def __init__(self,A,B,b,x0,tol,maxiter):
        if tol<0 or maxiter<0:raise ValueError('tol and maxiter must be nonnegative')
        self.Afun=A;self.Bfun=B;self.b=b.double();self.x=self.b.new_zeros(b.shape) if x0 is None else x0.clone().double()
        self.tol=float(tol);self.cap=int(maxiter);self.bn=norm(self.b) or 1.
        self.info=dict(updates=0,solution_updates=0,preconditioner_applications=0,matrix_vector_applications=0,
            true_residual_evaluations=0,diagnostic_matvecs=0,restarts=0,false_convergence_recoveries=0,
            near_breakdowns=0,stagnation_events=0,maximum_direction_image_defect=0.,
            maximum_image_orthogonality_defect=0.,history=[],trace=[],status='iteration_limit')
        self.tic=time.perf_counter()
    def A(self,v,diagnostic=False):
        self.info['matrix_vector_applications']+=1
        self.info['diagnostic_matvecs']+=int(diagnostic)
        y=self.Afun(v)
        if y.dtype!=torch.float64:raise TypeError('matvec must return float64')
        return y
    def B(self,v):
        self.info['preconditioner_applications']+=1
        y=self.Bfun(v)
        if y.dtype!=torch.float64:raise TypeError('preconditioner must return float64')
        return y
    def residual(self):
        self.info['true_residual_evaluations']+=1
        return self.b-self.A(self.x)
    def snapshot(self):return self.x.clone()
    def zero_vector(self):return torch.zeros_like(self.x)
    def start_update(self):return torch.zeros_like(self.x)
    def accumulate(self,delta,vector,weight):return delta+float(weight)*vector
    def commit(self,base,delta):self.x=base+delta
    def record_update(self,delta,r,row):pass
    def update_image(self,delta):return self.A(delta,diagnostic=True)
    def finish(self,r,status=None):
        true=self.residual();tr=norm(true)/self.bn;rr=norm(r)/self.bn
        ok=math.isfinite(tr) and tr<=self.tol
        self.info.update(converged=ok,status='converged' if ok else (status or self.info['status']),
            recursive_final_residual=rr,true_final_residual=tr,residual_gap=norm(true-r)/self.bn,
            estimated_final_residual=self.info['history'][-1] if self.info['history'] else rr,
            elapsed_seconds=time.perf_counter()-self.tic)
        return KrylovResult(self.x,self.info)


@torch.no_grad()
def repaired_gcr(A,B,b,x0=None,*,tol=1e-6,maxiter=200,restart=50,
                 variant='recompute',diagnostics=False):
    """Retained true-residual GCR path for the historical operator API.

    The released experiment entry points use FGMRES. This compatibility path
    preserves the recomputed-image GCR algorithm; exploratory repair variants
    are excluded from this package.
    """
    if variant != 'recompute':
        raise ValueError(variant)
    c=Context(A,B,b,x0,tol,maxiter);r=c.residual();P=[];Q=[]
    c.info['history']=[norm(r)/c.bn];no_progress=0;attempts=0
    if c.info['history'][-1]<=tol:return c.finish(r)
    cycle=int(restart) if restart is not None else max(1,maxiter)
    if cycle<0:raise ValueError('restart must be nonnegative or None')
    while c.info['updates']<c.cap and attempts<2*c.cap+10:
        attempts+=1;p=c.B(r)
        if not bool(torch.isfinite(p).all()):return c.finish(r,'nonfinite_preconditioner')
        q=c.A(p);rawqn=norm(q)
        for _ in range(2):
            for pk,qk in zip(P,Q):
                den=dot(qk,qk)
                if den>0:
                    beta=dot(q,qk)/den;p=p-beta*pk;q=q-beta*qk
        ap=c.A(p,diagnostic=diagnostics)
        eta=norm(q-ap)/max(norm(q)+norm(ap),1e-300)
        c.info['maximum_direction_image_defect']=max(c.info['maximum_direction_image_defect'],eta)
        c.info['maximum_pre_repair_direction_image_defect']=max(c.info.get('maximum_pre_repair_direction_image_defect',0.),eta)
        q=ap
        qn=norm(q)
        orth=max([abs(dot(q,qk))/(max(qn,1e-300)*max(norm(qk),1e-300)) for qk in Q],default=0.)
        c.info['maximum_image_orthogonality_defect']=max(c.info['maximum_image_orthogonality_defect'],orth)
        if not math.isfinite(qn):return c.finish(r,'nonfinite_image')
        if qn<=max(1e-150,64*torch.finfo(torch.float64).eps*rawqn):
            c.info['near_breakdowns']+=1
            true=c.residual()
            if norm(true)/c.bn<=tol:return c.finish(true)
            if not Q or no_progress>=2:return c.finish(r,'near_breakdown')
            r=true;P.clear();Q.clear();c.info['restarts']+=1;no_progress+=1;continue
        if diagnostics:
            actual_ap=c.A(p,diagnostic=True)
            actual_eta=norm(q-actual_ap)/max(norm(q)+norm(actual_ap),1e-300)
            c.info['maximum_actual_direction_image_defect']=max(c.info.get('maximum_actual_direction_image_defect',0.),actual_eta)
        alpha=dot(r,q)/(qn*qn)
        oldrn=norm(r)
        if not math.isfinite(alpha):return c.finish(r,'nonfinite_step')
        c.x=c.x+alpha*p;r=r-alpha*q
        c.info['updates']+=1;c.info['solution_updates']+=1
        rr=norm(r)/c.bn;c.info['history'].append(rr)
        step=dict(j=c.info['updates'],recursive_relative_residual=rr,alpha=alpha,
            direction_image_defect=eta if ap is not None else None,image_orthogonality=orth)
        c.info['trace'].append(step)
        if norm(r)>=oldrn*(1-32*torch.finfo(torch.float64).eps):
            no_progress+=1;c.info['stagnation_events']+=1
        else:no_progress=0
        if cycle!=0:P.append(p);Q.append(q)
        normal_restart=cycle>0 and len(Q)>=cycle
        candidate=rr<=tol
        if candidate or normal_restart:
            true=c.residual();tr=norm(true)/c.bn;gap=norm(true-r)/c.bn
            step.update(true_relative_residual=tr,residual_gap=gap)
            if tr<=tol:return c.finish(r)
            if candidate:c.info['false_convergence_recoveries']+=1
            if candidate or normal_restart or gap>.1*max(tr,tol):
                r=true;P.clear();Q.clear();c.info['restarts']+=1
        if no_progress>=3:return c.finish(r,'stagnation')
    return c.finish(r)


@torch.no_grad()
def fgmres(A,B,b,x0=None,*,tol=1e-6,maxiter=200,restart=50,diagnostics=False,observer=None):
    """Independent right FGMRES(m), retaining raw z_j and A z_j.

    Two-pass Arnoldi; Givens residual estimates; small CPU float64 SVD least
    squares at a candidate/cycle end. A failed true check restarts fully.
    """
    return _fgmres_engine(Context(A,B,b,x0,tol,maxiter),restart=restart,diagnostics=diagnostics,observer=observer)


def _fgmres_engine(c,*,restart=50,diagnostics=False,observer=None):
    """Shared Arnoldi/least-squares machinery; state policies own accumulation.

    The default Context preserves the original tensor arithmetic/order.
    Split-pressure contexts provide an explicit expanded state and residual.
    """
    tol=c.tol
    r=c.residual();rr=norm(r)/c.bn
    c.info['history']=[rr];badcycles=0
    if rr<=tol:return c.finish(r)
    m=max(1,c.cap) if restart is None else int(restart)
    if m<=0:raise ValueError('FGMRES restart must be positive or None')
    while c.info['updates']<c.cap:
        beta=norm(r);xbase=c.snapshot();cycle_initial=beta
        V=[r/beta];Z=[];AZ=[];H=torch.zeros((m+1,m),dtype=torch.float64)
        R=H.clone();g=torch.zeros(m+1,dtype=torch.float64);g[0]=beta
        cs=[];sn=[];endcycle=False
        for j in range(min(m,c.cap-c.info['updates'])):
            z=c.B(V[j])
            if not bool(torch.isfinite(z).all()):return c.finish(r,'nonfinite_preconditioner')
            w=c.A(z);raw=w.clone();rawnorm=norm(raw)
            if not bool(torch.isfinite(w).all()):return c.finish(r,'nonfinite_image')
            if diagnostics:
                independent=c.A(z,diagnostic=True)
                eta=norm(raw-independent)/max(norm(raw)+norm(independent),1e-300)
                c.info['maximum_direction_image_defect']=max(c.info['maximum_direction_image_defect'],eta)
            Z.append(z);AZ.append(raw)
            for _ in range(2):
                for i,v in enumerate(V):
                    hij=dot(v,w);H[i,j]+=hij;w=w-hij*v
            hn=norm(w);H[j+1,j]=hn
            near=hn<=64*torch.finfo(torch.float64).eps*max(rawnorm,1e-300)
            if not near:
                v=w/hn
                orth=max(abs(dot(v,vi)) for vi in V)
                c.info['maximum_image_orthogonality_defect']=max(c.info['maximum_image_orthogonality_defect'],orth)
                V.append(v)
            else:c.info['near_breakdowns']+=1
            if observer is not None:
                observer(dict(j=c.info['updates'],cycle=c.info['restarts'],column=j,
                    z=z,Az=raw,V=V,H_column=H[:j+2,j].clone(),near_breakdown=near))
            R[:j+2,j]=H[:j+2,j]
            for i in range(j):
                a=float(R[i,j]);bb=float(R[i+1,j]);R[i,j]=cs[i]*a+sn[i]*bb;R[i+1,j]=-sn[i]*a+cs[i]*bb
            aa=float(R[j,j]);bb=float(R[j+1,j]);rho=math.hypot(aa,bb)
            cc=aa/rho if rho else 1.;ss=bb/rho if rho else 0.;cs.append(cc);sn.append(ss)
            R[j,j]=rho;R[j+1,j]=0.;gg=float(g[j]);g[j]=cc*gg;g[j+1]=-ss*gg
            est=abs(float(g[j+1]))/c.bn
            c.info['updates']+=1;c.info['history'].append(est)
            row=dict(j=c.info['updates'],recursive_relative_residual=est,Arnoldi_h_next=hn,near_breakdown=near)
            c.info['trace'].append(row)
            if est<=tol or near or j+1==m or c.info['updates']==c.cap:
                rhs=torch.zeros(j+2,dtype=torch.float64);rhs[0]=beta
                y=torch.linalg.lstsq(H[:j+2,:j+1],rhs,driver='gelsd').solution
                dx=c.start_update()
                image=c.zero_vector()
                try:
                    for yi,zi,azi in zip(y,Z,AZ):dx=c.accumulate(dx,zi,yi);image=image+float(yi)*azi
                    c.commit(xbase,dx)
                except ArithmeticError:
                    return c.finish(r,'representation_breakdown')
                c.info['solution_updates']+=1
                r_est=r-image;true=c.residual();tr=norm(true)/c.bn
                row.update(true_relative_residual=tr,residual_gap=norm(true-r_est)/c.bn,
                    small_least_squares_relative_residual=float(torch.linalg.vector_norm(rhs-H[:j+2,:j+1]@y))/c.bn)
                c.record_update(dx,r,row)
                if diagnostics:
                    adx=c.update_image(dx)
                    c.info['maximum_direction_image_defect']=max(c.info['maximum_direction_image_defect'],norm(image-adx)/max(norm(image)+norm(adx),1e-300))
                if tr<=tol:return c.finish(r_est)
                if est<=tol:c.info['false_convergence_recoveries']+=1
                if c.info['updates']==c.cap:return c.finish(r_est)
                badcycles=badcycles+1 if norm(true)>=.999*cycle_initial else 0
                if badcycles>=3:
                    c.info['stagnation_events']+=1
                    # A finite total step cap already bounds ordinary residual
                    # replacement cycles. Do not abort a near-tolerance solve
                    # merely because a few floating point residuals plateau.
                    # Repeated actual Arnoldi breakdown still exits promptly.
                    if near:return c.finish(r_est,'near_breakdown_stagnation')
                r=true;c.info['restarts']+=1;endcycle=True;break
        if not endcycle:break
    return c.finish(r)
