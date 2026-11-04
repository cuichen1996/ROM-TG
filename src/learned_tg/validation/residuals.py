"""Numerical routines used by the paper; no experiment orchestration."""
from __future__ import annotations
from decimal import Decimal, localcontext
import numpy as np

def two_sum(a,b):
    s=a+b;v=s-a;return s,(a-(s-v))+(b-v)

def two_prod(a,b):
    p=a*b;split=134217729.
    ca=split*a;ah=ca-(ca-a);al=a-ah
    cb=split*b;bh=cb-(cb-b);bl=b-bh
    e=((ah*bh-p)+ah*bl+al*bh)+al*bl
    return p,e

def add(ah,al,bh,bl):
    s,e=two_sum(ah,bh);t,f=two_sum(al,bl)
    h,l=two_sum(s,e+t)
    return two_sum(h,l+f)

def neighbours(x):
    p=np.pad(x,1)
    return [p[1:-1,:-2],p[1:-1,2:],p[:-2,1:-1],p[2:,1:-1],x]

def evaluate(coefficients,high,low,b):
    rh=b.copy();rl=np.zeros_like(b)
    nh=neighbours(high);nl=neighbours(low)
    for c,xh,xl in zip(coefficients,nh,nl):
        ph,pl=two_prod(c,xh);rh,rl=add(rh,rl,-ph,-pl)
        ph,pl=two_prod(c,xl);rh,rl=add(rh,rl,-ph,-pl)
    return rh,rl

def decimal_check(coefficients,hi,lo,b,rh,rl):
    n=hi.shape[-1];rng=np.random.default_rng(20261030)
    chosen=set(rng.choice(n*n,size=512,replace=False).tolist())
    chosen.update([0,n-1,n*(n-1),n*n-1,n*(n//2)+n//2])
    nh=neighbours(hi);nl=neighbours(lo);max_error=Decimal(0)
    with localcontext() as ctx:
        ctx.prec=100
        for index in sorted(chosen):
            i,j=divmod(index,n);reference=Decimal.from_float(float(b[i,j]))
            for c,h,l in zip(coefficients,nh,nl):
                reference-=Decimal.from_float(float(c[i,j]))*(Decimal.from_float(float(h[i,j]))+Decimal.from_float(float(l[i,j])))
            computed=Decimal.from_float(float(rh[i,j]))+Decimal.from_float(float(rl[i,j]))
            max_error=max(max_error,abs(reference-computed))
    assert max_error<Decimal('1e-22'),str(max_error)
    return dict(selected_rows=len(chosen),decimal_precision=100,maximum_absolute_difference=str(max_error))
