"""Numerical routines used by the paper; no experiment orchestration."""
from __future__ import annotations
from decimal import Decimal

def decimal_quantities(high,low,kappa,i,j,alpha):
    n=high.shape[-1]
    D=lambda x:Decimal.from_float(float(x))
    value=lambda y,x:D(high[y,x])+D(low[y,x])
    hh=D(1/n)
    if n==1:gx=gy=Decimal(0)
    else:
        gx=value(i,0)/(hh/2) if j==0 else -value(i,n-1)/(hh/2) if j==n-1 else (value(i,j+1)-value(i,j-1))/(2*hh)
        gy=value(0,j)/(hh/2) if i==0 else -value(n-1,j)/(hh/2) if i==n-1 else (value(i+1,j)-value(i-1,j))/(2*hh)
    magnitude=(gx*gx+gy*gy+D(1e-300)).sqrt()
    effective=D(kappa[i,j])*(Decimal(alpha)*value(i,j)).exp()
    inverse=1/effective
    lam=2/(inverse+(inverse*inverse+4*D(1e-4)*magnitude).sqrt())
    return magnitude,lam
