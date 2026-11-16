"""Numerical routines used by the paper; no experiment orchestration."""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
from learned_tg.model.coarse_spaces.repeated_solve import build_threshold_geneo_raw_basis, ALL_DIRICHLET_BOUNDARY_TYPES
from learned_tg.solvers.picard import tensor_sha, write_json

def diagnostic_json(value):
    """Preserve infinite pencil eigenvalues as explicit JSON data."""
    if isinstance(value,dict):return {str(k):diagnostic_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [diagnostic_json(v) for v in value]
    if isinstance(value,np.ndarray):return diagnostic_json(value.tolist())
    if isinstance(value,np.generic):return diagnostic_json(value.item())
    if isinstance(value,float) and not math.isfinite(value):return {'nonfinite_float':str(value)}
    return value

class GenEOProvider:
    preparation={'coarse_basis_orth_metric':'direct_galerkin'}
    def __init__(self,method,field,cache,threshold=.30,max_local_modes=64):
        if method not in ['rebuild','frozen']:raise ValueError(method)
        self.method=method;self.field=field;self.cache=Path(cache)
        self.threshold=threshold;self.max_local_modes=max_local_modes;self.raw=None
        self.signature=dict(method=method,threshold=threshold,max_local_modes=max_local_modes,
            threshold_initial_modes=8,dense_cutoff=512,eig_tol=1e-9,eigen_residual_tol=1e-6,
            preparation='direct_galerkin',frozen_rule='initial_span_current_operator_Gram')
    def restore(self,op,history):
        """After a restart, rebuild the frozen span from the trajectory's first operator.

        The span exists only in memory, so it is recomputed and must reproduce the
        hash recorded at the first Picard step; otherwise the restart is refused.
        """
        if self.method!='frozen':return
        raw,_=build_threshold_geneo_raw_basis(op.asm,threshold=self.threshold,
            boundary_types=ALL_DIRICHLET_BOUNDARY_TYPES,sample_id=self.field*1000,
            max_local_modes=self.max_local_modes)
        if tensor_sha(raw)!=history[0]['raw_span_sha256']:
            raise RuntimeError('the frozen GenEO span of this interrupted trajectory could not be reproduced')
        self.raw=raw
    def __call__(self,op,step,mobility):
        reused=self.method=='frozen' and self.raw is not None
        diagnostics=None
        if not reused:
            raw,diagnostics=build_threshold_geneo_raw_basis(op.asm,threshold=self.threshold,
                boundary_types=ALL_DIRICHLET_BOUNDARY_TYPES,sample_id=self.field*1000+step-1,
                max_local_modes=self.max_local_modes)
            if self.method=='frozen':self.raw=raw
        else:raw=self.raw
        self.cache.mkdir(parents=True,exist_ok=True)
        if diagnostics is not None:write_json(self.cache/f'step_{step:02d}_spectrum.json',diagnostic_json(diagnostics))
        return raw,dict(method=self.method,span_reused=reused,span_sha256=tensor_sha(raw),
            geneo_signature=self.signature,threshold_complete=None if diagnostics is None else diagnostics['threshold_complete'],
            geneo_diagnostics=str(self.cache/f'step_{step:02d}_spectrum.json') if diagnostics is not None else None)
