"""Opt-in split-pressure representation and verified FGMRES state policy.

The correction solver is the existing right TG--FGMRES. All stored vectors
and scalar linear algebra use float64. Two-component pressure and residual
arithmetic are explicitly expanded representations, not a single float64 state.
No global direct correction is provided by this module.
"""
from dataclasses import dataclass
import math
import time

import torch
import torch.nn.functional as F

from learned_tg.model.krylov import Context, _fgmres_engine, norm, dot


class RepresentationError(ArithmeticError):
    pass


def finite64(x, name):
    if x.dtype != torch.float64:
        raise TypeError(f'{name} must be float64')
    if not bool(torch.isfinite(x).all()):
        raise RepresentationError(f'nonfinite {name}')


def two_sum(a, b):
    high = a + b
    virtual = high - a
    return high, (a - (high - virtual)) + (b - virtual)


def two_prod(a, b):
    """Dekker product, for finite normal-range inputs (checked by callers)."""
    product = a * b
    split = 134217729.0
    ca = split * a
    ah = ca - (ca - a)
    al = a - ah
    cb = split * b
    bh = cb - (cb - b)
    bl = b - bh
    error = ((ah * bh - product) + ah * bl + al * bh) + al * bl
    return product, error


def expansion_add(ah, al, bh, bl):
    s, e = two_sum(ah, bh)
    t, f = two_sum(al, bl)
    high, low = two_sum(s, e + t)
    return two_sum(high, low + f)


@dataclass
class SplitPressure:
    high: torch.Tensor
    low: torch.Tensor

    def __post_init__(self):
        finite64(self.high, 'pressure high')
        finite64(self.low, 'pressure low')
        if self.high.shape != self.low.shape or self.high.device != self.low.device:
            raise ValueError('pressure components must share shape and device')

    @classmethod
    def from_tensor(cls, pressure):
        finite64(pressure, 'initial pressure')
        return cls(pressure.clone(), torch.zeros_like(pressure))

    @classmethod
    def zeros_like(cls, vector):
        finite64(vector, 'shape template')
        return cls(torch.zeros_like(vector), torch.zeros_like(vector))

    def clone(self):
        return SplitPressure(self.high.clone(), self.low.clone())

    def add_expansion(self, high, low):
        finite64(high, 'correction high')
        finite64(low, 'correction low')
        if high.shape != self.high.shape or low.shape != self.low.shape:
            raise ValueError('correction shape mismatch')
        if high.device != self.high.device or low.device != self.low.device:
            raise ValueError('correction device mismatch')
        h, l = expansion_add(self.high, self.low, high, low)
        # Validate before replacing either component: failure cannot partially
        # mutate the pressure. Normalization is inherent in expansion_add.
        finite64(h, 'updated pressure high')
        finite64(l, 'updated pressure low')
        self.high, self.low = h, l
        return self

    def add(self, correction):
        return self.add_expansion(correction, torch.zeros_like(correction))

    def add_scaled(self, vector, scale):
        finite64(vector, 'basis vector')
        if not math.isfinite(float(scale)):
            raise RepresentationError('nonfinite basis coefficient')
        high, low = two_prod(vector, float(scale))
        return self.add_expansion(high, low)

    def renormalize(self):
        h, l = two_sum(self.high, self.low)
        finite64(h, 'renormalized pressure high')
        finite64(l, 'renormalized pressure low')
        self.high, self.low = h, l
        return self

    def norm(self):
        scale = max(float(self.high.abs().max()), float(self.low.abs().max()))
        if not scale:
            return 0.0
        h, l = self.high / scale, self.low / scale
        square = math.fsum([float(torch.sum(h*h)), 2*float(torch.sum(h*l)), float(torch.sum(l*l))])
        return scale * math.sqrt(max(square, 0.0))

    def export(self):
        return dict(high=self.high.detach().cpu().clone(), low=self.low.detach().cpu().clone())

    def collapsed_for_comparison(self):
        """Deliberately lossy; never used to update the split iterative state."""
        return self.high + self.low

    def __array__(self, *args, **kwargs):
        raise TypeError('export both pressure components; implicit collapse is forbidden')


class StoredFVMResidual:
    """Residual of the original stored stencil, using compensated products.

    Directly sums b - sum_j a_ij*p_high_j - sum_j a_ij*p_low_j in a
    two-component expansion. Stored diagonals/boundary entries/row defects
    are never reconstructed or replaced by a zero-row-sum assumption.
    Each call returns its residual rounded to float64, not collapsed pressure.
    """
    def __init__(self, asm):
        self.coefficients = tuple(getattr(asm, key) for key in ['sw', 'se', 'sn', 'ss', 'sc'])
        for coefficient in self.coefficients:
            finite64(coefficient, 'stored coefficient')
        self.evaluations = 0
        self.elapsed_seconds = 0.0

    @staticmethod
    def neighbours(x):
        pad = F.pad(x, (1, 1, 1, 1), value=0.)
        return (pad[..., 1:-1, :-2], pad[..., 1:-1, 2:],
                pad[..., :-2, 1:-1], pad[..., 2:, 1:-1], x)

    def residual(self, rhs, pressure):
        tic = time.perf_counter()
        finite64(rhs, 'residual RHS')
        if rhs.shape != pressure.high.shape or rhs.ndim != 4:
            raise ValueError('FVM residual expects matching (B,C,ny,nx) tensors')
        rh, rl = rhs.clone(), torch.zeros_like(rhs)
        for coefficient, high, low in zip(self.coefficients,
                self.neighbours(pressure.high), self.neighbours(pressure.low)):
            ph, pl = two_prod(coefficient, high)
            rh, rl = expansion_add(rh, rl, -ph, -pl)
            ph, pl = two_prod(coefficient, low)
            rh, rl = expansion_add(rh, rl, -ph, -pl)
        result = rh + rl
        finite64(result, 'compensated residual')
        self.evaluations += 1
        self.elapsed_seconds += time.perf_counter() - tic
        return result


@dataclass
class SplitKrylovResult:
    pressure: SplitPressure
    info: dict


class SplitContext(Context):
    """Expanded solution policy for the one shared FGMRES engine."""
    def __init__(self,A,B,b,accurate,initial,tol,safety,maxiter):
        if not 0<safety<=1:raise ValueError('invalid split acceptance safety')
        if initial is not None and not isinstance(initial,SplitPressure):
            raise TypeError('split warm starts must retain both components')
        super().__init__(A,B,b,None,tol*safety,maxiter)
        del self.x  # No single-vector pressure is maintained in this policy.
        self.pressure=SplitPressure.zeros_like(b) if initial is None else initial.clone()
        self.accurate=accurate;self.start_evaluations=accurate.evaluations
        self.requested_tol=tol
        self.info.update(algorithm='FGMRES_split_compensated',global_direct_corrections=0,
            outer_tol=tol,acceptance_goal=tol*safety,
            pressure_storage_bytes=2*b.numel()*b.element_size(),maximum_Arnoldi_relation_error=0.)

    def snapshot(self):return self.pressure.clone()
    def zero_vector(self):return torch.zeros_like(self.b)
    def start_update(self):return SplitPressure.zeros_like(self.b)
    def accumulate(self,delta,vector,weight):return delta.add_scaled(vector,float(weight))
    def commit(self,base,delta):
        self.pressure=base.clone().add_expansion(delta.high,delta.low)
    def residual(self):
        self.info['true_residual_evaluations']+=1
        return self.accurate.residual(self.b,self.pressure)
    def record_update(self,delta,r,row):
        check=self.accurate.residual(r,delta)
        row.update(true_cycle_correction_relative_residual=norm(check)/(norm(r) or 1.),
                   correction_norm=delta.norm())
    def update_image(self,delta):
        return -self.accurate.residual(torch.zeros_like(self.b),delta)
    def finish(self,r,status=None):
        true=self.residual();tr=norm(true)/self.bn;rr=norm(r)/self.bn
        ok=math.isfinite(tr) and tr<=self.tol
        self.info.update(converged=ok,status='converged' if ok else (status or self.info['status']),
            recursive_final_residual=rr,true_final_residual=tr,residual_gap=norm(true-r)/self.bn,
            estimated_final_residual=self.info['history'][-1] if self.info['history'] else rr,
            elapsed_seconds=time.perf_counter()-self.tic,
            accurate_residual_evaluations=self.accurate.evaluations-self.start_evaluations,
            correction_matrix_applications=self.info['matrix_vector_applications'],
            refinements=max(self.info['solution_updates']-1,0),
            pressure_low_norm=norm(self.pressure.low),pressure_norm=self.pressure.norm(),
            pressure_low_nonzero=int(torch.count_nonzero(self.pressure.low)),
            maximum_Arnoldi_orthogonality_defect=self.info['maximum_image_orthogonality_defect'])
        return SplitKrylovResult(self.pressure,self.info)


@torch.no_grad()
def fgmres_split(A,B,accurate,rhs,*,initial=None,tol=1e-6,safety=.5,
                 max_updates=300,restart=50,diagnostics=False,observer=None):
    """Explicit split state; shared Arnoldi, ordinary float64 Krylov vectors."""
    finite64(rhs,'split RHS')
    context=SplitContext(A,B,rhs,accurate,initial,tol,safety,max_updates)
    def observed(values):
        if diagnostics:
            reconstructed=torch.zeros_like(values['Az'])
            for coefficient,vector in zip(values['H_column'],values['V']):
                reconstructed=reconstructed+float(coefficient)*vector
            error=norm(values['Az']-reconstructed)/max(norm(values['Az']),1e-300)
            context.info['maximum_Arnoldi_relation_error']=max(context.info['maximum_Arnoldi_relation_error'],error)
        if observer is not None:observer(values)
    return _fgmres_engine(context,restart=restart,diagnostics=diagnostics,
        observer=observed if diagnostics or observer is not None else None)


