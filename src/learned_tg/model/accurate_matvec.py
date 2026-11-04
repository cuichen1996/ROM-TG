"""Compensated binary64 evaluation of the SAME stored five-point matrix.

This is an outer-matvec/residual evaluator. It never replaces the stencil
inside ASM/TG and never modifies an operator coefficient or coarse space.
All tensors and arithmetic remain float64; no float128 solve is performed.
"""
import torch
import torch.nn.functional as F


def two_sum(a,b):
    """Error-free (high,low) representation of a+b, absent overflow."""
    high=a+b;virtual=high-a
    low=(a-(high-virtual))+(b-virtual)
    return high,low


def compensated_sum(terms):
    total=torch.zeros_like(terms[0]);correction=torch.zeros_like(total)
    for term in terms:
        total,error=two_sum(total,term);correction=correction+error
    return total+correction


class AccurateFVMAction:
    """Apply a CCFVM matrix via differences plus its exact row-sum expansion.

    A x = sum_off c_ij (x_j-x_i) + (c_ii+sum_off c_ij) x_i.
    Retaining all TwoSum errors for the row sum preserves the stored matrix,
    including interior floating point row-sum defects and boundary anchors.
    Assuming a zero row sum would change the matrix and is not done here.
    """
    def __init__(self,asm):
        self.coefficients=[getattr(asm,k) for k in ['sw','se','sn','ss']]
        if asm.sc.dtype!=torch.float64:raise TypeError('stored FVM matrix must be float64')
        row=asm.sc;errors=[]
        for coefficient in self.coefficients:
            row,error=two_sum(row,coefficient);errors.append(error)
        self.row_high=row;self.row_errors=errors

    def __call__(self,x):
        if x.dtype!=torch.float64 or x.ndim!=4:raise TypeError('expected a float64 (B,C,ny,nx) tensor')
        pad=F.pad(x,(1,1,1,1),value=0.)
        neighbours=[pad[...,1:-1,:-2],pad[...,1:-1,2:],pad[...,:-2,1:-1],pad[...,2:,1:-1]]
        terms=[coefficient*(neighbour-x) for coefficient,neighbour in zip(self.coefficients,neighbours)]
        terms += [self.row_high*x]+[error*x for error in self.row_errors]
        return compensated_sum(terms)
