"""
A-orthonormalization and Euclidean QR orthonormalization utilities.
Operator-agnostic — callers supply pre-computed AV.
"""

from learned_tg.model.losses import primitives as coarse_loss


def a_orthonormalize_from_AV(V, AV, reg=1e-10):
    """
    Given V and AV, return A-orthonormal Q such that Q^T A Q ≈ I.
    Uses eigsh decomposition for numerical robustness near rank deficiency.
    """
    return coarse_loss.a_orthonormalize_from_AV(V, AV, reg=reg)


def euclidean_orthonormalize_qr(V):
    """
    QR-based Euclidean orthonormalization. Returns Q with Q^T Q ≈ I.
    """
    return coarse_loss.euclidean_orthonormalize(V)
