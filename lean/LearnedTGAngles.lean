module

public import LearnedTGOptimality
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse

/-! Principal angles defined by the spectrum of the genuine compression Q P Q. -/

@[expose] public section

open scoped RealInnerProductSpace BigOperators

namespace LearnedTG

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] {n : ℕ}

noncomputable def principalGram (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection] : Q →L[ℝ] Q :=
  (Q.orthogonalProjectionOnto ∘L P.starProjection) ∘L Q.subtypeL

theorem principalGram_inner (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection] (x y : Q) :
    ⟪principalGram P Q x, y⟫ = ⟪P.starProjection (x : V), (y : V)⟫ := by
  change ⟪Q.starProjection (P.starProjection (x : V)), (y : V)⟫ = _
  rw [Q.inner_starProjection_left_eq_right, Q.starProjection_mem_subspace_eq_self]

theorem principalGram_quadratic (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection] (x : Q) :
    ⟪principalGram P Q x, x⟫ = ‖P.starProjection (x : V)‖ ^ 2 := by
  rw [principalGram_inner]
  have h := P.inner_starProjection_left_eq_right (P.starProjection (x : V)) (x : V)
  rw [P.starProjection_eq_self_iff.mpr (P.starProjection_apply_mem _)] at h
  exact h.trans (real_inner_self_eq_norm_sq _)

theorem principalGram_positive (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection] : (principalGram P Q).IsPositive := by
  apply (ContinuousLinearMap.isPositive_iff _).mpr
  constructor
  · intro x y
    calc
      _ = ⟪P.starProjection (x : V), (y : V)⟫ := principalGram_inner P Q x y
      _ = ⟪(x : V), P.starProjection (y : V)⟫ := P.inner_starProjection_left_eq_right _ _
      _ = ⟪x, principalGram P Q y⟫ :=
        (real_inner_comm (x : V) (P.starProjection (y : V))).symm.trans
          ((principalGram_inner P Q y x).symm.trans (real_inner_comm _ _))
  · intro x
    rw [principalGram_quadratic]
    exact sq_nonneg _

noncomputable def principalEigenvalues (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection]
    (q : OrthonormalBasis (Fin n) ℝ Q) : Fin n → ℝ := by
  let _ : FiniteDimensional ℝ Q := Module.Finite.of_basis q.toBasis
  exact (principalGram_positive P Q).isSymmetric.eigenvalues
    ((Module.finrank_eq_card_basis q.toBasis).trans (Fintype.card_fin n))

noncomputable def principalAngles (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection]
    (q : OrthonormalBasis (Fin n) ℝ Q) (i : Fin n) : ℝ :=
  Real.arccos (Real.sqrt (principalEigenvalues P Q q i))

theorem principal_spectral_data (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection]
    (q : OrthonormalBasis (Fin n) ℝ Q) :
    (∀ i, 0 ≤ principalEigenvalues P Q q i ∧ principalEigenvalues P Q q i ≤ 1) ∧
      (∑ i, principalEigenvalues P Q q i) = ∑ j, ‖P.starProjection (q j : V)‖ ^ 2 := by
  let _ : FiniteDimensional ℝ Q := Module.Finite.of_basis q.toBasis
  have hdim : Module.finrank ℝ Q = n :=
    (Module.finrank_eq_card_basis q.toBasis).trans (Fintype.card_fin n)
  let hT := principalGram_positive P Q
  let b := hT.isSymmetric.eigenvectorBasis hdim
  let lambda := hT.isSymmetric.eigenvalues hdim
  have heigen (i : Fin n) : principalGram P Q (b i) = lambda i • b i :=
    hT.isSymmetric.apply_eigenvectorBasis hdim i
  have hvalues (i : Fin n) : principalEigenvalues P Q q i = lambda i := rfl
  constructor
  · intro i
    rw [hvalues]
    refine ⟨hT.toLinearMap.nonneg_eigenvalues hdim i, ?_⟩
    have hquad := principalGram_quadratic P Q (b i)
    rw [heigen, real_inner_smul_left, real_inner_self_eq_norm_sq,
      b.orthonormal.norm_eq_one i, one_pow, mul_one] at hquad
    have hnorm := P.norm_starProjection_apply_le (b i : V)
    have hb : ‖(b i : V)‖ = 1 := b.orthonormal.norm_eq_one i
    rw [hb] at hnorm
    have hsq := (sq_le_sq₀ (norm_nonneg _) zero_le_one).mpr hnorm
    nlinarith
  · have hsum : (∑ j, ⟪principalGram P Q (q j), q j⟫) = ∑ i, lambda i := by
      simp_rw [finite_quadratic_diagonal (principalGram P Q) b lambda heigen]
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro i _
      rw [← Finset.mul_sum]
      have hnorm : (∑ j, (b.repr (q j) i) ^ 2) = 1 := by
        simp_rw [b.repr_apply_apply]
        rw [q.sum_sq_inner_left, b.orthonormal.norm_eq_one i, one_pow]
      rw [hnorm, mul_one]
    simp_rw [principalGram_quadratic] at hsum
    exact (Finset.sum_congr rfl fun i _ => hvalues i).trans hsum.symm

/-- The principal-angle identity uses spectral principal angles, rather than defining angles
from the loss itself. It holds for any learned-space rank and reference rank n. -/
theorem whiteLoss_eq_mean_sin_sq (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection]
    (q : OrthonormalBasis (Fin n) ℝ Q) :
    whiteLoss P Q q = (∑ i, Real.sin (principalAngles P Q q i) ^ 2) / n := by
  obtain ⟨hvalues, hsum⟩ := principal_spectral_data P Q q
  have hpoint (i : Fin n) : Real.sin (principalAngles P Q q i) ^ 2 =
      1 - principalEigenvalues P Q q i := by
    unfold principalAngles
    rw [Real.sin_arccos, Real.sq_sqrt (hvalues i).1,
      Real.sq_sqrt (sub_nonneg.mpr (hvalues i).2)]
  unfold whiteLoss
  congr 1
  rw [residualSum_eq_rank_sub]
  simp_rw [hpoint]
  rw [Finset.sum_sub_distrib, hsum]
  simp

end LearnedTG
