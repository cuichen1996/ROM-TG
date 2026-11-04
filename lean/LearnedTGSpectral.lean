module

public import LearnedTGSubspaces
public import Mathlib.Analysis.InnerProductSpace.Spectrum
public import Mathlib.Analysis.InnerProductSpace.l2Space

/-! Spectral-tail estimates derived from an actual covariance eigensystem. -/

@[expose] public section

open scoped RealInnerProductSpace

namespace LearnedTG

variable {V Z I : Type*}
  [NormedAddCommGroup V] [InnerProductSpace ℝ V] [CompleteSpace V]
  [NormedAddCommGroup Z] [InnerProductSpace ℝ Z] [CompleteSpace Z]

noncomputable def covariance (K : Z →L[ℝ] V) : V →L[ℝ] V := K ∘L K.adjoint

theorem covariance_symmetric (K : Z →L[ℝ] V) : (covariance K).IsSymmetric := by
  intro x y
  exact (K.adjoint_inner_right (K.adjoint x) y).symm.trans
    (K.adjoint_inner_left (K.adjoint y) x)

theorem covariance_quadratic (K : Z →L[ℝ] V) (x : V) :
    ⟪covariance K x, x⟫ = ‖K.adjoint x‖ ^ 2 := by
  exact (K.adjoint_inner_right (K.adjoint x) x).symm.trans (real_inner_self_eq_norm_sq _)

omit [CompleteSpace V] [NormedAddCommGroup Z] [InnerProductSpace ℝ Z] [CompleteSpace Z] in
theorem spectral_quadratic_tail (A : V →L[ℝ] V) (hA : A.IsSymmetric)
    (b : HilbertBasis I ℝ V) (lambda : I → ℝ) (s : Finset I) (sigma : ℝ)
    (heigen : ∀ i, A (b i) = lambda i • b i)
    (htail : ∀ i, i ∉ s → lambda i ≤ sigma ^ 2)
    (x : V) (hzero : ∀ i ∈ s, ⟪b i, x⟫ = 0) :
    ⟪A x, x⟫ ≤ sigma ^ 2 * ‖x‖ ^ 2 := by
  have hterm (i : I) : ⟪A x, b i⟫ * ⟪b i, x⟫ = lambda i * ⟪b i, x⟫ ^ 2 := by
    have hs : ⟪A x, b i⟫ = ⟪x, A (b i)⟫ := hA x (b i)
    rw [hs, heigen i, inner_smul_right, real_inner_comm]
    ring
  have hleft : HasSum (fun i => lambda i * ⟪b i, x⟫ ^ 2) ⟪A x, x⟫ := by
    simpa only [hterm] using b.hasSum_inner_mul_inner (A x) x
  have hparseval : HasSum (fun i => ⟪b i, x⟫ ^ 2) (‖x‖ ^ 2) := by
    simpa only [real_inner_comm, ← pow_two, real_inner_self_eq_norm_sq] using
      b.hasSum_inner_mul_inner x x
  have hright := hparseval.mul_left (sigma ^ 2)
  calc
    _ = ∑' i, lambda i * ⟪b i, x⟫ ^ 2 := hleft.tsum_eq.symm
    _ ≤ ∑' i, sigma ^ 2 * ⟪b i, x⟫ ^ 2 := by
      apply hleft.summable.tsum_le_tsum _ hright.summable
      intro i
      by_cases hi : i ∈ s
      · rw [hzero i hi]
        simp
      · exact mul_le_mul_of_nonneg_right (htail i hi) (sq_nonneg _)
    _ = _ := hright.tsum_eq

/-- No spectral-tail residual bound is assumed: it follows from diagonal covariance data. -/
theorem snapshot_tail_from_eigenbasis (K : Z →L[ℝ] V)
    (b : HilbertBasis I ℝ V) (lambda : I → ℝ) (s : Finset I)
    (Q : Submodule ℝ V) [Q.HasOrthogonalProjection]
    (sigma : ℝ) (hsigma : 0 ≤ sigma)
    (heigen : ∀ i, covariance K (b i) = lambda i • b i)
    (hretained : ∀ i ∈ s, b i ∈ Q)
    (htail : ∀ i, i ∉ s → lambda i ≤ sigma ^ 2) :
    ‖(1 - Q.starProjection) ∘L K‖ ≤ sigma := by
  let T := (1 - Q.starProjection) ∘L K
  have hQadj : Q.starProjection.adjoint = Q.starProjection :=
    (isSelfAdjoint_starProjection Q).adjoint_eq
  have hTadj : T.adjoint = K.adjoint ∘L (1 - Q.starProjection) := by
    rw [ContinuousLinearMap.adjoint_comp, map_sub, ContinuousLinearMap.adjoint_one, hQadj]
  have hbound : ‖T.adjoint‖ ≤ sigma := by
    apply ContinuousLinearMap.opNorm_le_bound _ hsigma
    intro v
    let w := (1 - Q.starProjection) v
    have hzero (i : I) (hi : i ∈ s) : ⟪b i, w⟫ = 0 := by
      rw [real_inner_comm]
      exact Q.starProjection_inner_eq_zero v (b i) (hretained i hi)
    have hquad := spectral_quadratic_tail (covariance K) (covariance_symmetric K)
      b lambda s sigma heigen htail w hzero
    rw [covariance_quadratic] at hquad
    have hw : ‖w‖ ^ 2 ≤ ‖v‖ ^ 2 := (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mpr
      (complement_projection_apply_le Q v)
    rw [hTadj]
    change ‖K.adjoint w‖ ≤ sigma * ‖v‖
    apply (sq_le_sq₀ (norm_nonneg _) (mul_nonneg hsigma (norm_nonneg _))).mp
    rw [mul_pow]
    exact hquad.trans (mul_le_mul_of_nonneg_left hw (sq_nonneg _))
  exact (ContinuousLinearMap.adjoint.norm_map T).symm.le.trans hbound

/-- The spectral complement Rayleigh bound implies the snapshot tail bound by adjunction. -/
theorem snapshot_tail_from_rayleigh (K : Z →L[ℝ] V)
    (Q : Submodule ℝ V) [Q.HasOrthogonalProjection] (b : ℝ) (hb : 0 ≤ b)
    (htail : ∀ x ∈ Qᗮ, ⟪covariance K x, x⟫ ≤ b * ‖x‖ ^ 2) :
    ‖(1 - Q.starProjection) ∘L K‖ ≤ Real.sqrt b := by
  let T := (1 - Q.starProjection) ∘L K
  have hTadj : T.adjoint = K.adjoint ∘L (1 - Q.starProjection) := by
    rw [ContinuousLinearMap.adjoint_comp, map_sub, ContinuousLinearMap.adjoint_one,
      (isSelfAdjoint_starProjection Q).adjoint_eq]
  have hbound : ‖T.adjoint‖ ≤ Real.sqrt b := by
    apply ContinuousLinearMap.opNorm_le_bound _ (Real.sqrt_nonneg _)
    intro v
    let w := (1 - Q.starProjection) v
    have hw : w ∈ Qᗮ := by
      dsimp [w]
      rw [← Q.starProjection_orthogonal']
      exact Qᗮ.starProjection_apply_mem v
    have hquad := htail w hw
    rw [covariance_quadratic] at hquad
    have hnorm : ‖w‖ ^ 2 ≤ ‖v‖ ^ 2 :=
      (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mpr (complement_projection_apply_le Q v)
    rw [hTadj]
    change ‖K.adjoint w‖ ≤ Real.sqrt b * ‖v‖
    apply (sq_le_sq₀ (norm_nonneg _) (by positivity : 0 ≤ Real.sqrt b * ‖v‖)).mp
    rw [mul_pow, Real.sq_sqrt hb]
    exact hquad.trans (mul_le_mul_of_nonneg_left hnorm hb)
  exact (ContinuousLinearMap.adjoint.norm_map T).symm.le.trans hbound

end LearnedTG
