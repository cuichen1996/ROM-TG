module

public import LearnedTGSubspaces

/-! Raw weighted loss and the one-sided inclusion defect needed at reduced reference rank. -/

@[expose] public section

open scoped BigOperators

namespace LearnedTG

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] {n : ℕ}

noncomputable def weightedResidual (P Q : Submodule ℝ V) [P.HasOrthogonalProjection]
    (q : OrthonormalBasis (Fin n) ℝ Q) (weight : Fin n → ℝ) : ℝ :=
  ∑ i, weight i * ‖(1 - P.starProjection) (q i : V)‖ ^ 2

noncomputable def rawLoss (P Q : Submodule ℝ V) [P.HasOrthogonalProjection]
    (q : OrthonormalBasis (Fin n) ℝ Q) (weight : Fin n → ℝ) : ℝ :=
  weightedResidual P Q q weight / ∑ i, weight i

theorem weightedResidual_lower (P Q : Submodule ℝ V) [P.HasOrthogonalProjection]
    (q : OrthonormalBasis (Fin n) ℝ Q) (weight : Fin n → ℝ) (w : ℝ)
    (hmin : ∀ i, w ≤ weight i) :
    w * residualSum P Q q ≤ weightedResidual P Q q weight := by
  unfold residualSum weightedResidual
  rw [Finset.mul_sum]
  exact Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_right (hmin i) (sq_nonneg _)

theorem rawLoss_controls_projection_distance (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection]
    (p : OrthonormalBasis (Fin n) ℝ P) (q : OrthonormalBasis (Fin n) ℝ Q)
    (weight : Fin n → ℝ) (w : ℝ) (hn : 0 < n) (hw : 0 < w)
    (hmin : ∀ i, w ≤ weight i) :
    ‖P.starProjection - Q.starProjection‖ ^ 2 ≤
      ((∑ i, weight i) / w) * rawLoss P Q q weight := by
  have hsum : 0 < ∑ i, weight i := by
    apply Finset.sum_pos
    · intro i _
      exact hw.trans_le (hmin i)
    · exact Finset.univ_nonempty_iff.mpr ⟨⟨0, hn⟩⟩
  have hdelta := projection_difference_le_sqrt_residualSum P Q p q
  have hsquare := (sq_le_sq₀ (norm_nonneg _) (Real.sqrt_nonneg _)).mpr hdelta
  rw [Real.sq_sqrt (residualSum_nonneg P Q q)] at hsquare
  have hweight := weightedResidual_lower P Q q weight w hmin
  have hres : residualSum P Q q ≤ weightedResidual P Q q weight / w :=
    (le_div_iff₀ hw).mpr (by simpa only [mul_comm] using hweight)
  calc
    _ ≤ weightedResidual P Q q weight / w := hsquare.trans hres
    _ = _ := by unfold rawLoss; field_simp

/-- This one-sided defect is meaningful when the learned and reference ranks differ. -/
noncomputable def inclusionDefect (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection] : ℝ :=
  ‖(1 - P.starProjection) * Q.starProjection‖

theorem inclusionDefect_le_sqrt_residualSum (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection]
    (q : OrthonormalBasis (Fin n) ℝ Q) :
    inclusionDefect P Q ≤ Real.sqrt (residualSum P Q q) := by
  apply ContinuousLinearMap.opNorm_le_of_unit_norm (Real.sqrt_nonneg _)
  intro x hx
  have h := apply_sq_le_basis_sum Q q (1 - P.starProjection) (Q.orthogonalProjectionOnto x)
  have hnorm : ‖Q.starProjection x‖ ≤ 1 := by simpa only [hx] using Q.norm_starProjection_apply_le x
  have hnormsq : ‖Q.starProjection x‖ ^ 2 ≤ 1 := by
    simpa only [one_pow] using (sq_le_sq₀ (norm_nonneg _) zero_le_one).mpr hnorm
  apply Real.le_sqrt_of_sq_le
  change ‖(1 - P.starProjection) (Q.starProjection x)‖ ^ 2 ≤ _
  exact h.trans ((mul_le_mul_of_nonneg_left hnormsq (residualSum_nonneg P Q q)).trans_eq (by simp))

theorem whiteLoss_le_inclusionDefect_sq (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection]
    (q : OrthonormalBasis (Fin n) ℝ Q) (hn : 0 < n) :
    whiteLoss P Q q ≤ inclusionDefect P Q ^ 2 := by
  apply (div_le_iff₀ (Nat.cast_pos.mpr hn)).mpr
  unfold residualSum
  calc
    _ ≤ ∑ _i : Fin n, inclusionDefect P Q ^ 2 := by
      apply Finset.sum_le_sum
      intro i _
      have hq : Q.starProjection (q i : V) = (q i : V) := Q.starProjection_mem_subspace_eq_self (q i)
      have h := ((1 - P.starProjection) * Q.starProjection).le_opNorm (q i : V)
      change ‖(1 - P.starProjection) (Q.starProjection (q i : V))‖ ≤ _ at h
      have hnorm : ‖(q i : V)‖ = 1 := q.orthonormal.norm_eq_one i
      rw [hq, hnorm, mul_one] at h
      exact (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mpr h
    _ = _ := by simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]; ring


theorem inclusionDefect_le_one (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection] : inclusionDefect P Q ≤ 1 := by
  exact (norm_mul_le _ _).trans ((mul_le_mul (complement_projection_norm_le P)
    Q.starProjection_norm_le (norm_nonneg _) zero_le_one).trans_eq (by simp))

/-- Exact one-sided reference capture; the second factor is ‖Q R‖, not
the coarser ‖R‖ used in the deleted continuous coverage theorem. -/
theorem one_sided_reference_capture (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection] (R : V →L[ℝ] V) :
    ‖(1 - P.starProjection) * R‖ ≤ ‖(1 - Q.starProjection) * R‖ +
      inclusionDefect P Q * ‖Q.starProjection * R‖ := by
  have hQ : Q.starProjection * Q.starProjection = Q.starProjection := by
    ext v
    exact Q.starProjection_mem_subspace_eq_self (Q.orthogonalProjectionOnto v)
  have hdecomp : (1 - P.starProjection) * R =
      (1 - P.starProjection) * ((1 - Q.starProjection) * R) +
      ((1 - P.starProjection) * Q.starProjection) * (Q.starProjection * R) := by
    have hsecond : ((1 - P.starProjection) * Q.starProjection) * (Q.starProjection * R) =
        ((1 - P.starProjection) * Q.starProjection) * R := by
      simp only [mul_assoc, ← mul_assoc Q.starProjection Q.starProjection R, hQ]
    rw [hsecond]
    noncomm_ring
  rw [hdecomp]
  calc
    _ ≤ ‖(1 - P.starProjection) * ((1 - Q.starProjection) * R)‖ +
        ‖((1 - P.starProjection) * Q.starProjection) * (Q.starProjection * R)‖ := norm_add_le _ _
    _ ≤ 1 * ‖(1 - Q.starProjection) * R‖ + inclusionDefect P Q * ‖Q.starProjection * R‖ :=
      add_le_add ((norm_mul_le _ _).trans
        (mul_le_mul_of_nonneg_right (complement_projection_norm_le P) (norm_nonneg _)))
        (norm_mul_le _ _)
    _ = _ := by rw [one_mul]

theorem unequal_rank_whitened_capture (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection]
    (q : OrthonormalBasis (Fin n) ℝ Q) (hn : 0 < n) (R : V →L[ℝ] V) :
    ‖(1 - P.starProjection) * R‖ ≤ ‖(1 - Q.starProjection) * R‖ +
      min 1 (Real.sqrt ((n : ℝ) * whiteLoss P Q q)) * ‖Q.starProjection * R‖ := by
  have hn0 : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.ne_of_gt hn)
  have hroot : inclusionDefect P Q ≤ Real.sqrt ((n : ℝ) * whiteLoss P Q q) := by
    simpa only [whiteLoss, mul_div_cancel₀ _ hn0] using inclusionDefect_le_sqrt_residualSum P Q q
  exact (one_sided_reference_capture P Q R).trans
    (add_le_add le_rfl (mul_le_mul_of_nonneg_right
      (le_min (inclusionDefect_le_one P Q) hroot) (norm_nonneg _)))

end LearnedTG
