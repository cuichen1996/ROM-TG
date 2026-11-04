module

public import LearnedTGOperators
public import Mathlib.Analysis.InnerProductSpace.PiL2
public import Mathlib.Analysis.InnerProductSpace.Adjoint

/-! Finite-dimensional subspaces and whitened loss, using actual orthonormal bases. -/

@[expose] public section

open scoped BigOperators RealInnerProductSpace

namespace LearnedTG

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
variable {n : ℕ}

theorem norm_sum_smul_sq_le (a : Fin n → ℝ) (u : Fin n → V) :
    ‖∑ i, a i • u i‖ ^ 2 ≤ (∑ i, a i ^ 2) * (∑ i, ‖u i‖ ^ 2) := by
  have hnorm : ‖∑ i, a i • u i‖ ≤ ∑ i, |a i| * ‖u i‖ := by
    simpa only [norm_smul, Real.norm_eq_abs] using norm_sum_le (s := Finset.univ) (f := fun i => a i • u i)
  calc
    _ ≤ (∑ i, |a i| * ‖u i‖) ^ 2 :=
      (sq_le_sq₀ (norm_nonneg _) (Finset.sum_nonneg fun i _ => mul_nonneg (abs_nonneg _) (norm_nonneg _))).mpr hnorm
    _ ≤ (∑ i, |a i| ^ 2) * (∑ i, ‖u i‖ ^ 2) :=
      Finset.sum_mul_sq_le_sq_mul_sq Finset.univ _ _
    _ = _ := by simp only [sq_abs]

theorem apply_sq_le_basis_sum (Q : Submodule ℝ V) (b : OrthonormalBasis (Fin n) ℝ Q)
    (T : V →L[ℝ] V) (x : Q) :
    ‖T x‖ ^ 2 ≤ (∑ i, ‖T (b i : V)‖ ^ 2) * ‖x‖ ^ 2 := by
  have hexp : T x = ∑ i, b.repr x i • T (b i : V) := by
    conv_lhs => rw [← b.sum_repr x]
    simp only [Submodule.coe_sum, Submodule.coe_smul, map_sum, map_smul]
  rw [hexp]
  have hcoeff : ∑ i, (b.repr x i) ^ 2 = ‖x‖ ^ 2 := by
    rw [← EuclideanSpace.real_norm_sq_eq]
    exact congrArg (fun t : ℝ => t ^ 2) (b.repr.norm_map x)
  simpa only [hcoeff, mul_comm] using norm_sum_smul_sq_le (fun i => b.repr x i) (fun i => T (b i : V))

noncomputable def residualSum (P Q : Submodule ℝ V) [P.HasOrthogonalProjection]
    (b : OrthonormalBasis (Fin n) ℝ Q) : ℝ :=
  ∑ i, ‖(1 - P.starProjection) (b i : V)‖ ^ 2

noncomputable def whiteLoss (P Q : Submodule ℝ V) [P.HasOrthogonalProjection]
    (b : OrthonormalBasis (Fin n) ℝ Q) : ℝ := residualSum P Q b / n

theorem residualSum_nonneg (P Q : Submodule ℝ V) [P.HasOrthogonalProjection]
    (b : OrthonormalBasis (Fin n) ℝ Q) : 0 ≤ residualSum P Q b :=
  Finset.sum_nonneg fun _ _ => sq_nonneg _

theorem whiteLoss_nonneg (P Q : Submodule ℝ V) [P.HasOrthogonalProjection]
    (b : OrthonormalBasis (Fin n) ℝ Q) : 0 ≤ whiteLoss P Q b :=
  div_nonneg (residualSum_nonneg P Q b) (Nat.cast_nonneg n)

theorem projection_norm_sq_eq_sum (P : Submodule ℝ V) [P.HasOrthogonalProjection]
    (b : OrthonormalBasis (Fin n) ℝ P) (x : V) :
    ‖P.starProjection x‖ ^ 2 = ∑ i, ⟪(b i : V), x⟫ ^ 2 := by
  change ‖P.orthogonalProjectionOnto x‖ ^ 2 = _
  rw [← b.sum_sq_inner_right (P.orthogonalProjectionOnto x)]
  apply Finset.sum_congr rfl
  intro i _
  change ⟪(b i : V), P.starProjection x⟫ ^ 2 = _
  rw [← P.inner_starProjection_left_eq_right, P.starProjection_mem_subspace_eq_self]

theorem residualSum_eq_rank_sub (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] (b : OrthonormalBasis (Fin n) ℝ Q) :
    residualSum P Q b = n - ∑ i, ‖P.starProjection (b i : V)‖ ^ 2 := by
  have hpoint (i : Fin n) :
      ‖(1 - P.starProjection) (b i : V)‖ ^ 2 = 1 - ‖P.starProjection (b i : V)‖ ^ 2 := by
    have hpyth := P.norm_sq_eq_add_norm_sq_starProjection (b i : V)
    rw [P.starProjection_orthogonal'] at hpyth
    have hn : ‖(b i : V)‖ = 1 := b.orthonormal.norm_eq_one i
    rw [hn] at hpyth
    nlinarith
  unfold residualSum
  simp_rw [hpoint]
  rw [Finset.sum_sub_distrib]
  simp

/-- Equality of the two squared inclusion defects is proved by interchanging finite sums. -/
theorem residualSum_symm (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection]
    (p : OrthonormalBasis (Fin n) ℝ P) (q : OrthonormalBasis (Fin n) ℝ Q) :
    residualSum P Q q = residualSum Q P p := by
  rw [residualSum_eq_rank_sub, residualSum_eq_rank_sub]
  congr 1
  simp_rw [projection_norm_sq_eq_sum P p, projection_norm_sq_eq_sum Q q]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  rw [real_inner_comm]

theorem projected_complement_sq_le (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection]
    (p : OrthonormalBasis (Fin n) ℝ P) (x : V) :
    ‖P.starProjection ((1 - Q.starProjection) x)‖ ^ 2 ≤
      residualSum Q P p * ‖(1 - Q.starProjection) x‖ ^ 2 := by
  let z := (1 - Q.starProjection) x
  have hz : Q.starProjection z = 0 := by
    change Q.starProjection (x - Q.starProjection x) = 0
    rw [map_sub, Q.starProjection_eq_self_iff.mpr (Q.starProjection_apply_mem x), sub_self]
  rw [projection_norm_sq_eq_sum P p]
  unfold residualSum
  rw [Finset.sum_mul]
  apply Finset.sum_le_sum
  intro i _
  have hinner : ⟪(p i : V), z⟫ = ⟪(1 - Q.starProjection) (p i : V), z⟫ := by
    simp only [sub_apply, one_apply_eq_self, inner_sub_left]
    rw [Q.inner_starProjection_left_eq_right, hz, inner_zero_right, sub_zero]
  change ⟪(p i : V), z⟫ ^ 2 ≤ _
  rw [hinner, ← mul_pow]
  simpa only [sq_abs] using
    (sq_le_sq₀ (abs_nonneg ⟪(1 - Q.starProjection) (p i : V), z⟫)
      (mul_nonneg (norm_nonneg _) (norm_nonneg _))).mpr
      (abs_real_inner_le_norm ((1 - Q.starProjection) (p i : V)) z)

theorem projection_difference_sq_apply_le (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection]
    (p : OrthonormalBasis (Fin n) ℝ P) (q : OrthonormalBasis (Fin n) ℝ Q) (x : V) :
    ‖(P.starProjection - Q.starProjection) x‖ ^ 2 ≤ residualSum P Q q * ‖x‖ ^ 2 := by
  let z := (1 - Q.starProjection) x
  let a := P.starProjection z
  let b := (1 - P.starProjection) (Q.starProjection x)
  have hdecomp : (P.starProjection - Q.starProjection) x = a - b := by
    simp only [a, b, z, sub_apply, one_apply_eq_self, map_sub]
    abel
  have horth : ⟪a, b⟫ = 0 := by
    rw [real_inner_comm]
    exact P.starProjection_inner_eq_zero (Q.starProjection x) a
      (P.starProjection_apply_mem z)
  have ha := projected_complement_sq_le P Q p x
  rw [← residualSum_symm P Q p q] at ha
  have hb := apply_sq_le_basis_sum Q q (1 - P.starProjection) (Q.orthogonalProjectionOnto x)
  have hpyth := Q.norm_sq_eq_add_norm_sq_starProjection x
  rw [Q.starProjection_orthogonal'] at hpyth
  rw [hdecomp, norm_sub_sq_real, horth]
  change ‖a‖ ^ 2 - 2 * 0 + ‖b‖ ^ 2 ≤ _
  change ‖a‖ ^ 2 ≤ residualSum P Q q * ‖z‖ ^ 2 at ha
  change ‖b‖ ^ 2 ≤ residualSum P Q q * ‖Q.starProjection x‖ ^ 2 at hb
  change ‖x‖ ^ 2 = ‖Q.starProjection x‖ ^ 2 + ‖z‖ ^ 2 at hpyth
  calc
    _ = ‖a‖ ^ 2 + ‖b‖ ^ 2 := by ring
    _ ≤ residualSum P Q q * ‖z‖ ^ 2 +
        residualSum P Q q * ‖Q.starProjection x‖ ^ 2 := add_le_add ha hb
    _ = residualSum P Q q * ‖x‖ ^ 2 := by rw [hpyth]; ring

theorem projection_difference_le_sqrt_residualSum (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection]
    (p : OrthonormalBasis (Fin n) ℝ P) (q : OrthonormalBasis (Fin n) ℝ Q) :
    ‖P.starProjection - Q.starProjection‖ ≤ Real.sqrt (residualSum P Q q) := by
  apply ContinuousLinearMap.opNorm_le_of_unit_norm (Real.sqrt_nonneg _)
  intro x hx
  apply Real.le_sqrt_of_sq_le
  simpa only [hx, one_pow, mul_one] using projection_difference_sq_apply_le P Q p q x

theorem whiteLoss_controls_projection_distance (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection]
    (p : OrthonormalBasis (Fin n) ℝ P) (q : OrthonormalBasis (Fin n) ℝ Q)
    (hn : 0 < n) :
    ‖P.starProjection - Q.starProjection‖ ≤ Real.sqrt ((n : ℝ) * whiteLoss P Q q) := by
  have hn0 : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.ne_of_gt hn)
  simpa only [whiteLoss, mul_div_cancel₀ _ hn0] using
    projection_difference_le_sqrt_residualSum P Q p q

theorem whiteLoss_le_projection_distance_sq (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection]
    (q : OrthonormalBasis (Fin n) ℝ Q) (hn : 0 < n) :
    whiteLoss P Q q ≤ ‖P.starProjection - Q.starProjection‖ ^ 2 := by
  apply (div_le_iff₀ (Nat.cast_pos.mpr hn)).mpr
  calc
    residualSum P Q q ≤ ∑ _ : Fin n, ‖P.starProjection - Q.starProjection‖ ^ 2 := by
      apply Finset.sum_le_sum
      intro i _
      have heq : (1 - P.starProjection) (q i : V) =
          (Q.starProjection - P.starProjection) (q i : V) := by simp
      have hb := (Q.starProjection - P.starProjection).le_opNorm (q i : V)
      have hunit : ‖(q i : V)‖ = 1 := q.orthonormal.norm_eq_one i
      rw [hunit, mul_one, norm_sub_rev] at hb
      rw [heq]
      exact (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mpr hb
    _ = ‖P.starProjection - Q.starProjection‖ ^ 2 * (n : ℝ) := by simp [mul_comm]

theorem projection_difference_norm_le_one (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection] :
    ‖P.starProjection - Q.starProjection‖ ≤ 1 := by
  apply ContinuousLinearMap.opNorm_le_of_unit_norm zero_le_one
  intro x hx
  let z := (1 - Q.starProjection) x
  let a := P.starProjection z
  let b := (1 - P.starProjection) (Q.starProjection x)
  have heq : (P.starProjection - Q.starProjection) x = a - b := by
    simp only [a, b, z, sub_apply, one_apply_eq_self, map_sub]
    abel
  have horth : ⟪a, b⟫ = 0 := by
    rw [real_inner_comm]
    exact P.starProjection_inner_eq_zero (Q.starProjection x) a (P.starProjection_apply_mem z)
  have ha := (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mpr (P.norm_starProjection_apply_le z)
  have hb := (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mpr
    (complement_projection_apply_le P (Q.starProjection x))
  have hpyth := Q.norm_sq_eq_add_norm_sq_starProjection x
  rw [Q.starProjection_orthogonal', hx] at hpyth
  rw [heq]
  apply (sq_le_sq₀ (norm_nonneg _) zero_le_one).mp
  rw [norm_sub_sq_real, horth]
  change ‖a‖ ^ 2 ≤ ‖z‖ ^ 2 at ha
  change ‖b‖ ^ 2 ≤ ‖Q.starProjection x‖ ^ 2 at hb
  change 1 ^ 2 = ‖Q.starProjection x‖ ^ 2 + ‖z‖ ^ 2 at hpyth
  nlinarith

theorem exists_unit_mem_orthogonal (P W : Submodule ℝ V)
    [FiniteDimensional ℝ P] [FiniteDimensional ℝ W]
    (hdim : Module.finrank ℝ W < Module.finrank ℝ P) :
    ∃ v : V, v ∈ P ∧ v ∈ Wᗮ ∧ ‖v‖ = 1 := by
  let f : P →L[ℝ] W := W.orthogonalProjectionOnto ∘L P.subtypeL
  have hninj : ¬ Function.Injective f := by
    intro hinj
    exact (not_le_of_gt hdim) (f.toLinearMap.finrank_le_finrank_of_injective hinj)
  have hker : f.ker ≠ ⊥ := by
    intro h
    exact hninj (LinearMap.ker_eq_bot.mp h)
  obtain ⟨u, hu, hu0⟩ := Submodule.exists_mem_ne_zero_of_ne_bot hker
  change W.orthogonalProjectionOnto (u : V) = 0 at hu
  have huW : (u : V) ∈ Wᗮ := W.orthogonalProjectionOnto_eq_zero_iff.mp hu
  have huV : (u : V) ≠ 0 := by
    intro h
    apply hu0
    exact Subtype.ext h
  refine ⟨‖(u : V)‖⁻¹ • (u : V), P.smul_mem _ u.prop, Wᗮ.smul_mem _ huW, ?_⟩
  rw [norm_smul, Real.norm_eq_abs, abs_inv, abs_norm, inv_mul_cancel₀ (norm_ne_zero_iff.mpr huV)]

/-- Different finite ranks force the full orthogonal-projector distance to one.
This is the dimension obstruction in the paper's unequal-rank remark. -/
theorem unequal_rank_projection_distance (P Q : Submodule ℝ V)
    [FiniteDimensional ℝ P] [FiniteDimensional ℝ Q]
    (hrank : Module.finrank ℝ Q < Module.finrank ℝ P) :
    ‖P.starProjection - Q.starProjection‖ = 1 := by
  obtain ⟨v, hvP, hvQ, hvnorm⟩ := exists_unit_mem_orthogonal P Q hrank
  have hp : P.starProjection v = v := P.starProjection_eq_self_iff.mpr hvP
  have hq : Q.starProjection v = 0 := by
    change (Q.orthogonalProjectionOnto v : V) = 0
    exact congrArg (fun q : Q => (q : V)) (Q.orthogonalProjectionOnto_eq_zero_iff.mpr hvQ)
  apply le_antisymm (projection_difference_norm_le_one P Q)
  have h := (P.starProjection - Q.starProjection).le_opNorm v
  simpa only [sub_apply, hp, hq, sub_zero, hvnorm, mul_one] using h

section Perturbation

variable [CompleteSpace V]

theorem projection_cross_norm (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection] :
    ‖Q.starProjection * (1 - P.starProjection)‖ =
      ‖(1 - P.starProjection) * Q.starProjection‖ := by
  have hP : star P.starProjection = P.starProjection := isSelfAdjoint_starProjection P
  have hQ : star Q.starProjection = Q.starProjection := isSelfAdjoint_starProjection Q
  calc
    _ = ‖star ((1 - P.starProjection) * Q.starProjection)‖ := by
      rw [star_mul, star_sub, star_one, hP, hQ]
    _ = _ := by
      change ‖ContinuousLinearMap.adjoint ((1 - P.starProjection) * Q.starProjection)‖ = _
      exact ContinuousLinearMap.adjoint.norm_map _

theorem projection_distance_le_cross_sum (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection] :
    ‖P.starProjection - Q.starProjection‖ ≤
      ‖(1 - Q.starProjection) * P.starProjection‖ +
        ‖(1 - P.starProjection) * Q.starProjection‖ := by
  have heq : P.starProjection - Q.starProjection =
      (1 - Q.starProjection) * P.starProjection - Q.starProjection * (1 - P.starProjection) := by
    rw [sub_mul, mul_sub, one_mul, mul_one]
    abel
  rw [heq]
  exact (norm_sub_le _ _).trans_eq (by rw [projection_cross_norm P Q])

end Perturbation

end LearnedTG
