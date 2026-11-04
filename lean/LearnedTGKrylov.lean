module

public import LearnedTGOperators
public import LearnedTGPolynomial

/-! Exact Hilbert-space minimal-residual Krylov iteration for a fixed linear operator. -/

@[expose] public section

open scoped RealInnerProductSpace BigOperators

namespace LearnedTG

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

noncomputable def krylovSpace (T : V →L[ℝ] V) (r : V) : ℕ → Submodule ℝ V
  | 0 => ⊥
  | m + 1 => (ℝ ∙ r) ⊔ (krylovSpace T r m).map T.toLinearMap

theorem krylovSpace_nested (T : V →L[ℝ] V) (r : V) (m : ℕ) :
    krylovSpace T r m ≤ krylovSpace T r (m + 1) := by
  induction m with
  | zero => exact bot_le
  | succ m ih => exact sup_le_sup le_rfl (Submodule.map_mono ih)

theorem power_mem_krylov (T : V →L[ℝ] V) (r : V) (m i : ℕ) (hi : i < m) :
    (T ^ i) r ∈ krylovSpace T r m := by
  induction m generalizing i with
  | zero => omega
  | succ m ih =>
    cases i with
    | zero =>
      apply (show (ℝ ∙ r) ≤ krylovSpace T r (m + 1) from le_sup_left)
      simpa only [pow_zero, one_apply_eq_self] using Submodule.mem_span_singleton_self r
    | succ i =>
      have hmem := ih i (by omega)
      have hmap : T ((T ^ i) r) ∈ (krylovSpace T r m).map T.toLinearMap := ⟨_, hmem, rfl⟩
      apply (show (krylovSpace T r m).map T.toLinearMap ≤ krylovSpace T r (m + 1) from le_sup_right)
      simpa only [pow_succ', mul_apply_eq_comp] using hmap

/-- The recursive space used for residual minimization is the usual Krylov power span. -/
theorem krylovSpace_eq_span_powers (T : V →L[ℝ] V) (r : V) (m : ℕ) :
    krylovSpace T r m = Submodule.span ℝ {v | ∃ i < m, (T ^ i) r = v} := by
  apply le_antisymm
  · induction m with
    | zero => exact bot_le
    | succ m ih =>
      apply sup_le
      · apply Submodule.span_le.mpr
        intro v hv
        rcases Set.mem_singleton_iff.mp hv with rfl
        exact Submodule.subset_span ⟨0, by omega, by simp⟩
      · intro y hy
        obtain ⟨x, hx, rfl⟩ := hy
        have hxspan := ih hx
        refine Submodule.span_induction
          (p := fun x _ => T x ∈ Submodule.span ℝ {v | ∃ i < m + 1, (T ^ i) r = v})
          ?_ ?_ ?_ ?_ hxspan
        · intro x hx
          obtain ⟨i, hi, rfl⟩ := hx
          apply Submodule.subset_span
          exact ⟨i + 1, by omega, by rw [pow_succ']; rfl⟩
        · simpa only [map_zero] using (Submodule.zero_mem (Submodule.span ℝ _))
        · intro x y _ _ hx hy
          simpa only [map_add] using Submodule.add_mem _ hx hy
        · intro a x _ hx
          simpa only [map_smul] using Submodule.smul_mem _ a hx
  · apply Submodule.span_le.mpr
    rintro v ⟨i, hi, rfl⟩
    exact power_mem_krylov T r m i hi

instance krylovSpace_finite (T : V →L[ℝ] V) (r : V) (m : ℕ) :
    FiniteDimensional ℝ (krylovSpace T r m) := by
  induction m with
  | zero => change FiniteDimensional ℝ (⊥ : Submodule ℝ V); infer_instance
  | succ m ih =>
    let _ := ih
    change FiniteDimensional ℝ ((ℝ ∙ r) ⊔ (krylovSpace T r m).map T.toLinearMap : Submodule ℝ V)
    infer_instance

noncomputable def gmresResidual (T : V →L[ℝ] V) (r : V) (m : ℕ) : V :=
  (1 - ((krylovSpace T r m).map T.toLinearMap).starProjection) r

/-- Minimality is proved from the actual Krylov correction-space projection,
not supplied as an assumption or inferred from a stationary contraction. -/
theorem gmres_minimal_residual (T : V →L[ℝ] V) (r : V) (m : ℕ) (x : V)
    (hx : x ∈ krylovSpace T r m) : ‖gmresResidual T r m‖ ≤ ‖r - T x‖ := by
  let S := (krylovSpace T r m).map T.toLinearMap
  have hy : T x ∈ S := ⟨x, hx, rfl⟩
  have heq : (1 - S.starProjection) r = (1 - S.starProjection) (r - T x) := by
    simp only [sub_apply, one_apply_eq_self, map_sub, S.starProjection_eq_self_iff.mpr hy]
    abel
  change ‖(1 - S.starProjection) r‖ ≤ _
  rw [heq]
  exact complement_projection_apply_le S _

noncomputable def polynomialCorrection (a : ℕ → ℝ) (T : V →L[ℝ] V) (r : V) (m : ℕ) : V :=
  -(∑ i ∈ Finset.range m, a (i + 1) • (T ^ i) r)

theorem polynomialCorrection_mem_krylov (a : ℕ → ℝ) (T : V →L[ℝ] V) (r : V) (m : ℕ) :
    polynomialCorrection a T r m ∈ krylovSpace T r m := by
  apply Submodule.neg_mem
  apply Submodule.sum_mem
  intro i hi
  exact Submodule.smul_mem _ _ (power_mem_krylov T r m i (Finset.mem_range.mp hi))

theorem polynomialCorrection_residual (a : ℕ → ℝ) (ha : a 0 = 1)
    (T : V →L[ℝ] V) (r : V) (m : ℕ) :
    r - T (polynomialCorrection a T r m) = polynomialOperator a T m r := by
  unfold polynomialOperator
  rw [Finset.sum_range_succ']
  simp only [add_apply, sum_apply, smul_apply, pow_zero, one_apply_eq_self,
    ha, one_smul, pow_succ', mul_apply_eq_comp]
  simp only [polynomialCorrection, map_neg, map_sum, map_smul]
  abel

theorem gmres_candidate_polynomial (a : ℕ → ℝ) (ha : a 0 = 1)
    (T : V →L[ℝ] V) (r : V) (m : ℕ) :
    ‖gmresResidual T r m‖ ≤ ‖polynomialOperator a T m r‖ := by
  simpa only [polynomialCorrection_residual a ha] using
    gmres_minimal_residual T r m (polynomialCorrection a T r m)
      (polynomialCorrection_mem_krylov a T r m)

variable [Nontrivial V]

theorem gmres_polynomial_comparison (a : ℕ → ℝ) (ha : a 0 = 1)
    (T U : V →L[ℝ] V) (M N : ℝ) (hT : ‖T‖ ≤ M) (hU : ‖U‖ ≤ N) (r : V) (m : ℕ) :
    ‖gmresResidual T r m‖ ≤ ‖polynomialOperator a U m r‖ +
      (‖T - U‖ * polynomialPerturbationFactor a M N m) * ‖r‖ := by
  have heq : polynomialOperator a T m r = polynomialOperator a U m r +
      (polynomialOperator a T m - polynomialOperator a U m) r := by
    rw [sub_apply]
    abel
  calc
    _ ≤ ‖polynomialOperator a T m r‖ := gmres_candidate_polynomial a ha T r m
    _ ≤ ‖polynomialOperator a U m r‖ + ‖(polynomialOperator a T m - polynomialOperator a U m) r‖ :=
      by rw [heq]; exact norm_add_le _ _
    _ ≤ _ := add_le_add le_rfl
      (((polynomialOperator a T m - polynomialOperator a U m).le_opNorm r).trans
        (mul_le_mul_of_nonneg_right (polynomial_perturbation a T U M N hT hU m) (norm_nonneg r)))

theorem gmres_polynomial_ratio (a : ℕ → ℝ) (ha : a 0 = 1)
    (T U : V →L[ℝ] V) (M N : ℝ) (hT : ‖T‖ ≤ M) (hU : ‖U‖ ≤ N)
    (r : V) (hr : r ≠ 0) (m : ℕ) :
    ‖gmresResidual T r m‖ / ‖r‖ ≤ ‖polynomialOperator a U m r‖ / ‖r‖ +
      ‖T - U‖ * polynomialPerturbationFactor a M N m := by
  have h := div_le_div_of_nonneg_right
    (gmres_polynomial_comparison a ha T U M N hT hU r m) (norm_nonneg r)
  simpa only [add_div, mul_div_cancel_right₀ _ (norm_ne_zero_iff.mpr hr)] using h


end LearnedTG
