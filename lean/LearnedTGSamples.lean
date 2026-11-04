module

public import LearnedTGPOD

/-! The empirical covariance is constructed from the actual finite list of snapshots. -/

@[expose] public section

open scoped BigOperators RealInnerProductSpace

namespace LearnedTG

variable {D V : Type*} {q : ℕ}
  [NormedAddCommGroup D] [InnerProductSpace ℝ D]
  [NormedAddCommGroup V] [InnerProductSpace ℝ V]

noncomputable def sampleCovariance (e : Fin q → D) : D →L[ℝ] D :=
  (q : ℝ)⁻¹ • ∑ i, InnerProductSpace.rankOne ℝ (e i) (e i)

theorem sampleCovariance_positive (e : Fin q → D) : (sampleCovariance e).IsPositive := by
  apply ContinuousLinearMap.IsPositive.smul_of_nonneg
  · exact ContinuousLinearMap.isPositive_sum _ fun i _ => InnerProductSpace.isPositive_rankOne_self (e i)
  · positivity

theorem sampleCovariance_apply (e : Fin q → D) (x : D) :
    sampleCovariance e x = (q : ℝ)⁻¹ • ∑ i, ⟪e i, x⟫ • e i := by
  simp only [sampleCovariance, smul_apply, sum_apply,
    InnerProductSpace.rankOne_apply]

/-- The finite snapshot matrix, as a map from Euclidean coefficients into the energy space. -/
noncomputable def snapshotSynthesis (e : Fin q → D) : EuclideanSpace ℝ (Fin q) →L[ℝ] D :=
  ∑ i, InnerProductSpace.rankOne ℝ (e i) (EuclideanSpace.basisFun (Fin q) ℝ i)

theorem snapshotSynthesis_apply (e : Fin q → D) (a : EuclideanSpace ℝ (Fin q)) :
    snapshotSynthesis e a = ∑ i, a i • e i := by
  simp only [snapshotSynthesis, sum_apply, InnerProductSpace.rankOne_apply, EuclideanSpace.basisFun_inner]

variable [CompleteSpace D] [CompleteSpace V]

theorem snapshotSynthesis_covariance (e : Fin q → D) (hq : 0 < q) :
    covariance (snapshotSynthesis e) = (q : ℝ) • sampleCovariance e := by
  classical
  have hq0 : (q : ℝ) ≠ 0 := ne_of_gt (Nat.cast_pos.mpr hq)
  let b := EuclideanSpace.basisFun (Fin q) ℝ
  have hadj (x : D) : (snapshotSynthesis e).adjoint x = ∑ i, ⟪e i, x⟫ • b i := by
    simp only [b, snapshotSynthesis, map_sum, InnerProductSpace.adjoint_rankOne,
      sum_apply, InnerProductSpace.rankOne_apply]
  have hb (i : Fin q) : snapshotSynthesis e (b i) = e i := by
    change (∑ j, InnerProductSpace.rankOne ℝ (e j) (b j)) (b i) = _
    simp only [sum_apply, InnerProductSpace.rankOne_apply, orthonormal_iff_ite.mp b.orthonormal,
      ite_smul, one_smul, zero_smul, Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  ext x
  change snapshotSynthesis e ((snapshotSynthesis e).adjoint x) = _
  rw [hadj]
  simp only [map_sum, map_smul, hb, smul_apply, sampleCovariance_apply, smul_smul,
    mul_inv_cancel₀ hq0, one_smul]

theorem lifted_sampleCovariance (J : D →L[ℝ] V) (e : Fin q → D) :
    (J ∘L sampleCovariance e) ∘L J.adjoint = sampleCovariance (fun i => J (e i)) := by
  ext x
  simp only [ContinuousLinearMap.comp_apply, sampleCovariance_apply,
    map_smul, map_sum, J.adjoint_inner_right]

end LearnedTG
