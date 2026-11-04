module

public import LearnedTGKrylov

/-! Isometric transport between error and residual spaces, without contraction or spectral bounds. -/

@[expose] public section

open scoped RealInnerProductSpace

namespace LearnedTG

variable {V F : Type*}
  [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  [NormedAddCommGroup F] [InnerProductSpace ℝ F]

noncomputable def isometricConjugate (U : V ≃ₗᵢ[ℝ] F) (E : V →L[ℝ] V) : F →L[ℝ] F :=
  (U.toContinuousLinearEquiv.toContinuousLinearMap ∘L E) ∘L
    U.symm.toContinuousLinearEquiv.toContinuousLinearMap

theorem isometricConjugate_norm (U : V ≃ₗᵢ[ℝ] F) (E : V →L[ℝ] V) :
    ‖isometricConjugate U E‖ = ‖E‖ := by
  apply le_antisymm
  · apply ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg _)
    intro f
    change ‖U (E (U.symm f))‖ ≤ _
    rw [U.norm_map]
    exact (E.le_opNorm _).trans_eq (by rw [U.symm.norm_map])
  · apply ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg _)
    intro v
    have h := (isometricConjugate U E).le_opNorm (U v)
    change ‖U (E (U.symm (U v)))‖ ≤ _ at h
    simpa only [U.symm_apply_apply, U.norm_map] using h

/-- Taking U to be the energy Riesz isometry yields the paper's right-preconditioned operator. -/
theorem right_preconditioned_similarity (U : V ≃ₗᵢ[ℝ] F) (B : F →L[ℝ] V) :
    U.toContinuousLinearEquiv.toContinuousLinearMap ∘L B =
      isometricConjugate U (B ∘L U.toContinuousLinearEquiv.toContinuousLinearMap) := by
  ext f
  change U (B f) = U (B (U (U.symm f)))
  rw [U.apply_symm_apply]

theorem isometricConjugate_one_sub (U : V ≃ₗᵢ[ℝ] F) (E : V →L[ℝ] V) :
    isometricConjugate U (1 - E) = 1 - isometricConjugate U E := by
  ext f
  change U ((1 - E) (U.symm f)) = f - U (E (U.symm f))
  rw [sub_apply, one_apply_eq_self, map_sub, U.apply_symm_apply]


/-- The actual difference of right-preconditioned residual operators is the
negative isometric conjugate of the error-operator difference. -/
theorem right_preconditioned_difference (U : V ≃ₗᵢ[ℝ] F) (B C : F →L[ℝ] V) :
    U.toContinuousLinearEquiv.toContinuousLinearMap ∘L B -
      U.toContinuousLinearEquiv.toContinuousLinearMap ∘L C =
    -isometricConjugate U
      ((1 - B ∘L U.toContinuousLinearEquiv.toContinuousLinearMap) -
       (1 - C ∘L U.toContinuousLinearEquiv.toContinuousLinearMap)) := by
  ext f
  change U (B f) - U (C f) =
    -U ((U.symm f - B (U (U.symm f))) - (U.symm f - C (U (U.symm f))))
  simp only [U.apply_symm_apply, map_sub]
  abel

theorem right_preconditioned_difference_norm (U : V ≃ₗᵢ[ℝ] F) (B C : F →L[ℝ] V) :
    ‖U.toContinuousLinearEquiv.toContinuousLinearMap ∘L B -
      U.toContinuousLinearEquiv.toContinuousLinearMap ∘L C‖ =
    ‖(1 - B ∘L U.toContinuousLinearEquiv.toContinuousLinearMap) -
      (1 - C ∘L U.toContinuousLinearEquiv.toContinuousLinearMap)‖ := by
  rw [right_preconditioned_difference, norm_neg, isometricConjugate_norm]

theorem norm_precompose_isometry (U : V ≃ₗᵢ[ℝ] F) (B : F →L[ℝ] V) :
    ‖B ∘L U.toContinuousLinearEquiv.toContinuousLinearMap‖ = ‖B‖ := by
  apply le_antisymm
  · apply ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg B)
    intro v
    change ‖B (U v)‖ ≤ ‖B‖ * ‖v‖
    simpa only [U.norm_map] using B.le_opNorm (U v)
  · apply ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg _)
    intro f
    have h := (B ∘L U.toContinuousLinearEquiv.toContinuousLinearMap).le_opNorm (U.symm f)
    change ‖B (U (U.symm f))‖ ≤ _ at h
    simpa only [U.apply_symm_apply, U.symm.norm_map] using h

theorem preconditioner_difference_norm (U : V ≃ₗᵢ[ℝ] F) (B C : F →L[ℝ] V) :
    ‖(1 - B ∘L U.toContinuousLinearEquiv.toContinuousLinearMap) -
      (1 - C ∘L U.toContinuousLinearEquiv.toContinuousLinearMap)‖ = ‖B - C‖ := by
  have h : (1 - B ∘L U.toContinuousLinearEquiv.toContinuousLinearMap) -
      (1 - C ∘L U.toContinuousLinearEquiv.toContinuousLinearMap) =
      -((B - C) ∘L U.toContinuousLinearEquiv.toContinuousLinearMap) := by
    ext v
    simp only [sub_apply, one_apply_eq_self, neg_apply, ContinuousLinearMap.comp_apply]
    abel
  rw [h, norm_neg, norm_precompose_isometry]

end LearnedTG
