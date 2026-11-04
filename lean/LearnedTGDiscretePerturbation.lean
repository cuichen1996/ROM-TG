module

public import LearnedTGCore
public import LearnedTGLosses
import Mathlib.Tactic.Ring

/-! Fixed-grid TG differences in an operator algebra. No contraction,
symmetry, normality, coverage or cross-grid premise is used. -/

@[expose] public section

namespace LearnedTG

variable {A : Type*} [NormedRing A]

theorem discrete_tg_difference (L R P Q : A) :
    L * (1 - P) * R - L * (1 - Q) * R = L * (Q - P) * R := by
  simp only [mul_sub, sub_mul, mul_one]
  abel

theorem discrete_tg_difference_norm (L R P Q : A) :
    ‖L * (1 - P) * R - L * (1 - Q) * R‖ ≤
      ‖L‖ * ‖R‖ * ‖P - Q‖ := by
  rw [discrete_tg_difference]
  calc
    _ ≤ ‖L * (Q - P)‖ * ‖R‖ := norm_mul_le _ _
    _ ≤ (‖L‖ * ‖Q - P‖) * ‖R‖ :=
      mul_le_mul_of_nonneg_right (norm_mul_le _ _) (norm_nonneg _)
    _ = _ := by rw [norm_sub_rev]; ring

theorem discrete_online_weighted_difference (L R P Q : A) :
    ‖L * (1 - P) * R - L * (1 - Q) * R‖ ≤
      ‖L‖ * ‖(Q - P) * R‖ := by
  rw [discrete_tg_difference, mul_assoc]
  exact norm_mul_le _ _

theorem discrete_weighted_defect_eq (L R P Q : A) :
    ‖L * (1 - P) * R - L * (1 - Q) * R‖ = ‖L * (P - Q) * R‖ := by
  rw [discrete_tg_difference]
  have h : L * (Q - P) * R = -(L * (P - Q) * R) := by noncomm_ring
  rw [h, norm_neg]

/-- Allowance for arbitrary practical coarse corrections C and D relative
to the ideal projectors P and Q. No basis-invariance of regularization is assumed. -/
theorem regularized_tg_perturbation (L R P Q C D : A) :
    ‖L * (1 - C) * R - L * (1 - D) * R‖ ≤
      ‖L * (1 - P) * R - L * (1 - Q) * R‖ +
      ‖L‖ * ‖R‖ * (‖P - C‖ + ‖Q - D‖) := by
  have h : L * (1 - C) * R - L * (1 - D) * R =
      (L * (1 - C) * R - L * (1 - P) * R) +
      (L * (1 - P) * R - L * (1 - Q) * R) +
      (L * (1 - Q) * R - L * (1 - D) * R) := by abel
  rw [h]
  calc
    _ ≤ (‖L * (1 - C) * R - L * (1 - P) * R‖ +
        ‖L * (1 - P) * R - L * (1 - Q) * R‖) +
        ‖L * (1 - Q) * R - L * (1 - D) * R‖ :=
      (norm_add_le _ _).trans (add_le_add (norm_add_le _ _) le_rfl)
    _ ≤ _ := by
      have hC := discrete_tg_difference_norm L R C P
      rw [norm_sub_rev C P] at hC
      have hD := discrete_tg_difference_norm L R Q D
      linarith

section Hilbert

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] {n : ℕ}

theorem discrete_projector_loss_bound (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection]
    (p : OrthonormalBasis (Fin n) ℝ P) (q : OrthonormalBasis (Fin n) ℝ Q) (hn : 0 < n) :
    ‖P.starProjection - Q.starProjection‖ ≤ min 1 (Real.sqrt ((n : ℝ) * whiteLoss P Q q)) :=
  le_min (projection_difference_norm_le_one P Q) (whiteLoss_controls_projection_distance P Q p q hn)

theorem discrete_tg_loss_bound (L R : V →L[ℝ] V) (P Q : Submodule ℝ V)
    [P.HasOrthogonalProjection] [Q.HasOrthogonalProjection]
    (p : OrthonormalBasis (Fin n) ℝ P) (q : OrthonormalBasis (Fin n) ℝ Q) (hn : 0 < n) :
    ‖tg L R P - tg L R Q‖ ≤ ‖L‖ * ‖R‖ * min 1 (Real.sqrt ((n : ℝ) * whiteLoss P Q q)) := by
  exact (discrete_tg_difference_norm L R P.starProjection Q.starProjection).trans
    (mul_le_mul_of_nonneg_left (discrete_projector_loss_bound P Q p q hn)
      (mul_nonneg (norm_nonneg L) (norm_nonneg R)))

noncomputable def restrictedSmoothingNorm (L D : V →L[ℝ] V) : ℝ :=
  ‖L ∘L D.range.subtypeL‖

theorem restrictedSmoothingNorm_le (L D : V →L[ℝ] V) : restrictedSmoothingNorm L D ≤ ‖L‖ := by
  apply ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg _)
  intro v
  exact L.le_opNorm (v : V)

theorem restrictedSmoothingNorm_zero (L : V →L[ℝ] V) : restrictedSmoothingNorm L 0 = 0 := by
  have h : L ∘L (0 : V →L[ℝ] V).range.subtypeL = 0 := by
    ext v
    have hv : (v : V) = 0 := by
      obtain ⟨x, hx⟩ := v.prop
      exact hx.symm
    change L (v : V) = 0
    rw [hv, map_zero]
  rw [restrictedSmoothingNorm, h]
  exact ContinuousLinearMap.opNorm_zero

theorem operator_weighted_defect (L D R : V →L[ℝ] V) :
    ‖L * D * R‖ ≤ restrictedSmoothingNorm L D * ‖D * R‖ := by
  unfold restrictedSmoothingNorm
  apply ContinuousLinearMap.opNorm_le_bound (L * D * R)
    (mul_nonneg (norm_nonneg (L ∘L D.range.subtypeL)) (norm_nonneg (D * R)))
  intro v
  have h := (L ∘L D.range.subtypeL).le_opNorm (⟨D (R v), ⟨R v, rfl⟩⟩ : D.range)
  change ‖L (D (R v))‖ ≤ _ at h ⊢
  simpa only [mul_assoc] using
    h.trans (mul_le_mul_of_nonneg_left ((D * R).le_opNorm v) (norm_nonneg _))

theorem operator_weighted_product_bound (L D R : V →L[ℝ] V) :
    restrictedSmoothingNorm L D * ‖D * R‖ ≤ ‖L‖ * ‖R‖ * ‖D‖ := by
  calc
    _ ≤ ‖L‖ * (‖D‖ * ‖R‖) := mul_le_mul (restrictedSmoothingNorm_le L D)
      (norm_mul_le _ _) (norm_nonneg _) (norm_nonneg _)
    _ = _ := by ring

end Hilbert

end LearnedTG

#print axioms LearnedTG.discrete_tg_difference
#print axioms LearnedTG.discrete_tg_difference_norm
#print axioms LearnedTG.discrete_online_weighted_difference
