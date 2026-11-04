module

public import LearnedTGCore
public import Mathlib.Analysis.InnerProductSpace.Projection.FiniteDimensional

/-! Fixed Hilbert-space TG maps and actual orthogonal projections. -/

@[expose] public section

namespace LearnedTG

variable {V : Type*}
  [NormedAddCommGroup V] [InnerProductSpace ℝ V]

noncomputable def tg (L R : V →L[ℝ] V) (U : Submodule ℝ V)
    [U.HasOrthogonalProjection] : V →L[ℝ] V :=
  L * (1 - U.starProjection) * R

theorem tg_apply (L R : V →L[ℝ] V) (U : Submodule ℝ V)
    [U.HasOrthogonalProjection] (v : V) :
    tg L R U v = L (R v - U.starProjection (R v)) := rfl

theorem complement_projection_norm_le (U : Submodule ℝ V)
    [U.HasOrthogonalProjection] : ‖1 - U.starProjection‖ ≤ 1 := by
  rw [← U.starProjection_orthogonal']
  exact Uᗮ.starProjection_norm_le

theorem complement_projection_apply_le (U : Submodule ℝ V)
    [U.HasOrthogonalProjection] (v : V) :
    ‖(1 - U.starProjection) v‖ ≤ ‖v‖ := by
  simpa only [one_mul] using
    (1 - U.starProjection).le_of_opNorm_le (complement_projection_norm_le U) v

theorem tg_norm_le (L R : V →L[ℝ] V) (P : Submodule ℝ V)
    [P.HasOrthogonalProjection] : ‖tg L R P‖ ≤ ‖L‖ * ‖R‖ := by
  calc
    ‖tg L R P‖ ≤ ‖L‖ * ‖1 - P.starProjection‖ * ‖R‖ :=
      (norm_mul_le _ _).trans
        (mul_le_mul_of_nonneg_right (norm_mul_le _ _) (norm_nonneg R))
    _ ≤ ‖L‖ * 1 * ‖R‖ := mul_le_mul_of_nonneg_right
      (mul_le_mul_of_nonneg_left (complement_projection_norm_le P) (norm_nonneg L))
      (norm_nonneg R)
    _ = ‖L‖ * ‖R‖ := by ring


end LearnedTG
