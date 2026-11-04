module

public import Mathlib.Analysis.Normed.Ring.Basic
public import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Linarith

/-! Fixed-operator algebra. No continuous reconstruction, covariance event,
offline/online coverage or stationary-convergence premise is introduced. -/

@[expose] public section

namespace LearnedTG

variable {A : Type*} [NormedRing A]

/-- The algebraic decomposition underlying the learned-TG perturbation bound. -/
theorem tg_operator_perturbation (L R P Q : A) :
    ‖L * (1 - P) * R‖ ≤
      ‖L‖ * (‖(1 - Q) * R‖ + ‖(Q - P) * R‖) := by
  have hdecomp : (1 - P) * R = (1 - Q) * R + (Q - P) * R := by
    calc
      (1 - P) * R = R - P * R := by rw [sub_mul, one_mul]
      _ = (R - Q * R) + (Q * R - P * R) := by abel
      _ = (1 - Q) * R + (Q - P) * R := by
        rw [sub_mul, sub_mul, one_mul]
  calc
    ‖L * (1 - P) * R‖ = ‖L * ((1 - P) * R)‖ := by rw [mul_assoc]
    _ ≤ ‖L‖ * ‖(1 - P) * R‖ := norm_mul_le _ _
    _ = ‖L‖ * ‖(1 - Q) * R + (Q - P) * R‖ := by rw [hdecomp]
    _ ≤ ‖L‖ * (‖(1 - Q) * R‖ + ‖(Q - P) * R‖) :=
      mul_le_mul_of_nonneg_left (norm_add_le _ _) (norm_nonneg L)


end LearnedTG
