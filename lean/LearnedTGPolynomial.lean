module

public import LearnedTGCore
public import Mathlib.Analysis.Normed.Algebra.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic.NoncommRing

/-! Finite-degree perturbation for arbitrary fixed, possibly noncommuting
operators. Norm bounds are finite radii, never contraction assumptions. -/

@[expose] public section

open scoped BigOperators

namespace LearnedTG

noncomputable def powerPerturbationFactor (M N : ℝ) (k : ℕ) : ℝ :=
  ∑ i ∈ Finset.range k, M ^ (k - 1 - i) * N ^ i

noncomputable def polynomialPerturbationFactor (a : ℕ → ℝ) (M N : ℝ) (m : ℕ) : ℝ :=
  ∑ k ∈ Finset.range m, |a (k + 1)| * powerPerturbationFactor M N (k + 1)

noncomputable def polynomialOperator {A : Type*} [NormedRing A] [NormedAlgebra ℝ A]
    (a : ℕ → ℝ) (T : A) (m : ℕ) : A :=
  ∑ k ∈ Finset.range (m + 1), a k • T ^ k

theorem noncommutative_power_difference {A : Type*} [Ring A] (T U : A) (k : ℕ) :
    T ^ k - U ^ k = ∑ i ∈ Finset.range k, T ^ (k - 1 - i) * (T - U) * U ^ i := by
  induction k with
  | zero => simp
  | succ k ih =>
    calc
      T ^ (k + 1) - U ^ (k + 1) =
          (T ^ k - U ^ k) * U + T ^ k * (T - U) := by
        simp only [pow_succ]
        noncomm_ring
      _ = (∑ i ∈ Finset.range k, T ^ (k - 1 - i) * (T - U) * U ^ i) * U +
          T ^ k * (T - U) := by rw [ih]
      _ = _ := by
        rw [Finset.sum_range_succ', Finset.sum_mul]
        simp only [Nat.add_sub_cancel_right, Nat.sub_zero, pow_zero, mul_one]
        congr 1
        apply Finset.sum_congr rfl
        intro i _hi
        have hexponent : k - (i + 1) = k - 1 - i := by omega
        rw [hexponent, pow_succ]
        simp only [mul_assoc]

theorem power_difference_norm {A : Type*} [NormedRing A] [NormOneClass A]
    (T U : A) (M N : ℝ) (hT : ‖T‖ ≤ M) (hU : ‖U‖ ≤ N) (k : ℕ) :
    ‖T ^ k - U ^ k‖ ≤ ‖T - U‖ * powerPerturbationFactor M N k := by
  have hM : 0 ≤ M := (norm_nonneg T).trans hT
  rw [noncommutative_power_difference]
  calc
    _ ≤ ∑ i ∈ Finset.range k, ‖T ^ (k - 1 - i) * (T - U) * U ^ i‖ := norm_sum_le _ _
    _ ≤ ∑ i ∈ Finset.range k, M ^ (k - 1 - i) * ‖T - U‖ * N ^ i := by
      apply Finset.sum_le_sum
      intro i _hi
      have ha := (norm_pow_le T (k - 1 - i)).trans
        (pow_le_pow_left₀ (norm_nonneg T) hT (k - 1 - i))
      have hb := (norm_pow_le U i).trans (pow_le_pow_left₀ (norm_nonneg U) hU i)
      calc
        _ ≤ (‖T ^ (k - 1 - i)‖ * ‖T - U‖) * ‖U ^ i‖ :=
          (norm_mul_le _ _).trans
            (mul_le_mul_of_nonneg_right (norm_mul_le _ _) (norm_nonneg _))
        _ ≤ _ := mul_le_mul
          (mul_le_mul_of_nonneg_right ha (norm_nonneg _)) hb (norm_nonneg _)
          (mul_nonneg (pow_nonneg hM _) (norm_nonneg _))
    _ = _ := by
      unfold powerPerturbationFactor
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro i _hi
      ring

theorem polynomial_operator_difference {A : Type*} [NormedRing A] [NormedAlgebra ℝ A]
    (a : ℕ → ℝ) (T U : A) (m : ℕ) :
    polynomialOperator a T m - polynomialOperator a U m =
      ∑ k ∈ Finset.range m, a (k + 1) • (T ^ (k + 1) - U ^ (k + 1)) := by
  unfold polynomialOperator
  rw [← Finset.sum_sub_distrib]
  simp_rw [← smul_sub]
  rw [Finset.sum_range_succ']
  simp only [pow_zero, sub_self, smul_zero, add_zero]

/-- The paper's finite-step polynomial theorem, without commutativity,
normality, spectral separation, or any radius being smaller than one. -/
theorem polynomial_perturbation {A : Type*} [NormedRing A] [NormOneClass A]
    [NormedAlgebra ℝ A] (a : ℕ → ℝ) (T U : A) (M N : ℝ)
    (hT : ‖T‖ ≤ M) (hU : ‖U‖ ≤ N) (m : ℕ) :
    ‖polynomialOperator a T m - polynomialOperator a U m‖ ≤
      ‖T - U‖ * polynomialPerturbationFactor a M N m := by
  rw [polynomial_operator_difference]
  calc
    _ ≤ ∑ k ∈ Finset.range m, ‖a (k + 1) • (T ^ (k + 1) - U ^ (k + 1))‖ :=
      norm_sum_le _ _
    _ ≤ ∑ k ∈ Finset.range m,
        |a (k + 1)| * (‖T - U‖ * powerPerturbationFactor M N (k + 1)) := by
      apply Finset.sum_le_sum
      intro k _hk
      rw [norm_smul, Real.norm_eq_abs]
      exact mul_le_mul_of_nonneg_left (power_difference_norm T U M N hT hU (k + 1))
        (abs_nonneg _)
    _ = _ := by
      unfold polynomialPerturbationFactor
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro k _hk
      ring

end LearnedTG
