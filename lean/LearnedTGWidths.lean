module

public import LearnedTGSpectral

/-! Lower bounds and attained n-widths from finite singular systems. -/

@[expose] public section

open scoped RealInnerProductSpace BigOperators

namespace LearnedTG

variable {V Z : Type*}
  [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  [NormedAddCommGroup Z] [InnerProductSpace ℝ Z]

theorem orthonormal_sum_norm_sq {k : ℕ} (u : Fin k → Z) (hu : Orthonormal ℝ u)
    (a : Fin k → ℝ) : ‖∑ i, a i • u i‖ ^ 2 = ∑ i, a i ^ 2 := by
  rw [← real_inner_self_eq_norm_sq, sum_inner]
  simp only [real_inner_smul_left, hu.inner_right_fintype, pow_two]

theorem finite_singular_ball_inclusion (K : Z →L[ℝ] V)
    (S : Submodule ℝ V) {k : ℕ} (v : OrthonormalBasis (Fin k) ℝ S)
    (u : Fin k → Z) (hu : Orthonormal ℝ u)
    (singular : Fin k → ℝ) (sigma : ℝ) (hsigma : 0 < sigma)
    (hmin : ∀ i, sigma ≤ singular i)
    (hK : ∀ i, K (u i) = singular i • (v i : V))
    (y : S) (hy : ‖y‖ ≤ 1) : ∃ z : Z, ‖z‖ ≤ 1 ∧ K z = sigma • (y : V) := by
  let a : Fin k → ℝ := fun i => (sigma / singular i) * v.repr y i
  let z : Z := ∑ i, a i • u i
  have hpos (i : Fin k) : 0 < singular i := hsigma.trans_le (hmin i)
  have hratio0 (i : Fin k) : 0 ≤ sigma / singular i := div_nonneg hsigma.le (hpos i).le
  have hratio1 (i : Fin k) : sigma / singular i ≤ 1 := (div_le_one (hpos i)).mpr (hmin i)
  have hnorm : ‖z‖ ^ 2 ≤ ‖y‖ ^ 2 := by
    rw [orthonormal_sum_norm_sq u hu a]
    have hcoeff : ∑ i, (v.repr y i) ^ 2 = ‖y‖ ^ 2 := by
      rw [← EuclideanSpace.real_norm_sq_eq, v.repr.norm_map]
    rw [← hcoeff]
    apply Finset.sum_le_sum
    intro i _
    dsimp [a]
    rw [mul_pow]
    exact (mul_le_mul_of_nonneg_right
      ((sq_le_sq₀ (hratio0 i) zero_le_one).mpr (hratio1 i)) (sq_nonneg _)).trans_eq (by simp)
  refine ⟨z, ?_, ?_⟩
  · apply (sq_le_sq₀ (norm_nonneg _) zero_le_one).mp
    exact hnorm.trans ((sq_le_sq₀ (norm_nonneg _) zero_le_one).mpr hy)
  · have hcoeff (i : Fin k) : a i * singular i = sigma * v.repr y i := by
      dsimp [a]
      field_simp [ne_of_gt (hpos i)]
    calc
      K z = ∑ i, (a i * singular i) • (v i : V) := by
        simp only [z, map_sum, map_smul, hK, smul_smul]
      _ = sigma • ∑ i, v.repr y i • (v i : V) := by
        simp only [hcoeff, Finset.smul_sum, smul_smul]
      _ = sigma • (y : V) := by
        congr 1
        simpa only [Submodule.coe_sum, Submodule.coe_smul] using
          congrArg (fun x : S => (x : V)) (v.sum_repr y)

theorem snapshot_width_lower (K : Z →L[ℝ] V)
    (S W : Submodule ℝ V) [FiniteDimensional ℝ S] [FiniteDimensional ℝ W]
    (hdim : Module.finrank ℝ W < Module.finrank ℝ S)
    (sigma : ℝ) (hsigma : 0 ≤ sigma)
    (hball : ∀ y : S, ‖y‖ ≤ 1 → ∃ z : Z, ‖z‖ ≤ 1 ∧ K z = sigma • (y : V)) :
    sigma ≤ ‖(1 - W.starProjection) ∘L K‖ := by
  obtain ⟨y, hyS, hyW, hynorm⟩ := exists_unit_mem_orthogonal S W hdim
  obtain ⟨z, hznorm, hKz⟩ := hball ⟨y, hyS⟩ hynorm.le
  have hyproj : W.starProjection y = 0 := W.starProjection_apply_eq_zero_iff.mpr hyW
  have hval : ‖((1 - W.starProjection) ∘L K) z‖ = sigma := by
    rw [ContinuousLinearMap.comp_apply, hKz, map_smul]
    simp only [sub_apply, one_apply_eq_self, hyproj, sub_zero, norm_smul, Real.norm_eq_abs,
      abs_of_nonneg hsigma, hynorm, mul_one]
  calc
    sigma = ‖((1 - W.starProjection) ∘L K) z‖ := hval.symm
    _ ≤ ‖(1 - W.starProjection) ∘L K‖ * ‖z‖ := ContinuousLinearMap.le_opNorm _ _
    _ ≤ ‖(1 - W.starProjection) ∘L K‖ := by
      simpa only [mul_one] using mul_le_mul_of_nonneg_left hznorm (norm_nonneg _)

structure ApproximationSpace (n : ℕ) where
  space : Submodule ℝ V
  finite : FiniteDimensional ℝ space
  dimension : Module.finrank ℝ space = n

attribute [instance] ApproximationSpace.finite

noncomputable def snapshotWidth (K : Z →L[ℝ] V) (n : ℕ) : ℝ :=
  sInf (Set.range fun W : ApproximationSpace (V := V) n => ‖(1 - W.space.starProjection) ∘L K‖)

/-- The best-approximation distance is exactly the projection residual used in snapshotWidth. -/
theorem best_approximation_residual (K : Z →L[ℝ] V) (W : Submodule ℝ V)
    [W.HasOrthogonalProjection] (z : Z) :
    ‖((1 - W.starProjection) ∘L K) z‖ = ⨅ w : W, ‖K z - w‖ := by
  exact Submodule.starProjection_minimal (K z)

theorem snapshotWidth_eq_of_upper_lower (K : Z →L[ℝ] V) (n : ℕ)
    (Q : Submodule ℝ V) [FiniteDimensional ℝ Q] (hQdim : Module.finrank ℝ Q = n)
    (sigma : ℝ) (hupper : ‖(1 - Q.starProjection) ∘L K‖ ≤ sigma)
    (hlower : ∀ W : ApproximationSpace (V := V) n,
      sigma ≤ ‖(1 - W.space.starProjection) ∘L K‖) :
    snapshotWidth K n = sigma := by
  let WQ : ApproximationSpace (V := V) n := ⟨Q, inferInstance, hQdim⟩
  have hnorm : ‖(1 - Q.starProjection) ∘L K‖ = sigma := le_antisymm hupper (hlower WQ)
  apply IsLeast.csInf_eq
  constructor
  · exact ⟨WQ, hnorm⟩
  · intro a ha
    obtain ⟨W, rfl⟩ := ha
    exact hlower W

theorem snapshotWidth_eq_of_singular_system (K : Z →L[ℝ] V) (n : ℕ)
    (Q S : Submodule ℝ V) [FiniteDimensional ℝ Q] [FiniteDimensional ℝ S]
    (hQdim : Module.finrank ℝ Q = n) (hSdim : Module.finrank ℝ S = n + 1)
    (v : OrthonormalBasis (Fin (n + 1)) ℝ S)
    (u : Fin (n + 1) → Z) (hu : Orthonormal ℝ u)
    (singular : Fin (n + 1) → ℝ) (sigma : ℝ) (hsigma : 0 < sigma)
    (hmin : ∀ i, sigma ≤ singular i)
    (hK : ∀ i, K (u i) = singular i • (v i : V))
    (hupper : ‖(1 - Q.starProjection) ∘L K‖ ≤ sigma) : snapshotWidth K n = sigma := by
  apply snapshotWidth_eq_of_upper_lower K n Q hQdim sigma hupper
  intro W
  apply snapshot_width_lower K S W.space (by rw [W.dimension, hSdim]; omega) sigma hsigma.le
  exact finite_singular_ball_inclusion K S v u hu singular sigma hsigma hmin hK

theorem snapshotWidth_eq_zero (K : Z →L[ℝ] V) (n : ℕ)
    (Q : Submodule ℝ V) [FiniteDimensional ℝ Q] (hQdim : Module.finrank ℝ Q = n)
    (hupper : ‖(1 - Q.starProjection) ∘L K‖ = 0) : snapshotWidth K n = 0 := by
  exact snapshotWidth_eq_of_upper_lower K n Q hQdim 0 hupper.le (fun _ => norm_nonneg _)

end LearnedTG
