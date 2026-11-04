module

public import LearnedTGWidths
public import Mathlib.Analysis.Normed.Operator.NNNorm

/-! Finite-snapshot ellipsoid width from actual covariance spectral data; no spectral-gap transfer assumption. -/

@[expose] public section

open scoped RealInnerProductSpace BigOperators

namespace LearnedTG

variable {V Z I : Type*}
  [NormedAddCommGroup V] [InnerProductSpace ℝ V] [CompleteSpace V]
  [NormedAddCommGroup Z] [InnerProductSpace ℝ Z] [CompleteSpace Z]

noncomputable def spectralCutSpace (b : HilbertBasis I ℝ V) (s : Finset I) : Submodule ℝ V := by
  classical
  exact Submodule.span ℝ (↑(s.image b) : Set V)

noncomputable def spectralCutIndex (s : Finset I) : ↑s ≃ Fin s.card :=
  (Fintype.equivFin ↑s).trans (finCongr (Fintype.card_coe s))

noncomputable def spectralCutBasis (b : HilbertBasis I ℝ V) (s : Finset I) :
    OrthonormalBasis (Fin s.card) ℝ (spectralCutSpace b s) := by
  classical
  exact (OrthonormalBasis.span b.orthonormal s).reindex (spectralCutIndex s)

omit [CompleteSpace V] in
theorem spectralCutBasis_apply (b : HilbertBasis I ℝ V) (s : Finset I) (i : Fin s.card) :
    (spectralCutBasis b s i : V) = b ((spectralCutIndex s).symm i).val := by
  classical
  have h := congrArg (fun x : spectralCutSpace b s => (x : V))
    ((OrthonormalBasis.span b.orthonormal s).reindex_apply (spectralCutIndex s) i)
  exact h.trans (OrthonormalBasis.span_apply b.orthonormal s ((spectralCutIndex s).symm i))

theorem covariance_right_singular_system (K : Z →L[ℝ] V) {k : ℕ}
    (S : Submodule ℝ V) (v : OrthonormalBasis (Fin k) ℝ S) (sigma : Fin k → ℝ)
    (hpos : ∀ i, 0 < sigma i)
    (heigen : ∀ i, covariance K (v i : V) = (sigma i) ^ 2 • (v i : V)) :
    Orthonormal ℝ (fun i => (sigma i)⁻¹ • K.adjoint (v i : V)) ∧
      ∀ i, K ((sigma i)⁻¹ • K.adjoint (v i : V)) = sigma i • (v i : V) := by
  have hinner (i j : Fin k) :
      ⟪K.adjoint (v i : V), K.adjoint (v j : V)⟫ =
        (sigma i) ^ 2 * ⟪(v i : V), (v j : V)⟫ := by
    rw [K.adjoint_inner_right]
    change ⟪covariance K (v i : V), (v j : V)⟫ = _
    rw [heigen, real_inner_smul_left]
  constructor
  · constructor
    · intro i
      have hnorm : ‖K.adjoint (v i : V)‖ = sigma i := by
        apply (sq_eq_sq₀ (norm_nonneg _) (hpos i).le).mp
        have hvnorm : ‖(v i : V)‖ = 1 := v.orthonormal.norm_eq_one i
        rw [← real_inner_self_eq_norm_sq, hinner, real_inner_self_eq_norm_sq,
          hvnorm, one_pow, mul_one]
      rw [norm_smul, Real.norm_eq_abs, abs_inv, abs_of_pos (hpos i), hnorm,
        inv_mul_cancel₀ (ne_of_gt (hpos i))]
    · intro i j hij
      have hvorth : ⟪(v i : V), (v j : V)⟫ = 0 := v.orthonormal.inner_eq_zero hij
      rw [real_inner_smul_left, real_inner_smul_right, hinner, hvorth]
      ring
  · intro i
    rw [map_smul]
    change (sigma i)⁻¹ • covariance K (v i : V) = _
    rw [heigen, smul_smul]
    congr 1
    field_simp

/-- The positive-tail n-width equality, with right singular vectors derived from K K*. -/
theorem snapshotWidth_eq_from_covariance (K : Z →L[ℝ] V) (n : ℕ)
    (Q S : Submodule ℝ V) [FiniteDimensional ℝ Q] [FiniteDimensional ℝ S]
    (hQdim : Module.finrank ℝ Q = n) (hSdim : Module.finrank ℝ S = n + 1)
    (v : OrthonormalBasis (Fin (n + 1)) ℝ S) (sigma : Fin (n + 1) → ℝ)
    (tail : ℝ) (htail : 0 < tail) (hmin : ∀ i, tail ≤ sigma i)
    (heigen : ∀ i, covariance K (v i : V) = (sigma i) ^ 2 • (v i : V))
    (hlow : ∀ x ∈ Qᗮ, ⟪covariance K x, x⟫ ≤ tail ^ 2 * ‖x‖ ^ 2) :
    snapshotWidth K n = tail := by
  obtain ⟨hu, hK⟩ := covariance_right_singular_system K S v sigma (fun i => htail.trans_le (hmin i)) heigen
  apply snapshotWidth_eq_of_singular_system K n Q S hQdim hSdim v _ hu sigma tail htail hmin hK
  simpa only [Real.sqrt_sq htail.le] using snapshot_tail_from_rayleigh K Q (tail ^ 2)
    (sq_nonneg _) hlow

noncomputable def worstBestApproximation (K : Z →L[ℝ] V) (W : Submodule ℝ V) : ℝ :=
  sSup ((fun z => ⨅ w : W, ‖K z - (w : V)‖) '' Metric.closedBall 0 1)

omit [CompleteSpace V] [CompleteSpace Z] in
theorem worstBestApproximation_eq_residual (K : Z →L[ℝ] V) (W : Submodule ℝ V)
    [W.HasOrthogonalProjection] :
    worstBestApproximation K W = ‖(1 - W.starProjection) ∘L K‖ := by
  unfold worstBestApproximation
  simp_rw [← best_approximation_residual K W]
  exact ContinuousLinearMap.sSup_unitClosedBall_eq_norm _

noncomputable def ellipsoidWidth (K : Z →L[ℝ] V) (n : ℕ) : ℝ :=
  sInf (Set.range fun W : ApproximationSpace (V := V) n => worstBestApproximation K W.space)

omit [CompleteSpace V] [CompleteSpace Z] in
/-- Equality with the inf-space / sup-input / inf-approximant definition in the paper. -/
theorem ellipsoidWidth_eq_snapshotWidth (K : Z →L[ℝ] V) (n : ℕ) :
    ellipsoidWidth K n = snapshotWidth K n := by
  unfold ellipsoidWidth snapshotWidth
  simp_rw [worstBestApproximation_eq_residual]

/-- Exact n-width for a finite leading cut of an actual covariance eigenbasis.
The next eigenvalue is specified by an attaining index; no right singular system is assumed. -/
theorem ellipsoidWidth_eq_of_spectral_cut (K : Z →L[ℝ] V)
    (b : HilbertBasis I ℝ V) (lambda : I → ℝ) (s : Finset I)
    (cut : ℝ) (hcut : 0 ≤ cut)
    (heigen : ∀ i, covariance K (b i) = lambda i • b i)
    (hhigh : ∀ i ∈ s, cut ≤ lambda i) (hlow : ∀ i, i ∉ s → lambda i ≤ cut)
    (j : I) (hj : j ∉ s) (hnext : lambda j = cut) :
    ellipsoidWidth K s.card = Real.sqrt cut := by
  classical
  let Q := spectralCutSpace b s
  let _ : FiniteDimensional ℝ Q := Module.Finite.of_basis (spectralCutBasis b s).toBasis
  have hQdim : Module.finrank ℝ Q = s.card :=
    (Module.finrank_eq_card_basis (spectralCutBasis b s).toBasis).trans (Fintype.card_fin _)
  have hupper : ‖(1 - Q.starProjection) ∘L K‖ ≤ Real.sqrt cut := by
    apply snapshot_tail_from_eigenbasis K b lambda s Q (Real.sqrt cut) (Real.sqrt_nonneg _)
      heigen
    · intro i hi
      exact Submodule.subset_span (Finset.mem_image.mpr ⟨i, hi, rfl⟩)
    · simpa only [Real.sq_sqrt hcut] using hlow
  rw [ellipsoidWidth_eq_snapshotWidth]
  by_cases hc0 : cut = 0
  · rw [hc0, Real.sqrt_zero] at hupper ⊢
    exact snapshotWidth_eq_zero K s.card Q hQdim (le_antisymm hupper (norm_nonneg _))
  · have hcpos : 0 < cut := lt_of_le_of_ne hcut (Ne.symm hc0)
    let u := insert j s
    let S := spectralCutSpace b u
    let v := spectralCutBasis b u
    let _ : FiniteDimensional ℝ S := Module.Finite.of_basis v.toBasis
    let sigma : Fin u.card → ℝ := fun i => Real.sqrt (lambda ((spectralCutIndex u).symm i).val)
    have hlarge (i : Fin u.card) : cut ≤ lambda ((spectralCutIndex u).symm i).val := by
      have hi := ((spectralCutIndex u).symm i).prop
      rcases Finset.mem_insert.mp hi with hi | hi
      · rw [hi, hnext]
      · exact hhigh _ hi
    have hpos (i : Fin u.card) : 0 < sigma i := Real.sqrt_pos.mpr (hcpos.trans_le (hlarge i))
    have hmin (i : Fin u.card) : Real.sqrt cut ≤ sigma i := Real.sqrt_le_sqrt (hlarge i)
    have hSdim : Module.finrank ℝ S = s.card + 1 := by
      have hd := (Module.finrank_eq_card_basis v.toBasis).trans (Fintype.card_fin u.card)
      simpa only [u, Finset.card_insert_of_notMem hj] using hd
    have heigS (i : Fin u.card) : covariance K (v i : V) = sigma i ^ 2 • (v i : V) := by
      change covariance K (spectralCutBasis b u i : V) = _
      rw [spectralCutBasis_apply, heigen]
      dsimp only [sigma]
      rw [Real.sq_sqrt (hcut.trans (hlarge i))]
    obtain ⟨hortho, hK⟩ := covariance_right_singular_system K S v sigma hpos heigS
    apply snapshotWidth_eq_of_upper_lower K s.card Q hQdim (Real.sqrt cut) hupper
    intro W
    apply snapshot_width_lower K S W.space (by rw [W.dimension, hSdim]; omega)
      (Real.sqrt cut) (Real.sqrt_nonneg _)
    exact finite_singular_ball_inclusion K S v _ hortho sigma (Real.sqrt cut)
      (Real.sqrt_pos.mpr hcpos) hmin hK

end LearnedTG
