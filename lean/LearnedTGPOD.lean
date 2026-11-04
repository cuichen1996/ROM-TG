module

public import LearnedTGSpectral
public import Mathlib.Analysis.InnerProductSpace.Positive

/-! Ordered finite-dimensional POD spaces, constructed from the spectral theorem. -/

@[expose] public section

open scoped RealInnerProductSpace BigOperators

namespace LearnedTG

variable {D : Type*} [NormedAddCommGroup D] [InnerProductSpace ℝ D]
variable {N n : ℕ}

noncomputable def prefixSet (N n : ℕ) : Finset (Fin N) :=
  Finset.univ.filter fun i => (i : ℕ) < n

noncomputable def prefixSpace (b : OrthonormalBasis (Fin N) ℝ D) (n : ℕ) : Submodule ℝ D := by
  classical
  exact Submodule.span ℝ ((prefixSet N n).image b : Set D)

noncomputable def prefixIndexEquiv (hn : n ≤ N) : Fin n ≃ ↑(prefixSet N n) where
  toFun i := ⟨⟨i, lt_of_lt_of_le i.isLt hn⟩, by simp [prefixSet, i.isLt]⟩
  invFun i := ⟨i.val.val, by simpa only [prefixSet, Finset.mem_filter, Finset.mem_univ, true_and] using i.prop⟩
  left_inv i := rfl
  right_inv i := by apply Subtype.ext; rfl

noncomputable def prefixBasis (b : OrthonormalBasis (Fin N) ℝ D) (hn : n ≤ N) :
    OrthonormalBasis (Fin n) ℝ (prefixSpace b n) := by
  classical
  exact (OrthonormalBasis.span b.orthonormal (prefixSet N n)).reindex (prefixIndexEquiv hn).symm

theorem prefixBasis_apply (b : OrthonormalBasis (Fin N) ℝ D) (hn : n ≤ N) (i : Fin n) :
    (prefixBasis b hn i : D) = b ⟨i, lt_of_lt_of_le i.isLt hn⟩ := by
  classical
  have h := congrArg (fun x : prefixSpace b n => (x : D))
    ((OrthonormalBasis.span b.orthonormal (prefixSet N n)).reindex_apply
      (prefixIndexEquiv hn).symm i)
  exact h.trans (OrthonormalBasis.span_apply b.orthonormal (prefixSet N n) (prefixIndexEquiv hn i))

theorem prefixSpace_finrank (b : OrthonormalBasis (Fin N) ℝ D) (hn : n ≤ N) :
    Module.finrank ℝ (prefixSpace b n) = n := by
  exact (Module.finrank_eq_card_basis (prefixBasis b hn).toBasis).trans (Fintype.card_fin n)

instance prefixSpace_finite (b : OrthonormalBasis (Fin N) ℝ D) (n : ℕ) :
    FiniteDimensional ℝ (prefixSpace b n) := by
  classical
  exact FiniteDimensional.span_of_finite ℝ (Finset.finite_toSet _)

theorem prefixSpace_contains (b : OrthonormalBasis (Fin N) ℝ D) (i : Fin N) (hi : (i : ℕ) < n) :
    b i ∈ prefixSpace b n := by
  classical
  apply Submodule.subset_span
  exact Finset.mem_image.mpr ⟨i, by simp [prefixSet, hi], rfl⟩

theorem finite_quadratic_diagonal (A : D →L[ℝ] D) (b : OrthonormalBasis (Fin N) ℝ D)
    (lambda : Fin N → ℝ) (heigen : ∀ i, A (b i) = lambda i • b i) (x : D) :
    ⟪A x, x⟫ = ∑ i, lambda i * (b.repr x i) ^ 2 := by
  have hAx : A x = ∑ i, b.repr x i • A (b i) := by
    calc
      _ = A (∑ i, b.repr x i • b i) := congrArg A (b.sum_repr x).symm
      _ = _ := by simp only [map_sum, map_smul]
  rw [hAx]
  simp only [heigen, smul_smul, sum_inner, real_inner_smul_left]
  apply Finset.sum_congr rfl
  intro i _
  rw [← b.repr_apply_apply x i]
  ring

theorem prefix_invariant (A : D →L[ℝ] D) (b : OrthonormalBasis (Fin N) ℝ D)
    (lambda : Fin N → ℝ) (heigen : ∀ i, A (b i) = lambda i • b i)
    (hn : n ≤ N) : ∀ x ∈ prefixSpace b n, A x ∈ prefixSpace b n := by
  intro x hx
  let q := prefixBasis b hn
  have hexp : ∑ i, q.repr ⟨x, hx⟩ i • (q i : D) = x := by
    simpa only [Submodule.coe_sum, Submodule.coe_smul] using
      congrArg (fun y : prefixSpace b n => (y : D)) (q.sum_repr ⟨x, hx⟩)
  have hAx : A x = ∑ i, q.repr ⟨x, hx⟩ i • A (q i : D) := by
    conv_lhs => rw [← hexp]
    simp only [map_sum, map_smul]
  rw [hAx]
  apply Submodule.sum_mem
  intro i _
  apply Submodule.smul_mem
  change A (prefixBasis b hn i : D) ∈ _
  rw [prefixBasis_apply, heigen]
  exact Submodule.smul_mem _ _ (prefixSpace_contains b _ i.isLt)

theorem prefix_tail_rayleigh (A : D →L[ℝ] D) (hA : A.IsSymmetric)
    (b : OrthonormalBasis (Fin N) ℝ D) (lambda : Fin N → ℝ)
    (heigen : ∀ i, A (b i) = lambda i • b i) (hordered : Antitone lambda)
    (k : Fin N) (hlambda : 0 ≤ lambda k) (x : D) (hx : x ∈ (prefixSpace b k.val)ᗮ) :
    ⟪A x, x⟫ ≤ lambda k * ‖x‖ ^ 2 := by
  let _ : FiniteDimensional ℝ D := Module.Finite.of_basis b.toBasis
  have htail : ∀ i : Fin N, i ∉ prefixSet N k.val → lambda i ≤ (Real.sqrt (lambda k)) ^ 2 := by
    intro i hi
    rw [Real.sq_sqrt hlambda]
    apply hordered
    have hnot : ¬ (i : ℕ) < (k : ℕ) := by simpa [prefixSet] using hi
    exact le_of_not_gt hnot
  have hzero (i : Fin N) (hi : i ∈ prefixSet N k.val) : ⟪b.toHilbertBasis i, x⟫ = 0 := by
    have hi' : (i : ℕ) < (k : ℕ) := by simpa [prefixSet] using hi
    rw [congrFun b.coe_toHilbertBasis i]
    exact (prefixSpace b k.val).inner_right_of_mem_orthogonal (prefixSpace_contains b i hi') hx
  have h := spectral_quadratic_tail A hA b.toHilbertBasis lambda (prefixSet N k.val)
    (Real.sqrt (lambda k)) (by simpa using heigen) htail x hzero
  simpa only [Real.sq_sqrt hlambda] using h

section Empirical

variable [FiniteDimensional ℝ D]

noncomputable def empiricalPODSpace (C : D →L[ℝ] D) (hC : C.IsPositive) (n : ℕ) : Submodule ℝ D :=
  prefixSpace (hC.isSymmetric.eigenvectorBasis rfl) n

instance empiricalPODSpace_finite (C : D →L[ℝ] D) (hC : C.IsPositive) (n : ℕ) :
    FiniteDimensional ℝ (empiricalPODSpace C hC n) := prefixSpace_finite _ _

noncomputable def empiricalPODBasis (C : D →L[ℝ] D) (hC : C.IsPositive)
    (n : ℕ) (hn : n ≤ Module.finrank ℝ D) :
    OrthonormalBasis (Fin n) ℝ (empiricalPODSpace C hC n) := prefixBasis _ hn


end Empirical

end LearnedTG
