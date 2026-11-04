module

public import LearnedTGSamples

/-! Finite-snapshot POD optimality in the supplied energy inner product. -/

@[expose] public section

open scoped RealInnerProductSpace BigOperators

namespace LearnedTG

theorem spectral_weight_majorization {N : ℕ} (lambda weight : Fin N → ℝ)
    (s : Finset (Fin N)) (cut : ℝ) (hcut : 0 ≤ cut)
    (hweight0 : ∀ i, 0 ≤ weight i) (hweight1 : ∀ i, weight i ≤ 1)
    (hmass : ∑ i, weight i ≤ s.card)
    (hhigh : ∀ i ∈ s, cut ≤ lambda i) (hlow : ∀ i, i ∉ s → lambda i ≤ cut) :
    ∑ i, lambda i * weight i ≤ ∑ i ∈ s, lambda i := by
  classical
  have hterm (i : Fin N) : lambda i * weight i ≤
      cut * weight i + if i ∈ s then lambda i - cut else 0 := by
    by_cases hi : i ∈ s
    · rw [ite_eq_left hi]
      nlinarith [mul_nonneg (sub_nonneg.mpr (hhigh i hi)) (sub_nonneg.mpr (hweight1 i))]
    · rw [ite_eq_right hi]
      nlinarith [mul_nonneg (sub_nonneg.mpr (hlow i hi)) (hweight0 i)]
  have hsum := Finset.sum_le_sum fun i (_hi : i ∈ Finset.univ) => hterm i
  have hmass' := mul_le_mul_of_nonneg_left hmass hcut
  simp only [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_ite_mem,
    Finset.univ_inter, Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul] at hsum
  linarith

variable {D : Type*} [NormedAddCommGroup D] [InnerProductSpace ℝ D]
variable {N n q : ℕ}

theorem projection_weights_mass (b : OrthonormalBasis (Fin N) ℝ D)
    (P : Submodule ℝ D) [P.HasOrthogonalProjection] (p : OrthonormalBasis (Fin n) ℝ P) :
    ∑ i, ‖P.starProjection (b i)‖ ^ 2 = n := by
  simp_rw [projection_norm_sq_eq_sum P p]
  rw [Finset.sum_comm]
  have hpoint (j : Fin n) : ∑ i, ⟪(p j : D), b i⟫ ^ 2 = 1 := by
    rw [b.sum_sq_inner_left]
    have hp : ‖(p j : D)‖ = 1 := p.orthonormal.norm_eq_one j
    rw [hp, one_pow]
  simp only [hpoint, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]

theorem captured_energy_diagonal (A : D →L[ℝ] D) (b : OrthonormalBasis (Fin N) ℝ D)
    (lambda : Fin N → ℝ) (heigen : ∀ i, A (b i) = lambda i • b i)
    (P : Submodule ℝ D) [P.HasOrthogonalProjection] (p : OrthonormalBasis (Fin n) ℝ P) :
    ∑ j, ⟪A (p j : D), (p j : D)⟫ = ∑ i, lambda i * ‖P.starProjection (b i)‖ ^ 2 := by
  simp_rw [finite_quadratic_diagonal A b lambda heigen]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  rw [projection_norm_sq_eq_sum P p, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  rw [b.repr_apply_apply, real_inner_comm]

theorem captured_energy_prefix_maximal (A : D →L[ℝ] D)
    (b : OrthonormalBasis (Fin N) ℝ D) (lambda : Fin N → ℝ)
    (heigen : ∀ i, A (b i) = lambda i • b i) (hordered : Antitone lambda)
    (hn : n ≤ N) (hnonneg : ∀ i, 0 ≤ lambda i)
    (P : Submodule ℝ D) [P.HasOrthogonalProjection] (p : OrthonormalBasis (Fin n) ℝ P) :
    ∑ j, ⟪A (p j : D), (p j : D)⟫ ≤
      ∑ j, ⟪A (prefixBasis b hn j : D), (prefixBasis b hn j : D)⟫ := by
  classical
  let weight : Fin N → ℝ := fun i => ‖P.starProjection (b i)‖ ^ 2
  have hw0 (i : Fin N) : 0 ≤ weight i := sq_nonneg _
  have hw1 (i : Fin N) : weight i ≤ 1 := by
    have h := P.norm_starProjection_apply_le (b i)
    rw [b.orthonormal.norm_eq_one i] at h
    simpa only [one_pow] using (sq_le_sq₀ (norm_nonneg _) zero_le_one).mpr h
  have hright : ∑ j, ⟪A (prefixBasis b hn j : D), (prefixBasis b hn j : D)⟫ =
      ∑ i ∈ prefixSet N n, lambda i := by
    have hpoint (j : Fin n) :
        ⟪A (prefixBasis b hn j : D), (prefixBasis b hn j : D)⟫ =
          lambda (prefixIndexEquiv hn j).val := by
      rw [prefixBasis_apply, heigen, real_inner_smul_left, real_inner_self_eq_norm_sq,
        b.orthonormal.norm_eq_one, one_pow, mul_one]
      rfl
    simp_rw [hpoint]
    exact (Equiv.sum_comp (prefixIndexEquiv hn) fun i => lambda i.val).trans
      (Finset.sum_coe_sort _ _)
  rw [captured_energy_diagonal A b lambda heigen P p, hright]
  have hcard : (prefixSet N n).card = n := by
    have h := Fintype.card_congr (prefixIndexEquiv hn)
    simpa only [Fintype.card_fin, Fintype.card_coe] using h.symm
  have hmass : ∑ i, weight i ≤ (prefixSet N n).card := by
    rw [hcard]
    exact (projection_weights_mass b P p).le
  by_cases hnN : n < N
  · let k : Fin N := ⟨n, hnN⟩
    apply spectral_weight_majorization lambda weight (prefixSet N n) (lambda k)
      (hnonneg k) hw0 hw1 hmass
    · intro i hi
      apply hordered
      have hi' : (i : ℕ) < n := by simpa [prefixSet] using hi
      exact hi'.le
    · intro i hi
      apply hordered
      have hi' : ¬ (i : ℕ) < n := by simpa [prefixSet] using hi
      exact le_of_not_gt hi'
  · have hnN' : n = N := le_antisymm hn (le_of_not_gt hnN)
    have hs : prefixSet N n = Finset.univ := by
      ext i
      simp [prefixSet, hnN']
    rw [hs]
    exact Finset.sum_le_sum fun i _ => by
      simpa only [mul_one] using mul_le_mul_of_nonneg_left (hw1 i) (hnonneg i)

noncomputable def snapshotResidual (e : Fin q → D) (P : Submodule ℝ D)
    [P.HasOrthogonalProjection] : ℝ := ∑ i, ‖(1 - P.starProjection) (e i)‖ ^ 2

theorem snapshotResidual_eq_total_sub_capture (e : Fin q → D)
    (P : Submodule ℝ D) [P.HasOrthogonalProjection] (p : OrthonormalBasis (Fin n) ℝ P) :
    snapshotResidual e P = (∑ i, ‖e i‖ ^ 2) -
      ∑ j, ∑ i, ⟪e i, (p j : D)⟫ ^ 2 := by
  have hpoint (i : Fin q) : ‖(1 - P.starProjection) (e i)‖ ^ 2 =
      ‖e i‖ ^ 2 - ‖P.starProjection (e i)‖ ^ 2 := by
    have h := P.norm_sq_eq_add_norm_sq_starProjection (e i)
    rw [P.starProjection_orthogonal'] at h
    linarith
  unfold snapshotResidual
  simp_rw [hpoint, projection_norm_sq_eq_sum P p, real_inner_comm (e _)]
  rw [Finset.sum_sub_distrib, Finset.sum_comm]

theorem sampleCovariance_quadratic (e : Fin q → D) (x : D) :
    ⟪sampleCovariance e x, x⟫ = (q : ℝ)⁻¹ * ∑ i, ⟪e i, x⟫ ^ 2 := by
  rw [sampleCovariance_apply, real_inner_smul_left, sum_inner]
  congr 1
  simp only [real_inner_smul_left, pow_two]

theorem captured_sample_energy (e : Fin q → D) (P : Submodule ℝ D)
    (p : OrthonormalBasis (Fin n) ℝ P) :
    ∑ j, ⟪sampleCovariance e (p j : D), (p j : D)⟫ =
      (q : ℝ)⁻¹ * ∑ j, ∑ i, ⟪e i, (p j : D)⟫ ^ 2 := by
  simp only [sampleCovariance_quadratic, Finset.mul_sum]

variable [FiniteDimensional ℝ D]

/-- The empirical leading POD space minimizes the actual sum of squared snapshot errors
among all n-dimensional energy subspaces. -/
theorem empiricalPOD_minimizes_snapshotResidual (e : Fin q → D) (hq : 0 < q)
    (hn : n ≤ Module.finrank ℝ D) (P : Submodule ℝ D)
    (p : OrthonormalBasis (Fin n) ℝ P) :
    snapshotResidual e (empiricalPODSpace (sampleCovariance e) (sampleCovariance_positive e) n) ≤
      snapshotResidual e P := by
  let C := sampleCovariance e
  let hC := sampleCovariance_positive e
  let b := hC.isSymmetric.eigenvectorBasis rfl
  let lambda := hC.isSymmetric.eigenvalues rfl
  let Q := empiricalPODSpace C hC n
  let qb := empiricalPODBasis C hC n hn
  have heigen (i : Fin (Module.finrank ℝ D)) : C (b i) = lambda i • b i :=
    hC.isSymmetric.apply_eigenvectorBasis rfl i
  have h := captured_energy_prefix_maximal C b lambda heigen
    (hC.isSymmetric.eigenvalues_antitone rfl) hn (hC.toLinearMap.nonneg_eigenvalues rfl) P p
  change (∑ j, ⟪C (p j : D), (p j : D)⟫) ≤ ∑ j, ⟪C (qb j : D), (qb j : D)⟫ at h
  rw [captured_sample_energy e P p, captured_sample_energy e Q qb] at h
  have hinv : 0 < (q : ℝ)⁻¹ := inv_pos.mpr (Nat.cast_pos.mpr hq)
  have hcapture : (∑ j, ∑ i, ⟪e i, (p j : D)⟫ ^ 2) ≤
      ∑ j, ∑ i, ⟪e i, (qb j : D)⟫ ^ 2 := by nlinarith
  rw [snapshotResidual_eq_total_sub_capture e Q qb, snapshotResidual_eq_total_sub_capture e P p]
  linarith

/-- The attained POD objective equals the tail sum of normalized empirical eigenvalues.
For the unnormalized snapshot map these are sigma_j^2 / q. -/
theorem empiricalPOD_objective_eq_spectral_tail (e : Fin q → D)
    (hn : n ≤ Module.finrank ℝ D) :
    (q : ℝ)⁻¹ * snapshotResidual e
      (empiricalPODSpace (sampleCovariance e) (sampleCovariance_positive e) n) =
      ∑ i ∈ Finset.univ \ prefixSet (Module.finrank ℝ D) n,
        (sampleCovariance_positive e).isSymmetric.eigenvalues rfl i := by
  classical
  let C := sampleCovariance e
  let hC := sampleCovariance_positive e
  let b := hC.isSymmetric.eigenvectorBasis rfl
  let lambda := hC.isSymmetric.eigenvalues rfl
  let Q := empiricalPODSpace C hC n
  let qb := empiricalPODBasis C hC n hn
  have heigen (i : Fin (Module.finrank ℝ D)) : C (b i) = lambda i • b i :=
    hC.isSymmetric.apply_eigenvectorBasis rfl i
  have hpoint (i : Fin (Module.finrank ℝ D)) : ⟪C (b i), b i⟫ = lambda i := by
    rw [heigen, real_inner_smul_left, real_inner_self_eq_norm_sq,
      b.orthonormal.norm_eq_one i, one_pow, mul_one]
  have htotal : (q : ℝ)⁻¹ * (∑ i, ‖e i‖ ^ 2) = ∑ i, lambda i := by
    have ht : (∑ j, ⟪C (b j), b j⟫) = (q : ℝ)⁻¹ * (∑ i, ‖e i‖ ^ 2) := by
      change (∑ j, ⟪sampleCovariance e (b j), b j⟫) = _
      simp_rw [sampleCovariance_quadratic]
      rw [← Finset.mul_sum, Finset.sum_comm]
      simp_rw [b.sum_sq_inner_left]
    simpa only [hpoint] using ht.symm
  have hprefix : (∑ j, ⟪C (qb j : D), (qb j : D)⟫) =
      ∑ i ∈ prefixSet (Module.finrank ℝ D) n, lambda i := by
    have hp (j : Fin n) : ⟪C (qb j : D), (qb j : D)⟫ =
        lambda (prefixIndexEquiv hn j).val := by
      change ⟪C (prefixBasis b hn j : D), (prefixBasis b hn j : D)⟫ = _
      rw [prefixBasis_apply, hpoint]
      rfl
    simp_rw [hp]
    exact (Equiv.sum_comp (prefixIndexEquiv hn) fun i => lambda i.val).trans
      (Finset.sum_coe_sort _ _)
  have hcapture := captured_sample_energy e Q qb
  rw [snapshotResidual_eq_total_sub_capture e Q qb, mul_sub, htotal, ← hcapture, hprefix]
  change (∑ i, lambda i) - (∑ i ∈ prefixSet (Module.finrank ℝ D) n, lambda i) =
    ∑ i ∈ Finset.univ \ prefixSet (Module.finrank ℝ D) n, lambda i
  have htail : (∑ i ∈ Finset.univ \ prefixSet (Module.finrank ℝ D) n, lambda i) +
      (∑ i ∈ prefixSet (Module.finrank ℝ D) n, lambda i) = ∑ i, lambda i :=
    Finset.sum_sdiff (Finset.subset_univ _)
  linarith

end LearnedTG
