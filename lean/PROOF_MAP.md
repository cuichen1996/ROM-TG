# Proof correspondence and coverage

The labels below refer to the manuscript's section *Subspace approximation and perturbation analysis*. They are documentation labels only; no annotations are inserted in the manuscript.

## Assumptions

The real inner product represents the target grid's SPD energy metric. Left and right smoothing maps are arbitrary bounded linear operators: they need not be self-adjoint, normal, commuting, or contractive. Equal-rank bounds use two actual orthonormal `Fin n` bases and `n > 0`. Covariance self-adjointness follows from its finite-snapshot definition and is unrelated to smoother symmetry.

Krylov comparisons concern a fixed map, exact arithmetic, a shared nonzero cycle-initial residual, and the full ideal Krylov correction space. Norm upper bounds may exceed one. The proofs do not assume spectral clustering, a uniform spectral gap, or a continuum-limit Schwarz contraction.

## Main correspondence

| Manuscript item | Lean declarations | Coverage |
| --- | --- | --- |
| `eq:discrete-projector-loss` | `discrete_projector_loss_bound` | Derives the equal-rank projector-distance bound from the whitened loss. The exact largest-principal-angle norm identity is not formalized. |
| `eq:discrete-subspace-perturbation-bound` | `discrete_tg_loss_bound`, `discrete_tg_difference_norm` | Full product perturbation bound for the difference of two TG maps. |
| Regularization contribution | `regularized_tg_perturbation` | General correction-norm contribution; the closed prepared-Gram eigenvalue formula is not formalized. |
| `eq:operator-weighted-defect` | `discrete_weighted_defect_eq`, `operator_weighted_defect`, `operator_weighted_product_bound` | Exact weighted-defect identity and norm bounds, including the actual restriction to the defect range. |
| Zero defect range | `restrictedSmoothingNorm_zero` | Handles the zero-range case explicitly. |
| `eq:small-gram-perturbation` | No complete Lean counterpart | The square-root/singular-Gram finite-dimensional norm identity remains a mathematical proof in the manuscript. |
| Unequal-rank projection distance | `unequal_rank_projection_distance` | Constructs the orthogonal unit direction from the dimension difference. |
| One-sided reference inclusion | `one_sided_reference_capture`, `unequal_rank_whitened_capture` | Keeps the one-sided factor and the loss bound; does not assume the final inclusion estimate. |
| `eq:residual-operator-isometry` | `right_preconditioned_difference`, `right_preconditioned_difference_norm`, `preconditioner_difference_norm` | Sign, conjugacy, and norm identities for a supplied isometric equivalence. Concrete SPD/Riesz instance construction is not encoded. |
| Noncommutative telescoping | `noncommutative_power_difference` | Arbitrary ring; no commuting hypothesis. |
| `eq:polynomial-perturbation` | `power_difference_norm`, `polynomial_operator_difference`, `polynomial_perturbation` | Full double sum, including degree zero. Lean indexes the nonconstant coefficients as `a(k+1)`. |
| `eq:weighted-gmres-comparison` | `gmres_minimal_residual`, `gmres_candidate_polynomial`, `gmres_polynomial_comparison`, `gmres_polynomial_ratio` | Proves ideal residual minimality by projection onto the actual Krylov correction space, then constructs the normalized-polynomial candidate. |
| `eq:euclidean-krylov-comparison` | No complete Lean counterpart | The matrix spectral constants and concrete Euclidean conversion factor remain outside the formalization. |
| Numerical assessment | Numerical experiments | Observed counts, cross-grid performance, and finite-precision acceptance are not consequences of a Lean build. |

## Supporting modules

`Samples`, `POD`, and `Optimality` establish finite snapshot synthesis, covariance, and POD optimality. `Spectral`, `Widths`, and `Reference` establish spectral-tail/ellipsoid width facts and singular-vector constructions. `Subspaces`, `Angles`, and `Losses` establish projection, whitening, the average principal-angle loss identity, and weighted raw-snapshot loss properties. `Polynomial`, `Krylov`, `Preconditioning`, and `DiscretePerturbation` support the main section's operator bounds.

The current source set contains no continuous-network Gram coercivity or geometric Schwarz contraction proof. A successful build certifies the included statements with their explicit hypotheses; it does not mean that every matrix identity in the manuscript has been formalized.
