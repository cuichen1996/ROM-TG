import Lake

open Lake DSL

package learnedtg where
  version := v!"0.1.0"
  leanOptions := #[⟨`autoImplicit, false⟩, ⟨`warningAsError, true⟩]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @
  "56a005855ea17e0b0958f73ec801ae140b0a1679"

@[default_target]
lean_lib LearnedTG where
  roots := #[`LearnedTGCore, `LearnedTGOperators, `LearnedTGSubspaces,
    `LearnedTGSpectral, `LearnedTGWidths, `LearnedTGPOD, `LearnedTGSamples,
    `LearnedTGReference, `LearnedTGOptimality, `LearnedTGLosses,
    `LearnedTGAngles, `LearnedTGPolynomial, `LearnedTGKrylov,
    `LearnedTGPreconditioning, `LearnedTGDiscretePerturbation, `LearnedTGAxiomAudit]
