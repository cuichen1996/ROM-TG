# Lean formalization

This project contains the current discrete subspace and perturbation results supporting the manuscript. All 16 proof modules are explicit build roots and are imported transitively by `LearnedTGAxiomAudit.lean`.

## Build

Install Lean through elan, then run inside this directory:

```bash
lake exe cache get
./verify.sh
```

`lean-toolchain` pins Lean 4.35.0. `lakefile.lean` and `lake-manifest.json` pin Mathlib to `56a005855ea17e0b0958f73ec801ae140b0a1679`. The build treats warnings as errors and disables implicit variables. Initial dependency/cache acquisition requires network access; the checked-in project contains no dependency cache or compiled proof artifacts.

`verify.sh` checks source/build/audit scope, builds all modules, and runs the axiom audit. It prints verification output to the terminal. Generated files stay under `.lake/`, which is ignored by Git. It does not read a sibling manuscript or create a manuscript annotation.

The axiom audit checks 218 theorem/definition declarations and their dependencies. It permits only Lean's standard `propext`, `Classical.choice`, and `Quot.sound`. No `sorry`, custom axiom, self-adjoint smoother assumption, or contraction assumption is used to establish the discrete TG perturbation results.

## Mathematical scope

See [PROOF_MAP.md](PROOF_MAP.md) for the manuscript-to-declaration correspondence and explicit limits. The energy inner product and isometric coordinate maps are provided as abstract structures; constructing these instances from a concrete SPD matrix is outside the current formalization. The project does not verify Python execution, roundoff, restart trajectories, nonlinear Picard convergence, or empirical iteration counts.
