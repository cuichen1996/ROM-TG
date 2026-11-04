# Release validation

Validation performed on 2026-11-02. This is a source/package validation record, not a new full benchmark or a timing measurement. Generated QA artifacts and build logs are kept outside this release directory.

## Environment

| Component | Observed local version |
| --- | --- |
| Python | 3.13.5 |
| PyTorch | 2.12.0 |
| NumPy | 2.3.4 |
| SciPy | 1.17.1 |
| Matplotlib | 3.10.9 |
| PyYAML | 6.0.3 |
| Lean | 4.35.0 |
| Mathlib | `56a005855ea17e0b0958f73ec801ae140b0a1679` |
| Julia renderer | Julia 1.12.4, dependencies pinned in `plotting/julia/Manifest.toml` |

The Python dependency ranges in `pyproject.toml` describe intended supported versions; this validation does not test every combination in those ranges.

## Checks

- All numerical modules import without a sibling original source tree; all Python and JSON source/config files parse.
- Seven numerical contract tests pass: actual FGMRES update caps, rejection of a false convergence candidate, mixed-boundary pressure matrix/RHS agreement, singular small-Gram norm identity, manufactured direct residual, refusal to overwrite existing run data, and float32 TB inference under a float64 solver default.
- On one canonical 63-grid field, ASM, POD, GenEO, TB-SVD, and Global-POD all pass independent static residual acceptance. The rank-15 preparation checks remain active.
- Static and moving-network training each complete a two-sample, one-epoch CPU check. The channelized adaptive-whitening Forchheimer recipe completes the corresponding from-scratch functional check. These tiny test models are not the reported final weights.
- A two-step 16-grid moving trajectory and all three coarse-space strategies pass the CPU sparse pressure acceptance check on their sampled operators. This does not replace the pending full moving-interface FGMRES experiment.
- One-step 23-grid learned and rebuilt-GenEO Picard checks pass independent linear acceptance. The one-step trajectories intentionally do not satisfy the nonlinear endpoint criterion; they are not reported as full nonlinear convergence.
- The full 63-grid q=200 illustration completes all three solves with independent acceptance. The history entry is exercised on a one-epoch/one-field check, including retained TG iteration caps.
- A 63-grid paired perturbation check agrees between the small-Gram norm and an independent matrix-free generalized-eigenvalue calculation.
- Python history/smoothing/timing renderers run from saved data. The pinned Julia field renderer reproduces the four-panel Forchheimer layout from the saved full-resolution field arrays; the PNG was visually inspected.
- A fresh Lean project, without reusing this project's compiled proof modules, builds all 16 proof modules and passes the axiom audit for 218 declarations. Third-party Mathlib dependency caches are reused.
- The Python wheel builds and is installed into an isolated temporary directory. CLI/import/numerical checks run outside the original repository. The source distribution includes configurations, documentation, final summaries, tests, and Lean sources.
- Original selected source files retain their source hashes, and all 16 copied proof files are byte-identical to the current original Lean sources. Private paths, server credentials, caches, logs, generated arrays, and model checkpoints are absent from the release tree.


