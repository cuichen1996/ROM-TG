# Reproduction recipes

These commands assume installation from the repository root. Select a device explicitly, keep each output directory empty, and retain all failed cases. The examples below use CPU by default; add `--device cuda:0` on a suitable compute server. The full grids and training datasets are substantially larger than the quick-start examples.

## Numerical contract

Static evaluation uses float64 FVM/Krylov arithmetic, float32 network weights, averaged ASM with damping 0.8, one pre- and one postsmoothing step, exact local block-tridiagonal Cholesky with shift `1e-6`, and coarse regularization `1e-10`. Right FGMRES uses two-pass Euclidean Arnoldi, restart 50, and at most 200 completed updates. Acceptance requires both independent CPU sparse and double-double stored-stencil relative residuals at most `1e-6`; the initial residual check is not an update. Standalone TG uses the same 200-sweep cap. Counts and construction failures are retained separately.

The canonical static coefficient fields are generated on CPU. Field `sid` on grid `n` uses seed `1234 + 1000*n + sid`; geometries therefore differ across grids. All methods within one case share the exact field, RHS, and zero initial guess. Global-POD uses the separate 50-parameter pool with seeds 91234–91283, 15 probes per parameter, local rank 15, and final compressed rank 30. Static GenEO uses threshold 0.30 and a cap of 256 local modes; a threshold that is not bracketed is a construction failure.

## Model artifacts

Place final weights under `artifacts/` and compare their SHA-256 hashes with `reference_data/artifacts.json`. This directory contains no weights or download URLs. The raw/white model configurations describe the reported architecture and dataset/loss settings. A final model needs its matching configuration; a TB checkpoint needs its matching selection metadata.

Forchheimer's stored recipe starts from pretrained initialization. Its initialization artifact is not bundled. `train --config configs/forchheimer.json` therefore requires an explicit `--initialize-from PATH`, or an explicit `--from-scratch` to train a new model. A new training run does not recreate a frozen reported weight by implication.

## Static experiments

```bash
learned-tg train --config configs/train_static_white.json --output outputs/train_white
learned-tg train --config configs/train_static_raw.json --output outputs/train_raw

learned-tg static --config configs/static.json --methods all --include-tg \
  --checkpoint artifacts/static_white.pth --model-config configs/train_static_white.json \
  --raw-checkpoint artifacts/static_raw.pth --raw-model-config configs/train_static_raw.json \
  --tb-checkpoint artifacts/tb_min_train.pt --tb-metadata reference_data/tb_metadata.json \
  --output outputs/static

learned-tg summarize --kind static --input outputs/static/summary.json \
  --output outputs/static/table.csv
```

The full comparison has five grids, eight fields each, seven FGMRES methods, and two standalone learned TG cases per field: 360 configurations. The supplied final dataset records 351 accepted solves and nine capped failures. Summary means and population standard deviations use accepted solves; failures remain explicit.

Global-POD is constructed once per target grid and reused within the run. An existing cache can be supplied with `--global-cache 'artifacts/global_n{n}.pt'`; its tensor layout is `{'bases': {'global-POD-n30': tensor}}`. If regenerating a TB baseline, use its newly generated metadata with its newly trained weight:

```bash
learned-tg tb-train --config configs/tb_training.json --output outputs/tb
```

The TB baseline trains an independent solution DeepONet on certified labels, extracts its coordinate-only trunk, applies the retained rank-15 SVD selection and QR truncation, and uses the same TG/FGMRES evaluation path. The supplied metadata is tied to the reported minimum-training-loss checkpoint, not to an arbitrary retrained model.

## Snapshot studies and Figure 4

```bash
learned-tg study --config configs/static.json --kind q200 --output outputs/q200
learned-tg study --config configs/static.json --kind ablation --output outputs/ablation

learned-tg history --config configs/static.json \
  --white-config configs/train_static_white.json --raw-config configs/train_static_raw.json \
  --white-checkpoints 'outputs/train_white/checkpoints/epoch{epoch:04d}.pth' \
  --raw-checkpoints 'outputs/train_raw/checkpoints/epoch{epoch:04d}.pth' \
  --output outputs/history

learned-tg plot --kind history --input reference_data/validation_history.csv \
  --output outputs/figures/figure4.png
learned-tg plot --kind q200 --input reference_data/q200_curves.json \
  --output outputs/figures/figure3_iterations.png
learned-tg plot --kind spectrum --input reference_data/q200_spectrum.json \
  --output outputs/figures/figure3_spectrum.png
```

The q=200 example uses the first held-out 63-grid field and 50 ASM snapshot steps. The ablation shares nested Gaussian prefixes: the original batch of 20 probes followed by 55 additional probes. It tests six smoothing counts at q=20 and five additional q values at 50 smoothing steps, on eight fields with TG and FGMRES: 176 cases. Its final dataset retains 12 failures.

History evaluation uses eight independent validation fields, epochs 10–500 at spacing 10, two losses, and two solvers: 1600 solves and 200 aggregate rows. Figure 4 contains only the two iteration panels. Capped or failed cases are never silently discarded from consumed-update curves; construction failures have no invented solve count. The current plot marks epochs with incomplete acceptance.

## Forchheimer flow

```bash
learned-tg forchheimer --config configs/forchheimer.json \
  --checkpoint artifacts/forchheimer.pth --output outputs/forchheimer
learned-tg summarize --kind forchheimer --input outputs/forchheimer/summary.json \
  --output outputs/forchheimer/table.csv
learned-tg plot --kind fields --input outputs/forchheimer/representative_fields.npz \
  --output outputs/figures/forchheimer.png
```

This runs eight connected channel fields, alpha 0 and 10, and learned/rebuilt/frozen GenEO strategies: 48 trajectories. It uses grid 1023, base seed 20260720, permeability contrast `1e6`, source 10, rank 30 for the learned model, GenEO threshold 0.30 with at most 64 local modes, coarse regularization `1e-8`, and split-pressure compensated FGMRES(50) with at most 300 linear updates. Linear tolerance is `1e-6` with internal safety factor 0.5. Both the relative pressure update and nonlinear residual must meet `1e-6`, within 100 Picard updates. Every accepted linear step passes independent CPU reassembly, expanded-residual, and Decimal checks before the trajectory advances.

Pressure high/low components remain float64. Collapsing them to a single float64 array is permitted for displayed colors only, not residual acceptance. Field 0's converged learned endpoints are exported for plotting. The exact paper renderer is also supplied:

```bash
julia --project=plotting/julia -e 'using Pkg; Pkg.instantiate()'
julia --project=plotting/julia plotting/julia/forchheimer_fields.jl \
  outputs/forchheimer/representative_fields.npz outputs/figures/forchheimer_paper
```

The Julia renderer validates the paper's 1023-grid, alpha-0/10 array layout. Its manifest records Julia 1.12.4; the project compatibility starts at Julia 1.10.

## Moving-interface flow

```bash
learned-tg moving --config configs/moving.json --checkpoint artifacts/moving.pth \
  --output outputs/moving
learned-tg summarize --kind moving --input outputs/moving/comparison.json \
  --output outputs/moving/table.csv
```

The portable entry generates a 1000-step learned-TG/FGMRES trajectory, stores 80 uniformly sampled pre-transport states, and compares learned/rebuilt/frozen spaces on those identical operators. Each comparison method starts from zero on its first sampled system and uses its own previous sampled solution thereafter. The frozen GenEO span is reused while its current-operator Galerkin matrix is rebuilt. Effective RHS normalization includes the nonzero Dirichlet boundary contribution despite the zero body source. The FVM, level-set upwinding, SSPRK(2,2) transport, and CFL 0.3 are retained.

This FGMRES adapter has small-grid checks; the full new 1023-grid moving-interface benchmark has not been completed. The old Table 3 iteration values came from FGCR. They are not supplied as evidence for this FGMRES program. A full new run and controlled timings remain separate numerical work.

## Perturbation diagnostics and appendices

```bash
learned-tg perturbation --config configs/static.json \
  --checkpoint artifacts/static_white.pth --model-config configs/train_static_white.json \
  --static-results outputs/static/summary.json --output outputs/perturbation

learned-tg appendix --config configs/appendices.json --kind circle --output outputs/circle
learned-tg appendix --config configs/appendices.json --kind smoothing --output outputs/smoothing
learned-tg plot --kind circle --input reference_data/circle_convergence.csv \
  --output outputs/figures/circle.png
```

Perturbation evaluation uses the first two canonical fields per grid and requires accepted paired POD/learned solves with matching field hashes. It computes energy projector distances, small-Gram TG perturbation norms, and sparse generalized-eigenvalue estimates of the smoother norm. The norm calculations use the paper's ten-by-ten overlapping decomposition. They are not bounds on floating-point FGMRES trajectories.

## Timing and validation

All new experiment records explicitly set `timing_eligible=false`. Kernel counters or incidental elapsed seconds are not controlled runtime comparisons. Reproduce timing only under a separate fixed hardware/software and isolation protocol; this code collection does not replace the manuscript's retained timing figures with unmeasured values.


Use `python -m pytest tests -q` for the lightweight numerical contracts and `lean/verify.sh` for formal verification. Release validation details are in [validation.md](validation.md). Generated datasets, weights, result arrays, plots, caches, and logs belong under ignored output directories; none is needed in Git history.
