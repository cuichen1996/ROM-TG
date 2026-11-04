# Learned coarse spaces for two-grid methods

Research code for **Learning Coarse Spaces for Two-Grid Methods from Smoothed Errors with Applications to High-Contrast Porous Media Flow**, by Chenglong Bao, Chen Cui, and Ruohua Li.

The package contains the coarse-space network and losses, finite-volume operators, exact local Schwarz solves, POD and GenEO baselines, right-preconditioned FGMRES, split-pressure Forchheimer solves with independent residual checks, and the current Lean proofs. It is self-contained: no sibling manuscript, original working directory, server paths, or private launch scripts are needed.


## Installation

Use Python 3.10 or newer. Install the appropriate PyTorch build for the intended CPU or CUDA environment before installing this package.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
learned-tg --help
```

Run commands below from this repository root. The default device is CPU; select an explicit CUDA device with `--device cuda:0`. Results are written only to the specified empty output directory. Large runs should use a compute server.

## Quick start

A small pressure solve, requiring no pretrained model:

```bash
learned-tg static --config configs/static.json --grids 23 --samples 1 \
  --subdomains 2 --methods asm --output outputs/quickstart
python -m pytest tests -q
```

A single held-out paper-size case with three non-neural methods:

```bash
learned-tg static --config configs/static.json --grids 63 --samples 1 \
  --methods asm,pod,geneo --output outputs/static63
```

The full numerical recipes and artifact requirements are in [docs/reproduction.md](docs/reproduction.md). [docs/paper_map.md](docs/paper_map.md) maps the paper's experiments and proofs to the code. [reference_data/](reference_data/) contains small final numerical summaries for inspecting counts and redrawing figures without solving.

## Layout

```text
configs/                 Portable numerical and training configurations
src/learned_tg/
  model/                 Networks, losses, FVM operators, Krylov kernels
  data/                  Interface/channel/droplet snapshot generation
  training/              Paired-loss batch and checkpoint utilities
  coarse/                POD, Global-POD, GenEO, and TB-SVD construction
  baselines/deeponet_tb/  Independent solution-DeepONet trunk baseline
  solvers/               Static TG/FGMRES and split-pressure Picard kernels
  validation/            Independent residual and perturbation checks
  runs/                  Portable experiment entry points
  plotting/              Rendering from saved numerical data
lean/                    Pinned proof sources and axiom audit
reference_data/          Compact final data
docs/                    Reproduction and proof coverage
```

## Reproduction status

Static FGMRES and Forchheimer summaries are retained with failures and stopping criteria. The moving-interface FGMRES entry is supplied for a full remeasurement; the historical moving-interface table was produced with FGCR and is not bundled as FGMRES evidence. Historical static timing data and their renderer are retained with explicit FGCR provenance. New FGMRES timing experiments require separate controlled measurements.

Pretrained weights, epoch checkpoint series, generated datasets, full-resolution field arrays, and timing records are not included. Known weight hashes are listed in [reference_data/artifacts.json](reference_data/artifacts.json); no download location is currently assigned. Training a new model does not establish byte-for-byte reproduction of a reported checkpoint.

## Formal proofs

The Lean project uses Lean 4.35.0 and the pinned Mathlib revision in its lock file:

```bash
cd lean
lake exe cache get
./verify.sh
```

See [lean/README.md](lean/README.md) for build details and [lean/PROOF_MAP.md](lean/PROOF_MAP.md) for precise coverage and remaining unformalized matrix identities. Formal results concern the stated algebraic assumptions; they do not certify floating-point programs or empirical convergence counts.

## Citation and license

Citation metadata is supplied in [CITATION.cff](CITATION.cff). No license has been selected for this release directory.
