# Paper-to-code map

Experiment labels are stable identifiers from the manuscript; figure/table numbers can change during editing.

| Manuscript item | Entry point / implementation | Supplied final data |
| --- | --- | --- |
| Static elliptic interface problem; Tables 1–2 | `learned-tg static`; `runs/static.py`, `model/interface.py`, `model/operators/ccfvm_asm.py` | `static_per_solve.csv`, `static_summary.json` |
| Raw versus whitened loss | `learned-tg train`; `model/losses/`, `training/batches.py` | Portable raw/white configurations |
| Snapshot spectrum and iteration illustration, Figure 3 | `learned-tg study --kind q200`; `coarse/pod.py` | `q200_spectrum.json`, `q200_curves.json` |
| Independent validation history, Figure 4 | `learned-tg history`; `learned-tg plot --kind history` | `validation_history.csv`, all 200 aggregate rows; two panels only |
| Retained static timing figure | `learned-tg plot --kind historical-timing`; `plotting/historical_timing.py` | `historical_static_fgcr_timing.csv`; old FGCR provenance retained |
| Moving-interface flow, Figure 5 / Table 3 | `learned-tg moving`; `runs/moving.py`, `model/two_phase.py`, `model/operators/two_phase_fvm.py` | No historical FGCR records relabeled as FGMRES; full new measurement pending |
| Forchheimer flow, Figure 6 / Table 4 | `learned-tg forchheimer`; `solvers/picard.py`, `model/forchheimer_split.py`, `model/split_pressure.py` | 48 final trajectory summaries, aggregate iterations and terminal statistics |
| Independent Forchheimer acceptance | `validation/split.py`, `validation/residuals.py`, `validation/physics.py` | Acceptance outcomes are represented in the final summaries |
| Same-grid perturbation diagnostics, Table 5 | `learned-tg perturbation`; `validation/perturbation.py` | `perturbation_summary.csv`; counts joined only after matching field and energy spans |
| Circular-interface convergence appendix | `learned-tg appendix --kind circle`; `problems/manufactured.py` | `circle_convergence.csv` |
| ASM smoothing appendix | `learned-tg appendix --kind smoothing`; `problems/smoothing.py`, `plotting/smoothing.py` | Code regenerates the deterministic example |
| Smoothing/count ablation appendix | `learned-tg study --kind ablation` | `snapshot_ablation.csv`, `ablation_summary.json` |
| Solution-trained TB-SVD baseline | `learned-tg tb-train`; `baselines/deeponet_tb/` | `tb_metadata.json` contains the retained SVD selection |
| Subspace approximation and perturbation analysis | `lean/`; see `lean/PROOF_MAP.md` | Current proof sources and pinned dependency manifest |

All paths in the implementation column are relative to `src/learned_tg/`. Final data paths are relative to `reference_data/`.


