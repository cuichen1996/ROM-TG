# Compact final reference data

These are final numerical summaries, retained to inspect reported counts and redraw figures without solving. They are not training logs or intermediate campaign records.

| File | Meaning |
| --- | --- |
| `static_per_solve.csv` | All 360 static configurations, with accepted flags, failures, updates, dimensions, and field/RHS hashes |
| `static_summary.json` | Accepted-only means/population standard deviations and failure counts |
| `snapshot_ablation.csv`, `ablation_summary.json` | All 176 snapshot-ablation cases, including failures |
| `validation_history.csv` | All 200 Figure 4 aggregate rows from the independent validation cohort |
| `q200_spectrum.json`, `q200_curves.json` | Snapshot spectrum and independently evaluated residual curves |
| `circle_convergence.csv` | Manufactured circular-interface convergence results |
| `forchheimer_trajectories.csv` | All 48 completed trajectory summaries |
| `forchheimer_iterations.csv` | Means and population standard deviations across the eight fields; per-field mean linear counts are averaged across fields |
| `forchheimer_terminal_statistics.csv` | Final nonlinear residuals and coarse dimensions |
| `perturbation_summary.csv` | Five-grid, two-field same-grid norm averages with verified FGMRES counts |
| `historical_static_fgcr_timing.csv` | Retained historical timing values, explicitly identified as FGCR; not a new FGMRES timing experiment |
| `tb_metadata.json` | Retained TB-SVD selection; use only with its matching final checkpoint |
| `artifacts.json` | Known final weight hashes; omitted artifacts and unassigned download locations are explicit |
| `provenance.json` | Hashes of original final summary sources; machine paths are omitted |

Iteration failure counts remain part of the dataset. No new FGMRES timing data, model weights, full-resolution fields, or historical moving-interface FGCR data relabeled as FGMRES is included. Large artifacts can be generated with the documented commands or supplied separately with their hashes.
