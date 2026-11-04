"""Numerical routines used by the paper; no experiment orchestration."""
from __future__ import annotations
import math
from pathlib import Path

def _write_payload(smoothing_data: dict, path: Path) -> None:
    import numpy as np

    results = smoothing_data["results"]
    np.savez_compressed(
        path,
        errors=np.stack(
            [item["error"][0, 0].detach().cpu().numpy() for item in results],
            axis=0,
        ),
        phi=smoothing_data["sample"]["phi"][0].detach().cpu().numpy(),
        steps=np.asarray([int(item["steps"]) for item in results], dtype=np.int64),
        l2_norms=np.asarray([float(item["l2_norm"]) for item in results]),
        energy_norms=np.asarray([float(item["energy_norm"]) for item in results]),
    )

def _load_payload(path: Path) -> dict:
    import numpy as np

    with np.load(path) as archive:
        return {name: archive[name].copy() for name in archive.files}

def _format_ratio_math(value: float) -> str:
    """Format a positive ratio compactly for a Matplotlib mathtext title."""

    if value <= 0.0:
        return "0"
    exponent = int(math.floor(math.log10(value)))
    mantissa = value / (10.0**exponent)
    if exponent == 0:
        return f"{mantissa:.2f}"
    return rf"{mantissa:.2f}\times10^{{{exponent}}}"

def plot_experiment(
    payload: dict,
    output_path: Path,
    *,
    pdf_output: Path | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FormatStrFormatter

    steps = payload["steps"]
    errors = payload["errors"]
    l2_norms = payload["l2_norms"]
    initial_l2 = max(float(l2_norms[0]), 1.0e-300)
    ncols = 3
    nrows = math.ceil(len(steps) / ncols)
    if len(steps) != 6:
        raise ValueError("the report layout expects exactly six smoothing states")

    phi = payload["phi"]

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.titlesize": 7.0,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    }

    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(7.08, 4.45),
            layout="constrained",
        )
        axes_flat = axes.ravel()

        for panel_index, ax in enumerate(axes_flat):
            field = errors[panel_index]
            field_scale = max(float(np.max(np.abs(field))), 1.0e-300)
            image = ax.imshow(
                field,
                origin="lower",
                cmap="RdBu_r",
                vmin=-field_scale,
                vmax=field_scale,
                interpolation="nearest",
                rasterized=True,
            )
            ax.contour(
                phi,
                levels=[0.0],
                colors="#202020",
                linewidths=0.45,
                origin="lower",
            )

            step = int(steps[panel_index])
            l2_ratio = float(l2_norms[panel_index]) / initial_l2
            ratio_text = _format_ratio_math(l2_ratio)
            ax.set_title(
                rf"$k={step}$" + "\n" + rf"$\|e^k\|_2/\|e^0\|_2={ratio_text}$",
                pad=2.5,
            )
            ax.text(
                -0.035,
                1.025,
                chr(ord("a") + panel_index),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.0,
                fontweight="bold",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#4D4D4D")
                spine.set_linewidth(0.55)
            colorbar = fig.colorbar(
                image,
                ax=ax,
                orientation="vertical",
                fraction=0.046,
                pad=0.025,
                ticks=[-field_scale, 0.0, field_scale],
                format=FormatStrFormatter("%.1e"),
            )
            colorbar.outline.set_linewidth(0.5)
            colorbar.ax.tick_params(direction="out", length=2.0, width=0.5, pad=1.5)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, facecolor="white")
        if pdf_output is not None:
            pdf_output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(pdf_output, facecolor="white")
        plt.close(fig)
