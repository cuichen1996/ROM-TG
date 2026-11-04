"""Numerical routines used by the paper; no experiment orchestration."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

GRIDS = (63, 127, 255, 511, 1023)

MAIN_SUMMARY_METHODS = (
    "ASM-FGCR",
    "oracle-TG-FGCR",
    "learned-TG-FGCR",
)

COMPRESSED_METHOD = "compressed-local-spectral-15-TG-FGCR"

THRESHOLD_METHOD = "threshold-modified-GenEO-TG-FGCR"

GLOBAL_POD30_METHOD = "global-POD-30-TG-FGCR"

MAIN_PLOT_METHODS = (*MAIN_SUMMARY_METHODS, THRESHOLD_METHOD)

COLORS = {
    "ASM-FGCR": "#666666",
    "oracle-TG-FGCR": "#1F4E79",
    "learned-TG-FGCR": "#D97706",
    GLOBAL_POD30_METHOD: "#7A5195",
    THRESHOLD_METHOD: "#2E7D32",
    COMPRESSED_METHOD: "#7A5195",
}

MARKERS = {
    "ASM-FGCR": "o",
    "oracle-TG-FGCR": "s",
    "learned-TG-FGCR": "^",
    GLOBAL_POD30_METHOD: "P",
    THRESHOLD_METHOD: "D",
    COMPRESSED_METHOD: "v",
}

DISPLAY = {
    "ASM-FGCR": "ASM–FGCR",
    "oracle-TG-FGCR": "POD reference–TG–FGCR (rank 15)",
    "learned-TG-FGCR": "learned–TG–FGCR (rank 15)",
    GLOBAL_POD30_METHOD: "global-POD-30–TG–FGCR (rank 30)",
    THRESHOLD_METHOD: r"threshold modified GenEO ($\tau=0.30$)",
    COMPRESSED_METHOD: "compressed local spectral (rank 15)",
}

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.5,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.2,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

def save_all(fig: mpl.figure.Figure, image_dir: Path, stem: str) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            image_dir / f"{stem}.{suffix}",
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)

def rows_for_method(
    rows: list[dict[str, object]], method: str
) -> list[dict[str, object]]:
    selected = [row for row in rows if row["method"] == method]
    return sorted(selected, key=lambda row: int(row["Nside"]))

def plot_main_timing(main_rows: list[dict[str, object]], image_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.55), sharex=True)
    x = np.arange(len(GRIDS), dtype=float)
    metrics = (
        ("mean_basis_setup_time", "Basis setup"),
        ("mean_solve_time", "Solve"),
        ("mean_total_time", "Total"),
    )
    for ax, (field, title) in zip(axes, metrics):
        for method in MAIN_PLOT_METHODS:
            rows = rows_for_method(main_rows, method)
            y = np.asarray([float(row[field]) for row in rows])
            if field == "mean_basis_setup_time" and method == "ASM-FGCR":
                continue
            ax.plot(
                x,
                y,
                color=COLORS[method],
                marker=MARKERS[method],
                label=DISPLAY[method],
            )
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xticks(x, [str(grid) for grid in GRIDS], rotation=35, ha="right")
        ax.grid(axis="y", which="both", color="#E3E3E3", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Time (s, log scale)")
    axes[1].set_xlabel(r"Grid size $N_{\mathrm{side}}$")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.88), pad=0.55, w_pad=0.8)
    save_all(fig, image_dir, "static_formal_timing")
