"""Plot final numerical data without invoking a PDE solver."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def history(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.85), layout="constrained")
    for ax, solver, title in zip(axes, ["TG", "FGMRES"], ["Standalone TG", "TG–FGMRES(50)"]):
        for label, text, color, style in [("white", "Whitened subspace loss", "#9355B5", "-"),
                                          ("raw", "Raw-snapshot loss", "#3E61D8", "--")]:
            group = sorted([r for r in rows if r["label"] == label and r["solver"] == solver], key=lambda r: int(r["epoch"]))
            x = [int(r["epoch"]) for r in group]
            y = [float(r["mean_consumed_updates"]) for r in group]
            ax.plot(x, y, label=text, color=color, ls=style, lw=1.3)
            capped = [(i, v) for i, v, r in zip(x, y, group) if int(r["accepted"]) < int(r["attempted"])]
            if capped:
                ax.scatter(*zip(*capped), marker="x", s=12, color=color)
        ax.set(title=title, xlabel="Training epoch", ylabel="Mean completed updates",
               ylim=(0, 210) if solver == "TG" else (10, 24))
        ax.grid(axis="y", alpha=0.2)
        if solver == "TG":
            ax.axhline(200, color="0.5", ls=":", lw=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    return fig


def q200(path):
    data = json.loads(Path(path).read_text())
    fig, ax = plt.subplots(figsize=(4.0, 3.0), layout="constrained")
    for label, curve in data["curves"].items():
        ax.semilogy([r["update"] for r in curve], [r["residual"] for r in curve], label=label.replace("_", "–"))
    ax.axhline(1e-6, ls=":", color="0.5", lw=0.8)
    ax.set(xlabel="Completed updates", ylabel="Independently evaluated relative residual")
    ax.legend(frameon=False); ax.grid(alpha=0.2)
    return fig


def spectrum(path):
    data = json.loads(Path(path).read_text())
    values = np.asarray(data["energy_eigenvalues"]).ravel()
    fig, ax = plt.subplots(figsize=(4.0, 3.0), layout="constrained")
    positive = values > 0
    ax.semilogy(np.arange(1, len(values) + 1)[positive], values[positive] / values[0], "o-", markersize=2)
    ax.axvline(15, color="0.5", ls=":", lw=0.8)
    ax.set(xlabel="Snapshot mode", ylabel="Normalized energy-covariance eigenvalue")
    ax.grid(alpha=0.2)
    return fig


def circle(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6), layout="constrained")
    for ax, metric in zip(axes, ["L2", "Linf", "A"]):
        for mode in ["harmonic", "interface_aware"]:
            group = [r for r in rows if r["transmissibility"] == mode and r["case"] == "circle"]
            ax.loglog([float(r["h"]) for r in group], [float(r[f"{metric}_error"]) for r in group], "o-", label=mode.replace("_", " "))
        ax.set(xlabel="Mesh width", ylabel=f"{metric} error"); ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    return fig


def fields(path):
    with np.load(path, allow_pickle=False) as data:
        mobility = np.asarray(data["mobility"]).squeeze()
        pressure = (np.asarray(data["pressure"]) if "pressure" in data else
                    np.asarray(data["pressure_high"]) + np.asarray(data["pressure_low"])).squeeze()
        phi = np.asarray(data["phi"]).squeeze()
        alphas = data["alphas"].tolist() if "alphas" in data else None
    if mobility.ndim == 2:
        mobility, pressure = mobility[None], pressure[None]
    count = len(mobility)
    fig, axes = plt.subplots(1, 2 * count, figsize=(3.0 * 2 * count, 2.8), layout="constrained", squeeze=False)
    for i in range(count):
        for j, array, cmap, text in [(0, np.log10(mobility[i]), "inferno", "Log mobility"),
                                     (1, pressure[i], "cividis", "Pressure")]:
            ax = axes[0, 2 * i + j]
            image = ax.imshow(array, origin="lower", extent=(0, 1, 0, 1), cmap=cmap)
            interface = phi if phi.ndim == 2 else phi[i]
            coords = (np.arange(len(interface)) + 0.5) / len(interface)
            ax.contour(coords, coords, interface, levels=[0], colors="white", linewidths=0.6)
            ax.set(title=text + (f", α={alphas[i]:g}" if alphas else ""), xlabel="x", ylabel="y")
            fig.colorbar(image, ax=ax, shrink=0.8)
    return fig


def run(args):
    if args.kind == "historical-timing":
        from learned_tg.plotting.historical_timing import configure_matplotlib, read_csv, plot_main_timing
        configure_matplotlib()
        rows = read_csv(Path(args.input))
        if any(r.get("data_solver") != "FGCR" for r in rows):
            raise ValueError("The historical renderer requires explicit FGCR provenance.")
        plot_main_timing(rows, Path(args.output))
        return
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig = globals()[args.kind](args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
