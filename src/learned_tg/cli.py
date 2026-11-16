"""Command-line entry points; no machine-specific paths or GPU scheduling."""
from __future__ import annotations

import argparse


def _common(parser):
    parser.add_argument("--config", required=True, help="Explicit JSON experiment configuration")
    parser.add_argument("--output", required=True, help="Empty directory for generated results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)


def _campaign(parser):
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted run in its output directory; finished cases are kept")
    parser.add_argument("--save-bases", action="store_true",
                        help="Also store raw and prepared coarse-basis tensors (large); hashes are always recorded")


def parser():
    root = argparse.ArgumentParser(prog="learned-tg", description="Learned coarse-space two-grid experiments")
    root.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = root.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Generate snapshots and train a coarse-space network")
    _common(train)
    train.add_argument("--epochs", type=int)
    train.add_argument("--grid", type=int)
    train.add_argument("--samples", type=int)
    train.add_argument("--initialize-from")
    train.add_argument("--from-scratch", action="store_true")
    tb = commands.add_parser("tb-train", help="Train the independent solution-DeepONet trunk baseline")
    _common(tb)
    tb.add_argument("--epochs", type=int)
    tb.add_argument("--grid", type=int)
    tb.add_argument("--samples", type=int)
    tb.add_argument("--validation-samples", type=int)
    static = commands.add_parser("static", help="Static interface FGMRES/TG comparison")
    _common(static)
    static.add_argument("--grids", default="63,127,255,511,1023")
    static.add_argument("--samples", type=int)
    static.add_argument("--methods", default="asm,pod", help="asm,pod,learned,raw,global-pod,tb,geneo, or all")
    static.add_argument("--include-tg", action="store_true")
    static.add_argument("--subdomains", type=int)
    static.add_argument("--checkpoint")
    static.add_argument("--model-config")
    static.add_argument("--raw-checkpoint")
    static.add_argument("--raw-model-config")
    static.add_argument("--global-cache", help="Optional Global-POD cache; {n} expands to grid size")
    static.add_argument("--tb-checkpoint")
    static.add_argument("--tb-metadata")
    _campaign(static)
    forch = commands.add_parser("forchheimer", help="Independently validated split-pressure Picard comparison")
    _common(forch)
    forch.add_argument("--grid", type=int)
    forch.add_argument("--samples", type=int, default=8)
    forch.add_argument("--methods", default="learned,rebuild,frozen")
    forch.add_argument("--alphas", default="0,10")
    forch.add_argument("--max-picard-steps", type=int, default=100)
    forch.add_argument("--subdomains", type=int)
    forch.add_argument("--checkpoint")
    _campaign(forch)
    moving = commands.add_parser("moving", help="Moving-interface pressure/transport experiment")
    _common(moving)
    moving.add_argument("--grid", type=int)
    moving.add_argument("--steps", type=int)
    moving.add_argument("--subdomains", type=int)
    moving.add_argument("--checkpoint")
    moving.add_argument("--methods", default="learned,rebuild,frozen")
    study = commands.add_parser("study", help="q=200 illustration or nested snapshot ablation")
    _common(study)
    study.add_argument("--kind", choices=["q200", "ablation"], required=True)
    study.add_argument("--samples", type=int, default=8)
    study.add_argument("--subdomains", type=int)
    history = commands.add_parser("history", help="Evaluate raw/white epoch checkpoints on independent fields")
    _common(history)
    history.add_argument("--white-config", required=True)
    history.add_argument("--raw-config", required=True)
    history.add_argument("--white-checkpoints", required=True, help="Path pattern containing {epoch:04d}")
    history.add_argument("--raw-checkpoints", required=True)
    history.add_argument("--first-epoch", type=int, default=10)
    history.add_argument("--last-epoch", type=int, default=500)
    history.add_argument("--epoch-step", type=int, default=10)
    history.add_argument("--samples", type=int, default=8)
    appendix = commands.add_parser("appendix", help="Manufactured convergence or ASM smoothing")
    _common(appendix)
    appendix.add_argument("--kind", choices=["circle", "smoothing"], required=True)
    theory = commands.add_parser("perturbation", help="Paired subspace and TG-operator diagnostics")
    _common(theory)
    theory.add_argument("--checkpoint", required=True)
    theory.add_argument("--model-config", required=True)
    theory.add_argument("--static-results", required=True, help="summary.json from the paired static run")
    theory.add_argument("--grids", default="63,127,255,511,1023")
    theory.add_argument("--samples", type=int, default=2)
    plot = commands.add_parser("plot", help="Render final data without solving")
    plot.add_argument("--kind", choices=["history", "q200", "spectrum", "fields", "circle", "historical-timing"], required=True)
    plot.add_argument("--input", required=True)
    plot.add_argument("--output", required=True)
    summary = commands.add_parser("summarize", help="Table-ready counts with failures preserved")
    summary.add_argument("--kind", choices=["static", "moving", "forchheimer"], required=True)
    summary.add_argument("--input", required=True)
    summary.add_argument("--output", required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "train":
        from learned_tg.runs.train import run
    elif args.command == "tb-train":
        from learned_tg.runs.tb import run
    elif args.command == "static":
        from learned_tg.runs.static import run
    elif args.command == "forchheimer":
        from learned_tg.runs.forchheimer import run
    elif args.command == "moving":
        from learned_tg.runs.moving import run
    elif args.command == "study":
        from learned_tg.runs.studies import run
    elif args.command == "history":
        from learned_tg.runs.history import run
    elif args.command == "appendix":
        from learned_tg.runs.appendices import run
    elif args.command == "perturbation":
        from learned_tg.runs.perturbation import run
    elif args.command == "plot":
        from learned_tg.plotting.figures import run
    else:
        from learned_tg.runs.summarize import run
    run(args)
