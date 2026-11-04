"""Configuration adapter for the independent solution-trained TB baseline."""
from learned_tg.baselines.deeponet_tb.train import build_parser, main
from learned_tg.runtime import initialize, read_config
import torch


def run(args):
    cfg = read_config(args.config)
    initialize(cfg.get("seed", 1234), args.device, args.threads, dtype=torch.float32)
    cfg.update(output_dir=args.output, device=args.device)
    for source, target in [("epochs", "epochs"), ("grid", "n"), ("samples", "num_train_samples"),
                           ("validation_samples", "num_val_samples")]:
        if getattr(args, source) is not None:
            cfg[target] = getattr(args, source)
    argv = []
    for action in build_parser()._actions:
        if action.dest not in cfg or cfg[action.dest] is None:
            continue
        value = cfg[action.dest]
        flag = next(s for s in action.option_strings if s.startswith("--"))
        if action.nargs == 0:
            if value:
                argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    return main(argv)
