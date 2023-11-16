#!/usr/bin/env python3
from __future__ import annotations

import argparse

from rl_dev.utils import load_launcher_config
from rl_dev.utils.baselines import run_local_entrypoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_pickcube_rgb_ddpg.yaml")
    return parser.parse_args()


def main() -> None:
    parsed = parse_args()
    config = load_launcher_config(parsed.config)
    args = dict(config["args"])
    run_name = args.get("exp_name", "eval_ddpg_run")
    script_path = config.get("script_path", "src/rl_dev/algorithms/ddpg_rgb.py")
    run_dir = run_local_entrypoint(
        script_path=script_path,
        launcher_config_path=config["_config_path"],
        args=args,
        run_name=run_name,
        oom_fallback_ladder=[],
    )
    print(run_dir)


if __name__ == "__main__":
    main()

