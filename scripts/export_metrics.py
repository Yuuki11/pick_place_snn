#!/usr/bin/env python3
from __future__ import annotations

import argparse

from rl_dev.utils.metrics import export_tensorboard_scalars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_tensorboard_scalars(args.run_dir)
    if result is None:
        print("No tensorboard scalars found.")
    else:
        print(result[0])
        print(result[1])


if __name__ == "__main__":
    main()

