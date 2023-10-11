#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import rl_dev  # noqa: F401

from rl_dev.constants import REPO_ROOT


def to_frame_array(frame) -> np.ndarray:
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    else:
        frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="PandaLiftCube-v0")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=64)
    parser.add_argument("--camera-height", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = gym.make(
        args.env_id,
        num_envs=args.num_envs,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        sim_backend="gpu",
        sensor_configs={"width": args.camera_width, "height": args.camera_height},
    )
    env.reset(seed=args.seed)
    frames: list[np.ndarray] = []
    for _ in range(args.steps):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        frames.append(to_frame_array(env.render()))
        if bool(terminated[0] if hasattr(terminated, "__len__") else terminated):
            break
        if bool(truncated[0] if hasattr(truncated, "__len__") else truncated):
            break
    env.close()

    output_dir = REPO_ROOT / "artifacts" / "demos"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{args.env_id}_random.mp4"
    imageio.mimwrite(video_path, frames, fps=10)
    print(video_path)


if __name__ == "__main__":
    main()
