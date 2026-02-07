#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import rl_dev  # noqa: F401

from rl_dev.constants import REPO_ROOT


def to_frame_array(frame) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    else:
        frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    return frame


def run_command(command: list[str]) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode, completed.stdout.strip() or completed.stderr.strip()


def main() -> None:
    artifacts_dir = REPO_ROOT / "artifacts" / "validation"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("[validate] nvidia-smi")
    code, output = run_command(["nvidia-smi"])
    if code != 0:
        raise RuntimeError(output)
    print(output.splitlines()[0])

    vulkaninfo = shutil.which("vulkaninfo")
    if vulkaninfo:
        print("[validate] vulkaninfo --summary")
        code, output = run_command([vulkaninfo, "--summary"])
        if code != 0:
            raise RuntimeError(output)
        print("\n".join(output.splitlines()[:10]))
    else:
        print("[validate] vulkaninfo not installed; skipping CLI summary check.")

    env = gym.make(
        "PickCube-v1",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        sim_backend="gpu",
        sensor_configs={"width": 64, "height": 64},
    )
    obs, _ = env.reset(seed=0)
    if "sensor_data" not in obs:
        raise RuntimeError("Expected sensor_data in RGB observation.")
    action = torch.zeros(env.action_space.shape, dtype=torch.float32)
    frames: list[np.ndarray] = []
    for _ in range(5):
        obs, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        if frame is None:
            raise RuntimeError("env.render() returned None in rgb_array mode.")
        frames.append(to_frame_array(frame))
        if bool(terminated[0] if hasattr(terminated, "__len__") else terminated):
            break
        if bool(truncated[0] if hasattr(truncated, "__len__") else truncated):
            break
    env.close()

    frame_path = artifacts_dir / "pickcube_frame.png"
    video_path = artifacts_dir / "pickcube_validation.mp4"
    imageio.imwrite(frame_path, frames[0])
    imageio.mimwrite(video_path, frames, fps=10)
    print(f"[validate] wrote {frame_path}")
    print(f"[validate] wrote {video_path}")


if __name__ == "__main__":
    main()
