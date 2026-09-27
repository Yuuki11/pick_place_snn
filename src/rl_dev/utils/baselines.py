from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rl_dev.constants import DEFAULT_ASSET_DIR, MANISKILL_SAC_DIR, REPO_ROOT
from rl_dev.utils.metrics import export_tensorboard_scalars


def ensure_baseline_script(script_name: str) -> Path:
    path = MANISKILL_SAC_DIR / script_name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ManiSkill baseline script: {path}. "
            "Run scripts/bootstrap_env.sh to fetch the pinned upstream source."
        )
    return path


def ensure_local_script(script_path: str | Path) -> Path:
    path = Path(script_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Missing local entrypoint script: {path}")
    return path


def _serialize_cli_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def build_tyro_args(args: dict[str, Any], include_false_booleans: bool = False) -> list[str]:
    cli_args: list[str] = []
    for key, value in args.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cli_args.append(flag)
            elif include_false_booleans:
                cli_args.append(f"--no-{key.replace('_', '-')}")
            continue
        if value is None:
            continue
        cli_args.extend([flag, _serialize_cli_value(value)])
    return cli_args


def baseline_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(REPO_ROOT / "src")]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["MS_ASSET_DIR"] = env.get("MS_ASSET_DIR", str(DEFAULT_ASSET_DIR))
    env["MS_SKIP_ASSET_DOWNLOAD_PROMPT"] = "1"
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    return env


def stream_subprocess(command: list[str], cwd: Path, env: dict[str, str]) -> tuple[int, str]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    output_lines: list[str] = []
    for line in process.stdout:
        sys.stdout.write(line)
        output_lines.append(line)
    return process.wait(), "".join(output_lines)


def looks_like_oom(output: str) -> bool:
    lowered = output.lower()
    return "out of memory" in lowered or "cuda error" in lowered and "memory" in lowered


def write_run_manifest(
    run_dir: Path,
    *,
    baseline_script: str,
    launcher_config_path: Path,
    args: dict[str, Any],
    command: Iterable[str],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "baseline_script": baseline_script,
        "launcher_config_path": str(launcher_config_path),
        "args": args,
        "command": list(command),
    }
    with (run_dir / "launcher_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def run_baseline_entrypoint(
    *,
    baseline_script: str,
    launcher_config_path: Path,
    args: dict[str, Any],
    run_name: str,
    oom_fallback_ladder: list[dict[str, Any]] | None = None,
) -> Path:
    script_path = ensure_baseline_script(baseline_script)
    return run_python_entrypoint(
        script_path=script_path,
        script_label=baseline_script,
        launcher_config_path=launcher_config_path,
        args=args,
        run_name=run_name,
        oom_fallback_ladder=oom_fallback_ladder,
    )


def run_local_entrypoint(
    *,
    script_path: str | Path,
    launcher_config_path: Path,
    args: dict[str, Any],
    run_name: str,
    oom_fallback_ladder: list[dict[str, Any]] | None = None,
) -> Path:
    resolved_script_path = ensure_local_script(script_path)
    return run_python_entrypoint(
        script_path=resolved_script_path,
        script_label=str(resolved_script_path.relative_to(REPO_ROOT)),
        launcher_config_path=launcher_config_path,
        args=args,
        run_name=run_name,
        oom_fallback_ladder=oom_fallback_ladder,
    )


def run_python_entrypoint(
    *,
    script_path: Path,
    script_label: str,
    launcher_config_path: Path,
    args: dict[str, Any],
    run_name: str,
    oom_fallback_ladder: list[dict[str, Any]] | None = None,
) -> Path:
    env = baseline_subprocess_env()
    cwd = REPO_ROOT
    oom_fallback_ladder = oom_fallback_ladder or []
    attempt_args = dict(args)
    attempts = [attempt_args] + [dict(attempt_args, **fallback) for fallback in oom_fallback_ladder]

    bootstrap = (
        "import runpy, sys; "
        "import rl_dev; "
        "script = sys.argv[1]; "
        "sys.argv = sys.argv[1:]; "
        "runpy.run_path(script, run_name='__main__')"
    )

    last_output = ""
    for attempt_index, current_args in enumerate(attempts, start=1):
        launch_args = dict(current_args)
        base_run_name = launch_args.get("run_name") or launch_args.get("exp_name", run_name)
        if "run_name" in launch_args and len(attempts) > 1:
            launch_args["run_name"] = f"{base_run_name}__try{attempt_index}"
        cli_args = build_tyro_args(launch_args, include_false_booleans=True)
        command = [sys.executable, "-c", bootstrap, str(script_path), *cli_args]
        attempt_run_name = launch_args.get("run_name") or launch_args.get("exp_name", run_name)
        run_dir = cwd / "runs" / attempt_run_name
        write_run_manifest(
            run_dir,
            baseline_script=script_label,
            launcher_config_path=launcher_config_path,
            args=launch_args,
            command=command,
        )
        print(
            f"[rl_dev] Launching {script_label} attempt {attempt_index}/{len(attempts)}: "
            + " ".join(shlex.quote(part) for part in command[3:])
        )
        return_code, output = stream_subprocess(command, cwd=cwd, env=env)
        last_output = output
        export_tensorboard_scalars(run_dir)
        if return_code == 0:
            return run_dir
        if not looks_like_oom(output) or attempt_index == len(attempts):
            break
        print("[rl_dev] CUDA OOM detected. Applying the next fallback rung.")
    raise RuntimeError(last_output[-4000:] if last_output else "Baseline subprocess failed.")
