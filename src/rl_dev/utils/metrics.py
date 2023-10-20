from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def export_tensorboard_scalars(run_dir: str | Path) -> tuple[Path, Path] | None:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return None
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return None

    accumulator = event_accumulator.EventAccumulator(str(run_dir))
    accumulator.Reload()
    rows: list[dict] = []
    for tag in accumulator.Tags().get("scalars", []):
        for event in accumulator.Scalars(tag):
            rows.append(
                {
                    "tag": tag,
                    "step": event.step,
                    "wall_time": event.wall_time,
                    "value": event.value,
                }
            )
    if not rows:
        return None

    rows.sort(key=lambda item: (item["tag"], item["step"], item["wall_time"]))
    frame = pd.DataFrame(rows)
    csv_path = run_dir / "metrics.csv"
    jsonl_path = run_dir / "metrics.jsonl"
    frame.to_csv(csv_path, index=False)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return csv_path, jsonl_path

