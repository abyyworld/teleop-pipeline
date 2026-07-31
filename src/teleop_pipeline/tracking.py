"""Experiment tracking behind a two-method interface.

MLflow is the default because it self-hosts with zero infrastructure
(`file:./mlruns` is a directory) and a lab can later point every run at a shared
server by exporting `MLFLOW_TRACKING_URI` — no code change, no accounts, no
per-seat cost. Weights & Biases is a drop-in alternative and slots in as another
`Tracker` subclass if the lab already pays for it.

The `JsonlTracker` fallback exists so that the pipeline never fails because a
tracking server is unreachable. Losing a training run because the metrics
database was down is a self-inflicted wound.
"""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Tracker(AbstractContextManager):
    """Minimal tracking surface: params in, metrics and artifacts out."""

    run_id: str = ""

    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
    def log_artifact(self, path: Path) -> None: ...
    def set_tags(self, tags: dict[str, str]) -> None: ...

    def __exit__(self, *exc: object) -> None:
        return None


class NullTracker(Tracker):
    def __init__(self) -> None:
        self.run_id = "null"


class JsonlTracker(Tracker):
    """Append-only local log. Always works, never blocks a run."""

    def __init__(self, out_dir: Path, run_id: str) -> None:
        self.run_id = run_id
        self.dir = Path(out_dir) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "events.jsonl"

    def _emit(self, kind: str, payload: dict) -> None:
        with self.path.open("a") as fh:
            fh.write(
                json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "kind": kind, **payload})
                + "\n"
            )

    def log_params(self, params: dict[str, Any]) -> None:
        self._emit("params", {"params": _flatten(params)})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._emit("metrics", {"step": step, "metrics": metrics})

    def log_artifact(self, path: Path) -> None:
        self._emit("artifact", {"path": str(path)})

    def set_tags(self, tags: dict[str, str]) -> None:
        self._emit("tags", {"tags": tags})


class MlflowTracker(Tracker):
    def __init__(self, uri: str, experiment: str, run_name: str) -> None:
        import mlflow  # imported lazily: mlflow is an optional extra

        self._mlflow = mlflow
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        self._run = mlflow.start_run(run_name=run_name)
        self.run_id = self._run.info.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        # MLflow rejects params over 500 chars and nested values.
        self._mlflow.log_params({k: str(v)[:500] for k, v in _flatten(params).items()})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._mlflow.log_metrics(
            {k: float(v) for k, v in metrics.items() if v is not None}, step=step
        )

    def log_artifact(self, path: Path) -> None:
        self._mlflow.log_artifact(str(path))

    def set_tags(self, tags: dict[str, str]) -> None:
        self._mlflow.set_tags(tags)

    def __exit__(self, *exc: object) -> None:
        self._mlflow.end_run()


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def make_tracker(cfg, run_name: str, fallback_dir: Path) -> Tracker:
    """Build the configured tracker, degrading rather than failing."""
    backend = str(cfg.get("tracking.backend", "mlflow")).lower()

    if backend == "none":
        return NullTracker()
    if backend == "jsonl":
        return JsonlTracker(fallback_dir, run_name)

    try:
        return MlflowTracker(
            uri=cfg.tracking_uri,
            experiment=str(cfg.get("tracking.experiment", "teleop-pipeline")),
            run_name=run_name,
        )
    except Exception as exc:  # noqa: BLE001 — degrading is the whole point
        print(f"[tracking] mlflow unavailable ({exc}); falling back to jsonl")
        return JsonlTracker(fallback_dir, run_name)
