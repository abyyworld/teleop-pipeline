"""Stage definitions and the single-job runner behind the desktop app.

The app calls the same library functions the CLI calls, so a run started from a
button and a run started from a terminal produce identical artefacts. What the
app adds is a place to watch it happen and a place to read the result, which is
the difference between a pipeline you can operate at a rig and one you can only
operate from a laptop with the docs open.

One job at a time, deliberately, and for two reasons. Two stages writing the
episode store concurrently would race, and the failure would surface later as a
corrupt corpus rather than as an error here. And the log is captured by
redirecting `sys.stdout`, which is process-global: with a second job running,
each would collect the other's output. Anything else the process prints while a
stage runs therefore lands in that stage's log, which is the accepted cost of
in-process capture and the reason the app never starts two.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config, load_config


@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    blurb: str
    #: Stages that must have produced their output before this one can run.
    needs: tuple[str, ...] = ()
    #: Set for stages that are not part of a normal run.
    optional: bool = False


STAGES: tuple[Stage, ...] = (
    Stage(
        "synth",
        "Generate demo data",
        "Synthetic teleop sessions with realistic defects. Skip this once you have real "
        "recordings in the raw directory.",
        optional=True,
    ),
    Stage(
        "ingest",
        "Ingest",
        "Normalise every raw session onto a fixed time grid, hash it, and record the gaps "
        "that had to be interpolated.",
    ),
    Stage(
        "validate",
        "Validate",
        "Schema and physics checks. Episodes that fail are excluded from everything "
        "downstream rather than silently trained on.",
        needs=("ingest",),
    ),
    Stage(
        "score",
        "Score quality",
        "Nine quality metrics per episode, combined into a score and a tier.",
        needs=("ingest",),
    ),
    Stage(
        "dataset",
        "Build dataset",
        "Train and validation splits grouped by session, so no session contributes to both.",
        needs=("score",),
    ),
    Stage(
        "train",
        "Train policy",
        "Behaviour cloning with action chunking. Needs PyTorch.",
        needs=("dataset",),
    ),
    Stage(
        "eval",
        "Evaluate",
        "Offline evaluation against three baselines, with bootstrap confidence intervals.",
        needs=("train",),
    ),
    Stage(
        "report",
        "Data quality report",
        "Markdown report over the scored corpus.",
        needs=("score",),
        optional=True,
    ),
)

STAGE_BY_KEY = {s.key: s for s in STAGES}
#: The stages "Run all" executes, in order. `synth` is excluded because running
#: it against a real corpus would mix generated episodes into real data.
DEFAULT_RUN = ("ingest", "validate", "score", "dataset", "train", "eval", "report")


class LogBuffer:
    """Append-only log the UI polls by index.

    Presents as a text stream so `print` and rich both write here unchanged.
    `isatty` is False, which is what makes rich emit plain text instead of
    escape sequences the browser would have to strip.
    """

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._partial = ""
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        with self._lock:
            self._partial += text
            while "\n" in self._partial:
                line, self._partial = self._partial.split("\n", 1)
                self._lines.append(line)
        return len(text)

    def flush(self) -> None:  # part of the stream protocol
        return None

    def isatty(self) -> bool:
        return False

    def line(self, text: str) -> None:
        with self._lock:
            self._lines.append(text)

    def promote_partial(self) -> None:
        """Move a trailing unterminated line into the log.

        A stage that writes a progress line without a newline would otherwise
        leave its last output invisible until the next write.
        """
        with self._lock:
            if self._partial:
                self._lines.append(self._partial)
                self._partial = ""

    def since(self, index: int) -> tuple[list[str], int]:
        with self._lock:
            index = max(0, min(index, len(self._lines)))
            return self._lines[index:], len(self._lines)

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()
            self._partial = ""


# -- the stage implementations ------------------------------------------------
#
# Each returns a small dict that the UI renders. They call the same functions
# the CLI commands call; nothing here reimplements pipeline logic.


def _run_synth(cfg: Config, options: dict[str, Any]) -> dict[str, Any]:
    from ..synthetic import generate

    target = cfg.resolve("ingest.raw_dir")
    sessions = int(options.get("sessions", 12))
    written = generate(cfg, target, n_sessions=sessions, seed=int(options.get("seed", 0)))
    print(f"wrote {written} episode(s) across {sessions} session(s) -> {target}")
    return {"episodes": written, "sessions": sessions}


def _run_ingest(cfg: Config, options: dict[str, Any]) -> dict[str, Any]:
    from ..ingest import ingest_all

    result = ingest_all(cfg, prune=True)
    print(f"ingest {result.summary()}")
    if result.episodes == 0:
        raise RuntimeError(
            "no episodes ingested. Put raw sessions in "
            f"{cfg.resolve('ingest.raw_dir')}, or run Generate demo data first."
        )
    return {"episodes": result.episodes, "skipped": len(result.skipped)}


def _run_validate(cfg: Config, options: dict[str, Any]) -> dict[str, Any]:
    from ..validate import validate_store

    reports = validate_store(cfg, quarantine=False)
    failed = [r for r in reports if not r.ok]
    warned = [r for r in reports if r.ok and r.warnings]
    path = cfg.root / "reports" / "validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.model_dump() for r in reports], indent=2), encoding="utf-8")
    print(f"{len(reports)} episode(s): {len(reports) - len(failed)} ok, {len(failed)} failed")
    for r in failed[:10]:
        print(f"  fail {r.episode_id}: {', '.join(i.code for i in r.errors)}")
    return {"total": len(reports), "failed": len(failed), "warned": len(warned)}


def _run_score(cfg: Config, options: dict[str, Any]) -> dict[str, Any]:
    from ..quality import score_store, summarise

    reports = score_store(cfg, write_back=False)
    if not reports:
        raise RuntimeError("no episodes to score. Run Ingest first.")
    out = cfg.root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "quality.json").write_text(
        json.dumps([r.model_dump() for r in reports], indent=2), encoding="utf-8"
    )
    summary = summarise(reports)
    (out / "quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"{summary['n_episodes']} episodes  mean {summary['score']['mean']}  "
        f"median {summary['score']['median']}"
    )
    return summary


def _run_dataset(cfg: Config, options: dict[str, Any]) -> dict[str, Any]:
    from ..dataset import build_dataset

    manifest = build_dataset(cfg)
    for split in ("train", "val"):
        s = manifest["splits"][split]
        print(f"{split}: {s['n_episodes']} episodes, {s['n_steps']:,} steps")
    return manifest


def _run_train(cfg: Config, options: dict[str, Any]) -> dict[str, Any]:
    from ..train import train as run_train

    summary = run_train(cfg)
    print(f"best val {summary['best_val_loss']:.5f} at epoch {summary['best_epoch']}")
    history_path = cfg.root / "artifacts" / "history.json"
    if history_path.is_file():
        summary = {**summary, "history": json.loads(history_path.read_text(encoding="utf-8"))}
    return summary


def _run_eval(cfg: Config, options: dict[str, Any]) -> dict[str, Any]:
    from ..evaluate import evaluate

    ckpt = cfg.root / "artifacts" / "policy.pt"
    if not ckpt.is_file():
        raise RuntimeError("no checkpoint at artifacts/policy.pt. Run Train policy first.")
    report = evaluate(cfg, ckpt, split="val")
    m = report["metrics"]
    skill = m.get("skill_vs_best_baseline")
    print(f"action MAE {m['action_mae']:.5f}")
    if skill is not None:
        print(f"vs best baseline ({m.get('best_baseline')}): {skill:+.1%}")
    return report


def _run_report(cfg: Config, options: dict[str, Any]) -> dict[str, Any]:
    from .. import report as report_mod
    from ..schema import QualityReport, ValidationReport

    quality_path = cfg.root / "reports" / "quality.json"
    if not quality_path.is_file():
        raise RuntimeError("no quality report. Run Score quality first.")
    quality = [
        QualityReport.model_validate(r)
        for r in json.loads(quality_path.read_text(encoding="utf-8"))
    ]
    validation_path = cfg.root / "reports" / "validation.json"
    validation = None
    if validation_path.is_file():
        validation = [
            ValidationReport.model_validate(r)
            for r in json.loads(validation_path.read_text(encoding="utf-8"))
        ]
    path = report_mod.write(
        cfg.root / "reports" / "data_quality.md", report_mod.render(quality, validation)
    )
    print(f"wrote {path}")
    return {"path": str(path), "markdown": Path(path).read_text(encoding="utf-8")}


RUNNERS: dict[str, Callable[[Config, dict[str, Any]], dict[str, Any]]] = {
    "synth": _run_synth,
    "ingest": _run_ingest,
    "validate": _run_validate,
    "score": _run_score,
    "dataset": _run_dataset,
    "train": _run_train,
    "eval": _run_eval,
    "report": _run_report,
}


# -- artefacts on disk --------------------------------------------------------


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def stage_outputs(cfg: Config) -> dict[str, Path]:
    """The file whose presence means a stage has produced something."""
    root = cfg.root
    return {
        "synth": cfg.resolve("ingest.raw_dir"),
        "ingest": cfg.resolve("ingest.episode_dir"),
        "validate": root / "reports" / "validation.json",
        "score": root / "reports" / "quality_summary.json",
        "dataset": cfg.resolve("dataset.out_dir") / "manifest.json",
        "train": root / "artifacts" / "policy.pt",
        "eval": root / "reports" / "eval_val.json",
        "report": root / "reports" / "data_quality.md",
    }


def results_on_disk(cfg: Config) -> dict[str, Any]:
    """Everything the results panel renders, read fresh from the artefacts.

    Read from disk rather than kept in memory so the app shows the real state
    of the project after a restart, and so a run performed from the terminal
    shows up here too.
    """
    root = cfg.root
    train_summary = _read_json(root / "artifacts" / "latest_run.json")
    history = _read_json(root / "artifacts" / "history.json")
    if train_summary and history:
        train_summary = {**train_summary, "history": history}
    return {
        "quality": _read_json(root / "reports" / "quality_summary.json"),
        "dataset": _read_json(cfg.resolve("dataset.out_dir") / "manifest.json"),
        "train": train_summary,
        "eval": _read_json(root / "reports" / "eval_val.json"),
    }


def torch_available() -> bool:
    from importlib.util import find_spec

    return find_spec("torch") is not None


# -- the runner ---------------------------------------------------------------


@dataclass
class StageRun:
    key: str
    status: str = "running"  # running | done | failed | skipped
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None

    @property
    def elapsed(self) -> float:
        return (self.finished_at or time.time()) - self.started_at

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "status": self.status,
            "elapsed_s": round(self.elapsed, 2),
            "error": self.error,
        }


class Runner:
    """Runs stages in a worker thread, one job at a time."""

    def __init__(self, params: str | None = None):
        self._params = params
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self.log = LogBuffer()
        self.runs: list[StageRun] = []
        self.current: str | None = None
        self.last_error: str | None = None

    def config(self) -> Config:
        return load_config(self._params)

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, keys: list[str], options: dict[str, Any] | None = None) -> None:
        """Queue a list of stages. Raises if one is already running."""
        unknown = [k for k in keys if k not in RUNNERS]
        if unknown:
            raise ValueError(f"unknown stage(s): {', '.join(unknown)}")
        with self._lock:
            if self.busy:
                raise RuntimeError("a run is already in progress")
            self._cancel.clear()
            self.log.clear()
            self.runs = [StageRun(k, status="queued") for k in keys]
            self.last_error = None
            self._thread = threading.Thread(
                target=self._work, args=(keys, options or {}), daemon=True
            )
            self._thread.start()

    def cancel(self) -> None:
        """Ask the run to stop after the current stage.

        A stage is not interrupted mid-write: killing `ingest` halfway would
        leave a partially written episode store, which is worse than waiting.
        """
        self._cancel.set()

    def _work(self, keys: list[str], options: dict[str, Any]) -> None:
        cfg = self.config()
        for run in self.runs:
            if self._cancel.is_set():
                run.status = "skipped"
                run.finished_at = time.time()
                continue
            self.current = run.key
            run.status = "running"
            run.started_at = time.time()
            stage = STAGE_BY_KEY[run.key]
            self.log.line(f"=== {stage.title} ===")
            try:
                with contextlib.redirect_stdout(self.log), contextlib.redirect_stderr(self.log):
                    RUNNERS[run.key](cfg, options)
                run.status = "done"
            except Exception as exc:  # surfaced in the UI, not swallowed
                run.status = "failed"
                run.error = f"{type(exc).__name__}: {exc}"
                self.last_error = run.error
                self.log.line(f"FAILED  {run.error}")
                for line in traceback.format_exc().splitlines()[-12:]:
                    self.log.line("  " + line)
                # Later stages consume this one's output, so continuing would
                # either fail identically or train on stale data.
                run.finished_at = time.time()
                for later in self.runs[self.runs.index(run) + 1 :]:
                    later.status = "skipped"
                    later.finished_at = time.time()
                break
            finally:
                if run.finished_at is None:
                    run.finished_at = time.time()
                self.log.line(f"    {stage.title}: {run.status} in {run.elapsed:.1f}s")
        self.current = None

    def state(self) -> dict[str, Any]:
        return {
            "busy": self.busy,
            "current": self.current,
            "cancelling": self._cancel.is_set() and self.busy,
            "runs": [r.as_dict() for r in self.runs],
            "last_error": self.last_error,
        }


__all__ = [
    "DEFAULT_RUN",
    "RUNNERS",
    "STAGES",
    "STAGE_BY_KEY",
    "LogBuffer",
    "Runner",
    "Stage",
    "results_on_disk",
    "stage_outputs",
    "torch_available",
]
