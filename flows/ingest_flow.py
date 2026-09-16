"""Prefect flow: onboard new teleop sessions without anyone watching.

DVC and Prefect answer different questions and the split is deliberate:

* **DVC** answers *"can I rebuild the result from six months ago?"* It is a
  content-addressed DAG over a fixed corpus, run on demand.
* **Prefect** answers *"a session landed on the NAS at 6pm on a Friday — is it
  any good?"* It is the operational loop: watch, validate, score, quarantine,
  notify.

Prefect rather than Airflow because Airflow needs a scheduler, a metadata
database and a webserver kept alive by someone. In an academic lab that person
graduates, and eight months later the DAGs have been silently failing since
whenever the pod restarted. A Prefect flow is a Python function: it runs from
cron, from a terminal, or from a Prefect worker, and it does not rot when
nobody is maintaining a control plane.

Run it directly::

    python flows/ingest_flow.py

Or on a schedule (needs `pip install -e '.[orchestrate]'`)::

    prefect deploy flows/ingest_flow.py:teleop_ingest_flow \\
        --name nightly --cron "0 2 * * *"
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Keep the module importable (and testable) without Prefect installed. The
# decorators degrade to pass-throughs, so the flow is still an ordinary
# callable — which is also how it gets unit tested.
try:
    from prefect import flow, get_run_logger, task

    HAS_PREFECT = True
except ImportError:  # pragma: no cover - exercised only without the extra
    HAS_PREFECT = False

    def task(fn=None, **_kwargs):
        def wrap(f):
            f.submit = f  # type: ignore[attr-defined]
            return f

        return wrap(fn) if fn else wrap

    def flow(fn=None, **_kwargs):
        def wrap(f):
            return f

        return wrap(fn) if fn else wrap

    def get_run_logger():
        import logging

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        return logging.getLogger("teleop-ingest")


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from teleop_pipeline.config import Config, load_config  # noqa: E402
from teleop_pipeline.ingest import IngestError, ingest_session  # noqa: E402
from teleop_pipeline.io import iter_episodes  # noqa: E402
from teleop_pipeline.quality import score_episode  # noqa: E402
from teleop_pipeline.validate import validate_episode  # noqa: E402

STATE_FILE = ".ingest_state.json"


@dataclass
class SessionOutcome:
    session_id: str
    episodes: int = 0
    accepted: int = 0
    quarantined: int = 0
    rejected: int = 0
    error: str | None = None
    flags: dict[str, int] | None = None


@task(name="discover-new-sessions")
def discover_sessions(cfg: Config) -> list[Path]:
    """Raw session directories not yet seen, oldest first.

    State is a JSON file of processed session ids rather than a timestamp
    watermark: a session that is still being copied when the flow fires would
    otherwise be marked done while half-written, and never looked at again.
    """
    raw_root = cfg.resolve("ingest.raw_dir")
    if not raw_root.exists():
        return []

    state_path = cfg.root / STATE_FILE
    seen: set[str] = set()
    if state_path.exists():
        seen = set(json.loads(state_path.read_text(encoding="utf-8")).get("processed", []))

    candidates = []
    for d in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        if d.name in seen or not (d / "session.json").exists():
            continue
        candidates.append(d)
    return candidates


@task(name="process-session", retries=2, retry_delay_seconds=30)
def process_session(cfg: Config, session_dir: Path) -> SessionOutcome:
    """Ingest, validate and score one session; quarantine what fails."""
    logger = get_run_logger()
    outcome = SessionOutcome(session_id=session_dir.name)

    episode_root = cfg.resolve("ingest.episode_dir")
    try:
        n_episodes, _ = ingest_session(cfg, session_dir, episode_root)
    except (IngestError, ValueError) as exc:
        outcome.error = str(exc)
        logger.error("ingest failed for %s: %s", session_dir.name, exc)
        return outcome

    outcome.episodes = n_episodes
    store_dir = episode_root / session_dir.name

    # Duration context for `duration_zabs` comes from this session's own
    # episodes. Narrower than the corpus-wide comparison the batch scorer uses,
    # which is the honest trade for being able to score a session on arrival.
    metas = [(m, df) for m, df in iter_episodes(episode_root) if m.session_id == session_dir.name]
    durations = [m.duration_s for m, _ in metas]

    import numpy as np

    duration_array = np.asarray(durations, dtype=float)
    flags: dict[str, int] = {}

    for meta, df in metas:
        report = validate_episode(cfg, meta, df)
        if not report.ok:
            _quarantine(cfg, store_dir, meta.episode_id, report.model_dump())
            outcome.quarantined += 1
            logger.warning(
                "quarantined %s: %s",
                meta.episode_id,
                ", ".join(i.code for i in report.errors),
            )
            continue

        quality = score_episode(cfg, meta, df, duration_array)
        for f in quality.flags:
            flags[f] = flags.get(f, 0) + 1

        if quality.tier == "reject":
            _quarantine(cfg, store_dir, meta.episode_id, quality.model_dump())
            outcome.rejected += 1
            logger.warning(
                "rejected %s (score %.1f): %s",
                meta.episode_id,
                quality.score,
                ", ".join(quality.hard_reject_reasons or quality.flags) or "low score",
            )
        else:
            outcome.accepted += 1

    outcome.flags = flags
    return outcome


def _quarantine(cfg: Config, store_dir: Path, episode_id: str, reason: dict) -> None:
    dest = cfg.resolve("ingest.quarantine_dir") / store_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    for suffix in (".parquet", ".meta.json"):
        src = store_dir / f"{episode_id}{suffix}"
        if src.exists():
            shutil.move(str(src), str(dest / src.name))
    (dest / f"{episode_id}.reason.json").write_text(
        json.dumps(reason, indent=2, default=str), encoding="utf-8"
    )


@task(name="record-state")
def record_state(cfg: Config, outcomes: list[SessionOutcome]) -> None:
    """Mark sessions done — but only the ones that actually got through.

    A session that errored is deliberately left unrecorded so the next run
    retries it. Silently skipping a failed session forever is how data goes
    missing without anyone noticing.
    """
    state_path = cfg.root / STATE_FILE
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    processed = set(state.get("processed", []))
    processed |= {o.session_id for o in outcomes if o.error is None}

    state_path.write_text(
        json.dumps(
            {
                "processed": sorted(processed),
                "last_run": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@task(name="write-run-summary")
def write_summary(cfg: Config, outcomes: list[SessionOutcome]) -> dict:
    logger = get_run_logger()
    flags: dict[str, int] = {}
    for o in outcomes:
        for k, v in (o.flags or {}).items():
            flags[k] = flags.get(k, 0) + v

    summary = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "sessions": len(outcomes),
        "episodes": sum(o.episodes for o in outcomes),
        "accepted": sum(o.accepted for o in outcomes),
        "quarantined": sum(o.quarantined for o in outcomes),
        "rejected": sum(o.rejected for o in outcomes),
        "errors": {o.session_id: o.error for o in outcomes if o.error},
        "flags": dict(sorted(flags.items(), key=lambda kv: -kv[1])),
    }

    out = cfg.root / "reports" / "ingest_runs"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out / f"{stamp}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    total = summary["accepted"] + summary["quarantined"] + summary["rejected"]
    if total:
        drop_rate = (summary["quarantined"] + summary["rejected"]) / total
        logger.info(
            "%d session(s), %d episode(s): %d accepted, %.0f%% dropped",
            summary["sessions"],
            summary["episodes"],
            summary["accepted"],
            100 * drop_rate,
        )
        # A rig that has come loose shows up as a sudden jump in drop rate long
        # before anyone notices the policy got worse.
        if drop_rate > 0.4:
            logger.error(
                "drop rate %.0f%% is abnormally high — check the rig before collecting more",
                100 * drop_rate,
            )
    else:
        logger.info("no new sessions")

    return summary


@flow(name="teleop-ingest", log_prints=True)
def teleop_ingest_flow(params: str | None = None) -> dict:
    """Discover, ingest, validate and score every new teleop session."""
    cfg = load_config(params)
    sessions = discover_sessions(cfg)

    outcomes = [process_session(cfg, d) for d in sessions]
    record_state(cfg, outcomes)
    return write_summary(cfg, outcomes)


if __name__ == "__main__":
    teleop_ingest_flow()
