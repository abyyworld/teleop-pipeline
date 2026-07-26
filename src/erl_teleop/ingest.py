"""Raw teleop session -> canonical episode store.

Real labs hand you a directory of CSVs with whatever column names the logging
script happened to use that term. Rather than pretend otherwise, ingestion is
explicitly alias-driven: onboarding a new rig means adding entries to
`COLUMN_ALIASES`, not writing a new parser.

Two things happen here that are easy to skip and expensive to skip:

1. **Resampling.** Teleop loops do not run at their nominal rate. Training a
   fixed-dt model on jittery data quietly biases every velocity and delta in the
   set. We resample onto an exact 1/control_hz grid and record how much was
   interpolated so `quality.py` can penalise it.
2. **Gap preservation.** Interpolating across a two-second dropout invents
   motion that never happened. Gaps longer than `ingest.max_gap_s` are left as
   NaN and surface as `nan_fraction` downstream instead of being papered over.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .io import content_hash, write_episode, write_session_meta
from .schema import (
    EE_POS_COLS,
    EE_QUAT_COLS,
    EpisodeMeta,
    SessionMeta,
    joint_cols,
    timeseries_columns,
)

# Aliases seen in the wild, mapped to canonical names. Extend per rig.
COLUMN_ALIASES: dict[str, str] = {
    "time": "t",
    "timestamp": "t",
    "stamp": "t",
    "secs": "t",
    "gripper": "grip",
    "gripper_width": "grip",
    "gripper_pos": "grip",
    "cmd_gripper": "act_grip",
    "gripper_cmd": "act_grip",
    "action_gripper": "act_grip",
    "ee_pos_x": "ee_x",
    "ee_pos_y": "ee_y",
    "ee_pos_z": "ee_z",
    "ee_rot_x": "ee_qx",
    "ee_rot_y": "ee_qy",
    "ee_rot_z": "ee_qz",
    "ee_rot_w": "ee_qw",
}

# Indexed aliases: joint_0 -> q_0, vel_3 -> dq_3, cmd_1 -> act_q_1, ...
INDEXED_ALIASES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(?:joint|jpos|pos)_?(\d+)$"), "q_{}"),
    (re.compile(r"^(?:vel|jvel|qd|dq)_?(\d+)$"), "dq_{}"),
    (re.compile(r"^(?:cmd|action|act|target)_?(\d+)$"), "act_q_{}"),
]


class IngestError(RuntimeError):
    pass


@dataclass
class IngestResult:
    sessions: int = 0
    episodes: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)
    interpolated_fraction: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.episodes} episode(s) from {self.sessions} session(s); "
            f"{len(self.skipped)} skipped"
        )


def canonicalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw columns onto the canonical schema."""
    mapping: dict[str, str] = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in COLUMN_ALIASES:
            mapping[col] = COLUMN_ALIASES[key]
            continue
        for pattern, template in INDEXED_ALIASES:
            if m := pattern.match(key):
                mapping[col] = template.format(m.group(1))
                break
        else:
            if key != col:
                mapping[col] = key
    return df.rename(columns=mapping)


def _read_raw_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise IngestError(f"unsupported raw format: {path.name}")


@dataclass(frozen=True)
class RawTiming:
    """Timing facts about the pre-resample recording.

    Resampling makes `t` perfectly uniform, which destroys exactly the evidence
    needed to tell a clean session from one recorded on an overloaded machine.
    Measure it before it is gone.
    """

    jitter: float = 0.0  # std(dt) / mean(dt) of the raw timestamps
    gap_fraction: float = 0.0  # share of wall-clock span inside gaps > max_gap_s
    interp_fraction: float = 0.0


def measure_raw_timing(t: np.ndarray, max_gap_s: float) -> RawTiming:
    if t.size < 3:
        return RawTiming()
    dt = np.diff(t)
    mean_dt = float(np.mean(dt))
    jitter = float(np.std(dt) / mean_dt) if mean_dt > 0 else 0.0
    span = float(t[-1] - t[0])
    gap_time = float(np.sum(dt[dt > max_gap_s]))
    return RawTiming(jitter=jitter, gap_fraction=gap_time / span if span > 0 else 0.0)


def resample_uniform(
    df: pd.DataFrame, control_hz: float, max_gap_s: float
) -> tuple[pd.DataFrame, float]:
    """Resample onto an exact 1/control_hz grid.

    Returns the resampled frame and the fraction of output samples that were
    produced by interpolation rather than falling on a real sample. Samples
    inside a gap wider than `max_gap_s` are left NaN.
    """
    t = df["t"].to_numpy(dtype=np.float64)
    if t.size < 2:
        return df, 0.0

    dt = 1.0 / control_hz
    grid = np.arange(t[0], t[-1] + 0.5 * dt, dt)
    out = pd.DataFrame({"t": grid - grid[0]})

    # For each grid point, distance to the nearest real sample. Anything further
    # than half a nominal step away is interpolated rather than observed.
    idx = np.searchsorted(t, grid).clip(1, t.size - 1)
    nearest = np.minimum(np.abs(grid - t[idx - 1]), np.abs(t[idx] - grid))
    interpolated = float(np.mean(nearest > 0.5 * dt))

    # Grid points that fall inside a real dropout must not be invented.
    gap_widths = np.diff(t)
    in_gap = np.zeros(grid.shape, dtype=bool)
    for start, width in zip(t[:-1], gap_widths, strict=True):
        if width > max_gap_s:
            in_gap |= (grid > start) & (grid < start + width)

    for col in df.columns:
        if col == "t":
            continue
        values = np.interp(grid, t, df[col].to_numpy(dtype=np.float64))
        values[in_gap] = np.nan
        out[col] = values.astype(np.float32)

    return out, interpolated


def _coerce_schema(df: pd.DataFrame, n_joints: int) -> pd.DataFrame:
    """Order columns canonically and fill derivable ones that are absent."""
    cols = timeseries_columns(n_joints)

    missing_required = [c for c in ["t", *joint_cols("q", n_joints)] if c not in df.columns]
    if missing_required:
        raise IngestError(f"missing required column(s): {missing_required}")

    # Velocities are routinely absent from CSV dumps; finite-difference them.
    dq = joint_cols("dq", n_joints)
    if not all(c in df.columns for c in dq):
        t = df["t"].to_numpy(dtype=np.float64)
        dt = np.gradient(t) if t.size > 1 else np.ones_like(t)
        for i, col in enumerate(dq):
            q = df[f"q_{i}"].to_numpy(dtype=np.float64)
            df[col] = np.gradient(q) / np.where(dt == 0, np.nan, dt)

    # Commanded joint deltas are likewise often implicit in the recorded motion.
    act = joint_cols("act_q", n_joints)
    if not all(c in df.columns for c in act):
        for i, col in enumerate(act):
            q = df[f"q_{i}"].to_numpy(dtype=np.float64)
            df[col] = np.append(np.diff(q), 0.0)

    for col in [*EE_POS_COLS, *EE_QUAT_COLS, "grip", "act_grip"]:
        if col not in df.columns:
            df[col] = np.nan

    out = df.reindex(columns=cols)
    for col in cols:
        out[col] = out[col].astype(np.float64 if col == "t" else np.float32)
    return out


def ingest_session(cfg: Config, session_dir: Path, out_root: Path) -> tuple[int, list[float]]:
    """Ingest one raw session directory. Returns (n_episodes, interp_fractions)."""
    session_dir = Path(session_dir)
    meta_path = session_dir / "session.json"
    if not meta_path.exists():
        raise IngestError(f"{session_dir.name}: no session.json")

    session = SessionMeta.model_validate_json(meta_path.read_text())
    if session.n_joints != cfg.n_joints:
        raise IngestError(
            f"{session.session_id}: n_joints={session.n_joints} but params says {cfg.n_joints}"
        )

    out_dir = Path(out_root) / session.session_id
    write_session_meta(out_dir, session)

    episode_files = sorted(
        p
        for p in session_dir.iterdir()
        if p.suffix.lower() in {".csv", ".jsonl", ".ndjson", ".parquet"}
    )
    if not episode_files:
        raise IngestError(f"{session.session_id}: no episode files")

    resample = bool(cfg.get("ingest.resample", True))
    max_gap = float(cfg.get("ingest.max_gap_s", 0.25))

    n_written = 0
    interp_fractions: list[float] = []
    for path in episode_files:
        df = canonicalise_columns(_read_raw_table(path))
        if "t" not in df.columns:
            raise IngestError(f"{path.name}: no recognisable time column")
        df = df.sort_values("t").reset_index(drop=True)

        timing = measure_raw_timing(df["t"].to_numpy(dtype=np.float64), max_gap)
        interp = 0.0
        if resample:
            df, interp = resample_uniform(df, session.control_hz, max_gap)
        else:
            df["t"] = df["t"] - df["t"].iloc[0]
        timing = RawTiming(timing.jitter, timing.gap_fraction, interp)
        interp_fractions.append(interp)

        df = _coerce_schema(df, cfg.n_joints)

        # `success` is the operator's label. Encoded in the filename by the
        # teleop rig (…_ok.csv / …_fail.csv); default to True when unlabelled,
        # and let quality scoring catch the bad ones.
        stem = path.stem
        success = not re.search(r"(fail|bad|abort)", stem, flags=re.IGNORECASE)
        episode_id = f"{session.session_id}__{stem}"

        duration = float(df["t"].iloc[-1]) if len(df) else 0.0
        ep_meta = EpisodeMeta(
            episode_id=episode_id,
            session_id=session.session_id,
            operator_id=session.operator_id,
            task_id=session.task_id,
            success=success,
            n_steps=len(df),
            duration_s=duration,
            control_hz=session.control_hz,
            interp_fraction=timing.interp_fraction,
            raw_dt_jitter=timing.jitter,
            raw_gap_fraction=timing.gap_fraction,
            source_file=str(path.relative_to(session_dir.parent)),
            content_hash=content_hash(df),
        )
        write_episode(out_dir, ep_meta, df)
        n_written += 1

    return n_written, interp_fractions


def ingest_all(
    cfg: Config, raw_dir: Path | None = None, out_dir: Path | None = None
) -> IngestResult:
    raw_root = Path(raw_dir) if raw_dir else cfg.resolve("ingest.raw_dir")
    out_root = Path(out_dir) if out_dir else cfg.resolve("ingest.episode_dir")
    out_root.mkdir(parents=True, exist_ok=True)

    result = IngestResult()
    if not raw_root.exists():
        return result

    for session_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        try:
            n, interp = ingest_session(cfg, session_dir, out_root)
        except (IngestError, ValueError) as exc:
            result.skipped.append((str(session_dir), str(exc)))
            continue
        result.sessions += 1
        result.episodes += n
        if interp:
            result.interpolated_fraction[session_dir.name] = float(np.mean(interp))

    (out_root / "_ingest.json").write_text(
        pd.Series(
            {
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "sessions": result.sessions,
                "episodes": result.episodes,
                "skipped": len(result.skipped),
            }
        ).to_json(indent=2)
    )
    return result
