"""Structural and physical validation of canonical episodes.

Validation is a *hard gate*: an episode with an error is quarantined, never
trained on. It answers "is this file a well-formed recording of this robot?" —
not "is this a good demonstration". The second question is `quality.py`, and
keeping the two apart matters: a technically perfect recording of a clumsy
demonstration should be scored down, not thrown away, because it is still valid
data for some purposes.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .io import iter_episodes, read_episode_meta
from .schema import (
    EE_QUAT_COLS,
    EpisodeMeta,
    ValidationIssue,
    ValidationReport,
    joint_cols,
    timeseries_columns,
)


def _issue(code: str, message: str, severity: str = "error") -> ValidationIssue:
    return ValidationIssue(code=code, message=message, severity=severity)  # type: ignore[arg-type]


def validate_episode(cfg: Config, meta: EpisodeMeta, df: pd.DataFrame) -> ValidationReport:
    issues: list[ValidationIssue] = []
    n = cfg.n_joints
    v = cfg["validate"]

    # -- structure ---------------------------------------------------------
    expected = timeseries_columns(n)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        issues.append(_issue("missing_columns", f"missing columns: {missing}"))
        # Nothing below can be trusted without the columns.
        return ValidationReport(
            episode_id=meta.episode_id, session_id=meta.session_id, ok=False, issues=issues
        )

    n_steps = len(df)
    if n_steps < int(v["min_steps"]):
        issues.append(_issue("too_short", f"{n_steps} steps < min {v['min_steps']}"))
    if n_steps > int(v["max_steps"]):
        issues.append(_issue("too_long", f"{n_steps} steps > max {v['max_steps']}"))
    if n_steps < 2:
        return ValidationReport(
            episode_id=meta.episode_id, session_id=meta.session_id, ok=False, issues=issues
        )

    # -- timebase ----------------------------------------------------------
    t = df["t"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(t)):
        issues.append(_issue("nonfinite_time", "time column contains NaN/inf"))
    elif np.any(np.diff(t) <= 0):
        n_bad = int(np.sum(np.diff(t) <= 0))
        issues.append(_issue("time_not_increasing", f"{n_bad} non-increasing timestamp(s)"))
    else:
        mean_dt = float(np.mean(np.diff(t)))
        rel_err = abs(mean_dt - cfg.dt) / cfg.dt
        if rel_err > float(v["max_mean_dt_error"]):
            issues.append(
                _issue(
                    "dt_mismatch",
                    f"mean dt {mean_dt:.4f}s deviates {rel_err:.0%} from nominal {cfg.dt:.4f}s",
                )
            )

    # -- completeness ------------------------------------------------------
    # ee_* may be legitimately absent for rigs that log joint space only; an
    # all-NaN optional column is a warning, scattered NaNs are corruption.
    optional_all_nan = {c for c in df.columns if c != "t" and bool(df[c].isna().all())}
    if optional_all_nan:
        issues.append(
            _issue(
                "columns_absent",
                f"column(s) present but entirely empty: {sorted(optional_all_nan)}",
                "warning",
            )
        )
    scored = df.drop(columns=["t", *optional_all_nan], errors="ignore")
    nan_fraction = float(scored.isna().to_numpy().mean()) if scored.shape[1] else 0.0
    if nan_fraction > float(v["max_nan_fraction"]):
        issues.append(
            _issue(
                "excess_nan",
                f"{nan_fraction:.1%} missing samples > max {float(v['max_nan_fraction']):.1%}",
            )
        )

    # -- physical plausibility --------------------------------------------
    q = df[joint_cols("q", n)].to_numpy(dtype=np.float64)
    lower, upper = cfg.joint_lower, cfg.joint_upper
    with np.errstate(invalid="ignore"):
        below = np.nanmax(np.where(np.isnan(q), -np.inf, lower - q), initial=0.0)
        above = np.nanmax(np.where(np.isnan(q), -np.inf, q - upper), initial=0.0)
    excess = float(max(below, above))
    if excess > 0.1:
        # A large excursion means a frame or unit mismatch, not a tight demo.
        issues.append(
            _issue("joint_limit_violation", f"joint position {excess:.3f} rad outside limits")
        )
    elif excess > 0.0:
        issues.append(
            _issue(
                "joint_limit_grazed",
                f"joint position {excess:.4f} rad outside limits (within tolerance)",
                "warning",
            )
        )

    quat = df[EE_QUAT_COLS].to_numpy(dtype=np.float64)
    if not np.isnan(quat).all():
        norms = np.linalg.norm(quat, axis=1)
        norms = norms[np.isfinite(norms)]
        if norms.size and float(np.nanmax(np.abs(norms - 1.0))) > 0.05:
            issues.append(
                _issue(
                    "quaternion_not_unit",
                    f"max |‖q‖-1| = {float(np.nanmax(np.abs(norms - 1.0))):.3f}",
                    "warning",
                )
            )

    for col in ("grip", "act_grip"):
        vals = df[col].to_numpy(dtype=np.float64)
        if not np.isnan(vals).all():
            lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
            if lo < -0.01 or hi > 1.01:
                issues.append(
                    _issue(
                        "gripper_out_of_range",
                        f"{col} spans [{lo:.2f}, {hi:.2f}], expected [0, 1]",
                        "warning",
                    )
                )

    act = df[joint_cols("act_q", n)].to_numpy(dtype=np.float64)
    if not np.isnan(act).all():
        peak = float(np.nanmax(np.abs(act)))
        if peak > cfg.action_limit * 1.05:
            issues.append(
                _issue(
                    "action_over_limit",
                    f"peak |action| {peak:.4f} exceeds limit {cfg.action_limit:.4f}",
                    "warning",
                )
            )

    ok = not any(i.severity == "error" for i in issues)
    return ValidationReport(
        episode_id=meta.episode_id, session_id=meta.session_id, ok=ok, issues=issues
    )


def validate_store(
    cfg: Config, episode_root: Path | None = None, quarantine: bool = True
) -> list[ValidationReport]:
    """Validate every episode in the store, quarantining the failures."""
    root = Path(episode_root) if episode_root else cfg.resolve("ingest.episode_dir")
    quarantine_root = cfg.resolve("ingest.quarantine_dir")

    reports: list[ValidationReport] = []
    for meta, df in iter_episodes(root):
        report = validate_episode(cfg, meta, df)
        reports.append(report)
        if not report.ok and quarantine:
            _quarantine(root, quarantine_root, meta, report)
    return reports


def _quarantine(
    root: Path, quarantine_root: Path, meta: EpisodeMeta, report: ValidationReport
) -> None:
    dest = Path(quarantine_root) / meta.session_id
    dest.mkdir(parents=True, exist_ok=True)
    src_dir = Path(root) / meta.session_id
    for suffix in (".parquet", ".meta.json"):
        src = src_dir / f"{meta.episode_id}{suffix}"
        if src.exists():
            shutil.move(str(src), str(dest / src.name))
    (dest / f"{meta.episode_id}.validation.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )


def load_reports(path: Path) -> list[ValidationReport]:
    import json

    return [
        ValidationReport.model_validate(r)
        for r in json.loads(Path(path).read_text(encoding="utf-8"))
    ]


__all__ = [
    "read_episode_meta",
    "validate_episode",
    "validate_store",
    "load_reports",
]
