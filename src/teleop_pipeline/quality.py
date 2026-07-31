"""Automated quality scoring for teleoperation demonstrations.

The premise: most of what makes a teleop demonstration bad is *measurable from
the trajectory alone*, and a human reviewing 4000 episodes will not catch any of
it consistently. Each metric below targets a specific, recognisable failure mode
of a human driving a leader arm:

``dropped_frame_rate``
    Share of timesteps lost to logging dropouts. Holes in a demonstration teach
    the policy discontinuous jumps it can never reproduce.
``timing_jitter``
    Irregularity of the *raw* control loop. Measured at ingest, before
    resampling erases it. High jitter means the rig was compute-starved, and
    every recorded velocity and delta is correspondingly wrong.
``nan_fraction``
    Missing sensor channels.
``idle_fraction``
    Share of the episode with the arm essentially stationary. Long idle stretches
    are the operator thinking, and they teach the policy to freeze — a
    disproportionately common and disproportionately fatal BC failure.
``action_saturation``
    Share of steps with the commanded delta pinned at the rig's clip limit. The
    operator wanted to move faster than the leader arm allowed, so the recorded
    action is a censored version of their intent, not their intent.
``jerk_spike_rate``
    Rate of discontinuities in commanded motion, robust-thresholded per episode.
    Catches tracker glitches and dropped-then-recovered poses.
``gripper_chatter_hz``
    Gripper open/close transitions per second, hysteresis-debounced. Fast
    chatter is operator indecision, and gripper timing is what most manipulation
    policies actually get wrong.
``tracking_error``
    Mean discrepancy between the commanded joint delta and the delta actually
    achieved. Large values mean the follower arm was not tracking — the recorded
    action never caused the recorded state, which breaks the core assumption of
    behaviour cloning.
``duration_zabs``
    Robust z-score of episode length against other episodes of the same task.
    Catches both bailed-out attempts and ones where the operator got lost.

Each raw value maps to a penalty in [0, 1] by linear interpolation between the
``good`` and ``bad`` anchors in params.yaml; the score is
``100 * (1 - weighted mean penalty)``. Thresholds live in config, never here.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .io import iter_episodes, update_episode_meta
from .schema import (
    EpisodeMeta,
    MetricScore,
    QualityReport,
    QualityTier,
    joint_cols,
)


# Channels that must be present for a step to count as observed. ee_* is
# optional (joint-space-only rigs exist), so a dropout is defined on the
# channels every rig records.
def _core_columns(n_joints: int) -> list[str]:
    return joint_cols("q", n_joints) + joint_cols("dq", n_joints) + joint_cols("act_q", n_joints)


def robust_scale(x: np.ndarray) -> float:
    """Median absolute deviation, scaled to be comparable to a std dev.

    MAD rather than std because the outliers are precisely what we are trying to
    detect; a std-based threshold is inflated by the very spikes it should flag.
    """
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    mad = float(np.median(np.abs(x - np.median(x))))
    return 1.4826 * mad


# --------------------------------------------------------------------------
# individual metrics
# --------------------------------------------------------------------------


def dropped_frame_rate(df: pd.DataFrame, n_joints: int) -> float:
    core = df[_core_columns(n_joints)]
    if core.empty:
        return 1.0
    return float(core.isna().any(axis=1).mean())


def nan_fraction(df: pd.DataFrame) -> float:
    """Fraction of missing cells, ignoring columns this rig never records."""
    cols = [c for c in df.columns if c != "t" and not bool(df[c].isna().all())]
    if not cols:
        return 1.0
    return float(df[cols].isna().to_numpy().mean())


def idle_fraction(df: pd.DataFrame, n_joints: int, speed_eps: float) -> float:
    dq = df[joint_cols("dq", n_joints)].to_numpy(dtype=np.float64)
    speed = np.linalg.norm(np.nan_to_num(dq), axis=1)
    if speed.size == 0:
        return 1.0
    return float(np.mean(speed < speed_eps))


def action_saturation(df: pd.DataFrame, n_joints: int, action_limit: float) -> float:
    act = df[joint_cols("act_q", n_joints)].to_numpy(dtype=np.float64)
    if act.size == 0 or np.isnan(act).all():
        return 0.0
    peak = np.nanmax(np.abs(act), axis=1)
    peak = peak[np.isfinite(peak)]
    if peak.size == 0:
        return 0.0
    return float(np.mean(peak >= 0.98 * action_limit))


def jerk_spike_rate(df: pd.DataFrame, n_joints: int, spike_k: float) -> float:
    """Rate of steps containing a discontinuity in commanded motion.

    Threshold is per-episode and MAD-based, so a slow careful demonstration and
    a fast one are held to the same *relative* smoothness rather than the same
    absolute jerk — which is what actually distinguishes a glitch from speed.
    """
    act = df[joint_cols("act_q", n_joints)].to_numpy(dtype=np.float64)
    if act.shape[0] < 3 or np.isnan(act).all():
        return 0.0
    jerk = np.abs(np.diff(act, axis=0))
    spikes = np.zeros(jerk.shape[0], dtype=bool)
    for j in range(jerk.shape[1]):
        col = jerk[:, j]
        scale = robust_scale(col)
        if scale <= 0:
            continue
        thresh = float(np.nanmedian(col)) + spike_k * scale
        spikes |= np.nan_to_num(col) > thresh
    return float(np.mean(spikes))


def gripper_chatter_hz(df: pd.DataFrame, duration_s: float) -> float:
    """Debounced gripper state transitions per second.

    Hysteresis (0.4 / 0.6) rather than a single 0.5 threshold: without it,
    sensor noise on a half-open gripper reads as hundreds of transitions.
    """
    if duration_s <= 0 or "grip" not in df.columns:
        return 0.0
    g = df["grip"].to_numpy(dtype=np.float64)
    g = g[np.isfinite(g)]
    if g.size < 2:
        return 0.0

    state = g[0] > 0.5
    transitions = 0
    for value in g[1:]:
        if state and value < 0.4:
            state, transitions = False, transitions + 1
        elif not state and value > 0.6:
            state, transitions = True, transitions + 1
    return float(transitions / duration_s)


def tracking_error(df: pd.DataFrame, n_joints: int) -> float:
    """Mean per-joint gap between commanded delta and achieved delta (rad).

    If this is large the follower arm was not keeping up, which means the
    recorded action did not produce the recorded next state. Behaviour cloning
    on such an episode teaches a dynamics model that does not exist.
    """
    q = df[joint_cols("q", n_joints)].to_numpy(dtype=np.float64)
    act = df[joint_cols("act_q", n_joints)].to_numpy(dtype=np.float64)
    if q.shape[0] < 2 or np.isnan(act).all():
        return 0.0
    achieved = np.diff(q, axis=0)
    commanded = act[:-1]
    err = np.abs(achieved - commanded)
    if not np.isfinite(err).any():
        return 0.0
    return float(np.nanmean(err))


def duration_zabs(duration_s: float, task_durations: np.ndarray) -> float:
    """|robust z| of this episode's length against its own task's distribution."""
    if task_durations.size < 3:
        return 0.0
    scale = robust_scale(task_durations)
    if scale <= 0:
        return 0.0
    return float(abs(duration_s - float(np.median(task_durations))) / scale)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def penalty_from(value: float, good: float, bad: float) -> float:
    """Linear map from a raw metric value to a penalty in [0, 1]."""
    if bad == good:
        return 0.0 if value <= good else 1.0
    return float(np.clip((value - good) / (bad - good), 0.0, 1.0))


def compute_metrics(
    cfg: Config, meta: EpisodeMeta, df: pd.DataFrame, task_durations: np.ndarray
) -> dict[str, float]:
    n = cfg.n_joints
    speed_eps = float(cfg.get("quality.idle_speed_eps", 0.02))
    spike_k = float(cfg.get("quality.jerk_spike_k", 12.0))

    # Real dropouts show up two ways: as NaN rows we deliberately preserved, and
    # as raw gaps the resampler bridged. Take the worse of the two so neither
    # ingest path can hide a dropout.
    dropped = max(dropped_frame_rate(df, n), meta.raw_gap_fraction)

    return {
        "dropped_frame_rate": dropped,
        "timing_jitter": meta.raw_dt_jitter,
        "nan_fraction": nan_fraction(df),
        "idle_fraction": idle_fraction(df, n, speed_eps),
        "action_saturation": action_saturation(df, n, cfg.action_limit),
        "jerk_spike_rate": jerk_spike_rate(df, n, spike_k),
        "gripper_chatter_hz": gripper_chatter_hz(df, meta.duration_s),
        "tracking_error": tracking_error(df, n),
        "duration_zabs": duration_zabs(meta.duration_s, task_durations),
    }


def score_episode(
    cfg: Config, meta: EpisodeMeta, df: pd.DataFrame, task_durations: np.ndarray
) -> QualityReport:
    spec: dict[str, dict] = cfg["quality"]["metrics"]
    values = compute_metrics(cfg, meta, df, task_durations)

    scores: list[MetricScore] = []
    flags: list[str] = []
    total_weight = 0.0
    weighted_penalty = 0.0

    for name, conf in spec.items():
        value = float(values.get(name, 0.0))
        penalty = penalty_from(value, float(conf["good"]), float(conf["bad"]))
        weight = float(conf["weight"])
        flagged = value >= float(conf["flag_at"])
        if flagged:
            flags.append(name)
        scores.append(
            MetricScore(name=name, value=value, penalty=penalty, weight=weight, flagged=flagged)
        )
        total_weight += weight
        weighted_penalty += weight * penalty

    score = 100.0 * (1.0 - weighted_penalty / total_weight) if total_weight else 0.0

    # Hard rejects are independent of the composite: an episode can score well
    # on average and still be unusable, e.g. 15% of frames simply missing.
    hard = cfg["quality"]["hard_reject"]
    reasons: list[str] = []
    if values["nan_fraction"] > float(hard["max_nan_fraction"]):
        reasons.append(f"nan_fraction={values['nan_fraction']:.3f}")
    if values["dropped_frame_rate"] > float(hard["max_dropped_frame_rate"]):
        reasons.append(f"dropped_frame_rate={values['dropped_frame_rate']:.3f}")
    if meta.n_steps < int(hard["min_steps"]):
        reasons.append(f"n_steps={meta.n_steps}")

    tiers = cfg["quality"]["tiers"]
    tier: QualityTier
    if reasons:
        tier = "reject"
    elif score >= float(tiers["gold"]):
        tier = "gold"
    elif score >= float(tiers["silver"]):
        tier = "silver"
    else:
        tier = "reject"

    return QualityReport(
        episode_id=meta.episode_id,
        session_id=meta.session_id,
        operator_id=meta.operator_id,
        task_id=meta.task_id,
        score=round(score, 2),
        tier=tier,
        metrics=scores,
        flags=flags,
        hard_reject_reasons=reasons,
    )


def score_store(
    cfg: Config, episode_root: Path | None = None, write_back: bool = True
) -> list[QualityReport]:
    """Score every episode in the store.

    Two passes: `duration_zabs` compares an episode against its own task's
    distribution, which cannot be known until every episode has been seen.
    """
    root = Path(episode_root) if episode_root else cfg.resolve("ingest.episode_dir")

    episodes: list[tuple[EpisodeMeta, pd.DataFrame]] = list(iter_episodes(root))
    by_task: dict[str, list[float]] = defaultdict(list)
    for meta, _ in episodes:
        by_task[meta.task_id].append(meta.duration_s)
    task_durations = {k: np.asarray(v, dtype=np.float64) for k, v in by_task.items()}

    reports: list[QualityReport] = []
    for meta, df in episodes:
        report = score_episode(cfg, meta, df, task_durations[meta.task_id])
        reports.append(report)
        if write_back:
            meta.quality_score = report.score
            meta.quality_tier = report.tier
            meta.flags = report.flags
            update_episode_meta(root, meta)
    return reports


def summarise(reports: list[QualityReport]) -> dict:
    """Corpus-level roll-up, for the report and for CI thresholds."""
    if not reports:
        return {"n_episodes": 0}

    scores = np.asarray([r.score for r in reports], dtype=np.float64)
    tiers = defaultdict(int)
    flag_counts: dict[str, int] = defaultdict(int)
    per_operator: dict[str, list[float]] = defaultdict(list)
    for r in reports:
        tiers[r.tier] += 1
        per_operator[r.operator_id].append(r.score)
        for f in r.flags:
            flag_counts[f] += 1

    return {
        "n_episodes": len(reports),
        "score": {
            "mean": round(float(scores.mean()), 2),
            "median": round(float(np.median(scores)), 2),
            "p10": round(float(np.percentile(scores, 10)), 2),
            "min": round(float(scores.min()), 2),
            "max": round(float(scores.max()), 2),
        },
        "tiers": dict(tiers),
        "flags": dict(sorted(flag_counts.items(), key=lambda kv: -kv[1])),
        # Per-operator means are the most actionable output here: a consistently
        # low operator is a retraining conversation, not a data problem.
        "per_operator_mean": {
            op: round(float(np.mean(v)), 2) for op, v in sorted(per_operator.items())
        },
    }
