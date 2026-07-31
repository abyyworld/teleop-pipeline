"""Each test injects exactly one defect and asserts the metric meant to catch it does.

A quality gate that has never been shown a bad episode is decoration. These are
the tests that make the scoring thresholds in params.yaml meaningful — change a
threshold badly and one of these fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from teleop_pipeline import quality
from teleop_pipeline.schema import joint_cols


def test_clean_episode_scores_gold(cfg, clean_episode, episode_meta):
    df = clean_episode()
    meta = episode_meta(df)
    report = quality.score_episode(cfg, meta, df, np.asarray([meta.duration_s] * 5))
    assert report.tier == "gold", f"clean episode scored {report.score} with flags {report.flags}"
    assert report.flags == []


def test_dropped_frames_detected(cfg, clean_episode, episode_meta):
    df = clean_episode()
    # Blank out 12% of rows across the core channels — a logging dropout.
    rng = np.random.default_rng(0)
    victims = rng.choice(len(df), size=int(0.12 * len(df)), replace=False)
    df.loc[victims, joint_cols("q", cfg.n_joints)] = np.nan

    meta = episode_meta(df)
    value = quality.dropped_frame_rate(df, cfg.n_joints)
    assert value == pytest.approx(0.12, abs=0.02)

    report = quality.score_episode(cfg, meta, df, np.asarray([meta.duration_s] * 5))
    assert "dropped_frame_rate" in report.flags
    assert report.tier != "gold"


def test_raw_gap_fraction_counts_as_dropped(cfg, clean_episode, episode_meta):
    """A gap the resampler bridged must not be able to hide from the score."""
    df = clean_episode()
    meta = episode_meta(df, raw_gap_fraction=0.30)
    report = quality.score_episode(cfg, meta, df, np.asarray([meta.duration_s] * 5))
    assert report.tier == "reject"
    assert any("dropped_frame_rate" in r for r in report.hard_reject_reasons)


def test_idle_stretch_detected(cfg, clean_episode, episode_meta):
    df = clean_episode()
    # Operator stops moving for 60% of the episode.
    freeze_from = int(0.4 * len(df))
    for col in joint_cols("dq", cfg.n_joints):
        df.loc[freeze_from:, col] = 0.0

    value = quality.idle_fraction(df, cfg.n_joints, float(cfg.get("quality.idle_speed_eps")))
    assert value > 0.5

    meta = episode_meta(df)
    report = quality.score_episode(cfg, meta, df, np.asarray([meta.duration_s] * 5))
    assert "idle_fraction" in report.flags


def test_action_saturation_detected(cfg, clean_episode, episode_meta):
    df = clean_episode()
    limit = cfg.action_limit
    for col in joint_cols("act_q", cfg.n_joints):
        df[col] = limit  # operator pinned at the leader-arm limit throughout

    value = quality.action_saturation(df, cfg.n_joints, limit)
    assert value == pytest.approx(1.0)

    meta = episode_meta(df)
    report = quality.score_episode(cfg, meta, df, np.asarray([meta.duration_s] * 5))
    assert "action_saturation" in report.flags


def test_gripper_chatter_detected(cfg, clean_episode, episode_meta):
    df = clean_episode()
    df["grip"] = np.tile([0.0, 1.0], len(df) // 2)[: len(df)]

    value = quality.gripper_chatter_hz(df, float(df["t"].iloc[-1]))
    assert value > 1.0

    meta = episode_meta(df)
    report = quality.score_episode(cfg, meta, df, np.asarray([meta.duration_s] * 5))
    assert "gripper_chatter_hz" in report.flags


def test_gripper_hysteresis_ignores_noise(cfg, clean_episode):
    """Noise on a half-open gripper must not register as chatter."""
    df = clean_episode()
    rng = np.random.default_rng(1)
    df["grip"] = 0.5 + rng.normal(0, 0.03, len(df))
    assert quality.gripper_chatter_hz(df, float(df["t"].iloc[-1])) == 0.0


def test_tracking_error_detected(cfg, clean_episode, episode_meta):
    """Follower arm not keeping up: commanded delta never achieved."""
    df = clean_episode()
    for col in joint_cols("act_q", cfg.n_joints):
        df[col] = df[col] * 0.2  # only 20% of the command was realised

    value = quality.tracking_error(df, cfg.n_joints)
    baseline = quality.tracking_error(clean_episode(), cfg.n_joints)
    assert value > baseline


def test_jerk_spike_detected(cfg, clean_episode, episode_meta):
    df = clean_episode()
    clean_rate = quality.jerk_spike_rate(df, cfg.n_joints, float(cfg.get("quality.jerk_spike_k")))
    # Tracker glitch: a single discontinuous jump in the commanded delta.
    for col in joint_cols("act_q", cfg.n_joints):
        df.loc[100, col] = 0.5

    spiked = quality.jerk_spike_rate(df, cfg.n_joints, float(cfg.get("quality.jerk_spike_k")))
    assert spiked > clean_rate


def test_duration_outlier_detected(cfg):
    typical = np.asarray([8.0, 8.2, 7.9, 8.1, 8.3, 7.8])
    assert quality.duration_zabs(8.0, typical) < 1.0
    assert quality.duration_zabs(30.0, typical) > 5.0


def test_duration_zabs_needs_a_population(cfg):
    """With too few comparable episodes, report 0 rather than a fabricated z."""
    assert quality.duration_zabs(8.0, np.asarray([8.0, 9.0])) == 0.0


def test_penalty_is_clamped():
    assert quality.penalty_from(-5.0, 0.0, 1.0) == 0.0
    assert quality.penalty_from(0.5, 0.0, 1.0) == pytest.approx(0.5)
    assert quality.penalty_from(99.0, 0.0, 1.0) == 1.0
    # Degenerate anchors must not divide by zero.
    assert quality.penalty_from(1.0, 2.0, 2.0) == 0.0
    assert quality.penalty_from(3.0, 2.0, 2.0) == 1.0


def test_robust_scale_ignores_outliers():
    """MAD must not move when a few extreme values are added.

    This is the whole reason spike detection uses MAD rather than std: the
    outliers being detected would otherwise inflate their own threshold.
    """
    rng = np.random.default_rng(0)
    clean = rng.normal(0.0, 1.0, 1000)
    contaminated = np.concatenate([clean, np.full(10, 1e6)])

    assert quality.robust_scale(clean) == pytest.approx(1.0, rel=0.1)
    assert quality.robust_scale(contaminated) == pytest.approx(
        quality.robust_scale(clean), rel=0.05
    )
    # A std-based scale, by contrast, is destroyed by the same contamination.
    assert float(np.std(contaminated)) > 100 * float(np.std(clean))


def test_hard_reject_overrides_a_good_score(cfg, clean_episode, episode_meta):
    """An episode can score well on average and still be unusable.

    30% of timesteps missing a joint reading breaches
    `quality.hard_reject.max_dropped_frame_rate`, which is independent of the
    weighted composite — the remaining metrics are all fine here.
    """
    df = clean_episode()
    n_bad = int(0.30 * len(df))
    df.loc[: n_bad - 1, joint_cols("q", cfg.n_joints)] = np.nan

    meta = episode_meta(df)
    report = quality.score_episode(cfg, meta, df, np.asarray([meta.duration_s] * 5))

    assert report.score > float(cfg["quality"]["tiers"]["silver"]), (
        "fixture should still score above the silver cut, so the test proves the "
        "hard reject fired rather than the composite"
    )
    assert report.tier == "reject"
    assert any("dropped_frame_rate" in r for r in report.hard_reject_reasons)


def test_summarise_ranks_operators(cfg, clean_episode, episode_meta):
    reports = []
    for op, degrade in (("good_op", 0.0), ("bad_op", 1.0)):
        for i in range(3):
            df = clean_episode()
            if degrade:
                for col in joint_cols("act_q", cfg.n_joints):
                    df[col] = cfg.action_limit
            meta = episode_meta(df, episode_id=f"{op}_{i}", operator_id=op)
            reports.append(quality.score_episode(cfg, meta, df, np.asarray([meta.duration_s] * 5)))

    summary = quality.summarise(reports)
    assert summary["n_episodes"] == 6
    assert summary["per_operator_mean"]["good_op"] > summary["per_operator_mean"]["bad_op"]
