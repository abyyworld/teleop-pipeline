from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

from erl_teleop import ingest, validate
from erl_teleop.schema import joint_cols, timeseries_columns

# -- column aliasing --------------------------------------------------------


def test_aliases_map_messy_columns():
    df = pd.DataFrame(
        {
            "timestamp": [0.0, 0.05],
            "joint_0": [0.1, 0.2],
            "vel_3": [0.0, 0.1],
            "cmd_2": [0.01, 0.01],
            "gripper": [1.0, 0.0],
            "gripper_cmd": [0.0, 0.0],
            "ee_pos_x": [0.3, 0.3],
        }
    )
    out = ingest.canonicalise_columns(df)
    assert set(out.columns) >= {"t", "q_0", "dq_3", "act_q_2", "grip", "act_grip", "ee_x"}


def test_unknown_columns_are_lowercased_not_dropped():
    out = ingest.canonicalise_columns(pd.DataFrame({"WeirdSensor": [1.0]}))
    assert "weirdsensor" in out.columns


# -- resampling -------------------------------------------------------------


def test_resample_produces_a_uniform_grid():
    t = np.sort(np.random.default_rng(0).uniform(0, 5, 90))
    df = pd.DataFrame({"t": t, "q_0": np.sin(t)})
    out, interp = ingest.resample_uniform(df, control_hz=20.0, max_gap_s=0.25)

    dt = np.diff(out["t"].to_numpy())
    assert np.allclose(dt, 0.05, atol=1e-9)
    assert out["t"].iloc[0] == pytest.approx(0.0)
    assert 0.0 <= interp <= 1.0


def test_resample_does_not_invent_data_across_a_gap():
    """Interpolating over a dropout fabricates motion that never happened."""
    t = np.concatenate([np.arange(0, 1.0, 0.05), np.arange(3.0, 4.0, 0.05)])
    df = pd.DataFrame({"t": t, "q_0": np.zeros_like(t)})
    out, _ = ingest.resample_uniform(df, control_hz=20.0, max_gap_s=0.25)

    # The recording holds a hole between the samples at t=0.95 and t=3.0. Every
    # grid point strictly inside it must be NaN, every point outside must
    # survive. The tolerance is for float accumulation in the grid: 19 * 0.05
    # lands a hair above 0.95, and that endpoint is a real sample.
    eps = 1e-6
    inside_gap = (out["t"] > 0.95 + eps) & (out["t"] < 3.0 - eps)
    assert inside_gap.sum() > 30
    assert out.loc[inside_gap, "q_0"].isna().all(), "gap was silently bridged"
    assert out.loc[~inside_gap, "q_0"].notna().all(), "real samples were dropped"


def test_raw_timing_measures_jitter_before_it_is_erased():
    clean = np.arange(0, 5, 0.05)
    jittered = clean + np.random.default_rng(1).normal(0, 0.01, clean.size)
    jittered = np.maximum.accumulate(jittered) + 1e-6 * np.arange(clean.size)

    assert ingest.measure_raw_timing(clean, 0.25).jitter < 1e-6
    assert ingest.measure_raw_timing(jittered, 0.25).jitter > 0.05


def test_raw_timing_reports_gap_fraction():
    t = np.concatenate([np.arange(0, 1.0, 0.05), np.arange(3.0, 4.0, 0.05)])
    timing = ingest.measure_raw_timing(t, max_gap_s=0.25)
    # A 2.05 s hole in a ~3.95 s recording.
    assert timing.gap_fraction == pytest.approx(2.05 / 3.95, rel=0.05)


# -- validation -------------------------------------------------------------


def test_clean_episode_validates(cfg, clean_episode, episode_meta):
    df = clean_episode()
    report = validate.validate_episode(cfg, episode_meta(df), df)
    assert report.ok, [i.model_dump() for i in report.errors]


def test_missing_columns_are_an_error(cfg, clean_episode, episode_meta):
    df = clean_episode().drop(columns=["q_3"])
    report = validate.validate_episode(cfg, episode_meta(df), df)
    assert not report.ok
    assert report.errors[0].code == "missing_columns"


def test_non_monotonic_time_is_an_error(cfg, clean_episode, episode_meta):
    df = clean_episode()
    df.loc[50, "t"] = df.loc[10, "t"]
    report = validate.validate_episode(cfg, episode_meta(df), df)
    assert not report.ok
    assert any(i.code == "time_not_increasing" for i in report.errors)


def test_joint_limit_violation_is_an_error(cfg, clean_episode, episode_meta):
    df = clean_episode()
    df.loc[20, "q_0"] = 99.0  # a frame or unit mismatch, not a tight demo
    report = validate.validate_episode(cfg, episode_meta(df), df)
    assert not report.ok
    assert any(i.code == "joint_limit_violation" for i in report.errors)


def test_too_short_is_an_error(cfg, clean_episode, episode_meta):
    df = clean_episode(n_steps=5)
    report = validate.validate_episode(cfg, episode_meta(df), df)
    assert not report.ok
    assert any(i.code == "too_short" for i in report.errors)


def test_absent_optional_channel_is_only_a_warning(cfg, clean_episode, episode_meta):
    """Joint-space-only rigs are legitimate; they must not be rejected."""
    df = clean_episode()
    for col in ("ee_x", "ee_y", "ee_z"):
        df[col] = np.nan
    report = validate.validate_episode(cfg, episode_meta(df), df)
    assert report.ok
    assert any(i.code == "columns_absent" for i in report.warnings)


def test_scattered_nans_are_an_error(cfg, clean_episode, episode_meta):
    """Unlike an absent channel, holes inside a recorded channel are corruption."""
    df = clean_episode()
    rng = np.random.default_rng(0)
    victims = rng.choice(len(df), size=int(0.5 * len(df)), replace=False)
    df.loc[victims, joint_cols("q", cfg.n_joints)] = np.nan
    report = validate.validate_episode(cfg, episode_meta(df), df)
    assert not report.ok
    assert any(i.code == "excess_nan" for i in report.errors)


def test_non_unit_quaternion_is_a_warning(cfg, clean_episode, episode_meta):
    df = clean_episode()
    df["ee_qw"] = 5.0
    report = validate.validate_episode(cfg, episode_meta(df), df)
    assert any(i.code == "quaternion_not_unit" for i in report.warnings)


# -- store reconciliation ---------------------------------------------------


def _tiny_corpus(tmp_path, cfg, n_sessions=3):
    """A small raw dump plus a config pointed at it."""
    import shutil

    from erl_teleop.config import load_config
    from erl_teleop.synthetic import generate

    shutil.copy(cfg.path, tmp_path / "params.yaml")
    local = load_config(tmp_path / "params.yaml")
    generate(local, local.resolve("ingest.raw_dir"), n_sessions=n_sessions, seed=5)
    return local


def test_deleting_a_raw_session_removes_it_from_the_store(tmp_path, cfg):
    """A retracted session must actually disappear.

    Ingestion is a sync, not an append. Without this, deleting a raw session
    leaves its canonical episodes behind to be scored and trained on forever,
    with nothing in any report saying the source is gone.
    """
    from erl_teleop.ingest import ingest_all
    from erl_teleop.io import iter_episode_metas

    local = _tiny_corpus(tmp_path, cfg)
    raw_root = local.resolve("ingest.raw_dir")
    store = local.resolve("ingest.episode_dir")

    first = ingest_all(local)
    assert first.sessions == 3
    assert first.pruned == []

    victim = sorted(p for p in raw_root.iterdir() if p.is_dir())[0]
    victim_id = victim.name
    shutil.rmtree(victim)

    second = ingest_all(local)
    assert second.sessions == 2
    assert second.pruned == [victim_id]
    assert not (store / victim_id).exists()

    remaining = {m.session_id for m in iter_episode_metas(store)}
    assert victim_id not in remaining
    assert len(remaining) == 2


def test_prune_can_be_disabled(tmp_path, cfg):
    """Ingesting from a partial dump into an existing store must stay possible."""
    from erl_teleop.ingest import ingest_all

    local = _tiny_corpus(tmp_path, cfg)
    raw_root = local.resolve("ingest.raw_dir")
    store = local.resolve("ingest.episode_dir")

    ingest_all(local)
    victim = sorted(p for p in raw_root.iterdir() if p.is_dir())[0]
    shutil.rmtree(victim)

    result = ingest_all(local, prune=False)
    assert result.pruned == []
    assert (store / victim.name).exists()


def test_unreadable_session_metadata_does_not_trigger_pruning(tmp_path, cfg):
    """A corrupt session.json is a skip, not a licence to delete its data."""
    from erl_teleop.ingest import ingest_all

    local = _tiny_corpus(tmp_path, cfg)
    raw_root = local.resolve("ingest.raw_dir")
    store = local.resolve("ingest.episode_dir")

    ingest_all(local)
    victim = sorted(p for p in raw_root.iterdir() if p.is_dir())[0]
    (victim / "session.json").write_text("{ not valid json")

    result = ingest_all(local)

    # The session can no longer be ingested...
    assert any(victim.name in path for path, _ in result.skipped)
    # ...but its directory is still on disk, so its data must survive.
    # "I cannot read this" is not "this no longer exists".
    assert (store / victim.name).exists(), "corrupt metadata deleted good data"
    assert result.pruned == []


def test_schema_column_order_is_stable(cfg):
    cols = timeseries_columns(cfg.n_joints)
    assert cols[0] == "t"
    assert len(cols) == 1 + 3 * cfg.n_joints + 3 + 4 + 1 + 1
    assert len(set(cols)) == len(cols)
