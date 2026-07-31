"""Tests for the properties that make results trustworthy rather than merely produced.

Every test here corresponds to a way of accidentally inflating a validation
number. They are the reason to prefer this pipeline over a folder of CSVs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from teleop_pipeline import dataset
from teleop_pipeline.schema import action_columns, observation_columns


def test_split_is_grouped_by_session():
    """No session may appear on both sides. This is the leakage that matters."""
    sessions = [f"sess_{i:03d}" for i in range(200)]
    assignment = dataset.assign_splits(sessions, val_fraction=0.25, seed=17)
    train = {s for s, v in assignment.items() if v == "train"}
    val = {s for s, v in assignment.items() if v == "val"}
    assert train & val == set()
    assert train | val == set(sessions)


def test_split_is_deterministic():
    a = dataset.assign_splits([f"s{i}" for i in range(100)], 0.2, seed=17)
    b = dataset.assign_splits([f"s{i}" for i in range(100)], 0.2, seed=17)
    assert a == b


def test_split_is_stable_when_sessions_are_added():
    """Adding new sessions must not reshuffle the existing ones.

    A shuffle-based split silently reassigns old sessions whenever the corpus
    grows, which makes every previously reported number incomparable — and
    nothing in the output tells you it happened.
    """
    original = [f"sess_{i:03d}" for i in range(50)]
    grown = original + [f"sess_{i:03d}" for i in range(50, 90)]

    before = dataset.assign_splits(original, 0.2, seed=17)
    after = dataset.assign_splits(grown, 0.2, seed=17)

    for sid in original:
        assert before[sid] == after[sid], f"{sid} changed split when the corpus grew"


def test_split_respects_the_requested_fraction():
    assignment = dataset.assign_splits([f"s{i:04d}" for i in range(4000)], 0.25, seed=3)
    share = sum(v == "val" for v in assignment.values()) / len(assignment)
    assert share == pytest.approx(0.25, abs=0.02)


def test_norm_stats_floor_constant_columns():
    """A channel that never varies must not produce a divide-by-zero feature."""
    df = pd.DataFrame({"constant": np.ones(100), "varying": np.arange(100.0)})
    stats = dataset.compute_norm_stats(df, ["constant", "varying"])
    assert stats["constant"]["std"] >= 1e-6
    assert np.isfinite((1.0 - stats["constant"]["mean"]) / stats["constant"]["std"])
    assert stats["varying"]["std"] > 1.0


def test_norm_stats_ignore_nan():
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 3.0]})
    stats = dataset.compute_norm_stats(df, ["x"])
    assert stats["x"]["mean"] == pytest.approx(2.0)


def test_prev_action_columns_do_not_leak_the_current_action():
    act_cols = ["act_q_0", "act_grip"]
    df = pd.DataFrame({"act_q_0": [0.1, 0.2, 0.3, 0.4], "act_grip": [1.0, 1.0, 0.0, 0.0]})
    out = dataset.add_prev_action_columns(df, act_cols)

    # Each prev_* value is the preceding step's action, never the current one.
    assert list(out["prev_act_q_0"])[1:] == pytest.approx([0.1, 0.2, 0.3])
    # First step has no predecessor and is seeded with its own action.
    assert out["prev_act_q_0"].iloc[0] == pytest.approx(0.1)
    assert not out["prev_act_q_0"].isna().any()


def test_dataset_hash_changes_with_content():
    base = {
        "filters": {"min_tier": "silver"},
        "obs_columns": ["q_0"],
        "action_columns": ["act_q_0"],
        "splits": {
            "train": {"episodes": [{"episode_id": "a", "content_hash": "h1"}]},
            "val": {"episodes": [{"episode_id": "b", "content_hash": "h2"}]},
        },
    }
    original = dataset.dataset_hash(base)

    changed = {**base, "filters": {"min_tier": "gold"}}
    assert dataset.dataset_hash(changed) != original

    import copy

    recontent = copy.deepcopy(base)
    recontent["splits"]["train"]["episodes"][0]["content_hash"] = "h9"
    assert dataset.dataset_hash(recontent) != original

    # Rebuilding the same selection must produce the same hash.
    assert dataset.dataset_hash(copy.deepcopy(base)) == original


def test_observation_columns_resolve(cfg):
    cols = observation_columns(cfg.n_joints, ["q", "grip"])
    assert cols == [f"q_{i}" for i in range(cfg.n_joints)] + ["grip"]

    with pytest.raises(KeyError):
        observation_columns(cfg.n_joints, ["not_a_key"])


def test_action_columns_resolve(cfg):
    cols = action_columns(cfg.n_joints, ["act_q", "act_grip"])
    assert cols == [f"act_q_{i}" for i in range(cfg.n_joints)] + ["act_grip"]


def test_shipped_obs_keys_are_valid(cfg):
    """The config that ships must resolve — a typo here breaks `dvc repro`."""
    obs = observation_columns(cfg.n_joints, list(cfg.get("dataset.obs_keys")))
    act = action_columns(cfg.n_joints, list(cfg.get("dataset.action_keys")))
    assert obs and act
    assert not set(obs) & set(act), "an action column is also being fed in as an observation"
