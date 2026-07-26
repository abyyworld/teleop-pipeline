"""End-to-end: synthetic sessions in, trained-ready dataset out.

Runs the whole chain in a temp directory against a copy of the real params.yaml,
so a change that breaks the contract between two stages fails here rather than
in `dvc repro` twenty minutes later.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from erl_teleop.config import load_config
from erl_teleop.dataset import build_dataset, load_manifest
from erl_teleop.ingest import ingest_all
from erl_teleop.quality import score_store, summarise
from erl_teleop.report import render
from erl_teleop.synthetic import generate
from erl_teleop.validate import validate_store

torch = pytest.importorskip("torch", reason="training is an optional extra")


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory, cfg):
    """Run synth -> ingest -> validate -> score -> dataset once, share the result."""
    root = tmp_path_factory.mktemp("pipeline")
    shutil.copy(cfg.path, root / "params.yaml")
    local = load_config(root / "params.yaml")

    generate(local, local.resolve("ingest.raw_dir"), n_sessions=8, seed=3)
    ingest_result = ingest_all(local)

    validation = validate_store(local, quarantine=False)
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports" / "validation.json").write_text(
        json.dumps([r.model_dump() for r in validation], indent=2)
    )

    quality = score_store(local, write_back=False)
    (root / "reports" / "quality.json").write_text(
        json.dumps([r.model_dump() for r in quality], indent=2)
    )

    manifest = build_dataset(local)
    return {
        "cfg": local,
        "root": root,
        "ingest": ingest_result,
        "validation": validation,
        "quality": quality,
        "manifest": manifest,
    }


def test_ingest_produced_episodes(pipeline):
    assert pipeline["ingest"].sessions == 8
    assert pipeline["ingest"].episodes > 20
    assert pipeline["ingest"].skipped == []


def test_every_episode_was_validated(pipeline):
    assert len(pipeline["validation"]) == pipeline["ingest"].episodes


def test_every_episode_was_scored(pipeline):
    quality = pipeline["quality"]
    assert len(quality) == pipeline["ingest"].episodes
    assert all(0.0 <= r.score <= 100.0 for r in quality)
    assert all(r.tier in {"gold", "silver", "reject"} for r in quality)


def test_quality_recovers_the_generators_operator_ranking(pipeline):
    """The scorer never sees the skill levels; it should rediscover them.

    This is the strongest available evidence that the metrics measure
    demonstration quality rather than noise.
    """
    from erl_teleop.synthetic import OPERATORS

    summary = summarise(pipeline["quality"])
    means = summary["per_operator_mean"]
    skill = {o.operator_id: o.skill for o in OPERATORS}

    ranked_by_score = sorted(means, key=lambda op: means[op], reverse=True)
    ranked_by_skill = sorted((op for op in means), key=lambda op: skill[op], reverse=True)
    assert ranked_by_score == ranked_by_skill, (
        f"quality ranking {ranked_by_score} disagrees with generator skill "
        f"ranking {ranked_by_skill}; scores were {means}"
    )


def test_splits_share_no_session(pipeline):
    manifest = pipeline["manifest"]
    train = {e["episode_id"] for e in manifest["splits"]["train"]["episodes"]}
    val = {e["episode_id"] for e in manifest["splits"]["val"]["episodes"]}
    assert train and val
    assert train & val == set()

    root = pipeline["cfg"].resolve("dataset.out_dir")
    train_sessions = set(pd.read_parquet(root / "train.parquet")["session_id"])
    val_sessions = set(pd.read_parquet(root / "val.parquet")["session_id"])
    assert train_sessions & val_sessions == set()


def test_rejected_episodes_are_excluded(pipeline):
    manifest = pipeline["manifest"]
    included = {
        e["episode_id"] for split in ("train", "val") for e in manifest["splits"][split]["episodes"]
    }
    rejected = {r.episode_id for r in pipeline["quality"] if r.tier == "reject"}
    assert included & rejected == set()


def test_failed_demos_are_excluded(pipeline):
    """`require_success: true` must actually drop operator-marked failures."""
    from erl_teleop.io import iter_episode_metas

    failed = {
        m.episode_id
        for m in iter_episode_metas(pipeline["cfg"].resolve("ingest.episode_dir"))
        if not m.success
    }
    manifest = pipeline["manifest"]
    included = {
        e["episode_id"] for split in ("train", "val") for e in manifest["splits"][split]["episodes"]
    }
    assert failed, "fixture should contain some failed demonstrations"
    assert included & failed == set()


def test_manifest_is_reproducible(pipeline):
    """Rebuilding the same store must yield the same dataset hash."""
    original = pipeline["manifest"]["dataset_hash"]
    rebuilt = build_dataset(pipeline["cfg"])["dataset_hash"]
    assert rebuilt == original


def test_norm_stats_come_from_train_only(pipeline):
    """Fitting the normaliser on val would leak val statistics into training."""
    cfg = pipeline["cfg"]
    manifest = load_manifest(cfg)
    root = cfg.resolve("dataset.out_dir")
    train = pd.read_parquet(root / "train.parquet")

    for col, stats in manifest["norm"]["obs"].items():
        assert stats["mean"] == pytest.approx(float(train[col].mean()), rel=1e-4, abs=1e-6)


def test_report_renders(pipeline):
    markdown = render(pipeline["quality"], pipeline["validation"])
    assert markdown.startswith("# ")
    assert "By operator" in markdown
    assert "Metric distributions" in markdown


def test_train_and_eval_round_trip(pipeline):
    """A checkpoint must be loadable and evaluable without params.yaml."""
    from erl_teleop.evaluate import evaluate
    from erl_teleop.train import load_policy
    from erl_teleop.train import train as run_train

    cfg = pipeline["cfg"]
    # Keep the smoke run short; correctness of the plumbing is the point here,
    # not convergence.
    cfg.raw["train"]["epochs"] = 2
    cfg.raw["train"]["hidden_sizes"] = [32, 32]
    cfg.raw["tracking"]["backend"] = "jsonl"
    cfg.raw["evaluate"]["bootstrap_samples"] = 50

    summary = run_train(cfg)
    ckpt = Path(summary["checkpoint"])
    assert ckpt.exists()
    assert summary["dataset_hash"] == pipeline["manifest"]["dataset_hash"]

    policy, meta = load_policy(ckpt)
    assert meta["obs_columns"] == pipeline["manifest"]["obs_columns"]
    assert meta["dataset_hash"] == summary["dataset_hash"]

    report = evaluate(cfg, ckpt)
    metrics = report["metrics"]
    assert metrics["action_mae"] > 0
    lo, hi = metrics["action_mae_ci95"]
    assert lo <= metrics["action_mae"] <= hi
    # Every baseline must be present, or `skill_vs_best_baseline` is meaningless.
    assert {"zero", "train_mean", "persistence"} <= set(metrics["baselines"])


def test_lineage_pins_the_data_version(pipeline):
    lineage = json.loads((pipeline["root"] / "artifacts" / "lineage.json").read_text())
    assert lineage["dataset_hash"] == pipeline["manifest"]["dataset_hash"]
    assert lineage["checkpoint"]["sha256"]
    assert lineage["environment"]["python"]


def test_rollout_does_not_teacher_force(pipeline):
    """Open-loop drift must grow with the horizon.

    If the previous-action observation were fed ground truth instead of the
    policy's own output, drift would stay flat — the bug this guards against
    makes a policy look several times better than it is.
    """
    report = json.loads((pipeline["root"] / "reports" / "eval_val.json").read_text())
    metrics = report["metrics"]
    drifts = [
        metrics[f"rollout_l2@{n}"]
        for n in pipeline["cfg"].get("evaluate.rollout_horizons")
        if metrics.get(f"rollout_l2@{n}") is not None
    ]
    assert len(drifts) >= 2
    assert drifts == sorted(drifts), f"drift did not grow with horizon: {drifts}"
    assert drifts[-1] > drifts[0] * 2
