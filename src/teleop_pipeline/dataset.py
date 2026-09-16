"""Build train/val splits and the dataset manifest.

Three decisions in here are the ones that most often silently ruin a robot
learning result, so they are enforced rather than left to the caller:

1. **Group the split on session, not episode.** Episodes recorded in one sitting
   share an operator, a calibration and a scene layout. Splitting per-episode
   puts near-duplicates on both sides and produces validation numbers that do
   not survive contact with a new session.
2. **Fit normalisation statistics on the training split only.** Fitting on the
   full set leaks val statistics into training. It is a small leak and it makes
   every subsequent comparison slightly dishonest.
3. **Deduplicate by content hash.** Re-ingested sessions are routine — someone
   re-copies a folder — and duplicate demonstrations quietly reweight the
   dataset toward whatever happened to be copied twice.

The manifest is the unit of data versioning: it pins the exact episode content
hashes that went in, so any run can be traced back to a byte-exact dataset even
if the store has moved on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .io import iter_episodes
from .schema import EpisodeMeta, action_columns, observation_columns

TIER_ORDER = {"reject": 0, "silver": 1, "gold": 2}

# Populated by `load_gate`; keeps quality scores available for the manifest
# without threading them through every call site.
_SCORES: dict[str, float] = {}


@dataclass
class Split:
    name: str
    episode_ids: list[str]
    session_ids: list[str]
    n_steps: int


def load_gate(cfg: Config) -> tuple[dict[str, str], set[str]]:
    """Read the quality tiers and validation failures produced upstream.

    The tiers come from `reports/quality.json` rather than from the episode
    sidecars. That keeps every DVC stage purely functional — no stage rewrites
    another stage's outputs — which is what makes `dvc repro` able to tell what
    is actually stale. `score` can still write tiers back into the sidecars for
    interactive use; the pipeline just does not depend on it having done so.
    """
    reports = cfg.root / "reports"
    tiers: dict[str, str] = {}
    quality_path = reports / "quality.json"
    if quality_path.exists():
        for row in json.loads(quality_path.read_text(encoding="utf-8")):
            tiers[row["episode_id"]] = row["tier"]
            _SCORES[row["episode_id"]] = float(row["score"])

    invalid: set[str] = set()
    validation_path = reports / "validation.json"
    if validation_path.exists():
        for row in json.loads(validation_path.read_text(encoding="utf-8")):
            if not row.get("ok", True):
                invalid.add(row["episode_id"])

    return tiers, invalid


def _passes_filters(
    cfg: Config, meta: EpisodeMeta, tiers: dict[str, str], invalid: set[str]
) -> tuple[bool, str]:
    if meta.episode_id in invalid:
        return False, "invalid"

    min_tier = str(cfg.get("dataset.min_tier", "silver"))
    tier = tiers.get(meta.episode_id, meta.quality_tier)
    if tier is None:
        return False, "unscored"
    if TIER_ORDER[tier] < TIER_ORDER[min_tier]:
        return False, f"tier:{tier}"
    if bool(cfg.get("dataset.require_success", True)) and not meta.success:
        return False, "failed"
    return True, ""


def assign_splits(session_ids: list[str], val_fraction: float, seed: int) -> dict[str, str]:
    """Deterministically assign whole sessions to train/val.

    Hash-based rather than shuffle-based: adding a new session later must not
    reshuffle the existing ones, or every prior result becomes incomparable.
    """
    assignment: dict[str, str] = {}
    for sid in sorted(set(session_ids)):
        digest = hashlib.sha256(f"{seed}:{sid}".encode()).digest()
        # Top 32 bits as a uniform draw in [0, 1).
        draw = int.from_bytes(digest[:4], "big") / 2**32
        assignment[sid] = "val" if draw < val_fraction else "train"
    return assignment


def add_prev_action_columns(df: pd.DataFrame, act_cols: list[str]) -> pd.DataFrame:
    """Add `prev_<action>` columns by shifting each action one step forward.

    Done here rather than at ingest because it is a modelling choice, not a
    property of the recording — and because the shift must never cross an
    episode boundary, which is only guaranteed once episodes are separated.
    """
    df = df.copy()
    for col in act_cols:
        shifted = df[col].shift(1)
        # The first step has no predecessor. Seed it with its own action rather
        # than zero: zero is a real, meaningful action ("hold still") and
        # injecting it would teach the policy that every episode starts frozen.
        shifted.iloc[0] = df[col].iloc[0] if len(df) else 0.0
        df[f"prev_{col}"] = shifted
    return df


def compute_norm_stats(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    """Per-column mean/std over the training split only.

    Std is floored: constant channels (a joint the task never uses, a gripper
    that never opens) otherwise divide by ~0 and produce inf features.
    """
    stats: dict[str, dict[str, float]] = {}
    for col in columns:
        values = df[col].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        mean = float(values.mean()) if values.size else 0.0
        std = float(values.std()) if values.size else 1.0
        stats[col] = {"mean": mean, "std": max(std, 1e-6)}
    return stats


def build_dataset(
    cfg: Config, episode_root: Path | None = None, out_dir: Path | None = None
) -> dict:
    root = Path(episode_root) if episode_root else cfg.resolve("ingest.episode_dir")
    out = Path(out_dir) if out_dir else cfg.resolve("dataset.out_dir")
    out.mkdir(parents=True, exist_ok=True)

    n = cfg.n_joints
    obs_cols = observation_columns(n, list(cfg.get("dataset.obs_keys")))
    act_cols = action_columns(n, list(cfg.get("dataset.action_keys")))

    val_fraction = float(cfg.get("dataset.val_fraction", 0.2))
    seed = int(cfg.get("dataset.seed", 17))

    tiers, invalid = load_gate(cfg)

    kept: list[tuple[EpisodeMeta, pd.DataFrame]] = []
    excluded: dict[str, int] = {}
    seen_hashes: dict[str, str] = {}

    for meta, df in iter_episodes(root):
        ok, reason = _passes_filters(cfg, meta, tiers, invalid)
        if not ok:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        if meta.content_hash and meta.content_hash in seen_hashes:
            excluded["duplicate"] = excluded.get("duplicate", 0) + 1
            continue
        seen_hashes[meta.content_hash] = meta.episode_id
        kept.append((meta, df))

    if not kept:
        raise RuntimeError(
            "no episodes passed the dataset filters — run `teleop-pipeline score` first, "
            "or relax dataset.min_tier in params.yaml"
        )

    assignment = assign_splits([m.session_id for m, _ in kept], val_fraction, seed)

    frames: dict[str, list[pd.DataFrame]] = {"train": [], "val": []}
    members: dict[str, list[EpisodeMeta]] = {"train": [], "val": []}

    for meta, df in kept:
        split = assignment[meta.session_id]
        df = add_prev_action_columns(df, act_cols)
        # Rows with a hole in any needed channel cannot be a training pair and
        # must not be silently zero-filled.
        block = df[["t", *obs_cols, *act_cols]].dropna().reset_index(drop=True)
        if block.empty:
            excluded["all_nan"] = excluded.get("all_nan", 0) + 1
            continue
        block.insert(0, "episode_id", meta.episode_id)
        block.insert(1, "session_id", meta.session_id)
        block.insert(2, "step", np.arange(len(block), dtype=np.int32))
        frames[split].append(block)
        members[split].append(meta)

    if not frames["train"]:
        raise RuntimeError(
            f"training split is empty (val_fraction={val_fraction}); "
            "too few distinct sessions to split on session_id"
        )
    if not frames["val"]:
        raise RuntimeError(
            f"validation split is empty (val_fraction={val_fraction}); "
            "collect more sessions or lower dataset.val_fraction"
        )

    tables = {k: pd.concat(v, ignore_index=True) for k, v in frames.items()}
    # Fit on train only — see module docstring.
    norm = {
        "obs": compute_norm_stats(tables["train"], obs_cols),
        "action": compute_norm_stats(tables["train"], act_cols),
    }

    for split, table in tables.items():
        table.to_parquet(out / f"{split}.parquet", index=False, compression="zstd")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "obs_columns": obs_cols,
        "action_columns": act_cols,
        "norm": norm,
        "filters": {
            "min_tier": cfg.get("dataset.min_tier"),
            "require_success": cfg.get("dataset.require_success"),
            "split_on": cfg.get("dataset.split_on"),
            "val_fraction": val_fraction,
            "seed": seed,
        },
        "excluded": excluded,
        "splits": {
            split: {
                "n_episodes": len(members[split]),
                "n_sessions": len({m.session_id for m in members[split]}),
                "n_steps": int(len(tables[split])),
                "operators": sorted({m.operator_id for m in members[split]}),
                "tasks": sorted({m.task_id for m in members[split]}),
                "tiers": _count(tiers.get(m.episode_id, m.quality_tier) for m in members[split]),
                "mean_quality": round(
                    float(
                        np.mean(
                            [
                                _SCORES.get(m.episode_id, m.quality_score or 0.0)
                                for m in members[split]
                            ]
                        )
                    ),
                    2,
                ),
                "episodes": [
                    {"episode_id": m.episode_id, "content_hash": m.content_hash}
                    for m in sorted(members[split], key=lambda m: m.episode_id)
                ],
            }
            for split in ("train", "val")
        },
    }
    manifest["dataset_hash"] = dataset_hash(manifest)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        key = str(v)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def dataset_hash(manifest: dict) -> str:
    """Content-addressed identity of the dataset.

    Derived from the episode content hashes plus the filter settings, so it is
    stable across rebuilds on different machines and changes if and only if the
    actual data or the selection rule changed. This is the string that gets
    logged alongside every training run.
    """
    h = hashlib.sha256()
    h.update(json.dumps(manifest["filters"], sort_keys=True).encode())
    h.update(json.dumps(manifest["obs_columns"]).encode())
    h.update(json.dumps(manifest["action_columns"]).encode())
    for split in ("train", "val"):
        h.update(split.encode())
        for ep in manifest["splits"][split]["episodes"]:
            h.update(ep["episode_id"].encode())
            h.update(ep["content_hash"].encode())
    return h.hexdigest()[:16]


def load_manifest(cfg: Config, out_dir: Path | None = None) -> dict:
    out = Path(out_dir) if out_dir else cfg.resolve("dataset.out_dir")
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))
