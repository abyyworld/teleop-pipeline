"""Offline evaluation of a trained policy.

Scope, stated plainly: there is **no simulator in this repository**, so nothing
here measures task success. These are open-loop metrics on held-out
demonstrations. They are useful — they catch a broken checkpoint, a
normalisation mismatch, or a regression between data versions — and they are not
a substitute for closed-loop evaluation. Closed-loop benchmarking lives in
``erl-vla-evals``, which consumes the checkpoint and lineage this stage emits.

What is measured:

``action_mae`` / ``action_rmse``
    Error in physical units (rad, and gripper fraction), not normalised units.
    Normalised losses are not comparable across data versions, because the
    normaliser changes with the data.
``per_joint_mae``
    Same, per joint. A single joint dominating the error is the usual signature
    of a wrist the operator barely used.
``gripper_accuracy``
    Open/close agreement, thresholded. Reported separately from joint error
    because gripper timing is what manipulation policies actually fail at, and
    it is one dimension out of eight so it vanishes in an aggregate.
``rollout_l2@N``
    Compounding error: run the policy open-loop for N steps from a held-out
    state, integrating its own predicted deltas, and measure drift from the
    recorded trajectory. This is the number that degrades when action chunking
    is turned off, and the closest offline proxy for closed-loop behaviour.

Confidence intervals use a **cluster bootstrap resampling whole episodes**, not
individual timesteps. Timesteps within an episode are strongly correlated;
bootstrapping over them treats ~500 correlated samples as 500 independent ones
and produces intervals several times too narrow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .dataset import load_manifest
from .train import _require_torch, build_windows, load_policy, normalise, resolve_device


def denormalise(x: np.ndarray, stats: dict[str, dict[str, float]], cols: list[str]) -> np.ndarray:
    mean = np.asarray([stats[c]["mean"] for c in cols], dtype=np.float32)
    std = np.asarray([stats[c]["std"] for c in cols], dtype=np.float32)
    return x * std + mean


def cluster_bootstrap_ci(
    values: np.ndarray,
    groups: np.ndarray,
    n_samples: int = 1000,
    seed: int = 17,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI for the mean, resampling whole groups (episodes)."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    if unique.size < 2:
        return (float("nan"), float("nan"))

    index: dict[Any, np.ndarray] = {g: np.flatnonzero(groups == g) for g in unique}
    means = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        picked = rng.choice(unique, size=unique.size, replace=True)
        means[i] = np.mean(np.concatenate([values[index[g]] for g in picked]))
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _predict(policy, obs: np.ndarray, device, batch_size: int = 4096) -> np.ndarray:
    torch = _require_torch()
    outs = []
    with torch.no_grad():
        for start in range(0, obs.shape[0], batch_size):
            batch = torch.from_numpy(obs[start : start + batch_size]).float().to(device)
            outs.append(policy(batch).cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, 1, 1), dtype=np.float32)


def rollout_drift(
    policy,
    table: pd.DataFrame,
    obs_cols: list[str],
    act_cols: list[str],
    norm: dict,
    horizon: int,
    n_steps: int,
    device,
    n_joints: int,
) -> np.ndarray:
    """Open-loop drift after `n_steps`, one value per starting window.

    The policy is re-queried every `horizon` steps (as it would be deployed) and
    its own predicted joint deltas are integrated forward. Only the joint part of
    the observation is updated from the prediction; the remaining channels are
    held at their recorded values, since without a simulator we cannot evolve
    them. That makes this an optimistic bound on true compounding error — stated
    here rather than buried, because an optimistic metric presented as exact is
    worse than no metric.
    """
    q_cols = [c for c in obs_cols if c.startswith("q_")]
    if not q_cols:
        return np.asarray([])
    q_idx = [obs_cols.index(c) for c in q_cols]
    act_q_idx = [i for i, c in enumerate(act_cols) if c.startswith("act_q_")]
    if len(act_q_idx) != len(q_idx):
        return np.asarray([])

    # If the observation feeds back the previous action, that slot must be
    # filled with what the policy *itself* just predicted. Leaving the recorded
    # value there hands the policy the ground-truth action every step and turns
    # an open-loop rollout into a teacher-forced one — which reports drift
    # several times lower than the robot would actually see.
    feedback: list[tuple[int, int]] = []
    for obs_i, name in enumerate(obs_cols):
        if not name.startswith("prev_"):
            continue
        source = name[len("prev_") :]
        if source in act_cols:
            feedback.append((obs_i, act_cols.index(source)))

    drifts: list[float] = []
    for _, group in table.groupby("episode_id", sort=True):
        group = group.sort_values("step")
        obs_raw = group[obs_cols].to_numpy(dtype=np.float32)
        if obs_raw.shape[0] <= n_steps:
            continue

        # A handful of evenly spaced starts per episode: enough signal without
        # making evaluation cost scale with dataset size.
        starts = np.linspace(0, obs_raw.shape[0] - n_steps - 1, num=5, dtype=int)
        for start in np.unique(starts):
            state = obs_raw[start].copy()
            chunk: np.ndarray | None = None
            for step in range(n_steps):
                if step % horizon == 0:
                    normed = normalise(state[None, :], norm["obs"], obs_cols)
                    pred = _predict(policy, normed, device)[0]
                    chunk = denormalise(
                        pred.reshape(-1, len(act_cols)), norm["action"], act_cols
                    ).reshape(pred.shape)
                assert chunk is not None
                action = chunk[step % horizon]
                state[q_idx] = state[q_idx] + action[act_q_idx]
                for obs_i, act_i in feedback:
                    state[obs_i] = action[act_i]
            truth = obs_raw[start + n_steps][q_idx]
            drifts.append(float(np.linalg.norm(state[q_idx] - truth)))

    return np.asarray(drifts, dtype=np.float64)


def compute_baselines(
    windows, norm: dict, act_cols: list[str], horizon: int
) -> dict[str, dict[str, float]]:
    """Action MAE for three trivial predictors, in the model's own units.

    ``zero``
        Command no motion. The floor: a policy that cannot beat this has
        learned nothing at all.
    ``train_mean``
        Always emit the training-set mean action. Beats ``zero`` whenever the
        task has a directional bias — a bias the model gets for free.
    ``persistence``
        Repeat the previous action. On smooth 20 Hz teleop this is a genuinely
        strong predictor, and it is the one that embarrasses behaviour-cloning
        results reported only against zero.

    Each is scored two ways, because the difference is not cosmetic. A
    persistence baseline asked for one step is far stronger than the same
    baseline asked to hold that action for the whole chunk, while the policy is
    trained on the chunk. Comparing the policy's chunk output against a
    one-step baseline understates the policy, and it is the more flattering
    direction that tends to get published — so both are reported and
    ``skill_vs_best_baseline`` uses the chunk-level figure.
    """
    truth = windows.actions  # (N, H, A)
    mean_action = np.asarray([norm["action"][c]["mean"] for c in act_cols], dtype=np.float32)

    predictions: dict[str, np.ndarray] = {
        "zero": np.zeros_like(truth),
        "train_mean": np.broadcast_to(mean_action, truth.shape),
    }
    if windows.prev_actions is not None:
        # Hold the previous action for the whole chunk — the honest extension of
        # "repeat the last action" to a horizon.
        predictions["persistence"] = np.repeat(windows.prev_actions[:, None, :], horizon, axis=1)

    return {
        name: {
            "next_step_mae": float(np.abs(pred[:, 0, :] - truth[:, 0, :]).mean()),
            "chunk_mae": float(np.abs(pred - truth).mean()),
            # Per-offset, because the aggregate hides where a policy earns its
            # keep. A learned policy that loses to persistence at offset 0 and
            # wins by offset 6 is doing exactly what a policy should do:
            # committing to a plan instead of extrapolating the last command.
            "mae_by_offset": [
                float(np.abs(pred[:, k, :] - truth[:, k, :]).mean()) for k in range(horizon)
            ],
        }
        for name, pred in predictions.items()
    }


def evaluate(
    cfg: Config,
    checkpoint: Path,
    processed_dir: Path | None = None,
    out_dir: Path | None = None,
    split: str = "val",
) -> dict[str, Any]:
    processed = Path(processed_dir) if processed_dir else cfg.resolve("dataset.out_dir")
    out = Path(out_dir) if out_dir else cfg.root / "reports"
    out.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(cfg, processed)
    obs_cols = manifest["obs_columns"]
    act_cols = manifest["action_columns"]
    norm = manifest["norm"]

    device = resolve_device(str(cfg.get("train.device", "auto")))
    policy, ckpt = load_policy(Path(checkpoint), device)
    horizon = int(ckpt["action_horizon"])

    if ckpt["obs_columns"] != obs_cols or ckpt["action_columns"] != act_cols:
        raise RuntimeError(
            "checkpoint column layout does not match the dataset manifest — "
            "the checkpoint was trained on a different feature set"
        )
    if ckpt.get("dataset_hash") != manifest["dataset_hash"]:
        # Not fatal: evaluating an old checkpoint on new data is exactly how you
        # detect a data regression. But it must be visible in the report.
        print(
            f"[eval] note: checkpoint dataset {ckpt.get('dataset_hash')} "
            f"!= current dataset {manifest['dataset_hash']}"
        )

    table = pd.read_parquet(processed / f"{split}.parquet")
    windows = build_windows(table, obs_cols, act_cols, horizon)

    obs_n = normalise(windows.obs, norm["obs"], obs_cols)
    pred_n = _predict(policy, obs_n, device)
    pred = denormalise(pred_n.reshape(-1, len(act_cols)), norm["action"], act_cols).reshape(
        pred_n.shape
    )
    truth = windows.actions

    # Errors on the immediate next action — the one that would actually execute.
    err = np.abs(pred[:, 0, :] - truth[:, 0, :])
    per_sample_mae = err.mean(axis=1)
    groups = windows.episode_ids

    n_boot = int(cfg.get("evaluate.bootstrap_samples", 1000))
    boot_seed = int(cfg.get("evaluate.bootstrap_seed", 17))
    ci_lo, ci_hi = cluster_bootstrap_ci(per_sample_mae, groups, n_boot, boot_seed)

    joint_cols_idx = [i for i, c in enumerate(act_cols) if c.startswith("act_q_")]
    grip_idx = [i for i, c in enumerate(act_cols) if c == "act_grip"]

    metrics: dict[str, Any] = {
        "split": split,
        "n_windows": int(len(windows)),
        "n_episodes": int(np.unique(groups).size),
        "action_mae": float(per_sample_mae.mean()),
        "action_mae_ci95": [ci_lo, ci_hi],
        "action_rmse": float(np.sqrt(((pred[:, 0, :] - truth[:, 0, :]) ** 2).mean())),
        "per_joint_mae": {act_cols[i]: float(err[:, i].mean()) for i in joint_cols_idx},
        # Chunk-averaged error shows how fast prediction quality decays across
        # the horizon; a steep slope means the horizon is set too long.
        "chunk_mae_by_offset": [
            float(np.abs(pred[:, k, :] - truth[:, k, :]).mean()) for k in range(horizon)
        ],
    }

    if grip_idx:
        gi = grip_idx[0]
        metrics["gripper_mae"] = float(err[:, gi].mean())
        metrics["gripper_accuracy"] = float(
            np.mean((pred[:, 0, gi] > 0.5) == (truth[:, 0, gi] > 0.5))
        )

    # -- baselines ---------------------------------------------------------
    # An action MAE on its own is uninterpretable: whether 0.013 rad is good
    # depends entirely on how much the arm moves. These three say what the
    # number has to beat to mean anything, and `skill_vs_best_baseline` is the
    # single number worth putting in a paper.
    baselines = compute_baselines(windows, norm, act_cols, horizon)
    metrics["baselines"] = baselines
    metrics["chunk_mae"] = float(np.abs(pred - truth).mean())

    best_name = min(baselines, key=lambda k: baselines[k]["chunk_mae"])
    best_chunk = baselines[best_name]["chunk_mae"]
    metrics["best_baseline"] = best_name
    metrics["best_baseline_chunk_mae"] = best_chunk
    metrics["skill_vs_best_baseline"] = (
        float(1.0 - metrics["chunk_mae"] / best_chunk) if best_chunk > 0 else None
    )
    # Reported alongside so the horizon-mismatch effect stays visible rather
    # than being hidden behind a single headline number.
    best_next = min(b["next_step_mae"] for b in baselines.values())
    metrics["skill_vs_best_baseline_next_step"] = (
        float(1.0 - metrics["action_mae"] / best_next) if best_next > 0 else None
    )

    for n_steps in cfg.get("evaluate.rollout_horizons", [1, 8, 32]):
        drift = rollout_drift(
            policy, table, obs_cols, act_cols, norm, horizon, int(n_steps), device, cfg.n_joints
        )
        metrics[f"rollout_l2@{n_steps}"] = float(drift.mean()) if drift.size else None

    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_dataset_hash": ckpt.get("dataset_hash"),
        "eval_dataset_hash": manifest["dataset_hash"],
        "action_horizon": horizon,
        "device": str(device),
        "metrics": metrics,
    }
    path = out / f"eval_{split}.json"
    path.write_text(json.dumps(report, indent=2))

    # Flat metrics file for `dvc metrics show` / `dvc metrics diff`, which only
    # understands scalars at the top level.
    flat = {
        k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    (out / f"eval_{split}_metrics.json").write_text(json.dumps(flat, indent=2))
    return report
