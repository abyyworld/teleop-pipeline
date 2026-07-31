"""Behaviour-cloning policy training.

Deliberately a small MLP. The point of this repository is the pipeline around
the model — versioned data in, traceable checkpoint out — and a baseline you can
train in a minute is far more useful for validating that pipeline than one that
needs a GPU-hour. Swapping in a real architecture means replacing `BCPolicy` and
nothing else.

Two choices worth defending:

**Action chunking.** The policy predicts the next `action_horizon` actions in
one shot rather than one step. This is the cheapest known mitigation for
compounding error in behaviour cloning — the same idea ACT and the diffusion
policies use — and costs one config line. Set `action_horizon: 1` for vanilla BC.

**Huber loss.** Teleop actions have heavy tails: the operator jerks, corrects,
overshoots. Under plain MSE those tails dominate the gradient and the policy
learns to hedge toward the mean action, which in practice looks like an arm that
drifts and never commits.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .dataset import load_manifest
from .lineage import Lineage
from .tracking import make_tracker


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "training needs PyTorch, which is an optional extra.\n  pip install -e '.[train]'"
        ) from exc
    import torch

    return torch


def resolve_device(requested: str):
    torch = _require_torch()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Windows:
    """Flattened (obs, action-chunk) pairs plus the episode each came from."""

    obs: np.ndarray  # (N, obs_dim)
    actions: np.ndarray  # (N, horizon, action_dim)
    episode_ids: np.ndarray
    # The action executed immediately before each window. Carried so evaluation
    # can score a persistence baseline; unused by training.
    prev_actions: np.ndarray | None = None

    def __len__(self) -> int:
        return self.obs.shape[0]


def build_windows(
    table: pd.DataFrame, obs_cols: list[str], act_cols: list[str], horizon: int
) -> Windows:
    """Cut each episode into overlapping windows.

    Windows never cross an episode boundary — a chunk spanning two demonstrations
    is a trajectory that never happened.
    """
    obs_blocks, act_blocks, ep_blocks, prev_blocks = [], [], [], []

    for episode_id, group in table.groupby("episode_id", sort=True):
        group = group.sort_values("step")
        obs = group[obs_cols].to_numpy(dtype=np.float32)
        act = group[act_cols].to_numpy(dtype=np.float32)
        n_windows = len(group) - horizon + 1
        if n_windows <= 0:
            continue
        # Strided view: window i is act[i : i+horizon].
        idx = np.arange(n_windows)[:, None] + np.arange(horizon)[None, :]
        obs_blocks.append(obs[:n_windows])
        act_blocks.append(act[idx])
        ep_blocks.append(np.full(n_windows, episode_id, dtype=object))
        # Window 0 has no predecessor; reuse its own first action, which makes
        # the persistence baseline marginally optimistic on exactly one window
        # per episode.
        prev_blocks.append(np.vstack([act[:1], act[: n_windows - 1]]))

    if not obs_blocks:
        raise RuntimeError(
            f"no window of length {horizon} fits in any episode; "
            "lower train.action_horizon or check dataset.min_tier"
        )

    return Windows(
        obs=np.concatenate(obs_blocks),
        actions=np.concatenate(act_blocks),
        episode_ids=np.concatenate(ep_blocks),
        prev_actions=np.concatenate(prev_blocks),
    )


def normalise(x: np.ndarray, stats: dict[str, dict[str, float]], cols: list[str]) -> np.ndarray:
    mean = np.asarray([stats[c]["mean"] for c in cols], dtype=np.float32)
    std = np.asarray([stats[c]["std"] for c in cols], dtype=np.float32)
    return (x - mean) / std


def make_policy(obs_dim: int, action_dim: int, horizon: int, cfg: Config):
    _require_torch()  # raises a pointed error instead of a bare ImportError
    from torch import nn

    hidden = [int(h) for h in cfg.get("train.hidden_sizes", [512, 512])]
    dropout = float(cfg.get("train.dropout", 0.1))

    class BCPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            prev = obs_dim
            for width in hidden:
                layers += [nn.Linear(prev, width), nn.LayerNorm(width), nn.GELU()]
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev = width
            layers.append(nn.Linear(prev, horizon * action_dim))
            self.net = nn.Sequential(*layers)
            self.horizon = horizon
            self.action_dim = action_dim

        def forward(self, obs):  # (B, obs_dim) -> (B, horizon, action_dim)
            return self.net(obs).view(-1, self.horizon, self.action_dim)

    return BCPolicy()


def train(
    cfg: Config,
    processed_dir: Path | None = None,
    out_dir: Path | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    torch = _require_torch()
    from torch.utils.data import DataLoader, TensorDataset

    processed = Path(processed_dir) if processed_dir else cfg.resolve("dataset.out_dir")
    out = Path(out_dir) if out_dir else cfg.root / "artifacts"
    out.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(cfg, processed)
    obs_cols = manifest["obs_columns"]
    act_cols = manifest["action_columns"]
    horizon = int(cfg.get("train.action_horizon", 1))
    seed = int(cfg.get("train.seed", 17))

    torch.manual_seed(seed)
    np.random.seed(seed)

    splits = {
        name: build_windows(
            pd.read_parquet(processed / f"{name}.parquet"), obs_cols, act_cols, horizon
        )
        for name in ("train", "val")
    }

    tensors = {}
    for name, w in splits.items():
        obs = normalise(w.obs, manifest["norm"]["obs"], obs_cols)
        act = normalise(
            w.actions.reshape(-1, len(act_cols)), manifest["norm"]["action"], act_cols
        ).reshape(w.actions.shape)
        tensors[name] = TensorDataset(torch.from_numpy(obs).float(), torch.from_numpy(act).float())

    batch_size = int(cfg.get("train.batch_size", 256))
    loaders = {
        "train": DataLoader(tensors["train"], batch_size=batch_size, shuffle=True, drop_last=False),
        "val": DataLoader(tensors["val"], batch_size=batch_size, shuffle=False),
    }

    device = resolve_device(str(cfg.get("train.device", "auto")))
    policy = make_policy(len(obs_cols), len(act_cols), horizon, cfg).to(device)
    optimiser = torch.optim.AdamW(
        policy.parameters(),
        lr=float(cfg.get("train.lr", 3e-4)),
        weight_decay=float(cfg.get("train.weight_decay", 1e-4)),
    )
    epochs = int(cfg.get("train.epochs", 40))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(epochs, 1))
    loss_fn = torch.nn.SmoothL1Loss(beta=0.1)

    # Deterministic by default: the run name is a function of the data it was
    # trained on. Two runs of the same commit on the same data collide by
    # design — that is what makes `dvc repro` a no-op instead of a new run.
    run_name = run_name or f"bc-{manifest['dataset_hash']}"
    tracker = make_tracker(cfg, f"{run_name}-{int(time.time())}", out / "runs")

    patience = int(cfg.get("train.early_stop_patience", 8))
    best_val = float("inf")
    best_epoch = -1
    best_state: dict | None = None
    history: list[dict[str, float]] = []

    with tracker:
        tracker.set_tags(
            {
                "pipeline": "teleop-pipeline",
                "stage": "train",
                "dataset_hash": manifest["dataset_hash"],
            }
        )
        tracker.log_params(
            {
                "train": cfg["train"],
                "dataset": cfg["dataset"],
                "robot": {"n_joints": cfg.n_joints, "control_hz": cfg.control_hz},
                "obs_dim": len(obs_cols),
                "action_dim": len(act_cols),
                "n_train_windows": len(splits["train"]),
                "n_val_windows": len(splits["val"]),
                "device": str(device),
            }
        )

        for epoch in range(epochs):
            policy.train()
            train_loss = 0.0
            n_seen = 0
            for obs_b, act_b in loaders["train"]:
                obs_b, act_b = obs_b.to(device), act_b.to(device)
                optimiser.zero_grad(set_to_none=True)
                loss = loss_fn(policy(obs_b), act_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimiser.step()
                train_loss += loss.item() * obs_b.size(0)
                n_seen += obs_b.size(0)
            scheduler.step()

            val_loss = _evaluate_loss(policy, loaders["val"], loss_fn, device)
            row = {
                "epoch": epoch,
                "train_loss": train_loss / max(n_seen, 1),
                "val_loss": val_loss,
                "lr": scheduler.get_last_lr()[0],
            }
            history.append(row)
            tracker.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)

            if val_loss < best_val - 1e-6:
                best_val, best_epoch = val_loss, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
            elif epoch - best_epoch >= patience:
                print(
                    f"[train] early stop at epoch {epoch} (best {best_epoch}, val {best_val:.5f})"
                )
                break

        if best_state is not None:
            policy.load_state_dict(best_state)

        # The checkpoint carries everything needed to run the policy standalone:
        # weights, the exact normalisation, and the column order. A checkpoint
        # that needs the repo's config to be interpretable is not portable, and
        # policy-eval-harness loads these without importing this package.
        ckpt_path = out / "policy.pt"
        torch.save(
            {
                "format_version": 1,
                "state_dict": policy.state_dict(),
                "obs_columns": obs_cols,
                "action_columns": act_cols,
                "action_horizon": horizon,
                "norm": manifest["norm"],
                "arch": {
                    "hidden_sizes": list(cfg.get("train.hidden_sizes", [512, 512])),
                    "dropout": float(cfg.get("train.dropout", 0.1)),
                },
                "dataset_hash": manifest["dataset_hash"],
                "run_name": run_name,
            },
            ckpt_path,
        )

        summary = {
            "run_name": run_name,
            "run_id": tracker.run_id,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "epochs_run": len(history),
            "n_train_windows": len(splits["train"]),
            "n_val_windows": len(splits["val"]),
            "checkpoint": str(ckpt_path),
            "dataset_hash": manifest["dataset_hash"],
            "device": str(device),
        }
        tracker.log_metrics({"best_val_loss": best_val, "best_epoch": float(best_epoch)})

        lineage = Lineage.build(
            root=cfg.root,
            run_id=tracker.run_id or run_name,
            run_name=run_name,
            dataset_hash=manifest["dataset_hash"],
            params={"train": cfg["train"], "dataset": cfg["dataset"], "robot": cfg["robot"]},
        )
        lineage.metrics = {"best_val_loss": best_val}
        lineage.attach_checkpoint(ckpt_path, {"format_version": 1})
        lineage.tracking = {
            "backend": str(cfg.get("tracking.backend")),
            "uri": cfg.tracking_uri,
            "run_id": tracker.run_id,
        }
        # Fixed paths so DVC can declare them as stage outputs. The per-run
        # copy under runs/ is for browsing history; the top-level pair is the
        # pipeline's contract.
        lineage_path = lineage.write(out / "lineage.json")
        lineage.write(out / "runs" / run_name / "lineage.json")
        (out / "history.json").write_text(json.dumps(history, indent=2))
        tracker.log_artifact(lineage_path)

    summary["lineage"] = str(lineage_path)
    # `train_summary.json` is declared as a DVC metrics file, so `dvc metrics
    # diff` reports the change in val loss between any two commits.
    (out / "train_summary.json").write_text(
        json.dumps(
            {
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "epochs_run": len(history),
                "n_train_windows": len(splits["train"]),
                "n_val_windows": len(splits["val"]),
            },
            indent=2,
        )
    )
    (out / "latest_run.json").write_text(json.dumps(summary, indent=2))
    return summary


def _evaluate_loss(policy, loader, loss_fn, device) -> float:
    torch = _require_torch()
    policy.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for obs_b, act_b in loader:
            obs_b, act_b = obs_b.to(device), act_b.to(device)
            total += loss_fn(policy(obs_b), act_b).item() * obs_b.size(0)
            n += obs_b.size(0)
    return total / max(n, 1)


def load_policy(checkpoint_path: Path, device=None):
    """Rebuild a policy from a checkpoint alone, without params.yaml."""
    torch = _require_torch()
    from torch import nn

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    obs_dim = len(ckpt["obs_columns"])
    action_dim = len(ckpt["action_columns"])
    horizon = int(ckpt["action_horizon"])
    hidden = ckpt["arch"]["hidden_sizes"]
    dropout = float(ckpt["arch"]["dropout"])

    layers: list[nn.Module] = []
    prev = obs_dim
    for width in hidden:
        layers += [nn.Linear(prev, width), nn.LayerNorm(width), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = width
    layers.append(nn.Linear(prev, horizon * action_dim))

    class _Loaded(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(*layers)
            self.horizon = horizon
            self.action_dim = action_dim

        def forward(self, obs):
            return self.net(obs).view(-1, self.horizon, self.action_dim)

    policy = _Loaded()
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    if device is not None:
        policy = policy.to(device)
    return policy, ckpt
