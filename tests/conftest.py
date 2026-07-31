from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from teleop_pipeline.config import Config, load_config
from teleop_pipeline.schema import EpisodeMeta, timeseries_columns


@pytest.fixture(scope="session")
def cfg() -> Config:
    """The real params.yaml — tests should fail if the shipped config is broken."""
    return load_config()


@pytest.fixture
def clean_episode(cfg: Config):
    """A synthetic episode with no defects, as a baseline for the metric tests.

    Deliberately hand-built rather than drawn from the generator: a test that
    asserts "the scorer catches dropped frames" is only meaningful if the
    control case provably has none.
    """

    def _make(n_steps: int = 200) -> pd.DataFrame:
        n = cfg.n_joints
        t = np.arange(n_steps) * cfg.dt
        phase = np.linspace(0, 2 * np.pi, n_steps)
        df = pd.DataFrame({"t": t})
        # Oscillate about each joint's own midpoint, scaled to its own range.
        # Panda joint 3 is limited to [-3.07, -0.07] — a trajectory centred on
        # zero violates it, which the validator correctly rejects.
        centre = 0.5 * (cfg.joint_lower + cfg.joint_upper)
        amplitude = 0.3 * (cfg.joint_upper - cfg.joint_lower) / 2.0
        for j in range(n):
            df[f"q_{j}"] = centre[j] + amplitude[j] * np.sin(phase + j)
        for j in range(n):
            df[f"dq_{j}"] = np.gradient(df[f"q_{j}"].to_numpy(), cfg.dt)
        df["ee_x"], df["ee_y"], df["ee_z"] = 0.3, 0.0, 0.4
        df["ee_qx"], df["ee_qy"], df["ee_qz"], df["ee_qw"] = 0.0, 0.0, 0.0, 1.0
        df["grip"] = np.where(np.arange(n_steps) < n_steps // 2, 1.0, 0.0)
        for j in range(n):
            q = df[f"q_{j}"].to_numpy()
            df[f"act_q_{j}"] = np.append(np.diff(q), 0.0)
        df["act_grip"] = np.roll(df["grip"].to_numpy(), -1)
        return df[timeseries_columns(n)].astype(
            {c: np.float32 for c in timeseries_columns(n) if c != "t"}
        )

    return _make


@pytest.fixture
def episode_meta():
    def _make(df: pd.DataFrame, **overrides) -> EpisodeMeta:
        base = {
            "episode_id": "ep_test",
            "session_id": "sess_test",
            "operator_id": "op_test",
            "task_id": "task_test",
            "success": True,
            "n_steps": len(df),
            "duration_s": float(df["t"].iloc[-1]) if len(df) else 0.0,
            "control_hz": 20.0,
        }
        base.update(overrides)
        return EpisodeMeta(**base)

    return _make
