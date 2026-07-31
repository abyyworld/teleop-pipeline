"""Synthetic teleop session generator.

Exists for two reasons, both practical:

1. **A fresh clone runs end to end.** Nobody evaluating this repository has
   access to the lab's demonstrations. `teleop-pipeline synth && dvc repro` works on
   any machine, which is the difference between a pipeline someone can assess
   and a pipeline they have to take on faith.
2. **The quality scorer has known-bad input.** Each defect below is injected
   deliberately and independently, so the tests can assert that the metric
   meant to catch it actually does. A quality gate that has never been shown a
   bad episode is decoration.

Operators are given differing skill levels, so the per-operator breakdown in the
report has real signal rather than noise. Output is written as *messy* CSV —
`joint_0`, `vel_0`, `gripper`, `timestamp` — to exercise the alias mapping in
`ingest.py` rather than a format that happens to already be canonical.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .schema import SessionMeta

TASKS = ["pick_place_block", "open_drawer", "stack_cups", "wipe_surface"]


@dataclass(frozen=True)
class OperatorProfile:
    """How a given operator tends to fail. Drives defect probabilities."""

    operator_id: str
    skill: float  # 0 (novice) .. 1 (expert)
    hesitancy: float  # tendency to idle mid-episode
    aggression: float  # tendency to saturate the leader arm
    gripper_fidget: float  # tendency to chatter the gripper


OPERATORS = [
    OperatorProfile("op_amelia", skill=0.92, hesitancy=0.05, aggression=0.10, gripper_fidget=0.05),
    OperatorProfile("op_bram", skill=0.78, hesitancy=0.15, aggression=0.35, gripper_fidget=0.15),
    OperatorProfile("op_chidi", skill=0.55, hesitancy=0.40, aggression=0.20, gripper_fidget=0.45),
    OperatorProfile("op_dara", skill=0.35, hesitancy=0.55, aggression=0.50, gripper_fidget=0.60),
]


def _minimum_jerk(t: np.ndarray) -> np.ndarray:
    """Normalised minimum-jerk profile on t in [0, 1]. Smooth start and stop."""
    return 10 * t**3 - 15 * t**4 + 6 * t**5


# Typical per-step joint displacement for an unhurried operator, as a fraction
# of the rig's action limit. Waypoints are sampled from this rather than from
# the joint range, so that motion speed stays independent of episode length —
# otherwise short episodes are automatically saturated and `action_saturation`
# measures duration instead of operator aggression.
NOMINAL_SPEED_FRACTION = 0.22


def _base_trajectory(
    rng: np.random.Generator,
    n_steps: int,
    n_joints: int,
    lower: np.ndarray,
    upper: np.ndarray,
    action_limit: float,
) -> np.ndarray:
    """A plausible reach: min-jerk between waypoints, plus a little tremor."""
    n_waypoints = int(rng.integers(3, 6))
    n_segments = n_waypoints - 1
    segment_steps = max(n_steps // n_segments, 1)

    # Min-jerk peaks at 1.875x its mean velocity, so size the segment
    # displacement to hit the intended peak, not the intended average.
    peak_delta = NOMINAL_SPEED_FRACTION * action_limit
    displacement = peak_delta * segment_steps / 1.875

    mid = 0.5 * (upper + lower)
    waypoints = np.empty((n_waypoints, n_joints))
    waypoints[0] = np.clip(mid + rng.normal(0, 0.3, n_joints), lower, upper)
    for i in range(1, n_waypoints):
        step = rng.normal(0, displacement, n_joints)
        waypoints[i] = np.clip(waypoints[i - 1] + step, lower, upper)

    segments = np.array_split(np.arange(n_steps), n_segments)
    q = np.empty((n_steps, n_joints), dtype=np.float64)
    for i, seg in enumerate(segments):
        if seg.size == 0:
            continue
        s = _minimum_jerk(np.linspace(0, 1, seg.size))[:, None]
        q[seg] = waypoints[i] * (1 - s) + waypoints[i + 1] * s

    # Human tremor: low-amplitude, low-frequency, not white noise.
    tremor = rng.normal(0, 0.004, size=(n_steps, n_joints))
    kernel = np.ones(9) / 9
    for j in range(n_joints):
        tremor[:, j] = np.convolve(tremor[:, j], kernel, mode="same")
    return np.clip(q + tremor, lower, upper)


def _gripper_signal(rng: np.random.Generator, n_steps: int, fidget: float) -> np.ndarray:
    """Open, close on the object, hold, release. Optionally with chatter."""
    g = np.ones(n_steps)
    close_at = int(n_steps * rng.uniform(0.25, 0.45))
    open_at = int(n_steps * rng.uniform(0.70, 0.90))
    g[close_at:open_at] = 0.0
    # Smooth the transitions — a real gripper takes ~150 ms to travel.
    kernel = np.ones(5) / 5
    g = np.convolve(g, kernel, mode="same")

    if rng.random() < fidget:
        # Indecision right before the grasp: several aborted closes.
        for _ in range(int(rng.integers(3, 8))):
            start = int(rng.integers(max(close_at - 40, 1), max(close_at, 2)))
            g[start : start + int(rng.integers(2, 5))] = rng.random()
    return np.clip(g, 0.0, 1.0)


def _episode(
    cfg: Config, rng: np.random.Generator, profile: OperatorProfile, hz: float
) -> tuple[pd.DataFrame, bool]:
    """One episode, with defects injected according to the operator profile."""
    n_joints = cfg.n_joints
    lower, upper = cfg.joint_lower, cfg.joint_upper
    dt = 1.0 / hz

    duration = rng.uniform(4.0, 11.0)
    # Novices occasionally produce a wildly long or aborted attempt.
    if rng.random() < 0.10 * (1 - profile.skill):
        duration *= rng.choice([0.25, 3.0])
    n_steps = max(int(duration * hz), 8)

    q = _base_trajectory(rng, n_steps, n_joints, lower, upper, cfg.action_limit)

    # -- idle stretches: the operator stops to think ------------------------
    if rng.random() < profile.hesitancy:
        for _ in range(int(rng.integers(1, 4))):
            start = int(rng.integers(0, max(n_steps - 20, 1)))
            length = int(rng.integers(int(0.5 * hz), int(2.5 * hz)))
            q[start : start + length] = q[start]

    # -- tracker glitch: a discontinuous pose jump -------------------------
    if rng.random() < 0.18 * (1 - profile.skill):
        idx = int(rng.integers(5, max(n_steps - 5, 6)))
        q[idx:] += rng.normal(0, 0.12, size=n_joints)
        q = np.clip(q, lower, upper)

    dq = np.gradient(q, dt, axis=0)

    # -- actions: commanded delta, clipped at the leader-arm limit ----------
    commanded = np.vstack([np.diff(q, axis=0), np.zeros((1, n_joints))])
    limit = cfg.action_limit
    if rng.random() < profile.aggression:
        # Aggressive operators demand more than the rig will pass through, so
        # the recorded action is a censored version of their intent.
        commanded *= rng.uniform(1.5, 3.0)
    commanded = np.clip(commanded, -limit, limit)

    # -- follower lag: commanded != achieved -------------------------------
    if rng.random() < 0.25 * (1 - profile.skill):
        lag = rng.uniform(0.15, 0.45)
        commanded = commanded * (1 - lag)

    grip = _gripper_signal(rng, n_steps, profile.gripper_fidget)
    act_grip = np.roll(grip, -1)
    act_grip[-1] = grip[-1]

    # -- forward kinematics stand-in ---------------------------------------
    # Not a real FK chain — a smooth, deterministic function of joint angles is
    # enough for the pipeline, and a wrong FK would be worse than an honest
    # placeholder. Real rigs log measured ee pose directly.
    ee = np.column_stack(
        [
            0.30 + 0.25 * np.sin(q[:, 0]) * np.cos(q[:, 1]),
            0.25 * np.sin(q[:, 1]) * np.sin(q[:, 0]),
            0.40 + 0.20 * np.cos(q[:, 1] + q[:, 3]),
        ]
    )
    half = 0.5 * q[:, 5]
    quat = np.column_stack([np.sin(half), np.zeros(n_steps), np.zeros(n_steps), np.cos(half)])
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)

    # -- timing: jitter and dropped frames ---------------------------------
    t = np.arange(n_steps) * dt
    jitter_scale = 0.35 * (1 - profile.skill) * rng.random()
    t = t + rng.normal(0, jitter_scale * dt, size=n_steps)
    t = np.maximum.accumulate(t)
    t += 1e-4 * np.arange(n_steps)  # guarantee strict monotonicity

    frame = pd.DataFrame({"timestamp": t})
    for j in range(n_joints):
        frame[f"joint_{j}"] = q[:, j]
    for j in range(n_joints):
        frame[f"vel_{j}"] = dq[:, j]
    frame["ee_pos_x"], frame["ee_pos_y"], frame["ee_pos_z"] = ee.T
    frame["ee_rot_x"], frame["ee_rot_y"], frame["ee_rot_z"], frame["ee_rot_w"] = quat.T
    frame["gripper"] = grip
    for j in range(n_joints):
        frame[f"cmd_{j}"] = commanded[:, j]
    frame["gripper_cmd"] = act_grip

    # Logger dropouts: whole rows never written.
    drop_p = 0.06 * (1 - profile.skill) * rng.random()
    if drop_p > 0.005:
        keep = rng.random(n_steps) > drop_p
        keep[[0, -1]] = True
        frame = frame[keep].reset_index(drop=True)

    # Sensor dropouts: a channel goes missing for a stretch.
    if rng.random() < 0.12 * (1 - profile.skill):
        col = f"joint_{int(rng.integers(0, n_joints))}"
        start = int(rng.integers(0, max(len(frame) - 10, 1)))
        frame.loc[start : start + int(rng.integers(2, 15)), col] = np.nan

    # Operators mark roughly a fifth of attempts as failed, more when unskilled.
    success = rng.random() > (0.30 - 0.20 * profile.skill)
    return frame, success


def generate(cfg: Config, out_dir: Path, n_sessions: int = 12, seed: int = 0) -> int:
    """Write `n_sessions` raw session directories. Returns episodes written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    hz = cfg.control_hz
    start = datetime(2027, 2, 1, 9, 0, tzinfo=timezone.utc)

    n_episodes = 0
    for s in range(n_sessions):
        profile = OPERATORS[s % len(OPERATORS)]
        task = TASKS[int(rng.integers(0, len(TASKS)))]
        session_id = f"sess_{s:03d}_{task}"
        session_dir = out_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        meta = SessionMeta(
            session_id=session_id,
            operator_id=profile.operator_id,
            robot_id="panda_01",
            task_id=task,
            recorded_at=start + timedelta(days=s // 2, hours=3 * (s % 2)),
            control_hz=hz,
            n_joints=cfg.n_joints,
            notes="synthetic session generated by teleop_pipeline.synthetic",
        )
        (session_dir / "session.json").write_text(meta.model_dump_json(indent=2))

        for e in range(int(rng.integers(4, 9))):
            frame, success = _episode(cfg, rng, profile, hz)
            # The rig encodes the operator's own success label in the filename;
            # ingest.py parses it back out.
            suffix = "ok" if success else "fail"
            frame.to_csv(session_dir / f"episode_{e:03d}_{suffix}.csv", index=False)
            n_episodes += 1

    return n_episodes
