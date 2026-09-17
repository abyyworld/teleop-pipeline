"""The control loop, and the raw session it writes.

Output is deliberately the *raw* layout that `ingest.py` already consumes —
`session.json` plus one messy CSV per episode — and not the canonical interim
schema. A recorder that wrote canonical episodes directly would bypass
resampling, alias mapping, gap detection and hashing, which is to say it would
bypass every check that makes the corpus trustworthy. Recorded sessions take
exactly the same path as any other rig's dump.

Timestamps are measured, never assumed. A teleop loop does not hit its nominal
rate, and the jitter is information: `ingest.py` resamples onto an exact grid
and records how much it had to interpolate, and `quality.py` penalises the
episodes where that was a lot. Writing a perfect `i * dt` grid here would
fabricate a clean rig and silently disable that check.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..kinematics import ee_position, ee_quaternion
from ..schema import SessionMeta
from .arm import Arm
from .device import Command, InputDevice

# A step that starts more than this fraction of a period past its deadline is
# late enough that the operator can feel it. Counted per episode and reported,
# because a rig that runs late often is producing data whose timing cannot be
# trusted.
#
# Deadlines are absolute (`start + i * dt`), so the loop holds its nominal rate
# on average rather than drifting. One stall therefore makes *several* steps
# late: the loop runs without sleeping until it catches up, and each of those
# steps did genuinely start after its deadline. `late_steps` counts steps, not
# stalls, which is the quantity that matters for whether the timing can be
# trusted.
DEADLINE_SLACK = 0.25


@dataclass
class EpisodeRecording:
    frame: pd.DataFrame
    success: bool
    quit_requested: bool = False
    late_steps: int = 0

    @property
    def n_steps(self) -> int:
        return len(self.frame)


@dataclass
class _Columns:
    """Accumulates one episode in the raw (pre-ingest) column names."""

    n_joints: int
    t: list[float] = field(default_factory=list)
    q: list[np.ndarray] = field(default_factory=list)
    dq: list[np.ndarray] = field(default_factory=list)
    grip: list[float] = field(default_factory=list)
    cmd: list[np.ndarray] = field(default_factory=list)
    cmd_grip: list[float] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        q = np.asarray(self.q, dtype=np.float64).reshape(-1, self.n_joints)
        dq = np.asarray(self.dq, dtype=np.float64).reshape(-1, self.n_joints)
        cmd = np.asarray(self.cmd, dtype=np.float64).reshape(-1, self.n_joints)

        frame = pd.DataFrame({"timestamp": np.asarray(self.t, dtype=np.float64)})
        for j in range(self.n_joints):
            frame[f"joint_{j}"] = q[:, j]
        for j in range(self.n_joints):
            frame[f"vel_{j}"] = dq[:, j]
        if len(q):
            ee = ee_position(q)
            quat = ee_quaternion(q)
        else:  # an episode ended before its first step
            ee = np.zeros((0, 3))
            quat = np.zeros((0, 4))
        frame["ee_pos_x"], frame["ee_pos_y"], frame["ee_pos_z"] = ee.T
        frame["ee_rot_x"], frame["ee_rot_y"], frame["ee_rot_z"], frame["ee_rot_w"] = quat.T
        frame["gripper"] = np.asarray(self.grip, dtype=np.float64)
        for j in range(self.n_joints):
            frame[f"cmd_{j}"] = cmd[:, j]
        frame["gripper_cmd"] = np.asarray(self.cmd_grip, dtype=np.float64)
        return frame


def record_episode(
    arm: Arm,
    device: InputDevice,
    cfg: Config,
    *,
    max_steps: int = 100_000,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> EpisodeRecording:
    """Drive `arm` from `device` until the operator ends the episode.

    `clock` and `sleep` are injected so tests can run the loop on a virtual
    clock. In production they are the monotonic clock, which is the only one
    that cannot jump backwards mid-episode when the host syncs its time.
    """
    n_joints = cfg.n_joints
    dt = cfg.dt
    cols = _Columns(n_joints=n_joints)
    arm.reset()

    start = clock()
    late = 0
    success = False
    quit_requested = False
    grip_target = arm.state().grip

    for i in range(max_steps):
        now = clock()
        state = arm.state()
        command = device.poll(grip_target)

        if command.quit:
            quit_requested = True
            success = bool(command.success) if command.success is not None else False
            break
        if command.end_episode:
            success = bool(command.success)
            break

        applied, grip_target = arm.step(command, dt)

        # Row i holds the state measured at `now` and the action issued there,
        # so the action is the one that produced row i+1. `dataset.py` assumes
        # this alignment when it builds (observation, action) pairs.
        cols.t.append(now - start)
        cols.q.append(state.q)
        cols.dq.append(state.dq)
        cols.grip.append(state.grip)
        cols.cmd.append(applied)
        cols.cmd_grip.append(grip_target)

        deadline = start + (i + 1) * dt
        remaining = deadline - clock()
        if remaining > 0:
            sleep(remaining)
        elif -remaining > DEADLINE_SLACK * dt:
            late += 1

    return EpisodeRecording(
        frame=cols.to_frame(),
        success=success,
        quit_requested=quit_requested,
        late_steps=late,
    )


def write_session(
    out_dir: Path,
    episodes: list[EpisodeRecording],
    *,
    cfg: Config,
    session_id: str,
    operator_id: str,
    robot_id: str,
    task_id: str,
    notes: str = "",
    recorded_at: datetime | None = None,
) -> Path:
    """Write one raw session directory, in the layout `ingest` expects."""
    session_dir = Path(out_dir) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    meta = SessionMeta(
        session_id=session_id,
        operator_id=operator_id,
        robot_id=robot_id,
        task_id=task_id,
        recorded_at=recorded_at or datetime.now(timezone.utc),
        control_hz=cfg.control_hz,
        n_joints=cfg.n_joints,
        notes=notes,
    )
    (session_dir / "session.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")

    for e, episode in enumerate(episodes):
        # The operator's own label rides in the filename, which is the
        # convention `ingest.py` already parses.
        suffix = "ok" if episode.success else "fail"
        episode.frame.to_csv(session_dir / f"episode_{e:03d}_{suffix}.csv", index=False)

    return session_dir


def scripted_reach(
    cfg: Config, *, amplitude: float = 0.35, steps: int = 60
) -> Callable[[int], Command]:
    """A deterministic operator: one smooth reach, close the gripper, hold.

    Used by the tests and by `record --device scripted`, so that the recording
    path can be exercised end to end on a machine with no keyboard attached and
    no robot in the room.
    """
    n_joints = cfg.n_joints
    limit = cfg.action_limit

    def step(i: int) -> Command:
        phase = (i + 1) / steps
        # A half-sine so the reach accelerates and settles rather than
        # starting and stopping at full rate.
        scale = amplitude * np.sin(np.pi * min(phase, 1.0)) / max(steps, 1)
        delta = np.full(n_joints, scale, dtype=np.float64)
        delta[1::2] *= -1.0  # alternate direction so the arm does not fly off
        grip = 0.0 if phase < 0.5 else 1.0
        return Command(joint_delta=np.clip(delta, -limit, limit), grip=grip)

    return step
