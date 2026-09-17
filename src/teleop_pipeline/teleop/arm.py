"""The thing being driven.

`SimulatedArm` exists so the recorder can be run, tested and demonstrated
without a robot. It deliberately injects no defects. `synthetic.py` injects
defects because its job is to give the quality scorer known-bad input; a
recorder's job is to report what happened, and a simulator that invented
follower lag would put fabricated numbers into a corpus labelled as recorded.

What it does model is the part that is not optional for the recording to mean
anything: the leader arm cannot pass through more than `robot.action_limit` per
step, joints stop at their limits, and the gripper takes real time to travel.
Those three are why commanded and achieved differ on a real rig.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..config import Config
from .device import Command

# A real two-finger gripper takes roughly this long to go end to end. Without
# it the recorded `grip` channel is a square wave, which no real rig produces.
GRIPPER_TRAVEL_S = 0.15


@dataclass
class ArmState:
    q: np.ndarray  # measured joint positions (rad)
    dq: np.ndarray  # measured joint velocities (rad/s)
    grip: float  # measured gripper opening, normalised [0, 1]


@runtime_checkable
class Arm(Protocol):
    def reset(self) -> None: ...

    def state(self) -> ArmState: ...

    def step(self, command: Command, dt: float) -> tuple[np.ndarray, float]:
        """Apply one step. Returns what was actually commanded to the rig.

        The return value is the *censored* request — clipped to what the rig
        will pass through — not the achieved state. It is what gets recorded as
        `cmd_*`, so that a request the rig refused stays visible in the data.
        """
        ...


class SimulatedArm:
    """A kinematic stand-in: joint limits, a rate limit, and gripper travel."""

    def __init__(self, cfg: Config, *, home: np.ndarray | None = None) -> None:
        self.n_joints = cfg.n_joints
        self.lower = np.asarray(cfg.joint_lower, dtype=np.float64)
        self.upper = np.asarray(cfg.joint_upper, dtype=np.float64)
        self.action_limit = float(cfg.action_limit)
        if home is None:
            home = 0.5 * (self.lower + self.upper)
        self._home = np.asarray(home, dtype=np.float64).copy()
        self.reset()

    def reset(self) -> None:
        self._q = self._home.copy()
        self._dq = np.zeros(self.n_joints, dtype=np.float64)
        self._grip = 0.0

    def state(self) -> ArmState:
        return ArmState(q=self._q.copy(), dq=self._dq.copy(), grip=self._grip)

    def step(self, command: Command, dt: float) -> tuple[np.ndarray, float]:
        # The leader arm saturates: anything beyond the limit never reaches the
        # follower, and the recorded action is the censored version of intent.
        applied = np.clip(command.joint_delta, -self.action_limit, self.action_limit)

        previous = self._q.copy()
        self._q = np.clip(self._q + applied, self.lower, self.upper)
        # Velocity from the achieved change, not the request. A joint parked on
        # its limit reports zero velocity however hard the operator pushes.
        self._dq = (self._q - previous) / dt if dt > 0 else np.zeros_like(self._q)

        target = float(np.clip(command.grip, 0.0, 1.0))
        max_travel = dt / GRIPPER_TRAVEL_S if GRIPPER_TRAVEL_S > 0 else 1.0
        self._grip += float(np.clip(target - self._grip, -max_travel, max_travel))
        self._grip = float(np.clip(self._grip, 0.0, 1.0))

        return applied, target
