"""Driving a real arm, with the parts that are easy to get wrong already done.

`SimulatedArm` is enough to build and test the recorder. A real rig adds four
failure modes that a naive adapter written at the bench will have, and each one
either damages hardware or silently corrupts the corpus:

1. **Commanding beyond a joint limit.** The driver may accept it and the arm
   will try. Limits are enforced here, before anything reaches the driver.
2. **Commanding a step the arm cannot make.** A large delta at 20 Hz is a
   demand for a velocity the joint does not have. The rate limit is applied
   here, and what gets recorded is the censored request, so a refused command
   stays visible in the data rather than being rewritten as if it succeeded.
3. **Integrating the command instead of reading the arm.** An adapter that
   tracks where it *asked* the arm to go accumulates error against where the
   arm actually is, and the recorded state becomes fiction. Every step reads
   the driver.
4. **Carrying on after the driver fails.** A dropped read mid-episode is not a
   reason to keep issuing motion against a stale position. The arm is held and
   the failure is raised, which ends the episode rather than recording an
   invented one.

Writing an adapter for a specific arm therefore means implementing
`JointDriver`, which is three methods and no policy. Everything above is
already here and tested.
"""

from __future__ import annotations

import contextlib

import numpy as np

from ..config import Config
from .arm import GRIPPER_TRAVEL_S, ArmState
from .device import Command


class DriverError(RuntimeError):
    """The arm could not be read or commanded. Ends the episode."""


class JointDriver:
    """The arm-specific part. Implement these three against your robot's SDK.

    Positions are radians in the robot's own joint order, and the gripper is
    normalised to [0, 1] with 1 fully open, matching the recorded schema. Doing
    the unit conversion here rather than in the pipeline keeps every downstream
    stage identical across rigs.
    """

    def read(self) -> tuple[np.ndarray, float]:
        """Current joint positions (rad) and gripper opening in [0, 1]."""
        raise NotImplementedError

    def write(self, q_target: np.ndarray, grip_target: float) -> None:
        """Command absolute joint positions (rad) and a gripper opening."""
        raise NotImplementedError

    def hold(self) -> None:
        """Stop motion and stay where you are. Called when something failed.

        The default is to do nothing, which is correct for a position-controlled
        arm that holds its last target. An arm under velocity or torque control
        must override this, or a driver failure leaves it moving.
        """


class HardwareArm:
    """An `Arm` backed by a real robot, with the invariants enforced.

    Satisfies the same protocol as `SimulatedArm`, so the recorder, the CLI and
    the app take it without changing anything.
    """

    def __init__(self, cfg: Config, driver: JointDriver) -> None:
        self.driver = driver
        self.n_joints = cfg.n_joints
        self.lower = np.asarray(cfg.joint_lower, dtype=np.float64)
        self.upper = np.asarray(cfg.joint_upper, dtype=np.float64)
        self.action_limit = float(cfg.action_limit)
        if self.lower.shape != (self.n_joints,) or self.upper.shape != (self.n_joints,):
            raise ValueError(
                f"robot.joint_lower and joint_upper must each have {self.n_joints} entries; "
                f"got {self.lower.shape[0]} and {self.upper.shape[0]}. A mismatch here "
                "would clip the wrong joints."
            )
        self._q = np.zeros(self.n_joints, dtype=np.float64)
        self._dq = np.zeros(self.n_joints, dtype=np.float64)
        self._grip = 0.0
        self.reset()

    # -- reading -------------------------------------------------------------

    def _read(self) -> tuple[np.ndarray, float]:
        try:
            q, grip = self.driver.read()
        except Exception as exc:
            self._safe_hold()
            raise DriverError(f"could not read the arm: {exc}") from exc
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (self.n_joints,):
            self._safe_hold()
            raise DriverError(
                f"driver returned {q.shape[0]} joint positions, expected {self.n_joints}. "
                "Check robot.n_joints in params.yaml against the arm."
            )
        if not np.all(np.isfinite(q)) or not np.isfinite(grip):
            self._safe_hold()
            raise DriverError("driver returned a non-finite position; the arm is not readable")
        return q, float(np.clip(grip, 0.0, 1.0))

    def _safe_hold(self) -> None:
        """Best effort. A hold that itself fails must not mask the real error."""
        with contextlib.suppress(Exception):
            self.driver.hold()

    #: How far outside a configured limit a reading may sit before it is treated
    #: as a configuration error rather than an arm parked near its stop. Wide
    #: enough for encoder noise and a calibration offset, far too small to hide
    #: the limits belonging to a different robot.
    LIMIT_TOLERANCE = 0.05

    def reset(self) -> None:
        self._q, self._grip = self._read()
        self._check_limits(self._q)
        self._dq = np.zeros(self.n_joints, dtype=np.float64)

    def _check_limits(self, q: np.ndarray) -> None:
        """Refuse to start if the arm is outside the limits in params.yaml.

        Silently clipping instead would leave the offending joint unable to
        move for the whole session, which at a rig reads as a broken arm rather
        than as a wrong config file. The usual cause is limits copied from a
        different robot, so the message prints both numbers.
        """
        low = self.lower - self.LIMIT_TOLERANCE
        high = self.upper + self.LIMIT_TOLERANCE
        bad = np.nonzero((q < low) | (q > high))[0]
        if bad.size:
            detail = "; ".join(
                f"joint {i}: at {q[i]:+.4f}, limits [{self.lower[i]:+.4f}, {self.upper[i]:+.4f}]"
                for i in bad
            )
            self._safe_hold()
            raise DriverError(
                "the arm is outside the joint limits configured in params.yaml, so those "
                f"joints could never move. {detail}. Either the arm needs moving into "
                "range, or robot.joint_lower and robot.joint_upper belong to a different "
                "robot."
            )

    def state(self) -> ArmState:
        return ArmState(q=self._q.copy(), dq=self._dq.copy(), grip=self._grip)

    # -- commanding ----------------------------------------------------------

    def step(self, command: Command, dt: float) -> tuple[np.ndarray, float]:
        previous = self._q.copy()

        # The censored request, in the same two stages a leader arm imposes:
        # a rate limit, then the joint's own travel. Recording `applied` rather
        # than the raw request keeps a refused command visible in the data.
        rate_limited = np.clip(
            np.asarray(command.joint_delta, dtype=np.float64),
            -self.action_limit,
            self.action_limit,
        )
        target = np.clip(previous + rate_limited, self.lower, self.upper)
        applied = target - previous

        grip_request = float(np.clip(command.grip, 0.0, 1.0))
        # The gripper cannot cross its travel in one control step either, and a
        # request that ignores that produces a square wave no rig can follow.
        max_travel = dt / GRIPPER_TRAVEL_S if GRIPPER_TRAVEL_S > 0 else 1.0
        grip_target = self._grip + float(
            np.clip(grip_request - self._grip, -max_travel, max_travel)
        )
        grip_target = float(np.clip(grip_target, 0.0, 1.0))

        try:
            self.driver.write(target, grip_target)
        except Exception as exc:
            self._safe_hold()
            raise DriverError(f"could not command the arm: {exc}") from exc

        # Read back rather than assume. Where the arm ended up is the state;
        # where it was told to go is the action.
        self._q, self._grip = self._read()
        self._dq = (self._q - previous) / dt if dt > 0 else np.zeros(self.n_joints)
        return applied, grip_request


class EchoDriver(JointDriver):
    """A driver that moves exactly where it is told, for testing an adapter.

    Not a simulator: it models nothing. Use it to check wiring, units and joint
    order end to end before a real arm is powered on.
    """

    def __init__(self, n_joints: int, *, start: np.ndarray | None = None):
        self.q = (
            np.zeros(n_joints, dtype=np.float64)
            if start is None
            else np.asarray(start, dtype=np.float64).copy()
        )
        self.grip = 0.0
        self.holds = 0

    def read(self) -> tuple[np.ndarray, float]:
        return self.q.copy(), self.grip

    def write(self, q_target: np.ndarray, grip_target: float) -> None:
        self.q = np.asarray(q_target, dtype=np.float64).copy()
        self.grip = float(grip_target)

    def hold(self) -> None:
        self.holds += 1


__all__ = ["DriverError", "EchoDriver", "HardwareArm", "JointDriver"]
