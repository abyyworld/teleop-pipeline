"""Where an operator's intent comes from.

A device reports what the human asked for, never what the arm managed to do.
Keeping those apart is the whole point: `cmd_*` in the recorded episode is the
request, `joint_*` is the outcome, and the gap between them is the signal that
`quality.py` scores. A device that clipped its own output to what the rig can
follow would erase that gap at the source.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Command:
    """One control step's worth of operator intent."""

    joint_delta: np.ndarray  # (n_joints,) requested change in rad, unclipped
    grip: float  # requested gripper opening, normalised [0, 1]
    end_episode: bool = False
    success: bool | None = None  # the operator's own label, set when ending
    quit: bool = False  # stop the whole session, not just this episode

    @classmethod
    def idle(cls, n_joints: int, grip: float = 0.0) -> Command:
        return cls(joint_delta=np.zeros(n_joints, dtype=np.float64), grip=grip)


@runtime_checkable
class InputDevice(Protocol):
    """Anything that can answer "what does the operator want right now"."""

    def poll(self, grip: float) -> Command:
        """Return the intent for this step. Must not block longer than one step."""
        ...

    def close(self) -> None: ...


class ScriptedDevice:
    """A recorded operator, for tests, CI and `--device scripted` demos.

    Takes a callable of the step index so a test can describe a trajectory
    without materialising it. Returning ``None`` ends the episode.
    """

    def __init__(
        self,
        n_joints: int,
        step_fn,
        *,
        steps: int,
        success: bool = True,
    ) -> None:
        self.n_joints = n_joints
        self._step_fn = step_fn
        self._steps = steps
        self._success = success
        self._i = 0

    def poll(self, grip: float) -> Command:
        if self._i >= self._steps:
            return Command(
                joint_delta=np.zeros(self.n_joints),
                grip=grip,
                end_episode=True,
                success=self._success,
            )
        cmd = self._step_fn(self._i)
        self._i += 1
        return cmd

    def close(self) -> None:  # nothing to release
        return


# -- keyboard ---------------------------------------------------------------

# Joint j is driven by KEYS[j]: first raises, second lowers. Deliberately not
# arrow keys — those arrive as multi-byte escape sequences, and a half-read
# sequence on a 20 Hz loop shows up as a phantom command.
KEYS = ["qa", "ws", "ed", "rf", "tg", "yh", "uj"]
GRIP_OPEN = "]"
GRIP_CLOSE = "["
END_SUCCESS = "."
END_FAILURE = ","
QUIT = "x"

HELP = f"""\
joint 0..n   {'  '.join(f'{k[0]}/{k[1]}' for k in KEYS)}   (raise/lower)
gripper      {GRIP_CLOSE} close   {GRIP_OPEN} open
end episode  {END_SUCCESS} success   {END_FAILURE} failure
quit         {QUIT}
"""


class KeyboardDevice:
    """Drive the arm from a terminal, with no GUI and no extra dependency.

    Reads whatever keys arrived since the last step rather than waiting for one,
    because the control loop owns the timing. A step with no keypress is a hold,
    which is a real thing an operator does and must be recorded as such.
    """

    def __init__(self, n_joints: int, *, step_rad: float = 0.02, grip_rate: float = 0.08) -> None:
        if n_joints > len(KEYS):
            raise ValueError(f"keyboard layout covers {len(KEYS)} joints, rig has {n_joints}")
        self.n_joints = n_joints
        self.step_rad = step_rad
        self.grip_rate = grip_rate
        self._restore = None
        self._win = sys.platform == "win32"
        if not self._win:
            import termios
            import tty

            self._termios = termios
            fd = sys.stdin.fileno()
            self._fd = fd
            self._restore = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        else:
            import msvcrt

            self._msvcrt = msvcrt

    def _pending(self) -> str:
        """Every key buffered since the last call, as one string."""
        out = []
        if self._win:
            while self._msvcrt.kbhit():
                ch = self._msvcrt.getwch()
                out.append(ch)
        else:
            import select

            while select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if not ch:
                    break
                out.append(ch)
        return "".join(out)

    def poll(self, grip: float) -> Command:
        keys = self._pending()
        delta = np.zeros(self.n_joints, dtype=np.float64)
        for ch in keys:
            if ch == QUIT:
                return Command(joint_delta=delta, grip=grip, end_episode=True, quit=True)
            if ch == END_SUCCESS:
                return Command(joint_delta=delta, grip=grip, end_episode=True, success=True)
            if ch == END_FAILURE:
                return Command(joint_delta=delta, grip=grip, end_episode=True, success=False)
            if ch == GRIP_OPEN:
                grip = min(1.0, grip + self.grip_rate)
            elif ch == GRIP_CLOSE:
                grip = max(0.0, grip - self.grip_rate)
            else:
                for j, pair in enumerate(KEYS[: self.n_joints]):
                    if ch == pair[0]:
                        delta[j] += self.step_rad
                    elif ch == pair[1]:
                        delta[j] -= self.step_rad
        return Command(joint_delta=delta, grip=grip)

    def close(self) -> None:
        if self._restore is not None:
            self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._restore)
            self._restore = None
