"""End-effector pose from joint angles.

Not a real FK chain. A smooth, deterministic function of joint angles is enough
for the pipeline, and a wrong FK would be worse than an honest placeholder. Real
rigs log measured ee pose directly and never call this.

It lives here rather than inside one caller because both the synthetic generator
and the live recorder write the same raw columns. Two copies of a placeholder is
how the recorded corpus and the generated one quietly stop being comparable.
"""

from __future__ import annotations

import numpy as np

# The placeholder reads joints 0, 1, 3 and 5, so a rig with fewer than six would
# index off the end. Checked explicitly: a rig that small needs a real FK, and
# an IndexError three frames deep is a poor way to be told so.
MIN_JOINTS = 6


def _check(q: np.ndarray) -> np.ndarray:
    q = np.atleast_2d(q)
    if q.shape[1] < MIN_JOINTS:
        raise ValueError(
            f"the placeholder kinematics needs at least {MIN_JOINTS} joints, got {q.shape[1]}; "
            "a rig this size should log measured ee pose instead"
        )
    return q


def ee_position(q: np.ndarray) -> np.ndarray:
    """`(n_steps, n_joints)` joint angles -> `(n_steps, 3)` position in metres."""
    q = _check(q)
    return np.column_stack(
        [
            0.30 + 0.25 * np.sin(q[:, 0]) * np.cos(q[:, 1]),
            0.25 * np.sin(q[:, 1]) * np.sin(q[:, 0]),
            0.40 + 0.20 * np.cos(q[:, 1] + q[:, 3]),
        ]
    )


def ee_quaternion(q: np.ndarray) -> np.ndarray:
    """`(n_steps, n_joints)` joint angles -> `(n_steps, 4)` unit quaternion, xyzw."""
    q = _check(q)
    n_steps = q.shape[0]
    half = 0.5 * q[:, 5]
    quat = np.column_stack([np.sin(half), np.zeros(n_steps), np.zeros(n_steps), np.cos(half)])
    return quat / np.linalg.norm(quat, axis=1, keepdims=True)
