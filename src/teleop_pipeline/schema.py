"""Canonical on-disk schema for a teleoperation episode.

Every raw session, whatever its original layout, is normalised into this shape
before anything else touches it. Downstream stages are allowed to assume it
holds — that assumption is what `validate.py` exists to defend.

Layout on disk::

    data/interim/episodes/
      <session_id>/
        meta.json                  # SessionMeta
        <episode_id>.parquet       # timeseries, columns below
        <episode_id>.meta.json     # EpisodeMeta

Timeseries columns (n = robot.n_joints):

    t                       float64  seconds since episode start, strictly increasing
    q_0     .. q_{n-1}      float32  measured joint position (rad)
    dq_0    .. dq_{n-1}     float32  measured joint velocity (rad/s)
    ee_x, ee_y, ee_z        float32  end-effector position in robot base frame (m)
    ee_qx, ee_qy, ee_qz, ee_qw
                            float32  end-effector orientation, unit quaternion
    grip                    float32  measured gripper opening, normalised [0, 1]
    act_q_0 .. act_q_{n-1}  float32  commanded joint delta for this step (rad)
    act_grip                float32  commanded gripper opening, normalised [0, 1]
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

QualityTier = Literal["gold", "silver", "reject"]


def joint_cols(prefix: str, n: int) -> list[str]:
    return [f"{prefix}_{i}" for i in range(n)]


EE_POS_COLS = ["ee_x", "ee_y", "ee_z"]
EE_QUAT_COLS = ["ee_qx", "ee_qy", "ee_qz", "ee_qw"]


def timeseries_columns(n_joints: int) -> list[str]:
    """Full ordered column list for an episode parquet file."""
    return (
        ["t"]
        + joint_cols("q", n_joints)
        + joint_cols("dq", n_joints)
        + EE_POS_COLS
        + EE_QUAT_COLS
        + ["grip"]
        + joint_cols("act_q", n_joints)
        + ["act_grip"]
    )


def observation_columns(n_joints: int, obs_keys: list[str]) -> list[str]:
    """Resolve the symbolic `dataset.obs_keys` from params.yaml to real columns."""
    groups: dict[str, list[str]] = {
        "q": joint_cols("q", n_joints),
        "dq": joint_cols("dq", n_joints),
        "ee_pos": EE_POS_COLS,
        "ee_quat": EE_QUAT_COLS,
        "grip": ["grip"],
        # Previous executed action. Synthesised by `dataset.py`, not recorded.
        # Including it lets the policy express smoothness, without it a BC
        # policy cannot beat a persistence baseline on smooth teleop. It also
        # invites causal confusion — see the note in params.yaml before turning
        # it on.
        "prev_act_q": [f"prev_act_q_{i}" for i in range(n_joints)],
        "prev_act_grip": ["prev_act_grip"],
    }
    cols: list[str] = []
    for key in obs_keys:
        if key not in groups:
            raise KeyError(f"unknown obs key {key!r}; valid: {sorted(groups)}")
        cols.extend(groups[key])
    return cols


def action_columns(n_joints: int, action_keys: list[str]) -> list[str]:
    groups: dict[str, list[str]] = {
        "act_q": joint_cols("act_q", n_joints),
        "act_grip": ["act_grip"],
    }
    cols: list[str] = []
    for key in action_keys:
        if key not in groups:
            raise KeyError(f"unknown action key {key!r}; valid: {sorted(groups)}")
        cols.extend(groups[key])
    return cols


class SessionMeta(BaseModel):
    """One contiguous teleop sitting: an operator, a robot, a task, one setup."""

    session_id: str
    operator_id: str
    robot_id: str
    task_id: str
    recorded_at: datetime
    control_hz: float = Field(gt=0)
    n_joints: int = Field(gt=0)
    notes: str = ""

    @field_validator("session_id", "operator_id", "robot_id", "task_id")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("identifier must not be empty")
        return v.strip()


class EpisodeMeta(BaseModel):
    """One demonstration attempt. `success` is the operator's own label."""

    episode_id: str
    session_id: str
    operator_id: str
    task_id: str
    success: bool
    n_steps: int = Field(ge=0)
    duration_s: float = Field(ge=0)
    control_hz: float = Field(gt=0)

    # Facts about the *raw* recording that resampling destroys. Captured at
    # ingest because they are unrecoverable from the canonical file, and timing
    # quality is one of the strongest predictors of a bad teleop session.
    interp_fraction: float = 0.0
    raw_dt_jitter: float = 0.0
    raw_gap_fraction: float = 0.0

    # Filled in by later stages; absent until then.
    quality_score: float | None = None
    quality_tier: QualityTier | None = None
    flags: list[str] = Field(default_factory=list)
    source_file: str = ""
    content_hash: str = ""


class MetricScore(BaseModel):
    """One quality metric: its raw value and the penalty it contributed."""

    name: str
    value: float
    penalty: float = Field(ge=0, le=1)
    weight: float
    flagged: bool


class QualityReport(BaseModel):
    episode_id: str
    session_id: str
    operator_id: str
    task_id: str
    score: float
    tier: QualityTier
    metrics: list[MetricScore] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    hard_reject_reasons: list[str] = Field(default_factory=list)

    def value(self, name: str) -> float:
        for m in self.metrics:
            if m.name == name:
                return m.value
        raise KeyError(name)


class ValidationIssue(BaseModel):
    code: str
    message: str
    severity: Literal["error", "warning"]


class ValidationReport(BaseModel):
    episode_id: str
    session_id: str
    ok: bool
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]
