"""The recording path, from operator intent to a session `ingest` accepts.

The test that carries the most weight here is `test_recorded_session_ingests`:
a recorder whose output needs a special case in `ingest.py` has not removed the
problem it was built to remove, it has moved it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from teleop_pipeline.teleop import (
    Command,
    ScriptedDevice,
    SimulatedArm,
    record_episode,
    scripted_reach,
    write_session,
)
from teleop_pipeline.teleop.arm import GRIPPER_TRAVEL_S
from teleop_pipeline.teleop.recorder import DEADLINE_SLACK


class FakeClock:
    """A clock that only moves when the loop sleeps, plus optional overrun.

    Lets a 20 Hz loop be tested in microseconds, and lets a late step be
    injected deterministically rather than waited for.
    """

    def __init__(self, overruns: dict[int, float] | None = None) -> None:
        self.now = 0.0
        self._overruns = overruns or {}
        self._sleeps = 0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        extra = self._overruns.get(self._sleeps)
        if extra:
            self.now += extra
        self._sleeps += 1


def _record(cfg, *, steps=40, success=True, clock=None):
    arm = SimulatedArm(cfg)
    device = ScriptedDevice(
        cfg.n_joints, scripted_reach(cfg, steps=steps), steps=steps, success=success
    )
    fake = clock or FakeClock()
    return record_episode(arm, device, cfg, clock=fake.clock, sleep=fake.sleep)


# -- the arm ----------------------------------------------------------------


def test_arm_refuses_to_pass_more_than_the_action_limit(cfg):
    arm = SimulatedArm(cfg)
    huge = np.full(cfg.n_joints, 100.0)
    applied, _ = arm.step(Command(joint_delta=huge, grip=0.0), cfg.dt)
    assert np.all(np.abs(applied) <= cfg.action_limit + 1e-12)


def test_arm_stops_at_its_joint_limits(cfg):
    arm = SimulatedArm(cfg)
    for _ in range(5000):
        arm.step(Command(joint_delta=np.full(cfg.n_joints, 1.0), grip=0.0), cfg.dt)
    state = arm.state()
    assert np.all(state.q <= np.asarray(cfg.joint_upper) + 1e-9)


def test_velocity_reports_the_achieved_change_not_the_request(cfg):
    """A joint parked on its limit reports zero velocity, however hard you push."""
    arm = SimulatedArm(cfg)
    for _ in range(5000):
        arm.step(Command(joint_delta=np.full(cfg.n_joints, 1.0), grip=0.0), cfg.dt)
    arm.step(Command(joint_delta=np.full(cfg.n_joints, 1.0), grip=0.0), cfg.dt)
    assert np.allclose(arm.state().dq, 0.0, atol=1e-9)


def test_gripper_takes_real_time_to_travel(cfg):
    """A step response, not a square wave: no real gripper closes instantly."""
    arm = SimulatedArm(cfg)
    zero = np.zeros(cfg.n_joints)
    arm.step(Command(joint_delta=zero, grip=1.0), cfg.dt)
    after_one = arm.state().grip
    assert 0.0 < after_one < 1.0

    steps_needed = int(np.ceil(GRIPPER_TRAVEL_S / cfg.dt))
    for _ in range(steps_needed + 1):
        arm.step(Command(joint_delta=zero, grip=1.0), cfg.dt)
    assert arm.state().grip == pytest.approx(1.0)


# -- the loop ---------------------------------------------------------------


def test_recording_has_one_row_per_step(cfg):
    rec = _record(cfg, steps=40)
    assert rec.n_steps == 40
    assert rec.success is True


def test_operator_failure_label_is_kept(cfg):
    rec = _record(cfg, steps=12, success=False)
    assert rec.success is False


def test_timestamps_are_measured_and_strictly_increasing(cfg):
    rec = _record(cfg, steps=30)
    t = rec.frame["timestamp"].to_numpy()
    assert np.all(np.diff(t) > 0)
    # On an unloaded virtual clock the loop hits its nominal rate exactly; the
    # point of the assertion is that the column comes from the clock at all.
    assert t[-1] == pytest.approx((len(t) - 1) * cfg.dt, rel=1e-9)


def test_a_late_step_is_counted_not_hidden(cfg):
    """An overrun the operator could feel has to survive into the report."""
    rec = _record(cfg, steps=30, clock=FakeClock(overruns={5: 3.0 * cfg.dt}))
    assert rec.late_steps > 0


def test_one_stall_makes_every_step_it_delays_late(cfg):
    """Deadlines are absolute, so a stall is counted in steps, not in stalls.

    A 3-period stall leaves the loop two periods behind. It then runs without
    sleeping until it catches up, and both of those steps started after their
    own deadline, so both are late. Counting the stall once would understate
    how much of the episode was recorded off-schedule.
    """
    one = _record(cfg, steps=30, clock=FakeClock(overruns={5: 3.0 * cfg.dt}))
    two = _record(cfg, steps=30, clock=FakeClock(overruns={5: 3.0 * cfg.dt, 11: 3.0 * cfg.dt}))
    assert one.late_steps == 2
    assert two.late_steps == 2 * one.late_steps


def test_a_step_inside_the_slack_is_not_counted_late(cfg):
    inside = {7: 0.5 * DEADLINE_SLACK * cfg.dt}
    rec = _record(cfg, steps=20, clock=FakeClock(overruns=inside))
    assert rec.late_steps == 0


def test_quit_stops_the_episode_and_reports_it(cfg):
    class QuitAfter:
        def __init__(self, n, n_joints):
            self.n, self.n_joints, self.i = n, n_joints, 0

        def poll(self, grip):
            self.i += 1
            if self.i > self.n:
                return Command(joint_delta=np.zeros(self.n_joints), grip=grip, quit=True)
            return Command(joint_delta=np.zeros(self.n_joints), grip=grip)

        def close(self):
            return

    fake = FakeClock()
    rec = record_episode(
        SimulatedArm(cfg), QuitAfter(4, cfg.n_joints), cfg, clock=fake.clock, sleep=fake.sleep
    )
    assert rec.quit_requested is True
    assert rec.n_steps == 4


def test_commanded_column_holds_the_request_not_the_outcome(cfg):
    """The gap between asked-for and achieved is what quality.py scores."""
    n = cfg.n_joints

    def always_saturate(_i):
        return Command(joint_delta=np.full(n, 10.0), grip=0.0)

    device = ScriptedDevice(n, always_saturate, steps=25)
    fake = FakeClock()
    rec = record_episode(SimulatedArm(cfg), device, cfg, clock=fake.clock, sleep=fake.sleep)
    cmd = rec.frame[[f"cmd_{j}" for j in range(n)]].to_numpy()
    # Clipped to what the rig passes through, and never silently zeroed.
    assert np.all(np.abs(cmd) <= cfg.action_limit + 1e-12)
    assert np.any(np.abs(cmd) > 0)


# -- the contract with the rest of the pipeline ------------------------------


def test_written_columns_match_what_ingest_aliases(cfg, tmp_path):
    from teleop_pipeline.ingest import COLUMN_ALIASES

    rec = _record(cfg, steps=20)
    session_dir = write_session(
        tmp_path,
        [rec],
        cfg=cfg,
        session_id="sess_test_pick",
        operator_id="op_test",
        robot_id="sim_01",
        task_id="pick_place_block",
    )
    csv = next(session_dir.glob("episode_*.csv"))
    columns = set(pd.read_csv(csv).columns)

    expected_joint = {f"joint_{j}" for j in range(cfg.n_joints)}
    expected_cmd = {f"cmd_{j}" for j in range(cfg.n_joints)}
    assert expected_joint <= columns
    assert expected_cmd <= columns
    # Every non-joint column must be one ingest already knows how to rename.
    leftover = columns - expected_joint - expected_cmd - {f"vel_{j}" for j in range(cfg.n_joints)}
    assert leftover <= set(
        COLUMN_ALIASES
    ), f"ingest has no alias for {leftover - set(COLUMN_ALIASES)}"


def test_success_label_rides_in_the_filename(cfg, tmp_path):
    ok = _record(cfg, steps=10, success=True)
    bad = _record(cfg, steps=10, success=False)
    session_dir = write_session(
        tmp_path,
        [ok, bad],
        cfg=cfg,
        session_id="sess_labels",
        operator_id="op_test",
        robot_id="sim_01",
        task_id="pick_place_block",
    )
    names = sorted(p.name for p in session_dir.glob("episode_*.csv"))
    assert names == ["episode_000_ok.csv", "episode_001_fail.csv"]


def test_recorded_session_ingests(cfg, tmp_path):
    """The one that matters: a recording goes through the real ingest path.

    No special case, no bespoke reader. If this fails, the recorder has moved
    the problem rather than solved it.
    """
    from teleop_pipeline.ingest import ingest_all

    raw = tmp_path / "raw"
    recordings = [_record(cfg, steps=30, success=i % 2 == 0) for i in range(3)]
    write_session(
        raw,
        recordings,
        cfg=cfg,
        session_id="sess_recorded_pick_place_block",
        operator_id="op_recorder",
        robot_id="sim_01",
        task_id="pick_place_block",
        notes="recorded by tests",
    )

    out = tmp_path / "interim"
    result = ingest_all(cfg, raw_dir=raw, out_dir=out)
    assert result.skipped == []
    assert result.episodes == 3

    written = sorted((out / "sess_recorded_pick_place_block").glob("*.parquet"))
    assert len(written) == 3
    frame = pd.read_parquet(written[0])
    assert np.all(np.diff(frame["t"].to_numpy()) > 0)
