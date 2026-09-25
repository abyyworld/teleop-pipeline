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
    KeyboardDevice,
    ScriptedDevice,
    SimulatedArm,
    record_episode,
    scripted_reach,
    write_session,
)
from teleop_pipeline.teleop.arm import GRIPPER_TRAVEL_S
from teleop_pipeline.teleop.device import END_FAILURE, END_SUCCESS, KEYS, QUIT
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


# -- the keyboard the operator actually uses ---------------------------------


def _keyboard(cfg, script):
    """A KeyboardDevice fed a canned key sequence instead of a terminal."""
    queue = list(script)

    def read_keys() -> str:
        return queue.pop(0) if queue else ""

    return KeyboardDevice(cfg.n_joints, read_keys=read_keys)


def test_each_joint_key_pair_moves_its_own_joint(cfg):
    for j, pair in enumerate(KEYS[: cfg.n_joints]):
        up = _keyboard(cfg, [pair[0]]).poll(0.0)
        down = _keyboard(cfg, [pair[1]]).poll(0.0)
        assert up.joint_delta[j] > 0, f"{pair[0]} should raise joint {j}"
        assert down.joint_delta[j] < 0, f"{pair[1]} should lower joint {j}"
        others = [k for k in range(cfg.n_joints) if k != j]
        assert np.allclose(up.joint_delta[others], 0.0), f"{pair[0]} moved another joint"


def test_no_keypress_is_a_hold_not_a_dropped_step(cfg):
    """An operator pausing is data, not an absence of data."""
    cmd = _keyboard(cfg, [""]).poll(0.4)
    assert np.allclose(cmd.joint_delta, 0.0)
    assert cmd.grip == 0.4
    assert cmd.end_episode is False


def test_repeated_keys_in_one_step_accumulate(cfg):
    """Keys buffered between steps all count; the loop owns the timing."""
    dev = _keyboard(cfg, ["qqq"])
    cmd = dev.poll(0.0)
    assert cmd.joint_delta[0] == pytest.approx(3 * dev.step_rad)


def test_opposing_keys_in_one_step_cancel(cfg):
    cmd = _keyboard(cfg, ["qa"]).poll(0.0)
    assert cmd.joint_delta[0] == pytest.approx(0.0)


def test_gripper_keys_move_and_clamp(cfg):
    opened = _keyboard(cfg, ["]"]).poll(0.5)
    closed = _keyboard(cfg, ["["]).poll(0.5)
    assert opened.grip > 0.5
    assert closed.grip < 0.5
    assert _keyboard(cfg, ["]" * 50]).poll(0.5).grip == pytest.approx(1.0)
    assert _keyboard(cfg, ["[" * 50]).poll(0.5).grip == pytest.approx(0.0)


def test_end_keys_carry_the_operators_own_label(cfg):
    ok = _keyboard(cfg, [END_SUCCESS]).poll(0.0)
    bad = _keyboard(cfg, [END_FAILURE]).poll(0.0)
    assert (ok.end_episode, ok.success) == (True, True)
    assert (bad.end_episode, bad.success) == (True, False)


def test_quit_ends_the_episode_too(cfg):
    """Quitting mid-episode must not leave the loop running on a dead device."""
    cmd = _keyboard(cfg, [QUIT]).poll(0.0)
    assert cmd.quit is True
    assert cmd.end_episode is True


def test_an_end_key_wins_over_motion_buffered_behind_it(cfg):
    """Keys after the end marker belong to the next episode, not this one."""
    cmd = _keyboard(cfg, [f"q{END_SUCCESS}w"]).poll(0.0)
    assert cmd.end_episode is True
    assert cmd.success is True
    # Decoding stops at the marker: 'q' before it is still decoded, 'w' after it
    # is not. The recorder then breaks on end_episode and drops this last
    # partial step, so ending an episode cannot smear a stray keypress into the
    # demonstration.
    assert cmd.joint_delta[0] > 0
    assert cmd.joint_delta[1] == pytest.approx(0.0)


def test_unknown_keys_are_ignored(cfg):
    cmd = _keyboard(cfg, ["ZZ!5\n"]).poll(0.25)
    assert np.allclose(cmd.joint_delta, 0.0)
    assert cmd.grip == 0.25


def test_a_rig_wider_than_the_layout_is_refused(cfg):
    with pytest.raises(ValueError, match="keyboard layout"):
        KeyboardDevice(len(KEYS) + 1, read_keys=lambda: "")


def test_keyboard_drives_a_real_recording(cfg):
    """The decode path and the control loop, joined up, with no terminal."""
    script = ["q"] * 10 + ["]"] * 3 + [END_SUCCESS]
    queue = list(script)
    dev = KeyboardDevice(cfg.n_joints, read_keys=lambda: queue.pop(0) if queue else "")
    fake = FakeClock()
    rec = record_episode(SimulatedArm(cfg), dev, cfg, clock=fake.clock, sleep=fake.sleep)
    assert rec.success is True
    assert rec.n_steps == len(script) - 1
    assert rec.frame["joint_0"].iloc[-1] > rec.frame["joint_0"].iloc[0]
    assert rec.frame["gripper_cmd"].iloc[-1] > 0


# -- driving a real arm -------------------------------------------------------


def _midrange(cfg):
    """A pose every joint of the configured robot can legally hold."""
    return 0.5 * (np.asarray(cfg.joint_lower) + np.asarray(cfg.joint_upper))


def _hw(cfg, start=None):
    """A hardware arm over an echo driver, parked mid-range by default.

    Mid-range rather than zero: joint 4 of a Panda has an upper limit of
    -0.0698 rad, so an all-zero start is already out of range for it.
    """
    from teleop_pipeline.teleop.hardware import EchoDriver, HardwareArm

    if start is None:
        start = _midrange(cfg)
    driver = EchoDriver(cfg.n_joints, start=np.asarray(start, dtype=float))
    return HardwareArm(cfg, driver), driver


def test_hardware_arm_never_commands_past_a_joint_limit(cfg):
    """The driver may happily accept an out-of-range target. It never sees one."""
    arm, driver = _hw(cfg, start=np.asarray(cfg.joint_upper) - 1e-3)
    huge = np.full(cfg.n_joints, 10.0)
    arm.step(Command(joint_delta=huge, grip=0.0), dt=0.05)
    assert np.all(driver.q <= np.asarray(cfg.joint_upper) + 1e-9)


def test_hardware_arm_rate_limits_before_the_driver_sees_it(cfg):
    arm, driver = _hw(cfg)
    start = driver.q.copy()
    applied, _ = arm.step(Command(joint_delta=np.full(cfg.n_joints, 5.0), grip=0.0), dt=0.05)
    assert np.allclose(applied, cfg.action_limit)
    assert np.allclose(driver.q - start, cfg.action_limit)


def test_hardware_arm_records_the_censored_request_not_the_raw_one(cfg):
    """A command the rig refused has to stay visible as refused in the data."""
    arm, _ = _hw(cfg, start=np.asarray(cfg.joint_upper))
    applied, _ = arm.step(Command(joint_delta=np.full(cfg.n_joints, 1.0), grip=0.0), dt=0.05)
    assert np.allclose(applied, 0.0), "an arm already on its limit moved nowhere"


def test_hardware_arm_reads_the_arm_rather_than_integrating_its_own_commands(cfg):
    """A driver that lands somewhere else must be believed, not overwritten."""
    from teleop_pipeline.teleop.hardware import EchoDriver, HardwareArm

    class DriftingDriver(EchoDriver):
        def write(self, q_target, grip_target):
            super().write(np.asarray(q_target) - 0.01, grip_target)

    driver = DriftingDriver(cfg.n_joints, start=_midrange(cfg))
    arm = HardwareArm(cfg, driver)
    arm.step(Command(joint_delta=np.full(cfg.n_joints, 0.02), grip=0.0), dt=0.05)
    assert np.allclose(arm.state().q, driver.q), "state came from the command, not the arm"


def test_hardware_arm_stops_and_raises_when_a_read_fails(cfg):
    """Carrying on would record an episode the arm did not perform."""
    from teleop_pipeline.teleop.hardware import DriverError, EchoDriver, HardwareArm

    class FailsAfterFirstRead(EchoDriver):
        def __init__(self, n, start):
            super().__init__(n, start=start)
            self.reads = 0

        def read(self):
            self.reads += 1
            # Construction reads once; the failure has to land inside step().
            if self.reads > 1:
                raise OSError("bus timeout")
            return super().read()

    driver = FailsAfterFirstRead(cfg.n_joints, start=_midrange(cfg))
    arm = HardwareArm(cfg, driver)
    with pytest.raises(DriverError, match="could not read"):
        arm.step(Command.idle(cfg.n_joints), dt=0.05)
    assert driver.holds == 1, "the arm was left moving after a failed read"


def test_hardware_arm_holds_when_a_write_fails(cfg):
    from teleop_pipeline.teleop.hardware import DriverError, EchoDriver, HardwareArm

    class WriteFails(EchoDriver):
        def write(self, q_target, grip_target):
            raise OSError("no ack")

    driver = WriteFails(cfg.n_joints, start=_midrange(cfg))
    arm = HardwareArm(cfg, driver)
    with pytest.raises(DriverError, match="could not command"):
        arm.step(Command.idle(cfg.n_joints), dt=0.05)
    assert driver.holds == 1


def test_hardware_arm_rejects_a_wrong_joint_count(cfg):
    """Better to fail at the first read than to clip the wrong joints all session."""
    from teleop_pipeline.teleop.hardware import DriverError, EchoDriver, HardwareArm

    driver = EchoDriver(cfg.n_joints + 1)
    with pytest.raises(DriverError, match="expected"):
        HardwareArm(cfg, driver)


def test_hardware_arm_refuses_to_start_outside_the_configured_limits(cfg):
    """Clipping instead would leave that joint dead all session, looking like
    broken hardware rather than a params.yaml copied from another robot."""
    from teleop_pipeline.teleop.hardware import DriverError, EchoDriver, HardwareArm

    driver = EchoDriver(cfg.n_joints, start=np.zeros(cfg.n_joints))
    with pytest.raises(DriverError, match="outside the joint limits"):
        HardwareArm(cfg, driver)


def test_a_joint_resting_just_past_its_stop_is_tolerated(cfg):
    """Encoder noise and a calibration offset are not a configuration error."""
    from teleop_pipeline.teleop.hardware import EchoDriver, HardwareArm

    start = np.asarray(cfg.joint_upper, dtype=float) + 0.01
    HardwareArm(cfg, EchoDriver(cfg.n_joints, start=start))


def test_hardware_arm_rejects_a_non_finite_reading(cfg):
    from teleop_pipeline.teleop.hardware import DriverError, EchoDriver, HardwareArm

    driver = EchoDriver(cfg.n_joints, start=_midrange(cfg))
    driver.q[0] = np.nan
    with pytest.raises(DriverError, match="non-finite"):
        HardwareArm(cfg, driver)


def test_hardware_arm_gripper_cannot_cross_its_travel_in_one_step(cfg):
    arm, driver = _hw(cfg)
    arm.step(Command(joint_delta=np.zeros(cfg.n_joints), grip=1.0), dt=0.01)
    assert driver.grip < 1.0, "the gripper teleported, which no real rig does"


def test_hardware_arm_satisfies_the_same_protocol_as_the_simulated_one(cfg):
    from teleop_pipeline.teleop.arm import Arm

    arm, _ = _hw(cfg)
    assert isinstance(arm, Arm)
