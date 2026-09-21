# Running this on a real arm

Written to be followed at the rig, in order, by someone with a month of lab
access and no time to lose to a config file.

## Before the visit

Three answers, none of which need the lab:

| Question | Why it blocks everything |
| --- | --- |
| Who owns data and models generated here, in writing? | Every hour of collection is worth nothing if the answer comes back wrong afterwards. |
| Which task? | Named, with a definition of success an operator would agree with, before any recording. |
| Which arm, and what are its joint limits and control rate? | These go in `params.yaml`. Getting them wrong costs a day at the rig, not five minutes. |

## 1. Configure the robot

`params.yaml`, the `robot` block. The shipped values are a Franka Panda and
will be wrong for anything else.

```yaml
robot:
  name: <arm>
  n_joints: 6
  control_hz: 20.0            # what the rig can actually sustain, not what you want
  joint_lower: [...]          # radians, from the manufacturer, one per joint
  joint_upper: [...]
  action_limit: 0.08          # max commanded delta per step at control_hz
```

`action_limit` is the one people guess. Take the joint's rated velocity in
rad/s and divide by `control_hz`, then halve it for the first session. It is
the rate limit that stops a jerk on the input device becoming a demand the arm
answers at full speed.

## 2. Write the driver

One class, three methods, against whatever SDK the arm has:

```python
from teleop_pipeline.teleop.hardware import HardwareArm, JointDriver

class MyArm(JointDriver):
    def read(self):
        # -> (joint positions in radians, gripper opening in [0, 1])
        return self.sdk.joint_positions(), self.sdk.gripper() / MAX_OPENING

    def write(self, q_target, grip_target):
        self.sdk.move_to(q_target)
        self.sdk.set_gripper(grip_target * MAX_OPENING)

    def hold(self):
        self.sdk.stop()          # required if the arm is velocity or torque controlled
```

Everything else is already enforced by `HardwareArm` and covered by tests:
joint limits, the rate limit, gripper travel time, reading the arm rather than
integrating your own commands, and stopping on a driver error instead of
recording an episode the arm never performed. It also refuses to start if the
arm sits outside the limits in `params.yaml`, which is how a config copied from
a different robot announces itself instead of leaving one joint mysteriously
dead.

**Units and joint order are yours to get right.** Radians, and the robot's own
joint order. Convert in the driver so every stage downstream is identical
across rigs.

## 3. Dry run, with the arm powered off

```python
from teleop_pipeline.teleop.hardware import EchoDriver, HardwareArm
```

`EchoDriver` moves exactly where it is told and models nothing. Swap it for
yours to check wiring, units and joint order end to end before anything can
move. Then run the real driver with the arm powered down, if the SDK allows it,
and confirm `read()` returns plausible numbers in radians.

## 4. First episode

```bash
teleop-pipeline record --task <task> --operator <you> --robot <arm> --episodes 1
teleop-pipeline ingest
teleop-pipeline validate
teleop-pipeline score
```

Look at the score before recording anything else. A first episode that scores
badly is usually the rig, not the operator: check `late_steps` in the record
output and the timing jitter in the quality report. A loop that cannot hold its
nominal rate produces data whose timestamps cannot be trusted, and no amount of
collection fixes it afterwards.

## 5. Collect

Aim for **50 to 200 episodes of one task** before training anything. Vary the
object's starting position across the workspace, not the task. Keep failures:
an episode marked failed is supervision about what not to do, and deleting them
biases the corpus toward a world where nothing goes wrong.

Two operators is better than one, because a val split with a held-out operator
measures something a same-operator split cannot.

## 6. Train and evaluate

```bash
teleop-pipeline dataset
teleop-pipeline train
teleop-pipeline eval
```

Or open the app and press Run everything:

```bash
teleop-pipeline studio
```

**Read the baseline row, not the error.** On smooth teleoperation, repeating
the previous action is a very strong predictor, and an action MAE that looks
good against nothing usually loses to it. The eval prints the comparison per
horizon offset. A chunked policy that loses at offset 0 and wins further out is
behaving correctly; one that loses everywhere has learned nothing useful,
whatever its loss curve says.

## 7. Publish it either way

The result is worth reporting whichever direction it goes, and a negative one
is worth more here than elsewhere: almost nobody publishes real-hardware
manipulation results with an honest baseline, so a clean negative is a
contribution rather than an embarrassment. Include the failures, the quality
distribution of the corpus, and the exact dataset hash the checkpoint was
trained on, all of which the pipeline already records.

## What you do not need to buy

Nothing, if the lab has an arm and a GPU box. The one thing worth owning is a
teleoperation input device if the lab has none, because keyboard-driven
demonstrations are jerky and jerky demonstrations train worse policies.
