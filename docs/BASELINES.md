# Baselines, and what the eval harness caught

This document exists because the first version of this pipeline reported a
validation loss, the loss went down, and everything looked fine. It wasn't.

## The finding

The behaviour-cloning policy is **worse than repeating the previous action**.

Measured on the synthetic corpus (48 sessions, 289 episodes, 55 held-out
episodes across 10 held-out sessions):

| Predictor | chunk MAE (rad) | next-step MAE (rad) |
| --- | ---: | ---: |
| zero (don't move) | 0.07261 | 0.07255 |
| training-set mean action | 0.06670 | 0.06668 |
| **persistence (repeat last action)** | **0.01255** | **0.00362** |
| BC policy (this repo) | 0.01595 | 0.00779 |

Against `zero` the policy looks excellent — a 78% error reduction, the kind of
number that goes in a slide. Against `persistence` it is 27% *worse*. Only one
of those two comparisons is informative, and it is not the flattering one.

This is why `evaluate.py` computes all three baselines on every run and why
`skill_vs_best_baseline` is negative until the policy actually earns a positive
number. A loss curve cannot tell you this. A number without a baseline is not a
result.

## Where the policy does earn its keep

Broken out per chunk offset, the picture is more interesting than the aggregate:

| Offset | Policy MAE | Persistence MAE | Policy vs persistence |
| ---: | ---: | ---: | ---: |
| 0 | 0.00779 | 0.00362 | −115% |
| 1 | 0.01047 | 0.00666 | −57% |
| 2 | 0.01274 | 0.00919 | −39% |
| 3 | 0.01497 | 0.01152 | −30% |
| 4 | 0.01702 | 0.01369 | −24% |
| 5 | 0.01920 | 0.01606 | −20% |
| 6 | 0.02160 | 0.01859 | −16% |
| 7 | 0.02382 | 0.02106 | −13% |

The gap closes monotonically. Persistence degrades quickly as it is asked to
predict further ahead; the policy degrades more slowly, because it is committing
to a plan rather than extrapolating the last command. On this data the curves do
not cross inside the 8-step horizon, but the trend is the thing a longer horizon
or a stronger architecture would exploit.

## Why persistence is so strong *here*

It is largely an artefact of the fixture, and saying so matters.

The synthetic generator builds trajectories as minimum-jerk segments between
waypoints, plus a smoothed tremor term. The action is the per-step joint delta,
so consecutive actions are almost identical by construction — the smooth part is
trivially extrapolated, and the part that is *not* extrapolable is injected
noise that nothing can predict. Persistence is therefore close to Bayes-optimal
on synthetic data, and no policy should be expected to beat it by much.

**So do not read the −27% as "the model is broken."** Read it as: the harness
correctly refuses to award credit that the data does not support. On real
teleoperation the residual is operator intent, not white noise, and that is
learnable — `skill_vs_best_baseline` is the number to watch when the real
sessions arrive. If it is still negative on real data, the policy genuinely is
not learning, and you will know on day one instead of after a paper deadline.

## The other thing this caught

The first configuration did not include the previous action in the observation
at all (`obs_keys: [q, dq, ee_pos, ee_quat, grip]`). It scored **−220%** against
persistence, because the policy had no way to express "keep doing what you were
doing" — the information simply wasn't in its input.

Adding `prev_act_q` and `prev_act_grip` moved it to −35%, and more data moved it
to −27%.

That fix has a cost worth stating. Feeding the previous action back in is the
classic setup for **causal confusion** in behaviour cloning: the policy can learn
to copy its own last action and stop attending to the state, which looks superb
offline and drifts on hardware. Two things in this repo push back on it:

- `train.action_horizon > 1` forces the model to commit to a chunk, which the
  previous action alone cannot supply.
- `rollout_l2@N` in `evaluate.py` feeds the policy's *own* predictions back into
  the `prev_act_*` observation slots rather than the recorded ones. Getting this
  wrong turns an open-loop rollout into a teacher-forced one and understates
  drift by a wide margin — it is an easy bug to ship and a hard one to notice.

To measure the trade-off yourself, drop the two `prev_act_*` keys from
`dataset.obs_keys` and run `dvc repro`. DVC will invalidate the dataset, train
and evaluate stages and leave the rest alone.

## Reproducing this table

```bash
dvc repro
cat reports/eval_val.json
```
