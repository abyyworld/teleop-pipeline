# erl-teleop-pipeline

**Versioned, validated, reproducible teleoperation data for robot policy learning.**

Teleoperation datasets in academic labs are usually a folder tree of CSVs with a
naming convention that lived in one person's head. This repository is the
infrastructure that replaces it: raw sessions go in, a schema-checked and
quality-scored corpus comes out, and every trained policy can be traced back to
the exact bytes it was trained on.

```bash
git clone https://github.com/abyyworld/erl-teleop-pipeline
cd erl-teleop-pipeline
make install-all
dvc repro          # synthetic sessions -> corpus -> dataset -> policy -> eval
```

That command runs the entire pipeline on generated data, so the repository is
assessable without access to a lab's demonstrations. Point `ingest.raw_dir` at a
real teleop dump and delete the `synth` stage to use it for real.

---

## Why this exists

Four failures cost academic robotics labs more time than any modelling problem:

1. **A result cannot be rebuilt.** The paper says 87% success. Nobody can say
   which demonstrations produced it, and the folder has been added to since.
2. **Bad demonstrations are never noticed.** Nobody watches 4,000 episodes. The
   dropped frames, the operator who idles for six seconds mid-task, the session
   where the follower arm was not tracking — all of it silently becomes training
   data.
3. **Validation numbers are inflated.** Episodes from one sitting land on both
   sides of the split, and the model is graded on near-duplicates of what it
   trained on.
4. **A metric has no baseline.** Loss went down, so the policy must be learning.

Each has a specific answer here.

| Failure | What this repo does |
| --- | --- |
| Cannot rebuild a result | DVC DAG + `dvc.lock` + a `lineage.json` pinning commit, dataset hash and checkpoint SHA |
| Bad demos go unnoticed | Nine quality metrics, each targeting a named teleop failure mode, with tiering and flags |
| Inflated validation | Splits grouped on session; normalisation fitted on train only; duplicates dropped by content hash |
| Metrics without baselines | Every eval reports zero / train-mean / persistence baselines and `skill_vs_best_baseline` |

## Pipeline

```
raw sessions ──▶ ingest ──▶ validate ──▶ score ──▶ dataset ──▶ train ──▶ evaluate
   (CSV)         canonical   hard gate   quality   splits +    policy   baselines +
                  parquet               tiering    manifest    + lineage  bootstrap CI
```

Each box is a DVC stage and a CLI subcommand, so the pipeline and a human at a
terminal run the same code. No stage writes into another stage's outputs — the
rule that keeps `dvc repro`'s staleness detection honest.

### 1. Ingest — `erl-teleop ingest`

Normalises whatever the rig produced into one schema. Column aliasing is
table-driven (`joint_0`, `vel_3`, `timestamp`, `gripper_cmd` → canonical), so
onboarding a new rig means adding entries, not writing a parser.

Two things happen here that are easy to skip and expensive to skip:

- **Resampling** onto an exact `1/control_hz` grid. Teleop loops do not run at
  their nominal rate, and training a fixed-dt model on jittery data biases every
  velocity and delta in the corpus.
- **Gap preservation.** Interpolating across a two-second dropout invents motion
  that never happened. Gaps wider than `ingest.max_gap_s` stay NaN.

Raw timing statistics are measured *before* resampling erases them and carried
forward on the episode record, because jitter is one of the strongest available
predictors of a bad session.

### 2. Validate — `erl-teleop validate`

A hard gate: is this a well-formed recording of *this* robot? Monotonic
timebase, schema completeness, joint limits, unit quaternions, gripper range.
Errors exclude an episode; warnings do not. A joint-space-only rig with no
end-effector channel is legitimate and passes with a warning; scattered NaNs
inside a channel that *is* recorded are corruption and fail.

### 3. Score — `erl-teleop score`

The part that does not exist in a folder of CSVs. Nine metrics, each aimed at a
recognisable way a human driving a leader arm produces unusable data:

| Metric | What it catches |
| --- | --- |
| `dropped_frame_rate` | Logging dropouts — holes teach the policy jumps it cannot reproduce |
| `timing_jitter` | A compute-starved rig; every recorded velocity is correspondingly wrong |
| `nan_fraction` | Missing sensor channels |
| `idle_fraction` | The operator thinking. Teaches the policy to freeze — common and fatal |
| `action_saturation` | Operator pinned at the leader-arm limit, so the recorded action is *censored intent* |
| `jerk_spike_rate` | Tracker glitches, robust-thresholded per episode so speed is not penalised as noise |
| `gripper_chatter_hz` | Operator indecision at the grasp, hysteresis-debounced |
| `tracking_error` | Follower arm not keeping up — the recorded action never caused the recorded state |
| `duration_zabs` | Bailed-out attempts and ones where the operator got lost |

Values map to penalties through `(good, bad)` anchors in `params.yaml`; the
composite score assigns **gold / silver / reject**. Hard-reject conditions apply
independently, because an episode can average well and still be unusable.

**Does it work?** The synthetic generator gives four operators different skill
levels. The scorer never sees them, and recovers the ranking exactly:

| Operator | Generator skill | Mean quality score |
| --- | ---: | ---: |
| `op_amelia` | 0.92 | 96.1 |
| `op_bram` | 0.78 | 90.5 |
| `op_chidi` | 0.55 | 86.3 |
| `op_dara` | 0.35 | 83.6 |

That correspondence is asserted in the test suite
(`test_quality_recovers_the_generators_operator_ranking`), so it stays true.

### 4. Dataset — `erl-teleop dataset`

Three enforced decisions, each corresponding to a way of accidentally inflating
a validation number:

- **Split on session, never on episode.** Episodes from one sitting share an
  operator, a calibration and a scene layout.
- **Fit normalisation on train only.** A small leak, and it makes every
  subsequent comparison quietly dishonest.
- **Deduplicate by content hash.** Re-copied folders are routine and silently
  reweight the corpus.

The split is hash-based, not shuffle-based, so **adding sessions later does not
reshuffle existing ones** — otherwise every previously reported number becomes
incomparable and nothing tells you it happened.

Output includes a `manifest.json` pinning every episode's content hash, which is
what `dataset_hash` is derived from and what training runs record.

### 5. Train — `erl-teleop train`

A small MLP behaviour-cloning baseline. The point of this repository is the
infrastructure around the model, and a policy that trains in a minute validates
that infrastructure far better than one needing a GPU-hour. Replace `BCPolicy`
and nothing else changes.

- **Action chunking** (`train.action_horizon`): predicts the next *N* actions.
  The cheapest known mitigation for compounding error in BC, at one config line.
- **Huber loss**: teleop actions have heavy tails, and under MSE those tails
  dominate the gradient and produce an arm that drifts without committing.

Checkpoints are self-describing — weights, normalisation, column order — so they
load without this package or its config.

### 6. Evaluate — `erl-teleop eval`

**There is no simulator here.** These are open-loop metrics on held-out
demonstrations. They catch broken checkpoints, normalisation mismatches and
regressions between data versions; they do not measure task success. Closed-loop
benchmarking is [`erl-vla-evals`](https://github.com/abyyworld/erl-vla-evals),
which consumes the checkpoint and lineage emitted here.

- Errors in **physical units** (rad), not normalised — normalised losses are not
  comparable across data versions, because the normaliser changes with the data.
- **Cluster bootstrap CIs** resampling whole episodes. Timesteps within an
  episode are strongly correlated; bootstrapping over them treats 500 correlated
  samples as 500 independent ones and yields intervals several times too narrow.
- **Baselines on every run.** See below.

## What the eval harness caught

The BC policy is **27% worse than repeating the previous action**.

| Predictor | chunk MAE (rad) |
| --- | ---: |
| zero | 0.07261 |
| train-set mean | 0.06670 |
| **persistence** | **0.01255** |
| BC policy | 0.01595 |

Against `zero` the same policy looks like a 78% error reduction — the number
that would have gone on a slide. Only one of those comparisons is informative.

This is why baselines are computed on every evaluation and why the number is
reported even when it is unflattering. On this synthetic fixture persistence is
close to optimal by construction (the unpredictable residual is injected noise),
so the negative figure is expected here and is *not* evidence the pipeline is
broken — it is evidence the harness refuses to award credit the data does not
support. [`docs/BASELINES.md`](docs/BASELINES.md) has the full analysis,
including the earlier configuration that scored −220% and why.

## Reproducibility

Every training run writes a `lineage.json`:

```json
{
  "dataset_hash": "4ddbf4ffc8a625b6",
  "git": { "commit": "…", "branch": "main", "dirty": false },
  "dvc_lock_hash": "…",
  "checkpoint": { "sha256": "…", "bytes": 1259304 },
  "params": { "train": {…}, "dataset": {…} },
  "environment": { "python": "3.12.13", "platform": "…" }
}
```

`dirty` is recorded deliberately: a dirty tree means the commit does not
describe what actually ran, and that flag is the difference between
"reproducible" and "probably reproducible".

```bash
erl-teleop lineage       # what produced the latest run, and how to rebuild it
dvc metrics diff HEAD~1  # what changed, and by how much
```

## Orchestration

`dvc repro` and the Prefect flow answer different questions, and the split is
deliberate:

- **DVC** — *"can I rebuild the result from six months ago?"* A content-addressed
  DAG over a fixed corpus, run on demand.
- **Prefect** (`flows/ingest_flow.py`) — *"a session landed on the NAS at 6pm on
  a Friday, is it any good?"* Discover, ingest, validate, score, quarantine,
  report. Sessions that error are deliberately left unrecorded so the next run
  retries them, and an abnormal drop rate is logged loudly — a rig that has come
  loose shows up there long before anyone notices the policy got worse.

Prefect rather than Airflow because Airflow needs a scheduler, a metadata
database and a webserver kept alive by someone. In a lab, that person graduates.

## Tool choices

| Choice | Over | Because |
| --- | --- | --- |
| DVC | LakeFS | Git-native, no server to run. LakeFS needs a deployment a lab will not maintain. |
| Prefect | Airflow | A flow is a Python function. No control plane to rot. |
| MLflow | W&B | Self-hosts at `file:./mlruns` with zero infrastructure and no accounts. Point `MLFLOW_TRACKING_URI` at a lab server and nothing in the code changes. |
| Parquet | HDF5 | DVC dedupes it well, stays columnar for the stats passes, no C-extension dance. |

Tracking degrades rather than fails: if MLflow is unreachable the run falls back
to a local JSONL log. Losing a training run because a metrics database was down
is a self-inflicted wound.

## Layout

```
params.yaml              every threshold and hyperparameter, hashed by DVC
dvc.yaml                 the reproducible DAG
src/erl_teleop/
  schema.py              canonical episode format + pydantic records
  ingest.py              raw -> canonical, aliasing, resampling, gap handling
  validate.py            hard structural + physical gate
  quality.py             the nine metrics, scoring and tiering
  dataset.py             session-grouped splits, manifest, dataset hash
  train.py               BC policy with action chunking
  evaluate.py            open-loop metrics, baselines, cluster bootstrap
  lineage.py             commit + data version + checkpoint provenance
  tracking.py            MLflow / JSONL / null, behind one interface
  report.py              Markdown data-quality report
  synthetic.py           session generator with injected defects
flows/ingest_flow.py     Prefect operational ingestion
tests/                   54 tests; each quality metric has a defect-injection test
```

## Commands

```bash
make install-all    # venv + every extra
make test           # 54 tests
make lint           # ruff
make repro          # rebuild whatever is stale
make pipeline       # run every stage directly, without DVC
make metrics        # tracked metrics and the diff against HEAD
make ui             # MLflow UI on the local run store
```

## Status

Working end to end on synthetic data, with CI reproducing the full pipeline on
every push and failing the build if the reject rate exceeds 25% or lineage stops
being written. Not yet run against real teleoperation sessions — the schema and
alias table are the two places that will need extending when it is.

## Licence

MIT.
