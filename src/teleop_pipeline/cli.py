"""Command-line interface.

Every DVC stage shells out to one of these, so the pipeline and a human at a
terminal run identical code paths. Anything you can reproduce with `dvc repro`
you can also run and debug one stage at a time.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import load_config

app = typer.Typer(
    add_completion=False,
    help="Versioned, validated, reproducible teleoperation data pipeline.",
    no_args_is_help=True,
)
console = Console()

ParamsOpt = typer.Option(None, "--params", "-p", help="Path to params.yaml.")


def _cfg(params: str | None):
    return load_config(params)


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"teleop-pipeline {__version__}")


@app.command()
def synth(
    params: str = ParamsOpt,
    sessions: int = typer.Option(12, help="Number of synthetic teleop sessions."),
    seed: int = typer.Option(0, help="Generator seed."),
    out: str = typer.Option(None, help="Output directory (default: ingest.raw_dir)."),
) -> None:
    """Generate synthetic raw teleop sessions with realistic defects.

    Lets the whole pipeline run end to end on a fresh clone, and gives the
    quality scorer known-bad episodes to actually catch.
    """
    from .synthetic import generate

    cfg = _cfg(params)
    target = Path(out) if out else cfg.resolve("ingest.raw_dir")
    written = generate(cfg, target, n_sessions=sessions, seed=seed)
    console.print(f"[green]wrote[/] {written} episode(s) across {sessions} session(s) -> {target}")


@app.command()
def record(
    params: str = ParamsOpt,
    task: str = typer.Option(..., "--task", help="Task identifier, e.g. pick_place_block."),
    operator: str = typer.Option(..., "--operator", help="Operator identifier."),
    robot: str = typer.Option("sim_01", "--robot", help="Robot identifier."),
    device: str = typer.Option("keyboard", help="Input device: keyboard or scripted."),
    episodes: int = typer.Option(1, help="Episodes to record before stopping."),
    out: str = typer.Option(None, help="Output directory (default: ingest.raw_dir)."),
    session_id: str = typer.Option(None, help="Session id (default: derived from the clock)."),
    notes: str = typer.Option("", help="Free text stored in session.json."),
) -> None:
    """Record teleop demonstrations from a live operator.

    Writes the same raw layout any other rig produces, so the recording is
    ingested, validated, scored and hashed by the existing pipeline rather than
    a special path for our own data. Follow with `ingest`.

    The simulated arm is the default because it lets the recording path be run
    and tested with no robot present. Swap it for a real arm by implementing
    `teleop.arm.Arm`; nothing else in the pipeline changes.
    """
    from datetime import datetime, timezone

    from .teleop import KeyboardDevice, ScriptedDevice, SimulatedArm
    from .teleop.device import HELP
    from .teleop.recorder import record_episode, scripted_reach, write_session

    cfg = _cfg(params)
    target = Path(out) if out else cfg.resolve("ingest.raw_dir")
    started = datetime.now(timezone.utc)
    sid = session_id or f"sess_{started:%Y%m%dT%H%M%S}_{task}"

    arm = SimulatedArm(cfg)
    recordings = []
    for e in range(episodes):
        if device == "keyboard":
            dev = KeyboardDevice(cfg.n_joints)
            console.print(f"[bold]episode {e + 1}/{episodes}[/] — {task}")
            console.print(HELP)
        elif device == "scripted":
            dev = ScriptedDevice(cfg.n_joints, scripted_reach(cfg), steps=60)
        else:
            raise typer.BadParameter(f"unknown device {device!r}; use keyboard or scripted")
        try:
            rec = record_episode(arm, dev, cfg)
        finally:
            dev.close()
        recordings.append(rec)
        label = "ok" if rec.success else "fail"
        late = f", {rec.late_steps} late step(s)" if rec.late_steps else ""
        console.print(f"  {rec.n_steps} step(s), marked [bold]{label}[/]{late}")
        if rec.quit_requested:
            break

    session_dir = write_session(
        target,
        recordings,
        cfg=cfg,
        session_id=sid,
        operator_id=operator,
        robot_id=robot,
        task_id=task,
        notes=notes,
        recorded_at=started,
    )
    console.print(f"[green]wrote[/] {len(recordings)} episode(s) -> {session_dir}")


@app.command()
def ingest(
    params: str = ParamsOpt,
    prune: bool = typer.Option(
        True,
        "--prune/--no-prune",
        help="Remove stored sessions whose raw source is gone. On by default: "
        "ingestion is a sync, so a retracted session actually disappears "
        "instead of lingering in every future training set.",
    ),
) -> None:
    """Normalise raw sessions into the canonical episode store."""
    from .ingest import ingest_all

    cfg = _cfg(params)
    result = ingest_all(cfg, prune=prune)
    console.print(f"[green]ingest[/] {result.summary()}")
    for path, reason in result.skipped:
        console.print(f"  [yellow]skipped[/] {Path(path).name}: {reason}")
    for session_id in result.pruned:
        console.print(f"  [magenta]pruned[/] {session_id} (raw source removed)")
    if result.episodes == 0:
        raise typer.Exit(code=1)


@app.command()
def validate(
    params: str = ParamsOpt,
    quarantine: bool = typer.Option(
        False,
        "--quarantine/--no-quarantine",
        help="Physically move failing episodes into ingest.quarantine_dir. "
        "Off by default so the DVC stage stays pure; the Prefect ingestion "
        "flow turns it on, since there the point is to get bad sessions out "
        "of the way of the next run.",
    ),
    out: str = typer.Option("reports/validation.json", help="Where to write the report."),
) -> None:
    """Schema- and physics-check every episode. Failures are excluded downstream."""
    from .validate import validate_store

    cfg = _cfg(params)
    reports = validate_store(cfg, quarantine=quarantine)
    failed = [r for r in reports if not r.ok]
    warned = [r for r in reports if r.ok and r.warnings]

    path = cfg.root / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.model_dump() for r in reports], indent=2), encoding="utf-8")

    console.print(
        f"[green]validate[/] {len(reports)} episode(s): "
        f"{len(reports) - len(failed)} ok, {len(failed)} quarantined, {len(warned)} with warnings"
    )
    for r in failed[:10]:
        codes = ", ".join(i.code for i in r.errors)
        console.print(f"  [red]fail[/] {r.episode_id}: {codes}")
    if len(failed) > 10:
        console.print(f"  … and {len(failed) - 10} more")


@app.command()
def score(
    params: str = ParamsOpt,
    out: str = typer.Option("reports/quality.json", help="Where to write per-episode scores."),
    write_back: bool = typer.Option(
        False,
        "--write-back/--no-write-back",
        help="Also stamp the score and tier into each episode's sidecar. "
        "Convenient when browsing the store by hand; off in the pipeline so "
        "the stage does not mutate ingest's outputs.",
    ),
) -> None:
    """Score every episode for demonstration quality and assign a tier."""
    from .quality import score_store, summarise

    cfg = _cfg(params)
    reports = score_store(cfg, write_back=write_back)
    if not reports:
        console.print("[red]no episodes to score[/] — run `teleop-pipeline ingest` first")
        raise typer.Exit(code=1)

    path = cfg.root / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.model_dump() for r in reports], indent=2), encoding="utf-8")

    summary = summarise(reports)
    (path.parent / "quality_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    table = Table(title=f"Quality — {summary['n_episodes']} episodes")
    table.add_column("tier")
    table.add_column("count", justify="right")
    table.add_column("share", justify="right")
    for tier in ("gold", "silver", "reject"):
        n = summary["tiers"].get(tier, 0)
        table.add_row(tier, str(n), f"{n / summary['n_episodes']:.0%}")
    console.print(table)
    console.print(
        f"mean [bold]{summary['score']['mean']}[/]  "
        f"median [bold]{summary['score']['median']}[/]  "
        f"p10 [bold]{summary['score']['p10']}[/]"
    )
    if summary["flags"]:
        top = ", ".join(f"{k} ({v})" for k, v in list(summary["flags"].items())[:5])
        console.print(f"top flags: {top}")


@app.command()
def dataset(params: str = ParamsOpt) -> None:
    """Build session-grouped train/val splits and the dataset manifest."""
    from .dataset import build_dataset

    cfg = _cfg(params)
    manifest = build_dataset(cfg)

    table = Table(title=f"Dataset {manifest['dataset_hash']}")
    table.add_column("split")
    table.add_column("episodes", justify="right")
    table.add_column("sessions", justify="right")
    table.add_column("steps", justify="right")
    table.add_column("mean quality", justify="right")
    for split in ("train", "val"):
        s = manifest["splits"][split]
        table.add_row(
            split,
            str(s["n_episodes"]),
            str(s["n_sessions"]),
            f"{s['n_steps']:,}",
            f"{s['mean_quality']:.1f}",
        )
    console.print(table)
    if manifest["excluded"]:
        console.print("excluded: " + ", ".join(f"{k}={v}" for k, v in manifest["excluded"].items()))

    # Operator overlap between splits is legal but worth knowing about: it means
    # val measures generalisation to a new session, not to a new operator.
    tr = set(manifest["splits"]["train"]["operators"])
    va = set(manifest["splits"]["val"]["operators"])
    if va - tr:
        console.print(f"[green]held-out operator(s)[/]: {sorted(va - tr)}")
    else:
        console.print("[yellow]note[/]: every val operator also appears in train")


@app.command()
def train(
    params: str = ParamsOpt,
    run_name: str = typer.Option(None, help="Override the generated run name."),
) -> None:
    """Train the behaviour-cloning policy and emit a traceable checkpoint."""
    from .train import train as run_train

    cfg = _cfg(params)
    summary = run_train(cfg, run_name=run_name)
    console.print(
        f"[green]trained[/] {summary['run_name']}  "
        f"val {summary['best_val_loss']:.5f} @ epoch {summary['best_epoch']}"
    )
    console.print(f"checkpoint: {summary['checkpoint']}")
    console.print(f"lineage:    {summary['lineage']}")


@app.command("eval")
def eval_cmd(
    params: str = ParamsOpt,
    checkpoint: str = typer.Option(None, help="Checkpoint path (default: latest run)."),
    split: str = typer.Option("val", help="Which split to evaluate."),
) -> None:
    """Evaluate a checkpoint offline, with cluster-bootstrap confidence intervals."""
    from .evaluate import evaluate

    cfg = _cfg(params)
    ckpt = Path(checkpoint) if checkpoint else _latest_checkpoint(cfg)
    report = evaluate(cfg, ckpt, split=split)
    m = report["metrics"]

    table = Table(title=f"Eval — {Path(ckpt).name} on {split}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    lo, hi = m["action_mae_ci95"]
    table.add_row("action MAE (rad)", f"{m['action_mae']:.5f}  [{lo:.5f}, {hi:.5f}]")
    table.add_row("action RMSE (rad)", f"{m['action_rmse']:.5f}")
    table.add_row("chunk MAE (rad)", f"{m['chunk_mae']:.5f}")
    for name, values in sorted(m.get("baselines", {}).items(), key=lambda kv: kv[1]["chunk_mae"]):
        table.add_row(
            f"  baseline: {name}",
            f"{values['chunk_mae']:.5f}  (1-step {values['next_step_mae']:.5f})",
        )
    skill = m.get("skill_vs_best_baseline")
    if skill is not None:
        colour = "green" if skill > 0 else "red"
        table.add_row(f"vs best ({m.get('best_baseline')})", f"[{colour}]{skill:+.1%}[/]")
    if "gripper_accuracy" in m:
        table.add_row("gripper accuracy", f"{m['gripper_accuracy']:.3f}")
    for key in sorted(k for k in m if k.startswith("rollout_l2@")):
        if m[key] is not None:
            table.add_row(key, f"{m[key]:.4f}")
    table.add_row("windows / episodes", f"{m['n_windows']:,} / {m['n_episodes']}")
    console.print(table)


@app.command()
def report(
    params: str = ParamsOpt,
    out: str = typer.Option("reports/data_quality.md", help="Markdown output path."),
) -> None:
    """Render the Markdown data-quality report."""
    from . import report as report_mod
    from .schema import QualityReport, ValidationReport

    cfg = _cfg(params)
    quality_path = cfg.root / "reports" / "quality.json"
    if not quality_path.exists():
        console.print("[red]no quality report[/] — run `teleop-pipeline score` first")
        raise typer.Exit(code=1)

    quality = [
        QualityReport.model_validate(r)
        for r in json.loads(quality_path.read_text(encoding="utf-8"))
    ]

    validation = None
    validation_path = cfg.root / "reports" / "validation.json"
    if validation_path.exists():
        validation = [
            ValidationReport.model_validate(r)
            for r in json.loads(validation_path.read_text(encoding="utf-8"))
        ]

    path = report_mod.write(cfg.root / out, report_mod.render(quality, validation))
    console.print(f"[green]wrote[/] {path}")


@app.command()
def lineage(
    params: str = ParamsOpt,
    run: str = typer.Option(None, help="Run name (default: latest)."),
) -> None:
    """Show how a run was produced, and the command to reproduce it."""
    from .lineage import Lineage

    cfg = _cfg(params)
    runs_dir = cfg.root / "artifacts" / "runs"
    if run:
        path = runs_dir / run / "lineage.json"
    else:
        candidates = sorted(runs_dir.glob("*/lineage.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            console.print("[red]no runs found[/] — run `teleop-pipeline train` first")
            raise typer.Exit(code=1)
        path = candidates[-1]

    data = Lineage.read(path)
    table = Table(title=f"Lineage — {data['run_name']}")
    table.add_column("field")
    table.add_column("value")
    git = data.get("git") or {}
    commit = (git.get("commit") or "—")[:12]
    table.add_row("commit", commit + ("  [red](dirty tree)[/]" if git.get("dirty") else ""))
    table.add_row("branch", str(git.get("branch") or "—"))
    table.add_row("dataset", data["dataset_hash"])
    table.add_row("dvc.lock", str(data.get("dvc_lock_hash") or "—"))
    ckpt = data.get("checkpoint") or {}
    table.add_row("checkpoint", str(ckpt.get("filename") or "—"))
    table.add_row("sha256", str(ckpt.get("sha256", ""))[:16] or "—")
    for k, v in (data.get("metrics") or {}).items():
        table.add_row(k, f"{v:.6f}" if isinstance(v, float) else str(v))
    console.print(table)
    console.print(f"\nreproduce:\n  {Lineage(**data).reproduction_command()}")


def _latest_checkpoint(cfg) -> Path:
    latest = cfg.root / "artifacts" / "latest_run.json"
    if latest.exists():
        return Path(json.loads(latest.read_text(encoding="utf-8"))["checkpoint"])
    candidates = sorted((cfg.root / "artifacts").glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise typer.BadParameter("no checkpoint found; pass --checkpoint or run `train` first")
    return candidates[-1]


if __name__ == "__main__":
    app()
