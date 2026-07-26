"""Human-readable data-quality report.

Written as Markdown so it renders in a PR, a GitHub issue, or a lab wiki without
a viewer. The audience is the person who has to decide whether to collect more
data, retrain an operator, or fix a rig — so the report leads with per-operator
and per-flag breakdowns rather than a single aggregate score.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .schema import QualityReport, ValidationReport


def _bar(fraction: float, width: int = 20) -> str:
    filled = int(round(fraction * width))
    return "█" * filled + "·" * (width - filled)


def render(
    quality: list[QualityReport],
    validation: list[ValidationReport] | None = None,
    title: str = "Teleoperation data quality",
) -> str:
    lines: list[str] = [
        f"# {title}",
        "",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
    ]

    if not quality:
        lines += ["No scored episodes found.", ""]
        return "\n".join(lines)

    scores = np.asarray([r.score for r in quality])
    tiers: dict[str, int] = defaultdict(int)
    for r in quality:
        tiers[r.tier] += 1
    total = len(quality)

    lines += [
        "## Corpus",
        "",
        f"- **{total}** episodes scored",
        f"- mean score **{scores.mean():.1f}**, median **{np.median(scores):.1f}**, "
        f"p10 **{np.percentile(scores, 10):.1f}**",
        "",
        "| Tier | Episodes | Share | |",
        "| --- | ---: | ---: | --- |",
    ]
    for tier in ("gold", "silver", "reject"):
        n = tiers.get(tier, 0)
        lines.append(f"| {tier} | {n} | {n / total:.0%} | `{_bar(n / total)}` |")
    lines.append("")

    # -- flags -------------------------------------------------------------
    flag_counts: dict[str, int] = defaultdict(int)
    for r in quality:
        for f in r.flags:
            flag_counts[f] += 1

    if flag_counts:
        lines += [
            "## Flags raised",
            "",
            "Each flag is an episode-level threshold breach, independent of the composite score.",
            "",
            "| Flag | Episodes | Share |",
            "| --- | ---: | ---: |",
        ]
        for flag, n in sorted(flag_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{flag}` | {n} | {n / total:.0%} |")
        lines.append("")

    # -- operators ---------------------------------------------------------
    per_operator: dict[str, list[QualityReport]] = defaultdict(list)
    for r in quality:
        per_operator[r.operator_id].append(r)

    lines += [
        "## By operator",
        "",
        "A consistently low operator is a training conversation, not a data problem.",
        "",
        "| Operator | Episodes | Mean score | Gold | Reject | Most common flag |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for op, rows in sorted(per_operator.items(), key=lambda kv: np.mean([r.score for r in kv[1]])):
        op_flags: dict[str, int] = defaultdict(int)
        for r in rows:
            for f in r.flags:
                op_flags[f] += 1
        top = max(op_flags.items(), key=lambda kv: kv[1])[0] if op_flags else "—"
        lines.append(
            f"| {op} | {len(rows)} | {np.mean([r.score for r in rows]):.1f} | "
            f"{sum(r.tier == 'gold' for r in rows)} | "
            f"{sum(r.tier == 'reject' for r in rows)} | `{top}` |"
        )
    lines.append("")

    # -- tasks -------------------------------------------------------------
    per_task: dict[str, list[QualityReport]] = defaultdict(list)
    for r in quality:
        per_task[r.task_id].append(r)
    lines += [
        "## By task",
        "",
        "| Task | Episodes | Mean score | Reject |",
        "| --- | ---: | ---: | ---: |",
    ]
    for task, rows in sorted(per_task.items()):
        lines.append(
            f"| {task} | {len(rows)} | {np.mean([r.score for r in rows]):.1f} | "
            f"{sum(r.tier == 'reject' for r in rows)} |"
        )
    lines.append("")

    # -- metric distributions ---------------------------------------------
    metric_names = [m.name for m in quality[0].metrics]
    lines += [
        "## Metric distributions",
        "",
        "| Metric | Median | p90 | Max | Mean penalty |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in metric_names:
        values = np.asarray([r.value(name) for r in quality])
        penalties = np.asarray(
            [next(m.penalty for m in r.metrics if m.name == name) for r in quality]
        )
        lines.append(
            f"| `{name}` | {np.median(values):.4g} | {np.percentile(values, 90):.4g} | "
            f"{values.max():.4g} | {penalties.mean():.2f} |"
        )
    lines.append("")

    # -- worst offenders ---------------------------------------------------
    worst = sorted(quality, key=lambda r: r.score)[:10]
    lines += [
        "## Lowest-scoring episodes",
        "",
        "| Episode | Operator | Score | Tier | Flags |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for r in worst:
        flags = ", ".join(f"`{f}`" for f in r.flags) or "—"
        lines.append(f"| {r.episode_id} | {r.operator_id} | {r.score:.1f} | {r.tier} | {flags} |")
    lines.append("")

    # -- validation --------------------------------------------------------
    if validation:
        failed = [v for v in validation if not v.ok]
        lines += [
            "## Validation",
            "",
            f"- {len(validation)} episodes checked, **{len(failed)}** quarantined",
            "",
        ]
        if failed:
            codes: dict[str, int] = defaultdict(int)
            for v in failed:
                for issue in v.errors:
                    codes[issue.code] += 1
            lines += ["| Error | Episodes |", "| --- | ---: |"]
            for code, n in sorted(codes.items(), key=lambda kv: -kv[1]):
                lines.append(f"| `{code}` | {n} |")
            lines.append("")

    return "\n".join(lines)


def write(path: Path, content: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path
