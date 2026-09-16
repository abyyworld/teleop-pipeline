"""Run lineage: the record that makes a result reproducible a year later.

Every training run writes a `lineage.json` that pins the four things you need to
rebuild it and that are otherwise the four things nobody writes down:

* the **git commit** of the code, plus whether the tree was dirty
* the **dataset hash** from the manifest, and the DVC lock hash of the data
* the **resolved params** actually used, not the ones in the file today
* the **checkpoint hash**, so a model file found later can be traced back

This file is also the contract with the downstream evaluation repo
(``vla-evals``), which registers checkpoints by reading it.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import file_hash


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


@dataclass
class GitState:
    commit: str | None = None
    branch: str | None = None
    dirty: bool = False
    remote: str | None = None

    @classmethod
    def capture(cls, root: Path) -> GitState:
        commit = _git(["rev-parse", "HEAD"], root)
        if commit is None:
            return cls()
        status = _git(["status", "--porcelain"], root)
        return cls(
            commit=commit,
            branch=_git(["rev-parse", "--abbrev-ref", "HEAD"], root),
            # A dirty tree means the commit does not describe what actually ran.
            # Recording the flag is the difference between "reproducible" and
            # "probably reproducible".
            dirty=bool(status),
            remote=_git(["config", "--get", "remote.origin.url"], root),
        )


def dvc_lock_hash(root: Path) -> str | None:
    """Hash of dvc.lock — pins the exact data revision the stages consumed."""
    lock = Path(root) / "dvc.lock"
    return file_hash(lock)[:16] if lock.exists() else None


@dataclass
class Lineage:
    run_id: str
    run_name: str
    created_at: str
    git: dict[str, Any]
    dataset_hash: str
    dvc_lock_hash: str | None
    params: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    tracking: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        root: Path,
        run_id: str,
        run_name: str,
        dataset_hash: str,
        params: dict[str, Any],
    ) -> Lineage:
        return cls(
            run_id=run_id,
            run_name=run_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            git=asdict(GitState.capture(root)),
            dataset_hash=dataset_hash,
            dvc_lock_hash=dvc_lock_hash(root),
            params=params,
            environment={
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
        )

    def attach_checkpoint(self, path: Path, extra: dict | None = None) -> None:
        path = Path(path)
        self.checkpoint = {
            "path": str(path),
            "filename": path.name,
            "sha256": file_hash(path),
            "bytes": path.stat().st_size,
            **(extra or {}),
        }

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
        return path

    @staticmethod
    def read(path: Path) -> dict:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def reproduction_command(self) -> str:
        commit = (self.git or {}).get("commit") or "<commit>"
        return f"git checkout {commit[:12]} && dvc repro  # dataset {self.dataset_hash}"
