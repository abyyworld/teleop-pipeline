"""Typed access to params.yaml.

Stages never read YAML directly and never hardcode a constant that belongs in
params.yaml — otherwise DVC cannot tell which stages a config change
invalidates, and `dvc repro` silently reuses stale outputs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_PARAMS = "params.yaml"


def project_root(start: Path | None = None) -> Path:
    """Walk up from `start` until we find params.yaml; fall back to cwd."""
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / DEFAULT_PARAMS).exists():
            return candidate
    return Path.cwd()


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    root: Path
    path: Path

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        if path is None:
            root = project_root()
            path = root / DEFAULT_PARAMS
        path = Path(path).resolve()
        root = path.parent
        with path.open() as fh:
            raw = yaml.safe_load(fh) or {}
        return cls(raw=raw, root=root, path=path)

    # -- section accessors -------------------------------------------------

    def __getitem__(self, section: str) -> dict[str, Any]:
        return self.raw[section]

    def get(self, dotted: str, default: Any = None) -> Any:
        """`cfg.get("train.lr")` -> 0.0003."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def resolve(self, dotted_or_path: str) -> Path:
        """Resolve a params path (or a literal path) against the project root."""
        value = self.get(dotted_or_path)
        rel = value if isinstance(value, str) else dotted_or_path
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    # -- frequently used derived values ------------------------------------

    @property
    def n_joints(self) -> int:
        return int(self.raw["robot"]["n_joints"])

    @property
    def control_hz(self) -> float:
        return float(self.raw["robot"]["control_hz"])

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz

    @cached_property
    def joint_lower(self) -> np.ndarray:
        return np.asarray(self.raw["robot"]["joint_lower"], dtype=np.float64)

    @cached_property
    def joint_upper(self) -> np.ndarray:
        return np.asarray(self.raw["robot"]["joint_upper"], dtype=np.float64)

    @property
    def action_limit(self) -> float:
        return float(self.raw["robot"]["action_limit"])

    @property
    def tracking_uri(self) -> str:
        # An explicit env var always wins, so a lab MLflow server can be used
        # without editing (and accidentally committing) params.yaml.
        return os.environ.get("MLFLOW_TRACKING_URI") or str(self.get("tracking.uri"))


def load_config(path: str | Path | None = None) -> Config:
    return Config.load(path)
