"""teleop-pipeline — versioned, validated, reproducible teleoperation data pipeline."""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
