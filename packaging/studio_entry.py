"""PyInstaller entry point for the desktop app.

A bundled app has no repository to run from, so two things are resolved here
that the source checkout gets for free:

* The page is loaded from the extraction directory PyInstaller unpacks into.
* The working directory is where the user launched the app, which is where its
  `params.yaml` and `data/` live. `sys._MEIPASS` must never become the project
  root, or the app would write its corpus into a temporary directory that is
  deleted on exit.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _bundled_page() -> Path | None:
    """The unpacked copy of index.html, when running from a bundle."""
    base = getattr(sys, "_MEIPASS", None)
    if not base:
        return None
    candidate = Path(base) / "teleop_pipeline" / "studio" / "index.html"
    return candidate if candidate.is_file() else None


def main() -> int:
    from teleop_pipeline.studio import server
    from teleop_pipeline.studio.launch import main as launch

    page = _bundled_page()
    if page is not None:
        server.INDEX = page
    return launch()


if __name__ == "__main__":
    raise SystemExit(main())
