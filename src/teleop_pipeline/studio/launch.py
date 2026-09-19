"""Entry point for the double-clickable build.

Separate from the Typer CLI on purpose. Someone who opens the app expects a
window, not a usage message, so this parses a handful of flags itself and
otherwise just starts. It is also the PyInstaller entry point, where pulling in
Typer and Rich would add startup time and nothing else.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="teleop-studio",
        description="Run the teleoperation pipeline from a window instead of a terminal.",
    )
    parser.add_argument("--port", type=int, default=8765, help="Port on 127.0.0.1 (0 = any free).")
    parser.add_argument("--params", default=None, help="Path to params.yaml.")
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser.")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Check that this build can actually do its job, then exit.",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    from .server import serve

    try:
        serve(port=args.port, params=args.params, open_browser=not args.no_open)
    except SystemExit as exc:
        # A bundled app has no terminal to print a traceback into, so the
        # message is what the user gets. Keep it on stderr and exit non-zero.
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def self_test() -> int:
    """Check the things a bundle can silently lose, and say which one broke.

    A PyInstaller build can drop a data file or a lazily imported package and
    still start, failing only when someone presses the button that needs it.
    This runs in CI so that failure happens on the build machine instead.
    """
    from .pipeline import RUNNERS, STAGES
    from .server import INDEX

    failures: list[str] = []

    if not INDEX.is_file():
        failures.append(f"the app page is missing from the build ({INDEX})")

    missing = [s.key for s in STAGES if s.key not in RUNNERS]
    if missing:
        failures.append(f"stages with no implementation: {', '.join(missing)}")

    try:
        import torch

        print(f"torch {torch.__version__}")
    except Exception as exc:  # noqa: BLE001 - any failure here is the same failure
        failures.append(f"torch is not usable in this build: {type(exc).__name__}: {exc}")

    for problem in failures:
        print(f"FAIL  {problem}", file=sys.stderr)
    if failures:
        return 1
    print(f"ok: page present, {len(STAGES)} stages wired, torch importable")
    return 0
