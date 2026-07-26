#!/usr/bin/env python
"""Standalone entry point for the synthetic session generator.

Equivalent to `erl-teleop synth`; kept as a script so the generator can be run
against an arbitrary params file without installing the package.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from erl_teleop.config import load_config  # noqa: E402
from erl_teleop.synthetic import generate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--params", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.params)
    out = Path(args.out) if args.out else cfg.resolve("ingest.raw_dir")
    n = generate(cfg, out, n_sessions=args.sessions, seed=args.seed)
    print(f"wrote {n} episode(s) across {args.sessions} session(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
