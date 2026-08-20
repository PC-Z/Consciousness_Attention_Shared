from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .pipeline import build_alignment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Attention multimodal alignment")
    parser.add_argument("command", choices=("dry-run", "build"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--session", action="append", dest="sessions")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = build_alignment(
        config=args.config,
        dry_run=args.command == "dry-run",
        session_ids=args.sessions,
    )
    summary = pd.DataFrame([result.summary() for result in results.values()])
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
