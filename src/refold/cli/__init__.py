"""refold's command line.

    refold run         run one cell (strategy x model x benchmark x budget x concurrency)
    refold score       score a cell with the benchmark's official evaluator
    refold report      the run metrics for one or more cells
    refold strategies  list the registered strategies
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from refold import dashboard
    from refold.cli import report
    from refold.cli import run as run_cmd
    from refold.registry import available, get_strategy

    parser = argparse.ArgumentParser(prog="refold", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_cmd.add_args(sub.add_parser("run", help="run one cell"))
    score = sub.add_parser("score", help="score a cell with the benchmark's official evaluator")
    score.add_argument("cell", type=Path)
    score.add_argument("--benchmark", default=None, help="default: the one the cell's plan.json names")
    report.add_args(sub.add_parser("report", help="the run metrics for one or more cells"))
    sub.add_parser("strategies", help="list the registered strategies")
    dashboard.add_args(sub.add_parser("dashboard", help="a live page for one run (refold run --live / --dashboard)"))

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    try:
        if args.cmd == "run":
            return run_cmd.main(args)
        if args.cmd == "score":
            return _score(args.cell, args.benchmark)
        if args.cmd == "report":
            return report.main(args)
        if args.cmd == "dashboard":
            return dashboard.main(args)
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"refold {args.cmd}: {e.args[0] if e.args else e}", file=sys.stderr)
        return 2
    for n in available():
        s = get_strategy(n)
        notes = ["--compact: compact + ReFold"] if s.compact_agent else []
        print(f"{n:8s} {s.summary}" + (f"   [{'; '.join(notes)}]" if notes else ""))
    return 0


def _score(cell: Path, benchmark: str | None) -> int:
    """Run the official evaluator and write <cell>/resolved.json, which `refold report` reads."""
    from refold.benchmarks import get_benchmark

    plan = cell / "plan.json"
    name = benchmark or (json.loads(plan.read_text()).get("benchmark") if plan.is_file() else None) or "swebench"
    r = get_benchmark(name).score(cell)
    (cell / "resolved.json").write_text(json.dumps(
        {"benchmark": name, "submitted": r.submitted, "n_resolved": r.resolved, "resolved": sorted(r.resolved_ids),
         "report": str(r.report_path) if r.report_path else None}, indent=1))
    print(f"resolved {r.resolved} / {r.submitted}" + (f" ({r.empty_patch} empty patches)" if r.empty_patch else ""))
    if r.report_path:
        print(f"report: {r.report_path}")
    return 0
