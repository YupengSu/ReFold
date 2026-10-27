"""Terminal-Bench 1.0 and 2.1: state-based, scored inside the run.

The runner (runners/terminalbench.py) runs the task's own tests in the container once the
agent stops and writes results.json, so score() only reads it. Task sets are local
checkouts: set TBENCH1_TASKS / TBENCH21_TASKS.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from refold.benchmarks import register_benchmark
from refold.benchmarks.base import CONFIGS, ScoreReport, pick


class _TerminalBench:
    runner_module = "refold.benchmarks.runners.terminalbench"
    config = CONFIGS / "terminalbench.yaml"
    flavor = tasks_env = sets_file = ""

    def tasks_dir(self) -> Path:
        p = os.environ.get(self.tasks_env, "")
        if not p or not Path(p).is_dir():
            raise FileNotFoundError(f"{self.name}: set {self.tasks_env} to the benchmark's task directory "
                                    f"(got {p!r})")
        return Path(p)

    def data_flags(self) -> list[str]:
        return ["--flavor", self.flavor, "--tasks-dir", str(self.tasks_dir())]

    def provenance(self) -> dict:
        """The task directory, and the commit it is checked out at (a task set can change)."""
        d = self.tasks_dir()
        r = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True)
        return {"tasks_dir": str(d), "git_commit": r.stdout.strip() if r.returncode == 0 else None}

    def load(self, subset: str | None = None, limit: int | None = None) -> list[str]:
        names = sorted(d.name for d in self.tasks_dir().iterdir() if d.is_dir())
        return pick(names, self.sets_file, subset, limit)

    def score(self, out_dir: Path) -> ScoreReport:
        rf = out_dir / "results.json"
        if not rf.is_file():
            raise FileNotFoundError(f"{rf} not found: Terminal-Bench is scored in-run, so the cell did not run")
        d = json.loads(rf.read_text())
        ok = tuple(k for k, v in d.items() if v.get("resolved"))
        return ScoreReport(submitted=len(d), resolved=len(ok), resolved_ids=ok, report_path=rf)


@register_benchmark("tbench1")
class TerminalBench1(_TerminalBench):
    flavor, tasks_env, sets_file = "tb1", "TBENCH1_TASKS", "sets_tb1.json"  # Terminal-Bench's original-tasks/


@register_benchmark("tbench21")
class TerminalBench21(_TerminalBench):
    flavor, tasks_env, sets_file = "tb21", "TBENCH21_TASKS", "sets_tb21.json"  # Terminal-Bench 2.1's tasks/
