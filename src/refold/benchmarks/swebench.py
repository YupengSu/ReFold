"""The patch-based benchmarks: SWE-bench Verified, SWE-bench Pro, Multi-SWE-Bench.

All three run through runners/swebench.py (mini-swe-agent's own SWE-bench loop, with
per-benchmark image, working directory and problem statement) and are scored by their
official harness from the predictions the run wrote.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from refold.benchmarks import register_benchmark
from refold.benchmarks.base import CONFIGS, HF_DATASETS, ScoreReport, load_hf, pick


def _dataset_ids(flavor: str) -> list[str]:
    return [r["instance_id"] for r in load_hf(flavor)]


@register_benchmark("swebench")
class SweBenchVerified:
    runner_module = "refold.benchmarks.runners.swebench"
    config = CONFIGS / "swebench.yaml"
    flavor, sets_file = "verified", "sets.json"

    def data_flags(self) -> list[str]:
        return ["--flavor", self.flavor]

    def provenance(self) -> dict:
        name, split, revision = HF_DATASETS[self.flavor]
        return {"dataset": name, "split": split, "revision": revision}

    def load(self, subset: str | None = None, limit: int | None = None) -> list[str]:
        """`all`: the 50 tasks; `a100`: a seeded 100-task superset (for w = 64)."""
        return pick(_dataset_ids(self.flavor), self.sets_file, subset, limit)

    @staticmethod
    def run_id(out_dir: Path) -> str:
        """Derived from the predictions, not the path: the harness caches per-instance
        results under its run_id, so a path-derived id makes a re-run after a fix report
        the old run's scores."""
        # strip after the cut: a leading "-" makes the harness read the id as a flag
        slug = re.sub(r"[^A-Za-z0-9]+", "-", str(out_dir))[-60:].strip("-")
        preds = out_dir / "preds.json"
        if preds.is_file():
            slug += "-" + hashlib.sha256(preds.read_bytes()).hexdigest()[:8]
        return slug

    def score(self, out_dir: Path) -> ScoreReport:
        preds = out_dir / "preds.json"
        if not preds.is_file():
            raise FileNotFoundError(f"{preds} not found -- has the cell been run?")
        rid = self.run_id(out_dir)
        report_dir = out_dir / "score"
        report_dir.mkdir(parents=True, exist_ok=True)
        dataset = report_dir / "dataset.jsonl"  # the pinned split: the harness would load the latest
        with dataset.open("w") as f:
            for r in load_hf(self.flavor):
                f.write(json.dumps(dict(r)) + "\n")
        cmd = [sys.executable, "-m", "swebench.harness.run_evaluation",
               "--dataset_name", str(dataset.resolve()), "--split", "test",
               "--predictions_path", str(preds.resolve()), "--run_id", rid,
               "--max_workers", "8", "--report_dir", str(report_dir.resolve())]
        if _harness_accepts("--cache_level"):  # 4.x only; 5.x rejects the flag
            cmd += ["--cache_level", "env"]
        # cwd=report_dir: 4.1.0 writes <model>.<run_id>.json into the CURRENT directory,
        # whatever --report_dir says.
        if subprocess.run(cmd, check=False, cwd=report_dir).returncode != 0:
            raise RuntimeError("the SWE-bench harness failed; its output is above. "
                               "It ships as an extra: pip install -e '.[swebench]'")
        hits = [p for p in report_dir.glob("*.json") if rid in p.name] or list(report_dir.glob("*.json"))
        if not hits:
            raise FileNotFoundError(f"no harness report under {report_dir}")
        p = max(hits, key=lambda x: x.stat().st_mtime)
        d = json.loads(p.read_text())
        return ScoreReport(submitted=d["submitted_instances"], resolved=d["resolved_instances"],
                           empty_patch=d.get("empty_patch_instances", 0),
                           resolved_ids=tuple(d.get("resolved_ids") or ()), report_path=p)


@lru_cache(maxsize=None)
def _harness_accepts(flag: str) -> bool:
    try:
        out = subprocess.run([sys.executable, "-m", "swebench.harness.run_evaluation", "--help"],
                             capture_output=True, text=True, timeout=120)
    except Exception:
        return False
    return flag in (out.stdout or "") + (out.stderr or "")


def _trajectories(out_dir: Path):
    for t in sorted(out_dir.glob("*/*.traj.json")):
        yield json.loads(t.read_text())


@register_benchmark("swebenchpro")
class SweBenchPro:
    """ScaleAI's SWE-bench Pro. Repos live under /app; images are jefzda/sweap-images."""

    runner_module = "refold.benchmarks.runners.swebench"
    config = CONFIGS / "swebenchpro.yaml"
    flavor, sets_file = "pro", "sets_pro.json"

    def data_flags(self) -> list[str]:
        return ["--flavor", self.flavor]

    def provenance(self) -> dict:
        name, split, revision = HF_DATASETS[self.flavor]
        return {"dataset": name, "split": split, "revision": revision}

    def load(self, subset: str | None = None, limit: int | None = None) -> list[str]:
        return pick(_dataset_ids(self.flavor), self.sets_file, subset, limit)

    def score(self, out_dir: Path) -> ScoreReport:
        """The official harness is a repository, not a package:
            git clone https://github.com/scaleapi/SWE-bench_Pro-os.git
            pip install -r SWE-bench_Pro-os/requirements.txt
            export SWEBENCH_PRO_HARNESS=$PWD/SWE-bench_Pro-os
        """
        harness = os.environ.get("SWEBENCH_PRO_HARNESS", "")
        if not harness or not Path(harness).is_dir():
            raise FileNotFoundError(f"set SWEBENCH_PRO_HARNESS to a checkout of the Pro harness "
                                    f"(see {__name__}.SweBenchPro.score)")
        outd = (out_dir / "score").resolve()
        outd.mkdir(parents=True, exist_ok=True)
        patches = [{"instance_id": d["instance_id"], "patch": d["info"].get("submission") or "", "prefix": "run"}
                   for d in _trajectories(out_dir)]
        (outd / "patches.json").write_text(json.dumps(patches))
        with (outd / "dataset.jsonl").open("w") as f:
            for r in load_hf(self.flavor):
                f.write(json.dumps(dict(r)) + "\n")
        cmd = [sys.executable, "swe_bench_pro_eval.py", "--patch_path", str(outd / "patches.json"),
               "--raw_sample_path", str(outd / "dataset.jsonl"), "--use_local_docker",
               "--dockerhub_username", "jefzda", "--output_dir", str(outd / "eval"),
               "--scripts_dir", "./run_scripts", "--num_workers", "8"]
        if subprocess.run(cmd, check=False, cwd=harness).returncode != 0:
            raise RuntimeError("the SWE-bench Pro harness failed; its output is above")
        rf = outd / "eval" / "eval_results.json"
        res = json.loads(rf.read_text())
        return ScoreReport(submitted=len(patches), resolved=sum(1 for v in res.values() if v),
                           resolved_ids=tuple(k for k, v in res.items() if v), report_path=rf)


@register_benchmark("msb")
class MultiSweBench:
    """Multi-SWE-Bench flash (ByteDance). Multi-language; repos live under /home/<repo>."""

    runner_module = "refold.benchmarks.runners.swebench"
    config = CONFIGS / "multiswebench.yaml"
    flavor, sets_file = "msb", "sets_msb.json"

    def data_flags(self) -> list[str]:
        return ["--flavor", self.flavor]

    def provenance(self) -> dict:
        name, split, revision = HF_DATASETS[self.flavor]
        return {"dataset": name, "split": split, "revision": revision}

    def load(self, subset: str | None = None, limit: int | None = None) -> list[str]:
        return pick(_dataset_ids(self.flavor), self.sets_file, subset, limit)

    def score(self, out_dir: Path) -> ScoreReport:
        """Needs `pip install multi-swe-bench`."""
        preds = json.loads((out_dir / "preds.json").read_text())
        rows = {r["instance_id"]: r for r in load_hf(self.flavor)}
        outd = (out_dir / "score").resolve()
        for sub in ("workdir", "dataset", "repos", "logs"):
            (outd / sub).mkdir(parents=True, exist_ok=True)  # the harness requires them to exist
        pf, df = outd / "preds_msb.jsonl", outd / "dataset_msb.jsonl"
        with pf.open("w") as fp, df.open("w") as fd:
            for iid, p in preds.items():
                if (r := rows.get(iid)) is None:
                    continue
                fp.write(json.dumps({"org": r["org"], "repo": r["repo"], "number": r["number"],
                                     "fix_patch": p.get("model_patch") or ""}) + "\n")
                fd.write(json.dumps(dict(r)) + "\n")
        cfg = outd / "config.json"
        cfg.write_text(json.dumps({
            "mode": "evaluation", "patch_files": [str(pf)], "dataset_files": [str(df)],
            "workdir": str(outd / "workdir"), "output_dir": str(outd / "dataset"),
            "repo_dir": str(outd / "repos"), "log_dir": str(outd / "logs"), "log_level": "INFO",
            "specifics": [], "skips": [], "global_env": [], "clear_env": True, "force_build": False,
            "need_clone": False, "stop_on_error": False, "max_workers": 8,
            "max_workers_build_image": 4, "max_workers_run_instance": 8,
        }, indent=2))
        if subprocess.run([sys.executable, "-m", "multi_swe_bench.harness.run_evaluation",
                           "--config", str(cfg)], check=False).returncode != 0:
            raise RuntimeError("the Multi-SWE-Bench harness failed; its output is above. "
                               "It ships separately: pip install multi-swe-bench")
        hits = sorted(outd.rglob("final_report.json"))
        if not hits:
            raise FileNotFoundError(f"no final_report.json under {outd}")
        d = json.loads(hits[-1].read_text())
        # the harness names an instance org/repo:pr-N; the run names it by its instance_id
        run_id = {f"{r['org']}/{r['repo']}:pr-{r['number']}": iid for iid, r in rows.items()}
        return ScoreReport(submitted=d.get("total_instances", d.get("total", 0)),
                           resolved=d.get("resolved_instances", d.get("resolved", 0)),
                           resolved_ids=tuple(run_id.get(i, i) for i in d.get("resolved_ids") or ()),
                           report_path=hits[-1])
