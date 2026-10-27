"""Batch runner for the patch-based benchmarks: mini-swe-agent's own SWE-bench loop
(`process_instance`) with the agent class swapped in and per-benchmark instance setup.

    python -m refold.benchmarks.runners.swebench --flavor verified|pro|msb \\
        -o OUT -w N --model M --agent-class dotted.Class -c benchmark.yaml [-c ...]

    verified  princeton-nlp/SWE-Bench_Verified; images and /testbed as upstream
    pro       ScaleAI/SWE-bench_Pro; image jefzda/sweap-images:<tag>, repo at /app, the
              image's /bin/bash ENTRYPOINT neutralised so the keepalive runs
    msb       Multi-SWE-bench flash; problem statement from title + body + linked issues,
              image mswebench/<org>_m_<repo>:pr-<n>, repo at /home/<repo>
"""

from __future__ import annotations

import json
from pathlib import Path

import minisweagent.run.benchmarks.swebench as _sw
import typer
from minisweagent.utils.log import add_file_handler, logger

from refold.benchmarks.base import HF_DATASETS, load_hf
from refold.benchmarks.runners.common import merge_config, run_batch, with_progress
from refold.registry import import_class

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


def _pro(inst: dict) -> tuple[dict, dict]:
    inst = dict(inst, image_name=f"jefzda/sweap-images:{inst['dockerhub_tag']}")
    ps = inst.get("problem_statement") or ""
    if ps.startswith('"') and ps.endswith('"'):  # sometimes arrives JSON-quoted
        try:
            inst["problem_statement"] = json.loads(ps)
        except Exception:
            pass
    return inst, {"cwd": "/app", "run_args": ["--rm", "--entrypoint", ""]}


def _msb(inst: dict) -> tuple[dict, dict]:
    parts = [inst.get("title") or "", inst.get("body") or ""]
    for ri in inst.get("resolved_issues") or []:
        if isinstance(ri, dict) and (ri.get("title") or ri.get("body")):
            parts.append(f"\n--- linked issue #{ri.get('number', '')} ---\n{ri.get('title', '')}\n{ri.get('body', '') or ''}")
    inst = dict(inst, problem_statement="\n\n".join(p for p in parts if p).strip(),
                image_name=f"mswebench/{inst['org']}_m_{inst['repo']}:pr-{inst['number']}".lower())
    return inst, {"cwd": f"/home/{inst['repo']}"}


FLAVORS = {  # flavor -> instance -> (instance, environment overrides); the data is base.HF_DATASETS
    "verified": lambda inst: (inst, {}),
    "pro": _pro,
    "msb": _msb,
}


@app.command()
def main(
    flavor: str = typer.Option(..., "--flavor", help="verified | pro | msb"),
    filter_spec: str = typer.Option("", "--filter", help="instance-id regex"),
    output: str = typer.Option(..., "-o", "--output"),
    workers: int = typer.Option(1, "-w", "--workers"),
    model: str | None = typer.Option(None, "-m", "--model"),
    model_class: str | None = typer.Option(None, "--model-class"),
    agent_class: str = typer.Option(..., "--agent-class"),
    config_spec: list[str] = typer.Option(..., "-c", "--config"),
    redo_existing: bool = typer.Option(False, "--redo-existing"),
) -> None:
    adapt = FLAVORS[flavor]
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    add_file_handler(output_path / "minisweagent.log")
    logger.info(f"Loading {HF_DATASETS[flavor]}...")
    adapted = [adapt(x) for x in load_hf(flavor)]
    env_of = {inst["instance_id"]: env for inst, env in adapted}
    instances = _sw.filter_instances([inst for inst, _ in adapted], filter_spec=filter_spec)
    if not redo_existing and (output_path / "preds.json").exists():
        done = set(json.loads((output_path / "preds.json").read_text()))
        instances = [i for i in instances if i["instance_id"] not in done]
    logger.info(f"Running on {len(instances)} instances...")

    config = merge_config(config_spec, model, model_class)
    _sw.ProgressTrackingAgent = with_progress(import_class(agent_class))
    base_env = _sw.get_sb_environment

    def get_environment(cfg: dict, instance: dict):
        # A copy per instance: upstream writes the image into config["environment"], and the
        # config is shared by every worker thread.
        cfg = {**cfg, "environment": {**(cfg.get("environment") or {}), **env_of[instance["instance_id"]]}}
        return base_env(cfg, instance)

    _sw.get_sb_environment = get_environment
    run_batch(instances, lambda inst, pm: _sw.process_instance(inst, output_path, config, pm), output_path, workers)


if __name__ == "__main__":
    app()
