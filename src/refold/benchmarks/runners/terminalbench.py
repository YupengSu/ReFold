"""Batch runner for Terminal-Bench 1.0 and 2.1: state-based, scored in the container.

    python -m refold.benchmarks.runners.terminalbench --flavor tb1|tb21 --tasks-dir DIR \\
        -o OUT -w N --model M --agent-class dotted.Class -c terminalbench.yaml [-c ...]

The agent runs on the host and uses the task container as its shell. When it stops, the
task's own tests are copied in and run; the verdict goes to results.json (and into the
trajectory), so there is no separate scoring pass.

    tb1   task.yaml + Dockerfile, built locally; working dir = the Dockerfile's last WORKDIR;
          resolved = pytest reports >=1 passed and nothing failed
    tb21  task.toml + instruction.md + a prebuilt image; working dir = the image's WORKDIR;
          resolved = /logs/verifier/reward.txt (pytest summary as fallback)
"""

from __future__ import annotations

import fcntl
import os
import json
import re
import shlex
import subprocess
import threading
import tomllib
import traceback
from pathlib import Path

import typer
import yaml
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models import get_model
from minisweagent.utils.log import add_file_handler, logger

from refold.benchmarks.runners.common import merge_config, run_batch, stop_when_container_gone
from refold.live import LIVE_ENV, with_live_log
from refold.registry import import_class

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
_RESULTS_LOCK = threading.Lock()
_BUILD_LOCK = threading.Lock()


# ----------------------------------------------------------------------------- tasks
def dockerfile_workdir(dockerfile_text: str, default: str = "/app") -> str:
    matches = re.findall(r"^\s*WORKDIR\s+(\S+)", dockerfile_text, re.MULTILINE)
    return matches[-1] if matches else default


def load_tasks_tb1(tasks_dir: Path) -> list[dict]:
    out = []
    for d in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
        if not (d / "task.yaml").is_file():
            continue
        y = yaml.safe_load((d / "task.yaml").read_text())
        instruction = (y.get("instruction") or "").strip()
        if not instruction:
            continue
        dockerfile = (d / "Dockerfile").read_text() if (d / "Dockerfile").is_file() else ""
        out.append({"instance_id": d.name, "problem_statement": instruction, "task_dir": str(d),
                    "image_name": f"tbench-task-{d.name}".lower(), "workdir": dockerfile_workdir(dockerfile),
                    "max_test_timeout_sec": float(y.get("max_test_timeout_sec") or 300.0)})
    return out


def load_tasks_tb21(tasks_dir: Path) -> list[dict]:
    out = []
    for d in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
        if not ((d / "task.toml").is_file() and (d / "instruction.md").is_file()):
            continue
        meta = tomllib.loads((d / "task.toml").read_text())
        image = (meta.get("environment") or {}).get("docker_image")
        if not image:
            continue
        out.append({"instance_id": d.name, "problem_statement": (d / "instruction.md").read_text().strip(),
                    "task_dir": str(d), "image_name": image,
                    "max_test_timeout_sec": float((meta.get("verifier") or {}).get("timeout_sec") or 900.0)})
    return out


# ----------------------------------------------------------------------------- images
def build_image(instance: dict) -> str:
    """docker build, idempotent. A file lock on the tag keeps two concurrent cells (say base
    and refold) from building the same image at once."""
    image = instance["image_name"]
    with _BUILD_LOCK, open(f"/tmp/tbench-build-{re.sub(r'[^a-zA-Z0-9_.-]', '_', image)}.lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        if subprocess.run(["docker", "image", "inspect", image], capture_output=True, check=False).returncode != 0:
            logger.info(f"Building image {image} from {instance['task_dir']}")
            r = subprocess.run(["docker", "build", "-t", image, "-f", f"{instance['task_dir']}/Dockerfile",
                                instance["task_dir"]], capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                raise RuntimeError(f"docker build failed for {image}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return instance["workdir"]


def pull_image(instance: dict) -> str:
    image = instance["image_name"]
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True, check=False).returncode != 0:
        logger.info(f"Pulling {image}")
        r = subprocess.run(["docker", "pull", image], capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f"docker pull failed for {image}:\n{r.stderr[-1500:]}")
    wd = subprocess.run(["docker", "inspect", image, "--format", "{{.Config.WorkingDir}}"],
                        capture_output=True, text=True).stdout.strip()
    return wd if wd and wd != "/" else "/app"  # test.sh refuses to run from /


# ----------------------------------------------------------------------------- scoring
def parse_pytest_resolved(output: str) -> bool:
    """True iff the last pytest summary line reports >=1 passed and no failure or error."""
    summ = re.findall(r"=+\s*(.*?(?:passed|failed|error|no tests ran).*?)\s*=+", output or "", re.IGNORECASE)
    if not summ or "no tests ran" in summ[-1].lower():
        return False
    last = summ[-1].lower()
    n = {k: int(m.group(1)) if (m := re.search(rf"(\d+)\s+{k}", last)) else 0 for k in ("passed", "failed", "error")}
    return n["passed"] > 0 and n["failed"] == 0 and n["error"] == 0


def score_tb1(env: DockerEnvironment, instance: dict) -> tuple[bool, str]:
    cid, workdir = env.container_id, instance["workdir"]
    subprocess.run(["docker", "exec", cid, "mkdir", "-p", "/tests", workdir], capture_output=True, check=False)
    subprocess.run(["docker", "cp", f"{instance['task_dir']}/tests/.", f"{cid}:/tests/"], capture_output=True, check=True)
    subprocess.run(["docker", "cp", f"{instance['task_dir']}/run-tests.sh", f"{cid}:/run-tests.sh"],
                   capture_output=True, check=True)
    r = subprocess.run(["docker", "exec", "-e", "TEST_DIR=/tests", cid, "bash", "-lc",
                        f"cd {shlex.quote(workdir)} && TEST_DIR=/tests bash /run-tests.sh"],
                       capture_output=True, text=True, errors="replace",
                       timeout=int(instance["max_test_timeout_sec"]) + 120)
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    return parse_pytest_resolved(out), out


def score_tb21(env: DockerEnvironment, instance: dict) -> tuple[bool, str]:
    cid = env.container_id
    subprocess.run(["docker", "exec", cid, "mkdir", "-p", "/tests", "/logs/verifier"], capture_output=True, check=False)
    subprocess.run(["docker", "cp", f"{instance['task_dir']}/tests/.", f"{cid}:/tests/"], capture_output=True, check=True)
    r = subprocess.run(["docker", "exec", cid, "bash", "-lc", f"cd {shlex.quote(env.config.cwd)} && bash /tests/test.sh"],
                       capture_output=True, text=True, errors="replace",
                       timeout=int(instance["max_test_timeout_sec"]) + 180)
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    reward = subprocess.run(["docker", "exec", cid, "cat", "/logs/verifier/reward.txt"], capture_output=True, text=True)
    if reward.returncode == 0 and reward.stdout.strip() in ("0", "1"):
        return reward.stdout.strip() == "1", out
    return parse_pytest_resolved(out), out


FLAVORS = {  # flavor -> (load tasks, prepare image -> workdir, docker run args, score)
    "tb1": (load_tasks_tb1, build_image, None, score_tb1),
    "tb21": (load_tasks_tb21, pull_image, ["--rm", "--entrypoint", ""], score_tb21),
}


# ----------------------------------------------------------------------------- one task
def _update_results(path: Path, instance_id: str, record: dict | None) -> None:
    with _RESULTS_LOCK:
        data = json.loads(path.read_text()) if path.exists() else {}
        if record is None:
            data.pop(instance_id, None)
        else:
            data[instance_id] = record
        path.write_text(json.dumps(data, indent=2))


def process_instance(instance: dict, output_dir: Path, config: dict, agent_class: type, flavor: str, progress) -> None:
    _, prepare, run_args, score = FLAVORS[flavor]
    iid = instance["instance_id"]
    results = output_dir / "results.json"
    _update_results(results, iid, None)
    model = get_model(config=config.get("model", {}))
    progress.on_instance_start(iid)
    progress.update_instance_status(iid, "Preparing image")
    env = agent = exit_status = None
    resolved, test_output, extra = False, "", {}
    try:
        workdir = prepare(instance)
        env_cfg = {k: v for k, v in (config.get("environment") or {}).items() if k != "environment_class"}
        env_cfg.update(image=instance["image_name"], cwd=workdir)
        if run_args:
            env_cfg["run_args"] = run_args  # some prebuilt images set an ENTRYPOINT
        env = DockerEnvironment(**env_cfg)
        agent = agent_class(model, env, **config.get("agent", {}))
        agent.instance_id = iid  # names the task's live log (refold/live.py)
        progress.update_instance_status(iid, "Agent running")
        exit_status = agent.run(instance["problem_statement"]).get("exit_status")
        progress.update_instance_status(iid, "Scoring")
        resolved, test_output = score(env, instance)
    except Exception as e:
        logger.error(f"Error processing instance {iid}: {e}", exc_info=True)
        exit_status = exit_status or type(e).__name__
        extra = {"traceback": traceback.format_exc(), "exception_str": str(e)}
    finally:
        if agent is not None:
            agent.save(output_dir / iid / f"{iid}.traj.json",
                       {"info": {"exit_status": exit_status, "resolved": resolved,
                                 "test_output": test_output[-4000:], **extra}, "instance_id": iid})
        _update_results(results, iid, {"instance_id": iid, "exit_status": exit_status, "resolved": resolved})
        if env is not None:
            env.cleanup()
        progress.on_instance_end(iid, exit_status)


@app.command()
def main(
    flavor: str = typer.Option(..., "--flavor", help="tb1 | tb21"),
    tasks_dir: str = typer.Option(..., "--tasks-dir"),
    filter_spec: str = typer.Option("", "--filter", help="task-id regex"),
    output: str = typer.Option(..., "-o", "--output"),
    workers: int = typer.Option(1, "-w", "--workers"),
    model: str | None = typer.Option(None, "-m", "--model"),
    model_class: str | None = typer.Option(None, "--model-class"),
    agent_class: str = typer.Option(..., "--agent-class"),
    config_spec: list[str] = typer.Option(..., "-c", "--config"),
    redo_existing: bool = typer.Option(False, "--redo-existing"),
) -> None:
    load = FLAVORS[flavor][0]
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    add_file_handler(output_path / "minisweagent.log")
    instances = [i for i in load(Path(tasks_dir)) if re.search(filter_spec, i["instance_id"])]
    if not redo_existing and (output_path / "results.json").exists():
        done = set(json.loads((output_path / "results.json").read_text()))
        instances = [i for i in instances if i["instance_id"] not in done]
    logger.info(f"Running on {len(instances)} tasks...")
    config = merge_config(config_spec, model, model_class)
    agent_cls = stop_when_container_gone(import_class(agent_class))
    if live_dir := os.environ.get(LIVE_ENV):  # refold run --live
        agent_cls = with_live_log(agent_cls, live_dir)
    run_batch(instances, lambda inst, pm: process_instance(inst, output_path, config, agent_cls, flavor, pm),
              output_path, workers)


if __name__ == "__main__":
    app()
