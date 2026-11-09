"""Batch runner for Terminal-Bench 1.0 and 2.1: state-based, scored in the container.

    python -m refold.benchmarks.runners.terminalbench --flavor tb1|tb21 --tasks-dir DIR \\
        -o OUT -w N --model M --agent-class dotted.Class -c terminalbench.yaml [-c ...]

The agent runs on the host and uses the task container as its shell. When it stops, for any
reason, the task's own tests are copied in and run; the verdict goes to results.json (and
into the trajectory), so there is no separate scoring pass. The verdict follows the official
harness of each version:

    tb1   task.yaml + Dockerfile, built locally; working dir = the Dockerfile's last WORKDIR;
          resolved = every test in pytest's short test summary passed (tb run); tests/ is
          copied in when the task has one (some keep their tests in run-tests.sh)
    tb21  task.toml + instruction.md + a prebuilt image; working dir = the image's WORKDIR;
          resolved = the reward in /logs/verifier/reward.json or reward.txt is 1 (Harbor)

`resolved` is None, with a `verifier_error` reason, when the tests gave no verdict (no
summary or reward file, a test timeout, the container gone) or, on tb21, when a model API
error ended the agent's run. tb run counts such tasks as unresolved, Harbor leaves them out
of the mean; the resolved count is the same either way. Runs recorded before 2026-11 did not
score a task whose agent stopped on an error. As in Harbor, a tb21 task with no reward file
has no verdict even when pytest printed a summary, and a test timeout or a failed docker
exec is no verdict, not a failure.

The tb21 test scripts write reward 0 whenever their pytest command fails, also when it never
started because uv or curl could not be installed. Harbor counts that 0, and so does this
runner, but a task with reward 0 and no pytest short test summary in its output gets a
`verifier_warning`: its tests may not have run.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import shlex
import subprocess
import threading
import tomllib
import traceback
from pathlib import Path
from typing import NamedTuple

import typer
import yaml
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models import get_model
from minisweagent.utils.log import add_file_handler, logger

from refold.benchmarks.runners.common import merge_config, merge_env, run_batch, stop_when_container_gone
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
# The verdict rules of the official harnesses: Terminal-Bench's `tb run` for tb1
# (terminal_bench/parsers/pytest_parser.py, harness/harness.py) and Harbor for tb21
# (harbor/verifier/verifier.py, models/trial/error_policy.py). A task is resolved True or
# False by its tests, or None, with the reason, when the tests gave no verdict.
class Verdict(NamedTuple):
    resolved: bool | None
    output: str = ""
    reason: str | None = None  # why there is no verdict (resolved is None)
    warning: str | None = None  # why the verdict may not come from the tests (it is kept)


_SHORT_SUMMARY_RE = re.compile(r"=+\s*short test summary info\s*=+", re.IGNORECASE)
_PYTEST_PASSES = {"PASSED": True, "SKIPPED": True, "XFAIL": True, "FAILED": False, "XPASS": False, "ERROR": False}


def parse_pytest_tests(output: str) -> dict[str, bool] | None:
    """tb run's PytestParser: each test listed after pytest's "short test summary info" line,
    and whether it passed (PASSED, SKIPPED and XFAIL pass; FAILED, XPASS and ERROR fail).
    None when there is no such line, tb run's PARSE_ERROR."""
    parts = _SHORT_SUMMARY_RE.split(output or "", maxsplit=1)
    if len(parts) < 2:
        return None
    tests: dict[str, bool] = {}
    for line in parts[1].splitlines():
        if line.startswith("FAILED") and " - " in line:
            line = " - ".join(line.split(" - ")[:-1])  # drop the failure description
        bits = line.split(maxsplit=1)
        status = bits[0].strip().strip(":") if len(bits) == 2 else ""
        name = bits[1].strip().split("::", 1)[-1] if len(bits) == 2 else ""
        if status in _PYTEST_PASSES and name:
            tests[name] = _PYTEST_PASSES[status]
    return tests


def reward_verdict(reward_json: str | None, reward_txt: str | None) -> tuple[bool | None, str | None]:
    """Harbor's reading of the verifier's reward: reward.json if it exists, else reward.txt
    (None: the file does not exist). Resolved iff the reward is 1; no verdict when neither
    file exists or the one read is empty or not a finite number."""
    if reward_json is not None:
        if reward_json == "":
            return None, "empty reward.json"
        try:
            rewards = json.loads(reward_json)
        except ValueError:
            return None, "unparsable reward.json"
        values = rewards if isinstance(rewards, dict) else {"reward": rewards}
        if not values or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values.values()):
            return None, "unparsable reward.json"
        if "reward" not in values and len(values) > 1:
            return None, "several rewards in reward.json"
        return values.get("reward", next(iter(values.values()))) == 1, None
    if reward_txt is None:
        return None, "no reward file"
    if reward_txt == "":
        return None, "empty reward.txt"
    try:
        value = float(reward_txt)
    except ValueError:
        return None, "unparsable reward.txt"
    return (value == 1, None) if math.isfinite(value) else (None, "unparsable reward.txt")


def _text(out: str | bytes | None) -> str:
    return out.decode(errors="replace") if isinstance(out, bytes) else out or ""


def _run_tests(argv: list[str], timeout: int) -> tuple[str, bool]:
    """(stderr + stdout, whether the tests finished within `timeout`). stdout comes last, so
    the tail kept in the trajectory ends with pytest's summary, not with the log of the
    installs (uv, apt, pip) that the test scripts write to stderr."""
    try:
        r = subprocess.run(argv, capture_output=True, text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as e:  # its output is bytes whatever `text` says
        return _text(e.stderr) + "\n" + _text(e.stdout), False
    return (r.stderr or "") + "\n" + (r.stdout or ""), True


def score_tb1(env: DockerEnvironment, instance: dict) -> Verdict:
    cid, workdir, task_dir = env.container_id, instance["workdir"], instance["task_dir"]
    subprocess.run(["docker", "exec", cid, "mkdir", "-p", "/tests", workdir], capture_output=True, check=False)
    if Path(task_dir, "tests").is_dir():  # as in tb run: some tasks keep their tests in run-tests.sh
        subprocess.run(["docker", "cp", f"{task_dir}/tests/.", f"{cid}:/tests/"], capture_output=True, check=True)
    subprocess.run(["docker", "cp", f"{task_dir}/run-tests.sh", f"{cid}:/run-tests.sh"],
                   capture_output=True, check=True)
    out, finished = _run_tests(["docker", "exec", "-e", "TEST_DIR=/tests", cid, "bash", "-lc",
                                f"cd {shlex.quote(workdir)} && TEST_DIR=/tests bash /run-tests.sh"],
                               int(instance["max_test_timeout_sec"]) + 120)
    if not finished:
        return Verdict(None, out, "test timeout")
    tests = parse_pytest_tests(out)
    if tests is None:
        return Verdict(None, out, "no pytest short test summary")
    return Verdict(all(tests.values()), out)  # all([]) is True in tb run too


def score_tb21(env: DockerEnvironment, instance: dict) -> Verdict:
    cid = env.container_id
    subprocess.run(["docker", "exec", cid, "mkdir", "-p", "/tests", "/logs/verifier"], capture_output=True, check=False)
    # Harbor discards rewards left from before the tests run (the agent may have written one)
    subprocess.run(["docker", "exec", cid, "rm", "-f", "/logs/verifier/reward.txt", "/logs/verifier/reward.json"],
                   capture_output=True, check=True)
    subprocess.run(["docker", "cp", f"{instance['task_dir']}/tests/.", f"{cid}:/tests/"], capture_output=True, check=True)
    out, finished = _run_tests(["docker", "exec", cid, "bash", "-lc",
                                f"cd {shlex.quote(env.config.cwd)} && bash /tests/test.sh"],
                               int(instance["max_test_timeout_sec"]) + 180)
    if not finished:
        return Verdict(None, out, "test timeout")

    def read(path: str) -> str | None:
        r = subprocess.run(["docker", "exec", cid, "cat", path], capture_output=True, text=True, errors="replace")
        return r.stdout if r.returncode == 0 else None

    resolved, reason = reward_verdict(read("/logs/verifier/reward.json"), read("/logs/verifier/reward.txt"))
    # Harbor takes the reward as it is. The tasks' test.sh write 0 whenever the pytest command
    # fails, also when it never started (curl or uv failed to install), so a failed task whose
    # output has no pytest short test summary keeps its verdict but is marked.
    warning = "no pytest short test summary" if resolved is False and parse_pytest_tests(out) is None else None
    return Verdict(resolved, out, reason, warning)


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


# Agent errors after which Harbor keeps no valid reward (models/trial/error_policy.py): the
# model API, not the agent's work, ended the run. The tests still run and their verdict is
# kept in the trajectory's info, but the task gets none. tb run has no such rule.
TB21_NO_VALID_REWARD = frozenset({
    "Timeout", "RateLimitError", "InternalServerError", "ServiceUnavailableError", "APIConnectionError", "APIError",
    "BadRequestError", "UnprocessableEntityError", "UnsupportedParamsError", "APIResponseValidationError",
    "BudgetExceededError", "AuthenticationError", "PermissionDeniedError", "NotFoundError"})


def _error(e: Exception) -> dict:
    return {"traceback": traceback.format_exc(), "exception_str": str(e)}


def process_instance(instance: dict, output_dir: Path, config: dict, agent_class: type, flavor: str, progress) -> None:
    """Run one task and score it as the official harness would. The tests run after the agent
    stops for any reason, an error included; only a task that never started (its image or
    container failed) or whose container is gone is not scored."""
    _, prepare, run_args, score = FLAVORS[flavor]
    iid = instance["instance_id"]
    results = output_dir / "results.json"
    _update_results(results, iid, None)
    model = get_model(config=config.get("model", {}))
    progress.on_instance_start(iid)
    progress.update_instance_status(iid, "Preparing image")
    env = agent = exit_status = None
    verdict, extra = Verdict(None, "", "not run"), {}
    try:
        try:
            workdir = prepare(instance)
            flavor_env = {"image": instance["image_name"], "cwd": workdir,
                          **({"run_args": run_args} if run_args else {})}  # some prebuilt images set an ENTRYPOINT
            user_env = {k: v for k, v in (config.get("environment") or {}).items() if k != "environment_class"}
            env = DockerEnvironment(**merge_env(user_env, flavor_env))
            agent = agent_class(model, env, **config.get("agent", {}))
            agent.instance_id = iid  # names the task's live log (refold/live.py)
        except Exception as e:
            logger.error(f"Could not start instance {iid}: {e}", exc_info=True)
            exit_status, extra = type(e).__name__, _error(e)
            verdict = Verdict(None, "", f"setup failed: {exit_status}")
            return
        progress.update_instance_status(iid, "Agent running")
        try:
            exit_status = agent.run(instance["problem_statement"]).get("exit_status")
        except Exception as e:  # both official harnesses still grade the final state
            logger.error(f"Agent error on instance {iid}: {e}", exc_info=True)
            exit_status, extra = type(e).__name__, _error(e)
        if exit_status == "ContainerGone":
            verdict = Verdict(None, "", "container gone")
            return
        progress.update_instance_status(iid, "Scoring")
        try:
            verdict = score(env, instance)
        except Exception as e:  # docker cp / exec failed
            logger.error(f"Could not score instance {iid}: {e}", exc_info=True)
            verdict = Verdict(None, "", f"scoring failed: {type(e).__name__}")
            extra["scoring_error"] = _error(e)
        if flavor == "tb21" and exit_status in TB21_NO_VALID_REWARD:
            extra["test_verdict"] = verdict.resolved
            verdict = verdict._replace(resolved=None, reason=f"agent API error: {exit_status}")
    finally:
        record = {"exit_status": exit_status, "resolved": verdict.resolved,
                  **({"verifier_error": verdict.reason} if verdict.resolved is None else {}),
                  **({"verifier_warning": verdict.warning} if verdict.warning else {})}
        if agent is not None:
            agent.save(output_dir / iid / f"{iid}.traj.json",
                       {"info": {**record, "test_output": verdict.output[-4000:], **extra}, "instance_id": iid})
        _update_results(results, iid, {"instance_id": iid, **record})
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
