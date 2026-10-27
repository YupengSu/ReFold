"""What both runners share: config merging, the agent class, and the worker pool."""

from __future__ import annotations

import concurrent.futures
import os
import time
from pathlib import Path

from minisweagent.config import get_config_from_spec
from minisweagent.exceptions import InterruptAgentFlow
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import logger
from minisweagent.utils.serialize import UNSET, recursive_merge
from rich.live import Live

from refold.live import LIVE_ENV, with_live_log

_CONTAINER_GONE = "Error response from daemon: No such container"


def merge_config(config_spec: list[str], model: str | None, model_class: str | None) -> dict:
    """The -c layers in order (benchmark yaml, strategy yaml, key=value), then the flags."""
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({"model": {"model_name": model or UNSET, "model_class": model_class or UNSET}})
    return recursive_merge(*configs)


def stop_when_container_gone(agent_class: type) -> type:
    """The agent class, ending the task once its container is gone: the container outlived
    environment.container_timeout and was removed, so every further command fails."""

    class Guarded(agent_class):
        def step(self):
            added = super().step()
            if any(_CONTAINER_GONE in str(m.get("content", "")) for m in added or []):
                raise InterruptAgentFlow({"role": "exit", "content": "ContainerGone",
                                          "extra": {"exit_status": "ContainerGone", "submission": ""}})
            return added

    Guarded.__name__ = Guarded.__qualname__ = agent_class.__name__
    Guarded.__module__ = agent_class.__module__  # so the trajectory records the real class
    return Guarded


def with_progress(agent_class: type) -> type:
    """The agent class, taking the progress manager mini-swe-agent's SWE-bench loop passes
    and reporting each step to it; with REFOLD_LIVE_DIR set (`refold run --live`), also
    logging every model call there (refold/live.py)."""
    live_dir = os.environ.get(LIVE_ENV)
    inner = with_live_log(agent_class, live_dir) if live_dir else agent_class

    class Tracked(stop_when_container_gone(inner)):
        def __init__(self, *args, progress_manager, instance_id: str = "", **kwargs):
            super().__init__(*args, **kwargs)
            self.progress_manager, self.instance_id = progress_manager, instance_id

        def step(self):
            self.progress_manager.update_instance_status(self.instance_id,
                                                         f"Step {self.n_calls + 1:3d} (${self.cost:.2f})")
            return super().step()

    Tracked.__name__ = Tracked.__qualname__ = agent_class.__name__
    Tracked.__module__ = agent_class.__module__  # so the trajectory records the real class
    return Tracked


def run_batch(instances: list[dict], work, output_path: Path, workers: int) -> None:
    """`work(instance, progress_manager)` for every instance, `workers` at a time."""
    progress = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    def drain(futures):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error in future for instance {futures[future]}: {e}", exc_info=True)
                progress.on_uncaught_exception(futures[future], e)

    with Live(progress.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(work, inst, progress): inst["instance_id"] for inst in instances}
            try:
                drain(futures)
            except KeyboardInterrupt:
                for f in futures:
                    if not f.running() and not f.done():
                        f.cancel()
                drain(futures)
