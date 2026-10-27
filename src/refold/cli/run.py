"""refold run -- one cell: a strategy, a model, a benchmark, a context budget, a concurrency.

Serving is not this package's concern: a local vLLM and a cloud API are both reached
through litellm, so --model is a litellm model string and --api-base points at a local
server. The cell runs in a subprocess (the benchmark's runner module), and the resolved
plan and the exact command are written to plan.json and command.txt.

Against a local server the launcher also checks that the server serves B + 16,384 tokens
(the usable budget plus the reply ceiling) and samples its /metrics into
server_metrics.jsonl while the cell runs, for the per-run server metrics.

--live also logs every model call to OUT/live/<instance_id>.jsonl as it returns
(refold/live.py) and samples /metrics every second, for a dashboard that follows the run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from refold.benchmarks import available as benchmarks
from refold.benchmarks import get_benchmark
from refold.live import LIVE_ENV
from refold.registry import available, get_strategy, import_class

# The serving setup. The server's window is CONTEXT_WINDOW; a budget B is served as
# --max-model-len B + MAX_TOKENS, so the usable context at the full window is 245,760.
CONTEXT_WINDOW = 262_144
MAX_TOKENS = 16_384  # max generated tokens per step
FULL_BUDGET = CONTEXT_WINDOW - MAX_TOKENS
REQUEST_TIMEOUT = 600  # a request with no token for 600 s ends the task, without retry
STEP_LIMIT = 250
METRICS_INTERVAL = 5.0  # seconds between /metrics samples
LIVE_METRICS_INTERVAL = 1.0  # with --live: the dashboard's KV-cache curve needs the finer series


def build_config(strategy: str, model: str, api_base: str | None = None, overrides: list[str] = (),
                 compact: bool = False, context_budget: int = FULL_BUDGET, cost_limit: float | None = None,
                 model_class: str | None = None, benchmark: str = "swebench") -> dict:
    spec = get_strategy(strategy)
    agent_path, arm = spec.agent, strategy
    if compact:
        if spec.compact_agent is None:
            raise ValueError("--compact runs a strategy beneath compaction and is wired for refold only "
                             "(compact + ReFold); the compact baseline itself is --strategy compact")
        agent_path, arm = spec.compact_agent, f"compact+{strategy}"
    fields = import_class(agent_path).config_class.model_fields
    agent: dict = {}
    if "context_budget" in fields:  # the arm triggers on B: compact, compact + ReFold
        agent["context_budget"] = context_budget
    other: dict = {}  # model.* / environment.* overrides, passed through as given
    for kv in overrides:
        if "=" not in kv:
            raise ValueError(f"-c expects key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        if k.startswith(("model.", "environment.")):
            other[k] = _coerce(v)
            continue
        k = k.removeprefix("agent.")
        if k not in fields:
            knobs = sorted(set(fields) - {"system_template", "instance_template", "output_path"})
            raise ValueError(f"{arm!r} has no knob {k!r}; its knobs are {', '.join(knobs)}")
        agent[k] = _coerce(v)
    if api_base:
        # A local server: temperature 0, as every reported vLLM cell passed it.
        model = model if model.startswith("openai/") else f"openai/{model}"
        model_kwargs = {"temperature": 0.0, "api_base": api_base, "api_key": "dummy"}
    else:
        model_kwargs = {}  # a cloud API: nothing beyond the benchmark config
    return {"arm": arm, "strategy": strategy, "compact": compact, "agent_path": agent_path,
            "context_budget": context_budget, "agent": agent, "other": other,
            "model_name": model, "model_class": model_class,
            "model_kwargs": model_kwargs, "cost_limit": cost_limit}


def _coerce(v: str):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def build_command(ids: list[str], cfg: dict, out: Path, workers: int, bench) -> list[str]:
    cmd = [sys.executable, "-m", bench.runner_module, *bench.data_flags(),
           # anchored: an unanchored `a|b` also selects every id that merely starts with a
           "--filter", "^(" + "|".join(re.escape(i) for i in ids) + ")$",
           "-o", str(out), "-w", str(workers), "--model", cfg["model_name"],
           "--agent-class", cfg["agent_path"]]
    if cfg.get("model_class"):
        cmd += ["--model-class", cfg["model_class"]]
    cmd += ["-c", str(bench.config)]
    cmd += ["-c", f"agent.step_limit={STEP_LIMIT}",
            "-c", f"agent.cost_limit={cfg.get('cost_limit') or 0}",
            "-c", "model.cost_tracking=ignore_errors",
            "-c", f"model.model_kwargs.max_tokens={MAX_TOKENS}",
            "-c", f"model.model_kwargs.timeout={REQUEST_TIMEOUT}"]
    cmd += [x for k, v in cfg["model_kwargs"].items() for x in ("-c", f"model.model_kwargs.{k}={v}")]
    cmd += [x for k, v in cfg["agent"].items() for x in ("-c", f"agent.{k}={v}")]
    cmd += [x for k, v in cfg.get("other", {}).items() for x in ("-c", f"{k}={v}")]
    return cmd


def build_env(cfg: dict) -> dict:
    """Against a local vLLM, litellm's retries turn a stall into a request storm that lands
    in the token account as real traffic, so they are off there. Cloud APIs keep them."""
    env = dict(os.environ)
    if cfg["model_kwargs"].get("api_base"):
        env["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] = "1"
    return env


# ----------------------------------------------------------------------------- the server
def check_server(api_base: str, context_budget: int) -> None:
    """The budget is set in two places that must agree: the agent is told B, the server
    serves B + MAX_TOKENS. A mismatch silently measures another budget."""
    try:
        with urllib.request.urlopen(f"{api_base.rstrip('/')}/models", timeout=15) as r:
            served = [m.get("max_model_len") for m in json.load(r).get("data", [])]
    except Exception as e:
        raise ValueError(f"cannot reach the server at {api_base}: {e}") from None
    want = context_budget + MAX_TOKENS
    if served and served[0] is not None and served[0] != want:
        raise ValueError(f"the server at {api_base} serves max_model_len={served[0]}, but --context-budget "
                         f"{context_budget} needs {want}: restart vLLM with --max-model-len {want}, or pass "
                         f"--context-budget {served[0] - MAX_TOKENS}")


_METRIC_RE = re.compile(r"^(vllm:[a-z_0-9]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$")
_METRICS = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "vllm:num_requests_running",
            "vllm:num_requests_waiting", "vllm:num_preemptions_total", "vllm:request_queue_time_seconds_sum",
            "vllm:request_queue_time_seconds_count", "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
            "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total")


def parse_metrics(text: str) -> dict[str, float]:
    """The metrics of _METRICS in a Prometheus exposition, summed over their label sets."""
    row: dict[str, float] = {}
    for line in text.splitlines():
        m = _METRIC_RE.match(line)
        if m and m.group(1) in _METRICS:
            row[m.group(1)] = row.get(m.group(1), 0.0) + float(m.group(2))
    return row


def sample_metrics(api_base: str, out: Path, stop: threading.Event, interval: float = METRICS_INTERVAL) -> None:
    """One JSON line of the server's counters and gauges every `interval` seconds.
    KV usage is a gauge, so its peak needs the series, not two snapshots."""
    url = re.sub(r"/v1/?$", "", api_base.rstrip("/")) + "/metrics"
    with out.open("a") as f:
        while not stop.is_set():
            try:
                text = urllib.request.urlopen(url, timeout=10).read().decode()
                f.write(json.dumps({"t": time.time(), **parse_metrics(text)}) + "\n")
                f.flush()
            except Exception:
                pass  # a missed sample is a gap in the series, not a failed cell
            stop.wait(interval)


# ----------------------------------------------------------------------------- the command
def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--strategy", required=True, choices=available())
    p.add_argument("--compact", action="store_true", help="run ReFold beneath compaction (compact + ReFold)")
    p.add_argument("--model", required=True, help="litellm model string, e.g. Qwen/Qwen3.6-27B with --api-base")
    p.add_argument("--api-base", default=None, help="a local OpenAI-compatible server, e.g. http://127.0.0.1:8000/v1")
    p.add_argument("--model-class", default=None, help="dotted path of a custom Model class")
    p.add_argument("--context-budget", type=int, default=FULL_BUDGET, metavar="B",
                   help=f"usable context in tokens; the server must serve B + {MAX_TOKENS} "
                        f"(default {FULL_BUDGET}, the full window)")
    p.add_argument("--cost-limit", type=float, default=None, metavar="USD", help="per-task spend ceiling")
    p.add_argument("--benchmark", default="swebench", choices=benchmarks())
    p.add_argument("--subset", default="all", help="a named set in benchmarks/sets/; '' for the whole split")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=8, metavar="W", help="concurrent sessions (default 8)")
    p.add_argument("-c", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="knob override: -c chunk_size=5 (agent), -c environment.container_timeout=6h")
    p.add_argument("-o", dest="out", required=True, type=Path)
    p.add_argument("--live", action="store_true",
                   help="log every model call to OUT/live/ as it returns and sample /metrics every second")
    p.add_argument("--dashboard", nargs="?", type=int, const=8901, default=None, metavar="PORT",
                   help="--live, and serve a live page for this run on 127.0.0.1:PORT (default 8901) while it "
                        "lasts; OUT/dashboard.html replays it afterwards")
    p.add_argument("--dry-run", action="store_true", help="print the plan and argv; run nothing")


def main(args) -> int:
    bench = get_benchmark(args.benchmark)
    cfg = build_config(args.strategy, args.model, args.api_base, args.overrides, compact=args.compact,
                       context_budget=args.context_budget, cost_limit=args.cost_limit,
                       model_class=args.model_class, benchmark=args.benchmark)
    ids = bench.load(subset=args.subset or None, limit=args.limit)
    cmd = build_command(ids, cfg, args.out, args.workers, bench)
    env = build_env(cfg)
    dash_port = getattr(args, "dashboard", None)
    live = getattr(args, "live", False) or dash_port is not None
    if live:
        env[LIVE_ENV] = str((args.out / "live").resolve())
    plan = {"benchmark": args.benchmark, "data": bench.provenance(), "subset": args.subset, "instances": len(ids),
            "task_ids": ids, "workers": args.workers, **({"live": True} if live else {}), **cfg}
    if args.dry_run:
        print(json.dumps(plan, indent=1, default=str))
        print(" ".join(shlex.quote(c) for c in cmd))
        return 0
    if args.api_base:
        check_server(args.api_base, args.context_budget)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "plan.json").write_text(json.dumps(plan, indent=1, default=str))
    (args.out / "command.txt").write_text(
        "".join(f"{k}={env[k]} " for k in ("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", LIVE_ENV) if k in env)
        + " ".join(shlex.quote(c) for c in cmd) + "\n")
    print(f"{len(ids)} instances -> {args.out}", flush=True)
    dash = None
    if dash_port is not None:
        from refold import dashboard

        dash = dashboard.serve_in_background(args.out, dash_port)
        print(f"dashboard: http://127.0.0.1:{dash_port}/  (ssh -L {dash_port}:127.0.0.1:{dash_port} HOST)", flush=True)
    stop = threading.Event()
    sampler = None
    if args.api_base:
        sampler = threading.Thread(target=sample_metrics, args=(args.api_base, args.out / "server_metrics.jsonl", stop,
                                                                LIVE_METRICS_INTERVAL if live else METRICS_INTERVAL),
                                   daemon=True)
        sampler.start()
    try:
        rc = subprocess.run(cmd, env=env).returncode
    finally:
        stop.set()
        if sampler is not None:
            sampler.join(timeout=15)
        if dash is not None:
            from refold import dashboard

            dash.shutdown()
            print(f"dashboard replay: {dashboard.write_static(args.out, args.out / 'dashboard.html')}", flush=True)
    n = len(list(args.out.rglob("*.traj.json")))
    print(f"rc={rc}, {n} trajectories")
    if n == 0:  # zero trajectories is a failed cell, whatever the exit code says
        print("no trajectories were written -- treating as failure", file=sys.stderr)
        return rc or 3
    return rc
