"""refold report -- the run metrics for one or more cells.

    refold report runs/tab1/27b-swebench-base runs/tab1/27b-swebench-refold
    refold report CELL --tasks tasks.csv --calls calls.csv      # the per-task / per-call records

For task i of the N in a cell, model call j sends p_ij prompt tokens, of which c_ij are
served from the prefix cache, and receives g_ij completion tokens. A call an arm makes
for itself (a compaction summary) counts toward tokens and cost like any other.

    Resolve      tasks the official evaluator scores as solved (`refold score` writes them)
    Exhausted    tasks with a request longer than the usable context B
    Timeouts     tasks ended by a request that streams no token for 600 s
    Steps        model calls of a task; median over tasks, halves rounded up
    Tokens       mean over tasks of sum_j (p_ij + g_ij), in thousands
    Cost         mean over tasks of sum_j pi_u (p_ij - c_ij) + pi_c c_ij + pi_g g_ij, in US cents
    Cache hit    sum c_ij / sum p_ij
    Duration     median over tasks of the wall clock from its first to its last message
    KV cache     median over tasks of kappa * max_j p_ij + s, in GiB
    Compactions  mean summarize-and-restart events per task
    Overhead     mean context-management time per step

and per run, from the server's /metrics series that `refold run` samples:

    KV peak      largest share of the KV-cache pool in use
    Queue        mean wait before the scheduler admits a request
    Preemptions  requests evicted from the KV cache and later recomputed

Rates and KV constants are set for the two Qwen3.6 models; any other model needs
--price and --kv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

from refold.cli.run import FULL_BUDGET

# model -> ($/M uncached input, $/M cached input, $/M completion, KiB/token of KV cache
# over both GPUs at bf16, fixed GiB of the linear-attention layers).
MODELS = {
    "qwen3.6-35b-a3b": (0.10, 0.05, 0.90, 20, 0.06),
    "qwen3.6-27b": (0.30, 0.15, 2.70, 64, 0.14),
}

TASK_FIELDS = ["cell", "task", "resolved", "exit_status", "exhausted", "timeout", "steps", "prompt_tokens",
               "cached_tokens", "completion_tokens", "tokens", "cost_cents", "peak_prompt_tokens", "kv_gib",
               "duration_min", "compactions", "overhead_ms"]
CALL_FIELDS = ["cell", "task", "call", "auxiliary", "prompt_tokens", "cached_tokens", "completion_tokens",
               "overhead_ms"]


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("cells", nargs="+", type=Path)
    p.add_argument("--price", default=None, metavar="IN,CACHED,OUT", help="US$ per million tokens")
    p.add_argument("--kv", default=None, metavar="KIB,GIB", help="KV cache per token (KiB) and fixed state (GiB)")
    p.add_argument("--budget", type=int, default=None, metavar="B",
                   help="usable context, for cells without plan.json (default: the full window)")
    p.add_argument("--tasks", type=Path, default=None, metavar="CSV", help="write one row per task")
    p.add_argument("--calls", type=Path, default=None, metavar="CSV", help="write one row per model call")


def main(args) -> int:
    rows, tasks, calls = [], [], []
    for cell in args.cells:
        row, cell_tasks, cell_calls = cell_report(cell, args.price, args.kv, args.budget)
        rows.append(row)
        tasks += cell_tasks
        calls += cell_calls
    print(markdown(rows))
    for path, fields, data in ((args.tasks, TASK_FIELDS, tasks), (args.calls, CALL_FIELDS, calls)):
        if path is not None:
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(data)
    return 0


# ----------------------------------------------------------------------------- one cell
def cell_report(cell: Path, price: str | None = None, kv: str | None = None,
                budget: int | None = None) -> tuple[dict, list[dict], list[dict]]:
    trajs = sorted(cell.glob("*/*.traj.json"))
    if not trajs:
        raise FileNotFoundError(f"no trajectories under {cell}")
    plan = json.loads((cell / "plan.json").read_text()) if (cell / "plan.json").is_file() else {}
    first = json.loads(trajs[0].read_text())
    model = plan.get("model_name") or first.get("info", {}).get("config", {}).get("model", {}).get("model_name", "")
    b = budget or plan.get("context_budget") or FULL_BUDGET
    rates = _constants(model, price, kv)
    resolved, n_resolved = _resolved(cell)

    tasks, calls, saw_cached = [], [], False
    for path in trajs:
        task, task_calls, cached_reported = _task(json.loads(path.read_text()), path.parent.name, b, rates)
        task["cell"], task["resolved"] = str(cell), "" if resolved is None else int(task["task"] in resolved)
        tasks.append(task)
        calls += [{"cell": str(cell), "task": task["task"], **c} for c in task_calls]
        saw_cached |= cached_reported

    overhead = [c["overhead_ms"] for c in calls if c["overhead_ms"] != ""]
    compactions = [t["compactions"] for t in tasks if t["compactions"] != ""]
    row = {"cell": str(cell), "N": len(tasks),
           "Resolve": n_resolved if n_resolved is not None else "—",
           "Exhausted": sum(t["exhausted"] for t in tasks),
           "Timeouts": sum(t["timeout"] for t in tasks),
           "Steps": _median_int([t["steps"] for t in tasks]),
           "Tokens (k)": _mean([t["tokens"] for t in tasks]) / 1000,
           "Cost (¢)": _mean([t["cost_cents"] for t in tasks]) if rates else None,
           "Cache hit (%)": (100 * sum(t["cached_tokens"] for t in tasks)
                             / max(1, sum(t["prompt_tokens"] for t in tasks)) if saw_cached else None),
           "Duration (min)": st.median([t["duration_min"] for t in tasks if t["duration_min"] != ""] or [math.nan]),
           "KV cache (GiB)": st.median([t["kv_gib"] for t in tasks]) if rates else None,
           "Compactions": _mean(compactions) if compactions else None,
           "Overhead (ms/step)": _mean(overhead) if overhead else None,
           **server_metrics(cell / "server_metrics.jsonl")}
    if rates and not saw_cached:
        row["Cost (¢)"] = f"≤{row['Cost (¢)']:.1f}"  # no cache accounting: everything priced uncached
    return row, tasks, calls


def _task(traj: dict, name: str, budget: int, rates) -> tuple[dict, list[dict], bool]:
    info = traj.get("info") or {}
    calls, stamps, cached_reported = [], [], False
    for m in traj.get("messages", []):
        e = m.get("extra")
        if not isinstance(e, dict):
            continue
        if isinstance(e.get("timestamp"), (int, float)):
            stamps.append(e["timestamp"])
        r = e.get("response")
        usage = (r.get("usage") or {}) if isinstance(r, dict) else {}
        if not usage.get("prompt_tokens"):
            continue
        details = usage.get("prompt_tokens_details")  # vLLM leaves it null when nothing was cached
        cached_reported |= isinstance(details, dict)
        overhead = e.get("overhead_s")
        calls.append({"call": len(calls), "auxiliary": 0, "prompt_tokens": usage["prompt_tokens"],
                      "cached_tokens": (details.get("cached_tokens") or 0) if isinstance(details, dict) else 0,
                      "completion_tokens": usage.get("completion_tokens") or 0,
                      "overhead_ms": 1000 * overhead if isinstance(overhead, (int, float)) else ""})
    steps, peak = len(calls), max((c["prompt_tokens"] for c in calls), default=0)

    # A summary call counts like any other. Research-fork trajectories carry only totals.
    compact = info.get("compact") or {}
    aux = compact.get("summary_calls")
    if aux is None and info.get("summ_prompt_tokens"):
        aux = [{"prompt_tokens": info["summ_prompt_tokens"], "completion_tokens": info.get("summ_completion_tokens")}]
    for a in aux or []:
        calls.append({"call": len(calls), "auxiliary": 1, "overhead_ms": "",
                      **{k: a.get(k) or 0 for k in ("prompt_tokens", "cached_tokens", "completion_tokens")}})
    compactions = compact.get("compactions")
    if compactions is None and "summ_calls" in info:
        compactions = info["summ_calls"] - (info.get("summ_degenerate") or 0)

    p, c, g = (sum(x[k] for x in calls) for k in ("prompt_tokens", "cached_tokens", "completion_tokens"))
    exit_status = str(info.get("exit_status") or "")
    task = {"task": name, "exit_status": exit_status,
            "exhausted": int("Context" in exit_status or peak > budget),
            "timeout": int("Timeout" in exit_status), "steps": steps,
            "prompt_tokens": p, "cached_tokens": c, "completion_tokens": g, "tokens": p + g,
            "cost_cents": 100 * (rates[0] * (p - c) + rates[1] * c + rates[2] * g) / 1e6 if rates else "",
            "peak_prompt_tokens": peak,
            "kv_gib": rates[3] * peak / 2**20 + rates[4] if rates else "",
            "duration_min": (max(stamps) - min(stamps)) / 60 if len(stamps) >= 2 else "",
            "compactions": "" if compactions is None else compactions,
            "overhead_ms": sum(x["overhead_ms"] for x in calls if x["overhead_ms"] != "")}
    return task, calls, cached_reported


def server_metrics(path: Path) -> dict:
    """KV peak, queue and preemptions over one run's /metrics series."""
    out = {"KV peak (%)": None, "Queue (s)": None, "Preemptions": None}
    if not path.is_file():
        return out
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) < 2:
        return out
    usage = [r.get("vllm:kv_cache_usage_perc", r.get("vllm:gpu_cache_usage_perc")) for r in rows]
    usage = [u for u in usage if u is not None]
    first, last = rows[0], rows[-1]

    def delta(k):
        return (last.get(k) or 0.0) - (first.get(k) or 0.0)

    n = delta("vllm:request_queue_time_seconds_count")
    out["KV peak (%)"] = 100 * max(usage) if usage else None
    out["Queue (s)"] = delta("vllm:request_queue_time_seconds_sum") / n if n else None
    out["Preemptions"] = int(delta("vllm:num_preemptions_total"))
    return out


# ----------------------------------------------------------------------------- helpers
def _constants(model: str, price: str | None, kv: str | None):
    known = next((v for k, v in MODELS.items() if k in model.lower()), None)
    if price is None and kv is None:
        return known
    pu, pc, pg = [float(x) for x in price.split(",")] if price else (known or (None,) * 5)[:3]
    kib, gib = [float(x) for x in kv.split(",")] if kv else (known or (None,) * 5)[3:]
    if None in (pu, pc, pg, kib, gib):
        raise ValueError(f"no rates for {model!r}: pass both --price IN,CACHED,OUT and --kv KIB,GIB")
    return pu, pc, pg, kib, gib


def _resolved(cell: Path) -> tuple[set[str] | None, int | None]:
    """The ids `refold score` wrote, or Terminal-Bench's in-run results."""
    if (cell / "resolved.json").is_file():
        d = json.loads((cell / "resolved.json").read_text())
        ids = set(d.get("resolved") or ())
        return ids, d.get("n_resolved", len(ids))
    if (cell / "results.json").is_file():
        ids = {k for k, v in json.loads((cell / "results.json").read_text()).items() if v.get("resolved")}
        return ids, len(ids)
    return None, None


def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else math.nan


def _median_int(xs: list[int]) -> int | None:
    return math.floor(st.median(xs) + 0.5) if xs else None


def markdown(rows: list[dict]) -> str:
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)
