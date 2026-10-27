#!/usr/bin/env bash
# Shared by the run scripts. Each cell gets a fresh vLLM server (empty prefix cache) with the
# window its budget needs, runs, and is scored by the benchmark's official evaluator.
#
#   GPUS=0,1 bash scripts/main.sh                             # every cell, one after another
#   GPUS=2,3 PORT=8001 ONLY=a3b bash scripts/main.sh          # a second pair of cards takes a subset
#   DRY=1 bash scripts/main.sh                                # print each cell's command, start nothing
#
#   GPUS   two free cards on one NUMA node, for tensor parallelism 2 (48 GB each)
#   PORT   the server's port (default 8000)
#   RUNS   where cells go (default runs/)
#   ONLY   a regex: run only the cells whose name matches
#   DRY    set to print commands instead of running them
#   FORCE  set to use cards that nvidia-smi shows in use
#
# Cells resume: a scored cell is skipped, an unscored one continues from its preds.json.
# Long sweeps belong in tmux. Cells need the Docker daemon (every task runs in its
# benchmark's official container).

set -euo pipefail

REPLY=16384   # max generated tokens per step; the server serves B + REPLY for a budget B
FULL=245760   # the usable context at the full 262,144-token window
PORT=${PORT:-8000}
RUNS=${RUNS:-runs}
API=http://127.0.0.1:$PORT/v1
declare -A MODELS=([27b]=Qwen/Qwen3.6-27B [a3b]=Qwen/Qwen3.6-35B-A3B)
VLLM_PID=

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

check_gpus() {
  : "${GPUS:?set GPUS to two free cards, e.g. GPUS=0,1 (check nvidia-smi first)}"
  local busy
  busy=$(nvidia-smi --id="$GPUS" --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' '$2 > 1024 {print $1}' | paste -sd, -)
  if [ -n "$busy" ] && [ -z "${FORCE:-}" ]; then
    echo "GPU $busy already in use (nvidia-smi): pick free cards, or set FORCE=1" >&2
    exit 1
  fi
}

serve() {  # serve MODEL BUDGET LOG: vLLM with --max-model-len B + REPLY
  local model=$1 budget=$2 log=$3
  check_gpus
  CUDA_VISIBLE_DEVICES=$GPUS vllm serve "$model" \
    --tensor-parallel-size 2 --dtype bfloat16 --gpu-memory-utilization 0.92 \
    --max-num-seqs 128 --max-num-batched-tokens 16384 --max-model-len $((budget + REPLY)) \
    --enable-prefix-caching --enable-prompt-tokens-details \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --port "$PORT" > "$log" 2>&1 &
  VLLM_PID=$!
  for _ in $(seq 180); do
    curl -fsS "$API/models" > /dev/null 2>&1 && return 0
    kill -0 "$VLLM_PID" 2> /dev/null || { tail -n 30 "$log" >&2; return 1; }
    sleep 10
  done
  echo "vLLM did not answer within 30 minutes; see $log" >&2
  return 1
}

stop_server() {
  if [ -n "$VLLM_PID" ]; then
    kill "$VLLM_PID" 2> /dev/null || true
    wait "$VLLM_PID" 2> /dev/null || true
  fi
  VLLM_PID=
}
trap stop_server EXIT

# cell NAME MODEL BUDGET WORKERS BENCHMARK [refold run flags...]: one cell on a fresh local server
cell() {
  local name=$1 model=$2 budget=$3 workers=$4 bench=$5
  shift 5
  if [ -n "${ONLY:-}" ] && ! [[ $name =~ $ONLY ]]; then return 0; fi
  local out=$RUNS/$name
  local args=(--model "$model" --api-base "$API" --context-budget "$budget" --workers "$workers"
              --benchmark "$bench" -o "$out" -c environment.container_timeout=6h "$@")
  if [ -n "${DRY:-}" ]; then
    echo "# $name: vllm serve $model --max-model-len $((budget + REPLY))"
    echo "refold run ${args[*]}"
    return 0
  fi
  if [ -f "$out/resolved.json" ]; then log "$name: scored, skipped"; return 0; fi
  mkdir -p "$out"
  log "$name: serving $model with --max-model-len $((budget + REPLY))"
  serve "$model" "$budget" "$out/vllm.log"
  refold run "${args[@]}"
  stop_server
  refold score "$out"
}
