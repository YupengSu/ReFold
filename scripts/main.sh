#!/usr/bin/env bash
# Main results: base vs ReFold on five benchmarks and two models, 50 tasks each, at the
# full window (B = 245,760) with w = 8 concurrent sessions.
#
#   GPUS=0,1 bash scripts/main.sh
#   refold report runs/main/*
#
# Terminal-Bench needs its task directories (TBENCH1_TASKS, TBENCH21_TASKS) and SWE-bench
# Pro its scorer checkout (SWEBENCH_PRO_HARNESS); see README.
source "$(dirname "$0")/common.sh"

for model in 27b a3b; do
  for bench in swebench swebenchpro msb tbench1 tbench21; do
    for arm in base refold; do
      cell "main/$model-$bench-$arm" "${MODELS[$model]}" $FULL 8 $bench --strategy $arm
    done
  done
done
