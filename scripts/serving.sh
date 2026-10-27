#!/usr/bin/env bash
# Serving constraints: Qwen3.6-27B on SWE-bench Verified under capped context budgets
# (w = 8), higher concurrency (full window), and both at once, each with and without
# ReFold. The full-window, w = 8 pair is in scripts/main.sh (main/27b-swebench-*).
#
#   GPUS=0,1 bash scripts/serving.sh
#   refold report runs/main/27b-swebench-* runs/serving/*
source "$(dirname "$0")/common.sh"
M=${MODELS[27b]}

# Context budgets at w = 8: compact vs compact + ReFold.
for k in 64 32 24 16; do
  cell "serving/${k}k-w8-compact" "$M" $((k * 1024)) 8 swebench --strategy compact
  cell "serving/${k}k-w8-compact-refold" "$M" $((k * 1024)) 8 swebench --strategy refold --compact
done

# Concurrency at the full window: base vs ReFold.
for w in 16 32 48; do
  cell "serving/full-w$w-base" "$M" $FULL $w swebench --strategy base
  cell "serving/full-w$w-refold" "$M" $FULL $w swebench --strategy refold
done
# 50 tasks cannot fill 64 lanes: w = 64 runs a 100-task superset (`a100`); compare the
# arms on the tasks both finished.
cell serving/full-w64-base "$M" $FULL 64 swebench --strategy base --subset a100
cell serving/full-w64-refold "$M" $FULL 64 swebench --strategy refold --subset a100

# Both constraints: a 16k budget at w = 48.
cell serving/16k-w48-compact "$M" 16384 48 swebench --strategy compact
cell serving/16k-w48-compact-refold "$M" 16384 48 swebench --strategy refold --compact
