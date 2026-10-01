#!/bin/bash
set -euo pipefail
root=/home/jwipc/experiments/ouro-vime-20260930
python=/home/jwipc/projects/host-gateway/.vllm/bin/python
export PYTHONPATH=$root/rlt-training-contract-20261001:$root/vime-ouro-graph-integration-20260930:$root/megatron-runtime
run() {
  local name=$2
  local complete=$root/shared-${name%-resume}/complete-rank0.json
  if [ -f "$complete" ] && "$python" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["next_update"] < int(sys.argv[2]))' "$complete" "$3"; then
    return
  fi
  flock /tmp/codex-thor-perf.lock bash "$root/run_shared.sh" "$@" > "$root/shared-$name.log" 2>&1
}
run tiny tiny-eager 4
run tiny tiny-graph 3 --ouro-cuda-graphs
run tiny tiny-graph-resume 4 --ouro-cuda-graphs --ouro-resume --override-opt-param-scheduler \
  --load "$root/shared-tiny-graph/checkpoint" --save "$root/shared-tiny-graph/checkpoint" \
  --ouro-run-dir "$root/shared-tiny-graph" --ouro-export-hf "$root/shared-tiny-graph/hf-export"
"$python" "$root/compare_checkpoint.py" "$root/shared-tiny-eager/checkpoint/iter_0000003" \
  "$root/shared-tiny-graph/checkpoint/iter_0000003" "$root/shared-tiny-checkpoint-match.json" > "$root/shared-tiny-compare.log" 2>&1
"$python" "$root/vime-ouro-graph-integration-20260930/examples/ouro/check_hf_export.py" \
  "$root/shared-tiny-graph/checkpoint/iter_0000003" "$root/shared-tiny-graph/hf-export" \
  "$root/shared-tiny-hf-match.json" > "$root/shared-tiny-hf.log" 2>&1
run tiny tiny-early 2 --ouro-depths 4 --ouro-exit-threshold 0 --ouro-cuda-graphs
run tiny tiny-async 2 --ouro-depths 4 --ouro-exit-threshold 0 --ouro-cuda-graphs --ouro-async-scheduling
run tiny tiny-spec 2 --ouro-depths 4 --ouro-cuda-graphs --ouro-speculative-tokens 3
run full full-eager 5
run full full-graph 4 --ouro-cuda-graphs
run full full-graph-resume 5 --ouro-cuda-graphs --ouro-resume --override-opt-param-scheduler \
  --load "$root/shared-full-graph/checkpoint" --save "$root/shared-full-graph/checkpoint" \
  --ouro-run-dir "$root/shared-full-graph" --ouro-export-hf "$root/shared-full-graph/hf-export"
"$python" "$root/compare_checkpoint.py" "$root/shared-full-eager/checkpoint/iter_0000004" \
  "$root/shared-full-graph/checkpoint/iter_0000004" "$root/shared-full-checkpoint-match.json" > "$root/shared-full-compare.log" 2>&1
"$python" "$root/vime-ouro-graph-integration-20260930/examples/ouro/check_hf_export.py" \
  "$root/shared-full-graph/checkpoint/iter_0000004" "$root/shared-full-graph/hf-export" \
  "$root/shared-full-hf-match.json" > "$root/shared-full-hf.log" 2>&1
run full full-early 2 --ouro-depths 4 --ouro-exit-threshold 0 --ouro-cuda-graphs
run full full-async 2 --ouro-depths 4 --ouro-exit-threshold 0 --ouro-cuda-graphs --ouro-async-scheduling
run full full-spec 2 --ouro-depths 4 --ouro-cuda-graphs --ouro-speculative-tokens 3
date -u +%FT%TZ > "$root/shared-sequence-complete.txt"
