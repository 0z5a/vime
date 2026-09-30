#!/bin/bash
set -euo pipefail
root=/home/jwipc/experiments/ouro-vime-20260930
export PATH=/home/jwipc/projects/host-gateway/.vllm/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTHONPATH=$root:$root/rlt-ouro-graph-api-20260930:$root/vime-ouro-graph-integration-20260930:$root/megatron-runtime
bash "$root/run_full.sh" eager --num-rollout 5 > "$root/full-eager.log" 2>&1
bash "$root/run_full.sh" graph --ouro-cuda-graphs > "$root/full-graph.log" 2>&1
bash "$root/run_full.sh" graph --ouro-cuda-graphs --ouro-resume --load "$root/full-graph/checkpoint" --num-rollout 5 --use-checkpoint-opt-param-scheduler > "$root/full-resume.log" 2>&1
python "$root/compare_checkpoint.py" "$root/full-eager/checkpoint/iter_0000004" "$root/full-graph/checkpoint/iter_0000004" "$root/full-checkpoint-match.json" > "$root/full-checkpoint-match.log" 2>&1
printf '{"completed":true}\n' > "$root/full-validation-complete.json"
