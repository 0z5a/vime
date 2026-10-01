#!/bin/bash
set -euo pipefail
root=/home/jwipc/experiments/ouro-vime-20260930
size=${1:?tiny or full}
name=${2:?run name}
updates=${3:?number of updates}
shift 3
export GPUS_PER_NODE=1
export PATH=/home/jwipc/projects/host-gateway/.vllm/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTHONPATH=$root:$root/rlt-training-contract-20261001:$root/vime-ouro-graph-integration-20260930:$root/megatron-runtime
if [ "$size" = tiny ]; then
  model=$root/models/Ouro-tiny
  dimensions=(--num-layers 2 --hidden-size 256 --ffn-hidden-size 512 --num-attention-heads 4 --num-query-groups 2 --kv-channels 64)
else
  model=$root/models/Ouro-1.4B-parallel
  dimensions=()
fi
bash "$root/vime-ouro-graph-integration-20260930/examples/ouro/run.sh" "$model" "$root/train.jsonl" \
  "$root/shared-$name" "$updates" --ouro-depths 2 3 4 --ouro-kv-blocks 1024 --rollout-max-response-len 16 \
  --seq-length 512 --max-position-embeddings 2048 --ouro-reward-function smoke_reward.score \
  "${dimensions[@]}" "$@"
