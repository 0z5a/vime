#!/bin/bash
set -euo pipefail
root=/workspace/0z5a/work/ouro-vime-contract-20261001
size=${1:?tiny or full}
name=${2:?run name}
updates=${3:?updates}
shift 3
export GPUS_PER_NODE=1 MASTER_PORT=29871
export PATH=/venv/main/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTHONPATH=$root:$root/rlt:$root/vime:$root/megatron-runtime
export UCX_TLS=tcp,cuda_copy
if [ "$size" = tiny ]; then
  model=$root/models/Ouro-tiny
  dimensions=(--num-layers 2 --hidden-size 256 --ffn-hidden-size 512 --num-attention-heads 4 --num-query-groups 2 --kv-channels 64)
else
  model=$root/models/Ouro-1.4B
  dimensions=()
fi
bash "$root/vime/examples/ouro/run.sh" "$model" "$root/train.jsonl" "$root/$name" "$updates" \
  --ouro-depths 2 3 4 --ouro-kv-blocks 128 --rollout-max-response-len 16 \
  --seq-length 512 --max-position-embeddings 2048 --ouro-reward-function smoke_reward.score \
  --ouro-async-scheduling "${dimensions[@]}" "$@"
