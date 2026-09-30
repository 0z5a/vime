#!/bin/bash
set -euo pipefail
root=/home/jwipc/experiments/ouro-vime-20260930
export GPUS_PER_NODE=1
export PATH=/home/jwipc/projects/host-gateway/.vllm/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTHONPATH=$root:$root/rlt-ouro-graph-api-20260930:$root/vime-ouro-graph-integration-20260930:/home/jwipc/experiments/ouro-vime-20260930/megatron-runtime
bash "$root/vime-ouro-graph-integration-20260930/examples/ouro/run.sh" "$root/models/Ouro-tiny" "$root/train.jsonl" "$root/tiny-graph-r2" 3 --ouro-depths 2 3 4 --ouro-cuda-graphs --ouro-kv-blocks 128 --rollout-max-response-len 16 --seq-length 512 --max-position-embeddings 2048 --num-layers 2 --hidden-size 256 --ffn-hidden-size 512 --num-attention-heads 4 --num-query-groups 2 --kv-channels 64 --transformer-impl local --no-rope-fusion --ouro-reward-function smoke_reward.score "$@"
