#!/bin/bash
set -euo pipefail
root=/home/jwipc/experiments/ouro-vime-20260930
export GPUS_PER_NODE=1
export PATH=/home/jwipc/projects/host-gateway/.vllm/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTHONPATH=$root:$root/rlt-ouro-graph-api-20260930:$root/vime-ouro-graph-integration-20260930:$root/megatron-runtime
bash "$root/vime-ouro-graph-integration-20260930/examples/ouro/run.sh" "$root/models/Ouro-1.4B-parallel" "$root/train.jsonl" "$root/full-$1" 4 --ouro-depths 4 --rollout-max-response-len 128 --ouro-kv-blocks 1024 --seq-length 2048 --ouro-reward-function vime_plugins.ouro.reward.boxed_answer --ouro-eval-data "$root/eval.jsonl" --ouro-eval-prompts 4 --ouro-eval-interval 4 "${@:2}"
