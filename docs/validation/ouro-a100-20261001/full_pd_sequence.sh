#!/bin/bash
set -euo pipefail
root=/workspace/0z5a/work/ouro-vime-contract-20261001
cd "$root"
export PYTHONPATH=$root:$root/rlt:$root/vime:$root/megatron-runtime
common=(--ouro-prefill-devices 0 --ouro-decode-devices 1 --ouro-max-num-seqs 1 --rollout-batch-size 2 --n-samples-per-prompt 4 --global-batch-size 8)
bash run_pd_a100.sh full train-full-pd-eager 3 "${common[@]}" >train-full-pd-eager.log 2>&1
bash run_pd_a100.sh full train-full-pd-graph 2 "${common[@]}" --ouro-cuda-graphs >train-full-pd-graph.log 2>&1
bash run_pd_a100.sh full train-full-pd-graph 3 "${common[@]}" --ouro-cuda-graphs --ouro-resume \
  --load "$root/train-full-pd-graph/checkpoint" --override-opt-param-scheduler \
  --ouro-export-hf "$root/train-full-pd-graph/export" >train-full-pd-resume.log 2>&1
/venv/main/bin/python compare_checkpoint.py train-full-pd-eager/checkpoint/iter_0000002 \
  train-full-pd-graph/checkpoint/iter_0000002 full-pd-checkpoint.json >full-pd-compare.log 2>&1
/venv/main/bin/python check_hf_export.py train-full-pd-graph/checkpoint/iter_0000002 \
  train-full-pd-graph/export full-pd-hf.json >full-pd-hf.log 2>&1
/venv/main/bin/python cleanup_a100.py full
