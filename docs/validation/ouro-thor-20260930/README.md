# VIME Ouro GRPO on Thor — 2026-09-30

This extends the existing [budgeted GRPO draft](https://github.com/0z5a/vime/pull/4)
and [synchronous rollout draft](https://github.com/0z5a/vllm-rlt/pull/1), following
[issue 70's VIME integration work](https://github.com/ThinkFlowLab/vllm-rlt/issues/70#issuecomment-5887476683).
The recipe now accepts recurrent CUDA graphs while preserving synchronous full
weight publication, selected-token logprobs and deterministic budget/data cursors.

## Full model generation

| K | Eager (s) | CUDA graphs (s) | Speedup | Time saved |
|---|---:|---:|---:|---:|
| 2 | 1.0880 | 0.9597 | 1.134× | 11.79% |
| 3 | 1.5567 | 1.4105 | 1.104× | 9.40% |
| 4 | 1.9689 | 1.8269 | 1.078× | 7.21% |

Two prompts, 32-token cap per request, five alternating-order measured pairs per
K after per-arm warmup. Token IDs, logprobs and logical block-token counts match
exactly in every pair, including generation after full policy publication.
These medians exclude model load and capture.

## Real GRPO step timing

| Component | Eager (s) | CUDA graphs (s) | Speedup | Time saved |
|---|---:|---:|---:|---:|
| Rollout | 17.7078 | 16.7839 | 1.055× | 5.22% |
| Megatron training | 16.2651 | 16.7471 | 0.971× | -2.96% |
| Full weight publication | 0.0266 | 0.0268 | 0.993× | -0.72% |
| Whole GRPO step | 34.0036 | 33.7104 | 1.009× | 0.86% |

Medians of updates 1–3 at K=4: four prompts, four samples per prompt,
16 sequences per update, 128-token response cap, BF16, one Thor. Step time includes
rollout, exact boxed-answer scoring, VIME reward normalization/GRPO, Megatron
forward/backward/Adam and full policy publication. Both arms use identical prompts,
seeds, sampled tokens, rewards, work counts and optimizer settings. Update 0 and
held-out evaluation are outside these medians. Graph capture count stays at 8
through all four initial updates; replay count reaches 5,211 before final evaluation,
with zero fallbacks. Training is unchanged and its timing varies; the observed
whole-step gain is only 0.86%, smaller than the 5.22% rollout time reduction.

Eager ran five continuous updates in 249.13 s. Graph ran four updates in 210.19 s,
then resumed in a fresh process for update five in 97.07 s. The combined graph job
is 307.26 s (0.811× eager), including the extra process initialization, evaluation,
loading and save. Resume overhead must be included for that operational comparison.

## Gradient, publication and resume checks

- The full 1.43465B-parameter model runs real VIME/Megatron training. Gradient norms
  for updates 0–4 are 3.3697, 4.3443, 6.1851, 3.9051 and 8.8884 in both arms.
- Mean absolute training/rollout logprob differences are 0.01976, 0.02508, 0.02227,
  0.01639 and 0.01867, identical between eager and graphs. The BF16 Megatron and
  Triton paths have a nonzero numerical difference; the report preserves it.
- Every update publishes all physical weights in place and advances the policy
  version. The CUDA test additionally verifies stable addresses, graph reuse,
  K changes and released KV pages.
- Graph training saves after update four, then a fresh process loads model,
  optimizer, scheduler and RNG state and completes update five. Its final
  checkpoint is byte-identical to five continuous eager updates across all 272
  tensors: 269 model tensors plus FP32 master weights and both Adam moments.
  Adam group metadata, including step=5, also matches. See
  [checkpoint hashes](raw/full-checkpoint-match.json).
- A separate random tiny-model smoke run cycles K=2/3/4 and resumes at K=2.
  All 30 final tensors and Adam metadata match continuous training. Its byte-parity
  reward is an infrastructure probe; it is not a task-quality measurement.

## Held-out exact boxed-answer score

| K | Eager before | Eager after 4 updates | Graph before | Graph after 4 updates |
|---|---:|---:|---:|---:|
| 2 | 0/4 | 0/4 | 0/4 | 0/4 |
| 3 | 1/4 | 1/4 | 1/4 | 1/4 |
| 4 | 0/4 | 1/4 | 0/4 | 1/4 |

Four separate integer-arithmetic prompts, one sample per prompt per K. The resumed
job repeats the update-four evaluation; duplicate problem IDs are counted once.
These are small execution checks, not evidence of convergence. Training has
substantial 128-token truncation; raw response lengths and finish reasons are
preserved. Early exit, PD and speculative RL remain outside this change.

## Runtime and reproduction

Model `ByteDance/Ouro-1.4B`, pinned revision
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`; `model.safetensors` SHA256
`58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`.
Thor SM110/aarch64, driver 595.78, Python 3.12.3, PyTorch 2.13.0+cu130,
Triton 3.7.1, Transformers 5.17.0, NumPy 2.3.5, Safetensors 0.8.0.
[Package versions](raw/runtime.json) and [model config](raw/model-config.json) are recorded.

The existing Torch environment was used without package or driver updates.
Megatron-LM source revision `d113016bb2850c4b7d804f24ba72c055d7d4f861` runs from
an isolated worktree with [this native Adam patch](megatron-native-adam.patch),
adapted from the existing [0z5a checkpoint repair](https://github.com/0z5a/vime/commit/f8ee0915198d693daf5e97de090f216db34e3a3d).
It handles empty fresh-optimizer state, independent scalar step tensors and
excludes scalar steps from parameter-shaped shards. No Transformer Engine,
vLLM server, W&B or cached-metadata reader is needed for this synchronous recipe.

```bash
# MCORE points at the isolated source worktree with the patch applied.
export PYTHONPATH="$PWD:/path/to/vllm-rlt:$MCORE"
export GPUS_PER_NODE=1
bash examples/ouro/run.sh /models/Ouro-1.4B docs/validation/ouro-thor-20260930/raw/train.jsonl /runs/ouro-graph 4 \
  --ouro-depths 4 --ouro-cuda-graphs --ouro-kv-blocks 1024 \
  --rollout-max-response-len 128 --seq-length 2048 \
  --ouro-reward-function vime_plugins.ouro.reward.boxed_answer \
  --ouro-eval-data docs/validation/ouro-thor-20260930/raw/eval.jsonl --ouro-eval-interval 4
# Reuse those flags and the same run directory; also pass:
# --ouro-resume --load /runs/ouro-graph/checkpoint --num-rollout 5 --use-checkpoint-opt-param-scheduler
```

Exact machine launch scripts, dataset, metrics, evaluation responses, successful
training logs, completion markers and hashes are retained alongside this report.
Ruff, Python compilation and shell syntax checks passed. The recurrent graph CUDA
unit test passed; pytest is absent from this existing environment. Model downloads
and development overlapped. Validated model weights and optimizer checkpoints
were removed after their evidence was saved; cleanup records are in `raw/`.

The duplicate mirror download also completed naturally and passed the same SHA256
check before deletion. The task model directory is empty; see
[final cleanup](raw/mirror-cleanup.json). No process was killed.
