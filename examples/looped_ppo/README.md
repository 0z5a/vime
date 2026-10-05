# Fixed-depth recurrent policy training

The shared launcher runs VIME's standard Ray entry with native vLLM-RLT rollout.
[Nanbeige](../nanbeige) documents its model-specific command and all four algorithms.
[Huginn](../huginn) uses the same four objectives and replay contract.
Ouro also uses this entry with `--model /models/Ouro-1.4B` and checkpoint revision
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.

Use `--algorithm ppo`, `grpo`, `dppo`, `flow-dppo`, or `rltt`. PPO and GRPO reuse the existing
clipped policy loss. DPPO keeps rollout behavior scores separate from proximal
scores recomputed by the training provider, and applies VIME's existing bounded
importance correction. Flow-DPPO freezes full-vocabulary old-policy scores and
applies the categorical divergence gate. Actor/critic roles use GAE for PPO,
DPPO and Flow-DPPO; GRPO uses an actor and prompt-group reward normalization.
RLTT uses the explicit recurrent weighted-logprob objective, logical-batch
reduction and frozen initial reference described in [its contract](../../docs/looped_readout_contract.md).

Install the [pinned RLT engine](../../vime/backends/vllm_rlt_utils/README.md) and
Megatron `1dcf0dafa884ad52ffb243625717a3471643e087` with
[VIME's patch](../../docker/patch/latest/megatron.patch) in an isolated environment.
Use an existing two-GPU Ray cluster; `--ray-address host:port` selects its address.
The launcher propagates its Python executable and source path to workers.
For an existing single-GPU cluster, GRPO/RLTT can select `--colocate-resident`:
learner, frozen reference and rollout remain resident together. The launcher
disables both offload paths and uses the shared placement described in the
[backend contract](../../vime/backends/vllm_rlt_utils/README.md#single-gpu-resident-profile).
Capacity and actual Ray/CUDA training for this new profile are still unqualified;
it does not turn a small device into a sharded or offloaded execution path.
Math JSONL rows need `prompt` and `label` fields. Prompts may be strings or
structured chat messages when `--apply-chat-template` is enabled. For example:

```json
{"prompt":"Calculate 7-4. Reply with only \\boxed{answer}.\n###Response\n","label":"3"}
```

Defaults use SGD without momentum, FP32 gradient accumulation, four prompts,
128 prompt tokens, 48 response tokens and full-vocabulary sampling. Nanbeige
and Ouro use FP32; Huginn uses FP16. `--precision fp16` selects fixed loss scale
128. Local MCore checkpointing
restores optimizer parameter groups, and FP16 master parameters and loss scale.
`--recompute` checkpoints each physical block. `--rlt-cuda-graphs` selects the
engine's graph path; eager execution is the default. Cache capacity accounts
for all recurrence planes of two active sequences.

`--rollout-batch-size` counts prompts and `--n-samples-per-prompt` counts completions.
GRPO defaults to four completions and requires at least two. Its reward normalizer
runs before samples are packed; a second batch-wide advantage whitening is omitted.
PPO/DPPO use one optimizer step per rollout. Flow-DPPO uses two steps, each with
half the completion batch. Extra arguments pass through to VIME.

Use a separate output directory with `--stop-after 2`, then a fresh process with
`--resume` for interrupted/resumed comparisons against three uninterrupted updates.
Actor and critic checkpoints are separate; GRPO saves only the actor. The dataset
cursor and policy version continue, and restored weights publish before generation.
Compare full model/optimizer/scheduler/RNG state, tokens, rewards and traces.
The launcher enables deterministic training and passes the cuBLAS workspace and
NCCL algorithm settings to Ray workers for this comparison.
Report end-to-end timing including rollout, rewards, optimization, full publications
and checkpoints, with source pins, dtype and resource contention.

Periodic held-out evaluation uses the same committed native policy through
`--eval-interval` and `--eval-prompt-data` or `--eval-config`. It does not consume
the training dataset cursor. Initial and post-update results retain distinct
completed-update indices, and the output samples retain their policy and seed
traces. See the [native evaluation contract](../../vime/backends/vllm_rlt_utils/README.md#held-out-evaluation).
Existing short update/resume checks do not establish reward convergence.

The [RLTT source input profile](../../docs/rltt_source_contract.md) provides the
same audited public MATH split in native JSONL and original-converter Parquet.
It uses the source instruction and default chat options, separately from the
earlier thinking-enabled profile. Pass its JSONL with `--apply-chat-template`
and no thinking override. Its full training split and explicit 1,024-token
profile have different manifests; choose one before comparing systems.
