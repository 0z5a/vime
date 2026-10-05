# Native vLLM-RLT rollout backend

`--rollout-backend vllm-rlt` selects a native Ray-owned engine instead of an HTTP
server. Use `vime.rollout.vllm_rlt_rollout.generate_rollout` as the rollout
function. The ordinary sample, reward, dataset and training contracts remain in
use; backend imports stay lazy for other rollout engines.

The initial implementation supports Ouro, Nanbeige and Huginn at the checkpoint's
fixed full depth, full-vocabulary sampling and full physical-weight publication
through disk. Dedicated rollout resources remain the default. An opt-in resident
profile places one learner and one rollout engine on the same GPU, as described
below. Rollout offload/recovery, external HTTP and truncated training sampling
require additional backend contracts and are rejected by validation.

Install the pinned engine in VIME's isolated environment:

```bash
python -m pip install "git+https://github.com/0z5a/vllm-rlt.git@d2a358933393f25d74dd2bdd1068a741cd5d9226"
```

Supply the same commit with `--rlt-engine-revision`.

Pin the checkpoint and installed RLT source, then supply their identities with
`--rlt-model-revision` and `--rlt-engine-revision`. These flags record provenance;
they do not verify a checkout or download. The seeded Huginn runtime requires
the `like-init-cpu-f32-v1` engine profile. Persist PyTorch version, dtype and
host CPU architecture with its seed/position trace when comparing fresh runs.

Publication pauses generation, checks every safetensors shard, and calls RLT's
public start/update/finish weight transaction. Generation resumes only after
all physical parameters commit. Repeating a committed version requires the
same content digest. Huginn's fixed RoPE buffer is verified, and its LM head
alias must equal the tied embedding. Every sample carries the committed policy
version, digest, runtime epoch, model/engine identity and recurrence trace.

A fresh resume supplies `--rlt-start-version` before publishing the restored
actor. Standard training checks the engine's committed version after every
publication. Owned engines close and owned actors exit normally when training
finishes; the existing Ray cluster remains available.

Defaults use BF16 training dtype, Triton attention and eager execution.
`--rlt-attention-backend torch` selects the Torch reference path;
`--rlt-cuda-graphs` opts into graphs. Set cache capacity with `--rlt-kv-blocks`
and admission concurrency with `--rlt-max-num-seqs`.

## Single-GPU resident profile

For GRPO or RLTT, add `--colocate-resident` to `examples/looped_ppo/run.py`.
The launcher selects one physical GPU, `--colocate`, `--no-offload-train` and
`--no-offload-rollout`. For a direct `train.py` invocation the equivalent flags
are:

```bash
--colocate --no-offload-train --no-offload-rollout \
--actor-num-nodes 1 --actor-num-gpus-per-node 1 --num-gpus-per-node 1 \
--rollout-num-gpus 1 --rollout-num-gpus-per-engine 1
```

Both model roles remain resident in separate Ray actors. The rollout actor
reserves 0.5 CPU/GPU and the existing learner reserves 0.4, fitting the same
one-CPU/one-GPU placement bundle. The rollout's physical-device offset is zero.
Ray fractions express scheduling resources; they do not partition memory or
reserve half the device's memory. Model copies, frozen reference, gradients,
optimizer state, KV, activations and graph pools must all fit together.

This profile uses the ordinary synchronous `train.py` order: complete rollout,
update the learner, save when scheduled, publish every physical weight and
verify its version, then evaluate or generate again. The native publication
barrier and held-out evaluator are unchanged. Actor/rollout overlap, critic,
release-train, fault recovery and offload are outside this resident profile.
Bare `--colocate` still resolves to VIME's offload defaults and is rejected by
the native validator; choose the explicit resident flags.

Current coverage verifies configuration, both placement-resource budgets and
unchanged CPU numerical/update contracts. Actual shared-device Ray/CUDA startup,
memory capacity, complete training and fresh-worker recovery still require a
reserved compatible device. This profile is not yet a qualified performance or
quality result. See [native_colocation_results.md](../../../docs/native_colocation_results.md).

## Held-out evaluation

The standard `--eval-interval` and `--eval-prompt-data`/`--eval-config` paths now
support native text evaluation. Datasets use VIME's existing JSONL/Parquet loader,
chat-template settings and per-sample reward hooks. Evaluation supports its own
temperature, top-p/top-k, response length and token stops; training still uses
full-vocabulary sampling. String stops, minimum generation lengths, repetition
penalties and custom/server generation paths are explicitly unsupported.

Every configured prompt is evaluated. Empty or over-limit prompts fail instead
of silently reducing the held-out denominator. Generation is batched through the
same committed engine, and crossing a policy epoch/version/digest fails the entire
evaluation. Evaluation reads its own datasets without consuming the training data
source. Seeds derive from the run seed, dataset name, prompt position and completion
position under `native-heldout-v1`; they remain fixed across updates and resume.
Huginn records its actual seed and latent profile for every completion.

Native evaluation filenames and metric steps count completed updates: `eval_0`
is the initial actor, `eval_1` follows the first update, and so on. The committed
publication version is a separate trace field; it need not equal that step index.
Evaluation-only resume uses the restored training cursor. Per-dataset completion
counts are used when computing pass rates.

Add these flags to the existing qualified model training command:

```bash
--eval-interval 1 \
--eval-prompt-data heldout /data/heldout.jsonl \
--eval-temperature 0 \
--eval-max-response-len 48 \
--n-samples-per-eval-prompt 1 \
--save-debug-rollout-data '/runs/experiment/rollouts/{rollout_id}.pt'
```

Use a held-out dataset distinct from training, freeze both inputs and the source
revisions, and size the native cache for the declared prompt/response budget.
The example only selects the evaluation interface; it is not a validated
convergence configuration. CPU numerical and recovery evidence is in
[native_evaluation_results.md](../../../docs/native_evaluation_results.md).
