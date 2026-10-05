# Native held-out evaluation

The native `vllm-rlt` backend can now evaluate complete held-out datasets between
policy updates. The parent (`61903fbef8a06eff4a0f7ce9daf42f5da792aa0b`, draft #12)
rejects `--eval-interval`. This change removes that entry barrier; it does not
establish reward convergence or a performance improvement.

## Evaluation contract

The evaluator uses VIME's JSONL/Parquet dataset loader, chat-template settings,
math reward and custom per-sample reward hooks. Every prompt and requested
completion is retained in its original order. Empty datasets, invalid lengths
and prompts outside the declared context budget fail the evaluation instead of
silently reducing the held-out population. Generation uses bounded batches of
`--rlt-max-num-seqs`; all batches must report the same runtime epoch, publication
version and physical-weight digest.

Each completion has a stable seed derived from the experiment seed, dataset
name, prompt index and completion index. It does not depend on training rollout
IDs, request IDs or evaluation time. Huginn records and replays the actual latent
seed. Evaluation accepts separate temperature, top-p, top-k, response budget and
token stops; the full-vocabulary training restriction is unchanged. Generated
stop tokens remain in the trace. String stops, repetition penalties, minimum
length, group rewards, multimodal/tool/server execution and partial-dataset early
stopping are rejected by this initial text-only path.

Native evaluation uses completed-update indices: `eval_0.pt` is the initial
policy, and `eval_1.pt` follows the first update. Evaluation-only resume uses the
restored cursor. The publication version remains a separate field: an initial
weight publication may already be version one. Per-dataset completion counts
are passed to pass-rate logging rather than replaced by the global default.

The native sampler uses request-local seeds. Tests with the built-in math reward
also preserve Python and Torch global RNG state. An arbitrary custom reward hook
must manage its own external state and randomness; the evaluator cannot make
that guarantee on its behalf.

Usage and the supported argument profile are documented in the
[backend README](../vime/backends/vllm_rlt_utils/README.md#held-out-evaluation).
The short example remains a lifecycle probe, not a frozen convergence recipe.

## Validation

The two final, disjoint suites pass **508 CPU checks**, with six existing CUDA
checks skipped and three known `megatron.training` factory checks deselected
because Triton is unavailable in this existing CPU environment. The 506-case
release and two argument checks have distinct test identities. Earlier subset
runs overlap and are not added to that total.

| Gate | Result | Actual coverage |
| --- | --- | --- |
| Complete native evaluation | 17 passed | Real tiny Ouro, Nanbeige and Huginn engines; multiple datasets, stable seeds, overrides, rewards, order and policy-cohort checks |
| Update index and metric grouping | 5 passed | Actual `train.train` control flow and Torch debug files; actor/transport doubles; initial evaluation is not overwritten |
| Evaluation interleaved with learning | 3 passed | Real tiny native sampling, MCore RLTT loss, AdamW, weight publications and restored objects |
| Resolved evaluation arguments | 2 passed | Actual dataset config builder and native validator; separate evaluation top-p/top-k accepted, training restriction retained |
| Complete final CPU regression | 508 passed | The above cases and inherited tests; six CUDA skips and three factory deselections are not passes |

Each learning case performs three nonzero AdamW updates using fresh native
completions and output token-parity training rewards. A paired run inserts math
held-out evaluation after every update. Training tokens, rewards, request seeds,
losses, gradients, parameter deltas, physical publication digests, final tensors
and Adam state match the no-evaluation run exactly.

After update two, a real Torch checkpoint is restored into newly constructed
actor, frozen reference, optimizer and native engine objects. Held-out outputs
match the saved policy; update three and the final held-out evaluation match the
uninterrupted run exactly. These are fresh objects in the same test process,
not independent worker-process or distributed-checkpoint recovery.

| Tiny family | Nonzero updates | Training with/without evaluation | Restored next update and held-out samples |
| --- | ---: | --- | --- |
| Ouro | 3 | Exact | Exact |
| Nanbeige | 3 | Exact | Exact |
| Huginn | 3 | Exact | Exact, including latent seeds |

The engine and loss are real; Ray calls and tokenization are test doubles.
Fixtures use random tiny weights, numeric-token prompts and a test decoder.
Their held-out reward values are validation data, not reasoning quality. They
cannot support a learning or convergence claim. Full raw training/evaluation
records are in [cycles.json](../benchmarks/results/native-evaluation-cpu/cycles.json).

## Performance comparison

| Scope | Unchanged parent #12 | This change | Parent/current speed ratio |
| --- | --- | --- | --- |
| Native periodic held-out evaluation | Rejected at argument validation | CPU component path passes | Undefined: parent has no executable evaluation path |
| Official-model full RL | NOT_RUN | NOT_RUN | — |
| Peak CUDA memory under matched load | NOT_RUN | NOT_RUN | — |
| Independent-seed reward time-to-quality | NOT_RUN | NOT_RUN | — |

No timing comparison is inferred from pytest duration. Actual evaluation adds
work to a training run and must be included in the formal wall-time and GPU-hour
budget. A matched numerical oracle, complete Ray/MCore/CUDA lifecycle, official
tokenizer/checkpoint evaluation and at least three independent training seeds
remain required before the main quality comparison.

## Reproduction and provenance

Runtime: existing Python 3.12.14, Torch 2.13.0 and pytest 9.1.1, one OpenMP thread.
Native RLT is `7f367ee02abb5b8c700a705062fbe98842338d21` (runtime unchanged from
`7d6e08e`). MCore is `1dcf0dafa884ad52ffb243625717a3471643e087` with the standard
VIME patch SHA256 `6fa39fdfdac8dae6b9bb3e44014df6766d774e0968ac07d6025a6b40f211aa98`.
No dependency was installed, updated or replaced.

Focused commands in an existing compatible environment:

```sh
PYTHONPATH=.:../rlt OMP_NUM_THREADS=1 pytest -q \
  tests/rollout_backends/test_native_rlt_evaluation.py \
  tests/rollout_backends/test_native_eval_arguments.py \
  tests/test_native_eval_schedule.py \
  tests/plugins/test_native_heldout_training.py
```

The retained [validation manifest](../benchmarks/results/native-evaluation-cpu/validation.json)
records tested source hashes and disjoint-suite totals; `SHA256SUMS.json` covers
the raw XML/log files and records. Local validation used the existing resident
Python harness to avoid evicted filesystem modules without changing packages.
