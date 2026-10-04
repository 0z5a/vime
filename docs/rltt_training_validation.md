# Fixed-depth RLTT training bridge

`--loss-type rltt_loss` connects response-only per-loop readout to the standard
MCore model forward and loss callback. `examples/looped_ppo/run.py --algorithm rltt`
selects this path for Ouro, Nanbeige and Huginn. It is an opt-in objective
reimplementation, not a reproduction of the original RLTT stack or a qualified
official-model training result.

## Execution contract

- Native `vllm-rlt` rollout, checkpoint full depth, full-vocabulary sampling,
  one learner rank, TP=PP=CP=1, grouped GRPO advantages.
- The frozen initial reference is loaded even when the KL coefficient is zero.
  `--ref-update-interval` is rejected. On resume, the actor checkpoint changes;
  the reference path remains the initial checkpoint.
- Actor and reference replay use the same temperature and streamed FP32-product
  readout profile. Reference and old-policy evaluation retain terminal scores;
  training exposes all supervised loops. Huginn replays each recorded latent seed.
- The loss is unclipped weighted-logprob PG plus terminal sampled k3 and optional
  terminal entropy. Credit is uniform or progressive with fixed nonnegative alpha.
  Learned exit-PDF production and online full-vocabulary KL remain unfinished.
- `response_mean` averages each contributing response before averaging responses;
  `token_mean` averages all contributing tokens. Masked tokens contribute zero.
  Weights are calculated once for the actual logical-step sample indices, then
  sliced across microbatches. An entirely masked logical step is rejected.
- The callback cancels MCore's microbatch divisor, and compensates VIME's logging
  divisor separately. Changing microbatch boundaries preserves the declared loss
  and gradients. No DP/CP reduction equivalence is claimed by this single-rank path.

The existing recipe defaults remain a short lifecycle probe: FP32 for RLTT, SGD,
three updates, short responses, and a single seed. These are not the original
RLTT optimizer or a convergence recipe. Native held-out evaluation is still
unavailable through VIME's dataset evaluator. Do not use this probe as the main
quality experiment.

## Parent training-bridge validation result

This section records the immutable `fba761f` training-bridge delivery. Subsequent
packed replay validation is tracked separately in [packed replay results](packed_replay_results.md).

The final CPU run reports **257 passed, 3 failed**. All three failures occur while
importing `megatron.training`, which requires unavailable Triton in this existing
runtime. Their factory/HF-loader checks did not execute. No dependency was installed
or replaced. The full Ray launcher, `train_one_step`, GPU DDP/optimizer, and typed
distributed checkpoint lifecycle have not run for RLTT.

| Check | Passed | Evidence boundary |
| --- | ---: | --- |
| Readout/objective numerical oracles | 50 | FP64 finite differences, FP32/BF16, entropy/full KL, static/differentiable credit |
| Provider response/gradient comparisons | 42 | Real MCore/native tiny models; independent depth oracle, all parameters |
| Prior provider and objective regressions | 19 | Six actor/critic plus 13 update/publication/resume cases |
| RLTT packed loss and contract | 23 | Three families, both reductions, recompute on/off, three microbatch partitions; real MCore `forward_step_calc_loss` |
| Native sampled-reward update/resume | 3 | Three real tiny native engines, output-derived reward, three AdamW updates and exact fresh resume |
| Recipe, arguments, placement | 120 | Unit tests; CLI/placement use captured calls, not real Ray workers |
| Factory/HF-loader checks | 0 (3 failed) | `ModuleNotFoundError: triton`; retained as failures |

The native cycle uses two prompts, eight fresh completions per prompt, and a
token-parity reward calculated from the generated answer. Each update has mixed
reward groups, finite nonzero gradients, and nonzero parameter deltas. Every policy
is physically exported and loaded by the native engine. A new actor, frozen
reference, optimizer and engine restore step two and reproduce step three exactly:
tokens, rewards, seeds, loss, gradients, actor/reference tensors, Adam moments and
publication digest. This is a CPU component lifecycle, not a mathematical reasoning
benchmark, a multi-seed study, or reward convergence.

| Tiny family | Maximum rollout/replay score error | Gradient norm range | Parameter-delta L2 range | Fresh step-three replay |
| --- | ---: | ---: | ---: | --- |
| Ouro | 2.38e-7 | 1.057–1.585 | 0.01645–0.02912 | Exact |
| Nanbeige | 4.77e-7 | 0.557–0.833 | 0.01894–0.03059 | Exact |
| Huginn | 1.91e-6 | 2.478–3.177 | 0.02864–0.04783 | Exact |

Raw update records: [native-updates.jsonl](../benchmarks/results/rltt-tiny-cpu/native-updates.jsonl).
The [validation manifest](../benchmarks/results/rltt-tiny-cpu/validation.json)
records the raw JUnit hash, source hashes, exact failures and runtime identity.

## Speed and memory comparisons

No new matched full-RL timing or peak-memory result exists for this bridge.
The inherited, separately scoped readout measurements remain:

| CPU head case | Dense forward+backward (ms) | Streamed (ms) | Dense/streamed speed ratio | Saved intermediate bytes, dense → streamed |
| --- | ---: | ---: | ---: | ---: |
| N64/V1024/H128, selected score | 0.316 | 0.765 | 0.414× | 262144 → 1024 |
| N64/V1024/H128, + entropy/KL | 0.656 | 1.478 | 0.444× | 786432 → 1024 |
| N256/V4096/H256, selected score | 5.186 | 7.179 | 0.722× | 4194304 → 4096 |
| N256/V4096/H256, + entropy/KL | 10.284 | 15.720 | 0.654× | 12582912 → 4096 |
| Official-model full RL, high load/concurrency | NOT_RUN | NOT_RUN | — | — |
| Reward time-to-quality | NOT_RUN | NOT_RUN | — | — |

All four measured CPU cases are slower. Saved autograd intermediates exclude
input/parameter storage and are not peak GPU memory. Protocol, raw data and limits
are in [streamed readout results](streamed_readout_results.md).

## Runtime provenance and remaining gates

CPU: existing Python 3.12.14, Torch 2.13.0, pytest 9.1.1, one OpenMP thread.
Native RLT: `46145be03fc053da147f32c45f2a23893313cb70`.
MCore: `1dcf0dafa884ad52ffb243625717a3471643e087` plus VIME's standard
`docker/patch/latest/megatron.patch`, SHA256
`6fa39fdfdac8dae6b9bb3e44014df6766d774e0968ac07d6025a6b40f211aa98`.
All 2232 files in the existing tested MCore tree independently match the retained
patched-source manifest, with no additional private patches. This is distinct
from the separately frozen upstream Megatron framework-baseline archive.

Before performance or quality conclusions: qualify official-checkpoint G0 and
BF16 arithmetic, run the actual Ray/MCore CUDA lifecycle and typed resume, expose
held-out evaluation, freeze the optimizer/budget/reward and source-matched baseline
profiles, then run matched systems measurements and independent training seeds.
