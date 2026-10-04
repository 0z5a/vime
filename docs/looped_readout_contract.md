# Response-only loop readout and objective primitives

This is an explicit fixed-depth replay entry point. Existing dense provider
`forward` calls keep their output contract. The opt-in `rltt_loss` training branch
selects the response-only output through the standard MCore model call; its
Ray/CUDA lifecycle is not yet qualified. See [training validation](rltt_training_validation.md).

`response_log_probs` includes the last prompt position, which predicts the first
response token. It returns `[response_tokens, supervised_loops]`. Ouro normalizes
at every loop; Nanbeige retains its configured loop-final normalization; Huginn
applies its coda and final normalization for each supervised loop without feeding
that coda output back into the recurrent core. Huginn callers must supply the
recorded latent seed. There is no top-k/p processing or actual-depth replay here.

## Readout arithmetic

Vocabulary tiles compute log-normalizers with log-sum-exp. The backward rebuilds
each tile to obtain selected-score, entropy and full-vocabulary KL gradients.
It retains response hidden states, input weights and row-sized reductions, not
token-by-vocabulary logits. The full parameter gradient still exists; this does
not shard the head, optimizer or model. Only first-order differentiation is supported.

Products use FP32 for FP32/BF16 inputs and FP64 for FP64 inputs. The dense oracle
uses the same arithmetic. This is a declared arithmetic profile: it does not prove
parity with an existing BF16 head GEMM or processed rollout scores. Official-model
cross-engine qualification remains required. The reference policy is frozen.

## RLTT objective contract

`rltt_loss` implements `-advantage * sum(credit * loop_logprob)` plus terminal KL;
it has no PPO ratio or clipping. Loop credit can be uniform, progressive or an
explicit differentiable tensor. `credit_stopgrad` controls its gradient independently
of whether a producing gate's parameters are trainable. Advantages are detached.
Model-specific exit-PDF production is not implemented by this helper.

This is an **objective reimplementation**, not a reproduction of the original
RLTT stack. Sources are pinned to
[RLTT 0618985](https://github.com/jonwill8/RLTT/tree/06189850bb23a1b1b715ad29768e68f36b1c4a19).

| Contract | Visible source / paper | This primitive |
| --- | --- | --- |
| Policy loss | [`rltt_algos.py`](https://github.com/jonwill8/RLTT/blob/06189850bb23a1b1b715ad29768e68f36b1c4a19/rltt_experiments/verl_rltt/rltt_algos.py) sums weighted log-probabilities | Same unclipped expression; not a log-mixture |
| Reduction | Visible actor: microbatch token mean, then dynamic sample-count scaling or static accumulation divisor | Explicit logical-batch `token_mean` or nonempty `response_mean`; compute weights once and slice, never renormalize a microbatch |
| KL | [`rltt_actor.py`](https://github.com/jonwill8/RLTT/blob/06189850bb23a1b1b715ad29768e68f36b1c4a19/rltt_experiments/verl_rltt/rltt_actor.py) uses terminal sampled k3; generic helper instead aggregates loops | Terminal sampled k3 by default; supplied terminal full-vocabulary KL is a separate option |
| Reference | Visible actor falls back to old-policy scores if reference scores are absent | Frozen reference required, including at zero coefficient; no fallback |
| Credit gradients | Actor passes learned/exit weights onward; missing modified model source leaves internal behavior unresolved | Explicit stop-gradient option; loss and credit gradients tested |
| Paper objective | [v3 equations 4–7](https://arxiv.org/html/2602.10520v3) describe response averaging and terminal-policy KL | `response_mean` + streamed full KL supplies these primitive components; no complete paper-profile training result |
| Runtime / optimizer | Public config, batch-repeat semantics, optimizer and modified model require complete reproduction | No original runtime, optimizer, repeat count or training quality equivalence claimed |

Empty local partitions contribute a differentiable zero. A complete logical batch
must contain response tokens; `response_mean` excludes empty responses from its
denominator. Source-matched microbatch reduction must remain separately specified
if implemented; it is not interchangeable with these global denominators.

## Validation

111 CPU tests pass: 50 readout/objective oracles, 42 provider/independent-depth
comparisons, six existing actor/critic provider regressions and 13 existing
GRPO/PPO/DPPO/Flow-DPPO update/publication/resume regressions. Tests use real
MCore classes and native Ouro/Nanbeige/Huginn models, with small random weights.
Coverage includes FP64 finite differences; FP64/FP32/BF16 readout gradients;
uneven vocabulary tiles; entropy/full KL; empty response partitions; microbatch
partition invariance; differentiable credit; shared recurrent parameters; prompt
boundary alignment; Huginn coda and latent identity; recomputation on/off.

Shared recurrence and independent depth replays accumulate FP32 gradients in
different orders. Both dense and streamed heads are compared to independent depth
replays. Every trainable parameter is checked with `atol=1e-5, rtol=8e-5`; a separate
whole-gradient relative L2 bound of `2e-6` is required.

| Model | Streamed maximum absolute gradient error | Streamed relative L2 | Dense shared-recurrence relative L2 |
| --- | ---: | ---: | ---: |
| Ouro | 2.861e-6 | 1.327e-7 | 1.196e-7 |
| Nanbeige | 4.768e-7 | 9.548e-8 | 8.044e-8 |
| Huginn | 1.526e-5 | 2.853e-7 | 1.557e-7 |

These are maxima across the tested FP32 configurations, not official-checkpoint
error bounds. Three additional provider-factory checks could not run successfully
because the existing local MCore training import requires unavailable Triton.
No dependency was installed or replaced. The isolated readout timing and retained
intermediate-storage measurements, including slower cases, are in
[streamed_readout_results.md](streamed_readout_results.md).

GPU head optimization, official checkpoints, BF16 cross-engine G0, distributed
execution, full Ray/CUDA lifecycle timing and reward convergence remain unqualified.
The explicit recipe is available for qualification; it is not a measured speed or
quality result. Its initial scope uses static credit and terminal sampled k3;
the standalone full-KL primitive is not yet connected to that training recipe.
