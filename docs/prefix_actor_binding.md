# Bind a differentiable prefix to its producing actor

A prefix program previously acquired its parameter-version snapshot only when
`PrefixReplay` opened. A weight restore between prefix capture and replay could
therefore pass the freshness check. The same program also accepted a different
actor object or a frozen reference's output weight with a matching shape.

`PrefixProgram` now records its producing module, output weight and parameter
identity/version/dtype/device/gradient flags at construction. Replay validates
that binding before creating boundary leaves and checks the original versions
through the final prefix backward. Ouro and its Nanbeige subclass share this
capture path; Huginn records the same binding alongside its existing actual
latent identity. This adds no model copy or persistent cross-update cache.

Twelve regressions reproduce the previous acceptance of a different actor,
reference restore, restore round trip and reference readout across the three
families. The restores use actual PyTorch `load_state_dict` copies on tiny CPU
models. This is not a measurement of pinned snapshot transfers or the Ray actor
role-switch lifecycle. The final regression has **141 passes and 12 existing strict BF16 numerical
expected failures** in 152.36 seconds. The negative baseline has 12 failures
because all invalid bindings were accepted. The full existing prefix and
two-update numerical regressions are reported with the raw logs in
`benchmarks/results/prefix-actor-binding/`.

The actor's update generation still comes from the logical step. Behavior
publication versions and frozen reference scores remain distinct inputs; this
patch does not modify sampling, advantages or the RLTT objective. Actual MCore
and official-model GPU qualification are still pending, and the learner
schedule remains Ouro-only despite the three-family component checks.

| Comparison | Before | After | Speedup | GPU memory saving | Reward convergence |
|---|---|---|---|---|---|
| Invalid pre-replay actor bindings, tiny CPU | 12 accepted incorrectly | 12 rejected correctly | NOT_MEASURED | NOT_MEASURED | NOT_ESTABLISHED |
| Real MCore / official Ouro online RL | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_ESTABLISHED |

This is a correctness fix. It supplies no new throughput or convergence result.
