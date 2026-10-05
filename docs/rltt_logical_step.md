# Logical RLTT step and prefix loss scaling

The ordinary MCore loss returns a logical loss multiplied by the number of
microbatches; MCore divides it back before backward. Passing that return value
directly to prefix backward would amplify gradients. `logical_rltt_loss` now
returns the unchanged objective and detached statistics without schedule or
logging factors. The existing `megatron_loss` adapter preserves both factors
for the ordinary MCore path. Prefix backward accepts the optimizer's scaling
function once per suffix loss; the accumulated boundary adjoints are already
scaled and receive no second factor during prefix backward.

`logical_step.plan_step` selects the current step's actual `DataIterator`
indices without advancing its cursor. It retains the original microbatches,
sample IDs and GRPO group IDs, computes token/response weights over the whole
step, then groups compatible prompts into bounded suffix waves. Native rollout
transport now carries the original `group_indices` alongside `sample_indices`.
Regrouping never recomputes advantages or changes group membership.

The first plan supports canonical causal positions, recorded fixed-depth Ouro
and a caller-supplied current actor generation. Behavior policy versions are
not substituted for that generation. Prompt tokens, model revision and depth
must match; unsupported latent/position/mask contracts are rejected. Empty or
fully masked responses remain in the plan. A completely zero-contribution step,
duplicate sample/index, incomplete step or stale actor/cursor identity fails.
The existing prefix replay separately detects parameter mutation/replacement.

This increment supplies B1/B2 building blocks. It does **not** yet replace
`train_one_step` or establish B3's full DDP/finalize/optimizer contract.
There is no new optimizer, scheduler, persistent prefix cache or GPU campaign.

## Numerical results

The final suite has **170 CPU passes and 12 strict expected numerical failures**
in 49.51 s. Twenty-nine passes and the twelve BF16 failures are new; the other
141 passes exercise existing RLTT and three-family prefix behavior. All 170
MCore/deprecation warnings and earlier failing runs are retained.

The new oracle uses five ragged responses from two prompts, including an empty
response and a nonempty all-zero loss mask. It tests both logical reductions,
one/five/two original microbatches with reordering, suffix waves of one/two/three,
and optimizer scales one/eight. A dense independent-depth reference uses the
readout's specified FP32 linear products. Both the unchanged MCore
`forward_step_calc_loss`/`backward_step` functions and the prefix route run two
AdamW updates against a distinct frozen reference. Full parameter gradients,
losses, logging normalization and nonzero parameter deltas are checked.

| CPU component | Two-update cases | Maximum gradient relative L2 | Maximum parameter-delta error | Result |
|---|---:|---:|---:|---|
| FP32, ordinary MCore loss and prefix vs dense | 12 | 5.38e-7 | 1.20e-7 | PASS |
| BF16, ordinary MCore loss and prefix vs dense | 12 | 2.74e-2 | 1.77e-3 | NOT_QUALIFIED |

All 24 BF16 update observations fail the original delta gate; six also fail
the gradient gate after the first differing update. The strict expected-failure
marker catches only the explicit numerical-gate exception after both updates
have been recorded; unexpected exceptions and invariant failures still fail the
suite. Tolerances were not widened. BF16 failures affect the CPU Adam trajectory
and do not establish behavior of a CUDA mixed-precision MCore optimizer. The
first failing run also used BF16 dense-head products instead of the specified
FP32 readout products; correcting that oracle did not qualify BF16. The original
failures and corrected-oracle source are retained.

| Required comparison | Baseline throughput | Prefix throughput | Speedup | Peak-memory reduction | Reward convergence |
|---|---:|---:|---:|---:|---|
| Official Ouro single-learner RLTT | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_ESTABLISHED |
| RLTT / FlashLoop / slime / Megatron E2E | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_ESTABLISHED |

Evidence: `benchmarks/results/rltt-logical-step/` contains raw logs/XML,
per-case two-update measurements, the corrected intermediate oracle and hashes.
The retained qualification runtime manifest from draft #22 intentionally pins
older runtime code and must be frozen again before qualifying this runtime.
