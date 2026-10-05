# Native resident colocation

The dedicated native backend previously rejected `--colocate`. Simply removing
that check would leave the rollout actor requesting the entire 1-CPU/1-GPU
placement bundle, while the learner requests another 0.4 of each. The learner
could not be placed, and the native server would still report a separate-device
offset.

`--colocate-resident` in the looped recipe now selects a single learner and
rollout engine on one physical GPU. The native rollout reserves 0.5 CPU/GPU in
the same bundle as the existing 0.4 learner, and reports offset zero. Both roles
remain resident; Ray fractions are resource-accounting tokens, not fractions of
GPU memory. Dedicated placement remains unchanged.

The initial profile accepts GRPO/RLTT without a critic. Both offload switches
are explicitly disabled. Bare `--colocate` retains VIME's normal offload defaults
and is rejected by the native validator. Multi-rank learner placement, critic,
release-train and fault recovery require separate qualification and remain
outside this profile. The synchronous training order, losses, full physical
publication transaction, policy-cohort checks and held-out evaluator are unchanged.

## Validation

The final CPU regression passes **580 distinct tests**, skips six CUDA checks,
and deselects three previously identified factory checks that need unavailable
Triton. The focused 153-test run is a subset, not additional evidence to add to
that count. Existing three-family numerical, sampled-update, held-out isolation
and fresh-object Adam recovery tests are included. These remain tiny CPU
component checks with the previously documented transport/tokenizer doubles.

| Gate | Result | Scope |
| --- | --- | --- |
| Common and native argument resolution | Pass | Actual validators preserve explicit resident flags; implicit offload is rejected |
| Shared bundle budget | Pass | Actual deployment and training allocator emit 0.5 + 0.4 CPU/GPU on one bundle; Ray launch is captured |
| Dedicated and rollout-only deployment | Pass | Original whole-GPU reservation and device offsets retained |
| Family, algorithm and resume recipe matrix | Pass | Captured commands preserve model/depth/precision/reference/resume fields; critic algorithms reject resident mode |
| Entire final CPU regression | 580 passed | Six CUDA skips and three factory deselections are not passes |
| Actual single-device Ray/CUDA lifecycle | NOT_RUN | Requires an existing compatible environment and complete resource handoff |
| Official held-out reward convergence | NOT_RUN | Independent training seeds and frozen quality protocol remain required |

The initial focused run had one test-fixture failure: the raw-argument placement
test omitted `rollout_external`, normally filled by common validation. The fixture
now supplies that field. Production behavior was not changed to accommodate the
fixture. Both the failed receipt and completed rerun are retained.

## Resource and performance comparison

| Scope | Parent #13 | Resident candidate | Parent/current speed ratio |
| --- | --- | --- | --- |
| Native single-GPU resident launch | Rejected | Placement/argument checks pass | Undefined until actual runs |
| Matched official full-RL wall time | NOT_RUN | NOT_RUN | — |
| Whole-device peak memory | NOT_RUN | NOT_RUN | — |
| Held-out reward time-to-quality | NOT_RUN | NOT_RUN | — |

No speed or memory-saving claim follows from using one placement bundle.
Actor, reference, rollout, gradients, optimizer state, activations, KV and graph
pools must fit simultaneously, including allocations in both worker processes.
The formal run must measure actual device ownership, total allocated GPU-hours,
whole-device memory and the complete rollout/update/publication/evaluation/save
critical path. Resident and offloaded profiles are separate comparisons.

## Reproduction

Use an existing compatible environment and a reserved Ray cluster. Add
`--colocate-resident --algorithm rltt` or `--colocate-resident --algorithm grpo`
to the ordinary [looped recipe](../examples/looped_ppo/README.md), with pinned
model/engine inputs and a held-out dataset. No package change is part of this
delivery. The recipe's three-update SGD defaults remain a lifecycle probe,
not an optimizer or budget selected for convergence.

Local evidence uses Python 3.12.14, Torch 2.13.0, pytest 9.1.1 and one OpenMP
thread. RLT source is `fd993ec5512904b68e16f4d541682c076c487b5d` (runtime unchanged
from `7d6e08e`); tested MCore is `1dcf0dafa884ad52ffb243625717a3471643e087` with the
standard VIME patch SHA256 `6fa39fdfdac8dae6b9bb3e44014df6766d774e0968ac07d6025a6b40f211aa98`.

Raw receipts, exact test-file list and source hashes are retained under
[`benchmarks/results/native-colocation-cpu/`](../benchmarks/results/native-colocation-cpu/).
The CPU tests capture Ray scheduling calls; they do not establish that workers
started, shared the same physical device or fit its memory.
