# Opt-in MCore prefix learner schedule

`--rltt-prefix-wave-size N` with positive N selects an Ouro RLTT prefix schedule
inside the existing `train_one_step`; zero preserves the ordinary MCore route.
The first candidate requires one actual learner rank, FP32 parameters, real
MCore DDP, unsharded non-overlapped gradients and eager execution. Rematerialized
prefix scheduling and mixed precision are not enabled in this candidate.

After the existing gradient zeroing and hook, the schedule builds the current
logical-step plan. Each group's prefix enters through DDP `forward`, so MCore's
real parameter hooks accumulate suffix and prefix contributions into
`main_grad`. Packed suffix waves consume global logical weights and the frozen
reference scores already in the rollout data. Optimizer loss scaling happens
once per suffix; boundary adjoints carry that scale into a single prefix
backward. Groups execute sequentially and release their graphs before the next
prefix. MCore gradient finalization runs once after all groups.

The existing optimizer, scheduler, metrics, gradient cleanup, publication and
checkpoint paths remain responsible for the update. The prefix route advances
the original iterator by the original microbatch count only after a successful
optimizer/scheduler step. Invalid gradients raise without committing that
cursor. Neither wave count nor prompt grouping changes samples-seen increments.

The provider's `prefix_only` forward entry accepts one nonempty prompt with
canonical positions and causal attention. It rejects mixed packed/readout/trace
arguments. The scheduler rejects an unwrapped module, multiple learner ranks,
BF16/FP16, overlap/distributed-optimizer modes, CUDA Graph scope and existing
rematerialization switches. These checks delimit the candidate being qualified;
they do not establish that full GPU execution has passed.

## Validation and pending GPU gate

The broad local regression has **195 CPU passes, 2 CUDA skips and 12 strict
expected BF16 numerical failures** in 85.51 s. This includes 11 new provider and
argument cases, 14 existing native argument cases and the 170-pass logical-step/
RLTT/prefix regression. The provider entry matches every trainable gradient
exactly against its direct method. The actual CUDA tests are skipped before
framework startup, rather than replaced with a simulated DDP wrapper.

`tests/integration/test_native_prefix_train_step.py` is prepared to initialize
one NCCL rank and run the actual MCore DDP, FP32 optimizer wrapper, AdamW,
parameter scheduler and `train_one_step`. It compares two updates in each
logical reduction, captures real finalized `main_grad` values before cleanup,
checks all parameters and Adam moments, requires one finalize per update and
checks cursor/samples-seen counts. It also checks that gradient cleanup occurred.
This is a tiny fixed-trace test, not online generation, official model reward
qualification or a publication/checkpoint recovery test.

In an admitted GPU window with the complete existing environment, its command is:

```bash
PYTHONPATH=.:RLT_SOURCE:tests/plugins:MCORE_SOURCE \
  python -m pytest tests/integration/test_native_prefix_train_step.py \
  -q -o junit_family=legacy --junitxml=prefix-mcore-cuda.xml
```

The environment and shared device window have not been admitted for this run.
The fresh H20 lacks the requested `0z5a` environment and full learner dependencies
at the recorded inspection. No installation or base-environment substitution
was performed for this task. Other tasks' later environment changes do not
qualify this run. The existing three-update qualification profile requests
rematerialization, so it must not be silently relabeled as this B-only profile.
Freeze a matching runtime/profile after the required B/C work.

| Required comparison | Baseline throughput | Prefix throughput | Speedup | Peak-memory reduction | Reward convergence |
|---|---:|---:|---:|---:|---|
| Real MCore CUDA two-update fixture | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_ESTABLISHED |
| Official Ouro learner / full online RL | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_ESTABLISHED |
| RLTT / FlashLoop / slime / Megatron E2E | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_ESTABLISHED |

The broad CPU run precedes the final explicit rank/recompute guards and GPU
fixture model-type setup. A final focused provider/argument/collection run
checks the finalized sources separately; both runs overlap. Raw logs/XML,
source hashes and validation scope are in `benchmarks/results/rltt-prefix-schedule/`.
B remains incomplete until the actual wrapper and official-model short run pass;
C and the formal A campaign remain subsequent work.
