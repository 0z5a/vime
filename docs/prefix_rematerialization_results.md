# Joint prefix and suffix rematerialization

A response wave previously retained the complete Ouro suffix graph, while the
shared prompt retained its producer graph until the final accumulated prefix
backward. The new opt-in schedule checkpoints each complete suffix wave and the
prompt producer. Every physical-layer/loop K,V and first-response hidden boundary
remains available; the prefix adjoints are still accumulated before one producer
backward. This implementation supports one whole-depth segment, selected with
`rltt_loop_checkpoint == actor.readout_depth`. Layer/query subdivision and model
recompute remain rejected. It is the first C implementation, not the complete C
exploration matrix.

Tokens and labels are explicit checkpoint inputs. A regression test first
reproduced an incorrect replay when a response changed between forward and
backward; both suffix-token and final-label mutation now raise a version error.
Actor/readout versions are checked on replay as well.

## Numerical evidence

The CPU fixture compares the existing prefix schedule with joint replay across
12 configurations: terminal/all-loop readout, three ragged wave partitions, and
serial/packed suffixes. Each performs two nonzero AdamW updates with a distinct
frozen reference, masked response tokens and signed advantages. All trainable
gradients, parameters, updates and Adam moments retain the existing tolerances.

| Check | Result |
|---|---:|
| Full local regression | 158 passed, 4 CUDA skipped, 12 known strict BF16 xfails |
| Joint replay update observations | 24 |
| Maximum gradient absolute error / relative L2 error | 0 / 0 |
| Nonzero update L2 range | 0.0306714–0.0363083 |
| Response mutation regression before fix | 2 failed |
| Response mutation regression after fix | 2 passed |

Saved-tensor hooks measure unique nonparameter storage observed during forward,
not simultaneously live allocator memory. The tiny CPU fixture retains 2,560 B
of mandatory prefix boundaries in both arms. Storage categories can overlap and
must not be added or interpreted as a GPU memory reduction.

| Component observation | Existing prefix | Joint replay | Change |
|---|---:|---:|---:|
| Prompt forward observed saved storage | 24,544 B | 32 B | −99.87% |
| Suffix-wave forward observed saved storage, range | 56–45,656 B | 2,568–2,680 B | Workload dependent |
| Mandatory boundary storage | 2,560 B | 2,560 B | 0% |
| Physical decoder token calls per update | 80 | 160 | +100% |

The earlier H20 fixture tested parent commit
`d6229d638b08a3927d23d39854aa13a85df71933`, not this new replay code.
It passed both token-mean and response-mean cases with real MCore DDP,
`train_one_step`, FP32 AdamW and two updates per arm. Across all four comparisons,
maximum main-gradient error was 1.19209e−7 and parameter error was 5.96046e−8.
All trainable Adam moments, logical counters and gradient finalization passed.
The raw archive contains 20 SHA-verified payloads, including eight step receipts.
These are tiny fixed-trace correctness controls, not official online learning.

## Required speed, memory and quality comparison

| Workload | Baseline speed | Joint speed | Speedup | GPU memory reduction | Reward convergence |
|---|---:|---:|---:|---:|---|
| New C joint replay on H20 | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_ESTABLISHED |
| Official Ouro online RL | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_ESTABLISHED |
| Multi-seed high-load RL campaign | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_ESTABLISHED |

Raw logs, XML, all 24 observations, failure reproductions and the parent H20
archive are under `benchmarks/results/prefix-rematerialization/`. The initial
test fixture incorrectly read Adam state for frozen parameters; its 12 failures
are retained. Only trainable states are compared after that fixture correction.
No numerical threshold or BF16 failure gate was weakened.
