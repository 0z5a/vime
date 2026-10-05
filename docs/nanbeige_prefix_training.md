# ScaleRLT Nanbeige prefix training

Nanbeige already inherited the differentiable prefix and provider-forward
interfaces, but the logical-step planner rejected every trace whose family was
not Ouro. The actual MCore prefix schedule now passes the provider's declared
family to the planner. Ouro and Nanbeige preserve their separate math and loop
counts; a wrong or mixed family, revision, depth or latent identity is rejected.
Huginn remains outside this schedule until its latent-aware fallback is integrated.

The opt-in FP32 path supports ordinary prefix replay and the existing whole-depth
joint rematerialization. It keeps original sample/group IDs, token or response
normalization, distinct frozen-reference scores and all supervised loop outputs.
No sample budget, precision gate, model arithmetic or optimizer setting changes.

| Numerical check | Nanbeige result |
|---|---:|
| Independent dense / MCore loss adapter / prefix Adam update comparisons | 24 |
| Maximum whole-gradient relative L2 error | 2.0740e-7 |
| Maximum parameter-delta absolute error | 3.2783e-7 |
| B-only / joint replay Adam update comparisons | 24 |
| Joint replay maximum gradient absolute / relative L2 error | 0 / 0 |
| Joint replay nonzero delta L2 range | 0.0364186–0.0368734 |

The first comparison covers both logical reductions, three microbatch/wave
partitions and scale factors one/eight. The second covers terminal/all-loop
supervision, ragged waves, serial/packed suffixes and every trainable Adam moment.
Both use a distinct frozen reference; its parameters receive no gradients.
The tiny Nanbeige configuration has two recurrent loops, grouped-query attention
and final loop normalization enabled. These numbers are not official-checkpoint
error bounds.

The combined regression passes **188 CPU checks**, skips **8 actual CUDA cases**,
and retains **12 known strict Ouro BF16 expected failures**. Nanbeige BF16 is not
qualified. Real MCore CUDA tests now cover both families with/without joint replay,
but none of those new cases has run. Earlier H20 evidence belongs to the frozen
Ouro parent and cannot be transferred to this revision.

| Required workload | Baseline speed | ScaleRLT speed | Speedup | GPU memory reduction | Reward convergence |
|---|---:|---:|---:|---:|---|
| Nanbeige actual MCore CUDA update | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_ESTABLISHED |
| Official Nanbeige online RL | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_ESTABLISHED |
| Nanbeige scaled RL / concurrency | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_ESTABLISHED |

Raw XML, logs, all Nanbeige observations and source verification receipts are in
`benchmarks/results/nanbeige-prefix-training/`. Two earlier collection failures
came from iCloud-evicted MCore source files, before numerical tests ran. Final
validation used the existing task-owned MCore source packet after verifying all
490 files against its frozen manifest. The interpreter and installed dependencies
were unchanged; no stub or replacement package was introduced. The separate
original-file hydration completed naturally with four original SHAs verified.
