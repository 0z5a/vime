# ScaleRLT — IQuest explicit eager diagnostic

A separately frozen variant replaces SDPA with explicit repeated-KV matmul, FP32 softmax and value matmul, matching the pinned reference operator family. All original sixteen configurations, thirty-two paired AdamW updates, tokens/targets, optimizer settings and numerical bounds are identical. The ordinary and recomputed paths are both included. This CPU experiment uses the existing environment and read-only same-version cache recovery; no package is installed or changed.

| Attention variant | Output gate | Gradient gate | Complete parameter/update gate | Maximum logit error | Maximum gradient relative L2 | Maximum parameter delta error |
| --- | --- | --- | --- | ---: | ---: | ---: |
| SDPA AUTO | 32/32 | 32/32 | 24/32 | 7.152557e-07 | 3.218670e-07 | 2.399087e-06 |
| SDPA MATH | 32/32 | 32/32 | 24/32 | 6.258488e-07 | 3.045814e-07 | 1.981854e-06 |
| Explicit eager | 32/32 | 32/32 | 26/32 | 5.960464e-07 | 2.961169e-07 | 1.743436e-06 |

All profiles check every parameter, first/second Adam moment and step counter, with nonzero applied updates. The explicit eager variant still fails six complete update observations, including seed42 lengths8/65. The operator change raises the complete pass count from24 to26 but does not resolve qualification. Original failed AUTO/MATH receipts and all numerical gates remain unchanged. No official40B weights, GPU provider, inference quality, speed, memory or reward-convergence result follows from this diagnostic.

Raw result: `evidence/iquest-contract/functional-replay-eager-audit.json`; frozen module/driver: `bench/iquest_replay_eager.py`, `bench/audit_iquest_replay_eager.py`. The first two attempts failed only during local imports (evicted mpmath file, then command search path); their logs are retained. The third completed all32 observations and naturally exited1 for the numerical gate.

The two Python files are exact executed snapshots, retained as experiment data with their original local paths. The portable prior audit and pinned official reference remain in the parent artifact. They are not registered learner entrypoints.
