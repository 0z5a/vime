# Packed suffix execution for differentiable recurrent prefixes

`PrefixReplay.backward_suffixes(..., batch_suffixes=True)` batches suffix projections, normalization, MLPs and streamed response readout within each microbatch. It preserves each suffix's own causal history and all shared-prefix adjoints. The serial prefix reference remains the default and an independent comparison target.

This is an opt-in CPU-qualified execution path for Ouro, Nanbeige and same-latent Huginn. It does not yet integrate the production MCore logical-step schedule, joint rematerialization, distributed ownership or target CUDA kernels. No GPU memory saving, full-RL speedup or reward convergence is established here.

## Execution and gradient contract

The prefix forward still runs once. Each microbatch flattens its suffix token inputs for shared linear work. Position indices restart at the prompt length for each suffix. Segmented SDPA exposes the common prefix and only that suffix's causal keys, never another response's tokens. Readout restores the final prompt-position hidden state separately for each response at every supervised loop. Suffix backward accumulates all boundary-leaf adjoints; one final prefix backward precedes the optimizer step.

Zero-length responses contribute no rows, while one-token responses use only their first-response boundary. Unequal response lengths and reordered microbatch partitions retain the logical objective's original weights. Current policy, parameter versions, prompt, depth and latent identity checks are unchanged.

Huginn retains prelude/core/coda boundaries, repeated current-weight injection and absolute-position latent replay. Complete prefix reuse requires identical actual latent seed/profile. The benchmark's recorded seed 91 is a fixed-trace condition, not evidence that independently sampled latent groups can share recurrent prefixes. Distinct-latent deterministic-prelude reuse remains pending.

## Correctness evidence

The full regression has **448 passing CPU checks, six CUDA skips and three known missing-Triton factory deselections**. The 94-check prefix subset overlaps that count. Relative to the serial-reference draft, 46 checks were added: 36 packed full-gradient/nonzero-update configurations, nine empty/single-token/ragged cases, and an independent FP64 attention/adjoint/isolation check. Native models retain their existing mixed-precision gradient bounds; the independent FP64 attention bound is `1e-12`.

The first run passed 92 checks and failed two Huginn boundary checks with an unnormalized sum-of-squared-scores objective. Forwards were exactly equal; gradient relative L2 errors were `8.62e-8` and `8.92e-8`, with maximum absolute difference `2.44e-4`. Each failed one element of the fixed absolute/relative bound. The retained diagnostic also evaluates the logical-mean objective: all elements then satisfy the same bounds, with relative errors `7.44e-8` and `9.01e-8`. The final boundary tests use that mean and additionally enforce global relative L2 below `2e-6`. No production math or tolerance was changed to fix these checks. The original test source, failures and diagnostic remain available; the sum-reduction failures are not counted as passing tests.

## Matched CPU experiment

All providers have width 8, vocabulary 13, FP32 parameters, one CPU thread and 4/2/3 loops for Ouro/Nanbeige/Huginn. Suffix microbatches contain at most four responses. The three arms are packed full-sequence replay, serial prefix replay and packed prefix replay. Attention is segmented SDPA in all arms.

The fixed-trace RLTT objective uses progressive credit alpha 1.5, sampled terminal k3 coefficient 0.01, entropy coefficient 0.003, a separate frozen reference with its head scaled by 0.95, and logical token weights `1/(G*D)`. Each observation performs SGD at learning rate 0.03. Time covers zeroing gradients, actor forward, loss, complete backward and the optimizer step. Parameter reset, optimizer construction, reference evaluation, rollout/reward, publication and persistence are outside this measured scope.

Every observation checks loss, all parameter gradients and nonzero optimizer deltas against packed full-sequence replay. Frozen bounds are loss `atol=rtol=2e-6`; gradient `atol=5e-6, rtol=1e-4` plus relative L2 below `1e-5`; update delta `atol=2e-6, rtol=1e-4`. All **648 observations** pass, with maximum gradient relative L2 `7.332838e-7`. Three sequential process starts use seeds 31/32/33 and two symmetric full/serial/packed/packed/serial/full blocks per workload. A separate 108-observation seed-30 pilot is not pooled.

Ratios above one favor the packed prefix. Intervals use the three process-level log ratios and Student-t with two degrees of freedom. These shared-host CPU measurements have visible scheduling noise and are not GPU throughput measurements.

### Against packed full-sequence replay

| Family | P / D / G | packed-wave ms | Packed prefix ms | packed-wave / packed [95% CI] |
|---|---:|---:|---:|---:|
| huginn | 4 / 16 / 1 | 12.51 | 22.53 | 0.556 [0.462, 0.668] |
| huginn | 4 / 16 / 8 | 41.70 | 51.29 | 0.813 [0.612, 1.081] |
| huginn | 4 / 16 / 32 | 151.67 | 165.42 | 0.917 [0.818, 1.028] |
| huginn | 24 / 4 / 1 | 14.80 | 23.57 | 0.628 [0.349, 1.132] |
| huginn | 24 / 4 / 8 | 48.75 | 43.56 | 1.119 [0.784, 1.597] |
| huginn | 24 / 4 / 32 | 170.64 | 135.63 | 1.258 [1.121, 1.412] |
| nanbeige | 4 / 16 / 1 | 4.80 | 8.98 | 0.535 [0.305, 0.939] |
| nanbeige | 4 / 16 / 8 | 15.81 | 21.78 | 0.726 [0.622, 0.848] |
| nanbeige | 4 / 16 / 32 | 71.80 | 86.25 | 0.832 [0.594, 1.167] |
| nanbeige | 24 / 4 / 1 | 5.76 | 9.68 | 0.595 [0.270, 1.308] |
| nanbeige | 24 / 4 / 8 | 15.29 | 19.31 | 0.792 [0.743, 0.845] |
| nanbeige | 24 / 4 / 32 | 66.45 | 70.90 | 0.937 [0.747, 1.175] |
| ouro | 4 / 16 / 1 | 12.58 | 22.29 | 0.564 [0.432, 0.736] |
| ouro | 4 / 16 / 8 | 27.82 | 42.05 | 0.661 [0.442, 0.991] |
| ouro | 4 / 16 / 32 | 109.45 | 128.71 | 0.850 [0.726, 0.996] |
| ouro | 24 / 4 / 1 | 11.20 | 18.11 | 0.619 [0.541, 0.707] |
| ouro | 24 / 4 / 8 | 29.01 | 33.10 | 0.876 [0.768, 1.000] |
| ouro | 24 / 4 / 32 | 134.25 | 131.70 | 1.019 [0.927, 1.121] |

### Against the serial prefix reference

| Family | P / D / G | serial-prefix ms | Packed prefix ms | serial-prefix / packed [95% CI] |
|---|---:|---:|---:|---:|
| huginn | 4 / 16 / 1 | 23.42 | 22.53 | 1.040 [0.738, 1.464] |
| huginn | 4 / 16 / 8 | 108.03 | 51.29 | 2.106 [1.651, 2.688] |
| huginn | 4 / 16 / 32 | 371.50 | 165.42 | 2.246 [1.650, 3.056] |
| huginn | 24 / 4 / 1 | 22.74 | 23.57 | 0.965 [0.761, 1.223] |
| huginn | 24 / 4 / 8 | 100.68 | 43.56 | 2.311 [1.951, 2.738] |
| huginn | 24 / 4 / 32 | 341.04 | 135.63 | 2.515 [2.260, 2.797] |
| nanbeige | 4 / 16 / 1 | 8.17 | 8.98 | 0.910 [0.591, 1.403] |
| nanbeige | 4 / 16 / 8 | 49.44 | 21.78 | 2.270 [1.539, 3.349] |
| nanbeige | 4 / 16 / 32 | 186.69 | 86.25 | 2.164 [1.535, 3.051] |
| nanbeige | 24 / 4 / 1 | 10.04 | 9.68 | 1.037 [0.654, 1.644] |
| nanbeige | 24 / 4 / 8 | 42.92 | 19.31 | 2.223 [1.927, 2.566] |
| nanbeige | 24 / 4 / 32 | 157.74 | 70.90 | 2.225 [1.906, 2.597] |
| ouro | 4 / 16 / 1 | 21.52 | 22.29 | 0.965 [0.789, 1.181] |
| ouro | 4 / 16 / 8 | 90.78 | 42.05 | 2.159 [1.450, 3.214] |
| ouro | 4 / 16 / 32 | 364.86 | 128.71 | 2.835 [2.420, 3.320] |
| ouro | 24 / 4 / 1 | 17.94 | 18.11 | 0.990 [0.855, 1.146] |
| ouro | 24 / 4 / 8 | 83.99 | 33.10 | 2.538 [2.422, 2.659] |
| ouro | 24 / 4 / 32 | 347.50 | 131.70 | 2.639 [2.225, 3.129] |

Packing eliminates much of the earlier serial-suffix penalty for multi-response groups. It does not establish a general improvement over full-sequence packed replay: one-response and short-prompt cases often remain slower. Both comparisons and every tested workload are retained.

## Independent source comparisons

The full-sequence parent is the untouched draft #10 revision `5f7ce7faa26c6a95fcfee2a751ab332a39373e02` (688 verified blobs). The serial-prefix parent is untouched draft #11 revision `73d5ff32f316b47a7e36ed687f4e2fabd250921e` (755 verified blobs). Both audits preserve the recorded symlink. The clean driver verifies the actual three model-provider imports under each selected source root.

Fresh processes run full-parent/serial-parent/packed/packed/serial-parent/full-parent for each of three seeds. The matrix keeps the one-response control, suffix-heavy case and short-prefix case, and adds P256/D8/G32 to examine a larger prefix without selecting it from measured gains. The harness creates identical providers with positional capacity `max(32, P+D)` for every arm. Initial weights and all input/reference/advantage traces are matched.

An initial long-prefix run reached 11 observations before Huginn's original 32-position fixture rejected position 32. Those partial rows are excluded from final timing tables and retained with their failure. The revised harness changes only the explicit test configuration capacity, not provider implementation. A subsequent launcher attempt stopped on an existing log filename before starting a child; that receipt is also retained.

All **216 final observations** pass the frozen loss, gradient and nonzero-update checks; maximum gradient relative L2 is `7.324106e-7`. All 18 fresh child processes and their controller exit naturally with code 0. Full oracle vectors and import provenance are retained.

### Against the untouched packed-wave parent

| Family | P / D / G | parent ms | Packed prefix ms | parent / packed [95% CI] |
|---|---:|---:|---:|---:|
| huginn | 4 / 16 / 1 | 8.91 | 18.05 | 0.494 [0.274, 0.889] |
| huginn | 4 / 16 / 32 | 107.60 | 144.57 | 0.744 [0.420, 1.319] |
| huginn | 24 / 4 / 32 | 109.08 | 118.45 | 0.921 [0.548, 1.549] |
| huginn | 256 / 8 / 32 | 605.58 | 164.77 | 3.675 [3.100, 4.358] |
| nanbeige | 4 / 16 / 1 | 4.56 | 8.09 | 0.564 [0.470, 0.676] |
| nanbeige | 4 / 16 / 32 | 43.43 | 69.76 | 0.623 [0.391, 0.992] |
| nanbeige | 24 / 4 / 32 | 46.61 | 55.41 | 0.841 [0.621, 1.140] |
| nanbeige | 256 / 8 / 32 | 219.31 | 69.79 | 3.142 [2.039, 4.842] |
| ouro | 4 / 16 / 1 | 9.71 | 21.86 | 0.444 [0.250, 0.788] |
| ouro | 4 / 16 / 32 | 95.40 | 137.30 | 0.695 [0.401, 1.204] |
| ouro | 24 / 4 / 32 | 104.66 | 116.08 | 0.902 [0.667, 1.218] |
| ouro | 256 / 8 / 32 | 450.72 | 152.09 | 2.963 [2.099, 4.184] |

### Against the untouched serial-prefix parent

| Family | P / D / G | serial-prefix ms | Packed prefix ms | serial-prefix / packed [95% CI] |
|---|---:|---:|---:|---:|
| huginn | 4 / 16 / 1 | 16.17 | 18.05 | 0.896 [0.768, 1.046] |
| huginn | 4 / 16 / 32 | 306.21 | 144.57 | 2.118 [1.426, 3.146] |
| huginn | 24 / 4 / 32 | 288.21 | 118.45 | 2.433 [1.692, 3.500] |
| huginn | 256 / 8 / 32 | 335.44 | 164.77 | 2.036 [1.702, 2.434] |
| nanbeige | 4 / 16 / 1 | 7.28 | 8.09 | 0.900 [0.689, 1.175] |
| nanbeige | 4 / 16 / 32 | 143.95 | 69.76 | 2.064 [1.621, 2.627] |
| nanbeige | 24 / 4 / 32 | 141.42 | 55.41 | 2.552 [1.928, 3.379] |
| nanbeige | 256 / 8 / 32 | 147.37 | 69.79 | 2.112 [1.013, 4.402] |
| ouro | 4 / 16 / 1 | 20.14 | 21.86 | 0.921 [0.665, 1.277] |
| ouro | 4 / 16 / 32 | 317.84 | 137.30 | 2.315 [1.273, 4.209] |
| ouro | 24 / 4 / 32 | 328.83 | 116.08 | 2.833 [2.242, 3.580] |
| ouro | 256 / 8 / 32 | 361.79 | 152.09 | 2.379 [1.354, 4.179] |

For P256/D8/G32, packed-prefix replay is 2.963x/3.142x/3.675x faster than full-sequence packed replay for Ouro/Nanbeige/Huginn on these tiny CPU fixtures. All three intervals are above one. The independent serial-prefix comparison is also faster for all multi-response cases, while one-response controls do not improve. None of the three short-prefix P24/D4/G32 comparisons against full-sequence replay has an interval excluding one. The larger-prefix result establishes a limited CPU workload benefit, not official-model/GPU/full-RL acceleration or convergence.

## Reproduction

Run `benchmarks/bench_packed_prefix.py` separately for seeds 31/32/33 with `--blocks 2`. For independent comparisons, place the two unchanged source revisions in separate directories and invoke `benchmarks/bench_packed_prefix_clean.py` with explicit `--source`, `--arm`, `--seed`, `--order` and `--output`, as fresh sequential processes in the six-arm order above. The first full-parent arm writes complete loss/gradient/update oracle vectors; later arms verify those vectors.

`benchmarks/summarize_packed_prefix.py` regenerates each table using `--baseline packed-wave` or `serial-prefix` for the same-tree data and `--baseline parent` or `serial-prefix` for the clean data. Original inputs, full oracle vectors, raw observations, failure receipts and source/test SHA256 manifests accompany the results under `benchmarks/results/packed-prefix-{cpu,clean-cpu}`.

Next gates are measured CUDA attention and peak memory, ordinary independent-latent Huginn groups, joint rematerialization, complete production RL updates and held-out reward convergence. This change does not complete those gates.
