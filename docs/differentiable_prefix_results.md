# Differentiable recurrent-prefix reference

The fixed-depth reference reuses the current actor's prompt forward across suffix microbatches, accumulates all recurrent KV and first-response readout adjoints, then runs prefix backward once. It preserves full parameter gradients and nonzero optimizer updates for Ouro, Nanbeige and same-latent Huginn. It is currently slower than packed-wave replay on the tiny CPU workloads below.

This establishes an executable gradient reference for W07/T4. It does not establish a new scheduling contribution, production MCore/Ray integration, GPU memory savings, official-checkpoint training or reward convergence. The ordinary three-phase schedule follows [Schedule-Level Shared-Prefix Reuse v3](https://arxiv.org/html/2606.01143v3); the reference here explicitly preserves recurrent calls and per-loop response boundaries.

## Execution contract

Each prefix program records every physical layer at every loop, plus the final prompt-position hidden state that predicts the first response token. Suffixes read detached boundary leaves. Their backward passes accumulate the leaves' gradients and their own shared-parameter contributions. The final prefix backward injects all accumulated boundary gradients into the original graph. Prefix states are not permanently detached.

Ouro applies its loop normalization; Nanbeige respects `skip_loop_final_norm`. Huginn records prelude, recurrent-core and coda KV, retains repeated current-weight injection, and keeps coda output out of recurrent feedback. Complete Huginn prefix reuse requires identical actual latent seed/profile. Different seeds are rejected for that prefix; the implementation does not alter sampled group seeds to manufacture reuse. Sharing only the deterministic prelude across different latents remains pending.

The API is scoped to one current-actor logical group. It binds prompt, policy version, model revision, loop depth and latent identity; rejects mutated or replaced parameters; rejects duplicate/missing microbatch members; and permits prefix backward only once. Logical loss weights remain supplied by the existing objective. Zero-response groups keep differentiable zero gradients. There is no persistent rollout KV cache, distributed optimizer scheduling, activation offload, rematerialized prefix, or packed-suffix implementation in this reference.

## Numerical validation

The full regression passes **402 CPU checks**, skips six existing CUDA cases, and deselects three previously identified factory checks requiring unavailable Triton imports. No package is installed or changed. The added 48 checks include:

- 36 full-gradient/update configurations: three families, terminal/all-loop supervision, three microbatch partitions and FP32/FP64 parameter storage. The largest relative L2 gradient error is `1.818657e-7`.
- First-response-only cases proving nonzero hidden-boundary adjoints even when no suffix token is executed.
- Three independent FP64 causal-attention/GQA checks matching all Q/K/V gradients within `1e-12`.
- Latent/policy/prompt/depth isolation, parameter mutation/replacement, empty responses, partition validation and one-time backward.

An initial `1e-11` native FP64-weight gradient gate fails in 12 cases; a subsequent tighter mixed-precision gate leaves three Huginn failures. Both receipts are retained. Native RMSNorm always reduces in FP32, and Huginn rotary math also casts to FP32. Six diagnostic forwards are exactly equal; gradient relative errors are `2.33e-8`–`6.94e-8`, with maximum element error `3.27e-7`. Parameter storage dtype therefore does not make these native models FP64 mathematical oracles. The final full-model gate uses the existing mixed-precision bounds: all elements `atol=1e-5, rtol=8e-5`, and global relative L2 below `2e-6`. The separate pure-FP64 attention check retains its `1e-12` bound.

## Matched CPU experiments

Tiny random width-8/vocabulary-13 native providers run on one CPU thread. Ouro/Nanbeige/Huginn use 4/2/3 loops. Every workload has one shared prompt and group size G, with suffix microbatches of up to four. The full-sequence baseline batches projections, MLPs and readout within each wave; the prefix reference currently executes each suffix's projections separately. Both use segmented SDPA, the same parameters, FP32 products for streamed readout and the same loss/reduction.

The fixed RLTT objective uses progressive credit alpha 1.5, sampled terminal k3 coefficient 0.01, entropy coefficient 0.003, a distinct frozen reference with head weights scaled by 0.95, and logical token weights `1/(G*D)`. Each observation performs a real SGD update at learning rate 0.03. Timing covers gradient zeroing, actor forward, loss, all suffix/prefix backward and SGD. It excludes parameter reset, optimizer construction, frozen-reference evaluation, rollout, reward, publication and checkpoint work.

All observations compare loss, every gradient element, global gradient error and nonzero update against a baseline oracle. Benchmark bounds are `2e-6` for loss, gradient `atol=5e-6, rtol=1e-4` plus relative L2 below `1e-5`, and update-delta `atol=2e-6, rtol=1e-4`. The same-tree matrix has **648 observations** from three separate process starts, with three baseline/prefix/prefix/baseline blocks per workload. Maximum gradient relative error is `7.332838e-7`. A separate 72-observation seed-30 pilot is retained and not pooled into the table.

All Huginn rows use the same recorded latent seed 91 within the fixed trace. They do not establish a reuse opportunity for ordinary groups with distinct latent seeds. No peak-memory measurement or saving is claimed. Intervals use the three process-level log ratios with Student-t, two degrees of freedom; this shared-host CPU experiment is not GPU or full-RL performance evidence.

| Family | P / D / G | Baseline ms | Prefix ms | Baseline / prefix [95% CI] |
|---|---:|---:|---:|---:|
| huginn | 4 / 16 / 1 | 11.91 | 19.32 | 0.616 [0.456, 0.833] |
| huginn | 4 / 16 / 8 | 34.21 | 94.35 | 0.363 [0.263, 0.499] |
| huginn | 4 / 16 / 32 | 147.54 | 356.02 | 0.414 [0.337, 0.510] |
| huginn | 24 / 4 / 1 | 14.21 | 21.13 | 0.672 [0.528, 0.856] |
| huginn | 24 / 4 / 8 | 37.57 | 87.51 | 0.429 [0.359, 0.513] |
| huginn | 24 / 4 / 32 | 152.51 | 328.95 | 0.464 [0.369, 0.582] |
| nanbeige | 4 / 16 / 1 | 4.88 | 8.06 | 0.606 [0.431, 0.851] |
| nanbeige | 4 / 16 / 8 | 15.59 | 47.28 | 0.330 [0.307, 0.355] |
| nanbeige | 4 / 16 / 32 | 58.94 | 164.71 | 0.358 [0.329, 0.390] |
| nanbeige | 24 / 4 / 1 | 5.59 | 9.58 | 0.583 [0.504, 0.674] |
| nanbeige | 24 / 4 / 8 | 15.95 | 42.64 | 0.374 [0.314, 0.445] |
| nanbeige | 24 / 4 / 32 | 57.93 | 170.09 | 0.341 [0.230, 0.503] |
| ouro | 4 / 16 / 1 | 12.13 | 21.23 | 0.571 [0.512, 0.637] |
| ouro | 4 / 16 / 8 | 37.05 | 109.08 | 0.340 [0.281, 0.411] |
| ouro | 4 / 16 / 32 | 139.85 | 432.16 | 0.324 [0.292, 0.359] |
| ouro | 24 / 4 / 1 | 12.12 | 18.48 | 0.656 [0.567, 0.758] |
| ouro | 24 / 4 / 8 | 34.42 | 109.40 | 0.315 [0.221, 0.448] |
| ouro | 24 / 4 / 32 | 139.31 | 388.78 | 0.358 [0.258, 0.498] |

Every same-tree ratio is below one. The reference loses suffix projection batching and adds boundary bookkeeping on very short, tiny-width workloads. It is a correctness baseline, not an accelerated default. A subsequent implementation must preserve packed suffix work and be compared against this reference and an untouched upstream parent.

The untouched-parent check uses 12 fresh sequential APPA processes and 108 observations. The three workloads per family are a one-response control, a suffix-heavy group, and a prefix-heavy group. Full gradient and update vectors from the first parent arm are retained. All numerical/update gates pass; every interval remains below one.

| Family | P / D / G | Baseline ms | Prefix ms | Baseline / prefix [95% CI] |
|---|---:|---:|---:|---:|
| huginn | 4 / 16 / 1 | 13.98 | 19.85 | 0.705 [0.563, 0.881] |
| huginn | 4 / 16 / 32 | 124.49 | 362.64 | 0.343 [0.190, 0.619] |
| huginn | 24 / 4 / 32 | 152.97 | 324.89 | 0.471 [0.295, 0.750] |
| nanbeige | 4 / 16 / 1 | 5.38 | 8.71 | 0.618 [0.575, 0.664] |
| nanbeige | 4 / 16 / 32 | 57.06 | 161.72 | 0.353 [0.252, 0.494] |
| nanbeige | 24 / 4 / 32 | 55.96 | 171.83 | 0.326 [0.168, 0.633] |
| ouro | 4 / 16 / 1 | 12.60 | 20.21 | 0.624 [0.413, 0.941] |
| ouro | 4 / 16 / 32 | 119.92 | 374.80 | 0.320 [0.307, 0.334] |
| ouro | 24 / 4 / 32 | 135.50 | 335.78 | 0.404 [0.327, 0.498] |

## Reproduction and remaining work

Raw inputs, fixed reference values, advantages, observations and SHA256 manifests are retained in `benchmarks/results/differentiable-prefix-cpu`. Fresh-parent outputs and full gradient/update oracle vectors are in `benchmarks/results/differentiable-prefix-clean-cpu`. The unchanged parent is `5f7ce7faa26c6a95fcfee2a751ab332a39373e02`; all 688 blobs, including one symlink target, match.

Run `benchmarks/bench_differentiable_prefix.py` separately for seeds 31/32/33, with `--blocks 3`. `benchmarks/bench_prefix_clean.py` selects an explicit source root and checks that the imported provider belongs to that root; run parent/prefix/prefix/parent as fresh sequential processes per seed. `benchmarks/summarize_differentiable_prefix.py` reconstructs the comparison tables and intervals from raw files.

Next gates are packed suffix execution, ordinary prefix-reuse comparison under matching rematerialization, actual distinct-latent Huginn prelude sharing, production logical-step scheduling, target CUDA kernels/peak memory and full online RL with held-out reward convergence. This prototype completes none of those gates.

The exact measured clean-comparison driver is retained as `measured-driver.py.txt`. Its microbatch loss callback is consumed synchronously before the next workload. The portable driver now binds those loop values explicitly to resolve a lint warning; production math is unchanged. Separate import-resolution receipts recheck the actual provider paths in both original comparison environments.
