# Recurrent rematerialization: CPU actor-update evidence

The opt-in path checkpoints groups of recurrent loops together with response readout, groups physical layers, and splits attention queries while retaining each query's complete causal K/V prefix. Ouro and Nanbeige keep their own loop normalization. Huginn recomputes prelude injection with gradients and keeps coda readout out of the recurrent feedback state. No gradient is truncated or detached to obtain these results.

The default remains the existing replay path. Training options are `--rltt-loop-checkpoint`, `--rltt-layer-checkpoint` and `--rltt-token-chunk`; zero disables that axis. An explicit plan controls its own layer grouping instead of inheriting `--recompute` from the old path. `ResponseReadout.rematerialization` provides the same API to direct callers.

`candidate_plans` enumerates a finite grid. `select_plan` chooses only measured, gradient-verified records with the same workload contract, timing scope and memory metric, within the supplied budget. It does not mix actor-update timing with whole-RL timing or CPU saved-storage observations with CUDA peaks. Prefix residency and sharded weight residency are not implemented by this planner yet.

## Correctness and lifecycle

The final regression is **354 passed, 6 CUDA checks skipped, 3 known missing-Triton factory checks deselected**. Six schedules across three families and terminal/all-loop supervision give36 full-parameter comparisons against the prior serial path; the largest gradient relative L2 error is3.728e-7. Nonzero SGD parameter deltas match. Three FP64 query-chunk cases match full causal attention and all Q/K/V gradients within1e-12, including GQA, unequal sequence lengths and future-token isolation. Empty responses retain differentiable zero gradients.

The MCore loss bridge also passes logical microbatch/reduction comparisons with the new schedule. Three new Tiny native cycles sample fresh groups, derive token-parity reward, perform three nonzero AdamW updates, publish physical weights and exactly reproduce the third step after fresh actor/reference/optimizer/engine restore. Maximum rollout/replay score errors are4.77e-7 (Ouro),4.77e-7 (Nanbeige) and1.42e-6 (Huginn). These are CPU component tests, not official reasoning quality or Ray/CUDA distributed training. Counts overlap the broader regression.

Query chunking uses an explicit offset causal mask for SDPA. The unqualified CUDA branch uses the existing public Torch varlen API with the full prefix and unequal Q/K lengths. The inspected Torch2.13 source routes varlen to FlashAttention, whose causal mask aligns to the bottom right. Six registered CUDA FP16/BF16 cases compare chunk sizes0/3/11 against full SDPA, but they have not run. Backend semantics must be verified on the actual target runtime. Sources: [Torch varlen API tutorial](https://docs.pytorch.org/tutorials/intermediate/variable_length_attention_tutorial.html), [FlashAttention causal alignment](https://github.com/Dao-AILab/flash-attention#how-to-use-flashattention).

## Fixed-trace measurement contract

Shared macOS/arm64, existing Torch2.13.0/Python3.12.14, FP32, one CPU thread. Random tiny models use width8/vocab13 and their actual family stages, with model seed81. Each logical batch has four sequences of lengths16/19/22/25 and eight response tokens each. Process seeds31/32/33 generate the frozen token traces. The objective uses progressive credit(alpha1.5), advantages[-1,0.3,0.7,-0.2], terminal sampled-k3 coefficient0.01 and entropy coefficient0.003. A distinct frozen reference has head weights scaled by0.95. All32 response tokens share the fixed logical denominator.

Timing includes zero_grad, forward, RLTT loss, backward and an actual SGD step(lr0.03). Parameters reset outside timing so every observation starts from the same policy. Optimizer construction, reference evaluation, rollout, reward, publication, checkpoint, Ray and network are excluded. This is a fixed-trace actor-update measurement, not a complete RL step. Each observation checks loss, every gradient and a nonzero parameter update. Bounds are elementwise atol5e-6/rtol1e-4 for gradients plus global relative L2<1e-5; update deltas use atol2e-6/rtol1e-4.

The storage metric sums unique non-parameter/non-buffer storage addresses observed by forward autograd save hooks. It is measured separately from timing. It includes complete underlying storage for saved views, counts shared storage once, and excludes backward workspace, allocator overhead and optimizer state. **It is neither a live-memory peak nor GPU memory.** Query checkpointing can change temporary workspace without changing this count.

## Eight-candidate same-tree comparison

The baseline is the existing packed response readout with per-layer recomputation, inside the candidate source tree. Three independent processes run three baseline/candidate/candidate/baseline blocks per candidate and family:864 observations. Zero failures occurred under the declared numerical bounds. Approximate95% intervals use three process-level log ratios with df2; repeated observations are not independent training seeds. Loop/layer/query-chunk values denote their explicit intervals.

| Family | Loop / layer / query chunk | Speed ratio [95% CI] | Observed saved bytes, baseline → candidate | Change |
|---|---:|---:|---:|---:|
| huginn | 1 / 0 / 0 | 0.740 [0.705, 0.776] | 71,660 → 76,524 | 6.79% higher |
| huginn | 1 / 0 / 8 | 0.309 [0.257, 0.370] | 71,660 → 73,404 | 2.43% higher |
| huginn | 1 / 2 / 0 | 0.573 [0.538, 0.610] | 71,660 → 14,748 | 79.42% lower |
| huginn | 1 / 2 / 8 | 0.185 [0.162, 0.211] | 71,660 → 14,748 | 79.42% lower |
| huginn | 2 / 0 / 0 | 0.736 [0.709, 0.765] | 71,660 → 74,028 | 3.30% higher |
| huginn | 2 / 0 / 8 | 0.310 [0.269, 0.359] | 71,660 → 70,908 | 1.05% lower |
| huginn | 2 / 2 / 0 | 0.592 [0.447, 0.783] | 71,660 → 12,252 | 82.90% lower |
| huginn | 2 / 2 / 8 | 0.168 [0.103, 0.275] | 71,660 → 12,252 | 82.90% lower |
| nanbeige | 1 / 0 / 0 | 0.714 [0.646, 0.789] | 27,688 → 8,504 | 69.29% lower |
| nanbeige | 1 / 0 / 8 | 0.242 [0.225, 0.261] | 27,688 → 8,504 | 69.29% lower |
| nanbeige | 1 / 2 / 0 | 0.547 [0.499, 0.599] | 27,688 → 8,504 | 69.29% lower |
| nanbeige | 1 / 2 / 8 | 0.183 [0.156, 0.215] | 27,688 → 8,504 | 69.29% lower |
| nanbeige | 2 / 0 / 0 | 0.697 [0.655, 0.742] | 27,688 → 6,008 | 78.30% lower |
| nanbeige | 2 / 0 / 8 | 0.232 [0.212, 0.254] | 27,688 → 6,008 | 78.30% lower |
| nanbeige | 2 / 2 / 0 | 0.539 [0.447, 0.650] | 27,688 → 6,008 | 78.30% lower |
| nanbeige | 2 / 2 / 8 | 0.143 [0.117, 0.174] | 27,688 → 6,008 | 78.30% lower |
| ouro | 1 / 0 / 0 | 0.673 [0.582, 0.778] | 51,360 → 13,504 | 73.71% lower |
| ouro | 1 / 0 / 8 | 0.366 [0.308, 0.436] | 51,360 → 13,504 | 73.71% lower |
| ouro | 1 / 2 / 0 | 0.768 [0.749, 0.787] | 51,360 → 13,504 | 73.71% lower |
| ouro | 1 / 2 / 8 | 0.221 [0.209, 0.233] | 51,360 → 13,504 | 73.71% lower |
| ouro | 2 / 0 / 0 | 1.102 [0.737, 1.646] | 51,360 → 8,512 | 83.43% lower |
| ouro | 2 / 0 / 8 | 0.265 [0.189, 0.371] | 51,360 → 8,512 | 83.43% lower |
| ouro | 2 / 2 / 0 | 0.738 [0.519, 1.048] | 51,360 → 8,512 | 83.43% lower |
| ouro | 2 / 2 / 8 | 0.300 [0.229, 0.394] | 51,360 → 8,512 | 83.43% lower |

## Unmodified-parent comparison

Plans were frozen from the separate seed30 pilot before this measurement: loop2/layer0/query0 for Ouro and Nanbeige; loop2/layer2/query0 for Huginn. The baseline is unchanged #9, `d41ab1ca24cba8c5085d213f32e0cbcd4f507a9d`. All647 Git blobs match, including the one symlink target. Twelve sequential fresh processes run parent/candidate/candidate/parent for each seed, with one warmup and one measurement per family:36 observations. Each process verifies its imported source root; candidate loss, full gradients and updates compare against the first clean-parent process's retained oracle vectors.

| Family | Parent ms | Candidate ms | Clean-parent speed ratio [95% CI] |
|---|---:|---:|---:|
| huginn | 16.09 | 31.14 | 0.517 [0.261, 1.022] |
| nanbeige | 7.25 | 11.17 | 0.649 [0.268, 1.570] |
| ouro | 26.15 | 23.37 | 1.119 [0.514, 2.436] |

Both sets retain negative results. Huginn loop-only plans can retain more storage than the old per-layer baseline because prelude activations remain retained; layer grouping changes that tradeoff. Token chunking is much slower on these short CPU sequences. The clean-parent intervals all cover1, so no stable acceleration is established. Selected plans reduce the observed saved-storage metric by78.30–83.43%, which must not be reported as GPU capacity or peak-memory savings.

The prototype selects from observed points without extrapolating. Its example CPU budget is the minimum saved-storage count among measured candidates, only to exercise a constrained choice. That proxy budget is not a deployable GPU budget. Target-device profiling, larger shapes, comparison above a generic differentiable-prefix baseline, full RL timing and reward convergence are still required for W06 completion.

## Reproduction

- `benchmarks/bench_rematerialization.py --seed N --output FILE`: eight candidates with the same immutable parameter/data contract.
- `benchmarks/bench_rematerialization_clean.py`: one fresh parent/candidate arm using explicit `--source`, `--arm`, `--seed`, `--order`, `--output` and matching PYTHONPATH. Run APPA per seed so order0 creates the independent numerical oracle.
- `python benchmarks/summarize_rematerialization.py`: regenerate both tables and hash manifests.
- Raw observations, source/test receipts, full clean-parent oracle vectors, profile selections and Tiny native update records are in `benchmarks/results/rematerialization-cpu/` and `benchmarks/results/rematerialization-clean-cpu/`.

No environment was installed or updated. The existing read-only MCore tree matches upstream1dcf0dafa884ad52ffb243625717a3471643e087 plus VIME's standard patch; native model code is RLT7d6e08edf6327797337f361421245539ac0283e8. Other numerical runs owned by this task had exited before formal timing; the shared host was not reserved. Pilot data and the initially incorrect symlink-audit attempt remain separate from final measurements.
