# Packed fixed-depth replay: implementation and current evidence

The opt-in RLTT path can batch projection, normalization, MLP and response-head
work across real sequence tokens. Every sequence resets its position indices;
Huginn also retains its own replayed latent seed. The final unused input token and
trailing packing padding are excluded from recurrent computation. Empty responses
still have a differentiable zero contribution.

`--rltt-attention-backend` chooses an explicit execution contract:

| Value | Projection/MLP/readout | Attention | Current validation |
| --- | --- | --- | --- |
| `serial` (default) | Per sequence | Per-sequence SDPA | Existing CPU baseline and native cycle |
| `sdpa-reference` | Packed real tokens | Segmented SDPA | CPU numerical, gradient, optimizer and native-cycle checks |
| `varlen` | Packed real tokens | Torch `varlen_attn`, causal `(-1, 0)` window, GQA enabled | Implemented; CUDA tests NOT_RUN |

The CUDA branch requires FP16/BF16 and fails explicitly on CPU or FP32. It has no
silent fallback. It calls the existing [Torch varlen API](https://docs.pytorch.org/docs/main/nn.attention.varlen.html);
this does not identify the selected low-level kernel. Kernel provenance and actual
CUDA forward/backward correctness remain qualification gates. No dependency was
installed or upgraded. The current short recipe defaults to FP32, so selecting
varlen also requires an explicitly qualified precision profile.

The MCore bridge supplies original CPU sequence lengths. Packed provider metadata
does not read CUDA cumulative boundaries back to Python. This does **not** remove
every synchronization from VIME: existing `get_batch` metadata construction and
padding-mask validation remain, and GPU profiling must measure the whole path.
Only fixed-depth RLTT opts into this implementation; dense PPO-family and
actual-depth replay retain their existing contracts.

## Correctness and lifecycle

The main regression run passed 282 tests, skipped two CUDA-only checks, and
deselected the three previously demonstrated missing-Triton factory checks.
The latter remain unresolved; their failure receipts are retained by the parent
training-bridge validation. A subsequent 41-case RLTT run passed after adding
both attention modes to the loss and native-cycle checks. These runs overlap;
their counts must not be added.

- Eighteen new provider cases cover three model families, B1/B4/B16, and
  recomputation on/off. Mixed lengths and empty responses compare every trainable
  gradient and an actual nonzero SGD update. Maximum whole-gradient relative L2
  error: `6.701e-7`; the required bound is `2e-6`.
- Six isolation cases cover terminal/all-loop outputs, sequence permutation,
  unrelated-sequence changes, and causal response boundaries.
- Twenty-four RLTT numerical cases cover three families, both logical reductions,
  recomputation on/off, and both serial/packed modes. Each checks three microbatch
  partitions with real MCore loss scaling and an independent dense objective.
- Six native cycles cover both modes for all three families. Fresh sampled groups,
  output-derived token-parity reward, three nonzero AdamW updates, physical weight
  publication and exact fresh actor/reference/Adam recovery pass.

These are real CPU components with tiny random models. They do not execute the
Ray/CUDA distributed lifecycle, official weights, held-out reasoning quality or
reward convergence. Raw cycle records and source/test hashes accompany this report.

## CPU component timing

Three separate process starts use seeds 31/32/33. Each case warms both arms,
then runs three serial/packed/packed/serial blocks. Both arms process identical
valid tokens. Sequence lengths are 16–30, response lengths half the sequence,
hidden width 8 and vocabulary 13. Model loop counts are Ouro 4, Nanbeige 2, Huginn 3.
The shared macOS CPU uses the existing Torch 2.13.0 runtime with one thread.
This task ran no other numerical test concurrently with the final measurements.
The host is shared; CPU timing is not a GPU performance prediction.

Time covers forward and backward with response-only streamed score/entropy
readout. Gradient clearing, optimizer, rollout/reward/reference evaluation,
publication and checkpoint work are excluded. Both arms retain per-sequence
SDPA attention: the measurements isolate packed non-attention work and reduced
Python/readout invocation overhead, not CUDA varlen acceleration.

| Tiny model | B | Serial (ms) | Packed SDPA reference (ms) | Speed ratio | Process ratio range |
| --- | ---: | ---: | ---: | ---: | ---: |
| ouro | 1 | 9.476 | 9.555 | 0.992× | 0.950–1.041× |
| ouro | 4 | 38.604 | 14.251 | 2.709× | 2.631–2.809× |
| ouro | 16 | 162.150 | 27.279 | 5.944× | 5.802–6.084× |
| nanbeige | 1 | 4.088 | 4.426 | 0.924× | 0.778–1.031× |
| nanbeige | 4 | 17.239 | 6.252 | 2.757× | 2.578–3.075× |
| nanbeige | 16 | 61.747 | 13.132 | 4.702× | 4.563–4.782× |
| huginn | 1 | 9.189 | 9.062 | 1.014× | 0.855–1.144× |
| huginn | 4 | 35.603 | 14.802 | 2.405× | 2.311–2.482× |
| huginn | 16 | 153.068 | 40.972 | 3.736× | 3.451–4.107× |

Times are geometric means; speed is serial/packed. The range contains the three
per-process ratios, not a confidence interval. Each timed observation checks loss
and the complete gradient. No peak-memory result was measured here.

## Numerical calibration and retained failures

An initial benchmark exceeded the tiny Huginn fixture's 32-position context and
terminated before its Huginn observations; the partial raw records are excluded.
The corrected common workload stays within all fixture contexts.

One FP32 Ouro B4 seed-33 comparison failed the initial `2e-6` relative-gradient
threshold: measured relative L2 `3.188e-6`, maximum absolute error `1.699e-6`,
and zero displayed loss difference. Repeating that input in FP64 gives relative
L2 `3.021e-16`. Against the same FP64 oracle, serial FP32 error is `2.321e-6` and
packed FP32 error is `1.060e-6`, consistent with a finite-precision accumulation
difference from changing GEMM/gradient aggregation shapes.

Final timing qualification therefore requires every gradient element to satisfy
`atol=5e-6, rtol=1e-4` **and** whole-gradient relative L2 `<1e-5`. This benchmark
tolerance is separate from the stricter provider-unit bound. All first attempts,
partial observations and the FP64 diagnostic remain in the task evidence ledger;
the final three-start data are committed in
[`benchmarks/results/packed-replay-cpu`](../benchmarks/results/packed-replay-cpu).
This calibration does not establish an official-checkpoint or BF16 error bound.

## Remaining W03 work

Qualify the actual CUDA varlen backend on the existing target runtime, record its
selected kernels, and compare official-model scores, all gradients and complete
updates against the pinned serial baseline. Measure B1/B4/B16 at the planned
1K/8K prompt and 512/2K response lengths, complete RL phase timing, peak memory
and convergence. GPU admission and runtime constraints still apply. This CPU
delivery is not completion of W03 or the full T3 baseline.
