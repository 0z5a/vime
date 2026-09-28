# Resident actor validation on 2×H20

Date: 2026-09-28. Baseline code `e8f2a3e512cc9265cc06b58edf7eb6272aadf859`; candidate code `8f8ed82240cde34ea30b835e1cbb1e6e18b6641e`. Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`.

## Recipe

Two H20 GPUs (97871 MiB reported per GPU), TP=2 actor and rollout, colocated, resident actor, rollout offload, BF16, full activation recomputation, 16 prompts × 8 completions, response limit 4096, context limit 8192, GRPO, Adam LR 1e-6, betas (0.9, 0.98), weight decay 0.1, seed 42. Training uses fused attention; rollout CUDA graphs are disabled. VIME runtime: Torch 2.13.0+cu129, vLLM 0.30.0+cu129, TransformerEngine 2.16.1. The model revision, input data and configuration are fixed across the code comparison. Inputs derive from the deduplicated DAPO math 17k pilot split at dataset revision `2e65612930298bde4c5d58fd97b3f23a483aaff9`; the split-manifest SHA256 is `0d3252426ca77552d3c7e46a4ad60931650e8c9170f05283deb115d3c1b9f551`. The companion JSON preserves exact run arguments, metrics and replay-batch hashes.

## Matched replay timing

Both arms replay the same four saved batches (128 samples per batch) from the same initial model. The real vLLM engine remains active, and updated weights are published after every update. Driver wall time starts immediately before loading each batch and ends after training, snapshot/cleanup, weight publication and KV reactivation. Checkpoint saving and evaluation are disabled in this timing run. Drop the first step as warmup; retain all four raw timings in the companion JSON.

| Driver interval | Snapshot baseline | Live actor | Ratio / latency reduction |
| --- | ---: | ---: | ---: |
| Step 0 (warmup, excluded) | 87.088 s | 86.074 s | 1.0118× / 1.16% |
| Step 1 | 80.086 s | 79.946 s | 1.0017× / 0.17% |
| Step 2 | 77.979 s | 77.847 s | 1.0017× / 0.17% |
| Step 3 | 85.289 s | 85.014 s | 1.0032× / 0.32% |
| Median, steps 1–3 | 80.086 s | 79.946 s | 1.0017× / 0.17% |

The measured reduction is **0.17%**. This short run does not demonstrate a material speed benefit.

This measures the training/publication path with generation bypassed. It cannot establish online E2E acceleration, reward equivalence or time-to-target benefit. A single short pair also does not establish statistical significance.

The earlier single cold replay measured actor training alone at 84.079 s (baseline control) and 85.168 s (candidate), or 0.987× / −1.29% latency reduction. That inner timer excludes the snapshot operation and therefore cannot measure the intended optimization. Online generation time is never compared against replay time.

## Correctness and recovery

| Check | Result |
| --- | --- |
| CPU live-weight storage/update and tag checks | PASS |
| Three CUDA AdamW steps, optimizer states and published weights | Exact equality in component test on H20 and H100 |
| H20 runtime CLI/config/engine tests | 110 passed |
| Full candidate 4B replay, optimizer step, complete checkpoint and publication | Completed |
| Baseline online E2E | 128 generated/scored samples, finite nonzero gradient, full save, publication, normal shutdown |
| Candidate online resume | PASS: restored baseline iteration 5, generated 128 new samples, updated at iteration 6, saved full state, published weights and exited normally |
| Baseline full resume | Model, optimizer, scheduler and data cursor restored; five further online updates, full save and publication |
| Full 4B checkpoint bit-exact equality | **Not established**; finite differences below |
| Held-out reward / convergence | Not measured |

All 16 aggregated distributed tensor keys are compared, covering model layers, FP32 master parameters and both Adam optimizer groups. Scheduler and common optimizer metadata match. The following maxima compare the original online baseline and candidate replay separately against an independent baseline self-replay from the same initial model and exact trajectories.

| Tensor category | Baseline self-variation, max abs | Candidate vs control, max abs | Baseline / candidate maximum relative L2 |
| --- | ---: | ---: | ---: |
| BF16 model | 3.8147e-6 | 3.8147e-6 | 5.3490e-7 / 5.8083e-7 |
| FP32 master parameters | 1.9947e-6 | 1.9962e-6 | 1.0283e-6 / 1.1871e-6 |
| Adam first moment | 3.0518e-6 | 3.7909e-6 | 0.007083 / 0.007793 |
| Adam second moment | 2.2376e-9 | 3.0788e-9 | 0.010593 / 0.011893 |

All values are finite. Candidate and control differences have similar scale, but candidate differences are not bounded by every baseline difference. This is not proof of full-model numerical equivalence. Deterministic fused-attention attempts failed with 256.25 GiB and 64.13 GiB backward-workspace requests; neither is counted as a passing test.

## Online smoke and external baseline

| Diagnostic | VIME compatibility fix | Slime baseline | Speedup claim |
| --- | ---: | ---: | --- |
| Real generated/scored samples | 128 | 128 | N/A |
| Mean training reward | 0.125 | 0.1640625 | N/A: one sampled batch |
| Truncation fraction | 0.875 | 0.8359375 | N/A |
| Actor training | 88.415 s | 83.871 s | Not a matched speed pair |
| Reported step, excluding startup/final save | 252.854 s | 234.979 s | Not a matched speed pair |
| Process wall, including startup/save | 768.372 s | 396.334 s | Not comparable: disk vs tmpfs save |
| Full save/publication/shutdown | PASS | PASS on retry | N/A |

Slime source `559977aceab1c2703b25478a6389698d1e2e8929`, Torch 2.11.0+cu129 and SGLang 0.5.15.post1 follow the pinned CUDA 12 recipe. The two frameworks' actual 128 first-batch problem IDs and prompt token sequences match exactly; sampled responses differ. Runtime versions and KV capacity differ, so these smoke timings do not support a framework speedup claim.

The first Slime attempt completed its optimizer update, then failed during checkpoint saving with Ray `ActorUnavailableError` / `Socket closed`. The retry completed with 60 s Ray keepalive settings and fewer retained tmpfs files. These simultaneous changes leave the original cause unisolated. Failed evidence is retained separately; incomplete checkpoint shards were removed.

The VIME resume run produced 640 additional samples, reward 0.1609375, truncation 0.8265625 and a reported median step of 190.601 s excluding its first resumed step. It increased KV capacity from 16 to 32 GiB per GPU, so that change is configuration tuning, not a benefit from this patch.

Only the latest usable checkpoint for each active framework is retained on the remote machine. Superseded comparisons retain compact numerical evidence. No model weights or full checkpoints are stored on the Mac.

The candidate online resume reported reward 0.1484375, truncation 0.84375 and gradient norm 0.0799511 on its new 128-sample batch. Reported step time was 190.742 s and process wall time 321.346 s, including checkpoint saving but excluding subsequent checkpoint relocation. This follows the baseline's five resumed updates and is a correctness/recovery run, not an independent equal-progress speed or reward comparison.
