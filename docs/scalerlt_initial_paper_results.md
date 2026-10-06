# ScaleRLT — Initial paper v0.1 results

**ScaleRLT: I/O-Aware End-to-End Reinforcement Learning for Recurrent Looped Transformers**

Author: 0z5a. Evidence cutoff: October 6, 2026.

This draft combines the existing audited component results into a systems manuscript. Accuracy, held-out reward convergence and scaled-RL stability remain the primary endpoints. Complete official-model online RL and time-to-quality have not run. Every speed ratio below retains its workload, denominator and numerical boundary.

## Exact actor replay: measured speed comparisons

CPU FP32 tiny width-8/vocabulary-13 models, P256/D8/G32; three independent process cohorts. Time includes zeroing gradients, RLTT loss, forward/backward and SGD. It excludes rollout, reward, reference, publication and checkpoint I/O. All timed observations check loss, full gradients and a nonzero update. Huginn shares the same recorded latent in this fixed trace.

| Family | Untouched packed-wave parent | Packed prefix | Baseline / candidate | Approximate 95% CI |
| --- | ---: | ---: | ---: | --- |
| Ouro | 450.72 ms | 152.09 ms | 2.963× | [2.099, 4.184] |
| Nanbeige | 219.31 ms | 69.79 ms | 3.142× | [2.039, 4.842] |
| Huginn | 605.58 ms | 164.77 ms | 3.675× | [3.100, 4.358] |

These are component CPU improvements. They do not establish official-model GPU, complete-RL or reward convergence gains. [Original packed-prefix report](packed_prefix_results.md) retains all workloads and the serial-prefix ablation; raw estimates are in `benchmarks/results/packed-prefix-clean-cpu/summary-parent.json`.

| Family | Negative control P/D/G | Baseline / candidate | Approximate 95% CI |
| --- | --- | ---: | --- |
| Ouro | 4/16/1 | 0.444× | [0.250, 0.788] |
| Nanbeige | 4/16/1 | 0.564× | [0.470, 0.676] |
| Huginn | 4/16/1 | 0.494× | [0.274, 0.889] |
| Ouro | 24/4/32 | 0.902× | [0.667, 1.218] |
| Nanbeige | 24/4/32 | 0.841× | [0.621, 1.140] |
| Huginn | 24/4/32 | 0.921× | [0.548, 1.549] |

All three single-response controls regress; all short P24 intervals include one. The measured online Huginn sharing fraction is unavailable.

## Actual CUDA learner accuracy

| Source / device epoch | Scope | Complete update gate | Maximum gradient error | Maximum parameter error | Speed ratio |
| --- | --- | --- | ---: | ---: | --- |
| 2ad271d / historical H20 | 12 tiny three-family configurations; 24 paired candidate AdamW updates | PASS: gradients, parameters, both moments, counters and accounting | 1.1772e-6 | 2.0949e-7 | NOT_MEASURED |
| d6229d6 / continued H20 8ee | Two original B controls; four paired candidate updates | PASS at unchanged gates | 1.1921e-7 | 5.9605e-8 | NOT_MEASURED |

Verification durations of 22.66 s and 15.706 s are not speed measurements. The current new H20 d952 at SSH 35679 has only resource metadata/coordination registration; it contributes no new scientific result to v0.1. [Historical CUDA details](../benchmarks/results/h20-mcore-all-families/README.md), [current control details](scalerlt_h20_current_native_gate.md).

## Official Ouro approximate inference baseline

ByteDance/Ouro-1.4B@574fa66c, source e8aa1ee, FP32/Triton. Fixed work uses P512/T128, 64 GiB KV allocation, C32 queued requests and actual resident A16/S16. Two fresh processes per arm, five warmups plus five measurements each; 60 trials, 1,920 requests and 245,760 tokens overall. Ratios are geometric means of the two fresh-process mean-throughput ratios.

| Arm | Fresh 0 tokens/s | Fresh 1 tokens/s | Speed vs D4/off | Allocated bytes | Quality status |
| --- | ---: | ---: | ---: | ---: | --- |
| D4/off | 143.8722 | 144.4825 | 1.0000× | 74,501,625,856 | Reference |
| D3/off | 166.7335 | 167.5084 | 1.1591× | 74,501,625,856 | UNCERTAIN |
| D3/guided | 164.6464 | 164.7555 | 1.1424× | 74,501,756,928 | UNCERTAIN |

Reducing decode depth changes the execution policy. Guidance is about 1.45% slower than D3/off and allocates 128 KiB more; no VRAM saving is demonstrated. This is a generation cost comparison, not exact-schedule learner or complete RL acceleration.

| GSM8K development arm | Correct / 128 | Accuracy | Paired status vs D4/off at unchanged 1 pp margin |
| --- | ---: | ---: | --- |
| D4/off | 80 | 62.50% | Reference |
| D4/two-head | 83 | 64.84% | UNCERTAIN |
| D3/off | 79 | 61.72% | UNCERTAIN |
| D3/two-head | 80 | 62.50% | UNCERTAIN |
| D2/off | 59 | 46.09% | DEGRADED |
| D2/two-head | 62 | 48.44% | DEGRADED |

All six arms use the same 128 frozen development IDs, strict grader and P4 prefill. The D3/two-head versus D4/off paired interval is [-11.54, +11.54] pp. Equal correct totals do not prove noninferiority. The independent 505-math/148-code confirmation remains NOT_RUN. These evaluations are not training reward curves. [Official numerical, quality and cost evidence](scalerlt_looped_backend_accuracy.md).

## Memory, failures and pending endpoints

| Result | Measured evidence | Boundary |
| --- | --- | --- |
| Huginn R32 stage KV | Tiny CPU KV tensor storage decreases 48.44%; 256 → 132 planes | No official GPU peak, Graph-pool or full-learner saving |
| Streamed readout | CPU dense/streamed ratios 0.414–0.722; lower saved intermediates | All measured cases slower |
| Rematerialization | Saved storage falls 78.30–83.43% | All three clean speed intervals include one; saved storage is not peak VRAM |
| Official FP32 Torch/Triton | Each passes all 12 cases/108 readouts/96 continuations/21 ownership checks | Inference math, not full RL |
| Official BF16/Triton | First prefill fails: 998/2,048 mismatches; max abs 0.230469 | Original 0.02 gate retained; subsequent stages not reached |
| CPU BF16 learner | All 24 parameter-update observations fail; six later gradient failures | No BF16 qualification |
| Tiny IQuest Auto/Math | 32/32 output and full-gradient checks per backend; 24/32 strict updates pass | Eight strict-update failures per backend |
| Tiny IQuest eager | 32/32 output/gradient, 26/32 strict updates pass | Six update failures; zero official tensor-body download |
| Complete official online RL | NOT_RUN | No learning, recovery or time-to-quality claim |
| ≥3-seed held-out reward convergence | NOT_RUN | Main quality endpoint |
| Long-running high-load / DP/TP/CP | NOT_QUALIFIED | Short output equivalence is insufficient |
| Matched RLTT / FlashLoop / Slime framework campaigns | NOT_RUN | Common domain and objective must be matched |

## Immutable raw anchors

| Evidence | Payload files | Archive SHA-256 |
| --- | ---: | --- |
| Historical three-family H20 MCore | 84 | 5cbea08375482e57936c9808a352fa3d12b22aabbb4a04a21723deac117f42ff |
| Continued-device B MCore | 20 | d1dc4b22da2ce28d29e07106267ff5802134ab3faae4cade8d548dcec743e618 |
| Official C32 fixed-work | 99 | 06d960ebb83e97eb1dee8af3a98b5c3cc956815eb558c7cdabccc37077ec5cb4 |

Counts exclude the extra SHA manifest. The original source receipts, negative attempts and natural handbacks remain intact. Component ratios are not multiplied into an end-to-end claim.

## Updated execution priority

1. Complete [RFC 465](https://github.com/vllm-project/vime/issues/465) through the standard `train.py` / `RolloutManager` backend: pinned ordinary Ouro, full depth four, GRPO without reference/critic/KL, at least two real updates, full publication and fresh-process recovery. Thinking/RLTT startup remains a separate profile.
2. Produce the first actual 1/2/4/8 GPU scaling figure crossed with batch/concurrency and sequence length. Compare vanilla VIME/vLLM, fixed-depth RLT and ScaleRLT under matched supported model/precision/sampling/objective/budget. Main metric: correctly graded valid samples divided by complete RL wall time. Incorrect samples and every RL phase remain in the denominator.
3. Train a loop-aware draft/exit controller using [merged PR 454](https://github.com/vllm-project/vime/pull/454), then establish reward/accuracy noninferiority and higher complete-RL goodput. Collection overhead, controller training and target verification must be counted. Exact target-verified proposals and approximate exits retain separate contracts.

All three formal endpoints remain pending. The new 35679 instance is one H20 and has no scientific scaling measurement yet. These priorities follow the user's October 6 steering; existing accuracy and reward-convergence requirements remain unchanged.
