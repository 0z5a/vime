# Colocate span accounting results (2026-09-27)

Validation ran with `/home/gongji/0z5a/bin/python` on `gj-5090-2`. The candidate is the accounting fix in this PR; A is its parent (`01dc1b77`). The CPU benchmark measures one context-manager invocation, not a VIME training step. Each arm ran in a fresh Python process with 50,000 invocations per cycle. The first of six cycles was discarded; the table uses the median of the remaining five. Pair order was A→B, B→A, A→B. Per-cycle measurements are in [offload_spans_benchmark_20260927.jsonl](offload_spans_benchmark_20260927.jsonl).

| Mode | Pair | A, ns/span | B, ns/span | A/B speed ratio |
|---|---:|---:|---:|---:|
| Disabled | 01 | 1112.55 | 1120.30 | 0.993× |
| Disabled | 02 | 1313.94 | 1082.24 | 1.214× |
| Disabled | 03 | 1125.07 | 1095.26 | 1.027× |
| **Disabled, median pair ratio** | | | | **1.027×** |
| Enabled | 01 | 1466.25 | 1495.13 | 0.981× |
| Enabled | 02 | 1458.75 | 1496.85 | 0.975× |
| Enabled | 03 | 1469.39 | 1496.17 | 0.982× |
| **Enabled, median pair ratio** | | | | **0.981×** |

The disabled-path measurements vary enough to make a speed gain unproven. Enabled spans cost about 2% more per invocation in this CPU microbenchmark while recording nested intervals correctly. These numbers do not establish an E2E training speedup.

| Validation | Result |
|---|---|
| `python -m pytest -q tests/observability/test_offload_spans.py` | 8 passed in the `0z5a` environment |
| `ruff check` on the changed Python files; `ruff format --check` on the span module and its tests | Passed |
| RTX 5090 VIME training A/B | Both one-round E2E smokes and all three paired six-round runs passed in an isolated child environment |

## RTX 5090 full-step comparison

The smoke runs established vLLM rollout, Megatron reference and actor forwards, backward/optimizer, checkpoint save and weight sync. They are cold-start checks and are excluded from the speed ratios. The paired workload uses Qwen3-0.6B, two prompts × two samples, 16 maximum response tokens and a deterministic within-group GRPO reward. Each arm starts from the same checkpoint in a fresh process on the same RTX 5090. The disabled spans path is the reported candidate; round 0 is warmup, followed by five measured rounds. Driver round wall includes generation, training, offload and weight sync. Checkpoint save is disabled during speed measurement after the smoke verified it.

| Pair / order | Parent, median round s | Candidate, median round s | Parent/candidate speed ratio | Parent/candidate useful tokens/s | Parent/candidate mean response tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 / parent→candidate | 10.761 | 9.782 | 1.100× | 4.190 / 5.123 | 11.15 / 12.45 |
| 2 / candidate→parent | 11.836 | 11.300 | 1.047× | 3.274 / 3.856 | 10.70 / 10.90 |
| 3 / parent→candidate | 11.563 | 11.516 | 1.004× | 3.691 / 4.168 | 10.90 / 11.15 |
| **Median paired ratio** | | | **1.047× observed** | | |

Original six-round logs are stored under `/home/gongji/0z5a/work/vime-20260927/runs/` on the 5090 host. The runner records `VIME_ROUND_WALL_S`, per-round rollout metrics and `VIME_E2E_WALL_S`. Generated response lengths are shown because they can affect timing independently of this accounting change.

The disabled span code only changes accounting overhead. These noisy, variable-length runs do not establish that the accounting change caused a 4.7% speedup. The paired ratio is the observed wall-time comparison, not a claimed optimization.

## Enabled span ledger on RTX 5090

One separate Qwen3-0.6B E2E round with `VIME_OFFLOAD_SPANS=1` completed (`spans-diag1.log`). It confirms the emitted root accounting covers each measured call. This is a diagnostic round, not a speed arm.

| Call | Total s | Accounted s | Unaccounted s | Largest segment |
| --- | ---: | ---: | ---: | --- |
| `sleep` | 1.435661 | 1.435575 | 0.000087 | TMS pause, 1.133446 s |
| `wake_up` | 1.029749 | 1.029621 | 0.000128 | TMS resume, 0.617490 s |

## Explicit train-residency configuration on RTX 5090

This is a configuration comparison on the same base source, separate from the code A/B above. Both arms use Qwen3-0.6B, the same prompts and training settings, and an explicit 1 GiB vLLM KV cache budget. A passes `--offload-train`; B passes `--no-offload-train`. The resident one-round E2E smoke passed with a nonzero gradient norm and an optimizer update. Each paired arm starts a fresh process from the same checkpoint, runs one warmup plus five measured rounds, and samples total device memory used every 250 ms. Checkpoint saving is disabled in these speed arms. Pairs 1–2 used GPU 1; pair 3 used GPU 7 on the same eight-RTX-5090 host after contention on GPU 1.

| Pair / order / GPU | Offload median round s | Resident median round s | Offload/resident speed ratio | Useful tokens/s, offload / resident | Mean response tokens, offload / resident | Peak GPU used MiB, offload / resident |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / offload→resident / 1 | 13.563 | 2.816 | 4.816× | 2.997 / 17.226 | 11.75 / 12.25 | 16,571 / 18,730 |
| 2 / resident→offload / 1 | 9.997 | 2.417 | 4.136× | 4.892 / 20.397 | 12.95 / 12.25 | 16,429 / 18,730 |
| 3 / offload→resident / 7 | 8.872 | 2.042 | 4.345× | 5.258 / 22.282 | 11.70 / 11.60 | 15,626 / 18,730 |
| **Median paired ratio** | | | **4.345× observed** | | | |

The GPU reports 32,607 MiB total; the largest successful resident peak left 13,877 MiB unused at the sampled instant. This establishes a safe observed boundary for this small model and 1 GiB KV budget, not a general policy for larger models. The gain comes from selecting an existing configuration; this PR's span-accounting code is not the cause.

Failed attempts are retained in `resident-smoke.log`, `baseline-residency3.log`, and `resident-residency3-retry1.log`. The first failed before training because vLLM's dynamic memory profile observed free memory rising from 21.07 to 29.32 GiB; the explicit KV budget avoided that profile. The two GPU 1 pair-3 attempts later hit CUDA OOM when total device use reached 32,146 and 32,555 MiB respectively, including another concurrent GPU process. A completed offload retry without a matching resident arm was excluded from the speed ratio. The final pair used GPU 7, where both arms passed. Memory figures include all processes on the device and are not per-rank allocations.
