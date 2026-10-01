# Ouro budgeted GRPO: 2×H20 short pilot

Six updates per variant, one seed, 16 completions/update, 512 response tokens, four held-out prompts. The same pinned 1.4B checkpoint, dataset order and optimizer settings are used in every arm. These are exploratory cost/quality points, not convergence or noninferiority evidence.

| Variant | Mean step (s) | Ratio vs K=4 / latency reduction | Job (s) / GPU-h | Training reward | Truncated | Mean response tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed4 | 44.526 | 1.000× / 0.00% | 633.63 / 0.3520 | 1/96 | 73/96 | 436.1 |
| fixed2 | 22.786 | 1.954× / 48.83% | 499.86 / 0.2777 | 0/96 | 61/96 | 378.0 |
| fixed3 | 33.568 | 1.326× / 24.61% | 556.56 / 0.3092 | 1/96 | 70/96 | 428.5 |
| mixed | 33.415 | 1.333× / 24.96% | 558.63 / 0.3104 | 1/96 | 78/96 | 450.6 |

The step column averages all six measured max-rank intervals, including the first update and both complete K=2/3/4 cycles. It includes generation, scoring, training and publication; it excludes evaluation/checkpoint time. Job time includes initialization, evaluations and checkpoint save. It does not include prior model download/environment setup. The ratios do not establish speedup at equal quality.

| Variant | Evaluation update | K=2 success | K=3 success | K=4 success |
| --- | ---: | ---: | ---: | ---: |
| fixed4 | 0 | 0/4 | 0/4 | 0/4 |
| fixed4 | 3 | 0/4 | 0/4 | 0/4 |
| fixed4 | 6 | 0/4 | 0/4 | 1/4 |
| fixed2 | 0 | 0/4 | 0/4 | 0/4 |
| fixed2 | 3 | 0/4 | 0/4 | 0/4 |
| fixed2 | 6 | 0/4 | 0/4 | 0/4 |
| fixed3 | 0 | 0/4 | 0/4 | 0/4 |
| fixed3 | 3 | 0/4 | 0/4 | 0/4 |
| fixed3 | 6 | 0/4 | 0/4 | 0/4 |
| mixed | 0 | 0/4 | 0/4 | 0/4 |
| mixed | 3 | 0/4 | 0/4 | 0/4 |
| mixed | 6 | 0/4 | 0/4 | 0/4 |

| Variant | Prefill block-tokens | Decode block-tokens | Trainer layer tokens including recompute | Nonzero-gradient updates |
| --- | ---: | ---: | ---: | ---: |
| fixed4 | 1,544,832 | 4,009,632 | 12,165,120 | 1/6 |
| fixed2 | 772,416 | 1,737,120 | 5,603,328 | 0/6 |
| fixed3 | 1,158,624 | 2,955,096 | 9,013,248 | 1/6 |
| mixed | 1,128,480 | 3,169,992 | 9,394,176 | 1/6 |

Block-token counters measure actual layer invocations. Trainer counts include packing padding and recomputation, without separating them. Backward FLOPs/cost are not profiled. Different response lengths and truncation rates affect these totals. No learned halting or time-to-target result is claimed.

## Decision and validation

The simple mixed schedule reduces the measured step interval by 24.96%, but ends at 0/4 held-out success at every K, versus 1/4 at K=4 for the fixed-four-loop arm. Its truncation rate is also higher (78/96 versus 73/96). This short pilot does not establish a useful quality/compute tradeoff. Keep the global-budget recipe experimental and do not expand it into mixed-depth bucketing or claim equal-quality acceleration. Four held-out prompts are insufficient to establish general inferiority either.

| Correctness / execution check | Result |
| --- | --- |
| Budget/group identity and unchanged reward normalization | 14 unit tests pass |
| Shared provider, packed isolation and recomputed gradients | 1 provider test passes; independent HF checks also pass |
| Two-rank global plan agreement and mismatch rejection | PASS, real Gloo processes |
| Tiny two-rank K=2/3/4 online/save/resume | PASS; infrastructure-only zero-reward run |
| Full 1.4B K=4 online/save/resume | Six updates, resume from iteration 5, new update 6 and complete checkpoint; both ranks and process complete |
| Full 1.4B fixed2/fixed3/mixed | Six updates, three evaluations and complete checkpoint per arm; exit code 0 |
| Final retention | Only Ouro mixed iteration 5 retained; prior arm and tiny checkpoints removed |

Tested recipe: `b1c903650432a0746a4b6c09e029eaed34462a80`; provider: `bb1ea8a7b350a1cad9e6f414278759be5bf0ea67`; RL companion: `74ec97df222728fbeda9041fb82555be29aa4e84`; Megatron: `1dcf0dafa884ad52ffb243625717a3471643e087`. Model: `ByteDance/Ouro-1.4B@574fa66cb8bf5abdc979642d01cf2b79b16bfab1` (weight SHA256 `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`). Environment: PyTorch 2.13.0+cu129, vLLM 0.30.0+cu129, Transformers 5.14.1, Transformer Engine 2.16.1; the separate HF reference overlay uses Transformers 4.55.0. TP/PP/CP=1, DP=2, H20 95.6 GiB per card.

Training and validation sets contain 512 and 1,024 disjoint problem IDs; the pilot evaluates the first four validation IDs. Dataset/source hashes, every rank's measured intervals and numerical training metrics are in the raw JSON companion. A predeclared protocol is committed separately. Run `examples/ouro/run.sh` with each of `--ouro-depths 4`, `2`, `3`, and `2 3 4` from the same initial checkpoint and six total updates. No environment, driver or system CUDA update is required by the launcher.
