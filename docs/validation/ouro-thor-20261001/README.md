# Shared Ouro/VIME contract on Thor — 2026-10-01

The recipe now uses the ordinary vllm-rlt `LLM` rollout/update API, general VIME
parameter conversion and per-token execution-depth replay. The engine source is
based on upstream `ecb1f8b505b7e831815b40aec3b4598619cca23a`.

## Full-model rollout timings

| Rollout | Eager (s) | CUDA graphs (s) | Speedup | Time saved |
|---|---:|---:|---:|---:|
| K=2 decode | 1.0689 | 0.9939 | 1.076× | 7.02% |
| K=3 decode | 1.5190 | 1.4207 | 1.069× | 6.47% |
| K=4 decode | 1.9414 | 1.8234 | 1.065× | 6.08% |
| Self-speculative K=4 target, K=2 draft, 3 candidates | 1.4952 | 1.3741 | 1.088× | 8.11% |

Five alternating-order pairs after each arm's warmup and full-weight publication;
two prompts, 32-token responses, temperature 0.9 and fixed per-request seeds.
Tokens, selected-token raw logprobs, depths and versions match exactly in every
pair. Prefill always uses K=4. Loading and initial capture are excluded. See
[raw pairs](raw/shared-full-rollout-benchmark.json).

## Real GRPO step timings

| Component | Eager (s) | CUDA graphs (s) | Speedup | Time saved |
|---|---:|---:|---:|---:|
| Rollout | 1.8644 | 1.8612 | 1.002× | 0.17% |
| Megatron forward/backward/Adam | 10.9735 | 11.7205 | 0.936× | -6.81% |
| Full publication | 0.0287 | 0.0285 | 1.007× | 0.65% |
| Whole step | 12.8710 | 13.5940 | 0.947× | -5.62% |

Medians of matched updates 1–3, cycling K=3/4/2, four prompts and four samples per
prompt, response cap 16, sequence length 512, BF16, full 1.43465B-parameter Ouro.
The step includes scoring, normalization, GRPO, training and publication. These
runs show a warmed generation improvement but **no whole-training-step speedup**.
The byte-parity reward probes the infrastructure; it does not measure task quality.
Response truncation and the short run do not establish convergence.

## Consistency, conversion and resume

| Check | Tiny model | Full Ouro-1.4B |
|---|---|---|
| Continuous eager vs graph + fresh-process resume | All 30 tensors and Adam metadata exact, step=4 | All 272 tensors and Adam metadata exact, step=5 |
| General HF export vs distributed checkpoint | All 27 physical tensors exact | All 269 physical tensors exact |
| Early-exit training | Two real updates | Two real updates |
| Async delayed-exit training | Two real updates | Two real updates |
| Self-speculative target-policy training | Two real updates | Two real updates |

Full eager training runs five updates; graphs run four, save and resume for update
five. All 269 physical tensors, FP32 master weights and both Adam moments match
exactly in the [final checkpoint comparison](raw/shared-full-checkpoint-match.json).
The [HF comparison](raw/shared-full-hf-match.json) uses VIME's generic exporter.
The corresponding [tiny checkpoint](raw/shared-tiny-checkpoint-match.json) and
[tiny export](raw/shared-tiny-hf-match.json) also match.

Both full-model arms have gradient norms 23.93328, 18.38594, 21.85287, 20.35000 and
21.98155. Mean absolute training/rollout logprob differences are 0.02642, 0.03018,
0.02602, 0.02226 and 0.02568 in both arms. BF16 SDPA and Triton have a nonzero
numerical difference, preserved in the raw logs. The independent tiny FP32 depth
replay oracle agrees within 2e-6 and recomputed gradients also agree.

First output depth is always four. Threshold-zero early exit uses depth two
thereafter; delayed async exit uses depth three; speculation verifies at depth
four. Each feature arm has finite nonzero gradients and aligned policy versions.
Ordinary graphs keep three captures through the first four updates with 334
replays and zero fallbacks. Speculation records 16 captures and 298 replays;
verification work includes rejected candidates rather than only committed tokens.

Eager total job time is 105.08 s. Fresh-process graph update five alone takes
60.67 s including initialization/load/save/export. The initial graph completion
marker is overwritten on resume, so a combined total-job speedup is not reported.

## Reproduction and cleanup

Model revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`; safetensors SHA256
`58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`.
Thor SM110, driver 595.78, PyTorch 2.13.0+cu130, Triton 3.7.1, Transformers 5.17.
The existing runtime is unchanged. Megatron source `d113016bb2850c4b7d804f24ba72c055d7d4f861`
uses the isolated [native Adam patch](../ouro-thor-20260930/megatron-native-adam.patch).

[run_shared.sh](run_shared.sh) and [shared_sequence.sh](shared_sequence.sh) preserve
the exact launch/checkpoint/export recipe. Tests exercise general conversion,
packed execution, trace replay, checkpoint equality, native/processed/speculative
probabilities, prefix invalidation and failed-publication recovery. Raw data,
metrics, completion markers, hashes and logs are retained in `raw/`.

[Cleanup](raw/shared-full-cleanup.json) removes 106,200,185,997 bytes of task-owned
full-model/checkpoint/export files; the [tiny cleanup](raw/shared-tiny-cleanup.json)
removes 2,459,624,347 bytes. Downloads overlap development. No processes were
killed, packages upgraded or NFS files accessed. Thor has only one GPU; real PD
is checked separately on the dual-A100 host.
