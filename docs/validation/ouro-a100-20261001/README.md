# Ouro/VIME training with real P/D workers — 2026-10-01

Full Ouro-1.4B completes three real GRPO updates through the shared `LLM`
rollout/update contract. Prefill and decode run in separate spawned GPU processes
on A100 devices 0 and 1; the single Megatron trainer shares device 0. The graph
arm saves after update two and resumes in a fresh process for update three.
Every publication reaches both workers, and shutdown uses their stop/ack protocol.

## Consistency and save/resume

| Check | Random tiny BF16 Ouro | Full BF16 Ouro-1.4B |
|---|---|---|
| Continuous PD eager vs PD graphs + fresh-process resume | Four updates; all 30 checkpoint tensors and Adam metadata exact, step=4 | Three updates; all 272 checkpoint tensors and Adam metadata exact, step=3 |
| Generic HF export vs distributed checkpoint | All 27 physical tensors exact | All 269 physical tensors exact |
| Tokens, selected-token scores, rewards, depths and policy versions | Exact at every matched update | Exact at every matched update |
| Final worker ownership | Both workers stop with zero used blocks | Both workers stop with zero used blocks |

See [full checkpoint hashes](raw/full-pd-checkpoint.json),
[full HF comparison](raw/full-pd-hf.json),
[tiny checkpoint hashes](raw/tiny-pd-matched-checkpoint.json) and
[tiny HF comparison](raw/tiny-pd-matched-hf.json).
The checkpoint tensor count includes the physical model and distributed optimizer
buffers containing FP32 master weights and both Adam moments. Optimizer group
metadata is compared separately.

Full-model gradient norms are 26.38163, 36.34220 and 36.95280 in both arms.
Mean absolute training/rollout logprob differences are 0.02512, 0.03133 and 0.02761
in both arms; BF16 SDPA and Triton scores have this measured numerical difference.
The [paired summary](raw/paired-summary.json) checks the recorded output arrays
and corresponding gradient/logprob metrics, including the resumed update.

The full run uses two prompts with four samples each, three updates cycling
K=2/3/4, response cap 16, sequence length 512, processed logprobs at temperature
0.9, GRPO, BF16 and delayed async scheduling. Prefill always uses depth four.
The tiny run uses four prompts with four samples each and four updates.
The deterministic byte-parity reward tests training infrastructure; these short,
truncated rollouts do not establish model quality or convergence.

The fresh full-model resume has one decode capture, 470 replays and zero fallbacks.
Eight NIXL transfers send 138,444,800 bytes. Both final workers report version four
and zero used blocks; see [worker counters](raw/train-full-pd-graph/pd-workers.json).
Counters from the first graph process are overwritten by its fresh-process resume;
they are not aggregated across the two processes. The separate engine probe
checks graph reuse through several publications and failed-update recovery.

## Measured speed comparisons

| Full PD component | Eager median (s) | Graph median (s) | Speedup | Time saved |
|---|---:|---:|---:|---:|
| Rollout | 51.9759 | 51.2218 | 1.015× | 1.45% |
| Megatron forward/backward/Adam | 8.0184 | 8.4180 | 0.953× | -4.98% |
| Full publication to P/D workers | 2.6391 | 2.7243 | 0.969× | -3.23% |
| Whole GRPO step | 62.1185 | 62.2978 | 0.997× | -0.29% |

Medians of all three matched updates, including first-use capture and the fresh
resume. Other pre-existing jobs use both GPUs, and transport is host staged.
These are descriptive e2e timings, not a controlled throughput result: the whole
step does not improve. The analogous tiny whole-step medians are 6.2716 s eager
and 7.4705 s graphs (0.840×, -19.12%).
The [Thor report](../ouro-thor-20261001/README.md) provides the controlled warmed
rollout comparison: 6.08–7.02% ordinary decode time saved and 8.11% self-speculative
time saved. Its real GRPO whole-step result is also reported separately.

## Transport and numerical scope

The installed NIXL/UCX default GPU IPC path reports completion but does not copy
the independent two-process probe's destination. Every passing run uses
`UCX_TLS=tcp,cuda_copy` scoped to its command. Real registered GPU buffers exchange
NIXL WRITE transfers via host staging; direct GPU IPC/RDMA performance is unverified.
The [engine report](https://github.com/0z5a/vllm-rlt/blob/feat/training-contract/docs/validation/training-contract-a100-20261001/README.md)
contains the transport probe, raw/processed probability oracle, prefix invalidation,
stop/abort, full publication and rejected-update recovery checks.

Both controlled training arms set `--ouro-max-num-seqs 1`. The ordinary default
remains eight. At default BF16 batching, local-vs-PD and PD continuous-vs-resume
comparisons have 19 different checkpoint tensors despite equal rewards; the
[rejected cross-backend comparison](raw/tiny-pd-checkpoint-match.json) and
[rejected dynamic-batch resume comparison](raw/tiny-pd-resume-match.json) are retained.
The matched one-slot runs above are exact. Different BF16 batch shapes are not
claimed to be bitwise equivalent; the resume contract records a nondefault slot
count and rejects an incompatible requested configuration.

## Reproduction and cleanup

The model is `ByteDance/Ouro-1.4B`, revision
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, safetensors SHA256
`58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`.
Two A100-SXM4-40GB GPUs, driver 580.105.08, PyTorch 2.13.0+cu130,
Triton 3.7.1, NIXL 1.4.1 and Transformers 5.17.0; see [runtime](raw/runtime.json).
The existing `/venv/main` runtime is unchanged. Isolated Megatron source
`d113016bb2850c4b7d804f24ba72c055d7d4f861` reuses the
[native Adam patch](../ouro-thor-20260930/megatron-native-adam.patch).

[run_pd_a100.sh](run_pd_a100.sh) and [full_pd_sequence.sh](full_pd_sequence.sh)
preserve this host's exact launch/save/resume/export/check/cleanup commands.
[Input prompts](raw/train.jsonl) and [reward](raw/smoke_reward.py) are retained.
[compare_checkpoint.py](compare_checkpoint.py) hashes tensors loaded from the
distributed checkpoint and compares optimizer metadata. All three VIME regression
checks for packed execution/recomputed gradients, mixed-depth replay and general
parameter conversion pass in 6.31 s; see [test log](raw/provider-regression-final.log).

[Full cleanup](raw/full-cleanup.json) removes 45,928,504,188 bytes of task-owned
weights, checkpoints and exports immediately after both comparisons pass;
[tiny cleanup](raw/tiny-cleanup.json) removes 2,086,991,163 bytes.
The task's model directory is empty. Logs, metrics, hashes and completion markers
remain. Downloads overlap development; no processes are killed, environments
updated or lcpu NFS files touched.
