# VIME / Phase 1 coordination checks — 2026-10-01

This follow-up tests [MinhaoLi0318's proposed engine contract](https://github.com/ThinkFlowLab/vllm-rlt/issues/70#issuecomment-5927860064)
against the contributor reference implementation, with the
[coordination reply](https://github.com/ThinkFlowLab/vllm-rlt/issues/70#issuecomment-5930413915)
recording the consumer requirements and proposed upstream ownership.
Minhao's PR 1 implementation branch is not yet available for testing.

VIME needs four-loop prompt prefill. Decode budgets belong in
`SamplingParams.max_loops`; the checkpoint's `total_ut_steps` remains four.
The old `RLEngine.set_loop_budget` behavior is superseded. For each completion,
the trainer replays `[4] * prompt_length + exit_depths[1:] + [1]`, with the
last input loss masked, including `last_exited` KV at deeper planes.
VIME will use upstream `RequestOutput.logprobs` and engine-level
`processed_logprobs` when Minhao's branch is available. The reference uses
`log_probs` and request-level `processed`; a duplicate compatibility API is
not needed.

## Checks and method

The minimal companion fix resolves `seed=None` at request creation and returns
the effective seed. Six native CUDA contract tests and seven CPU contract/PD
control tests pass. VIME's independent FP32 mixed-depth replay/recomputed-gradient
and generic physical-parameter conversion tests pass **2/2 in 0.302 s**; see
[provider log](raw/minhao-provider-cpu.log). Earlier full training/save/resume
results remain in the [Thor e2e report](../ouro-thor-20261001/README.md).

`examples/ouro/check_rollout_replay.py` compares rollout selected-token scores
with the same physical weights in `OuroMegatronModel`: actual execution trace,
uniform decode K, and uniform full depth. All use processed probabilities at
T=0.9 without filtering, one ten-token prompt and eight output tokens. Each
rollout also repeats from its returned entropy seed and checks exact token IDs
and selected scores. First-token depth must be four. Recomputed BF16 forward
scores are measured with absolute errors, rather than required to be bitwise
identical across different execution/batch shapes.

A random two-layer BF16 smoke model passed all five replay cases before the full
model run. Its [raw replay](raw/minhao-tiny-replay.json) is retained and its
[57,039,722 bytes of weights were removed](raw/minhao-tiny-cleanup.json).
The companion's logprob benchmark smoke also completed all three paths and
six profiler windows per path before testing the full weights.

## Full Ouro-1.4B score comparison

| Rollout | Actual depths | Actual trace MAE | Actual trace max | Uniform decode K MAE | Uniform four-loop MAE |
|---|---|---:|---:|---:|---:|
| decode-2 | 4 then 2 | 0.034261 | 0.075311 | 0.179502 | 0.202643 |
| decode-3 | 4 then 3 | 0.011586 | 0.052898 | 0.014414 | 0.064761 |
| decode-4 | 4 throughout | 0.026607 | 0.092323 | 0.025122 | 0.025122 |
| early-exit | 4 then 2 | 0.011142 | 0.024446 | 3.622689 | 3.622689 |
| delayed-exit | 4 then 3 | 0.018904 | 0.052898 | 0.056929 | 0.056929 |

**All five cases pass exact effective-seed token/score replay.**
[Full raw JSON](raw/minhao-full-replay.json), [execution log](raw/minhao-full-replay.log).

For early exit, replaying the actual trace reduces mean absolute score error
from 3.622689 (uniform four loops) to 0.011142. A residual remains in all cases,
including fixed K=4, where actual-trace MAE is 0.026607 and uniform-four-loop
MAE is 0.025122. This check therefore does not establish bitwise equivalence
between batched training recomputation and incremental engine inference.
The shared-weight FP32 oracle and recomputed-gradient checks pass separately.

The full-model run overlapped another Qwen-Image training phase. The recorded
GPU utilization before these checks was 98%. Selected-score throughput,
latency, peak allocation and all 18 profiler operator summaries are in the
[companion Markdown speed table](https://github.com/0z5a/vllm-rlt/blob/feat/training-contract/docs/validation/minhao-coordination-thor-20261001/README.md);
the timing observations include resource contention.

The [completed full-model copy and replacement shard were removed](raw/minhao-cleanup.json)
(3,232,344,846 bytes), along with the completed local transfer copy. The original
slower download completed naturally and its
[2,873,677,792-byte model copy was removed](raw/minhao-slow-cleanup.json) by the
existing queue without repeating GPU validation. All task-owned model copies
are now absent.

## Reproduction

Use the unchanged Thor runtime with the VIME source, companion engine and
existing Megatron runtime on `PYTHONPATH`:

```bash
python examples/ouro/check_rollout_replay.py /local/Ouro-1.4B replay.json
python /local/vllm-rlt/benchmarks/logprob_overhead.py /local/Ouro-1.4B overhead.json
```

Model revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`; safetensors SHA256
`58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`.
Thor runtime: PyTorch 2.13.0+cu130, Triton 3.7.1, Transformers 5.17.0,
NVIDIA driver 595.78. No package or driver changes, process termination or lcpu
NFS access. Shared GPU validation was authorized; timing evidence must be read
with the recorded resource conditions.
