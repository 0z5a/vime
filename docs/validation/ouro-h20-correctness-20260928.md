# Ouro shared-parameter correctness on H20

Model: ByteDance/Ouro-1.4B revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
The 2,869,336,434-byte safetensors file has SHA256
`58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`.
The provider owns 24 physical layers, keeps four sandwich norms per layer,
normalizes at every recurrence boundary, reuses token positions and freezes the
unused exit gate. TP/PP/CP=1; the tiny online integration used DP=2 on two H20s.

| Check | K=2 | K=3 | K=4 |
| --- | ---: | ---: | ---: |
| Tiny FP32 HF logit max absolute error | 8.94e-8 | 1.19e-7 | 1.04e-7 |
| Tiny shared-gradient / recompute / cache checks | PASS | PASS | PASS |
| Full BF16 logits vs HF implicit-causal SDPA | 0 | 0 | 0 |
| Full BF16 logits vs HF default explicit-mask SDPA | 0.125 | 0.1875 | 0.1797 |
| Full BF16 cached selected-logprob vs dense SDPA max absolute error | 0.0478 | 0.1057 | 0.1046 |

Full K=4 FP32 default-HF vs training logits have max absolute difference 0;
FP32 cached selected-token logprobs differ by at most 6.68e-6. The full numerical
probe uses eight fixed input tokens and eight generated tokens. These are bounded
correctness probes, not long-context or quality validation.

The initial BF16 strict comparison failed because HF generated an explicit
boolean causal mask while the provider uses implicit causal SDPA. Layer-by-layer
tracing and matched-mask execution isolate that backend-dispatch difference.
The default-mask and cached BF16 paths are not bit-exact; their differences are
reported rather than hidden by widening the initial threshold.

Provider tests additionally compare packed independent sequences against the
native functional dense oracle, verify recomputed gradients and constant physical
parameter count. A random tiny model completed real cached generation, VIME's
existing GRPO/Megatron optimizer, full distributed save and restart on two H20s.
Its next resumed depth matched the saved schedule. Tiny rewards were all zero,
so that run establishes infrastructure execution only, not learning.

Companion runtime: [0z5a/vllm-rlt#1](https://github.com/0z5a/vllm-rlt/pull/1),
commit `74ec97df222728fbeda9041fb82555be29aa4e84`.
Full-model online reward training and long-context probability bounds are separate
validation work. No throughput or time-to-reward speedup is claimed here.
