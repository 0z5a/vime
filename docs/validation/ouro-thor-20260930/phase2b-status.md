# Phase 2B acceptance status — 2026-09-30

Overall status: **in progress**. The validated synchronous prototype does not
complete every requirement of [issue 70](https://github.com/ThinkFlowLab/vllm-rlt/issues/70).

| Requirement | Current evidence | Remaining work |
|---|---|---|
| Ouro training adaptation | Shared physical layers, VIME/Megatron GRPO, DDP, Adam and distributed checkpoints; TP/PP/CP=1, prior DP=2 evidence | Validate compatibility beyond the prototype's execution restrictions |
| Parameter conversion | Provider loads native HF tensors and publishes the same physical names/shapes; all 269 full-model tensors survive save/resume | Register Ouro in VIME's general conversion/export path; test a round trip through that path |
| Shared engine rollout/update contract | Prototype `RLEngine.generate_batch` / `publish`, versioned full updates and recurrent CUDA graphs | Replace prototype-only operations with the agreed Phase 1 public contract |
| Reproducible multi-iteration recipe | Full Ouro-1.4B: five real updates, matching gradients/probability metrics, fresh-process resume; all 272 checkpoint tensors and Adam metadata match continuous eager training | Re-run the recipe through the shared contract after integration |
| Existing inference-feature compatibility | Recurrent CUDA graph replay remains active across publication | Preserve and validate early exit, PD and speculative capabilities; the present recipe enforces fixed depth and the RL engine excludes speculative execution |

The upstream `LLMEngine` at `299bf14b117f42a38d15852886d673f89e123307` has none of
`start_weight_update`, `update_weights`, `finish_weight_update` or
`get_weight_version`, as proposed in the
[Phase 1 plan](https://github.com/ThinkFlowLab/vllm-rlt/issues/70#issuecomment-5888101273).
The formal shared-contract integration depends on that delivery. The RFC explicitly
requires preserving existing inference capabilities from the start; the prototype's
fixed-depth restrictions cannot be treated as completion of that requirement.

The [Thor e2e report](README.md) remains valid for its stated synchronous scope.
Its completed-run markers describe execution of those runs, not overall Phase 2B
acceptance. The draft stack remains open pending the work above.
