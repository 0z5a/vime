# Fixed-budget Ouro GRPO

This experimental recipe reuses VIME's Megatron optimizer, GRPO loss, gradient
reduction and distributed checkpoint implementation. The provider keeps one copy
of each physical Ouro layer. It preserves all four sandwich norms, inter-loop
normalization and the original token positions. The unused exit gate is frozen
but retained in checkpoints and weight publication.

The separate `vllm-rlt` RL companion supplies selected-token log probabilities,
synchronous full-weight publication and uniform prefill/decode depth. It is not
upstream vLLM. The companion base is
`ThinkFlowLab/vllm-rlt@ea680f7735eeaa357d2e09ba24355747496041dd`; use companion
[0z5a/vllm-rlt#1](https://github.com/0z5a/vllm-rlt/pull/1), tested revision
`74ec97df222728fbeda9041fb82555be29aa4e84`, for the RL interface. Model weights are pinned to
`ByteDance/Ouro-1.4B@574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.

## Execution

Use the existing custom provider option:
`--custom-model-provider-path vime_plugins.ouro.model.model_provider`.
Launch `examples/ouro/train.py` with `torchrun`, TP/PP/CP=1, BF16, sequence
parallelism disabled and `--no-gradient-accumulation-fusion`. DP ranks each run a
local rollout engine and train the same shared-parameter architecture. The first
recipe supports full finetuning with GRPO, zero KL/reference/entropy costs and
unfiltered positive-temperature sampling. It does not support pipeline/tensor
parallel recurrent execution or LoRA.

`--debug-train-only` suppresses VIME's built-in vLLM setup during argument parsing;
this recipe still generates real online samples through its explicit RL engine.
This compatibility option does not turn the recipe into fixed-trajectory replay.

- `--ouro-depths 4` is the fixed-four-loop baseline.
- `--ouro-depths 2` and `--ouro-depths 3` are fixed-small-budget controls.
- `--ouro-depths 2 3 4` cycles global K deterministically by update ID. Every rank
  verifies the same depth, policy version and execution configuration before work.
- `--ouro-run-dir DIR` stores raw per-rank metrics, evaluations and completion markers.
- `--ouro-eval-data FILE --ouro-eval-prompts N --ouro-eval-interval M` evaluates each
  of K=2/3/4 on a separate fixed dataset. Use a meaningful held-out sample count for
  quality claims; tiny smoke evaluations do not establish convergence.
- `--ouro-resume --load CHECKPOINT --use-checkpoint-opt-param-scheduler` restores
  training state and checks the schedule, data hash, sampling seed/temperature,
  response cap and group size against `ouro-plan.json`.

Pass the ordinary VIME model dimensions, optimizer, data, batch and loss options
matching the pinned model. `--load` and `--ref-load` should initially point to the
HF directory; the custom provider loads the complete physical weights directly.
The final distributed checkpoint includes optimizer and RNG state. After a new
checkpoint is complete, earlier `iter_*` directories in that run's save directory
are removed. Use a separate save directory for each experiment.

The two-GPU pilot launcher accepts paths without changing the environment:

```bash
bash examples/ouro/run.sh /models/ouro pilot.jsonl /runs/ouro-mixed 6 \
  --ouro-depths 2 3 4 --ouro-eval-data validation.jsonl
```

Make VIME, Megatron and the pinned RL companion importable in the existing
environment first. JSONL rows contain `prompt` (text or chat messages), `label`
and `metadata.problem_id`. Training and evaluation IDs must be disjoint. This
small launcher uses four prompts with four completions each and a 512-token cap;
those defaults are a smoke workload, not a convergence protocol.

## Semantics and measurements

Each group retains `(prompt ID, K, policy version, execution config hash)` before
reward normalization. The existing reward normalization and GRPO loss are reused.
The training forward uses exactly the depth that generated its old logprobs.
No constant `lambda*K` reward penalty is introduced: it would cancel within a
fixed-K group. This is externally budget-conditioned training, not learned halting.

Physical layer hooks measure rollout prefill/decode block-token counts. Training
reports layer invocation tokens including packing padding and recomputation;
these counters are not FLOPs or a measured backward cost. Step timing includes
rollout, scoring, training and publication. Job timing also includes initialization,
evaluation and the final checkpoint in the completion marker. GPU-hours count DP
ranks once, even though trainer and rollout share their devices.

Mixed-depth microbatch bucketing is not implemented. Establish fixed-K correctness
and a useful quality/compute tradeoff before adding that scheduler. Compare fixed
K=4, fixed K=2/3 and the simple mixed schedule from the same checkpoint and data.
Do not infer equal reward, convergence or a speedup from reduced loop counts alone.
