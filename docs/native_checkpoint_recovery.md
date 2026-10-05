# Stored-state and Adam update audit

This adds the checkpoint gate needed by the official OuroThinking qualification
in [the preceding protocol](native_reward_qualification.md). It changes no learner,
rollout, optimizer or checkpoint writer. **No new official-model training or
fresh Ray worker has run.**

`native_checkpoint.py` reads both DCP payloads freshly, one chunk at a time.
It compares tensor shape/layout metadata, every stored model/optimizer/RNG chunk
and every common state field except launch arguments. Typed hashes include dtype
and shape; nonfinite numerical state is rejected. Historical `fingerprints.json`
files are never used. Metadata/common-file hashes are recorded alongside chunk
hashes so the result identifies the consumed control files.

`common.args` is excluded because continuous and resumed jobs have different
load/output paths. This exclusion is explicit in the report. Actual worker source
and effective argument verification remain separate required gates. Equality of
stored state does not prove that a fresh worker loaded it, or that an incomplete
checkpoint contains every parameter expected by the model loader.

## New gradients versus decay and old momentum

`audit_native_adam.py` checks the three unsharded FP32 checkpoints against the
initial safetensors weights, then compares continuous/resumed state and rollout
counters. It reloads both runs' raw rollout/train/gradient evidence through the
existing signal auditor; a separately supplied signal summary cannot bypass this.

For each saved trainable parameter, consider an AdamW update with **zero new
gradient**, while retaining the preceding moments:

\[
m_t^0=\beta_1m_{t-1},\quad v_t^0=\beta_2v_{t-1},\qquad
\theta_t^0=(1-\eta\lambda)\theta_{t-1}
-\eta\frac{m_t^0/(1-\beta_1^t)}{\sqrt{v_t^0/(1-\beta_2^t)}+\epsilon}.
\]

The first update uses the hash-checked initial weights and zero initial moments.
The audit requires both a fresh first-moment contribution and a parameter change
beyond this counterfactual. It allows four FP32 relative rounding units, scaled
by the compared values. Both standard MCore decay and no-decay possibilities
are checked, without guessing the parameter-to-group assignment. A result must
also coincide with a mixed-reward/nonzero-advantage/nonzero-gradient update in
**both** logical runs. This still does not establish held-out improvement or
attribute all of the gradient to the reward term rather than KL.

The supported stored format has a common Adam step, full FP32 parameter/moment
tensors, constant configured LR and an explicit `decoupled_weight_decay=True`
optimizer-group marker. Older or fused optimizer formats that omit this marker
cannot be qualified from their parameter values alone. Their actual worker
optimizer identity and state layout must be captured and validated first.
No claim is made for BF16, distributed optimizer shards, AMSGrad or a different
rounding profile. The generic stored-state comparison is separate from this
specific AdamW update gate.

```bash
python -m benchmarks.audit_native_adam \
  --continuous /task/qualification-runs/rltt/continuous \
  --resumed /task/qualification-runs/rltt/resumed \
  --packet /task/qualification \
  --initial-weights /models/Ouro-1.4B-Thinking/model.safetensors \
  --initial-sha256 VERIFIED_MODEL_MANIFEST_SHA256 \
  --input-audit /task/native-qualification-input-audit.json \
  --algorithm rltt --output /task/rltt-state-audit.json
```

The caller must obtain the expected weight hash from the independently verified
model manifest. The hash argument verifies those input bytes; it does not
establish checkpoint provenance by itself. Run only after all writers finish,
within the admitted storage/IO window, on trusted task-owned checkpoint files.

## Validation and comparison table

The final suite has **155 unique CPU passes**: 19 new checkpoint/Adam/CLI checks
plus 136 prior qualification checks. Six cases use actual CPU `torch.optim.AdamW`
and actual Torch DCP serialization at LR 1e-3 and the qualification LR 1e-6.
The integrated CLI tests use synthetic saved rollout evidence; their optimizer
gradients are control inputs, not gradients produced by an RL model.

| Case | Previous evidence | New measured check | Speed ratio |
| --- | --- | --- | --- |
| Nonzero-gradient CPU AdamW | Raw rollout signal alone cannot establish parameter change | Fresh moments and parameter effects detected at both LRs | Not a timing comparison |
| Weight decay only | Parameters can move with zero reward-learning signal | Rejected as a useful update | — |
| Old momentum, zero new gradient | Parameters keep moving after the first update | Only the first update is counted | — |
| Corrupted model/moment/RNG/scheduler/step | Cached chunk hashes can hide changed payloads if reused | Fresh reads reject all six corruption controls | — |
| Retained prior Ray PPO/SGD checkpoint | Prior stored-state report exists | 17 chunks, common state and rollout counters match exactly on fresh reads | — |
| Adam qualification of that SGD checkpoint | No Adam moments exist | Explicitly rejected | — |
| Official Ray/Adam fresh-worker recovery | NOT_RUN | NOT_RUN | — |
| Full-RL time-to-quality / reward convergence | NOT_ESTABLISHED | NOT_ESTABLISHED | — |

The retained-file audit reads the earlier synthetic-tiny Ray/PPO continuous and
resumed runs from the machine. It does not rerun those workers or reinterpret
their SGD result as Adam. CUDA remains uninitialized. The old raw files remain
unchanged. Initial inspection attempts omitted the existing MCore source path
and then assumed non-null optional planner metadata; both caller failures and
their corrected inspection are retained. No package was installed or changed.

Raw receipts and source hashes are in
`benchmarks/results/native-checkpoint-recovery/`. Repeated test runs overlap and
are not added together; validation durations are excluded from speed ratios.
The official qualification still requires actual worker imports, full framework
execution, held-out generation, fresh-worker identity and natural resource
handback. The original multi-model, high-load and three-seed quality campaign
remains open.
