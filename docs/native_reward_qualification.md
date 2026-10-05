# Official OuroThinking online-RL qualification

This packet prepares the real standard-Ray run needed before the reward
convergence campaign. **No training phase has executed.** It replaces the
short recipe's SGD/48-token probe settings explicitly, while keeping the actual
VIME entry, native rollout, Megatron learner, reward function, full publication,
held-out evaluator and checkpoint path.

The selected public checkpoint is `ByteDance/Ouro-1.4B-Thinking` at
`3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`. RLT is pinned to
`fd993ec5512904b68e16f4d541682c076c487b5d`; the unqualified W08 group-fork flag
is not enabled. The VIME runtime is unchanged from parent
`67da3243aab3c82704d8b75e20696a34c9272830`.

## Frozen workload

`prepare_native_qualification.py` verifies the complete public source-profile
manifest SHA256 `a272932944329fa41b4da023b2d115606a1dbf3a364a1d1d300df6f251233b53`
and its explicit P1024 training file
`f3e52c440246c455b7a7fa089c63ab08f69fdac4a29bc9c828fdd567b0a72a90`.
It sorts by a fixed hash of the source problem ID and selects 12 training rows
and eight development rows. Selection never observes model output or reward.
The original 7,479-row training profile, full 256-row development set and
MATH500 remain the inputs for subsequent formal quality experiments.

The generated manifest SHA256 is
`d7bf3edcd0a05ee25b93bf7abaf1f739a4f8da4a769ac5fc00cf87a423e67f9c`.
Two preparations produce byte-identical data and manifest files. All 20 rows
pass actual official-tokenizer/VIME Dataset token equality and preserve labels,
metadata and order. The longest selected prompt is 1,013 tokens, below the fixed
1,024-token limit. This audit loads configuration/tokenizer files only.

| Setting | Qualification value |
| --- | --- |
| Algorithms | Native GRPO and uniform-credit RLTT, separately |
| Precision / parallelism | FP32; one learner, TP=PP=CP=DP=1 |
| Updates / logical completions | 3 × 4 prompts × 8 samples = 96 per logical run |
| Prompt / response caps | 1,024 / 2,048; natural EOS retained |
| Train sampling | Temperature 0.9, full vocabulary, seed 42 |
| Optimizer | MCore `adam`, beta=(0.9,0.999), epsilon=1e-8, LR=1e-6, WD=0.1, clip=0.1 |
| Learning-rate schedule | Constant three-update horizon in every phase |
| RLTT | Response mean, uniform credit, frozen initial reference, KL coefficient 0.001 |
| Activation handling | Physical-block recomputation; RLTT loop/layer interval 1 and query chunk 256 |
| Rollout cache | 1,536 physical 16-token pages, two active sequences |
| Held-out check | All eight selected development rows, greedy, before training and after each update |
| Evidence | Rollout/train dumps, per-step gradient norm, checkpoint every update, phase PID/exit receipt |

The public original RLTT stack uses AdamW8bit, a different grader and additional
source-specific choices. These native settings are a declared qualification
profile, not an original-stack reproduction or a convergence hyperparameter
selection. The eight-row held-out check is a lifecycle gate, not an accuracy
estimate. A useful signal is not guaranteed in 96 completions; zero-signal results
must be retained and diagnosed before extending the run.

For the official FP32 Ouro geometry, the declared KV pool alone contains 9 GiB
of K/V tensor storage. This follows from its tensor shape and excludes weights,
gradients, Adam state, activations, reference backups, graph/allocator overhead
and other processes. Actual resident capacity is unqualified. Ray's resource
fractions do not partition physical memory.

## Execution and recovery

Use an existing compatible `0z5a` environment and a completely admitted resource
window with an already reserved Ray cluster. Do not start or stop another cluster,
install dependencies or terminate processes. The local complete framework parser
still requires unavailable Triton; the command tests below capture the recipe
boundary and do not replace that parser or worker startup.

```bash
python benchmarks/prepare_native_qualification.py \
  --source /task/math-v2-rltt-source-release --output /task/qualification

python benchmarks/run_native_qualification.py \
  --packet /task/qualification --model /models/Ouro-1.4B-Thinking \
  --output /task/qualification-runs --ray-address RESERVED_HOST:PORT \
  --algorithm rltt --phase continuous --execute
```

Run `split` in a separate process, then `resume` only after the split phase
naturally returns zero. Use the same `--output`, packet, sources, optimizer
and sampling settings; `split` stops after update two while retaining the
three-update horizon. `resume` starts a fresh driver and restores iteration one
to perform update three. Repeat the three phases independently for GRPO.
The launcher rejects `--ray-address local`, checks frozen input hashes, records
the child PID before waiting, and uses ordinary `wait()` without a timeout or
signal. Omitting `--execute` prints the exact command and launches nothing.
The controller's execute path has not run in this delivery. Revision arguments
declare the intended inputs; they do not authenticate model files or worker
imports. Verify actual model hashes and loaded source identities before execution.
The retained dry-run commands reference a local tokenizer/configuration directory,
which contains no weights and must be replaced by the verified model directory.

Do not infer worker shutdown from driver exit alone. Verify actual Ray worker
identities/source paths at startup and fresh-worker replacement on resume, plus
natural process completion and resource handback. The two logical runs repeat
the same 96 completion identities; counting 192 executions as 192 independent
training samples would be incorrect. Checkpoint and publication IO must fit the
admitted local storage budget before launch. Retain complete state until its
comparison and all readers finish.

## Learning-signal audit

The older short-run auditor assumes zero weight decay and a separately exported
actor score field. Its rule that zero gradient implies an unchanged publication
is unsuitable for Adam with weight decay or momentum, and RLTT's training dump
need not contain the ordinary policy-loss score capture.

`audit_native_learning_signal.py` checks complete saved groups and sample indices,
source prompt-token hashes, actual boxed-answer regrading, rollout/train trace
alignment, fixed-depth policy metadata, finite response fields and saved gradient
norms. It requires a **joint** mixed-reward group, nonzero masked advantages and
finite nonzero gradient in at least one update. GRPO scores must match behavior
scores; RLTT's initial frozen-reference scores must match initial behavior scores
within fixed `atol=1e-4, rtol=3e-5`. Later reference score fields must be finite;
the audit does not independently establish their frozen-policy identity.

```bash
python benchmarks/audit_native_learning_signal.py \
  --run /task/qualification-runs/rltt/continuous \
  --packet /task/qualification --input-audit /task/native-qualification-input-audit.json \
  --algorithm rltt --output /task/continuous-signal.json
```

This audit deliberately returns only a signal-gate result. A pass still needs
independent checks of actual parameter deltas beyond decay, advancing Adam
moments, complete model/optimizer/scheduler/RNG restoration, held-out output
equality and fresh Ray worker identity. Existing chunkwise checkpoint comparison
tools can be reused for actual checkpoint files after the resource window opens.
No checkpoint comparison or new worker run has occurred in this delivery.

## Current evidence and speed table

The final focused suite has 136 unique CPU passes: 121 existing recipe checks,
seven new command/data checks and eight synthetic saved-evidence audit cases.
Synthetic fixtures test rejection of wrong rewards, prompt tokens, sample indices,
nonfinite gradients and mismatched initial reference scores. Their passing result
does not represent model generation, an optimizer update or real reward learning.

| Required comparison | Prior short recipe | Prepared qualification | Measured speed ratio |
| --- | --- | --- | --- |
| Adam / long-response command boundary | SGD and 48-token defaults | Explicit settings verified | Not a timing comparison |
| Actual official online RL | No new result | NOT_RUN | — |
| Held-out reward / time-to-quality | NOT_ESTABLISHED | NOT_RUN | — |
| Fresh-worker Adam recovery | NOT_RUN for this profile | Protocol and evidence gates prepared | — |
| Resident GPU peak / high-load stability | NOT_MEASURED | NOT_MEASURED | — |

One input-audit attempt passed a `Path` where the existing Dataset requires a
string and failed before tokenization. The audit invocation was corrected; the
production data loader is unchanged. Both receipts are retained. The final
token audit initializes no CUDA and runs no Ray/training phase. The original
W00–W14 baseline, model, scaling and convergence requirements remain open.

Raw inputs, the measured token-audit script, all test/input-audit receipts and
their hash manifest are in `benchmarks/results/native-reward-qualification/`.
Validation timings measure CPU checks only and are excluded from speed ratios.
