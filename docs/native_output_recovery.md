# Native output and held-out recovery qualification

Checkpoint equality alone cannot establish that a resumed worker generated the
same samples, used the same publication, or evaluated every held-out example.
`benchmarks/audit_native_outputs.py` checks those outputs for the frozen
three-update Ouro Thinking qualification. It builds on the runtime records in
[draft #20](https://github.com/0z5a/vime/pull/20) and the raw learning-signal audit.
No training or serving behavior changes in this increment.

For each continuous/resumed run, it reopens all 96 training samples and all
32 held-out samples: eight development questions at completed updates 0,1,2,3.
It verifies actual train/development IDs against the frozen selection and token
audit, includes every development row in order, and checks seeds, R4 traces,
greedy evaluation settings, policy versions, EOS/truncation and finite scores.
Saved response text must decode from the saved response tokens; rewards are
regraded from that text and the fixed label.

Each evaluation has one publication digest. Evaluations 0–2 must match the
corresponding subsequent training rollouts; the final evaluation must use a
fourth distinct publication. The first three resumed evaluations belong to the
split worker, while evaluation 3 belongs to the newly started worker. The raw
startup and process records are rechecked, not replaced by a detached summary.

Across runs, tokens, answers, rewards, statuses, metadata and semantic traces
must match. Only `runtime_epoch` and `request_id` are excluded from direct trace
equality: epoch is separately bound to the phase's worker record, and held-out
request IDs must be nonempty and unique within their round. Scores, advantages,
returns, KL, reference/current policy scores and gradient norms use the declared
FP32 comparison tolerance `atol=1e-5, rtol=3e-5`; maximum absolute differences are
reported. Masks match exactly. This tolerance is fixed before official model
execution and is not a BF16 qualification.

An identical pair with no mixed-reward/nonzero-advantage/gradient update remains
`output_equivalence_pass=true`, but `output_and_signal_pass=false` and the CLI
returns 1. The report hashes all 39 consumed evidence inputs and the local
tokenizer files, and preserves both four-point reward curves. It does not turn
this short qualification into a convergence experiment.

```bash
python -m benchmarks.audit_native_outputs \
  --continuous RUNS/rltt/continuous --resumed RUNS/rltt/resumed \
  --packet PACKET --input-audit INPUT_AUDIT.json \
  --tokenizer LOCAL_TOKENIZER --algorithm rltt --output output-audit.json
```

Use `--algorithm grpo` for the matched GRPO run. The tokenizer loads from local
files with remote code disabled. This command consumes completed, task-owned
experiment dumps; it does not launch workers. Weight identity, checkpoint/Adam
state, frozen-source membership and full-framework/GPU execution still require
their separate evidence. The older/fused Adam format restriction is unchanged.

## Validation

The final suite has **245 distinct CPU passes** in 34.25 seconds, with 42 retained
MCore/DCP warnings. It includes 20 new output-gate cases, three expanded existing
native sampling/RLTT/Adam tests, and 222 existing scheduling, runtime, checkpoint,
signal, recipe and adapter checks. Earlier 18- and 3-case runs overlap it.

The CLI controls use synthetic saved runs and Ray/process records with a real
WordLevel tokenizer. Sixteen corruption cases cover missing rounds/rows,
incorrect IDs/prefixes/text/seeds/versions/epochs/EOS, nonfinite scores, wrong or
stale publications, changed output, changed training tensors, changed gradient
norm and training text that does not decode from its tokens. Separate controls
exercise zero learning signal and a small finite score difference.

The three existing model tests generate real tiny native samples, compute
token-parity training rewards, perform AdamW updates and restore actor/reference/
optimizer state into fresh objects in the **same process**. Their held-out
tokenizer and Ray transport are doubles. The new comparator consumes their raw
held-out sample dictionaries before and after the recovered update:

| Tiny CPU model | Paired held-out samples | Tokens/reward/semantic trace | Maximum score difference | Speedup |
|---|---:|---|---:|---|
| Ouro | 8 | Exact | 0 | NOT_MEASURED |
| Nanbeige | 8 | Exact | 0 | NOT_MEASURED |
| Huginn | 8 | Exact | 0 | NOT_MEASURED |

The retained official Ouro Thinking tokenizer also passes 40 answer-replay
controls over the 12 training and eight development inputs. These use known
labels and invented scores, and verify decoding, reward grading and EOS ID 2;
they are **not model-generated answers**. Six local configuration/tokenizer files
match the retained model manifest at `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`;
all input prefixes match the retained qualification evidence. No weight file or
remote node was read for this check. No CUDA initialization or environment
mutation occurred.

| Required E2E comparison | Baseline throughput | ScaleRLT throughput | Speedup | Peak memory reduction | Reward convergence |
|---|---:|---:|---:|---:|---|
| Official Ouro Thinking continuous/fresh-process recovery | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_ESTABLISHED |
| RLTT / FlashLoop / slime / Megatron campaign | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_ESTABLISHED |

Raw logs/XML, tokenizer-control records, validation receipt and source/artifact
hashes are in `benchmarks/results/native-output-recovery/`. CPU suite elapsed
time is not a model speedup. The full experiment matrix and paper's quality and
system claims remain pending their actual measurements.
