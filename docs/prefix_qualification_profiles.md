# Matched B-stage native RLTT qualification

The original three-update profile enables model, loop and layer recomputation.
The first prefix learner schedule rejects those switches until the joint C
implementation is qualified. Two additional frozen profiles make B's official
model check executable through the existing complete Ray recipe:

| Profile | Learner schedule | Suffix wave | Model/loop/layer/query rematerialization | Workload |
|---|---|---:|---|---|
| `b-baseline` | Existing MCore | 0 | Off | Unchanged three-update qualification |
| `b-prefix` | Ouro prefix | 2 | Off | Identical inputs and budget |

These are development qualification controls. They do not replace the strong
engineering baseline for the final A campaign. The default `legacy-remat`
profile remains byte-identical to the previous release and still supports its
GRPO and RLTT configurations. The two new profiles select RLTT only.

Both retain the same official Thinking revision, fixed full depth, FP32, Adam
configuration, RLTT objective, masks, sampling parameters, prompt/response limits,
12 training questions and 8 development questions. One continuous arm still
requires 96 new training completions and 32 development completions. The
split/resume arm repeats the same update budget using fresh workers. This is
startup and recovery qualification, not reward convergence or a performance
campaign. Separate output roots prevent mixing the two profiles.

Preparation uses the existing entrypoint:

```bash
python -m benchmarks.prepare_native_qualification \
  --source FROZEN_MATH_SOURCE --output B_BASELINE_PACKET --variant b-baseline
python -m benchmarks.prepare_native_qualification \
  --source FROZEN_MATH_SOURCE --output B_PREFIX_PACKET --variant b-prefix
```

Run each packet through `benchmarks.run_native_qualification` with the existing
model/source manifests and an already reserved Ray cluster. The first real
MCore two-update GPU fixture remains a prerequisite. Re-freeze runtime sources
at the candidate commit; earlier manifests intentionally refer to older code.
The original 0z5a environment requirement remains unresolved for this task.
No environment installation or base-Python substitution is implied by a packet.

The actual learner startup record now includes the schedule and rematerialization
configuration. When RLTT runtime reporting is enabled, a successful
`train_one_step` writes a separate step receipt after forward/backward, optimizer,
scheduler and prefix cursor commitment. It records the completed dispatch,
process identity, logical sample count, gradient norm and scheduler counters.
The B-profile audit requires all six phase-step receipts, joins them to their
startup worker and rejects missing dispatches, the wrong schedule, changed
recomputation, a different PID, invalid norms or incorrect counter advancement.
This is saved-record evidence, not live-memory or binary attestation; the
existing source, output, useful-Adam-update and recovery checks still apply.

Validation: **86 CPU checks passed; 2 actual CUDA checks skipped** in 20.92 s.
Recipe tests execute the real argument-forwarding code with explicitly replaced
Ray/entrypoint transports. Audit tests use synthetic worker and step records;
MCore identity tests use existing CPU config classes with wrapper doubles.
They establish neither an official GPU update nor an actual Ray dispatch.
The CUDA fixture has been extended to check receipts from its real train step
when it is eventually run in the admitted environment.

The prepared B packets have identical train/development bytes. Their prompt-row
contract is retained from the prior measured input audit after that byte
comparison; it was not re-tokenized here. The old profile SHA remains
`d7bf3edcd0a05ee25b93bf7abaf1f739a4f8da4a769ac5fc00cf87a423e67f9c`.
Six phase commands are prepared, none executed. Profiles, raw test logs/XML,
commands and source hashes are in `benchmarks/results/prefix-qualification-profiles/`.

| Required result | B baseline | B prefix | Speedup | GPU-memory reduction | Reward convergence |
|---|---|---|---|---|---|
| Actual MCore CUDA two-update control | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_ESTABLISHED |
| Official Ouro complete qualification | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_ESTABLISHED |
| Formal multi-seed online quality campaign | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_ESTABLISHED |

The official seven-file Thinking cache is now present on the H20. Its download
and independent SHA verification are preparation, not model execution. The
cache remains needed for unfinished experiments and is not cleanup-eligible.
