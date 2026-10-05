# Native qualification evidence chain

Independent state and output checks can pass while referring to different
published weights. This increment joins the retained model, source, process,
checkpoint, output and useful-update evidence. The launcher verifies all seven
Ouro model/tokenizer files before starting a child and records their hashes.
Resume requires the split phase's model manifest. The auditor requires each
worker's reported model path to match its driver's verified path.

Each arm retains four publications: the initial policy and the policy after
each of three updates. The auditor recomputes the native digest from sorted
safetensor filenames and bytes, checks it against the held-out output trace,
and compares every published physical tensor exactly with the initial FP32
conversion or the corresponding actor checkpoint. Missing, extra or duplicate
tensors fail, including frozen tensors. Production multi-file safetensor
publications are supported; the qualified checkpoint layout is unsharded FP32
Ouro with explicitly saved AdamW semantics.

All prior source, fresh-worker identity, checkpoint recovery, Adam
counterfactual, sampled-output and reward checks are rerun. At least one
update must simultaneously have a fresh Adam moment, a parameter change beyond
zero-gradient momentum/decay, mixed rewards, advantages and nonzero gradients.
The report merges consumed-file hashes and rejects disagreement between
audits. DCP tensors retain semantic value fingerprints; the report does not
claim byte hashes for whole DCP shard files or live-memory attestation.

```bash
python -m benchmarks.audit_native_qualification \
  --continuous RUNS/rltt/continuous --resumed RUNS/rltt/resumed \
  --packet PACKET --input-audit input-audit.json \
  --model VERIFIED_MODEL --model-manifest model-manifest.json \
  --source-manifest sources.json --algorithm rltt --output qualification-audit.json
```

Repeat independently for GRPO. Launch commands also require `--model-manifest`;
its revision and seven unique `path`, `bytes`, `sha256` entries must describe
`config.json`, `model.safetensors`, `merges.txt`, `vocab.json`, `tokenizer.json`,
`tokenizer_config.json` and `special_tokens_map.json`. These are trusted
task-owned manifests, not signatures. Resource/process/reader handback remains
a separate audit. Three updates cannot establish reward convergence.

## Validation

The final targeted suite has **109 distinct CPU passes in 19.69 s**, comprising
20 new cases and 89 existing cases. The 57 MCore/DCP warnings remain in the logs.
The result package contains this CPU suite, earlier overlapping
runs and the two corrected fixture failures. The complete-chain tests use
synthetic worker/output records with real CPU AdamW and DCP files; their pass
is not a Ray/CUDA run. Controls reject changed model/tokenizer bytes, a different
manifest or worker model path, incomplete preflight, changed publication bytes,
rehashed wrong-update tensors, missing/duplicate tensors or versions, and
disjoint useful-update signals. Invalid preflight is rejected before `Popen`.

A separate real tiny Ouro/RLTT test generates samples with the native engine,
performs three AdamW updates against a frozen reference and publishes each
policy. All 16 physical tensors match at all four publication versions. Each
update has mixed token-parity rewards, nonzero gradients and a nonzero parameter
delta. Its DCP files are component fixtures, not full MCore checkpoints. The
production safetensor writer additionally repartitions the final state into two
files with unchanged values; that layout control is not a second generation run.
This tiny model uses synthetic rewards and its existing test configuration,
not the official Ouro qualification profile.

| Tiny CPU update | Mixed groups | Gradient norm | Parameter delta | Actor/native maximum score error |
|---|---:|---:|---:|---:|
| 0 | 2 | 1.585172 | 0.0290883 | 2.38e-7 |
| 1 | 2 | 1.460079 | 0.0220454 | 2.38e-7 |
| 2 | 2 | 1.056497 | 0.0164092 | 4.77e-7 |

| Comparison | Baseline throughput | FlashRLT throughput | Speedup | Peak-memory reduction | Reward convergence |
|---|---:|---:|---:|---:|---|
| Official Ouro continuous/fresh-worker recovery | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_ESTABLISHED |
| RLTT / FlashLoop / slime / Megatron campaign | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_ESTABLISHED |

Raw logs/XML, extracted tiny update measurements and source/artifact hashes are
in `benchmarks/results/native-qualification-chain/`. No GPU speed or memory
measurement is inferred from these CPU correctness tests. The official campaign
still requires an admitted GPU window and an existing compatible environment.
