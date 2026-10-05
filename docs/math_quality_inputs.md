# Audited MATH inputs for native recurrent RL

This change prepares a reproducible input pack for the pending online reward experiments. It does not report model accuracy, reward convergence, throughput or GPU memory. Training uses a reconstructed public MATH corpus, not RLTT's unavailable private JSONL or an exact reproduction of its prompt template. All matched comparison arms must use the same prepared files, tokenizer profile, sampling budget and reward code.

## Immutable sources and transformation

The source lock pins [EleutherAI/hendrycks_math at 21a5633](https://huggingface.co/datasets/EleutherAI/hendrycks_math/tree/21a5633873b6a120296cce3e2df9d5550074f4a3) and [HuggingFaceH4/MATH-500 at 6e4ed1a](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/tree/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be). All 18 downloaded data/card/config files are checked against their published size and LFS SHA256 or Git blob SHA1. A SHA256 receipt is retained for every file. No credentials are required. Existing files are reused only after verification; corrupt files are not silently replaced.

All 7,500 training and 5,000 test rows are read. Deduplication hashes case-sensitive NFC text with Unicode whitespace collapsed; it does not apply NFKC, case folding, fuzzy matching or semantic similarity. Thus this audit cannot rule out paraphrased duplicates or pretraining contamination. Source question text is unchanged in the actual prompt. A fixed instruction is appended: `Please reason step by step, and put your final answer within \boxed{}.` Solution text never enters the prompt.

| Artifact | Rows | Rule |
|---|---:|---|
| Full prepared training | 7,498 | Remove one normalized duplicate and one overlap with the full test split |
| Development | 256 | Lowest fixed SHA256 values from the 4,500 non-MATH-500 test questions |
| MATH-500 held-out | 500 | Preserve all source questions, source order and labels |
| Test reserve | 4,244 | Remaining test questions; not a training pool |
| Ouro thinking, prompt cap 1,024 | 7,480 | Explicit profile excludes 18 over-cap training rows; full training stays intact |

All four primary partitions are disjoint under the declared normalization. Every MATH-500 question matches a question in the full test split; none overlaps the original training split. `algebra/train/959` duplicates `algebra/train/925` with the same label 2. `geometry/train/433` overlaps `precalculus/test/202`, both label 315. Both exclusions and all original row IDs are in the manifest. Row indices are zero-based within each pinned source file. Two unknown training difficulty labels (`Level ?`) remain in the data as null metadata; these rows are not dropped.

Four explicit label repairs are bound to the SHA256 of the original solution. Two source answers use unbraced `\boxed 2` / `\boxed 9`; their labels are 2 and 9. Two number-theory solutions have empty boxes; divisibility proves that every candidate is composite, giving label 0. The full reasoning and source hashes are in `benchmarks/datasets/math/label_repairs.json`. No solution text or runtime grader is changed. All 500 MATH-500 supplied answers exactly match extraction from their supplied solutions.

## Token budget audit

Six official configuration/tokenizer files, 4,796,519 bytes in total, are SHA256-verified at [ByteDance/Ouro-1.4B-Thinking revision 3aaa222](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking/tree/3aaa2224253a92ca45cf2e3d427c360e1ef9c93d). No model weights are downloaded for this audit. The actual native tokenizer loader returns `GPT2TokenizerFast`, EOS token ID 2. All 12,498 prepared rows are tokenized in three profiles through the actual VIME `Dataset`, with `add_special_tokens=False`; 37,494 per-row token counts and token-ID hashes are retained. Quantiles use `sorted[floor((n-1)*q)]`.

| Split | Plain max | Chat max | Thinking max | Thinking p95 | Thinking over 1,024 |
|---|---:|---:|---:|---:|---:|
| Full training | 2,058 | 2,078 | 2,080 | 310 | 18 |
| Development | 973 | 993 | 995 | 257 | 0 |
| MATH-500 | 932 | 952 | 954 | 274 | 0 |
| Reserve | 1,539 | 1,559 | 1,561 | 281 | 7 |

The selected Ouro input profile uses the official chat template with `enable_thinking=true` and a 1,024-token prompt cap. `train-ouro-thinking-p1024.jsonl` contains raw prompt records, not pre-rendered chat strings. Its 18 exclusions are explicit and reproducible. No held-out question is removed. The reserve requires a larger prompt cap if later evaluated. Response tokens must be budgeted separately; these counts do not establish model context capacity or worker memory fit. Other model families need their own tokenizer profiles. Native chat-template handling through complete training rollout remains a separate integration gate.

## Validation and speed comparison

| Comparison | Parent #14 | This change | Speedup |
|---|---|---|---|
| Pinned quality inputs | No audited corpus/length roster | Source hashes, partitions, repairs and per-row tokenizer evidence | Not a timing measurement |
| Actual data/reward path | No full-corpus input audit | 12,498 rows loaded; 12,498 canonical boxed answers score 1; 12,498 missing-box controls score 0 | Not measured |
| Full online RL / held-out reward | Pending | Pending | Not measured |

The reward checks use the real `rm_type=math` route, with no signal-based timeout or process termination. They are known-answer plumbing checks, not generated model answers. Grading all 13,000 source solution records (including the intentional MATH-500 subset) produces exactly the four expected source-format failures. Regenerating all prepared JSONL and the manifest is byte-identical. Unit/contract tests protect hash mismatch handling, corrupted-file reuse, normalization, disjoint splits, conflicting duplicates, source-bound repairs, unknown levels and solution isolation, alongside existing reward/evaluation tests.

Initial failures are retained: namespace-package collision during preparation/test collection; a subsequently discovered unknown difficulty label; an audit launched before successful preparation. They were corrected without installing dependencies. All final job exit statuses and test counts are recorded in `benchmarks/results/math-quality-inputs/validation.json`. There is no model execution, new Ray worker, optimizer update or reward convergence result in this dataset change.

## Reproduce

Use an existing compatible environment; no installation or environment update is required by these commands. Set `PYTHONPATH` to this checkout. Paths below are task-owned outputs outside the repository.

```bash
python -m benchmarks.download_math_data --output /path/to/math/raw
python -m benchmarks.prepare_math_data --raw /path/to/math/raw --output /path/to/math/prepared
python -m benchmarks.audit_math_data --raw /path/to/math/raw --prepared /path/to/math/prepared --output /path/to/math/audit.json
python -m benchmarks.download_math_data --sources benchmarks/datasets/math/ouro_tokenizer.json --output /path/to/math/tokenizer
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m benchmarks.audit_math_tokenization --prepared /path/to/math/prepared --tokenizer /path/to/math/tokenizer/ByteDance/Ouro-1.4B-Thinking/3aaa2224253a92ca45cf2e3d427c360e1ef9c93d --output /path/to/math/tokenization
pytest tests/test_math_data_preparation.py tests/test_rm_math.py tests/rollout_backends/test_native_rlt_evaluation.py
```

The committed evidence includes all source/output hashes and row rosters; full question text is regenerated from the immutable upstream sources. Development inputs are for pilot choices. Freeze quality thresholds, non-inferiority tolerance and sampling/seed/budget contracts before inspecting held-out model results. The planned primary comparison still requires three independent seeds, actual logical rollout counts, lifecycle timing and fresh-worker checkpoint recovery.
