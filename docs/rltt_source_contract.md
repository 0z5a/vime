# RLTT source and public-input contract

The earlier MATH v1 JSONL has `prompt/label` fields. The original RLTT converter
reads `problem/answer` instead: executing it on a v1 row produces an empty
problem and empty gold label. The separate source-prompt profile now carries
both interfaces. Its native JSONL and original-converter Parquet give exactly
equal rendered prompts, token IDs and labels for all **12,498** public rows.
No production training, reward or rollout code changes in this increment.

## Frozen inputs and executed checks

[RLTT 0618985](https://github.com/jonwill8/RLTT/tree/06189850bb23a1b1b715ad29768e68f36b1c4a19)
defines the zero-shot instruction, one user message and default chat options.
Seventeen source files are pinned by Git blob and SHA256 in
[`rltt_source.json`](../benchmarks/datasets/math/rltt_source.json).
The unchanged prompting module and selected unchanged answer/conversion
function bodies run with actual pandas and PyArrow. Selecting these functions
avoids the package's unrelated reward import and exit-time output writer;
this is not an execution of the full original framework or grader.

The official Ouro-1.4B-Thinking tokenizer is revision
`3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`. Actual VIME `Dataset` rendering and
the source conversion agree on every row. Default chat options equal
`enable_thinking=False` and differ from the v1 selected thinking profile.
Both use `add_generation_prompt=True`, then encoding without extra special
tokens. The source verl dataset uses these text-rendering options too; its
padding, collation, filter and DataProto execution have not been run locally.

All source rows, order, split membership, labels and metadata match the v1
public reconstruction. Its two deduplication exclusions and four explicit
solution-hash-bound repairs remain. This does not reconstruct the authors'
private JSONL. The instruction and chat mode are explicit experimental
differences from v1 and must not change between a systems comparison's arms.

| Split/profile | Rows | Maximum prompt tokens | Rows over 1,024 | Action |
|---|---:|---:|---:|---|
| Full training | 7,498 | 2,098 | 19 | Retained |
| Training `train-p1024` | 7,479 | ≤1,024 | 0 | Exactly the 19 listed rows excluded |
| Development | 256 | 1,013 | 0 | All retained |
| MATH-500 | 500 | 972 | 0 | All retained |
| Reserve | 4,244 | 1,579 | 8 | Retained; not training |

No prompt is truncated. Two fresh processes produce byte-identical outputs
for all 11 files (five JSONL, five Parquet and one manifest), plus identical
12,498-row token/label/problem hash rosters. The frozen v1 manifest and every
v1 split remain unchanged. The existing data/reward/evaluation regression
passes 62 distinct CPU tests without skips. These checks measure input
integrity; no generated answer or reward-convergence result is implied.

The prepared manifest SHA256 is
`a272932944329fa41b4da023b2d115606a1dbf3a364a1d1d300df6f251233b53`.
The selected 7,479-row JSONL SHA256 is
`f3e52c440246c455b7a7fa089c63ab08f69fdac4a29bc9c828fdd567b0a72a90`;
the matching original-converter Parquet SHA256 is
`cf07ae5aab414d35f8d2ca0bfd61e5ee0b9d15a8cc58570aa250a5cd459aef02`.
All remaining artifact hashes and over-cap row identities are in the raw
[`manifest`](../benchmarks/results/rltt-source-contract/inputs-manifest.json).

## Conditional batch accounting

The original unchanged argument/configuration functions were executed with
explicit four-GPU configuration, without importing Torch or querying CUDA.
RLTT pins verl 0.6.1. Its uninstalled wheel was hash-verified and inspected:
the inherited dataloader consumes `train_batch_size` prompt rows when
`gen_batch_size` is absent. RLTT sets that size to CLI P×G, then repeats each
generation row G times. Its second repetition aligns training metadata and
does not produce another generation. The actor worker normalizes its
minibatch by G and data-parallel degree; the actor takes one optimizer step
per minibatch with one PPO epoch.

Assume DP4, sequence parallel one, no private modification or generation
batch override, and 140 complete rollout batches. These are conditional
source counts, **not observed original training counts**:

| CLI P/G/A | Prompt rows / rollout | New completions / rollout | Optimizer steps / full actor call | Total completions | Actor calls | Total optimizer steps |
|---|---:|---:|---:|---:|---:|---:|
| 32/8/1 | 256 | 2,048 | 2 | 286,720 | 140 | 280 |
| 32/8/2 | 256 | 2,048 | 4 | 286,720 | 70 | 280 |
| 4/8/1, explicit nominal-count override | 32 | 256 | 1 | 35,840 | 140 | 140 |

A counts accumulated rollout batches. The launcher defaults to A=2, while
the CLI parser defaults to A=1. The final row changes the public CLI
configuration; it does not identify what the authors ran. Accumulation
persists across epoch boundaries. The launcher's default five-epoch schedule
estimates steps using ceil(N/P), while the dataloader drops partial P×G
batches; if the epochs end before the configured last step, pending
accumulation is not flushed. Actual available rows and runtime counters are
therefore essential.

The plan's nominal 280×32×8 = 71,680-completion extension is twice its nominal
paper budget but only one quarter of the conditional source count above.
A 2× extension of that conditional count is 573,440 new completions. Formal
budgets must be updated from observed prompt IDs, completion IDs, rank rows,
actor calls and optimizer steps in the complete original stack. Reusing a
rollout for extra optimizer passes does not create new samples.

The source launcher and CLI defaults also differ in temperature, scheduler
and accumulation. Source reward uses math-verify 0.8.0 plus its own fallback;
VIME's math reward is a separately declared grader. Source AdamW8bit uses
betas (0.9, 0.999); the existing short native SGD recipe is a lifecycle probe.
The inspected complete 335-entry public tree contains no required modified
Ouro implementation from the private RLTT checkpoint path. These differences
remain open baseline qualifications, not reasons to label a reimplementation
as original-source E2E.

## Performance and runtime status

| Comparison | Parent / original | Candidate | Speed ratio |
|---|---|---|---|
| Original converter on v1 vs new public input | Empty problem and label | All 12,498 rows match native prompts and labels | Not a timing comparison |
| Original RLTT vs ScaleRLT full RL | Not run | Not run | Not measured |
| GPU peak memory / high-concurrency stability | Not run | Not run | Not measured |
| Held-out time-to-quality | Not run | Not run | Not measured |

Official `AutoConfig` loading succeeds with existing Transformers 4.54.1 and
the hash-verified official `configuration_ouro.py`. Its loaded source hash is
`950443e32929047aa08d02abad2e1888bc1914b3db988d3d675f70787f65dafb`.
The checkpoint's custom Python configuration is needed in addition to JSON
and tokenizer files. This check loads no model or weights. Importing the
full local VIME/MCore parser separately fails because Triton is absent;
no dependency was installed or stubbed. Full Ray/CUDA startup, native Adam
updates, held-out generation and fresh-worker restoration remain required.

## Reproduction

Use the existing environment and the pinned raw MATH/tokenizer files from
[the v1 input procedure](math_quality_inputs.md). Check out RLTT at the exact
revision above into `$RLTT_SOURCE`; no installation is needed for these
source-function audits. Output/evidence directories must be fresh.

```bash
python benchmarks/prepare_rltt_math_data.py \
  --raw "$MATH_RAW" --source "$RLTT_SOURCE" --tokenizer "$OURO_TOKENIZER" \
  --output "$TASK_OUTPUT/math-source" --evidence "$TASK_OUTPUT/math-source-evidence"
python benchmarks/audit_rltt_accounting.py \
  --source "$RLTT_SOURCE" --output "$TASK_OUTPUT/source-accounting.json"
pytest tests/test_math_data_preparation.py tests/test_rm_math.py \
  tests/rollout_backends/test_native_rlt_evaluation.py -q
```

For VIME select the JSONL with `--apply-chat-template`, `--input-key prompt`
and `--label-key label`, without a thinking override. For original RLTT use
the same rows' Parquet or pass this JSONL through its converter. Freeze the
grader, sampling and optimizer independently; input equality alone does not
make two training runs comparable. Raw outputs, source receipts, conditional
accounting, the unchanged-v1 probe and both success/failure startup logs are
under [`benchmarks/results/rltt-source-contract/`](../benchmarks/results/rltt-source-contract/).
