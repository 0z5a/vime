# External draft collect-only validation (2026-09-27)

Scope: the optional LM-head-input collector in this PR, stacked on the #452 feature contract. The output is a versioned feature batch for later draft training; this PR does not add a draft optimizer or draft weight publication.

## RTX 5090 correctness

| Check | Result |
| --- | --- |
| Contract, provenance, CPU/CUDA collector tests | 18 passed with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, including the CUDA owned-storage and head-logit reconstruction test |
| Qwen3-0.6B, collector off | Full vLLM rollout → Megatron reference/actor forward → backward/optimizer → checkpoint → weight sync passed; `draft-off-smoke.log` |
| Qwen3-0.6B, collector on | Same full path passed; `draft-on-smoke3.log` |
| Real-model export | One ready batch, 35 selected tokens in two groups, 311,236,608 bytes including a tied LM-head snapshot; token targets, source/head versions and model/tokenizer digests validated |
| Immutable head | Exported `151936 × 1024` BF16 head matched the initial Qwen3 checkpoint bitwise; the collector owns its snapshot before the optimizer step |
| Six-round version progression | Target/head source versions 1–6 each produced a ready manifest; the six head snapshots had distinct SHA256 digests |

The 18-test command on the 5090 host was `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=1 <child-venv>/bin/python -m pytest -o addopts= -q tests/utils/test_draft_feature_contract.py tests/backends/megatron/test_draft_feature_collector.py tests/rollout/test_draft_feature_provenance.py`, with the VIME and pinned Megatron source directories on `PYTHONPATH`.

The smoke runs used a math reward that expects boxed answers. Their zero gradients make them functional checks, not training or speed evidence. The repeated comparison below uses a deterministic reward of `sample.index % 2`, which gives opposite rewards within each two-sample GRPO group. The six-round baseline under this reward had nonzero gradient norms in five rounds.

For a fixed-data correctness check, one real baseline rollout was saved and replayed through the baseline, collector-off and collector-on training paths. All three completed a backward pass with the same nonzero gradient norm; their observed metrics matched exactly:

| Path | Mean response tokens | Mean raw reward | Loss | Entropy metric | Gradient norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline capture | 8.75 | 0.5 | 0 | 0.250118047 | 28.461925036 |
| Collector off, fixed replay | 8.75 | 0.5 | 0 | 0.250118047 | 28.461925036 |
| Collector on, fixed replay | 8.75 | 0.5 | 0 | 0.250118047 | 28.461925036 |

The replay payload and exported feature/head files were removed after checking the manifests. This check uses one rollout; the three independent pairs below measure timing with live generation.

The current vLLM response did not include per-sample `weight_version`, so all exported sequence `weight_versions` are empty. The manifest preserves that unknown rollout provenance explicitly; its target/head source version remains verified. A consumer requiring rollout policy provenance must reject these batches until serving supplies it.

## Same-node speed comparison

Each arm uses a fresh process, the same Qwen3-0.6B checkpoint, two prompts × two samples, maximum 16 response tokens, one RTX 5090, and the same Ray/Megatron/vLLM configuration. Round 0 is warmup; five later rounds define each run median. The driver round wall includes generation, training, offload and weight sync. Checkpoint save is disabled during speed runs because it was separately verified in the smoke run. The full run wall includes startup. Generated response lengths are reported because they can change the wall time independently of collection.

| Pair / order | Off, median round s | On, median round s | Off/on speed ratio | Off/on useful tokens/s | Off/on mean response tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 / off→on | 8.233 | 14.901 | 0.553× | 5.438 / 3.031 | 11.15 / 11.75 |
| 2 / on→off | 11.918 | 11.645 | 1.023× | 4.381 / 4.534 | 12.70 / 13.00 |
| 3 / off→on | 8.575 | 11.666 | 0.735× | 5.575 / 3.739 | 12.10 / 10.90 |
| **Median paired ratio** | | | **0.735× observed** | | |

Original logs are under `/home/gongji/0z5a/work/vime-20260927/runs/` on the 5090 host. Verified feature and head files from completed runs were removed to preserve disk space; the version, token-map, digest, SHA256-distinctness and byte-count checks above were recorded before removal. The test-only runner and metrics extractor are in the task evidence directory. All model files stayed on the remote host.

Collection was slower in two of three pairs. The measured median paired round ratio is 0.735×, with variable generated lengths and node load; no speed benefit is claimed. Each active round copied an approximately 311 MB immutable head snapshot plus selected features, with observed copy times between 0.55 and 1.35 seconds across the first two pairs.
