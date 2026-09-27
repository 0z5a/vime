# Colocate span accounting results (2026-09-27)

Validation ran with `/home/gongji/0z5a/bin/python` on `gj-5090-2`. The candidate is the accounting fix in this PR; A is its parent (`01dc1b77`). The CPU benchmark measures one context-manager invocation, not a VIME training step. Each arm ran in a fresh Python process with 50,000 invocations per cycle. The first of six cycles was discarded; the table uses the median of the remaining five. Pair order was A→B, B→A, A→B. Per-cycle measurements are in [offload_spans_benchmark_20260927.jsonl](offload_spans_benchmark_20260927.jsonl).

| Mode | Pair | A, ns/span | B, ns/span | A/B speed ratio |
|---|---:|---:|---:|---:|
| Disabled | 01 | 1112.55 | 1120.30 | 0.993× |
| Disabled | 02 | 1313.94 | 1082.24 | 1.214× |
| Disabled | 03 | 1125.07 | 1095.26 | 1.027× |
| **Disabled, median pair ratio** | | | | **1.027×** |
| Enabled | 01 | 1466.25 | 1495.13 | 0.981× |
| Enabled | 02 | 1458.75 | 1496.85 | 0.975× |
| Enabled | 03 | 1469.39 | 1496.17 | 0.982× |
| **Enabled, median pair ratio** | | | | **0.981×** |

The disabled-path measurements vary enough to make a speed gain unproven. Enabled spans cost about 2% more per invocation in this CPU microbenchmark while recording nested intervals correctly. These numbers do not establish an E2E training speedup.

| Validation | Result |
|---|---|
| `python -m pytest -q tests/observability/test_offload_spans.py` | 8 passed in the `0z5a` environment |
| `ruff check` on the changed Python files; `ruff format --check` on the span module and its tests | Passed |
| RTX 5090 VIME training A/B | Not run: the existing `0z5a` environment lacks `torch_memory_saver` |
