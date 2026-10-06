# ScaleRLT — IQuest semantic qualification

The fixed official source and sixteen physical checkpoint headers are in `reference/`. No weight tensor body is included. The unique parameter count is 39,794,696,320. These are tiny CPU FP32 tests of the checkpoint's architecture, not a 40B run or RL reward curve.

| Actual comparison | AUTO SDPA | MATH SDPA | Status |
| --- | ---: | ---: | --- |
| Paired AdamW steps, two seeds × four lengths × ordinary/recomputed × two steps | 32 | 32 | All executed |
| Maximum logits error against unmodified official token-by-token autograd | 7.15256×10⁻⁷ | 6.25849×10⁻⁷ | 32/32 pass original 10⁻⁵ absolute/relative |
| Maximum whole-gradient relative L2 error | 3.21867×10⁻⁷ | 3.04581×10⁻⁷ | 32/32 pass original 2×10⁻⁶ |
| Maximum parameter-delta difference | 2.39909×10⁻⁶ | 1.98185×10⁻⁶ | Original 10⁻⁷ absolute/10⁻⁵ relative gate fails |
| Complete update checks passed | 24/32 | 24/32 | **FAILED_NUMERICAL_GATE** |
| Adam moments and counters | 32/32 | 32/32 | Original checks pass |
| Speed ratio / GPU memory / RL reward convergence | NOT_MEASURED | NOT_MEASURED | No claim |

The unmodified public full-prefill path differs from decode by 0.41336 at length eight and 0.50415 at length 65. Cache-free and HF checkpointed training lose the first-loop global KV. The default cache/mask API also fails in the existing Transformers 4.54.1 environment; explicit mask inputs isolate the mathematical tests without changing source or packages. All failed records remain alongside the complete matrices.

The experimental replay is not registered as a learner provider. See [the contract report](../../../docs/iquest_semantic_qualification.md) for the complete boundaries. No failed check is promoted to full update qualification.

Run in the existing matching environment:

```sh
python benchmarks/audit_iquest_replay.py --output /tmp/scalerlt-iquest-auto.json
python benchmarks/audit_iquest_replay.py --sdpa-math --output /tmp/scalerlt-iquest-math.json
python benchmarks/probe_iquest_source.py --case decode --length 65 --explicit-mask --output /tmp/scalerlt-iquest-decode.json
```

Both complete replay audits currently return exit code 1 after writing all sixteen cases and thirty-two paired steps. The decode probe records an agreement failure in its JSON. Its exit code alone is not a numerical pass. Do not replace existing result files. The source manifest and `SHA256SUMS.json` bind the retained artifacts; original source notices and LICENSE are included.
