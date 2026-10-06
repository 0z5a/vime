# ScaleRLT — IQuest 40B contract and semantic qualification

The official loop checkpoint is [IQuest-Coder-V1-40B-Loop-Instruct](https://huggingface.co/IQuestLab/IQuest-Coder-V1-40B-Loop-Instruct/tree/61e8589747f6987ec7725e4ffe205f7a84561bd2), frozen at `61e8589747f6987ec7725e4ffe205f7a84561bd2`. This is W13 preparation. No official weight tensor was downloaded, and no 40B update, recovery, quality or reward convergence run has completed.

| Check | Actual result | Scope |
| --- | --- | --- |
| Source identity | Eleven source/config/tokenizer/index/license files hashed at the fixed revision | Original source retained |
| Physical checkpoint headers | 16/16, 883 BF16 tensors, complete index/name/shape/storage agreement | Bounded HTTP range reads |
| Unique parameters | **39,794,696,320** | Counted from tensor shapes, without multiplying by loops |
| Tensor payload size | 79,589,392,640 bytes | Excludes file headers |
| Verified header reads | 100,960 bytes including repeated length prefixes | Failed transfer traffic is not included; no speed claim |
| Downloaded weight tensor bodies | **0 bytes** | Tensor values are unverified |
| Tiny CPU cached forward/backward | PASS with explicit causal-mask input; all 21,800 parameters have finite gradients | Existing Torch 2.13.0 / Transformers 4.54.1; no environment update |
| Default cache/mask interface | FAIL: `IndexError` in existing HF mask builder | Public config records Transformers 4.55.4 |
| Cache-free and HF checkpointed paths | FAIL: missing first-loop KV | Unmodified source, explicit-mask input |
| Unmodified prefill/decode agreement | FAIL: max logits error 0.41336 at length 8; 0.50415 at length 65 | Same tiny parameters, tokens and explicit causal masks |
| Functional replay, AUTO SDPA | 32/32 output and whole-gradient checks pass; strict Adam gate 24/32 | Max gradient relative L2 3.21867×10⁻⁷; full update qualification FAIL |
| Functional replay, MATH SDPA | 32/32 output and whole-gradient checks pass; strict Adam gate 24/32 | Max gradient relative L2 3.04581×10⁻⁷; full update qualification FAIL |
| Official 40B update and fresh-worker recovery | **NOT_RUN** | Requires the original sharding/resource gates |
| Reward convergence and large-scale RL | **NOT_RUN** | Header/configuration evidence cannot qualify these |

The physical parameter count includes the untied token embedding and output head, eighty decoder layers, final norm, and eighty query gate projections. The two loop passes reuse physical layers. The gate tensors contribute 412,800 parameters; they are present in the checkpoint index.

The configuration specifies 5,120 hidden channels, 27,648 intermediate channels, forty query heads, eight KV heads and explicit head dimension 128. It uses two loops, a local window of 64, maximum total context 131,072, RoPE base 500,000 and RMS epsilon 10⁻⁵. [Frozen configuration](https://huggingface.co/IQuestLab/IQuest-Coder-V1-40B-Loop-Instruct/blob/61e8589747f6987ec7725e4ffe205f7a84561bd2/config.json).

The first loop stores global causal KV. The second loop combines attention to that global KV with attention to its own local KV through a learned, per-head sigmoid gate computed from the rotated query. The output norm occurs after the loop stack. The adapter must retain this data dependence and the gate gradients. [Frozen model implementation](https://huggingface.co/IQuestLab/IQuest-Coder-V1-40B-Loop-Instruct/blob/61e8589747f6987ec7725e4ffe205f7a84561bd2/modeling_iquestloopcoder.py).

The public implementation obtains the second loop's global KV from its cache object. Actual tiny calls reproduce missing-KV failures with `use_cache=False` and HF gradient checkpointing. The default cache/mask interface also fails in the existing 4.54.1 environment. Explicit additive causal masks use a supported model input and isolate model math without editing the official source or upgrading packages.

The public second-loop implementation reshapes head-major mixed attention directly into token-major hidden states. Its prefill also trims local KV before computing all query positions. The unmodified token-by-token path avoids the first layout error and retains the proper local window for each query. It is used as the independent numerical reference; the failed full-prefill result is retained, not silently replaced.

The experimental functional replay passes first-loop KV as differentiable tensors, mixes global/local attention in a consistent layout, and uses a causal window per query. Two initialization seeds, lengths 1/4/8/65, ordinary/recomputed graphs and two AdamW updates produce sixteen configurations and thirty-two paired steps per backend. Both AUTO and MATH SDPA pass the predeclared logits bounds (10⁻⁵ absolute/relative) and whole-gradient relative-L2 bound (2×10⁻⁶). Their maximum logits errors are 7.15256×10⁻⁷ and 6.25849×10⁻⁷, respectively.

The full update gate remains **FAILED**. Under the original 10⁻⁷ absolute/10⁻⁵ relative delta criterion, each backend passes 24/32 steps. Maximum delta differences are 2.39909×10⁻⁶ for AUTO and 1.98185×10⁻⁶ for MATH; parameter-state failures also remain. All first/second Adam moments and counters pass their original checks. Small gradient differences can affect Adam updates near zero; this is a diagnosis to investigate, not a reason to relax the gate. The prototype is not wired into the learner, and no throughput or GPU-memory improvement is asserted.

The first header attempt ended with a TLS EOF after fourteen complete headers. Their SHA values and the original auditor SHA were frozen; a new invocation completed the two missing headers. Both attempts remain archived. Initial CPU imports failed or waited on iCloud-evicted installed files. Exact installed-RECORD copies were recovered into task storage, with process-only source/metadata readers; original environment files and waiting processes remain untouched. The first incomplete update audit and both complete matrix results are retained.

Raw evidence is in [the pinned source manifest](../benchmarks/results/iquest-contract/reference/source_manifest.json) and [the complete physical-header audit](../benchmarks/results/iquest-contract/reference/weight-header-audit.json). The header audit SHA is `c284e84c02f38171e186e5c5b395f6771f3f2ff44b9896999d3991478fa370f8`. Reproduction scripts and failed logs are preserved in `benchmarks/` and `benchmarks/results/iquest-contract/`.

The frozen repository's modified MIT license and source notices are retained with the source snapshot. This work does not substitute the dense 7B/14B models for the loop checkpoint.
