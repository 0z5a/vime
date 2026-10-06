# ScaleRLT — Looped backend accuracy evidence

The official checkpoint is ByteDance/Ouro-1.4B revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`. These records audit the LoopCD/e8 inference baseline on the original H20. They do not establish ScaleRLT online training or reward convergence.

| Precision / attention backend | Numerical scope | Frozen tolerance | Observed evidence | Result |
| --- | --- | --- | --- | --- |
| FP32 / Torch | P4→D2/D3, prompt lengths3/4/5, guidance0/.3:12cases×9readouts;96forced continuations; lifecycle | abs/rel1e-4; exact readout state | Maxstate1.907e-5, logits3.528e-5, selectedLP1.907e-5;21ownership checks | PASS |
| BF16 / Triton | First P4 prefill, prompt3, guidance0; decode was not reached | stateabs/rel.02; selectedLPabs.03 predefined | Hidden998/2048 mismatches, maxabs.23046875 | FAILED; original record retained |
| FP32 / Triton | Same complete12case/108readout/96forced/lifecycle scope | Originalabs/rel1e-4; exact readout state | Maxstate1.907e-5, logits3.433e-5, selectedLP1.335e-5;21ownership checks | PASS |

The BF16 failure occurred before K/V, logits, selected log-probability, reduced-depth continuation or lifecycle checks. Those checks remain NOT_RUN. The archive contains no complete per-layer tensor trace, so it does not identify whether the difference arises from the reference, the production computation or both. Static source review identifies differing BF16 rounding paths in attention; that is a diagnostic hypothesis, not a verified cause.

The original FP32 qualification selected `attention_backend="torch"`. A later FP32/Triton request therefore required its own complete numerical qualification. The new test changes only that backend literal in the original FP32 script, preserving cases, data, original1e-4 thresholds, retained history and lifecycle. No BF16 failure is converted to a pass by changing precision or thresholds.

| Preservation gate | Evidence |
| --- | --- |
| BF16 child/controller/shell natural termination | All exit1; recorded original PIDs absent |
| Finalizer natural termination | Exit0; recorded PID absent |
| Raw offbox archive | 4,222,245bytes,19payloadSHA plus manifest20regular files,931cb207… |
| Original complete source | 4,205,380byte read-only Git bundle,4489e22f…; fresh independent bare fetch/fsck resolves e8aa1ee… |

Full inference quality, stable load, training throughput, memory saving and multi-seed online reward convergence require their separately scoped runs.

Raw proofs: `evidence/loopcd-v3-independent-handback.json`, `evidence/loopcd-bf16-w2-independent-failure-handback.json`, `evidence/reloop-fp32-v1-backend-admission-audit.json`, `evidence/loopcd-w2-fp32-triton-local-admission-audit.json`, and `evidence/loopcd-fp32-triton-w2-independent-handback.json`.

## Re:Loop FP32/Triton GSM8K development — independent audit

All six arms completed the same128 frozen training/development IDs (768 outputs,82041generated tokens), with P4 prefill and the declared subsequent decode depth. All106payload hashes, original8packet payloads, immutable e8 source bundle, complete prompt/arm/trace identities, independent grading and original-resource natural handback pass. The scorer is the frozen strict GSM8K recipe. This is local peer baseline evidence, not ScaleRLT online RL.

| Arm | Correct /128 | Accuracy | Paired status versus full P4D4 off | Fixed-work speed |
| --- | ---: | ---: | --- | --- |
| P4D4 off |80|62.50%|Reference|NOT_RUN|
| P4D4 two-head |83|64.84%|UNCERTAIN|NOT_RUN|
| P4D3 off |79|61.72%|UNCERTAIN|NOT_RUN|
| P4D3 two-head |80|62.50%|UNCERTAIN|NOT_RUN|
| P4D2 off |59|46.09%|DEGRADED|NOT_RUN|
| P4D2 two-head |62|48.44%|DEGRADED|NOT_RUN|

The unchanged paired delta is1percentage point; P4D3 two-head versus full-off has interval[-11.54,+11.54]percentage points, and versus P4D3 off[-9.85,+11.33]. Equal correct totals do not establish noninferiority. D3 is PROVISIONAL_FOR_COST under a separate exploratory selection rule; independent505math/148code confirmation and complete fixed-work cost have not run. Cross-platform interval re-score differs by at most1.1463e-14, below the original1e-10 statistical-check bound; discrete outcomes and classifications match exactly. The initial local bitwise-CI assertion failure remains preserved. No generation/attention tolerance changed.

Wholehandback confirms seven workload PIDs plus finalizer153627 recordedABSENT and natural0; original lock dev/inodes/boot/UUID/sourceclean/emptycompute checks pass. Each GPU child verified nine generation inputs before CUDA; the collector did not rehash all eleven canonical inputs. A new admitted window must perform its own fresh checks. Detailed proofs: evidence/reloop-fp32-v2-independent-quality-and-handback-audit.json and evidence/reloop-fp32-v2-independent-resource-handback.json.

## Complete C32 fixed-work cost comparison

The two fresh processes per arm completed 60 trials (five warmups and five measurements per process), 1,920 requests and 245,760 generated tokens on the original H20. FP32/Triton, P512/T128, 64GiB KV allocation, initial/total queue C32 and actual A16/S16 stay fixed. This is a generation cost experiment with reduced decode depth; C32 does not mean 32 simultaneously resident sequences.

| Arm | Decode depth / guidance | Fresh0 measured tokens/s | Fresh1 measured tokens/s | Speed versus full D4 | Allocated bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| A | D4 / off | 143.8722 | 144.4825 | 1.0000× | 74,501,625,856 |
| B | D3 / off | 166.7335 | 167.5084 | 1.1591× | 74,501,625,856 |
| C | D3 / guided | 164.6464 | 164.7555 | 1.1424× | 74,501,756,928 |

Ratios use the geometric mean of the two fresh-process mean-throughput ratios. C/B is0.9855×; guidance is about1.45% slower than the same depth without guidance. C allocates128KiB more, so this case supplies no memory-saving claim. The independent paired-trial analysis uses a different aggregation and reports C/A1.142596×. Both definitions are retained. Original quality status remains UNCERTAIN at the unchanged1pp margin; fixed-work acceleration does not prove quality preservation, RL throughput or reward convergence. C64 and the frozen653-prompt confirmation remain NOT_RUN.

The independently verified whole archive is2,407,222bytes/SHA06d960ebb83e97eb1dee8af3a98b5c3cc956815eb558c7cdabccc37077ec5cb4,99payloads plus manifest. All six natural exits, original11inputs/resources and finalizer absence pass. Proof: `evidence/loopcd-fp32-c32-independent-handback.json`.
