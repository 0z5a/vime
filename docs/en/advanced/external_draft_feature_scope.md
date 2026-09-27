# External draft feature collection: forward and version audit

Audited on `0z5a-branch` at `791f101e` on 2026-09-27. The first payload is the input to the target LM head. `draft_feature_contract.py` defines its metadata and causal token map.

| Question | Finding |
| --- | --- |
| Existing capture | DSpark `HiddenStateCapture` hooks embedding, selected decoder layers, and final decoder layer, then clones complete tensors. Its last-layer value is before the model's final norm. `forward_with_dspark` also executes a draft forward, so it is not a collect-only entrypoint. |
| Correct forward | `train_actor` runs ref/teacher before switching to `old_actor` or `actor`. Its separate target `compute_log_prob` can be skipped for rollout logprobs or reusable training logprobs. A collector on that call alone could silently produce no features. |
| Token map | `get_batch` concatenates and pads sequences; response masks align to causal input positions, with the final position masked. Dynamic batching reorders microbatches. The contract records original sample IDs, packed offsets, selected positions, next-token targets, and original rollout versions. |
| Source version | `rollout_id` is an execution identity. `Sample.weight_versions` refers to generation and may differ from the target snapshot used to recompute logits. `weights_backuper` names actor/old_actor tags but does not expose an immutable snapshot ID for each forward. The producer must supply and verify that identity before publishing a ready manifest. |

The current schema accepts only LM-head input, TP=PP=CP=DP=1, and a head snapshot matching the target feature version. The opt-in collector captures selected tokens from the actor LM head input during the existing target log-prob forward, before the optimizer update. It records `weight_updater.weight_version` as the target source version and copies the head at that forward. It publishes a ready manifest only after the owned feature and head files exist. Unsupported layouts and known skipped-forward configurations are rejected. No draft optimizer or vLLM draft publish path is included here.

The six-round Qwen3-0.6B RTX 5090 run exported target/head versions 1 through 6 with distinct head snapshots. The current vLLM response omitted per-sample `weight_version`; the collector preserves the empty generation-version tuple in each sequence instead of inferring it from the target version. Consumers that require verified rollout policy provenance must reject those samples until the serving path supplies it.
