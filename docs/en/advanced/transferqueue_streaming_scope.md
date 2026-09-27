# TransferQueue streaming scope and current integration gap

Audited against `0z5a-branch` at `791f101e` on 2026-09-27. This document describes current code and the contract in `vime/rollout/transfer_queue_contract.py`; it does not claim a working TransferQueue adapter.

| Concern | Current VIME base | Required for a queue adapter |
| --- | --- | --- |
| Production | `generate_and_rm_group` supplies generated and reward-scored groups; `fully_async_rollout.py` keeps an in-process background queue | Publish one complete, versioned group after payload write finishes |
| Conversion | `RolloutManager._get_rollout_data` flattens groups; `_convert_samples_to_train_data` computes rewards and preserves masks/logprobs; `_split_train_data_by_dp` builds the training schedule | Reuse these transformations exactly once, including optional top-p and routing fields |
| Transfer | `rollout_data_transport` supports Ray object store and NIXL | Optional TransferQueue client/backend path; no TransferQueue symbol or dependency in the base |
| Version | `Sample.weight_versions` records generation versions, while `Sample.rollout_id` identifies a rollout execution | Require engine provenance and exact consumer policy version; reject missing/mixed versions |
| Completion | `train.py` waits for `async_train` before its next weight update | A lease/commit ledger must distinguish data fetch, training start, and completed optimization plan |
| Recovery | Data-source checkpoint tracks sample offsets and indices | A durable queue consumer ledger is absent; a training-time unknown outcome must stop consumption |

The historical [#265](https://github.com/vllm-project/vime/pull/265) and [#320](https://github.com/vllm-project/vime/pull/320) are closed without merge. Their designs are useful references, but neither is an open stack dependency or code in this base.

The current [Ascend TransferQueue](https://github.com/Ascend/TransferQueue) API was inspected at `7b31c0b6147413bf4b55d0a49e197c85250d814b`. `AsyncTransferQueueClient.async_get_meta` requests data through the controller. The controller marks the sampler's `consumed_indexes` during that metadata request; its `GRPOGroupNSampler` recognizes groups by adjacent global indexes. Neither action proves VIME prompt-draw identity or that an optimizer update committed. The adapter must retain its own group manifest and ledger, and must verify backend behavior before data can be deleted. CUDA storage support and failure recovery remain unverified for this task.

The first supported schema is one text GRPO group with complete children, finite rewards, exact policy version, and response-aligned optional logprob/top-p metadata. `GroupLedger` permits requeue only before training begins. Lease expiry during training enters `UNKNOWN`, which requires checkpoint recovery; it is not an exactly-once optimizer guarantee. The adapter, backend E2E, CUDA run, and throughput comparison are still required before enabling this path.
