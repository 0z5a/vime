# TransferQueue contract validation — 2026-09-27

| Rollout path | Validation | Median driver round wall | Speed vs default |
| --- | --- | ---: | ---: |
| Default VIME rollout | Qwen3-0.6B, RTX 5090, three six-round runs on the shared base | 11.563 s | 1.000× reference |
| Existing fully async rollout | Source audit only | NOT_RUN | NOT_MEASURED |
| TransferQueue streaming | Group/lease CPU tests only | NOT_RUN | NOT_MEASURED |

Command: `/home/gongji/0z5a/bin/python -m pytest -q tests/rollout/test_transfer_queue_contract.py` in `/home/gongji/0z5a/work/vime-20260927/queue-test`. Result: **12 passed** in 9.39 s, including Python/Torch import time. The default-path reference uses the separate base-source E2E harness on the same host: `baseline-bench1/2/3.log`, with one warmup and five measured rounds per run. The individual run medians were 10.761, 11.836 and 11.563 s. This PR only adds a CPU contract; its head was not used for that GPU reference. The TransferQueue adapter, real backend and queue CUDA/full training path have not been executed, so no queue speed ratio is available.
