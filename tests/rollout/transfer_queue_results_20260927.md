# TransferQueue contract validation — 2026-09-27

| Rollout path | Validation | E2E step time | Speed vs default |
| --- | --- | ---: | ---: |
| Default VIME rollout | Source audit only | NOT_RUN | baseline unavailable |
| Existing fully async rollout | Source audit only | NOT_RUN | NOT_MEASURED |
| TransferQueue streaming | Group/lease CPU tests only | NOT_RUN | NOT_MEASURED |

Command: `/home/gongji/0z5a/bin/python -m pytest -q tests/rollout/test_transfer_queue_contract.py` in `/home/gongji/0z5a/work/vime-20260927/queue-test`. Result: **12 passed** in 9.39 s, including Python/Torch import time. The TransferQueue adapter, real backend, CUDA and full training run have not been executed. This table records the comparison target without inventing a speed result.
