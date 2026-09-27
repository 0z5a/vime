# Draft feature contract validation — 2026-09-27

| Path | Completed scope | E2E step time | Speed vs baseline |
| --- | --- | ---: | ---: |
| Existing VIME target forward | Source audit only | NOT_RUN | baseline unavailable |
| Versioned collect-only | Schema and token map CPU tests | NOT_RUN | NOT_MEASURED |

Command: `/home/gongji/0z5a/bin/python -m pytest -q tests/utils/test_draft_feature_contract.py` in `/home/gongji/0z5a/work/vime-20260927/draft-test`. Result: **12 passed** in 8.82 s, including Python/Torch import time. This is a CPU schema test, not a training-speed benchmark. A real CUDA comparison requires a working 0z5a VIME runtime, model weights, and the collector connection; those are not available in this run. No speed claim is made.
