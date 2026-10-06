# ScaleRLT Thor validation

The human authorized continuing ScaleRLT verification on Thor on 2026-10-06. Accuracy, reward preservation and convergence retain priority. This machine is an additional hardware validation target; its results require their own receipts.

The normal jump route reaches the target SSH service, but Root's own batch authentication returned exit 255 (`Permission denied`). The current GPU, memory, lock owners, disk capacity and existing `0z5a` Python environment remain unknown. Historical A3 receipts cannot establish current resource availability. No remote files, environments or processes have been changed.

Source is frozen at `322c7550c81f34d76ff77adb82559f5fee7b2ecf`, the head of [draft #37](https://github.com/0z5a/vime/pull/37). The 18,206,790-byte source packet and complete per-file hashes are retained in the private local task archive, `evidence/thor-validation-v1`. Its [preparation receipt](../benchmarks/results/thor-validation-preparation/preparation.json) records the source hash and authentication result. It contains no model weights.

| Stage | Required evidence | Current result | Throughput improvement |
| --- | --- | --- | --- |
| Existing work and environment | Fresh process/lock state, unified memory, local disk, existing Python and package identities | Authentication required | — |
| Actual CUDA learner equivalence | Existing 12 MCore prefix train-step cases: complete gradients, parameters, Adam moments, useful updates and accounting | NOT_RUN | Verification time is not a speedup |
| Native Adam restoration | Existing four patched-MCore Adam/AdamW restoration cases; actual patch identity required | NOT_RUN | — |
| Ordinary Ouro online GRPO | Exact 574fa66c model, full K4 FP32, natural EOS, unchanged original tolerances, standard train.py/RolloutManager | NOT_RUN | End-to-end wall time required |
| Continuous versus fresh resume | Full model, optimizer, scheduler, RNG, data cursor, publication versions and new runtime identities | NOT_RUN | — |
| Reward and load campaign | Fixed held-out data, paired accuracy/reward, multiple seeds, high concurrency, complete RL wall time | NOT_RUN | Reward-correct samples/s |

After normal authentication, inspect only the known private local task paths and existing runtime. Preserve all current owners and waiters. Enter the existing GPU coordination lock without signals or timeout termination, and confirm current memory and source identities before importing a CUDA runtime for a test. If the required dependency is absent, retain that failure and proceed with work supported by the existing environment; do not install or upgrade it.

The first CUDA learner command, after admission in the existing environment, is:

```bash
PYTHONPATH="$PWD/tests/plugins:$PYTHONPATH" "$SCALERLT_PYTHON" -m pytest \
  tests/integration/test_native_prefix_train_step.py \
  --junitxml="$SCALERLT_RESULTS/mcore-prefix.xml"
```

No skipped test qualifies CUDA correctness. The separate Adam restoration test requires the exact native-Adam Megatron patch and must not be run as though an arbitrary MCore installation has that patch.

One physical Thor GPU cannot by itself satisfy the initial RFC465 profile's two independently owned trainer/rollout resources, or supply a 2/4/8-GPU scaling point. Any single-device colocation experiment must be reported separately. Tiny learner checks do not establish official-model online reward convergence. Keep the full backend and convergence campaign pending until their actual evidence is available.

Completed task-owned model bodies may be cleaned only after evidence is retained and all readers naturally finish. No process is killed, no environment is updated, and lcpu NFS is never accessed.
