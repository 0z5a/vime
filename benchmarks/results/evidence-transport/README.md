# ScaleRLT — Evidence archive transport

This qualification measures evidence transport. Inputs are the original licensed Megatron source archive. Each asset is independently downloaded and checked against its server digest and local SHA-256.

| Transport | Input bytes | End-to-end time (s) | Verified MiB/s | Speed relative to same-input H20 SOCKS route |
| --- | ---: | ---: | ---: | ---: |
| Local TLS client through own H20 SOCKS route | 10178626 | 68.670 | 0.141 | 1.00× |
| Local TLS client, direct route | 10178626 | 24.183 | 0.401 | 2.84× |
| Direct route, complete uncompressed source tar | 38015488 | 84.780 | 0.428 | Different input; unpaired |

Times include upload, independent download, digest verification and control requests. Each row is one completed attempt. The route comparison uses identical compressed bytes. These are transport results; learner throughput, GPU memory savings and reward convergence require their own experiments.

At the observed uncompressed rate, preserving 75 GiB serially would take about 49.9 hours. This planning extrapolation from 38,015,488 bytes does not qualify the full online-state archive before H20 retirement.

| Online evidence gate | Current state |
| --- | --- |
| Source archive upload/download SHA and natural exits | PASS |
| Complete checkpoint, Adam and publication archive capacity | Unqualified |
| Native continuous three-update official-model run | NOT_RUN |
| Fresh-worker recovery | NOT_RUN |
| Multi-seed reward convergence and time-to-quality | NOT_RUN |

A durable writable destination with enough capacity is still needed for full states. Full optimizer state and publications remain required.

Evidence: `mcore-source-offbox-complete.json`, `mcore-source-h20route-offbox-complete.json`, `mcore-uncompressed-source-offbox-complete-v2.json`, their three part indices. The first uncompressed-source invocation failed before upload because the release argument expected a metadata path; its failure receipt/log and producer natural exit 0 remain preserved. Corrected v2 records archive format explicitly and passes both natural exits 0.

Use `benchmarks/archive_draft_evidence.py` with metadata from an existing draft release. The default 512 MiB part size bounds local disk use. `--format tar` records an uncompressed tar; the default is `tar.gz`. Completed source files remain intact, and verified temporary parts are removed only after the independent download passes. A retained failed part requires inspection before resuming. For stdin, retain the external producer’s natural-exit receipt as well as the part index.

```bash
gh api repos/0z5a/vime/releases/RELEASE_ID > release.json
python benchmarks/archive_draft_evidence.py --release release.json --index part-index.json --name scalert-source --input immutable-source.tar.gz
```

Raw receipts are retained in `raw/`; their SHA-256 manifest is `manifest.json`. Archive source 772e1abe… is an existing licensed Megatron source snapshot. The archived `.tar` bytes hash to 7e263b90….


Current-device preparation and the same-input Mac proxy measurement are recorded in [online state preservation](../../../docs/online_state_preservation.md) and [complete state storage](../../../docs/native_checkpoint_storage.md). The new source measurement is 6.761 s including upload and independent download, versus 24.183 s direct TLS (3.58×, one timed attempt per route). Full native state transfer and reward convergence remain unmeasured.
