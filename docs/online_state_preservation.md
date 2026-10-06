# ScaleRLT — Online state preservation qualification

This qualification measures evidence transport. Inputs are the original licensed Megatron source archive. Each asset is independently downloaded and checked against its server digest and local SHA-256.

| Transport | Input bytes | End-to-end time (s) | Verified MiB/s | Speed relative to same-input H20 SOCKS route |
| --- | ---: | ---: | ---: | ---: |
| Local TLS client through own H20 SOCKS route | 10178626 | 68.670 | 0.141 | 1.00× |
| Local TLS client, direct route | 10178626 | 24.183 | 0.401 | 2.84× |
| Local TLS client through the existing Mac HTTP proxy | 10178626 | 6.761 | 1.436 | 10.16× |
| Direct route, complete uncompressed source tar | 38015488 | 84.780 | 0.428 | Different input; unpaired |

Times include upload, independent download, digest verification and control requests. Each row is one completed attempt. The route comparison uses identical compressed bytes. These are transport results; learner throughput, GPU memory savings and reward convergence require their own experiments.

The new proxy measurement is 3.58× faster than the direct route for the same input. Each route still has one timed attempt. The historical uncompressed measurement implies 49.9 hours for a serial 75 GiB transfer; it is a planning extrapolation, not the current route's throughput or a full-state result.

| Online evidence gate | Current state |
| --- | --- |
| Source archive upload/download SHA and natural exits | PASS |
| Writable full-state destination and published quota | PASS; actual full-state transfer NOT_RUN |
| Current H20 source, model, environment and disk preflight | PASS; 770 sources, seven model files, 93.719 GiB free private XFS |
| Native continuous three-update official-model run | NOT_RUN |
| Fresh-worker recovery | NOT_RUN |
| Multi-seed reward convergence and time-to-quality | NOT_RUN |

The supplied Hugging Face credential was independently checked through the official identity endpoint: account `0z5a`, scoped `repo.content.read` only. The new own draft release, ID 404381963, is independently verified writable. Its published quota accommodates 150 parts of 512 MiB for a 75 GiB corpus. Complete optimizer state and publications remain required. [GitHub Releases quotas](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

Evidence: `mcore-source-offbox-complete.json`, `mcore-source-h20route-offbox-complete.json`, `mcore-uncompressed-source-offbox-complete-v2.json`, their three part indices, and `scalert-hf-identity-capability.json`. The first uncompressed-source invocation failed before upload because the release argument expected a metadata path; its failure receipt/log and producer natural exit 0 remain preserved. Corrected v2 records archive format explicitly and passes both natural exits 0.

Current-device evidence: `scalerlt-native-storage-proxy-timed-result.json`, its complete part index, `scalerlt-native-startup-release-verified.json`, and `h20-epoch2-online-io-whole-handback.json`. The proxy is selected per process; system settings and installed environments are unchanged. The first preflight attempt exits 127 because the remote shell has no `python3` command. The next attempt uses the existing absolute 0z5a Python path, passes all checks and exits 0; both attempts are retained.
