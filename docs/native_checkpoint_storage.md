# ScaleRLT — Complete native state storage

The new ScaleRLT draft release, ID 404381963, has two actual source uploads with matching server digests and independent download hashes. GitHub documents up to 1,000 assets per release, each under 2 GiB, with no total release-size or bandwidth limit. A 75 GiB startup corpus fits within this quota as 150 parts of 512 MiB. The quota and writable destination are verified; the complete state transfer remains NOT_RUN. [GitHub Releases quotas](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

| Requirement | Current evidence | Remaining actual check |
| --- | --- | --- |
| Durable writable own destination | New 0z5a/vime draft release; two source artifacts verified | Complete native state parts |
| Full startup states | Original three checkpoints and four publications retained in protocol; 93.719 GiB free private XFS verified | Actual full states from training |
| Per-part size/count | 512 MiB × 150 estimated parts, within published limits | Actual count and sizes from the stream |
| Complete optimizer/scheduler/RNG | Required by original source and controller | Actual saved states and fresh-worker recovery |
| Transfer speed | Same 10,178,626-byte archive through the existing Mac proxy: 6.761 s, 3.58× relative to direct TLS | Complete state transfer and repeated measurements |

The current H20 preflight verifies all 770 frozen source files and all seven Ouro-1.4B-Thinking files at revision `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`. The existing 0z5a environment and four staged controls are recorded. The preflight exits naturally with code 0; an independent check confirms its actor is absent, both original locks are free and GPU compute is empty. This is preparation evidence; current-device CUDA qualification and online learning remain NOT_RUN.

The producer must finish naturally. All part hashes, server states and independent downloads must match before the archive is complete. Failed uploads retain the source and pending part. Cache files can be reconstructed; checkpoint, publication and control records are all retained. This startup does not replace the required multi-seed reward curves.
