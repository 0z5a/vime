# Resident actor: H20 copy ledger and completion regression

Same Qwen3-4B model, TP=2, 128 matched replay trajectories per update, response cap 4096, 32 GiB rollout KV per card and resident trainer. Both arms use the same resident implementation; the validation hook selects the normal snapshot backuper for the control. Three updates per arm; update 0 is excluded from timing summaries. No checkpoints or evaluations are saved in these timing runs.

| Measurement | Snapshot control | Live actor | Difference |
| --- | ---: | ---: | --- |
| Actor D2H backup, both ranks/update | 7.493 GiB | 0.000 GiB | Removed host/device payload |
| Actor H2D restore, both ranks/update | 7.493 GiB | 0.000 GiB | Removed host/device payload |
| Publication H2D staging, both ranks/update | 7.493 GiB | 0.000 GiB | Removed host/device payload |
| Backup wall interval, warm max-rank median | 0.079847 s | 0.000006 s | Includes each method's waits |
| Restore wall interval, warm max-rank median | 0.113199 s | 0.000002 s | Includes each method's waits |
| Complete weight publication, warm max-rank median | 0.328720 s | 0.319547 s | Includes each method's waits |
| Driver replay-through-publication median | 79.091 s | 78.782 s | 1.0039×; 0.39% latency reduction |

Driver timing starts before replay loading and ends after training, snapshots, cleanup, weight publication and rollout KV onload. This is a fixed-trajectory system comparison, not online generation throughput. Per-method times include synchronization/waits and must not be added to claim an equal E2E saving. The ledger counts actual source tensor bytes handled by the existing copy paths; it is not a PCIe profiler or a count of NCCL/D2D traffic.

All three per-arm driver intervals are preserved in JSON. The two warm observations per arm do not establish statistical significance. The earlier four-step uninstrumented comparison found only 0.17% latency reduction. These results support the small copy-elimination path, not a broader state-management framework or a large speedup claim.

Validation completed: 12 tests passed on H20, including CUDA AdamW parity, reacquisition of live storage after load_state_dict(assign=True), failed-publication propagation, and receiver RPC completion ordering. Ruff passed. The delayed-receiver test covers the existing synchronous RPC boundary, not arbitrary concurrent GPU consumers.

Production code remains unchanged at 8f8ed82240cde34ea30b835e1cbb1e6e18b6641e (documentation head 73d40b9af303e0ebdcba33974798a074f9c65dcd), stacked on e8f2a3e512cc9265cc06b58edf7eb6272aadf859. The paired ledger removes 22.48 GiB of logical host/device payload per update across both ranks. Existing online checkpoint-resume evidence remains valid; this new replay comparison does not establish quality or convergence.
