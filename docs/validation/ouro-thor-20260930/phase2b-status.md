# Phase 2B contributor status — 2026-10-01

The VIME integration is implemented and validated through the shared engine
contract, including real full-model training, conversion and fresh-process
save/resume. The contributor drafts await review; this does not assert formal
upstream acceptance of the Phase 1 API or Phase 2B work.

| Requirement | Current evidence | Validated scope |
|---|---|---|
| Ouro training adaptation | Shared physical layers, existing VIME/Megatron GRPO, Adam and distributed checkpoints; replay of actual token depths | TP/PP/CP=1; local DP and single-rank PD recipe |
| Parameter conversion | Ouro registered in general HF→Megatron and Megatron→HF conversion; generic export matches all 269 full-model physical tensors | Three parameter wrapper prefixes and separate tiny/full export checks |
| Shared rollout/update contract | Public `LLM` generation, selected-token raw/processed scores, depth traces and versions; full chunked in-place publication, prefix invalidation and failed-update recovery | Native engine and real NIXL P/D workers on two A100 GPUs |
| Multi-iteration consistency and resume | Thor five eager updates vs four graph updates + fresh resume: all 272 checkpoint tensors and Adam metadata exact. PD three eager vs two graph updates + fresh resume: all 272 exact | Full BF16 Ouro-1.4B; PD uses one active slot for reproducible batch shapes |
| Existing inference-feature compatibility | Early exit, delayed async execution and self-speculation each complete two full-model train/update steps; real PD contract/save/resume checks pass | Existing upstream self-speculation remains synchronous, full-target-depth and unavailable in PD |
| Performance evidence | Committed Markdown tables with raw pairs, real GRPO components and negative whole-step results | Thor warmed rollout saves 6.08–8.11%; no whole-training speedup claim |

See the [current Thor report](../ouro-thor-20261001/README.md),
[dual-A100 PD training report](../ouro-a100-20261001/README.md),
[VIME draft](https://github.com/0z5a/vime/pull/5), and
[engine draft #4](https://github.com/0z5a/vllm-rlt/pull/4).
The engine changes apply to current upstream
`ecb1f8b505b7e831815b40aec3b4598619cca23a` rather than the earlier prototype
`RLEngine`. The original 2026-09-30 report remains historical evidence for its
stated synchronous scope.

BF16 numerical equivalence requires matched batching; default dynamic PD batching
does not pass the bitwise checkpoint comparison. The container's GPU IPC path
also fails its independent transport probe, so passing PD runs use command-scoped
host-staged NIXL transport. Both limits and rejected checks are recorded in the
A100 report. Completed task-owned model/checkpoint/export files are cleaned;
raw evidence remains available for review.
