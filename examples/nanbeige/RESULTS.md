# Nanbeige recurrent RL verification

Runtime `0ca616352e1aa6166217b21a068ba545f52bb4c7` passes 101 CPU tests with exit zero, no failures, errors or skips. Coverage includes the four objectives' tiny three-update versus two-save/fresh-resume checks, grouped GRPO, categorical KL, independent two-loop logits/value/gradient checks, parameter conversion and draft-feature contracts. These synthetic tests do not establish official-model accuracy or reward convergence. Ruff 0.14.7, Black 24.3.0 and isort 5.13.2 pass.

| Algorithm | Tiny FP32 CPU training / fresh resume | Actual recipe argument parsing | Complete official-weight GPU e2e | Matched speed improvement |
| --- | --- | --- | --- | --- |
| PPO | PASS | PASS | Pending | Not measured |
| GRPO | PASS | PASS | Pending | Not measured |
| DPPO | PASS | PASS | Pending | Not measured |
| Flow-DPPO | PASS | PASS | Pending | Not measured |

Native SGD uses Megatron's selected optimizer implementation; the Torch fallback does not receive fused-SGD state initialization. The actual training actor module imports with TMS absent and CUDA uninitialized after TMS imports move into offload paths. The published recipe enables `--offload-train` and still requires native TMS for execution. Unchanged TMS source `8d30c59` builds two AArch64 CUDA 13 libraries using CUDA 13.2 headers and libraries, with build and actor-module-import exit zero. CUDA allocation hooks, offload, recovery and GPU training have not been tested.

| Dependency verification | Before | Current CPU result | Matched training speed improvement |
| --- | --- | --- | --- |
| Actor module import with TMS absent | `ModuleNotFoundError` | PASS; CUDA uninitialized | Not measured |
| Pinned native TMS build | Dependency unavailable | Two AArch64 CUDA 13 libraries compile | Not measured |
| Published recipes and MCore argument parsing | Pending | Four algorithms PASS; FP32 and offload preserved | Not measured |

The recipe argument check intercepts Ray launch and performs no rollouts or optimizer updates. It preserves three updates, GRPO group size four, Flow-DPPO's two steps, 48 response tokens and distinct physical training/rollout GPUs. The full GPU campaign remains three continuous updates versus two saved updates plus one fresh-process update for each objective. Acceptance compares complete actor/critic parameters, optimizer, scheduler, RNG, tokens, rewards and policy publication. FP32 and the selected-logprob error bound of 0.03 remain unchanged. Native RLT is pinned to `d2a358933393f25d74dd2bdd1068a741cd5d9226`; model revision is `b82e54bd609793562a75cbf9337970a93369eab5`. The official-weight GPU campaign is incomplete. No speed or memory improvement is inferred from CPU test duration.

Official-configuration FP32 meta layouts match configuration Git blob `e542b36686258dcc58de083c0fa7c81c9ebd65d6`:

| Role | Unique physical parameters | Physical layers | Logical layer visits |
| --- | --- | --- | --- |
| Actor | 4,169,800,704 | 22 | 44 |
| Critic | 3,659,409,408 | 22 | 44 |

The critic excludes the policy LM head. This verifies architecture and parameter layout; zero weight shards were loaded and CUDA remained uninitialized. [Public validation summaries](../../benchmarks/results/nanbeige-validation-20261007/) retain measured results, source pins, library hashes and a checksum manifest. Original logs and deployment records are retained privately. Public summaries omit infrastructure identifiers and operational details.
