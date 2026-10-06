# ScaleRLT — Exploration coverage and quality gates

The current paper title is **ScaleRLT: I/O-Aware End-to-End Reinforcement Learning for Recurrent Looped Transformers**. This ledger follows the three supplied execution documents. Their original filenames and frozen input identifiers remain provenance records. The user's latest priority is accuracy, reward convergence and scaled RL stability; small speed or memory gains are sufficient.

## Measured comparisons

| Comparison | Model/workload | Baseline | Candidate | Speed ratio | Evidence boundary |
| --- | --- | ---: | ---: | ---: | --- |
| Packed prefix update | Tiny Ouro, CPU P256/D8/G32, SGD | 450.72 ms | 152.09 ms | 2.963× [2.099, 4.184] | Fixed trace; complete gradients and update checked |
| Packed prefix update | Tiny Nanbeige, same CPU workload | 219.31 ms | 69.79 ms | 3.142× [2.039, 4.842] | Fixed trace; complete gradients and update checked |
| Packed prefix update | Tiny Huginn, same CPU workload, shared recorded latent | 605.58 ms | 164.77 ms | 3.675× [3.100, 4.358] | Artificial sharing; online reuse fraction unmeasured |
| Evidence transport | Identical 10,178,626-byte source archive | 68.670 s via H20 SOCKS | 24.183 s direct TLS | 2.84× | One attempt per route; upload, download and SHA verification |
| Current evidence transport | Identical 10,178,626-byte source archive | 24.183 s direct TLS | 6.761 s through existing Mac proxy | 3.58× | One timed attempt per route; full server-digest and independent-download checks |
| Production MCore prefix schedule | Tiny three-family H20 CUDA, two reductions | Ordinary schedule | Prefix/joint/latent schedule | NOT_MEASURED | Twelve cases, paired Adam updates and all moments pass |
| Complete official-model RL | Natural rollout/reward/reference/update/publication/eval/save | NOT_RUN | NOT_RUN | NOT_MEASURED | Main endpoint remains open |

The first three ratios use three independent process cohorts. Negative controls and shorter-prompt regressions remain in the original report. Component ratios are not multiplied into an end-to-end claim.

## B → C → A

| Work package | Implemented and verified | Remaining exit gate |
| --- | --- | --- |
| B1 logical samples and prefix identity | Logical denominators, sample/group identity, reorder and masking checks | Formal official-model workloads |
| B2 loss and schedule scaling | Complete FP32 gradients, nonzero paired Adam updates, loss scales one/eight | BF16 remains failed at unchanged gates |
| B3 actual MCore schedule | Twelve real H20 CUDA cases at 2ad271d; 24 paired candidate updates, all gradients/parameters/moments pass | Official online learning and performance |
| B4 model and policy identity | Ouro and Nanbeige prefix paths; Huginn actual-latent singleton fallback and shared-group replay | Official Nanbeige/Huginn online quality; Huginn joint replay unsupported |
| C1 joint rematerialization | Ouro and Nanbeige CPU and actual tiny CUDA paired updates preserve gradients and Adam state | Official models and capacity measurements |
| C2 prefix lifetime and capacity | Tiny saved-storage observations and release/replay checks | GPU allocator peaks and actual capacity frontier |
| C3 role-switch I/O | Resident placement/lifecycle CPU checks; current source/model/private disk preflight and writable destination quota pass | Actual resident Ray/CUDA workers, complete saved states and fresh-worker recovery |
| A1 comparable freeze | Source, model/tokenizer, data, loss and three-update controller frozen | Actual workers and original-library compatibility |
| A2 fixed work and natural RL | Earlier fixed CPU traces and limited CUDA controls retained separately | Matched N0/NB/NC/NBC official runs and independent timing pairs |
| A3 quality and recovery | Continuous, split/resume and learning-signal audits prepared | Mixed-reward groups, nonzero reward-driven Adam, fresh-worker recovery and ≥3 training seeds |
| A4 full system metrics | Transport and tiny component results above | Complete iteration/campaign wall time, GPU peaks, high concurrency and time-to-quality |

The latest Huginn regression has **216 CPU passes**, twelve CUDA skips and twelve retained strict Ouro BF16 expected failures. Passing counts from overlapping suites are not added together. Its largest gradient relative L2 error is 1.936698×10⁻⁶ under the unchanged 2×10⁻⁶ gate.

## W00–W14 and associative execution mapping

| Plan ID | Current evidence | Open requirement |
| --- | --- | --- |
| W00 / BASE-0 / RL0 | Frozen source/input contracts and numerical references | Full original-stack baseline, official cross-engine score gaps and complete observed sample accounting |
| W01 terminal output | Sixteen official Ouro-1.4B CUDA pairs preserve final outputs at C1/4/16/64 | Longer responses, other checkpoints and independent performance pairs |
| W02 stage KV | Tiny Huginn R32 allocated KV tensor storage decreases 48.44% | Actual official CUDA peak, Graph pools and total learner capacity |
| W03 batched replay | Tiny three-family packed replay and gradient/update checks | Official-model CUDA and real workloads |
| W04 / RL1–RL2 readout | Selected scores, entropy and complete categorical KL primitives checked | Target GPU full-step benefit and complete original-objective match |
| W05 RLTT objective | Uniform/progressive primitives and source differences documented | Complete original RLTT and native online qualification |
| W06 / RL5 rematerialization | CPU loop/layer/query candidates; complete causal gradient checks | Measured CUDA planner and joint capacity frontier |
| W07 / RL3–RL4 prefix gradients | All boundary adjoints, first-response readout and actual latent identity preserved | Official online groups and load balance; twelve tiny MCore CUDA cases now pass |
| W08 rollout group fork | Tiny CPU cold-group fork, ownership and preemption checks | Official CUDA stream ordering, long-running load and Huginn deterministic prelude |
| W09 DP/publication | CPU cohort/version/recovery contracts | Actual DP2/4, optimizer sharding, fresh processes and full publication ledger |
| W10 CP/context | No completed CP qualification | CP2/4/8, serving capacity and useful long-context reward |
| W11 TP/physical weights | No completed TP qualification | TP2/4/8 checkpoint, update and publication equivalence |
| W12 actual depth/Graph | Prior fixed-depth output Graph evidence retained | Actual-depth source map and combined CP/Graph qualification |
| W13 IQuest 40B | Source/16 headers/883 tensors pinned; tiny replay output/gradient gates pass 32 steps per backend; strict Adam delta gate fails | Resolve update gate, native provider/loader, sharded real update and recovery; zero official weight tensor bodies downloaded |
| W14 final scale/quality | Interim component and short-run evidence only | Four-model core matrix, extended new samples, ≥3 seeds and full time-to-quality |

The associative document's RNN, KV, quantization, FlashNS and communication sections belong to coordinated peer projects. Their results are not counted as ScaleRLT reward convergence. Peer numerical or PDE convergence does not replace an RL update.

## Library and model baselines

| Baseline | Verified scope | Main unresolved comparison |
| --- | --- | --- |
| VIME / Megatron | Native tiny CPU and twelve actual H20 CUDA MCore controls | Complete official N0/NB/NC/NBC campaigns |
| Original RLTT @0618985 | Prompt/data equality for 12,498 public rows; conditional source batch accounting | Complete original environment, modified checkpoint, grader, AdamW8bit and observed batch counts |
| FlashLoop | Separate rollout baseline in the common supported domain | Matched supported-model numerical, quality and high-load measurements |
| Slime | Listed framework baseline; looped runtime compatibility not qualified | Exact supported model/runtime, complete learner and quality run |
| Optimized vllm-rlt | Existing official output and prior short-run evidence retained | New full online learning and stable high-concurrency comparisons |

Core official checkpoints remain Ouro 1.4B/2.6B, Nanbeige4.2 and Huginn-0125. IQuest 40B is a separate full-update extension. Tiny configurations do not qualify those checkpoints.

## Execution state

The frozen twelve-case CUDA packet 79ff7ca7 completed:12PASS,24 paired candidate updates,84 offbox payloads plus manifest,5cbea083 archive. Source2ad and original gates remained unchanged. See `h20_mcore_all_families.md`. Official Thinking continuous qualification retains 96 new training completions, 32 held-out completions, three full checkpoints and four publications; it is NOT_RUN. Fresh-worker recovery and full multi-seed reward convergence are also NOT_RUN.

The continued H20 is a separate physical GPU epoch. Its original B-source `d6229d6` passes two actual CUDA/MCore/Adam configurations with four paired updates and unchanged full numerical/state assertions. All 20 payloads and their manifest are verified offbox; all five recorded actors are independently absent. See `h20_current_native_gate.md`. The pinned Thinking model's seven files, 770 source files and 93.719 GiB free private XFS were verified before this gate. The new draft destination's quota and small verified uploads pass; complete state transfer remains NOT_RUN. Historical device timings are kept separate.

The historical uncompressed transport rate is 0.428 MiB/s; its 49.9-hour serial extrapolation is not a measurement of the current route. The current task-owned draft destination is writable, and matched small source uploads pass server-digest and independent-download checks. Its published quota supports 150 parts of 512 MiB for 75 GiB, but actual complete optimizer/publication transfer and fresh-worker recovery remain NOT_RUN.

IQuest source/header and strict update failures are in draft [#32](https://github.com/0z5a/vime/pull/32). Completed source transport evidence is in draft [#31](https://github.com/0z5a/vime/pull/31). Production Huginn scheduling is in draft [#30](https://github.com/0z5a/vime/pull/30). Numerical result records and pending gates remain separate from a finished paper or convergence claim.
