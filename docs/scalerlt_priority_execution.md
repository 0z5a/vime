# ScaleRLT — execution priority freeze, October 6

User priority: standard backend acceptance, actual reward-correct scaling, then trained loop-native compute policy. Accuracy and reward convergence remain the primary quality gates. This ordering supersedes using the prepared Thinking/RLTT startup as the first native acceptance experiment; that frozen profile remains intact as a later comparison.

## P1: standard RolloutManager backend / RFC 465

Start from the existing family factory, native Ray engine and standard `train.py` control plane. Keep the default vLLM path independent. The first official gate uses ordinary pinned Ouro-1.4B@574fa66c, full depth K4, positive temperature/full-vocabulary sampling, standard online GRPO and no reference/critic/KL. Broaden family coverage through the same backend after its first real milestone. The RFC's initial resource profile separates trainer and rollout; a one-device colocation run remains a separately qualified combination.

| Acceptance evidence to produce | Current scope | Exit condition |
| --- | --- | --- |
| Standard entry / real workers | Code and CPU contracts exist; exact source freeze pending | Actual v0 rollout → real update/publish v1 → v1 rollout → real update/publish v2; no debug-train-only |
| Default vLLM independent imports/recipe | Existing import tests; full default recipe not audited here | Real supported default recipe with no RLT installation/import dependency |
| Trace serialization/conversion | Existing tiny/CPU and limited inference evidence | Mixed-length tokens, scores, depths, masks, rewards/groups and committed identities survive actual manager → packing → learner conversion |
| Publication / cache isolation | Existing component checks | Initial physical actor publication; controlled missing/failed publication excludes generation; full recovery resumes it; no stale-policy KV reuse |
| Fresh process recovery | Earlier inherited records await independent raw audit | Separate worker epoch restores all native actor/optimizer/scheduler/RNG/cursor state and republishes; next update matches continuous run |
| Resource lifecycle | Existing owned close methods | Start/ready/drain/cancel/close/restart with cooperative natural exits and measured owner handback; no process kill |

Keep PASS, CPU-MOCK, GPU-LOCAL, MULTI-RANK and NOT-RUN evidence separate. Do not mark the RFC complete from the tiny MCore gate or legacy model-specific scripts.

## P2: first actual scaling figure

| Dimension | Frozen comparison requirement |
| --- | --- |
| Allocated GPUs | Actual 1, 2, 4, 8; homogeneous/topology-controlled primary series, heterogeneous placement separately identified |
| Arms | Vanilla VIME/vLLM, fixed-depth RLT, ScaleRLT, on matched supported checkpoint/math/objective/sampling/budget |
| Workload | Cross batch/group size, measured resident + queued concurrency, and sequence length; retain short-prefix / one-response controls |
| Main goodput | `valid correctly graded samples / complete RL wall seconds`; incorrect/cancelled work and all RL phases remain in the denominator |
| Additional goodput | Generated tokens from correctly graded valid samples per complete RL second, with fixed completion budget; raw rollout TPS separately |
| Quality | Fixed grader/held-out IDs/margin, ≥3 independent training seeds, accuracy/reward uncertainty and time-to-quality |
| Stability/cost | Failure/OOM counts, p50/p95/p99 latency and staleness, GPU allocated/reserved peaks, publication/checkpoint cost, allocated GPU-hours |
| Reporting | Raw phase receipts and immutable source/runtime/data/worker identities; unsupported cells marked UNSUPPORTED, never zero throughput |

Current node: new SSH 35679 provides one H20 d952. No real 2/4/8-GPU or scaling data exist yet. Multiple logical ranks on one card are not an 8-GPU result. Source transport, inference depth reduction and CPU prefix ratios cannot fill these curves.

## P3: external draft → trained adaptive speculation

PR 454 is merged (merge5315305d/head76ad2be4). It collects versioned target head inputs and immutable same-version heads; it provides no trained controller or speculative speed result. After P1, freeze the loop-index/profile and policy version of every training feature. Train and retain an actual loop-aware proposal/exit controller, including its optimizer and held-out evaluation. Do not mix target feature/head versions or count replayed answers as learning.

Exact proposals verified against the full-depth target and approximate early-exit policies use separate scoring and quality contracts. The primary claim requires reward/accuracy noninferiority plus higher complete-RL goodput. Feature collection, controller training, target verification and all policy updates enter the full cost. Acceptance rate and fewer loop calls alone do not establish the claim.

## Resource execution boundary

New d952 boot4620b8a9 is independent of historical H20 epochs. Its CPU-only NS environment grant is active until actual full offbox/natural handback; it authorizes no Root GPU, CD model upload or automatic continuation. Reuse an environment only after its actual path/version ledger and readonly borrow are verified. No base updates, process kill, peer credentials/socket borrowing or lcpu NFS access. Existing scientific packets retain their original datasets and thresholds; new resource bindings are separate files.
