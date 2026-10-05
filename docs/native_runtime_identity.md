# Native startup identity and fresh-worker evidence

Native resume qualification now records the learner and rollout workers when
`--rlt-runtime-report-dir` is set. Ordinary runs keep the default `None` and do
not import the recorder or write reports. The frozen qualification launcher
enables separate `runtime/continuous`, `runtime/split` and `runtime/resume`
directories. This is an incremental change on the checkpoint audit in
[draft #19](https://github.com/0z5a/vime/pull/19).

The learner records after checkpoint loading and critic reinitialization. Each
record contains actual Ray job/node/worker/actor IDs, PID and Linux process start
ticks; Python/Torch/CUDA/Ray versions; model class and state shapes/dtypes; and
actual optimizer class, effective MCore configuration and parameter-group
settings. It records the loader's **checkpoint iteration**, not a completed
rollout count: this HF loader returns 0 initially, while the two-update resume
loads iteration 1. The rollout record binds its native runtime epoch.

For imported VIME, plugin, RLT and MCore modules, the recorder hashes the bytes
at their loaded file paths. Model `forward` and optimizer `step` also record
class/function source hashes and the unwrapped Python entry point's marshalled
code hash. Hashing does not copy parameter values, initialize optimizer moments
or launch CUDA work. Reports are individual exclusive-create JSON files. A
missing Ray actor identity fails the opt-in startup; a driver is not a worker.

These are startup observations, not a complete dependency or kernel attestation.
Source bytes on disk need not describe every already-loaded function. Unimported
modules, compiled kernels and later code changes remain outside this record.
Marshalled code hashes depend on Python version and source path. Revision strings
remain declarations: a separate frozen-source manifest must establish membership
in the requested revisions. The recorder does not infer AdamW from a saved
argument default or relax the checkpoint auditor's explicit AdamW requirement.

After all three phases finish naturally, run this on the retained task-owned
records and rollout dumps:

```bash
python -m benchmarks.audit_native_runtime \
  --continuous RUNS/rltt/continuous --resumed RUNS/rltt/resumed \
  --packet PACKET --output runtime-audit.json
```

The auditor requires exactly one rank-0 learner and rollout report per phase,
six distinct Ray actors/workers and Linux process identities, one Ray job per
phase, unchanged reported environment and overlapping module hashes, stable
model/optimizer contracts, loader iterations 0/0/1, and three different runtime
epochs. It reopens all six saved rollout rounds, binds each sample to its phase's
epoch and policy version, and hashes every consumed report, receipt, profile and
dump. Failed/unfinished phases, extra startup reports and stale worker traces
are rejected. This gate tests consistency of trusted experiment evidence; it
does not authenticate an arbitrary supplied JSON file.

Weight identity, stored checkpoint/Adam state, complete rollout learning signal,
held-out output equality and frozen-source verification remain separate gates.
The previous Adam audit still rejects older/fused optimizer formats without its
required saved mode marker. Capturing their actual implementation provides input
for later qualification, not automatic approval of those formats.

## Validation and performance comparison

The new tests use actual CPU PyTorch models and Adam/AdamW implementations.
Ray context, model transport and DDP/optimizer wrappers are explicit test doubles.
MCore `OptimizerConfig` and `ChainedOptimizer` are imported from the existing
checkout; no package was installed or changed. The actual learner initializer
body is tested in isolation because importing the complete training framework
still requires unavailable Triton. No Ray cluster or worker was started.

| Check | Result | Evidence scope |
|---|---|---|
| Startup capture, missing Ray IDs, opt-in hooks | 12 CPU cases pass | Real model/optimizer identities; Ray/transport doubles |
| Fresh-worker audit, including 12 corruptions | 13 CPU cases pass | Synthetic six-worker/process records and saved traces |
| Existing MCore config and chained optimizer handling | 3 CPU cases pass | Real config/chain classes, wrapper and writer doubles |
| Qualification command forwarding | 7 CPU cases pass | Both algorithms, three phases, existing recipe boundary |
| Full Ray/CUDA startup and fresh-worker resume | NOT_RUN | Requires a compatible admitted runtime |

The final combined regression has **217 distinct CPU passes**, including all
28 new capture/audit/MCore cases and 189 existing recipe, stored-state, learning
signal, adapter, colocation and evaluation checks. It completed naturally in
62.70 seconds with 19 existing single-process/overwrite DCP warnings retained.
Earlier 19-, 32- and 3-case runs overlap this final suite.

| E2E comparison | Baseline rate | FlashRLT rate | Speedup | Peak memory reduction | Reward convergence |
|---|---:|---:|---:|---:|---|
| Native official Ouro Thinking, runtime records disabled/enabled | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_ESTABLISHED |
| RLTT / FlashLoop / slime / Megatron full online RL | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_ESTABLISHED |

Test-suite elapsed times are validation timings, not model throughput or speedup.
Raw logs, XML, command receipts and source/artifact hashes accompany this report
in `benchmarks/results/native-runtime-identity/`. Repeated suites overlap and
must not be added together. No official model results or paper claims change.
