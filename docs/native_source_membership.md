# Native source membership qualification

Matching revision strings across workers can conceal a different installed
tree or an unregistered patch. This increment binds the native qualification to
file bytes from fixed VIME/RLT Git objects and the retained MCore archive. It
adds launch preflight and a post-run source audit without changing model,
optimizer, sampling or serving code.

The frozen manifest covers the runtime package roots, including non-Python
files, plus `train.py` and `examples/looped_ppo/run.py`. The benchmark controller
and auditors have separate hashes in the results package; they are not included
in their own runtime manifest.

| Source | Frozen identity | Files | Local byte check |
|---|---|---:|---|
| VIME | `62a86331b5a9e9135aeffa64b8f3c4b122a31960` (draft #21) | 224 | PASS |
| RLT | `fd993ec5512904b68e16f4d541682c076c487b5d` (qualification pin) | 54 | PASS |
| MCore | Retained archive SHA256 `772e1abe21ae85b6ccfa29ce03fc4e86cbe7c7082d3614a9fe397690fac2f043` | 490 | PASS |

MCore's original copy receipt records base
`1dcf0dafa884ad52ffb243625717a3471643e087` and patch SHA256
`6fa39fdfdac8dae6b9bb3e44014df6766d774e0968ac07d6025a6b40f211aa98`.
The archive hash is verified against that receipt. This is membership in the
retained patched source, not an independent reconstruction from pristine upstream.
No archive extraction or environment installation is required.

Two freezes produce byte-identical 157,755-byte manifests with SHA256
`f421d2be6c52eaf402afac520a637399615f3f6286f09f93e7af86b617975ec3`.
The later RLT working tree at `334f243e1b7b374682e138b641ce9cb8236a9157`
is correctly rejected at `vllm_rlt/config.py`; the successful local check uses
the already retained qualification-parent files, without changing that tree.

## Launch and audit

Freeze from retained source objects using the existing environment:

```bash
python -m benchmarks.native_sources \
  --vime VIME_REPO --vime-revision 62a86331b5a9e9135aeffa64b8f3c4b122a31960 \
  --rlt RLT_REPO --rlt-revision fd993ec5512904b68e16f4d541682c076c487b5d \
  --mcore-archive megatron-validated.tar.gz --mcore-receipt mcore-source-copy.json \
  --output sources.json
```

Create a roots JSON object with `vime`, `rlt` and `megatron` pointing to the
existing local source directories. The VIME root must contain the child
entrypoint used by the controller. The same absolute paths must be available
on the already admitted Ray worker nodes.

```bash
python -m benchmarks.run_native_qualification \
  --packet PACKET --model VERIFIED_MODEL --output RUNS \
  --ray-address RESERVED_HOST:PORT --algorithm rltt --phase continuous \
  --source-manifest sources.json --source-roots roots.json --execute
```

Execution now requires both source arguments. Before creating the run directory
or launching a child, the controller checks every selected file's size and
SHA256, rejects symlinked source files, and checks the RLT pin against the frozen
qualification profile. It prepends the verified roots to the child's
`PYTHONPATH`; the existing recipe passes that path to Ray. Parent environment
settings and installed packages are not changed. The phase receipt retains the
manifest hash, checked counts and resolved roots. Resume requires the split
phase to have used the same manifest. A dry run still only prints the command.

After the continuous, split and resumed phases finish naturally:

```bash
python -m benchmarks.audit_native_sources \
  --continuous RUNS/rltt/continuous --resumed RUNS/rltt/resumed \
  --packet PACKET --source-manifest sources.json --output source-audit.json
```

The auditor first rechecks the raw runtime/process/rollout evidence. Every
reported package module must map to the frozen source and have its exact file
hash, even if all phases consistently report the same wrong hash. Each actor
must include the native Ouro learner and MCore optimizer module; each rollout
worker must include native Ouro. All three phase receipts must bind the same
manifest and complete file counts. The report hashes its 17 consumed inputs.

This gate covers driver preflight files and worker module files observed at
startup. It relies on trusted task-owned records. It does not attest live Python
code objects, unobserved worker files, HF dynamic modules, Torch/Transformer
Engine/CUDA dependencies or compiled kernels. Separate model-weight,
checkpoint/Adam, output and learning-signal audits remain required. Run it
independently for GRPO and RLTT.

## Validation and outstanding measurements

The final targeted suite has **89 distinct CPU passes**, including 22 new source
and controller cases and 67 existing recipe/runtime/output/learning-signal/
checkpoint cases. The earlier 41- and 89-case runs overlap this suite. Nineteen
MCore/DCP warnings are retained in the logs.

Controls cover dirty Git checkouts, changed archive bytes, changed/missing/
symlinked local files, ambiguous manifests, unchanged revision declarations
with changed module hashes, unknown or missing implementation modules,
incomplete preflight and a different resume manifest. Invalid source inputs
are rejected before `Popen`. A finite real Python child imports the verified
RLT fixture and naturally exits zero; it does not start Ray.

A separate real CPU import check validates all 768 retained files and matches
202 imported modules, including the native Ouro learner, rollout model, RLTT
loss and actual MCore optimizer/configuration classes. CUDA remains
uninitialized. No model weight is read, and no training or generation runs.

| Required E2E comparison | Baseline throughput | FlashRLT throughput | Speedup | Peak memory reduction | Reward convergence |
|---|---:|---:|---:|---:|---|
| Official Ouro Thinking continuous/fresh-worker recovery | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_ESTABLISHED |
| RLTT / FlashLoop / slime / Megatron campaign | NOT_RUN | NOT_RUN | NOT_MEASURED | NOT_MEASURED | NOT_ESTABLISHED |

Raw logs/XML, the frozen manifest, copy receipt, CPU import evidence, harnesses
and source/artifact hashes are retained in
`benchmarks/results/native-source-membership/`. The official GPU and reward
campaign remains pending resource access and full framework execution.
