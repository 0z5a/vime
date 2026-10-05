# ScaleRLT: Huginn latent grouping and complete replay

Huginn completions with independent recorded latents need separate recurrent trajectories. The learner now groups the actual prompt, revision, actor generation, depth, latent seed and replay profile. Shared identities use the existing differentiable prefix path; singleton latent groups execute the complete provider replay. Sampling seeds, behavior scores and reference values retain their original provenance.

The opt-in single-rank FP32 MCore schedule accepts the actual Huginn provider. Its prefix entry requires the recorded seed, and the planner rejects missing, boolean or incompatible latent identities. Huginn joint prefix rematerialization remains unsupported. Prelude, core and coda retain their existing computation boundaries.

After gradient finalization, `prefix-plans/generation<generation>-offset<offset>.json` records sample identities, shared samples and complete-replay samples. The denominator includes every logical sample, including empty and masked responses. This receipt describes completed backward work; the existing optimizer-step receipt remains the evidence for a successful update.

## Numerical comparison

Each configuration compares two nonzero AdamW updates against the complete packed-sequence provider replay. The frozen reference differs from the initial actor. Cases cover two loss reductions, two microbatch/wave partitions, scales 1 and 8, ragged/empty responses, zero masks, all recurrent supervision depths, KL and entropy. Every parameter, gradient, update and Adam moment is checked with the existing tolerances. Trace fixtures and behavior log probabilities are checked for mutation.

The three latent arrangements are explicit fixed-trace fixtures. They do not estimate how often independent online samples will share latents.

| Recorded latent arrangement | Shared samples / logical samples | Complete-replay samples | Configurations / paired updates | Max gradient relative L2 | Max update absolute difference |
|---|---:|---:|---:|---:|---:|
| Independent | 0 / 5 | 5 | 8 / 16 | 7.175e-7 | 5.306e-7 |
| Mixed | 2 / 5 | 3 | 8 / 16 | 9.231e-7 | 2.941e-7 |
| Shared | 5 / 5 | 0 | 8 / 16 | 1.937e-6 | 1.460e-6 |

The unchanged gradient relative-L2 gate is `< 2e-6`; elementwise gradients use `atol=1e-5, rtol=8e-5`, and parameters/updates use `atol=2e-6, rtol=8e-5`. Adam moments and counters also pass. The first focused run reports 28 passed and 12 CUDA skips. The complete related regression reports **216 passed, 12 CUDA skips and 12 existing strict Ouro BF16 xfails**, in 9.75 seconds.

| Comparison | Baseline | New path | Speedup | GPU memory saving | Accuracy / reward evidence |
|---|---|---|---|---|---|
| Fixed-trace tiny CPU updates | Complete provider replay | Actual shared-group dispatch or complete replay | Not measured | Not measured | Loss, full gradients, two updates and Adam state pass |
| Actual MCore CUDA updates | Ordinary schedule | Prefix/fallback schedule | NOT_RUN | NOT_RUN | Twelve cases prepared; CUDA skipped locally |
| Official Huginn online RL | Matched complete-replay RL | Recorded-latent grouping | NOT_RUN | NOT_RUN | Online sharing frequency and reward convergence pending |

CPU test duration includes both arms and assertions and cannot be used as a throughput ratio. These CPU calls use the production group helper with actual providers; they do not qualify a CUDA DDP wrapper or an official checkpoint. The existing CUDA fixture now covers independent and mixed Huginn latents alongside Ouro and Nanbeige, including the backward-report counts, finalization, optimizer, scheduler and cursor checks.

## Reproduction and retained evidence

Use the pinned RLT and MCore sources with the repository's existing dependencies:

```sh
PYTHONPATH=.:../rlt:tests:tests/plugins:tests/rollout_backends:$MCORE_ROOT \
OMP_NUM_THREADS=1 pytest \
  tests/plugins/test_differentiable_prefix.py \
  tests/plugins/test_looped_logical_step.py \
  tests/plugins/test_prefix_actor_binding.py \
  tests/plugins/test_prefix_provider_entry.py \
  tests/plugins/test_prefix_rematerialization.py \
  tests/plugins/test_huginn_prefix_fallback.py \
  tests/integration/test_native_prefix_train_step.py \
  -q -p no:cacheprovider --tb=short --show-capture=no \
  -o junit_family=legacy --junitxml=huginn-prefix.xml
```

[Raw logs, XML, numerical summary and SHA-256 manifest](../benchmarks/results/huginn-prefix-training/) retain the initial dependency-collection failures and successful runs. The four earlier iCloud-blocked processes were left to wait naturally; their pre-validation snapshot is retained separately from numerical results. A later import attempt also remains naturally waiting on an evicted PyTorch source. No waiting process is counted as a failed numerical test.

Validation used the original Python 3.12.14 / Torch 2.13.0 environment and a pre-existing, hash-verified MCore source copy. A process-only reader loaded evicted text from existing caches or official same-version artifacts, requiring the original installed `RECORD` hash. Entry-point metadata additionally requires byte-identical distribution metadata. Source recovery and artifact hashes are retained; no installed package or binary was replaced. Third-party pytest and device-backend autoload were disabled for these explicit synchronous CPU tests. Official CUDA, BF16, large-concurrency RL and independent-seed reward convergence remain open.
