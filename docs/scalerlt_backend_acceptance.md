# ScaleRLT standard backend acceptance

The first RFC465 experiment uses the standard `train.py` and `RolloutManager`, ordinary Ouro-1.4B at `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, full K4, FP32 and online GRPO. It has no reference model, critic or KL term. Trainer and rollout require two independently owned physical GPUs. One-card colocation is a separate capability and cannot fill this initial gate.

The existing Thinking/RLTT packets remain unchanged. The new `rfc465-ouro-grpo-separate-v1` profile selects 12 training and eight development prompts by ID hash from the [official GSM8K training file](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl). The complete file has 7,473 rows and SHA-256 `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`. Selection never reads model output or reward. The official test split is untouched; these eight development items are startup controls, not a final accuracy experiment.

Three updates use four prompts and eight samples each: 96 new training completions per logical run. A continuous run is compared with two naturally completed updates followed by a fresh process for update three. Initial and three subsequent policies each generate eight development completions. Sampling uses temperature 0.9, full vocabulary and natural EOS; caps are 1,024 prompt and 2,048 response tokens. AdamW uses LR 1e-6, betas 0.9/0.999, epsilon 1e-8, weight decay 0.1 and clip 0.1. The native MCore schedule retains full recomputation. Torch attention is the initial numerical control; other attention backends require their own receipts.

Raw committed publication versions are 1/2/3/4 for logical update states v0/v1/v2/v3. The initial actor is physically published before generation. Resume creates new worker/runtime identities and republishes logical v2 as committed version 3. Audits compare raw version fields rather than renaming them.

```bash
python -m benchmarks.prepare_backend_acceptance \
  --source /task/inputs/gsm8k-train-3101c7d.jsonl \
  --output /task/rfc465-packet

python -m benchmarks.run_native_qualification \
  --packet /task/rfc465-packet --model /task/models/ouro-574fa66c \
  --output /task/rfc465-runs --algorithm grpo --phase continuous \
  --ray-address RESERVED_RAY_ADDRESS \
  --source-manifest /task/frozen-sources.json --source-roots /task/source-roots.json \
  --model-manifest /task/verified-model-manifest.json --execute
```

Use the same command with `--phase split`, then `--phase resume` only after the split process naturally returns zero and its full checkpoint is durable. The launcher never creates a Ray cluster or kills a process. Its recipe enters the standard training driver without `debug-train-only`. Existing source/model, output, useful-Adam-update and full-state comparison auditors remain in the evidence chain. Model weights, optimizer state, scheduler, RNG and dataset cursor all require actual fresh-worker checks; matching saved hashes alone is insufficient.

The package retains the current experimental RLT pin `fd993ec5512904b68e16f4d541682c076c487b5d`. RFC465 records companion `d2a3589`; acceptance of the successor dependency and its public score contract remains explicit work. No upstream support claim follows from a local command probe.

| Check | Previous profile | New profile | Speed ratio | Evidence scope |
| --- | --- | --- | --- | --- |
| Recipe boundary, all three phases | Thinking GRPO resident | Ordinary GRPO separate | Not a timing comparison | Six real-config CPU-MOCK probes and 30 recipe tests pass |
| Deterministic preparation | Existing Thinking IDs preserved | New 12/8 ID-only GSM inputs | — | Two preparations produce identical bytes |
| Actual tokenizer and full input preservation | Existing Thinking receipt | Running local audit | — | Await natural receipt; no model generation |
| Standard real engine/update/publication | NOT_RUN | NOT_RUN | — | Requires two physical GPUs and complete runtime/source/model preflight |
| Fresh worker/model/Adam/scheduler/RNG/plan | NOT_RUN | NOT_RUN | — | Continuous/split/resume raw evidence required |
| Default vLLM without RLT; publication failure/recovery; KV isolation; lifecycle | Existing CPU controls | Real acceptance pending | — | Independent remaining RFC465 gates |
| Reward convergence and 1/2/4/8-GPU reward-correct scaling | NOT_RUN | NOT_RUN | — | Preserve the full campaign after backend acceptance |

Prepared inputs, hashes, command boundaries and pending local checks are retained in [the preparation archive](../benchmarks/results/backend-acceptance-preparation/). The new H20 has one physical GPU and is executing the separately prioritized FlashNS campaign. It provides no admission for this two-resource backend run. Completed task-owned model files are removed only after all required evidence has been retained and readers have naturally finished; no lcpu NFS or other environment is modified.
