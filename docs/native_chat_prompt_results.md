# Preserve prepared chat prompts in native rollout

With `--apply-chat-template`, the standard VIME `Dataset` already produces a rendered string. Native rollout previously passed that string to `apply_chat_template` a second time. The pinned official Ouro-Thinking tokenizer raises `UndefinedError: 'str object' has no attribute 'role'` before the engine receives a request. The original failure is retained at the parent revision.

The production change is one condition: apply the template only to structured message lists. Already-rendered strings are encoded as received. Plain strings with templating disabled remain unchanged; structured samples retain their per-sample template options. Group identity checks, tokenizer loading, decoding, rewards and model execution are unchanged. Custom data sources using the chat flag must provide a rendered string or a message list with the desired template options.

| Comparison | Parent #15 | Candidate | Speedup |
|---|---|---|---|
| Standard rendered chat prompt | Fails before native engine call | Preserves exact prepared prompt token IDs | Undefined: parent cannot execute |
| Structured conversation / plain text | Existing paths | Same behavior in regression | Not measured |
| Full online RL / reward convergence | Pending | Pending | Not measured |

## Validation

- Six prompt-contract cases cover plain text, structured messages and dataset-rendered strings, each with thinking on/off. Each case sends two complete samples through grouping, response decoding and the actual math reward route. A corrected test run against unchanged production source reproduces two rendered-string failures and four passing controls; the candidate passes all six.
- The full 29-file CPU regression passes **631 distinct checks**, with **6 CUDA skips** and **3 known missing-Triton factory deselections**, in 45.20 seconds. The suite includes previous three-family numerical, nonzero-update, optimizer recovery and held-out insertion checks. This wall time is a test receipt, not a speed benchmark.
- A separate official-tokenizer audit uses all 500 pinned MATH-500 prompts with thinking both off and on, two completions per prompt: **2,000 exact prompt-token matches and 2,000 expected known-answer rewards**. The actual `Dataset`, native rollout adapter, tokenizer, decoder and math reward path run. Ray is an explicit synchronous transport seam, and the engine replays canonical answer tokens. No model generation, GPU worker or quality measurement occurs.

The original official-tokenizer failure and all initial test failures are retained. Two early runs failed because the test tokenizer omitted the standard `tools` keyword; this fixture was corrected before the meaningful parent reproduction. Those fixture failures are not counted as production regressions. The audit's initial B023 lint error was fixed by binding the synchronous callback's loop values before execution. No tolerance, reward implementation or model math changed.

Raw receipts, per-completion token hashes and source hashes are in `benchmarks/results/native-chat-prompt/`. The full real Ray/CUDA training/evaluation/resume path and reward convergence remain unrun in this change.

## Reproduce

```bash
pytest tests/rollout_backends/test_native_chat_prompt.py tests/rollout_backends/test_native_rlt_groups.py tests/rollout_backends/test_native_rlt_evaluation.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m benchmarks.audit_native_chat --data /path/to/math/prepared/math500.jsonl --tokenizer /path/to/math/tokenizer/ByteDance/Ouro-1.4B-Thinking/3aaa2224253a92ca45cf2e3d427c360e1ef9c93d --output /path/to/math/native-chat.json
```

Use the existing environment and the verified data/tokenizer inputs from `docs/math_quality_inputs.md`; the commands require no environment installation or update.
