"""Validate answer-replay controls with the retained official tokenizer, not model output."""

import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from benchmarks.audit_native_outputs import validate_response
from vime.rollout.rm_hub.math_utils import grade_answer_verl
from vime.utils.data import Dataset

task = Path(__file__).resolve().parents[1]
packet = task / "data/native-qualification-v1"
model = task / "data/math-v1/model-config/ouro-thinking"
profile = json.loads((packet / "qualification.json").read_text())
inputs = json.loads((task / "evidence/native-qualification-input-audit.json").read_text())
assert inputs["qualification_sha256"] == hashlib.sha256((packet / "qualification.json").read_bytes()).hexdigest()
token_hashes = {row["id"]: row["token_ids_sha256"] for row in inputs["rows"]}
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
assert tokenizer.eos_token_id == json.loads((model / "config.json").read_text())["eos_token_id"] == 2
rows = []
for split in ("train", "development"):
    source = packet / f"{split}.jsonl"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == profile["files"][source.name]["sha256"]
    dataset = Dataset(str(source), tokenizer, None, None, prompt_key="prompt", label_key="label",
                      metadata_key="metadata", apply_chat_template=True, apply_chat_template_kwargs={})
    for sample in dataset.samples:
        prefix = tokenizer.encode(sample.prompt, add_special_tokens=False)
        assert hashlib.sha256(json.dumps(prefix).encode()).hexdigest() == token_hashes[sample.metadata["id"]]
        wrong = r"\boxed{0}" if not grade_answer_verl(r"\boxed{0}", sample.label) else r"\boxed{1}"
        for expected, response in ((1, r"\boxed{" + sample.label + "}"), (0, wrong)):
            output = tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            assert int(grade_answer_verl(decoded, sample.label)) == expected
            record = dict(tokens=prefix + output, response_length=len(output), response=decoded, label=sample.label,
                          reward=expected, status="completed", recurrent_trace={"finish_reason": "stop"},
                          rollout_log_probs=[-1.0] * len(output))
            validate_response(record, profile["response_limit"], tokenizer)
            rows.append(dict(split=split, id=sample.metadata["id"], expected_reward=expected,
                             response_tokens=len(output), token_sha256=hashlib.sha256(json.dumps(output).encode()).hexdigest()))
assert len(rows) == 40 and not torch.cuda.is_initialized()
result = {
    "scope": "answer-replay controls from known labels with invented scores; no model generation or training",
    "qualification_sha256": inputs["qualification_sha256"], "eos_token_id": tokenizer.eos_token_id,
    "tokenizer_class": type(tokenizer).__name__, "trust_remote_code": False, "local_files_only": True,
    "rows": rows, "cuda_initialized": False, "model_generation": "NOT_RUN",
    "source_sha256": hashlib.sha256((task / "vime/benchmarks/audit_native_outputs.py").read_bytes()).hexdigest(),
    "tokenizer_sha256": {name: hashlib.sha256((model / name).read_bytes()).hexdigest() for name in
                         ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")},
}
(task / "evidence/native-output-tokenizer.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"answer_replay_controls": len(rows), "eos_token_id": tokenizer.eos_token_id,
                  "model_generation": "NOT_RUN", "cuda_initialized": False}))
