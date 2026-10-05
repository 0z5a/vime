import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from benchmarks.prepare_math_data import INSTRUCTION
from benchmarks.prepare_rltt_math_data import load_source
from vime.rollout.vllm_rlt_rollout import _tokenizer

root = Path(__file__).resolve().parents[1]
first = root / "data/math-v2-rltt-source"
release = root / "data/math-v2-rltt-source-release"
v1 = root / "data/math-v1/prepared"
evidence = root / "evidence"
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
original = json.loads((v1 / "manifest.json").read_text())
assert digest(v1 / "manifest.json") == "10ce26681d2d07f91add2e5a304aca931b3d9300670fe7f3b55bbd00d4341037"
assert {p.name for p in first.iterdir()} == {p.name for p in release.iterdir()}
for path in first.iterdir():
    assert path.read_bytes() == (release / path.name).read_bytes(), path.name
assert (evidence / "rltt-source-inputs/token-identity.jsonl").read_bytes() == (
    evidence / "rltt-source-inputs-release/token-identity.jsonl"
).read_bytes()
matched = 0
for split, artifact in original["artifacts"].items():
    path = v1 / f"{split}.jsonl"
    assert digest(path) == artifact["sha256"]
    before = [json.loads(line) for line in path.read_text().splitlines()]
    after = [json.loads(line) for line in (release / f"{split}.jsonl").read_text().splitlines()]
    for old, new in zip(before, after, strict=True):
        assert old["metadata"] == new["metadata"]
        assert old["label"] == new["label"] == new["answer"]
        assert old["prompt"] == new["problem"] + INSTRUCTION
        matched += 1
upstream, _ = load_source(root / "baselines/rltt-source")
probe = evidence / "rltt-v1-converter-probe"
probe.mkdir(exist_ok=False)
old = json.loads((v1 / "train.jsonl").read_text().splitlines()[0])
(probe / "v1-row.jsonl").write_text(json.dumps(old) + "\n")
upstream["convert_math_to_parquet"](str(probe / "v1-row.jsonl"), str(probe / "v1-row.parquet"))
converted = pq.ParquetFile(probe / "v1-row.parquet").read().to_pylist()[0]
assert converted["extra_info"]["problem"] == ""
assert converted["reward_model"]["ground_truth"] == ""
tokenizer = _tokenizer(str(root / "data/math-v1/model-config/ouro-thinking"))
new = json.loads((release / "train.jsonl").read_text().splitlines()[0])
options = {"tokenize": False, "add_generation_prompt": True}
default = tokenizer.apply_chat_template(new["prompt"], **options)
assert default == tokenizer.apply_chat_template(new["prompt"], enable_thinking=False, **options)
assert default != tokenizer.apply_chat_template(new["prompt"], enable_thinking=True, **options)
manifest = json.loads((release / "manifest.json").read_text())
tokens = [json.loads(line) for line in (evidence / "rltt-source-inputs-release/token-identity.jsonl").read_text().splitlines()]
assert len(tokens) == len({(r["split"], r["id"]) for r in tokens}) == matched == 12498
assert (evidence / "rltt-source-accounting.json").read_bytes() == (evidence / "rltt-source-accounting-release.json").read_bytes()
result = {
    "public_rows_with_v1_identity_order_labels_unchanged": matched,
    "byte_identical_files_between_fresh_runs": len(list(first.iterdir())),
    "token_rosters_byte_identical": True,
    "v1_manifest_and_all_splits_unchanged": True,
    "original_converter_on_old_v1": {"rows": 1, "problem_empty": True, "gold_empty": True},
    "official_template_default_equals_thinking_false": True,
    "official_template_default_differs_from_thinking_true": True,
    "prepared_manifest_sha256": digest(release / "manifest.json"),
    "lengths": {split: {"max": max(r["tokens"] for r in tokens if r["split"] == split),
                         "over_1024": len(manifest["over_prompt_cap_1024"][split])}
                for split in original["artifacts"]},
    "artifacts": manifest["artifacts"],
    "accounting_unchanged": True,
    "scope": "Input/source-function/AutoTokenizer/PyArrow checks only, no model or reward accuracy",
}
(evidence / "rltt-source-contract-check.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
