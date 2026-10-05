"""Freeze matched B profiles and preserve the prior measured prompt contract."""

import hashlib
import json
from pathlib import Path

from benchmarks.prepare_native_qualification import prepare
from benchmarks.run_native_qualification import command

root = Path(__file__).resolve().parents[1]
source = root / "data/math-v2-rltt-source"
old = root / "data/native-qualification-v1"
prior_audit = root / "evidence/native-qualification-input-audit.json"
prior = json.loads(prior_audit.read_text())
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
rows = []
for variant in ("legacy-remat", "b-baseline", "b-prefix"):
    packet = root / "data" / f"native-{variant}-20261006"
    profile = prepare(source, packet, variant)
    for name in ("train.jsonl", "development.jsonl"):
        assert (packet / name).read_bytes() == (old / name).read_bytes()
    if variant == "legacy-remat":
        assert (packet / "qualification.json").read_bytes() == (old / "qualification.json").read_bytes()
        assert digest(packet / "qualification.json") == prior["qualification_sha256"]
        continue
    audit = {
        "qualification_sha256": digest(packet / "qualification.json"),
        "rows": prior["rows"],
        "source_record_sha256": digest(prior_audit),
        "scope": "Prior measured prompt rows retained after exact train/development byte comparison; not re-tokenized",
        "executed_training": False,
    }
    (packet / "input-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    commands = [
        command(
            packet,
            Path("/root/autodl-tmp/0z5a/flashrlt-h20-20261006/model-stage-v2/models"),
            root / "planned-prefix-runs" / variant,
            "RESERVED_CLUSTER_REQUIRED",
            "rltt",
            phase,
        )
        for phase in ("continuous", "split", "resume")
    ]
    assert all("--recompute" not in argv for argv in commands)
    rows.append({
        "variant": variant, "profile_sha256": digest(packet / "qualification.json"),
        "execution": profile["learner_execution"], "files": profile["files"],
        "input_audit_sha256": digest(packet / "input-audit.json"), "commands_not_run": commands,
    })
assert rows[0]["files"] == rows[1]["files"]
report = {"profiles": rows, "legacy_byte_identical": True, "training_executed": False,
          "full_framework_parse": "NOT_RUN: local Triton unavailable", "reward_convergence": "NOT_ESTABLISHED"}
(root / "evidence/prefix-profiles-freeze.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"profiles": [{k: row[k] for k in ("variant", "profile_sha256", "execution")} for row in rows],
                  "legacy_byte_identical": True, "commands_prepared_not_run": 6}))
