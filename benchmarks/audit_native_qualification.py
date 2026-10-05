"""Reopen the complete three-update model/source/state/output evidence chain."""

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.audit_native_adam import audit as audit_adam
from benchmarks.audit_native_outputs import audit as audit_outputs
from benchmarks.audit_native_sources import audit as audit_sources
from benchmarks.native_publications import audit_publications
from benchmarks.native_model_manifest import MODEL_FILES, verify_model


def audit(
    continuous: Path,
    resumed: Path,
    packet: Path,
    input_audit: Path,
    model: Path,
    model_manifest: Path,
    source_manifest: Path,
    algorithm: str,
) -> dict:
    from transformers import AutoTokenizer

    profile = json.loads((packet / "qualification.json").read_text())
    consumed = verify_model(model, model_manifest, profile["model_revision"])
    sources = audit_sources(continuous, resumed, packet, source_manifest)
    model_sha = consumed[str(model_manifest.resolve())]
    expected_files = {name: consumed[str((model / name).resolve())] for name in MODEL_FILES}

    def read_bound(path):
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == sources["consumed_sha256"][str(path.resolve())]
        return json.loads(data)

    for phase, run in (("continuous", continuous), ("split", resumed), ("resume", resumed)):
        receipt = read_bound(run / f"{phase}-process.json")
        assert receipt["model_manifest_sha256"] == model_sha, "Phase did not use the verified model manifest"
        assert receipt["model_preflight"]["files"] == expected_files, "Incomplete or changed model preflight"
        for path in (run / "runtime" / phase).glob("*.json"):
            worker = read_bound(path)
            assert worker["details"]["model_path"] == receipt["model_preflight"]["model"], (
                "Worker used a different model path"
            )
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    outputs = audit_outputs(continuous, resumed, packet, input_audit, algorithm, tokenizer)
    initial = model / "model.safetensors"
    state = audit_adam(continuous, resumed, packet, initial, consumed[str(initial.resolve())], input_audit, algorithm)
    publications = [
        audit_publications(run, initial, curve)
        for run, curve in ((continuous, outputs["continuous_curve"]), (resumed, outputs["resumed_curve"]))
    ]
    for result in (sources, outputs, state, *publications):
        for path, digest in result["consumed_sha256"].items():
            assert path not in consumed or consumed[path] == digest, f"Evidence changed between audits: {path}"
            consumed[path] = digest
    assert state["inputs"]["qualification_sha256"] == consumed[str((packet / "qualification.json").resolve())]
    assert state["inputs"]["token_audit_sha256"] == consumed[str(input_audit.resolve())]
    useful = sorted(set(state["joint_useful_updates"]) & set(outputs["joint_signal_updates"]))
    return {
        "evidence_chain_pass": bool(useful),
        "joint_useful_updates": useful,
        "model_revision": profile["model_revision"],
        "algorithm": algorithm,
        "model_files_verified": len(MODEL_FILES),
        "sources": sources,
        "outputs": outputs,
        "stored_state": state,
        "publications": publications,
        "consumed_sha256": consumed,
        "scope": "trusted saved three-update evidence including at least one useful update; not convergence or resource handback",
        "resource_handback": "REQUIRES_SEPARATE_PROCESS_AND_READER_AUDIT",
        "reward_convergence": "NOT_ESTABLISHED",
        "performance": "NOT_MEASURED",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "continuous",
        "resumed",
        "packet",
        "input-audit",
        "model",
        "model-manifest",
        "source-manifest",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--algorithm", choices=("grpo", "rltt"), required=True)
    args = parser.parse_args()
    report = audit(
        args.continuous,
        args.resumed,
        args.packet,
        args.input_audit,
        args.model,
        args.model_manifest,
        args.source_manifest,
        args.algorithm,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["evidence_chain_pass"] else 1)


if __name__ == "__main__":
    main()
