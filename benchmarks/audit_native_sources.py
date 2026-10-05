"""Match native worker startup files to a separately frozen runtime source manifest."""

import argparse
import json
from pathlib import Path

from benchmarks.audit_native_runtime import audit as audit_runtime
from benchmarks.native_sources import module_index, read_resident, sha


def audit(continuous: Path, resumed: Path, packet: Path, manifest_path: Path) -> dict:
    manifest_bytes = read_resident(manifest_path)
    manifest = json.loads(manifest_bytes)
    expected = module_index(manifest)
    digest = sha(manifest_bytes)
    profile = json.loads((packet / "qualification.json").read_text())
    assert manifest["sources"]["rlt"]["commit"] == profile["engine_revision"], "Wrong frozen engine revision"
    runtime = audit_runtime(continuous, resumed, packet)
    consumed = {**runtime["consumed_sha256"], str(manifest_path.resolve()): digest}
    matched = {}
    for module, observed in runtime["reported_module_sha256"].items():
        assert module in expected, f"Loaded module is outside the frozen source: {module}"
        assert observed == expected[module]["sha256"], f"Loaded module differs from the frozen source: {module}"
        matched[module] = expected[module]
    phases = {}
    for phase, run in (("continuous", continuous), ("split", resumed), ("resume", resumed)):
        path = run / f"{phase}-process.json"
        contents = read_resident(path)
        assert sha(contents) == consumed[str(path.resolve())], "Process receipt changed during audit"
        receipt = json.loads(contents)
        assert receipt["source_manifest_sha256"] == digest, "Phase used a different source manifest"
        counts = {name: sum(row["source"] == name for row in manifest["files"]) for name in manifest["sources"]}
        assert receipt["source_preflight"]["files_checked"] == counts, "Incomplete source preflight"
        for path in sorted((run / "runtime" / phase).glob("*.json")):
            contents = read_resident(path)
            assert sha(contents) == consumed[str(path.resolve())], "Worker record changed during audit"
            worker = json.loads(contents)
            required = (
                {"vime_plugins.ouro.model", "megatron.core.optimizer.optimizer"}
                if worker["role"] == "actor"
                else {"vllm_rlt.models.ouro"}
            )
            assert required <= worker["loaded_modules"].keys(), "Missing Ouro or MCore implementation modules"
        phases[phase] = receipt["source_preflight"]
    return {
        "source_membership_pass": True,
        "source_manifest_sha256": digest,
        "sources": manifest["sources"],
        "matched_modules": matched,
        "preflight": phases,
        "consumed_sha256": consumed,
        "scope": "driver preflight files and observed worker module files at startup; trusted records, not loaded-code attestation",
        "excluded": [
            "unobserved worker files",
            "HF dynamic modules",
            "Torch/TE/CUDA dependencies",
            "compiled kernels",
        ],
        "weights_checkpoint_outputs": "REQUIRES_SEPARATE_AUDITS",
        "reward_convergence": "NOT_ESTABLISHED",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.continuous, args.resumed, args.packet, args.source_manifest)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
