"""Source provenance and launch rejection controls; no Ray or GPU jobs."""

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from benchmarks import run_native_qualification as launcher
from benchmarks.audit_native_sources import audit, main
from benchmarks.native_sources import (
    ROOTS,
    SCHEMA,
    archive_source,
    file_record,
    git_source,
    module_index,
    sha,
    verify_local,
)

pytest_plugins = ["tests.test_native_runtime_audit", "tests.test_native_qualification_recipe"]


@pytest.fixture
def frozen(tmp_path):
    roots = {name: tmp_path / name for name in ROOTS}
    files = []
    for source, paths in {
        "vime": [
            "vime/observability/native_runtime.py",
            "vime/backends/megatron_utils/model.py",
            "vime/backends/vllm_rlt_utils/engine.py",
            "vime_plugins/ouro/model.py",
            "train.py",
            "examples/looped_ppo/run.py",
        ],
        "rlt": ["vllm_rlt/__init__.py", "vllm_rlt/models/ouro.py"],
        "megatron": ["megatron/core/optimizer/optimizer.py"],
    }.items():
        for name in paths:
            path = roots[source] / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# control " + name + "\n")
            files.append(file_record(source, name, path.read_bytes()))
    manifest = {
        "schema": SCHEMA,
        "sources": {
            "vime": {"kind": "git", "commit": "a" * 40},
            "rlt": {"kind": "git", "commit": "engine-pin"},
            "megatron": {"kind": "archive"},
        },
        "files": files,
    }
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(manifest))
    return manifest, roots, path


def test_git_reads_commit_objects_instead_of_dirty_checkout(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = tmp_path / "vllm_rlt/__init__.py"
    path.parent.mkdir()
    path.write_text("pinned = True\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=0z5a",
            "-c",
            "user.email=dezhen.lu@student.uni-tuebingen.de",
            "commit",
            "-qm",
            "Source control",
        ],
        check=True,
    )
    path.write_text("pinned = False\n")
    source, files = git_source(tmp_path, "HEAD", "rlt")
    assert len(source["commit"]) == len(source["tree"]) == 40
    assert files == [file_record("rlt", "vllm_rlt/__init__.py", b"pinned = True\n")]


@pytest.mark.parametrize("corrupt", [False, True])
def test_archive_bytes_bound_to_retained_receipt(tmp_path, corrupt):
    archive, receipt = tmp_path / "sources.tar.gz", tmp_path / "copy.json"
    data = b"# source\n"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("./megatron/core/__init__.py")
        member.size = len(data)
        stream.addfile(member, io.BytesIO(data))
    receipt.write_text(
        json.dumps({"sha256": "a" * 64 if corrupt else sha(archive.read_bytes()), "base": "base", "patch": "patch"})
    )
    if corrupt:
        with pytest.raises(ValueError, match="differs from the retained"):
            archive_source(archive, receipt)
    else:
        source, files = archive_source(archive, receipt)
        assert source["receipt_sha256"] == sha(receipt.read_bytes())
        assert files == [file_record("megatron", "megatron/core/__init__.py", data)]


@pytest.mark.parametrize("change", ["bytes", "symlink", "missing"])
def test_preflight_rejects_changed_source(frozen, change):
    manifest, roots, _ = frozen
    assert verify_local(manifest, roots)["files_checked"] == {"vime": 6, "rlt": 2, "megatron": 1}
    path = roots["rlt"] / "vllm_rlt/__init__.py"
    if change == "bytes":
        path.write_text("changed\n")
    elif change == "missing":
        path.unlink()
    else:
        target = path.with_suffix(".saved")
        path.rename(target)
        path.symlink_to(target)
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_local(manifest, roots)


@pytest.mark.parametrize("change", ["traversal", "duplicate", "entrypoint", "module_alias"])
def test_rejects_ambiguous_manifest(frozen, change):
    manifest, _, _ = frozen
    if change == "traversal":
        manifest["files"][0]["path"] = "vime/../../escape.py"
    elif change == "duplicate":
        manifest["files"].append(manifest["files"][0])
    elif change == "entrypoint":
        manifest["files"] = [row for row in manifest["files"] if row["path"] != "train.py"]
    else:
        manifest["files"].extend(
            [file_record("vime", "vime/alias.py", b""), file_record("vime", "vime/alias/__init__.py", b"")]
        )
    with pytest.raises(ValueError):
        module_index(manifest)


@pytest.fixture
def sourced_records(records, frozen):
    manifest, roots, path = frozen
    modules = module_index(manifest)
    for run in records[:2]:
        for receipt_path in run.glob("*-process.json"):
            receipt = json.loads(receipt_path.read_text())
            receipt.update(
                source_manifest_sha256=sha(path.read_bytes()), source_preflight=verify_local(manifest, roots)
            )
            receipt_path.write_text(json.dumps(receipt))
        for record in (run / "runtime").glob("*/*.json"):
            row = json.loads(record.read_text())
            required = (
                ("vime_plugins.ouro.model", "megatron.core.optimizer.optimizer")
                if row["role"] == "actor"
                else ("vllm_rlt.models.ouro",)
            )
            for name in required:
                row["loaded_modules"][name] = "/" + modules[name]["path"]
            row["source_sha256"] = {
                filename: modules[name]["sha256"] for name, filename in row["loaded_modules"].items()
            }
            record.write_text(json.dumps(row))
    return (*records, path)


def test_cli_binds_frozen_sources_to_all_phases(sourced_records, tmp_path, monkeypatch):
    continuous, resumed, packet, manifest = sourced_records
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--continuous",
            str(continuous),
            "--resumed",
            str(resumed),
            "--packet",
            str(packet),
            "--source-manifest",
            str(manifest),
            "--output",
            str(output),
        ],
    )
    main()
    report = json.loads(output.read_text())
    assert report["source_membership_pass"] and len(report["matched_modules"]) == 6
    assert len(report["consumed_sha256"]) == 17 and report["reward_convergence"] == "NOT_ESTABLISHED"
    for filename, digest in report["consumed_sha256"].items():
        assert sha(Path(filename).read_bytes()) == digest


@pytest.mark.parametrize("change", ["all_phases_changed", "unknown_module", "manifest", "incomplete", "missing_mcore"])
def test_same_revision_declarations_do_not_qualify_changed_sources(sourced_records, change):
    continuous, resumed, packet, manifest = sourced_records
    if change in {"all_phases_changed", "unknown_module", "missing_mcore"}:
        for run in (continuous, resumed):
            for path in (run / "runtime").glob("*/*.json"):
                row = json.loads(path.read_text())
                module = "vime.observability.native_runtime"
                if change == "all_phases_changed":
                    row["source_sha256"][row["loaded_modules"][module]] = "b" * 64
                elif change == "unknown_module":
                    row["loaded_modules"]["vime.unknown"] = "/unknown.py"
                    row["source_sha256"]["/unknown.py"] = "c" * 64
                elif row["role"] == "actor":
                    del row["loaded_modules"]["megatron.core.optimizer.optimizer"]
                path.write_text(json.dumps(row))
    else:
        path = resumed / "resume-process.json"
        receipt = json.loads(path.read_text())
        if change == "manifest":
            receipt["source_manifest_sha256"] = "d" * 64
        else:
            receipt["source_preflight"]["files_checked"]["vime"] -= 1
        path.write_text(json.dumps(receipt))
    with pytest.raises(AssertionError):
        audit(continuous, resumed, packet, manifest)


@pytest.mark.parametrize(
    "case", ["missing_options", "revision", "entrypoint", "changed_file", "resume_manifest", "success"]
)
def test_controller_checks_sources_before_launch(packet, frozen, tmp_path, monkeypatch, case):
    manifest, roots, path = frozen
    entrypoint = roots["vime"] / "examples/looped_ppo/run.py"
    if case == "success":
        entrypoint.write_text(
            "import vllm_rlt\nfrom pathlib import Path\n"
            'Path(__file__).with_name("import-origin.txt").write_text(vllm_rlt.__file__)\n'
        )
        manifest["files"] = [
            file_record("vime", row["path"], entrypoint.read_bytes())
            if row["path"] == "examples/looped_ppo/run.py"
            else row
            for row in manifest["files"]
        ]
    profile = json.loads((packet / "qualification.json").read_text())
    manifest["sources"]["rlt"]["commit"] = "wrong" if case == "revision" else profile["engine_revision"]
    path.write_text(json.dumps(manifest))
    roots_path = tmp_path / "roots.json"
    roots_path.write_text(json.dumps({name: str(root) for name, root in roots.items()}))
    phase = "resume" if case == "resume_manifest" else "continuous"
    output = tmp_path / "runs"
    run = output / "rltt" / ("resumed" if phase == "resume" else "continuous")
    if phase == "resume":
        run.mkdir(parents=True)
        (run / "split-process.json").write_text(
            json.dumps(
                {
                    "returncode": 0,
                    "qualification_sha256": sha((packet / "qualification.json").read_bytes()),
                    "source_manifest_sha256": "old",
                }
            )
        )
    if case == "changed_file":
        (roots["megatron"] / "megatron/core/optimizer/optimizer.py").write_text("changed\n")
    # Only the child command is replaced: all source checks and Popen/wait are real.
    monkeypatch.setattr(
        launcher,
        "command",
        lambda *args: [sys.executable, str(tmp_path / "wrong.py" if case == "entrypoint" else entrypoint)],
    )
    argv = [
        "launcher",
        "--packet",
        str(packet),
        "--model",
        str(tmp_path),
        "--output",
        str(output),
        "--ray-address",
        "reserved:6379",
        "--algorithm",
        "rltt",
        "--phase",
        phase,
        "--execute",
    ]
    if case != "missing_options":
        argv.extend(["--source-manifest", str(path), "--source-roots", str(roots_path)])
    monkeypatch.setattr(sys, "argv", argv)
    if case == "success":
        with pytest.raises(SystemExit) as finished:
            launcher.main()
        assert finished.value.code == 0
        receipt = json.loads((run / "continuous-process.json").read_text())
        assert receipt["returncode"] == 0 and receipt["source_manifest_sha256"] == sha(path.read_bytes())
        assert receipt["source_preflight"]["files_checked"] == {"vime": 6, "rlt": 2, "megatron": 1}
        assert receipt["finished_ns"] > receipt["started_ns"]
        assert Path(entrypoint.with_name("import-origin.txt").read_text()) == roots["rlt"] / "vllm_rlt/__init__.py"
    else:

        def forbidden(*args, **kwargs):
            pytest.fail("Invalid source reached Popen")

        monkeypatch.setattr(launcher.subprocess, "Popen", forbidden)
        with pytest.raises(ValueError):
            launcher.main()
        assert not (run / f"{phase}.log").exists()
