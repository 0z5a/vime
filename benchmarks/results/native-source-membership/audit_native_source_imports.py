"""Check retained files and real CPU imports against frozen native runtime sources."""

import importlib
import json
import sys
from pathlib import Path

import torch

from benchmarks.native_sources import module_index, read_resident, sha, verify_local

task = Path(__file__).resolve().parents[1]
manifest_path = task / "evidence/native-sources-v1.json"
manifest = json.loads(read_resident(manifest_path))
roots = {
    "vime": task / "vime",
    "rlt": task / "evidence/w08-gpu-packet-v1/parent",
    "megatron": task.parent / "looped-grpo-integration-20261003/runtime/megatron",
}
preflight = verify_local(manifest, roots)
expected = module_index(manifest)
for name in (
    "vime.observability.native_runtime",
    "vime_plugins.ouro.model",
    "vime_plugins.looped.rltt",
    "vllm_rlt.models.ouro",
    "megatron.core.optimizer.optimizer",
    "megatron.core.optimizer.optimizer_config",
):
    importlib.import_module(name)
observed = {}
for name, module in sorted(list(sys.modules.items())):
    if name.split(".", 1)[0] not in {"vime", "vime_plugins", "vllm_rlt", "megatron"} or module is None:
        continue
    origin = vars(module).get("__file__")
    if origin is None:
        continue
    path = Path(origin).resolve()
    assert name in expected, name
    record = expected[name]
    assert path == roots[record["source"]] / record["path"], (name, path)
    assert sha(read_resident(path)) == record["sha256"], name
    observed[name] = {"path": str(path), "sha256": record["sha256"]}

# Current RLT includes the later group-prefill change; it must not pass the older pin.
wrong_roots = {**roots, "rlt": task / "rlt"}
try:
    verify_local(manifest, wrong_roots)
except ValueError as error:
    rejected = str(error)
else:
    raise AssertionError("The different RLT working tree unexpectedly matched the qualification pin")
assert rejected.startswith("Changed source: rlt:")
assert not torch.cuda.is_initialized()
result = {
    "harness_sha256": sha(read_resident(Path(__file__))),
    "verifier_sha256": sha(read_resident(task / "vime/benchmarks/native_sources.py")),
    "source_manifest_sha256": sha(read_resident(manifest_path)),
    "preflight": preflight,
    "real_cpu_imports": observed,
    "changed_rlt_working_tree_rejected": rejected,
    "python": sys.executable,
    "torch_version": torch.__version__,
    "cuda_initialized": False,
    "scope": "read-only local files and real CPU imports; no Ray workers, model weights, generation or optimizer steps",
    "full_framework_ray_cuda": "NOT_RUN",
    "reward_convergence": "NOT_ESTABLISHED",
}
(task / "evidence/native-source-local-audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            "files_checked": preflight["files_checked"],
            "imported_modules": len(observed),
            "changed_rlt_rejected": rejected,
            "cuda_initialized": False,
        }
    )
)
