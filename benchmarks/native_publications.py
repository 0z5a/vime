"""Bind native publication file digests to complete initial/checkpoint tensors."""

import hashlib
from pathlib import Path

import torch
from safetensors import safe_open

from benchmarks.native_checkpoint import Checkpoint


def publication_index(folder: Path, consumed: dict[str, str]) -> tuple[str, dict[str, Path]]:
    digest, tensors = hashlib.sha256(), {}
    files = sorted(folder.glob("*.safetensors"))
    assert files, f"Missing publication weights: {folder}"
    for path in files:
        digest.update(path.name.encode())
        file_digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                file_digest.update(chunk)
        consumed[str(path.resolve())] = file_digest.hexdigest()
        with safe_open(path, framework="pt", device="cpu") as weights:
            for name in weights.keys():
                assert name not in tensors, f"Duplicate published tensor: {name}"
                tensors[name] = path
    return digest.hexdigest(), tensors


def audit_publications(run: Path, initial_weights: Path, curve: list[dict]) -> dict:
    assert len(curve) == 4 and [row["completed_updates"] for row in curve] == list(range(4))
    consumed, rounds = {}, []
    with safe_open(initial_weights, framework="pt", device="cpu") as initial:
        names = set(initial.keys())
        for step, row in enumerate(curve):
            folder = run / "publications" / f"weight_v{step + 1:06d}"
            digest, published = publication_index(folder, consumed)
            assert digest == row["publication_digest"], f"Output did not use the retained publication: {folder}"
            assert published.keys() == names, "Published physical tensor set differs from the initial model"
            checkpoint = Checkpoint(run / f"checkpoints/actor/iter_{step - 1:07d}") if step else None
            if checkpoint is not None:
                assert checkpoint.common["iteration"] == step - 1
                checkpoint_names = {
                    key[0] for key in checkpoint.index if not key[0].startswith(("optimizer.", "rng_state/"))
                }
                assert checkpoint_names == names, "Checkpoint physical tensor set differs from the initial model"
                for name, value in checkpoint.control_sha256.items():
                    consumed[str((checkpoint.folder / name).resolve())] = value
            for name, path in published.items():
                reference = initial.get_tensor(name).float() if checkpoint is None else checkpoint.full_tensor(name)
                with safe_open(path, framework="pt", device="cpu") as weights:
                    actual = weights.get_tensor(name)
                assert actual.dtype == reference.dtype == torch.float32
                assert bool(torch.isfinite(actual).all()) and bool(torch.isfinite(reference).all())
                assert actual.shape == reference.shape and torch.equal(actual, reference), (
                    f"Publication differs from initial/checkpoint state at update {step}: {name}"
                )
            rounds.append(
                {
                    "completed_updates": step,
                    "policy_version": step + 1,
                    "publication_digest": digest,
                    "physical_tensors": len(names),
                    "tensor_values_exact": True,
                }
            )
    return {
        "publication_checkpoint_pass": True,
        "rounds": rounds,
        "consumed_sha256": consumed,
        "scope": "retained unsharded FP32 Ouro publication files and complete physical tensors; not memory attestation",
    }
