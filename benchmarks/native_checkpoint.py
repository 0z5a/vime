"""Read trusted, task-owned Megatron DCP state one chunk at a time on CPU."""

import hashlib
import io
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch


def fingerprint(value: object) -> str:
    digest = hashlib.sha256()
    digest.update(f"{type(value).__module__}.{type(value).__qualname__}\0".encode())
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError("Nonfinite checkpoint tensor")
        digest.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    elif isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.inexact) and not bool(np.isfinite(array).all()):
            raise ValueError("Nonfinite checkpoint array")
        digest.update(json.dumps([array.dtype.str, list(array.shape)]).encode())
        digest.update(array.tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=fingerprint):
            digest.update(bytes.fromhex(fingerprint(key)))
            digest.update(bytes.fromhex(fingerprint(value[key])))
    elif isinstance(value, (list, tuple)):
        for item in value:
            digest.update(bytes.fromhex(fingerprint(item)))
    elif isinstance(value, torch.dtype):
        digest.update(str(value).encode())
    elif isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite checkpoint scalar")
        digest.update(json.dumps(value, ensure_ascii=False, allow_nan=False).encode())
    else:
        raise TypeError(f"Unsupported checkpoint value: {type(value).__qualname__}")
    return digest.hexdigest()


class Checkpoint:
    def __init__(self, folder: Path):
        self.folder = folder
        common_data, metadata_data = ((folder / name).read_bytes() for name in ("common.pt", ".metadata"))
        self.control_sha256 = {
            "common.pt": hashlib.sha256(common_data).hexdigest(),
            ".metadata": hashlib.sha256(metadata_data).hexdigest(),
        }
        self.common = torch.load(io.BytesIO(common_data), map_location="cpu", weights_only=False)
        metadata = pickle.loads(metadata_data)
        self.tensor_metadata = metadata.state_dict_metadata
        self.index = {(key.fqn, tuple(key.offset or ())): value for key, value in metadata.storage_data.items()}
        assert self.index and len(self.index) == len(metadata.storage_data)

    def read(self, key: tuple[str, tuple[int, ...]]) -> object:
        storage = self.index[key]
        with (self.folder / storage.relative_path).open("rb") as stream:
            stream.seek(storage.offset)
            data = stream.read(storage.length)
        assert len(data) == storage.length, f"Truncated DCP chunk: {key}"
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)

    def full_tensor(self, name: str) -> torch.Tensor:
        keys = [key for key in self.index if key[0] == name]
        assert len(keys) == 1 and not any(keys[0][1]), "Adam update audit requires unsharded FP32 parameters"
        value = self.read(keys[0])
        assert isinstance(value, torch.Tensor) and value.dtype == torch.float32
        assert bool(torch.isfinite(value).all())
        return value


def compare_checkpoints(left: Checkpoint, right: Checkpoint) -> dict:
    """Compare all stored state freshly; historical fingerprints are never read."""
    assert left.index.keys() == right.index.keys(), "DCP chunk identities differ"
    assert left.tensor_metadata == right.tensor_metadata, "DCP tensor shape/layout metadata differs"
    names = {key[0] for key in left.index}
    assert any(name.startswith("rng_state/") for name in names), "RNG checkpoint is missing"
    assert any(name.startswith("model.") for name in names), "Model checkpoint is missing"
    assert left.common.keys() == right.common.keys(), "Common checkpoint keys differ"
    for key in ("iteration", "checkpoint_version", "optimizer", "opt_param_scheduler"):
        assert key in left.common, f"Missing common state: {key}"
    # Launch arguments contain different output/load paths; separately qualify
    # their semantic contract with the frozen recipe and actual worker sources.
    common = {}
    for key in sorted(left.common.keys() - {"args"}):
        digest = fingerprint(left.common[key])
        assert digest == fingerprint(right.common[key]), f"Common state differs: {key}"
        common[key] = digest
    chunks = []
    for key in sorted(left.index):
        digest = fingerprint(left.read(key))
        assert digest == fingerprint(right.read(key)), f"DCP state differs: {key}"
        chunks.append({"name": key[0], "offsets": list(key[1]), "sha256": digest})
    return {
        "stored_state_exact": True,
        "control_sha256": {"left": left.control_sha256, "right": right.control_sha256},
        "common_sha256": common,
        "chunks": chunks,
        "excluded": ["common.args"],
        "fresh_worker_execution_proven": False,
    }
