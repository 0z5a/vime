"""Real tiny CPU Ouro/RLTT publications, with synthetic token-parity rewards."""

import copy
import json
import shutil

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from test_looped_grpo import cpu_engine, publish
from test_rltt_online_cycle import pair, update

from benchmarks.native_publications import audit_publications
from vime.backends.megatron_utils.hf_checkpoint_saver import _finalize_local_shards, _SafetensorShardWriter
from vime.backends.vllm_rlt_utils.engine import weight_digest


def test_real_native_updates_bind_every_published_physical_tensor(tmp_path, record_property):
    native, actor = pair("ouro")
    reference = copy.deepcopy(actor).requires_grad_(False)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.1)
    engine = cpu_engine(native, "ouro")
    initial = tmp_path / "initial.safetensors"
    save_file({name: value.detach().clone() for name, value in actor.state_dict().items()}, initial)
    publications = tmp_path / "publications"
    publications.mkdir()
    publish(actor, engine, "ouro", publications / "weight_v000001", 1)
    curve = [{"completed_updates": 0, "publication_digest": engine.committed_digest}]
    generated = []
    for step in range(3):
        observed = update(actor, reference, optimizer, engine, "ouro", tmp_path, step, "sdpa-reference")
        generated.append(observed)
        folder = tmp_path / f"checkpoints/actor/iter_{step:07d}"
        state = {name: value.detach() for name, value in actor.state_dict().items()}
        for name, parameter in actor.named_parameters():
            if parameter.requires_grad:
                for key in ("exp_avg", "exp_avg_sq"):
                    state[f"optimizer.state.{key}.{name}"] = optimizer.state[parameter][key]
        state["rng_state/shard_0.0_1.1"] = [torch.get_rng_state()]
        dcp.save(state, checkpoint_id=folder, no_dist=True, planner=dcp.DefaultSavePlanner(flatten_state_dict=False))
        torch.save({"iteration": step}, folder / "common.pt")
        shutil.copytree(tmp_path / f"after-{step}", publications / f"weight_v{step + 2:06d}")
        curve.append({"completed_updates": step + 1, "publication_digest": observed["publication_digest"]})
    report = audit_publications(tmp_path, initial, curve)
    assert report["publication_checkpoint_pass"]
    assert [row["physical_tensors"] for row in report["rounds"]] == [len(actor.state_dict())] * 4
    assert len({row["publication_digest"] for row in report["rounds"]}) == 4
    assert all(row["mixed_groups"] > 0 and row["grad_norm"] > 0 and row["parameter_delta"] > 0 for row in generated)
    record_property("physical_tensors", len(actor.state_dict()))
    record_property(
        "updates",
        json.dumps(
            [
                {
                    key: row[key]
                    for key in (
                        "step",
                        "mixed_groups",
                        "grad_norm",
                        "parameter_delta",
                        "score_error",
                        "publication_digest",
                    )
                }
                for row in generated
            ]
        ),
    )

    # The production shard writer changes file layout while preserving tensors.
    # Its digest must still use filenames and bytes exactly as NativeEngine does.
    sharded = tmp_path / "sharded"
    sharded.mkdir()
    writer = _SafetensorShardWriter(sharded, enabled=True)
    items = [(name, value.detach().clone()) for name, value in actor.state_dict().items()]
    half = len(items) // 2
    writer.write(items[:half], shard_idx=0)
    writer.write(items[half:], shard_idx=1)
    local = writer.state()
    _finalize_local_shards(sharded, local, [local], write_index=True)
    target = publications / "weight_v000004"
    (target / "model.safetensors").unlink()
    for path in sharded.iterdir():
        shutil.copyfile(path, target / path.name)
    curve[-1]["publication_digest"] = weight_digest(target)[0]
    rewritten = audit_publications(tmp_path, initial, curve)
    assert rewritten["rounds"][-1]["tensor_values_exact"]
    assert len(list(target.glob("*.safetensors"))) == 2
    record_property("shard_writer_relayout_exact", True)
