"""CUDA-only: actual single-rank MCore DDP, Adam and train_one_step.

This fixed-trace tiny test is not official-model online RL or publication
qualification. Run only in an admitted window with the complete environment.
"""

import copy
import json
from argparse import Namespace

import pytest
import torch


@pytest.fixture
def single_rank(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("Actual MCore DDP/optimizer validation requires CUDA")
    from megatron.core import parallel_state

    torch.distributed.init_process_group("nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    yield
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("reduction", ["token_mean", "response_mean"])
@pytest.mark.parametrize("rematerialize", [False, True])
def test_actual_mcore_prefix_train_step(single_rank, monkeypatch, record_property, reduction, rematerialize, tmp_path):
    from megatron.core.distributed import DistributedDataParallel as DDP
    from megatron.core.distributed import DistributedDataParallelConfig, finalize_model_grads
    from megatron.core.enums import ModelType
    from megatron.core.optimizer import OptimizerConfig
    from megatron.core.optimizer.optimizer import FP32Optimizer
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
    from test_looped_logical_step import rollout
    from test_looped_response_readout import make_actor

    from vime.backends.megatron_utils import model as backend
    from vime.backends.megatron_utils.data import DataIterator
    from vime_plugins.looped.response import ResponseReadout

    source, depth = make_actor("ouro", False)
    source.model_type = ModelType.encoder_or_decoder  # Assigned by MCore get_model in production.
    source = source.cuda()
    reference = copy.deepcopy(source).requires_grad_(False)
    with torch.no_grad():
        reference.lm_head.weight.mul_(0.9)
    data = rollout(depth)
    for key in ("tokens", "advantages", "loss_masks"):
        data[key] = [tensor.cuda() for tensor in data[key]]
    data["total_lengths"] = [len(tokens) for tokens in data["tokens"]]
    with torch.no_grad():
        ref = reference(
            input_ids=torch.cat(data["tokens"])[None],
            recurrent_inputs=data["recurrent_inputs"],
            readout=ResponseReadout(
                tuple(data["response_lengths"]),
                5,
                0.7,
                False,
                sequence_lengths=tuple(data["total_lengths"]),
                attention_backend="sdpa-reference",
            ),
        )
    data["ref_log_probs"] = list(ref[:, -2].split(data["response_lengths"]))
    groups = [[4, 1], [3, 0, 2]]
    args = Namespace(
        loss_type="rltt_loss",
        rollout_backend="vllm-rlt",
        rlt_runtime_report_dir=None,
        rank=0,
        recompute_granularity=None,
        rltt_prefix_wave_size=0,
        rltt_reduction=reduction,
        rlt_model_revision="tiny-cpu",
        rltt_vocab_tile=5,
        rollout_temperature=0.7,
        rltt_progressive_alpha=1.5,
        kl_loss_coef=0.1,
        entropy_coef=0.03,
        rltt_loop_checkpoint=0,
        rltt_layer_checkpoint=0,
        rltt_token_chunk=0,
        rltt_attention_backend="sdpa-reference",
        custom_megatron_before_train_step_hook_path=None,
        data_pad_size_multiplier=1,
        allgather_cp=False,
        rollout_top_p=1.0,
        save_debug_train_data=None,
        enable_mtp_training=False,
        dspark_enabled=False,
        check_for_nan_in_loss_and_grad=True,
        ci_test=False,
        calculate_per_token_loss=False,
        seq_length=32,
        micro_batch_size=1,
        decoder_seq_length=None,
    )
    monkeypatch.setattr(backend, "get_args", lambda: args)
    snapshots, counts, histories = {}, {}, {}
    for wave in (0, 2):
        actor = copy.deepcopy(source)
        wrapped = DDP(
            actor.config,
            DistributedDataParallelConfig(overlap_grad_reduce=False, use_distributed_optimizer=False),
            actor,
        )
        optimizer = FP32Optimizer(
            torch.optim.AdamW(actor.parameters(), lr=1e-3, eps=1e-5, weight_decay=0.1),
            OptimizerConfig(optimizer="adam", lr=1e-3, clip_grad=1.0),
            None,
        )
        scheduler = OptimizerParamScheduler(optimizer, 1e-3, 1e-3, 1e-3, 0, 100, "constant", 0.1, 0.1, 100, "constant")
        actor.config.grad_scale_func = optimizer.scale_loss
        actor.config.deallocate_pipeline_outputs = False
        counts[wave], snapshots[wave], histories[wave] = 0, [], []

        def capture(models, num_tokens, wave=wave, **kwargs):
            finalize_model_grads(models, num_tokens, **kwargs)
            counts[wave] += 1
            snapshots[wave].append(
                torch.cat(
                    [p.main_grad.detach().flatten().clone() for p in models[0].module.parameters() if p.requires_grad]
                )
            )

        actor.config.finalize_model_grads_func = capture
        iterator = DataIterator(data, groups * 2)
        args.rltt_prefix_wave_size = wave
        args.rltt_loop_checkpoint = depth if rematerialize and wave else 0
        args.rlt_runtime_report_dir = str(tmp_path / f"wave{wave}")
        for update in range(2):
            before = torch.cat([p.detach().flatten().clone() for p in actor.parameters()])
            loss, norm = backend.train_one_step(args, 0, update, [iterator], [wrapped], optimizer, scheduler, 2, 5)
            after = torch.cat([p.detach().flatten().clone() for p in actor.parameters()])
            assert iterator.offset == 2 * (update + 1) and scheduler.num_steps == 5 * (update + 1)
            assert counts[wave] == update + 1 and bool((after - before).norm() > 0)
            receipt = json.loads((tmp_path / f"wave{wave}/steps/rollout0-step{update}-rank0.json").read_text())
            assert receipt["actor_generation_after"] == scheduler.num_steps
            assert receipt["schedule_completed"] == ("prefix" if wave else "mcore")
            assert all(not bool(p.main_grad.any()) for p in actor.parameters() if p.requires_grad)
            histories[wave].append(
                {
                    "parameters": after,
                    "delta": after - before,
                    "loss": loss,
                    "norm": norm,
                    "moments": [
                        (
                            optimizer.optimizer.state[p]["exp_avg"].clone(),
                            optimizer.optimizer.state[p]["exp_avg_sq"].clone(),
                        )
                        for p in actor.parameters()
                        if p.requires_grad
                    ],
                }
            )
    rows = []
    for update in range(2):
        left, right = histories[0][update], histories[2][update]
        torch.testing.assert_close(snapshots[2][update], snapshots[0][update], atol=1e-5, rtol=8e-5)
        for key in ("parameters", "delta"):
            torch.testing.assert_close(left[key], right[key], atol=2e-6, rtol=8e-5)
        for expected_pair, actual_pair in zip(left["moments"], right["moments"], strict=True):
            for expected_moment, actual_moment in zip(expected_pair, actual_pair, strict=True):
                torch.testing.assert_close(expected_moment, actual_moment, atol=1e-6, rtol=8e-5)
        assert left["loss"].keys() == right["loss"].keys()
        for key in left["loss"]:
            assert left["loss"][key] == pytest.approx(right["loss"][key], abs=2e-6, rel=2e-6)
        rows.append(
            {
                "update": update,
                "gradient_max_error": float((snapshots[2][update] - snapshots[0][update]).abs().max()),
                "parameter_max_error": float((left["parameters"] - right["parameters"]).abs().max()),
                "delta_norm": float(right["delta"].norm()),
                "loss": right["loss"],
            }
        )
    assert all(p.grad is None for p in reference.parameters())
    record_property("actual_mcore_two_update", json.dumps(rows))
