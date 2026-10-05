"""The actual backend validator admits evaluation without relaxing training sampling."""

import json
from argparse import Namespace

import pytest

from vime.backends.vllm_rlt_utils.arguments import validate_args
from vime.utils.eval_config import build_eval_dataset_configs


@pytest.mark.parametrize("training_top_p", [1.0, 0.9])
def test_eval_sampling_is_separate_from_training_validation(tmp_path, training_top_p):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "ouro", "total_ut_steps": 4}))
    args = Namespace(
        hf_checkpoint=str(tmp_path),
        rlt_depth=None,
        rlt_kv_blocks=128,
        rlt_max_num_seqs=2,
        rlt_start_version=0,
        rltt_prefix_wave_size=0,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        colocate=False,
        offload_rollout=False,
        release_train=False,
        use_fault_tolerance=False,
        rollout_external=False,
        use_opd=False,
        use_rollout_routing_replay=False,
        check_weight_update_equal=False,
        update_weight_mode="full",
        update_weight_transport="disk",
        update_weight_local_checkpoint_dir=None,
        rollout_top_p=training_top_p,
        rollout_top_k=-1,
        rollout_function_path="vime.rollout.vllm_rlt_rollout.generate_rollout",
        eval_interval=1,
        group_rm=False,
        multimodal_keys=None,
        eval_temperature=0.0,
        eval_top_p=0.7,
        eval_top_k=3,
        eval_max_response_len=16,
        n_samples_per_eval_prompt=2,
        input_key="prompt",
        label_key="label",
    )
    args.eval_datasets = build_eval_dataset_configs(args, [{"name": "heldout", "path": "heldout.jsonl"}], {})
    if training_top_p == 1.0:
        validate_args(args)
        assert args.rlt_depth == 4
        assert args.eval_datasets[0].temperature == 0 and args.eval_datasets[0].top_p == 0.7
    else:
        with pytest.raises(ValueError, match="training requires full-vocabulary"):
            validate_args(args)
