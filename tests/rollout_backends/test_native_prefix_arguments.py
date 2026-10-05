"""Opt-in schedule selection is limited to the declared native objective."""

import json

import pytest
from test_native_colocation import arguments

from vime.backends.vllm_rlt_utils.arguments import validate_args


@pytest.mark.parametrize(
    "wave,loss,family,accepted",
    [
        (0, "policy_loss", "ouro", True),
        (2, "rltt_loss", "ouro", True),
        (-1, "rltt_loss", "ouro", False),
        (2, "policy_loss", "ouro", False),
        (2, "rltt_loss", "nanbeige", False),
    ],
)
def test_prefix_backend_selection(tmp_path, wave, loss, family, accepted):
    args = arguments(tmp_path, rltt_prefix_wave_size=wave, loss_type=loss)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": family, "total_ut_steps": 4, "num_loops": 4}))
    if accepted:
        validate_args(args)
    else:
        with pytest.raises(ValueError, match="prefix"):
            validate_args(args)
