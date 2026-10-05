"""Exercise the provider forward used beneath the real MCore DDP wrapper."""

import copy
from argparse import Namespace

import pytest
import torch
from test_looped_response_readout import make_actor

from vime.backends.megatron_utils.looped_prefix_schedule import forward_backward_prefix
from vime_plugins.looped.prefix import PrefixIdentity, PrefixReplay


def test_prefix_forward_entry_preserves_all_parameter_gradients():
    direct, depth = make_actor("ouro", False)
    entry = copy.deepcopy(direct)
    prompt = torch.tensor([1, 3, 5])
    identity = PrefixIdentity("tiny", 1, tuple(prompt.tolist()), depth)
    for actor, program in (
        (direct, direct.prefix_program(prompt)),
        (entry, entry(input_ids=prompt[None], prefix_only=True)),
    ):
        replay = PrefixReplay(
            program, identity, actor, actor.lm_head.weight, vocab_tile=5, temperature=0.7, all_loops=True, entropy=True
        )
        replay.scores_batch((torch.tensor([7]), torch.tensor([3, 5])), identity=identity).mean().backward()
        replay.backward_prefix(identity=identity)
    for left, right in zip(direct.parameters(), entry.parameters(), strict=True):
        if left.requires_grad:
            assert left.grad is not None and right.grad is not None
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)


@pytest.mark.parametrize(
    "key,value",
    [
        ("loss_mask", torch.ones(1)),
        ("recurrent_inputs", []),
        ("position_ids", torch.arange(2)),
        ("attention_mask", torch.ones(2)),
    ],
)
def test_prefix_forward_rejects_mixed_contract(key, value):
    actor, _ = make_actor("ouro", False)
    with pytest.raises(ValueError):
        actor(input_ids=torch.tensor([[1, 3]]), prefix_only=True, **{key: value})


def test_schedule_does_not_masquerade_an_unwrapped_component_as_mcore():
    actor, _ = make_actor("ouro", False)
    with pytest.raises(ValueError, match="actual FP32 MCore DDP"):
        forward_backward_prefix(Namespace(), None, actor, 1, 1, 0)
