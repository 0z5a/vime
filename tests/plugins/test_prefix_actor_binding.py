"""A prefix graph belongs to the actor that produced it, before replay opens."""

import copy

import pytest
import torch
from test_differentiable_prefix import replay
from test_looped_response_readout import make_actor

from vime_plugins.looped.prefix import PrefixReplay


@pytest.mark.parametrize("family", ["ouro", "nanbeige", "huginn"])
@pytest.mark.parametrize("change", ["other_actor", "restore", "restore_roundtrip", "reference_head"])
def test_prefix_rejects_role_or_weight_changes_before_replay(family, change):
    actor, depth = make_actor(family, False)
    reference = copy.deepcopy(actor).requires_grad_(False)
    with torch.no_grad():
        reference.lm_head.weight.mul_(0.9)
    saved = copy.deepcopy(actor.state_dict())
    state, identity = replay(actor, torch.tensor([1, 3, 5]), depth, True, 31 if family == "huginn" else None)
    program = state.program
    del state
    weight = actor.lm_head.weight
    if change == "other_actor":
        actor = copy.deepcopy(actor)
        weight = actor.lm_head.weight
    elif change in ("restore", "restore_roundtrip"):
        actor.load_state_dict(reference.state_dict())
        if change == "restore_roundtrip":
            actor.load_state_dict(saved)
    else:
        weight = reference.lm_head.weight
    with pytest.raises(RuntimeError, match="actor binding"):
        PrefixReplay(
            program,
            identity,
            actor,
            weight,
            vocab_tile=5,
            temperature=0.7,
            all_loops=True,
            entropy=True,
        )
