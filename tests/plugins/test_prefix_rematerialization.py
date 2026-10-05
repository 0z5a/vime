"""Joint prefix/suffix replay preserves all-loop loss and two Adam updates."""

import copy
import json
from functools import partial

import pytest
import torch
from test_differentiable_prefix import objective
from test_looped_response_readout import make_actor
from test_rltt_training import packed, trace

from vime_plugins.looped.prefix import PrefixIdentity, PrefixReplay


def observe_saved(actor, call):
    parameter_storage = {p.untyped_storage().data_ptr() for p in actor.parameters()}
    saved = {}
    tensors = 0

    def pack(tensor):
        nonlocal tensors
        tensors += 1
        storage = tensor.untyped_storage()
        if storage.data_ptr() not in parameter_storage:
            saved[storage.data_ptr()] = storage.nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = call()
    return result, {"saved_tensors": tensors, "observed_unique_nonparameter_storage_bytes": sum(saved.values())}


@pytest.mark.parametrize("family", ["ouro", "nanbeige"])
@pytest.mark.parametrize("all_loops", [False, True])
@pytest.mark.parametrize("groups", [((0, 1, 2),), ((0,), (1,), (2,)), ((2,), (0, 1))])
@pytest.mark.parametrize("batch_suffixes", [False, True])
def test_joint_replay_two_adam_updates(all_loops, groups, batch_suffixes, family, record_property):
    source, depth = make_actor(family, False)
    actors = [copy.deepcopy(source), copy.deepcopy(source)]
    reference = copy.deepcopy(source).requires_grad_(False)
    with torch.no_grad():
        reference.lm_head.weight.mul_(0.9)
    prompt = torch.tensor([1, 3, 5, 7])
    responses = tuple(torch.tensor(values) for values in ([2], [4, 6, 8], [9, 10, 11, 12, 2]))
    inputs = packed(
        [
            (torch.cat((prompt, response)), len(response), trace(family, depth, i, len(response)))
            for i, response in enumerate(responses)
        ],
        all_loops=all_loops,
    )
    with torch.no_grad():
        refs = reference(**inputs)[:, -2].split([1, 3, 5])
    advantages = torch.tensor([0.7, -0.3, 0.2, -0.4, 0.5, 0.1, -0.9, 0.3, 0.8]).split([1, 3, 5])
    weights = (torch.tensor([1, 1, 0, 1, 1, 1, 0, 0, 1]) / 6).split([1, 3, 5])
    optimizers = [torch.optim.AdamW(a.parameters(), lr=1e-3, eps=1e-5, weight_decay=0.1) for a in actors]
    records = []
    for update in range(2):
        losses, gradients, resources, deltas = [], [], [], []
        for actor, optimizer, rematerialize in zip(actors, optimizers, (False, True), strict=True):
            optimizer.zero_grad(set_to_none=True)
            actor.block_tokens = 0
            before = torch.cat([p.detach().flatten().clone() for p in actor.parameters()])
            program, prefix_saved = observe_saved(
                actor, partial(actor.prefix_program, prompt, rematerialize=rematerialize)
            )
            boundary_storage = {
                p.untyped_storage().data_ptr(): p.untyped_storage().nbytes() for p in program.boundaries
            }
            identity = PrefixIdentity("tiny-remat", update, tuple(prompt.tolist()), depth)
            replay = PrefixReplay(
                program,
                identity,
                actor,
                actor.lm_head.weight,
                vocab_tile=5,
                temperature=0.7,
                all_loops=all_loops,
                entropy=True,
            )
            total, waves = 0.0, []
            for group in groups:

                def scores(group=group, replay=replay, identity=identity):
                    return (
                        replay.scores_batch(tuple(responses[i] for i in group), identity=identity)
                        if batch_suffixes
                        else torch.cat([replay.scores(responses[i], identity=identity) for i in group])
                    )

                output, wave_saved = observe_saved(actor, scores)
                waves.append(wave_saved)
                loss = objective(
                    output,
                    torch.cat([refs[i] for i in group]),
                    torch.cat([advantages[i] for i in group]),
                    torch.cat([weights[i] for i in group]),
                )
                total += float(loss.detach())
                loss.backward()
                del output, loss
            replay.backward_prefix(identity=identity)
            assert replay.closed
            gradient = torch.cat([p.grad.detach().flatten().clone() for p in actor.parameters() if p.requires_grad])
            assert bool(torch.isfinite(gradient).all()) and float(gradient.norm()) > 0
            gradients.append(gradient)
            losses.append(total)
            resources.append(
                {
                    "prefix_forward": prefix_saved,
                    "suffix_wave_forwards": waves,
                    "retained_boundary_unique_storage_bytes": sum(boundary_storage.values()),
                    "physical_decoder_token_calls": actor.block_tokens,
                }
            )
            optimizer.step()
            delta = torch.cat([p.detach().flatten() for p in actor.parameters()]) - before
            assert float(delta.norm()) > 0
            deltas.append(delta)
            del replay, program
        torch.testing.assert_close(gradients[1], gradients[0], atol=1e-5, rtol=8e-5)
        torch.testing.assert_close(deltas[1], deltas[0], atol=2e-6, rtol=8e-5)
        assert losses[1] == pytest.approx(losses[0], abs=2e-6, rel=2e-6)
        for left, right in zip(actors[0].parameters(), actors[1].parameters(), strict=True):
            torch.testing.assert_close(left, right, atol=2e-6, rtol=8e-5)
            if left.requires_grad:
                for field in ("exp_avg", "exp_avg_sq", "step"):
                    torch.testing.assert_close(
                        optimizers[0].state[left][field], optimizers[1].state[right][field], atol=1e-6, rtol=8e-5
                    )
        assert resources[1]["physical_decoder_token_calls"] > resources[0]["physical_decoder_token_calls"]
        assert (
            resources[1]["prefix_forward"]["observed_unique_nonparameter_storage_bytes"]
            < (resources[0]["prefix_forward"]["observed_unique_nonparameter_storage_bytes"])
        )
        records.append(
            {
                "update": update,
                "loss": losses[1],
                "gradient_max_error": float((gradients[1] - gradients[0]).abs().max()),
                "gradient_relative_l2": float((gradients[1] - gradients[0]).norm() / gradients[0].norm()),
                "delta_norm": float(deltas[1].norm()),
                "resources": resources,
            }
        )
    assert all(p.grad is None for p in reference.parameters())
    record_property("joint_prefix_two_update", json.dumps(records))


def test_rematerialized_prefix_rejects_update_before_backward():
    actor, depth = make_actor("ouro", False)
    prompt = torch.tensor([1, 3])
    identity = PrefixIdentity("tiny-remat", 0, tuple(prompt.tolist()), depth)
    replay = PrefixReplay(
        actor.prefix_program(prompt, rematerialize=True),
        identity,
        actor,
        actor.lm_head.weight,
        vocab_tile=5,
        temperature=0.7,
        all_loops=True,
        entropy=False,
    )
    scores = replay.scores(torch.tensor([5, 7]), identity=identity)
    with torch.no_grad():
        actor.lm_head.weight.add_(0.01)
    with pytest.raises(RuntimeError, match="parameters changed"):
        scores.sum().backward()


@pytest.mark.parametrize("index", [0, -1])
def test_rematerialized_suffix_rejects_token_mutation(index):
    actor, depth = make_actor("ouro", False)
    prompt, response = torch.tensor([1, 3]), torch.tensor([5, 7, 9])
    identity = PrefixIdentity("tiny-remat", 0, tuple(prompt.tolist()), depth)
    replay = PrefixReplay(
        actor.prefix_program(prompt, rematerialize=True),
        identity,
        actor,
        actor.lm_head.weight,
        vocab_tile=5,
        temperature=0.7,
        all_loops=True,
        entropy=False,
    )
    scores = replay.scores(response, identity=identity)
    response[index] = 11
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        scores.sum().backward()
