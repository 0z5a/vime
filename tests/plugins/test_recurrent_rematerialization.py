"""Loop/layer/token schedules preserve the complete recurrent gradient graph."""

import copy
from dataclasses import replace

import pytest
import torch
from test_looped_response_readout import make_actor
from test_packed_recurrent_replay import batched, objective
from test_rltt_training import packed, trace

from vime_plugins.looped.execution import RematPlan
from vime_plugins.looped.packing import ReplayLayout, causal_attention


@pytest.mark.parametrize("family", ["ouro", "nanbeige", "huginn"])
@pytest.mark.parametrize("all_loops", [False, True])
@pytest.mark.parametrize(
    "plan", [RematPlan(), RematPlan(1), RematPlan(2), RematPlan(8), RematPlan(0, 2, 3), RematPlan(2, 1, 3)]
)
def test_full_parameter_gradient_and_nonzero_update(family, all_loops, plan, record_property):
    baseline, depth = make_actor(family, False)
    candidate = copy.deepcopy(baseline)
    items = [
        (torch.tensor(tokens), response, trace(family, depth, 71 + i, response))
        for i, (tokens, response) in enumerate((([1, 3, 5, 7, 9], 3), ([2, 3, 8, 1, 6, 4, 2], 2), ([1], 0)))
    ]
    inputs = batched(items, all_loops=all_loops)
    inputs["readout"] = replace(inputs["readout"], rematerialization=plan)
    expected = baseline(**packed(items, all_loops=all_loops))
    actual = candidate(**inputs)
    torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-6)
    objective(expected).backward()
    objective(actual).backward()
    errors, norms = [], []
    for (name, left), (_, right) in zip(baseline.named_parameters(), candidate.named_parameters(), strict=True):
        if left.requires_grad:
            assert left.grad is not None and right.grad is not None, name
            torch.testing.assert_close(right.grad, left.grad, atol=1e-5, rtol=8e-5, msg=name)
            errors.append((right.grad - left.grad).double().square().sum())
            norms.append(left.grad.double().square().sum())
    relative = (sum(errors).sqrt() / sum(norms).sqrt().clamp_min(1e-20)).item()
    assert relative < 2e-6
    record_property("gradient_relative_l2", relative)
    before = [value.detach().clone() for value in baseline.parameters()]
    for actor in (baseline, candidate):
        torch.optim.SGD(actor.parameters(), lr=0.03, momentum=0.9).step()
    assert sum((left - old).square().sum() for left, old in zip(baseline.parameters(), before, strict=True)) > 0
    for left, right, old in zip(baseline.parameters(), candidate.parameters(), before, strict=True):
        torch.testing.assert_close(left - old, right - old, atol=2e-6, rtol=8e-5)


@pytest.mark.parametrize("chunk", [1, 3, 32])
def test_query_chunks_keep_full_prefix_and_key_value_gradients(chunk):
    torch.manual_seed(33)
    tensors = [torch.randn(14, heads, 8, dtype=torch.float64, requires_grad=True) for heads in (4, 2, 2)]
    oracle = [tensor.detach().clone().requires_grad_(True) for tensor in tensors]
    layout = ReplayLayout.create((5, 9), (4, 5), "sdpa-reference", torch.device("cpu"), token_chunk=chunk)
    actual = causal_attention(*tensors, layout)
    expected = causal_attention(*oracle, replace(layout, chunks=()))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    weight = torch.randn_like(actual)
    actual.backward(weight)
    expected.backward(weight)
    for left, right in zip(tensors, oracle, strict=True):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-12, rtol=1e-12)
    altered = [tensor.detach().clone() for tensor in tensors]
    altered[1][-1] += 5
    changed = causal_attention(*altered, layout)
    torch.testing.assert_close(changed[:-1], actual.detach()[:-1], atol=0, rtol=0)


@pytest.mark.parametrize("family", ["ouro", "nanbeige", "huginn"])
def test_serial_empty_response_retains_zero_gradient(family):
    actor, depth = make_actor(family, False)
    items = [(torch.tensor([1]), 0, trace(family, depth, 81, 0))]
    inputs = packed(items, all_loops=True)
    inputs["readout"] = replace(inputs["readout"], rematerialization=RematPlan(2, 1, 2))
    output = actor(**inputs)
    assert output.shape == (0, depth + 1)
    output.sum().backward()
    assert all(
        parameter.grad is not None and not bool(parameter.grad.any())
        for parameter in actor.parameters()
        if parameter.requires_grad
    )
