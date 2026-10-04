"""Packed projections and explicit varlen attention against serial replay."""

import copy
from dataclasses import replace

import pytest
import torch
from test_looped_response_readout import make_actor
from test_rltt_training import packed, trace

from vime_plugins.looped.packing import ReplayLayout, causal_attention


def batched(items, *, all_loops=True):
    inputs = packed(items, all_loops=all_loops)
    inputs["readout"] = replace(
        inputs["readout"], sequence_lengths=tuple(len(item[0]) for item in items), attention_backend="sdpa-reference"
    )
    return inputs


def objective(output):
    # Exercise selected scores at every loop and the terminal entropy column.
    return (output.sin() + 0.1 * output.square()).sum() / max(1, output.numel())


@pytest.mark.parametrize("family", ["ouro", "nanbeige", "huginn"])
@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_ragged_scores_gradients_and_optimizer_delta(family, recompute, batch_size, record_property):
    serial, depth = make_actor(family, recompute)
    packed_actor = copy.deepcopy(serial)
    items = []
    for index in range(batch_size):
        length = 1 if index % 7 == 6 else 5 + index % 7
        response = 0 if index % 5 == 2 else min(3, length - 1)
        tokens = (torch.arange(length) + index) % 11
        items.append((tokens, response, trace(family, depth, index + 10, response)))
    expected = serial(**packed(items, all_loops=True))
    actual = packed_actor(**batched(items))
    torch.testing.assert_close(actual, expected, atol=4e-6, rtol=4e-6)
    objective(expected).backward()
    objective(actual).backward()
    errors, norms = [], []
    for (name, left), (_, right) in zip(serial.named_parameters(), packed_actor.named_parameters(), strict=True):
        if left.requires_grad:
            assert left.grad is not None and right.grad is not None, name
            torch.testing.assert_close(right.grad, left.grad, atol=1e-5, rtol=8e-5, msg=name)
            errors.append((right.grad - left.grad).double().square().sum())
            norms.append(left.grad.double().square().sum())
    relative = (sum(errors).sqrt() / sum(norms).sqrt().clamp_min(1e-20)).item()
    assert relative < 2e-6
    old = [p.detach().clone() for p in serial.parameters()]
    for actor in (serial, packed_actor):
        torch.optim.SGD(actor.parameters(), lr=0.03, momentum=0.9).step()
    assert sum((after - before).square().sum() for after, before in zip(serial.parameters(), old, strict=True)) > 0
    for left, right, before in zip(serial.parameters(), packed_actor.parameters(), old, strict=True):
        torch.testing.assert_close(right - before, left - before, atol=2e-6, rtol=8e-5)
    record_property("gradient_relative_l2", relative)
    record_property("backend", "packed projections with segmented SDPA reference; not CUDA varlen")


@pytest.mark.parametrize("family", ["ouro", "nanbeige", "huginn"])
@pytest.mark.parametrize("all_loops", [False, True])
def test_sequence_isolation_and_reordered_packs(family, all_loops):
    actor, depth = make_actor(family, False)
    items = [
        (torch.tensor(tokens), response, trace(family, depth, 81 + i, response))
        for i, (tokens, response) in enumerate([([1, 3, 5, 7], 2), ([2, 2, 4, 6, 8], 3), ([1], 0)])
    ]
    with torch.no_grad():
        expected = actor(**batched(items, all_loops=all_loops))
        changed = [(torch.tensor([9, 8, 7, 6]), 2, items[0][2]), *items[1:]]
        altered = actor(**batched(changed, all_loops=all_loops))
        torch.testing.assert_close(altered[2:], expected[2:], rtol=0, atol=0)
        reordered = actor(**batched([items[1], items[2], items[0]], all_loops=all_loops))
        torch.testing.assert_close(reordered, torch.cat([expected[2:], expected[:2]]), atol=3e-6, rtol=3e-6)
        # A response token cannot change scores predicting it or an earlier token.
        future = [items[0], (torch.tensor([2, 2, 4, 9, 8]), 3, items[1][2]), items[2]]
        modified = actor(**batched(future, all_loops=all_loops))
        torch.testing.assert_close(modified[2], expected[2], rtol=0, atol=0)


def test_varlen_is_not_silently_replaced_on_cpu():
    layout = ReplayLayout.create((2, 3), (4, 5), "varlen", torch.device("cpu"))
    tensor = torch.randn(5, 2, 8)
    with pytest.raises(ValueError, match="CUDA FP16/BF16"):
        causal_attention(tensor, tensor, tensor, layout)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an existing CUDA runtime")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("token_chunk", [0, 3, 11])
def test_cuda_varlen_forward_and_backward(dtype, token_chunk):
    torch.manual_seed(41)
    layout = ReplayLayout.create((5, 17, 31), (0, 0, 0), "varlen", torch.device("cuda"), token_chunk=token_chunk)
    tensors = [torch.randn(53, heads, 64, device="cuda", dtype=dtype, requires_grad=True) for heads in (4, 2, 2)]
    oracle = [value.detach().clone().requires_grad_(True) for value in tensors]
    actual = causal_attention(*tensors, layout)
    expected = causal_attention(*oracle, replace(layout, backend="sdpa-reference", chunks=()))
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    gradient = torch.randn_like(actual) / 53
    actual.backward(gradient)
    expected.backward(gradient)
    for left, right in zip(tensors, oracle, strict=True):
        torch.testing.assert_close(left.grad, right.grad, atol=0.002, rtol=0.03)
