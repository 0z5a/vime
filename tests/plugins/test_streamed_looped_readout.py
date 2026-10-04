"""Dense and finite-difference oracles for first-order vocabulary streaming."""

import pytest
import torch
import torch.nn.functional as F

from vime_plugins.looped.readout import streamed_readout
from vime_plugins.looped.rltt import loop_weights, rltt_loss, token_weights


def dense(hidden, weight, targets, temperature, reference_hidden, reference_weight):
    dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
    log_p = F.log_softmax(hidden.to(dtype) @ weight.to(dtype).T / temperature, -1)
    selected = log_p.gather(1, targets[:, None])[:, 0]
    entropy = -(log_p.exp() * log_p).sum(-1)
    log_q = F.log_softmax(reference_hidden.to(dtype) @ reference_weight.to(dtype).T / temperature, -1)
    full_kl = (log_p.exp() * (log_p - log_q)).sum(-1)
    return selected, entropy, full_kl


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tile", [1, 4, 11, 32])
@pytest.mark.parametrize("temperature", [0.7, 1.0, 2.0])
def test_loss_and_all_readout_gradients(dtype, tile, temperature):
    torch.manual_seed(13)
    h = torch.randn(7, 5, dtype=dtype, requires_grad=True)
    w = torch.randn(11, 5, dtype=dtype, requires_grad=True)
    targets = torch.tensor([0, 3, 4, 10, 8, 5, 1])
    rh, rw = torch.randn_like(h), torch.randn_like(w)
    expected = dense(h, w, targets, temperature, rh, rw)
    cotangents = [torch.randn_like(value) for value in expected]
    expected_grads = torch.autograd.grad(
        sum((value * cotangent).sum() for value, cotangent in zip(expected, cotangents, strict=True)), (h, w)
    )
    result = streamed_readout(
        h, w, targets, vocab_tile=tile, temperature=temperature, entropy=True, reference_hidden=rh, reference_weight=rw
    )
    actual = (result.log_probs, result.entropy, result.full_kl)
    actual_grads = torch.autograd.grad(
        sum((value * cotangent).sum() for value, cotangent in zip(actual, cotangents, strict=True)), (h, w)
    )
    tolerance = 2e-11 if dtype == torch.float64 else 2e-5
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, atol=tolerance, rtol=tolerance)
    for left, right in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(
            left,
            right,
            atol=0.016 if dtype == torch.bfloat16 else tolerance,
            rtol=0.01 if dtype == torch.bfloat16 else tolerance,
        )


def test_finite_difference_gradcheck():
    torch.manual_seed(14)
    h = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    w = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    rh, rw = torch.randn_like(h), torch.randn_like(w)

    def operation(hidden, weight):
        output = streamed_readout(
            hidden,
            weight,
            torch.tensor([0, 4]),
            vocab_tile=2,
            temperature=0.9,
            entropy=True,
            reference_hidden=rh,
            reference_weight=rw,
        )
        return output.log_probs, output.entropy, output.full_kl

    assert torch.autograd.gradcheck(operation, (h, w), atol=1e-6, rtol=1e-5)


def test_empty_local_partition_and_disabled_outputs():
    h = torch.randn(0, 3, requires_grad=True)
    w = torch.randn(7, 3, requires_grad=True)
    result = streamed_readout(h, w, torch.empty(0, dtype=torch.long), vocab_tile=4)
    result.log_probs.sum().backward()
    assert h.grad.shape == h.shape and torch.equal(w.grad, torch.zeros_like(w))
    h = torch.randn(2, 3, requires_grad=True)
    result = streamed_readout(h, w, torch.tensor([1, 5]), vocab_tile=4)
    (result.entropy.sum() + result.full_kl.sum()).backward()
    assert torch.equal(h.grad, torch.zeros_like(h))


def test_no_saved_vocabulary_logits_and_large_logit_stability():
    torch.manual_seed(22)
    h = (torch.randn(8, 5) * 20).requires_grad_()
    w = (torch.randn(17, 5) * 20).requires_grad_()
    labels = torch.arange(8)
    saved = []

    def pack(tensor):
        saved.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = streamed_readout(h, w, labels, vocab_tile=4, entropy=True)
    assert (8, 17) not in saved and all(shape in [(8, 5), (17, 5), (8,)] for shape in saved)
    expected = F.log_softmax(h @ w.T, -1).gather(1, labels[:, None])[:, 0]
    torch.testing.assert_close(result.log_probs, expected, atol=5e-4, rtol=1e-5)
    (result.log_probs.sum() + result.entropy.sum()).backward()
    assert torch.isfinite(h.grad).all() and torch.isfinite(w.grad).all()


@pytest.mark.parametrize("reduction", ["token_mean", "response_mean"])
@pytest.mark.parametrize("stopgrad", [True, False])
@pytest.mark.parametrize("alpha", [0.0, 1.5])
def test_recurrent_parameters_credit_and_partition_invariance(reduction, stopgrad, alpha):
    torch.manual_seed(23)
    base = [torch.randn(*shape, dtype=torch.float64) / 3 for shape in [(9, 4), (4, 4), (13, 4), (9, 3)]]
    labels = torch.tensor([0, 1, 3, 4, 12, 8, 7, 9, 2])
    advantage = torch.tensor([1.0, 1.0, -1.0, -1.0, -1.0, 0.5, 0.5, 0.5, 0.5], dtype=torch.float64)
    ref_h, ref_w = torch.randn(9, 4, dtype=torch.float64), torch.randn(13, 4, dtype=torch.float64)
    normalization = token_weights([2, 0, 3, 4], reduction=reduction, like=base[0])
    outcomes = []
    for tiled, boundaries in [(False, [0, 9]), (True, [0, 9]), (True, [0, 2, 2, 5, 9])]:
        h, core, head, gates = [value.clone().requires_grad_() for value in base]
        total = h.sum() * 0
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            state = h[start:end]
            scores = []
            for _ in range(3):
                state = torch.tanh(state @ core)
                if tiled:
                    result = streamed_readout(
                        state,
                        head,
                        labels[start:end],
                        vocab_tile=4,
                        entropy=True,
                        reference_hidden=ref_h[start:end],
                        reference_weight=ref_w,
                    )
                    score, full_kl = result.log_probs, result.full_kl
                else:
                    score, _, full_kl = dense(state, head, labels[start:end], 1, ref_h[start:end], ref_w)
                scores.append(score)
            # Trainable gates test the logp * grad(credit) term independently of the head.
            credit = (gates[start:end] + loop_weights(3, alpha=alpha, like=h).log()).softmax(-1)
            reference = F.log_softmax(ref_h[start:end] @ ref_w.T, -1).gather(1, labels[start:end, None])[:, 0]
            total = total + rltt_loss(
                torch.stack(scores, -1),
                advantage[start:end],
                credit,
                normalization[start:end],
                reference_log_probs=reference,
                kl_coefficient=0.1,
                credit_stopgrad=stopgrad,
                terminal_full_kl=full_kl,
            )
        gradients = torch.autograd.grad(total, (h, core, head, gates), allow_unused=True)
        outcomes.append((total, gradients))
    for value, gradients in outcomes[1:]:
        torch.testing.assert_close(value, outcomes[0][0], atol=1e-11, rtol=1e-11)
        for actual, expected in zip(gradients, outcomes[0][1], strict=True):
            if expected is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)


def test_sampled_k3_terminal_contract_and_mixture_distinction():
    scores = torch.tensor([[-1.0, -3.0], [-2.0, -1.0]], dtype=torch.float64, requires_grad=True)
    ref = torch.tensor([-4.0, -2.0], dtype=torch.float64)
    credit = torch.tensor([0.25, 0.75], dtype=torch.float64)
    advantage = torch.tensor([1.0, -1.0], dtype=torch.float64)
    weight = torch.tensor([0.5, 0.5], dtype=torch.float64)
    loss = rltt_loss(
        scores, advantage, credit, weight, reference_log_probs=ref, kl_coefficient=0.3, credit_stopgrad=True
    )
    expected = (
        -advantage * (scores * credit).sum(-1) + 0.3 * ((ref - scores[:, -1]).exp() - (ref - scores[:, -1]) - 1)
    ).mean()
    torch.testing.assert_close(loss, expected)
    assert not torch.allclose((scores * credit).sum(-1), (scores.exp() * credit).sum(-1).log())


def test_reject_trainable_reference():
    with pytest.raises(ValueError, match="frozen"):
        streamed_readout(
            torch.zeros(2, 3),
            torch.zeros(5, 3),
            torch.tensor([0, 1]),
            vocab_tile=2,
            reference_hidden=torch.zeros(2, 3, requires_grad=True),
            reference_weight=torch.zeros(5, 3),
        )


def test_zero_kl_coefficient_does_not_evaluate_exponential():
    scores = torch.tensor([[-1000.0, -1000.0]], requires_grad=True)
    loss = rltt_loss(
        scores,
        torch.ones(1),
        torch.tensor([0.5, 0.5]),
        torch.ones(1),
        reference_log_probs=torch.zeros(1),
        kl_coefficient=0,
        credit_stopgrad=True,
    )
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(1000.0))
    torch.testing.assert_close(scores.grad, torch.tensor([[-0.5, -0.5]]))
