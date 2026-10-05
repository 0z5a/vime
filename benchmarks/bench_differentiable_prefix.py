"""Matched CPU packed-wave versus shared-prefix fixed-trace RLTT updates."""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests/plugins"))
from test_differentiable_prefix import replay
from test_looped_response_readout import make_actor
from test_packed_recurrent_replay import batched
from test_rltt_training import trace

from vime_plugins.looped.rltt import loop_weights, rltt_loss


def loss_value(scores, reference, advantages, total):
    credit = loop_weights(scores.shape[1] - 1, alpha=1.5, like=scores)
    weights = torch.full_like(advantages, 1 / total)
    return (
        rltt_loss(
            scores[:, :-1],
            advantages,
            credit,
            weights,
            reference_log_probs=reference,
            kl_coefficient=0.01,
            credit_stopgrad=True,
        )
        - 0.003 * scores[:, -1].sum() / total
    )


def execute(actor, family, depth, initial, prompt, responses, inputs, refs, advantages, shared):
    actor.load_state_dict(initial)
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.03)
    groups = tuple(tuple(range(begin, min(begin + 4, len(responses)))) for begin in range(0, len(responses), 4))
    total = sum(map(len, responses))

    def objective(scores, indices):
        return loss_value(
            scores, torch.cat([refs[i] for i in indices]), torch.cat([advantages[i] for i in indices]), total
        )

    started = time.perf_counter_ns()
    optimizer.zero_grad(set_to_none=True)
    if shared:
        state, identity = replay(actor, prompt, depth, True, 91 if family == "huginn" else None)
        loss = state.backward_suffixes(responses, groups, objective, identity=identity)
    else:
        loss = actor.lm_head.weight.new_zeros(())
        for group, batch in zip(groups, inputs, strict=True):
            value = objective(actor(**batch), group)
            value.backward()
            loss = loss + value.detach()
    optimizer.step()
    seconds = (time.perf_counter_ns() - started) / 1e9
    gradient = torch.cat([p.grad.flatten() for p in actor.parameters() if p.requires_grad])
    updated = torch.cat([p.detach().flatten() for p in actor.parameters() if p.requires_grad])
    return seconds, loss, gradient, updated


def main(args):
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    workloads = []
    with args.output.open("x") as stream:
        for family in ("ouro", "nanbeige", "huginn"):
            for prompt_length, response_length in ((4, 16), (24, 4)):
                for group_size in (1, 8, 32):
                    baseline, depth = make_actor(family, False)
                    candidate = copy.deepcopy(baseline)
                    reference = copy.deepcopy(baseline).requires_grad_(False)
                    initial = copy.deepcopy(baseline.state_dict())
                    generator = torch.Generator().manual_seed(args.seed)
                    prompt = torch.randint(0, 13, (prompt_length,), generator=generator)
                    responses = tuple(
                        torch.randint(0, 13, (response_length,), generator=generator) for _ in range(group_size)
                    )
                    items = [
                        (torch.cat((prompt, response)), response_length, trace(family, depth, 91, response_length))
                        for response in responses
                    ]
                    inputs = [batched(items[begin : begin + 4]) for begin in range(0, group_size, 4)]
                    advantages = tuple(torch.randn(response_length, generator=generator) for _ in responses)
                    with torch.no_grad():
                        reference.lm_head.weight.mul_(0.95)
                        refs = reference(**batched(items))[:, -2].split([response_length] * group_size)
                    common = (family, depth, initial, prompt, responses, inputs, refs, advantages)
                    _, expected_loss, expected_gradient, expected_update = execute(baseline, *common, False)
                    old = torch.cat(
                        [initial[name].flatten() for name, p in baseline.named_parameters() if p.requires_grad]
                    )
                    execute(candidate, *common, True)
                    workloads.append(
                        {
                            "family": family,
                            "P": prompt_length,
                            "D": response_length,
                            "G": group_size,
                            "prompt": prompt.tolist(),
                            "responses": [r.tolist() for r in responses],
                            "advantages": [a.tolist() for a in advantages],
                            "reference_scores": [r.tolist() for r in refs],
                            "latent_seed": 91 if family == "huginn" else None,
                        }
                    )
                    for block in range(args.blocks):
                        for order, arm in enumerate(("packed-wave", "shared-prefix", "shared-prefix", "packed-wave")):
                            shared = arm == "shared-prefix"
                            seconds, loss, gradient, updated = execute(
                                candidate if shared else baseline, *common, shared
                            )
                            torch.testing.assert_close(loss, expected_loss, atol=2e-6, rtol=2e-6)
                            torch.testing.assert_close(gradient, expected_gradient, atol=5e-6, rtol=1e-4)
                            relative = (
                                (gradient - expected_gradient).double().norm() / expected_gradient.double().norm()
                            ).item()
                            assert relative < 1e-5
                            torch.testing.assert_close(updated - old, expected_update - old, atol=2e-6, rtol=1e-4)
                            delta = (updated - old).double().norm().item()
                            assert delta > 0
                            stream.write(
                                json.dumps(
                                    {
                                        "seed": args.seed,
                                        "family": family,
                                        "P": prompt_length,
                                        "D": response_length,
                                        "G": group_size,
                                        "microbatch": 4,
                                        "block": block,
                                        "order": order,
                                        "arm": arm,
                                        "actor_update_seconds": seconds,
                                        "loss": loss.item(),
                                        "gradient_relative_l2": relative,
                                        "gradient_max_abs": (gradient - expected_gradient).abs().max().item(),
                                        "parameter_delta": delta,
                                        "scope": "tiny FP32 CPU fixed-trace RLTT forward/backward/SGD; no online RL or memory claim",
                                    }
                                )
                                + "\n"
                            )
                            stream.flush()
                    print(
                        f"PASS seed{args.seed} {family} P{prompt_length} D{response_length} G{group_size}", flush=True
                    )
    args.output.with_suffix(".inputs.json").write_text(json.dumps(workloads, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
