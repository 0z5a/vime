"""Fresh-process comparison with an untouched parent and retained gradient oracle."""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import vime


def main(args):
    source = Path(vime.__file__).resolve().parents[1]
    assert source == args.source.resolve(), source
    sys.path.insert(0, str(source / "tests/plugins"))
    from test_looped_response_readout import make_actor
    from test_packed_recurrent_replay import batched
    from test_rltt_training import trace
    from vime_plugins.looped.rltt import loop_weights, rltt_loss

    if args.arm == "shared-prefix":
        from test_differentiable_prefix import replay

    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for family in ("ouro", "nanbeige", "huginn"):
            for prompt_length, response_length, group_size in ((4, 16, 1), (4, 16, 32), (24, 4, 32)):
                actor, depth = make_actor(family, False)
                initial = copy.deepcopy(actor.state_dict())
                reference = copy.deepcopy(actor).requires_grad_(False)
                generator = torch.Generator().manual_seed(args.seed)
                prompt = torch.randint(0, 13, (prompt_length,), generator=generator)
                responses = tuple(
                    torch.randint(0, 13, (response_length,), generator=generator) for _ in range(group_size)
                )
                items = [
                    (torch.cat((prompt, response)), response_length, trace(family, depth, 91, response_length))
                    for response in responses
                ]
                groups = tuple(tuple(range(begin, min(begin + 4, group_size))) for begin in range(0, group_size, 4))
                inputs = [batched([items[i] for i in group]) for group in groups]
                advantages = tuple(torch.randn(response_length, generator=generator) for _ in responses)
                with torch.no_grad():
                    reference.lm_head.weight.mul_(0.95)
                    refs = reference(**batched(items))[:, -2].split([response_length] * group_size)

                def objective(
                    scores, indices, advantages=advantages, depth=depth, refs=refs, total=group_size * response_length
                ):
                    advantage = torch.cat([advantages[i] for i in indices])
                    credit = loop_weights(depth, alpha=1.5, like=scores)
                    return (
                        rltt_loss(
                            scores[:, :-1],
                            advantage,
                            credit,
                            torch.full_like(advantage, 1 / total),
                            reference_log_probs=torch.cat([refs[i] for i in indices]),
                            kl_coefficient=0.01,
                            credit_stopgrad=True,
                        )
                        - 0.003 * scores[:, -1].sum() / total
                    )

                old = torch.cat([initial[name].flatten() for name, p in actor.named_parameters() if p.requires_grad])
                for _ in range(2):
                    actor.load_state_dict(initial)
                    optimizer = torch.optim.SGD(actor.parameters(), lr=0.03)
                    started = time.perf_counter_ns()
                    optimizer.zero_grad(set_to_none=True)
                    if args.arm == "shared-prefix":
                        state, identity = replay(actor, prompt, depth, True, 91 if family == "huginn" else None)
                        loss = state.backward_suffixes(responses, groups, objective, identity=identity)
                    else:
                        loss = actor.lm_head.weight.new_zeros(())
                        for indices, inputs_for_wave in zip(groups, inputs, strict=True):
                            value = objective(actor(**inputs_for_wave), indices)
                            value.backward()
                            loss = loss + value.detach()
                    optimizer.step()
                    seconds = (time.perf_counter_ns() - started) / 1e9
                gradient = torch.cat([p.grad.flatten() for p in actor.parameters() if p.requires_grad])
                updated = torch.cat([p.detach().flatten() for p in actor.parameters() if p.requires_grad])
                oracle_path = (
                    args.output.parent
                    / f"oracle-{args.seed}-{family}-{prompt_length}-{response_length}-{group_size}.json"
                )
                if args.order == 0:
                    assert args.arm == "parent"
                    with oracle_path.open("x") as oracle_file:
                        json.dump(
                            {"loss": loss.item(), "gradient": gradient.tolist(), "updated": updated.tolist()},
                            oracle_file,
                        )
                oracle = json.loads(oracle_path.read_text())
                expected_gradient = torch.tensor(oracle["gradient"])
                torch.testing.assert_close(loss, torch.tensor(oracle["loss"]), atol=2e-6, rtol=2e-6)
                torch.testing.assert_close(gradient, expected_gradient, atol=5e-6, rtol=1e-4)
                relative = ((gradient - expected_gradient).double().norm() / expected_gradient.double().norm()).item()
                assert relative < 1e-5
                torch.testing.assert_close(updated - old, torch.tensor(oracle["updated"]) - old, atol=2e-6, rtol=1e-4)
                delta = (updated - old).double().norm().item()
                assert delta > 0
                stream.write(
                    json.dumps(
                        {
                            "family": family,
                            "P": prompt_length,
                            "D": response_length,
                            "G": group_size,
                            "seed": args.seed,
                            "order": args.order,
                            "arm": args.arm,
                            "source_root": str(source),
                            "actor_update_seconds": seconds,
                            "gradient_relative_l2": relative,
                            "parameter_delta": delta,
                            "loss": loss.item(),
                        }
                    )
                    + "\n"
                )
                stream.flush()
                print(
                    f"PASS seed{args.seed} order{args.order} {args.arm} {family} P{prompt_length} D{response_length} G{group_size}",
                    flush=True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--order", type=int, required=True)
    parser.add_argument("--arm", choices=("parent", "shared-prefix"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
