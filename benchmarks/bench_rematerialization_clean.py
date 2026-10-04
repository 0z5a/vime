"""Fresh unmodified-parent comparison of plans frozen from the separate pilot."""

import argparse
import copy
import json
import sys
import time
from dataclasses import replace
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

    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for family in ("ouro", "nanbeige", "huginn"):
            model, depth = make_actor(family, True)
            initial = copy.deepcopy(model.state_dict())
            generator = torch.Generator().manual_seed(args.seed)
            items = [
                (torch.randint(0, 11, (16 + 3 * i,), generator=generator), 8, trace(family, depth, 91 + i, 8))
                for i in range(4)
            ]
            inputs = batched(items)
            reference_model = copy.deepcopy(model).requires_grad_(False)
            with torch.no_grad():
                reference_model.lm_head.weight.mul_(0.95)
                reference = reference_model(**inputs)[:, -2]
            if args.arm == "remat":
                from vime_plugins.looped.execution import RematPlan

                plan = RematPlan(2, 2 if family == "huginn" else 0, 0)
                inputs["readout"] = replace(inputs["readout"], rematerialization=plan)
            advantages = torch.tensor([-1.0, 0.3, 0.7, -0.2]).repeat_interleave(8)
            weights = torch.full_like(advantages, 1 / 32)
            old = torch.cat([initial[name].flatten() for name, p in model.named_parameters() if p.requires_grad])
            for _ in range(2):
                model.load_state_dict(initial)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
                start = time.perf_counter_ns()
                optimizer.zero_grad(set_to_none=True)
                output = model(**inputs)
                credit = loop_weights(depth, alpha=1.5, like=output)
                loss = (
                    rltt_loss(
                        output[:, :-1],
                        advantages,
                        credit,
                        weights,
                        reference_log_probs=reference,
                        kl_coefficient=0.01,
                        credit_stopgrad=True,
                    )
                    - 0.003 * output[:, -1].mean()
                )
                loss.backward()
                optimizer.step()
                seconds = (time.perf_counter_ns() - start) / 1e9
            gradient = torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad])
            updated = torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])
            oracle_path = args.output.parent / f"oracle-{args.seed}-{family}.json"
            if args.order == 0:
                assert args.arm == "parent"
                with oracle_path.open("x") as oracle_file:
                    json.dump(
                        {"loss": loss.item(), "gradient": gradient.tolist(), "updated": updated.tolist()}, oracle_file
                    )
            oracle = json.loads(oracle_path.read_text())
            wanted_gradient = torch.tensor(oracle["gradient"])
            torch.testing.assert_close(loss, torch.tensor(oracle["loss"]), atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(gradient, wanted_gradient, atol=5e-6, rtol=1e-4)
            relative = ((gradient - wanted_gradient).double().norm() / wanted_gradient.double().norm()).item()
            assert relative < 1e-5
            torch.testing.assert_close(updated - old, torch.tensor(oracle["updated"]) - old, atol=2e-6, rtol=1e-4)
            assert (updated - old).norm() > 0
            stream.write(
                json.dumps(
                    {
                        "family": family,
                        "seed": args.seed,
                        "order": args.order,
                        "arm": args.arm,
                        "source_root": str(source),
                        "actor_update_seconds": seconds,
                        "gradient_relative_l2": relative,
                        "parameter_delta": (updated - old).double().norm().item(),
                        "loss": loss.item(),
                    }
                )
                + "\n"
            )
            stream.flush()
            print(f"seed={args.seed} order={args.order} {args.arm} {family} exact contract", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--order", required=True, type=int)
    parser.add_argument("--arm", choices=("parent", "remat"), required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    main(parser.parse_args())
