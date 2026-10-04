"""Tiny fixed-trace RLTT actor-update timing and retained-storage observations."""

import argparse
import copy
import json
import statistics
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests/plugins"))
from test_looped_response_readout import make_actor
from test_packed_recurrent_replay import batched
from test_rltt_training import trace

from vime_plugins.looped.planner import ScheduleProfile, candidate_plans, select_plan
from vime_plugins.looped.rltt import loop_weights, rltt_loss


def loss_value(output, advantages, reference):
    weights = torch.full_like(advantages, 1 / advantages.numel())
    credit = loop_weights(output.shape[1] - 1, alpha=1.5, like=output)
    return (
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


def execute(model, inputs, initial, advantages, reference):
    model.load_state_dict(initial)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    start = time.perf_counter_ns()
    optimizer.zero_grad(set_to_none=True)
    loss = loss_value(model(**inputs), advantages, reference)
    loss.backward()
    optimizer.step()
    seconds = (time.perf_counter_ns() - start) / 1e9
    gradients = torch.cat([parameter.grad.flatten() for parameter in model.parameters() if parameter.requires_grad])
    updated = torch.cat([parameter.detach().flatten() for parameter in model.parameters() if parameter.requires_grad])
    return seconds, loss.detach(), gradients, updated


def observed_storage(model, inputs, initial, advantages, reference):
    model.load_state_dict(initial)
    excluded = {tensor.untyped_storage().data_ptr() for tensor in (*model.parameters(), *model.buffers())}
    observed = {}

    def pack(tensor):
        storage = tensor.untyped_storage()
        if storage.data_ptr() not in excluded:
            observed[storage.data_ptr()] = storage.nbytes()
        return tensor

    model.zero_grad(set_to_none=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = loss_value(model(**inputs), advantages, reference)
    # Unique storage observed by forward save hooks, not allocator or RSS peak.
    size = sum(observed.values())
    loss.backward()
    return size


def main(args):
    torch.set_num_threads(1)
    plans = candidate_plans((1, 2), (0, 2), (0, 8))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    choices = []
    with args.output.open("x") as stream:
        for family in ("ouro", "nanbeige", "huginn"):
            baseline, depth = make_actor(family, True)
            candidate = copy.deepcopy(baseline)
            initial = copy.deepcopy(baseline.state_dict())
            generator = torch.Generator().manual_seed(args.seed)
            items = [
                (torch.randint(0, 11, (16 + 3 * i,), generator=generator), 8, trace(family, depth, 91 + i, 8))
                for i in range(4)
            ]
            inputs = batched(items)
            advantages = torch.tensor([-1.0, 0.3, 0.7, -0.2]).repeat_interleave(8)
            reference_actor = copy.deepcopy(baseline).requires_grad_(False)
            with torch.no_grad():
                reference_actor.lm_head.weight.mul_(0.95)
                reference = reference_actor(**inputs)[:, -2]
            _, expected_loss, expected_grad, expected_update = execute(
                baseline, inputs, initial, advantages, reference
            )
            old = torch.cat([initial[name].flatten() for name, p in baseline.named_parameters() if p.requires_grad])
            assert (expected_update - old).norm() > 0
            baseline_storage = observed_storage(baseline, inputs, initial, advantages, reference)
            profiles = []
            contract = f"tiny-{family}-model81-data{args.seed}-B4-FP32-CPU-RLTT"
            for plan in plans:
                candidate_inputs = {**inputs, "readout": replace(inputs["readout"], rematerialization=plan)}
                candidate_storage = observed_storage(candidate, candidate_inputs, initial, advantages, reference)
                execute(candidate, candidate_inputs, initial, advantages, reference)
                plan_times = []
                for block in range(args.blocks):
                    for order, arm in enumerate(("layer-baseline", "remat", "remat", "layer-baseline")):
                        model, batch, size = (
                            (baseline, inputs, baseline_storage)
                            if arm == "layer-baseline"
                            else (candidate, candidate_inputs, candidate_storage)
                        )
                        seconds, loss, gradient, updated = execute(model, batch, initial, advantages, reference)
                        torch.testing.assert_close(loss, expected_loss, atol=2e-6, rtol=2e-6)
                        torch.testing.assert_close(gradient, expected_grad, atol=5e-6, rtol=1e-4)
                        relative = ((gradient - expected_grad).double().norm() / expected_grad.double().norm()).item()
                        assert relative < 1e-5
                        torch.testing.assert_close(updated - old, expected_update - old, atol=2e-6, rtol=1e-4)
                        if arm == "remat":
                            plan_times.append(seconds)
                        stream.write(
                            json.dumps(
                                {
                                    "seed": args.seed,
                                    "family": family,
                                    "plan": asdict(plan),
                                    "block": block,
                                    "order": order,
                                    "arm": arm,
                                    "actor_update_seconds": seconds,
                                    "forward_observed_saved_storage_bytes": size,
                                    "gradient_relative_l2": relative,
                                    "gradient_max_abs": (gradient - expected_grad).abs().max().item(),
                                    "loss": loss.item(),
                                    "parameter_delta": (updated - old).double().norm().item(),
                                    "scope": "CPU fixed-trace RLTT forward/backward/SGD; no rollout, reward, reference timing, publication or checkpoint",
                                }
                            )
                            + "\n"
                        )
                        stream.flush()
                profiles.append(
                    ScheduleProfile(
                        plan,
                        contract,
                        "actor_update",
                        "forward_observed_saved_storage",
                        statistics.geometric_mean(plan_times),
                        candidate_storage,
                        True,
                    )
                )
            # This CPU proxy exercises selection; it is not a GPU capacity decision.
            budget = min(profile.memory_bytes for profile in profiles)
            selected = select_plan(
                tuple(profiles),
                contract=contract,
                scope="actor_update",
                memory_metric="forward_observed_saved_storage",
                budget_bytes=budget,
            )
            choices.append(
                {
                    "contract": contract,
                    "memory_metric": "forward_observed_saved_storage",
                    "budget_bytes": budget,
                    "selected": asdict(selected),
                    "profiles": [asdict(profile) for profile in profiles],
                }
            )
            print(f"seed={args.seed} family={family} all gradients/updates matched; selected={selected}", flush=True)
    args.output.with_suffix(".profiles.json").write_text(json.dumps(choices, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    main(parser.parse_args())
