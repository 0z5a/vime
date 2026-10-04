"""CPU tiny-model replay timing; not CUDA varlen or a complete RL step."""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests/plugins"))
from test_looped_response_readout import make_actor
from test_packed_recurrent_replay import batched, objective
from test_rltt_training import packed, trace


def execute(model, inputs):
    model.zero_grad(set_to_none=True)
    start = time.perf_counter_ns()
    loss = objective(model(**inputs))
    loss.backward()
    elapsed = (time.perf_counter_ns() - start) / 1e6
    gradients = torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad])
    return elapsed, loss.detach(), gradients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with args.output.open("w") as stream:
        for family in ("ouro", "nanbeige", "huginn"):
            for count in (1, 4, 16):
                serial, depth = make_actor(family, False)
                candidate = copy.deepcopy(serial)
                generator = torch.Generator().manual_seed(args.seed)
                items = []
                for i in range(count):
                    length = 16 + 2 * (i % 8)
                    response = length // 2
                    tokens = torch.randint(0, 11, (length,), generator=generator)
                    items.append((tokens, response, trace(family, depth, 100 + i, response)))
                inputs = {"serial": packed(items, all_loops=True), "packed": batched(items)}
                models = {"serial": serial, "packed": candidate}
                _, expected, gradient = execute(serial, inputs["serial"])
                execute(candidate, inputs["packed"])
                for block in range(3):
                    for arm in ("serial", "packed", "packed", "serial"):
                        elapsed, loss, actual = execute(models[arm], inputs[arm])
                        torch.testing.assert_close(loss, expected, atol=2e-6, rtol=2e-6)
                        relative = ((actual - gradient).double().norm() / gradient.double().norm()).item()
                        torch.testing.assert_close(actual, gradient, atol=5e-6, rtol=1e-4)
                        assert relative < 1e-5, (family, count, arm, relative)
                        stream.write(
                            json.dumps(
                                {
                                    "process_seed": args.seed,
                                    "family": family,
                                    "batch": count,
                                    "sequence_lengths": [len(item[0]) for item in items],
                                    "response_tokens": sum(item[1] for item in items),
                                    "loops": depth,
                                    "parameters": sum(p.numel() for p in serial.parameters()),
                                    "arm": arm,
                                    "block": block,
                                    "forward_backward_ms": elapsed,
                                    "loss": loss.item(),
                                "gradient_relative_l2": relative,
                                "gradient_max_abs": (actual - gradient).abs().max().item(),
                                    "scope": "CPU tiny random models, packed projections with segmented SDPA; no optimizer, Ray, CUDA varlen or RL timing",
                                }
                            )
                            + "\n"
                        )
                        stream.flush()


if __name__ == "__main__":
    main()
