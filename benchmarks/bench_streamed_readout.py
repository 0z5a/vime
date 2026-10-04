"""Readout forward/backward microbenchmark; excludes recurrent model and RL work."""

import argparse
import json
import platform
import time
from functools import partial

import torch
import torch.nn.functional as F

from vime_plugins.looped.readout import streamed_readout


def readout_loss(arm, *, inputs, full, tile):
    hidden, weight, targets, reference_hidden, reference_weight = inputs
    if arm == "streamed":
        result = streamed_readout(
            hidden,
            weight,
            targets,
            vocab_tile=tile,
            entropy=full,
            reference_hidden=reference_hidden if full else None,
            reference_weight=reference_weight if full else None,
        )
        if not full:
            return result.log_probs.mean()
        return result.log_probs.mean() + 0.01 * result.entropy.mean() + 0.1 * result.full_kl.mean()
    log_p = F.log_softmax(hidden @ weight.T, -1)
    selected = log_p.gather(1, targets[:, None]).mean()
    if not full:
        return selected
    log_q = F.log_softmax(reference_hidden @ reference_weight.T, -1)
    probability = log_p.exp()
    return (
        selected - 0.01 * (probability * log_p).sum(-1).mean() + 0.1 * (probability * (log_p - log_q)).sum(-1).mean()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    for rows, vocabulary, width, tile in [(64, 1024, 128, 256), (256, 4096, 256, 512)]:
        hidden = torch.randn(rows, width).requires_grad_()
        weight = (torch.randn(vocabulary, width) / width**0.5).requires_grad_()
        targets = torch.randint(vocabulary, (rows,))
        reference_hidden, reference_weight = torch.randn_like(hidden), torch.randn_like(weight) / width**0.5
        inputs = (hidden, weight, targets, reference_hidden, reference_weight)
        input_storages = {value.untyped_storage().data_ptr() for value in inputs}
        for full in (False, True):
            loss = partial(readout_loss, inputs=inputs, full=full, tile=tile)

            oracle = None
            for index, arm in enumerate(["dense", "streamed"] + ["dense", "streamed", "streamed", "dense"] * 3):
                hidden.grad = weight.grad = None
                start = time.perf_counter()
                value = loss(arm)
                value.backward()
                elapsed = time.perf_counter() - start
                observed = (value.detach(), hidden.grad.detach().clone(), weight.grad.detach().clone())
                if oracle is None:
                    oracle = observed
                for actual, expected in zip(observed, oracle, strict=True):
                    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
                if index >= 2:
                    print(
                        json.dumps(
                            dict(
                                seed=args.seed,
                                rows=rows,
                                vocabulary=vocabulary,
                                width=width,
                                tile=tile,
                                full=full,
                                arm=arm,
                                seconds=elapsed,
                                torch=torch.__version__,
                                platform=platform.platform(),
                            )
                        ),
                        flush=True,
                    )
            for arm in ("dense", "streamed"):
                retained = {}

                def pack(tensor, retained=retained, input_storages=input_storages):
                    storage = tensor.untyped_storage()
                    if storage.data_ptr() not in input_storages:
                        retained[storage.data_ptr()] = storage.nbytes()
                    return tensor

                with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                    value = loss(arm)
                print(
                    json.dumps(
                        dict(
                            seed=args.seed,
                            rows=rows,
                            vocabulary=vocabulary,
                            width=width,
                            tile=tile,
                            full=full,
                            arm=arm,
                            saved_intermediate_bytes=sum(retained.values()),
                        )
                    ),
                    flush=True,
                )
                del value


if __name__ == "__main__":
    main()
