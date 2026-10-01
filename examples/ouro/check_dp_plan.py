"""Run with torchrun --standalone --nproc_per_node=2 on CPU."""

import pytest
import torch.distributed as dist

from vime_plugins.ouro.budget import ExecutionPlan, synchronize_plan


def main():
    dist.init_process_group("gloo")
    assert dist.get_world_size() == 2
    for loops in (2, 3, 4):
        synchronize_plan(ExecutionPlan(loops, 7))
    with pytest.raises(ValueError, match="DP ranks disagree"):
        synchronize_plan(ExecutionPlan(2 + dist.get_rank(), 7))
    dist.barrier()
    if dist.get_rank() == 0:
        print("PASS: global K=2/3/4 and all-rank mismatch rejection", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
