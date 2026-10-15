"""Expose the FSDP2 trainer as pure data parallelism to the shared training code.

The loss, advantage and metric helpers shared with the Megatron backend query
``megatron.core.mpu`` for their process groups. An FSDP2 trainer shards over
every rank and has no tensor, pipeline or context parallelism, so it registers
exactly that layout (TP = PP = CP = 1, DP = world) instead of re-implementing
those helpers.
"""


def initialize_data_parallel_state(args) -> None:
    from megatron.core import mpu

    if mpu.model_parallel_is_initialized():
        return
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        distributed_timeout_minutes=args.distributed_timeout_minutes,
        create_gloo_process_groups=True,
    )
