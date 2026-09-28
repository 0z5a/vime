"""Validation-only accounting around the real actor and backuper operations."""

import json
import os
import time
from pathlib import Path

import torch.distributed as dist
from vime.backends.megatron_utils.actor import MegatronTrainRayActor
from vime.backends.megatron_utils.update_weight import hf_weight_iterator_direct
from vime.utils.tensor_backper import TensorBackuper


def record(values: dict) -> None:
    path = Path(os.environ['REWARD_RUN_DIR']) / f'copy-ledger-rank{dist.get_rank()}.jsonl'
    with path.open('a') as output:
        output.write(json.dumps(values) + '\n')


class MeasuredActor(MegatronTrainRayActor):
    def init(self, args, role, with_ref=False, with_opd_teacher=False):
        create_original = TensorBackuper.create
        materialize_original = hf_weight_iterator_direct._get_megatron_full_params

        def materialize(infos, weights, broadcast_buffer_size, ep_broadcast_src_rank_map):
            cpu_bytes = sum(info.size for info in infos if info.src_rank == dist.get_rank()
                            and weights[info.name].device.type == 'cpu')
            result = materialize_original(infos, weights, broadcast_buffer_size, ep_broadcast_src_rank_map)
            record({'event': 'publication_staging_h2d', 'logical_copy_bytes': cpu_bytes})
            return result

        hf_weight_iterator_direct._get_megatron_full_params = materialize

        def create(source_getter, single_tag):
            selected = None if os.environ['RESIDENT_FORCE_SNAPSHOT'] == '1' else single_tag
            backuper = create_original(source_getter, selected)
            payload = sum(tensor.numel() * tensor.element_size() for _, tensor in source_getter())
            record({'event': 'selection', 'single_tag': selected, 'logical_parameter_bytes': payload})

            def measure(operation, method, copied_bytes):
                def wrapped(tag):
                    started = time.perf_counter()
                    result = method(tag)
                    seconds = time.perf_counter() - started
                    record({'event': operation, 'tag': tag, 'seconds': seconds,
                            'logical_copy_bytes': copied_bytes})
                    return result
                return wrapped

            copied = payload if selected is None else 0
            backuper.backup = measure('backup_d2h', backuper.backup, copied)
            backuper.restore = measure('restore_h2d', backuper.restore, copied)
            backuper.get = measure('get_publication_view', backuper.get, 0)
            return backuper

        TensorBackuper.create = staticmethod(create)
        return super().init(args, role, with_ref, with_opd_teacher)

    def update_weights(self):
        record({'event': 'publication_start'})
        started = time.perf_counter()
        result = super().update_weights()
        record({'event': 'publication_complete', 'seconds': time.perf_counter() - started})
        return result
