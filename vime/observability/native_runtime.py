"""Opt-in native worker identity, loaded implementations and optimizer contract."""

import hashlib
import inspect
import json
import marshal
import os
import platform
import sys
import time
from pathlib import Path

import torch


def source_hash(path: Path) -> str:
    if sys.platform == "darwin" and path.stat().st_flags & 0x40000000:
        raise OSError(f"Source is not locally resident: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation(value, function) -> dict:
    cls = type(value)
    function = inspect.unwrap(function)
    paths = {Path(inspect.getfile(cls)), Path(function.__code__.co_filename)}
    files = {}
    for path in sorted(paths):
        files[str(path.resolve())] = source_hash(path)
    return {
        "class": f"{cls.__module__}.{cls.__qualname__}",
        "source_sha256": files,
        "method": function.__name__,
        "code_sha256": hashlib.sha256(marshal.dumps(function.__code__)).hexdigest(),
    }


def model_identity(model: torch.nn.Module) -> dict:
    return {
        "implementation": implementation(model, model.forward),
        "state": {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in model.state_dict().items()
        },
    }


def optimizer_identity(components) -> list[dict]:
    records = []
    for component in components:
        config, optimizer = component.config, component.optimizer
        records.append(
            {
                "wrapper": f"{type(component).__module__}.{type(component).__qualname__}",
                "implementation": implementation(optimizer, optimizer.step),
                "effective_config": {
                    "optimizer": config.optimizer,
                    "decoupled_weight_decay": config.decoupled_weight_decay,
                    "adam_beta1": config.adam_beta1,
                    "adam_beta2": config.adam_beta2,
                    "adam_eps": config.adam_eps,
                    "lr": config.lr,
                    "weight_decay": config.weight_decay,
                    "clip_grad": config.clip_grad,
                    "fp16": config.fp16,
                    "bf16": config.bf16,
                    "use_distributed_optimizer": config.use_distributed_optimizer,
                },
                "groups": [
                    {
                        "settings": {
                            key: group[key]
                            for key in (
                                "lr",
                                "weight_decay",
                                "betas",
                                "eps",
                                "decoupled_weight_decay",
                                "adam_w_mode",
                                "bias_correction",
                                "amsgrad",
                                "maximize",
                                "foreach",
                                "fused",
                            )
                            if key in group
                        },
                        "parameters": len(group["params"]),
                        "numel": sum(parameter.numel() for parameter in group["params"]),
                    }
                    for group in optimizer.param_groups
                ],
            }
        )
    return records


def learner_execution(args) -> dict[str, str | int | bool]:
    return {
        "actor_schedule": "prefix" if args.loss_type == "rltt_loss" and args.rltt_prefix_wave_size > 0 else "mcore",
        "prefix_wave_size": args.rltt_prefix_wave_size,
        "recompute": args.recompute_granularity is not None,
        "loop_checkpoint": args.rltt_loop_checkpoint,
        "layer_checkpoint": args.rltt_layer_checkpoint,
        "token_chunk": args.rltt_token_chunk,
    }


def record_step(args, rollout_id, step_id, schedule, generation_before, generation_after, samples, grad_norm) -> Path:
    """A completed learner dispatch/update, bound to its startup process record."""
    proc = Path("/proc/self/stat")
    record = {
        "pid": os.getpid(),
        "process_start_ticks": int(proc.read_text().rsplit(")", 1)[1].split()[19]) if proc.exists() else None,
        "rank": args.rank,
        "rollout_id": rollout_id,
        "step_id": step_id,
        "schedule_completed": schedule,
        "actor_generation_before": generation_before,
        "actor_generation_after": generation_after,
        "logical_samples": samples,
        "gradient_norm": float(grad_norm),
        "learner_execution": learner_execution(args),
    }
    folder = Path(args.rlt_runtime_report_dir) / "steps"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"rollout{rollout_id}-step{step_id}-rank{args.rank}.json"
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def write_worker_report(directory: str, role: str, rank: int, details: dict) -> Path:
    import ray

    context = ray.get_runtime_context()
    identity = {
        "job": context.get_job_id(),
        "node": context.get_node_id(),
        "worker": context.get_worker_id(),
        "actor": context.get_actor_id(),
    }
    if not all(identity.values()):
        raise RuntimeError("Native runtime reports must be written inside an actual Ray actor")
    proc = Path("/proc/self/stat")
    start_ticks = int(proc.read_text().rsplit(")", 1)[1].split()[19]) if proc.exists() else None
    files, modules = {}, {}
    for name, module in sorted(list(sys.modules.items())):
        if name.split(".", 1)[0] not in {"vime", "vime_plugins", "vllm_rlt", "megatron"} or module is None:
            continue
        origin = vars(module).get("__file__")
        if origin is not None:
            path = str(Path(origin).resolve())
            if path not in files:
                files[path] = source_hash(Path(path))
            modules[name] = path
    report = {
        "schema": "native-runtime-identity-v1",
        "role": role,
        "rank": rank,
        "ray": identity,
        "pid": os.getpid(),
        "process_start_ticks": start_ticks,
        "host": platform.node(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "ray_version": ray.__version__,
        "captured_ns": time.time_ns(),
        "loaded_modules": modules,
        "source_sha256": files,
        "details": details,
    }
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{role}-rank{rank}-{identity['worker']}-{report['captured_ns']}.json"
    with path.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def record_learner(args, role: str, models, optimizer, loaded_checkpoint_iteration: int) -> Path:
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    components = (
        []
        if optimizer is None
        else (optimizer.chained_optimizers if isinstance(optimizer, ChainedOptimizer) else [optimizer])
    )
    return write_worker_report(
        args.rlt_runtime_report_dir,
        role,
        args.rank,
        {
            "loaded_checkpoint_iteration": loaded_checkpoint_iteration,
            "models": [model_identity(model.module) for model in models],
            "optimizers": optimizer_identity(components),
            "declared_model_revision": args.rlt_model_revision,
            "declared_engine_revision": args.rlt_engine_revision,
            "model_path": str(Path(args.hf_checkpoint).resolve()),
            "load_path": str(Path(args.load).resolve()),
            "parameters_dtype": str(args.params_dtype),
            "learner_execution": learner_execution(args),
            "parallelism": {
                "tp": args.tensor_model_parallel_size,
                "pp": args.pipeline_model_parallel_size,
                "cp": args.context_parallel_size,
            },
        },
    )


def record_rollout(args, model, runtime_epoch: str) -> Path:
    return write_worker_report(
        args.rlt_runtime_report_dir,
        "rollout",
        0,
        {
            "runtime_epoch": runtime_epoch,
            "model": model_identity(model),
            "declared_model_revision": args.rlt_model_revision,
            "declared_engine_revision": args.rlt_engine_revision,
            "model_path": str(Path(args.hf_checkpoint).resolve()),
            "parameters_dtype": str(args.params_dtype),
            "depth": args.rlt_depth,
            "attention_backend": args.rlt_attention_backend,
            "cuda_graphs": args.rlt_cuda_graphs,
        },
    )
