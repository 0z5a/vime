"""Official-tokenizer prompt boundary audit with an explicit answer-replay transport."""

import argparse
import copy
import hashlib
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.download_math_data import SOURCES, Source, verify
from vime.rollout import vllm_rlt_rollout
from vime.utils.data import Dataset
from vime.utils.types import Sample


def audit(data: Path, checkpoint: Path) -> dict:
    source: Source = json.loads(SOURCES.with_name("ouro_tokenizer.json").read_text())[0]
    for item in source["files"]:
        verify((checkpoint / item["path"]).read_bytes(), item)
    tokenizer = vllm_rlt_rollout._tokenizer(str(checkpoint))
    records = []
    for thinking in (False, True):
        dataset = Dataset(
            str(data),
            tokenizer,
            None,
            None,
            prompt_key="prompt",
            label_key="label",
            apply_chat_template=True,
            apply_chat_template_kwargs={"enable_thinking": thinking},
        )
        expected = {
            sample.metadata["id"]: tokenizer.encode(sample.prompt, add_special_tokens=False)
            for sample in dataset.samples
        }

        def replay(samples: list[Sample], rollout_id: int, expected=expected, thinking=thinking) -> list[Sample]:
            assert rollout_id == 0
            for sample in samples:
                identity = sample.metadata["id"]
                assert sample.tokens == expected[identity], identity
                records.append(
                    {
                        "id": identity,
                        "thinking": thinking,
                        "completion": sample.index % 2,
                        "prompt_tokens": len(sample.tokens),
                        "token_sha256": hashlib.sha256(json.dumps(sample.tokens).encode()).hexdigest(),
                    }
                )
                answer = tokenizer.encode(rf"\boxed{{{sample.label}}}", add_special_tokens=False)
                sample.tokens = sample.tokens + answer
                sample.response_length = len(answer)
            return samples

        args = Namespace(
            hf_checkpoint=str(checkpoint),
            n_samples_per_prompt=2,
            apply_chat_template=True,
            custom_rm_path=None,
            rm_type="math",
            rlt_engine=SimpleNamespace(generate=SimpleNamespace(remote=replay)),
        )
        with patch.object(vllm_rlt_rollout.ray, "get", lambda result: result):
            for start in range(0, len(dataset), 16):
                groups = [[copy.deepcopy(sample) for _ in range(2)] for sample in dataset.samples[start : start + 16]]
                for group_index, group in enumerate(groups, start=start):
                    for completion, sample in enumerate(group):
                        sample.group_index, sample.index = group_index, 2 * group_index + completion
                args.rollout_batch_size = len(groups)
                result = vllm_rlt_rollout.generate_rollout(
                    args, 0, SimpleNamespace(get_samples=lambda count, groups=groups: groups)
                )
                assert all(sample.reward == 1 for group in result.samples for sample in group)
        print(f"thinking={thinking}: {len(dataset)} prompts, {2 * len(dataset)} replayed completions", flush=True)
    return {
        "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "tokenizer_repository": source["repository"],
        "tokenizer_revision": source["revision"],
        "tokenizer_class": type(tokenizer).__name__,
        "rows": records,
        "exact_prompt_token_matches": len(records),
        "known_answer_rewards_one": len(records),
        "ray_transport": "explicit synchronous test seam",
        "engine": "canonical answer token replay",
        "model_generation": False,
        "training": False,
        "quality_measurement": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data, args.tokenizer)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
