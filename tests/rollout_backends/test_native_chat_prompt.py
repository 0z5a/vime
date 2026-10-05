"""Dataset-rendered prompts must reach native generation without a second template."""

import copy
import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from vime.rollout import vllm_rlt_rollout
from vime.utils.data import Dataset
from vime.utils.types import Sample


class Tokenizer:
    def __init__(self):
        self.render_count = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking=False, tools=None):
        assert isinstance(messages, list), "The dataset already rendered this prompt"
        assert not tokenize and add_generation_prompt
        assert tools is None
        self.render_count += 1
        return f"user:{messages[0]['content']}\nassistant:" + ("<think>" if enable_thinking else "")

    def encode(self, prompt, *, add_special_tokens):
        assert not add_special_tokens
        return list(prompt.encode())

    def decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens
        return bytes(tokens).decode()


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("source", ["rendered", "conversation", "plain"])
def test_prompt_rendered_once_and_preserved_through_reward(tmp_path, monkeypatch, thinking, source):
    tokenizer = Tokenizer()
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps({"prompt": "What is 1+1?", "label": "2"}) + "\n")
    dataset = Dataset(
        str(path),
        tokenizer,
        None,
        None,
        prompt_key="prompt",
        label_key="label",
        apply_chat_template=source == "rendered",
        apply_chat_template_kwargs={"enable_thinking": thinking},
    )
    sample = dataset[0]
    if source == "conversation":
        sample = Sample(prompt=[{"role": "user", "content": "What is 1+1?"}], label="2")
        # Structured samples carry their own template options.
        sample.apply_chat_template_kwargs = {"enable_thinking": thinking}
    expected_prompt = (
        "What is 1+1?" if source == "plain" else ("user:What is 1+1?\nassistant:" + ("<think>" if thinking else ""))
    )
    expected_tokens = tokenizer.encode(expected_prompt, add_special_tokens=False)
    groups = [[copy.deepcopy(sample) for _ in range(2)]]
    for index, item in enumerate(groups[0]):
        item.group_index, item.index = 0, index
    answer = tokenizer.encode(r"\boxed{2}", add_special_tokens=False)

    def generate(samples, rollout_id):
        assert rollout_id == 7
        assert [item.tokens for item in samples] == [expected_tokens, expected_tokens]
        for item in samples:
            item.tokens = item.tokens + answer
            item.response_length = len(answer)
        return samples

    monkeypatch.setattr(vllm_rlt_rollout, "_tokenizer", lambda checkpoint: tokenizer)
    monkeypatch.setattr(vllm_rlt_rollout.ray, "get", lambda value: value)
    args = Namespace(
        rollout_batch_size=1,
        n_samples_per_prompt=2,
        hf_checkpoint="fixture",
        apply_chat_template=source != "plain",
        apply_chat_template_kwargs={"enable_thinking": not thinking},
        custom_rm_path=None,
        rm_type="math",
        rlt_engine=SimpleNamespace(generate=SimpleNamespace(remote=generate)),
    )
    output = vllm_rlt_rollout.generate_rollout(args, 7, SimpleNamespace(get_samples=lambda count: groups))
    assert [item.reward for item in output.samples[0]] == [1, 1]
    assert [item.response for item in output.samples[0]] == [r"\boxed{2}", r"\boxed{2}"]
    assert tokenizer.render_count == {"plain": 0, "rendered": 1, "conversation": 2}[source]
