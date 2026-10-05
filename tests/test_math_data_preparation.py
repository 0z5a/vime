"""Protect source integrity, held-out isolation and explicit answer repairs."""

import hashlib
import io
from dataclasses import replace

import pytest

from benchmarks.download_math_data import download, verify
from benchmarks.prepare_math_data import Row, digest, problem_key, source_label, source_level, split_rows


def row(identity: str, problem: str, label: str = "2") -> Row:
    return Row(identity, problem, rf"\boxed{{{label}}}", label, "Algebra", 2, "source.parquet", 0, "", False)


@pytest.mark.parametrize("algorithm", ["sha256", "git_blob_sha1"])
def test_source_content_not_just_length(algorithm):
    data = b"source"
    expected = (
        hashlib.sha256(data).hexdigest() if algorithm == "sha256" else hashlib.sha1(b"blob 6\0" + data).hexdigest()
    )
    item = {"path": "source", "size": 6, algorithm: expected}
    assert verify(data, item) == hashlib.sha256(data).hexdigest()
    with pytest.raises(ValueError, match="content hash"):
        verify(b"edited", item)
    with pytest.raises(ValueError, match="size"):
        verify(b"short", item)


@pytest.mark.parametrize("kind,prefix", [("dataset", "datasets/"), ("model", "")])
def test_download_checks_content_and_reuses_only_verified_file(tmp_path, monkeypatch, kind, prefix):
    data = b"source"
    item = {"path": "input.json", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    source = {"repository": "owner/repo", "revision": "abc", "kind": kind, "files": [item]}
    calls = []

    def fetch(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(data)

    monkeypatch.setattr("benchmarks.download_math_data.urllib.request.urlopen", fetch)
    first = download(tmp_path, source, item)
    assert download(tmp_path, source, item) == first
    assert calls == [(f"https://huggingface.co/{prefix}owner/repo/resolve/abc/input.json", 60)]
    target = tmp_path / first["path"]
    target.write_bytes(b"edited")
    with pytest.raises(ValueError, match="content hash"):
        download(tmp_path, source, item)
    assert target.read_bytes() == b"edited"
    assert len(calls) == 1


def test_normalization_does_not_conflate_mathematical_characters():
    assert problem_key("café\n x") == problem_key("cafe\u0301   x")
    assert problem_key("x²") != problem_key("x2")
    assert problem_key("X") != problem_key("x")


def test_unknown_source_difficulty_is_preserved():
    assert source_level("Level ?") is None
    assert source_level("Level 3") == 3
    assert replace(row("unknown", "Problem"), level=source_level("Level ?")).record()["metadata"]["level"] is None


def test_missing_answers_require_source_bound_repairs():
    solution = r"The answer is \boxed 2."
    repairs = {"a": {"solution_sha256": digest(solution), "label": "2", "reason": "Unbraced source answer"}}
    assert source_label("a", solution, repairs) == "2"
    with pytest.raises(ValueError, match="source changed"):
        source_label("a", solution + " changed", repairs)
    for missing in (r"\boxed{}", solution, "No answer"):
        with pytest.raises(ValueError, match="Missing label"):
            source_label("other", missing, repairs)
    assert source_label("normal", r"\boxed{\frac{1}{2}}", repairs) == r"\frac{1}{2}"


def fixture_groups():
    return {
        "train": [row("keep", "one  question"), row("duplicate", "one question"), row("leak", "held out")],
        "test": [row("test-a", "held out"), row("test-b", "dev b"), row("test-c", "dev c")],
        "math500": [row("eval", "held out")],
    }


def test_complete_partition_and_explicit_exclusions():
    groups = fixture_groups()
    splits, excluded = split_rows(groups, 1)
    assert [r.id for r in splits["train"]] == ["keep"]
    assert excluded == [
        {"id": "duplicate", "reason": "train_duplicate", "matches": "keep"},
        {"id": "leak", "reason": "test_overlap", "matches": "test-a"},
    ]
    sets = [{r.key for r in rows} for rows in splits.values()]
    assert sum(len(keys) for keys in sets) == len(set.union(*sets))
    assert {r.id for r in splits["development"] + splits["reserve"]} == {"test-b", "test-c"}
    reversed_groups = {name: list(reversed(rows)) if name == "test" else rows for name, rows in groups.items()}
    assert split_rows(reversed_groups, 1)[0] == splits
    assert len(groups["train"]) == 3


def test_duplicate_conflicting_labels_fail():
    groups = fixture_groups()
    groups["train"][1] = replace(groups["train"][1], label="3")
    with pytest.raises(ValueError, match="conflicting labels"):
        split_rows(groups, 1)


@pytest.mark.parametrize("name", ["test", "math500"])
def test_duplicate_heldout_fails(name):
    groups = fixture_groups()
    groups[name].append(groups[name][0])
    with pytest.raises(ValueError, match="duplicate normalized"):
        split_rows(groups, 1)


def test_unknown_math500_question_fails():
    groups = fixture_groups()
    groups["math500"] = [row("outside", "unrecognized test question")]
    with pytest.raises(ValueError, match="outside"):
        split_rows(groups, 1)


def test_record_contains_no_solution_or_answer_in_prompt():
    original = row("a", "Find the value.", "123456789")
    record = original.record()
    assert "123456789" not in record["prompt"]
    assert "solution" not in record and "solution" not in record["metadata"]
    assert record["label"] == "123456789"
    assert record["metadata"]["solution_sha256"] == digest(original.solution)
    assert record["metadata"]["rm_type"] == "math"
