from dataclasses import replace

import pytest

from vime_plugins.ouro.budget import BudgetSchedule, ExecutionPlan


def test_schedule_resume_keeps_the_next_depth():
    schedule = BudgetSchedule((2, 4, 3, 4))
    restored = BudgetSchedule(tuple([2, 4, 3, 4]))
    assert [schedule.at(i) for i in range(5, 12)] == [restored.at(i) for i in range(5, 12)]


@pytest.mark.parametrize("depths", [(), (0,), (5,), (True,)])
def test_unsupported_budget(depths):
    with pytest.raises(ValueError):
        BudgetSchedule(depths)


@pytest.mark.parametrize(
    "field,value",
    [
        ("loop_budget", 4),
        ("prefill_loop_budget", 4),
        ("decode_loop_budget", 4),
        ("policy_version", 8),
        ("prompt_id", "another-prompt"),
        ("execution_config_hash", "different"),
        ("temperature", 0.5),
    ],
)
def test_mixed_execution_or_group_is_rejected(field, value):
    plan = ExecutionPlan(2, 7)
    metadata = [plan.metadata("prompt") for _ in range(4)]
    metadata[2][field] = value
    with pytest.raises(ValueError, match="Mixed"):
        plan.validate(metadata, group_size=4)


def test_full_groups_and_execution_identity():
    plan = ExecutionPlan(2, 7)
    metadata = [plan.metadata(prompt) for prompt in ("a", "b") for _ in range(4)]
    plan.validate(metadata, 4)
    with pytest.raises(ValueError, match="complete"):
        plan.validate(metadata[:-1], 4)
    assert plan.group_key("a") != plan.group_key("b")
    assert plan.group_key("a") != replace(plan, loop_budget=3).group_key("a")
    assert plan.group_key("a") != replace(plan, policy_version=8).group_key("a")
    assert plan.config_hash != replace(plan, temperature=0.5).config_hash
