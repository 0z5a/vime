from argparse import Namespace

import pytest

from vime.utils.reward_normalization import normalize_rewards


def test_grpo_preserves_prompt_groups_and_constant_groups():
    args = Namespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        n_samples_per_prompt=4,
        rollout_batch_size=2,
    )
    # The second prompt's constant reward must not change the first prompt's advantages.
    assert normalize_rewards(args, [0, 0, 0, 1, 5, 5, 5, 5]) == pytest.approx(
        [-0.5, -0.5, -0.5, 1.5, 0, 0, 0, 0], abs=4e-6
    )
    args.rewards_normalization = False
    assert normalize_rewards(args, [0, 1]) == [0, 1]
