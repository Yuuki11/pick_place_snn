from __future__ import annotations

import os

import pytest
import torch

from rl_dev.envs.lift_cube import (
    LIFT_DROP_PENALTY,
    LIFT_HOLD_PROGRESS_MAX,
    LIFT_HOLD_PROGRESS_SCALE,
    LIFT_NORMALIZATION_DENOM,
    LIFT_POST_LIFT_DROP_PENALTY,
    compute_lift_dense_reward,
    update_success_hold_counter,
)


def test_update_success_hold_counter_resets_false_entries():
    previous = torch.tensor([0, 2, 9], dtype=torch.int32)
    condition = torch.tensor([True, False, True])
    updated = update_success_hold_counter(previous, condition)
    assert updated.tolist() == [1, 0, 10]


def test_dense_reward_matches_expected_formula():
    reward = compute_lift_dense_reward(
        tcp_to_obj_dist=torch.tensor([0.0]),
        is_grasped=torch.tensor([True]),
        z_target_error=torch.tensor([0.0]),
        arm_qvel_norm=torch.tensor([0.0]),
        is_lifted=torch.tensor([True]),
        success=torch.tensor([True]),
        ever_grasped=torch.tensor([True]),
        ever_lifted=torch.tensor([True]),
        success_hold_counter=torch.tensor([30], dtype=torch.int32),
    )
    assert torch.allclose(reward, torch.tensor([13.0]))


def test_requested_normalization_divisor_is_retained():
    reward = compute_lift_dense_reward(
        tcp_to_obj_dist=torch.tensor([0.0]),
        is_grasped=torch.tensor([True]),
        z_target_error=torch.tensor([0.0]),
        arm_qvel_norm=torch.tensor([0.0]),
        is_lifted=torch.tensor([True]),
        success=torch.tensor([True]),
        ever_grasped=torch.tensor([True]),
        ever_lifted=torch.tensor([True]),
        success_hold_counter=torch.tensor([30], dtype=torch.int32),
    )
    normalized = reward / LIFT_NORMALIZATION_DENOM
    assert torch.allclose(normalized, torch.tensor([1.0]))


def test_drop_penalty_applies_after_grasp_lost():
    # Once dropped, reaching reward (1.0) is reduced by drop penalty (-0.7)
    reward = compute_lift_dense_reward(
        tcp_to_obj_dist=torch.tensor([0.0]),
        is_grasped=torch.tensor([False]),
        z_target_error=torch.tensor([1.0]),
        arm_qvel_norm=torch.tensor([1.0]),
        is_lifted=torch.tensor([False]),
        success=torch.tensor([False]),
        ever_grasped=torch.tensor([True]),
        ever_lifted=torch.tensor([False]),
        success_hold_counter=torch.tensor([0], dtype=torch.int32),
    )
    # reaching=1.0 (dist=0), drop_penalty=-0.7 => 0.3
    assert torch.allclose(reward, torch.tensor([1.0 - LIFT_DROP_PENALTY]))


def test_extra_drop_penalty_applies_after_lift_lost():
    reward = compute_lift_dense_reward(
        tcp_to_obj_dist=torch.tensor([0.0]),
        is_grasped=torch.tensor([False]),
        z_target_error=torch.tensor([1.0]),
        arm_qvel_norm=torch.tensor([1.0]),
        is_lifted=torch.tensor([False]),
        success=torch.tensor([False]),
        ever_grasped=torch.tensor([True]),
        ever_lifted=torch.tensor([True]),
        success_hold_counter=torch.tensor([0], dtype=torch.int32),
    )
    expected = 1.0 - LIFT_DROP_PENALTY - LIFT_POST_LIFT_DROP_PENALTY
    assert torch.allclose(reward, torch.tensor([expected]))


def test_hold_progress_reward_is_capped():
    reward = compute_lift_dense_reward(
        tcp_to_obj_dist=torch.tensor([1.0]),
        is_grasped=torch.tensor([False]),
        z_target_error=torch.tensor([1.0]),
        arm_qvel_norm=torch.tensor([1.0]),
        is_lifted=torch.tensor([False]),
        success=torch.tensor([False]),
        ever_grasped=torch.tensor([False]),
        ever_lifted=torch.tensor([False]),
        success_hold_counter=torch.tensor([100], dtype=torch.int32),
    )
    reaching_reward = 1 - torch.tanh(torch.tensor([5.0]))
    assert torch.allclose(reward, reaching_reward + LIFT_HOLD_PROGRESS_MAX)


def test_hold_progress_reward_scales_before_cap():
    reward = compute_lift_dense_reward(
        tcp_to_obj_dist=torch.tensor([1.0]),
        is_grasped=torch.tensor([False]),
        z_target_error=torch.tensor([1.0]),
        arm_qvel_norm=torch.tensor([1.0]),
        is_lifted=torch.tensor([False]),
        success=torch.tensor([False]),
        ever_grasped=torch.tensor([False]),
        ever_lifted=torch.tensor([False]),
        success_hold_counter=torch.tensor([4], dtype=torch.int32),
    )
    reaching_reward = 1 - torch.tanh(torch.tensor([5.0]))
    expected = reaching_reward + 4 * LIFT_HOLD_PROGRESS_SCALE
    assert torch.allclose(reward, expected)


def test_no_drop_penalty_before_first_grasp():
    # Before any grasp, behavior is unchanged
    reward = compute_lift_dense_reward(
        tcp_to_obj_dist=torch.tensor([0.0]),
        is_grasped=torch.tensor([False]),
        z_target_error=torch.tensor([1.0]),
        arm_qvel_norm=torch.tensor([1.0]),
        is_lifted=torch.tensor([False]),
        success=torch.tensor([False]),
        ever_grasped=torch.tensor([False]),
        ever_lifted=torch.tensor([False]),
        success_hold_counter=torch.tensor([0], dtype=torch.int32),
    )
    assert torch.allclose(reward, torch.tensor([1.0]))


@pytest.mark.integration
def test_lift_env_smoke():
    if os.environ.get("VISRL_RUN_MANISKILL_TESTS") != "1":
        pytest.skip("Set VISRL_RUN_MANISKILL_TESTS=1 to run ManiSkill integration tests.")

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import rl_dev  # noqa: F401

    env = gym.make(
        "PandaLiftCube-v0",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        sim_backend="gpu",
        sensor_configs={"width": 64, "height": 64},
    )
    obs, _ = env.reset(seed=0)
    assert "sensor_data" in obs
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert reward.shape[0] == 1
    assert "success_hold_counter" in info
    assert "max_success_hold_counter" in info
    assert "max_consecutive_grasp_steps" in info
    assert "max_consecutive_lift_steps" in info
    assert "dropped_after_lift" in info
    frame = env.render()
    assert frame is not None
    env.close()
