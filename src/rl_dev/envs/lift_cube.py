from typing import Any

import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

LIFT_SUCCESS_BONUS = 5.0
LIFT_DROP_PENALTY = 0.7
LIFT_POST_LIFT_DROP_PENALTY = 1.0
LIFT_HOLD_PROGRESS_SCALE = 0.05
LIFT_HOLD_PROGRESS_MAX = 1.5
LIFT_NORMALIZATION_DENOM = 13.0


def update_success_hold_counter(
    previous: torch.Tensor, hold_condition: torch.Tensor
) -> torch.Tensor:
    return torch.where(
        hold_condition,
        previous + 1,
        torch.zeros_like(previous),
    )


def compute_lift_dense_reward(
    tcp_to_obj_dist: torch.Tensor,
    is_grasped: torch.Tensor,
    z_target_error: torch.Tensor,
    arm_qvel_norm: torch.Tensor,
    is_lifted: torch.Tensor,
    success: torch.Tensor,
    ever_grasped: torch.Tensor,
    ever_lifted: torch.Tensor,
    success_hold_counter: torch.Tensor,
) -> torch.Tensor:
    reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
    grasp_reward = is_grasped.to(torch.float32)
    lift_reward = 1 - torch.tanh(10 * z_target_error)
    static_reward = 1 - torch.tanh(5 * arm_qvel_norm)
    dropped = torch.logical_and(ever_grasped, torch.logical_not(is_grasped))
    dropped_after_lift = torch.logical_and(dropped, ever_lifted)
    hold_progress_reward = torch.clamp(
        success_hold_counter.to(torch.float32) * LIFT_HOLD_PROGRESS_SCALE,
        max=LIFT_HOLD_PROGRESS_MAX,
    )
    reward = reaching_reward
    reward += grasp_reward
    reward += 4.0 * lift_reward * grasp_reward
    reward += 0.5 * static_reward * is_lifted.to(torch.float32)
    reward += hold_progress_reward
    reward += LIFT_SUCCESS_BONUS * success.to(torch.float32)
    reward -= LIFT_DROP_PENALTY * dropped.to(torch.float32)
    reward -= LIFT_POST_LIFT_DROP_PENALTY * dropped_after_lift.to(torch.float32)
    return reward


@register_env("PandaLiftCube-v0", max_episode_steps=100)
class PandaLiftCubeEnv(PickCubeEnv):
    SUPPORTED_ROBOTS = ["panda"]

    def __init__(
        self,
        *args,
        robot_uids: str = "panda",
        lift_height: float = 0.05,
        success_hold_steps: int = 30,
        success_velocity_thresh: float = 0.1,
        **kwargs,
    ):
        if robot_uids != "panda":
            raise ValueError("PandaLiftCube-v0 only supports robot_uids='panda'.")
        self.lift_height = lift_height
        self.success_hold_steps = success_hold_steps
        self.success_velocity_thresh = success_velocity_thresh
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _ensure_tracking_tensors(self) -> None:
        if (
            not hasattr(self, "_initial_cube_heights")
            or self._initial_cube_heights.shape[0] != self.num_envs
        ):
            self._initial_cube_heights = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            self._success_hold_counter = torch.zeros(
                self.num_envs, dtype=torch.int32, device=self.device
            )
            self._ever_grasped = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._ever_lifted = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._dropped_after_lift = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._grasp_hold_counter = torch.zeros(
                self.num_envs, dtype=torch.int32, device=self.device
            )
            self._lift_hold_counter = torch.zeros(
                self.num_envs, dtype=torch.int32, device=self.device
            )
            self._max_success_hold_counter = torch.zeros(
                self.num_envs, dtype=torch.int32, device=self.device
            )
            self._max_consecutive_grasp_steps = torch.zeros(
                self.num_envs, dtype=torch.int32, device=self.device
            )
            self._max_consecutive_lift_steps = torch.zeros(
                self.num_envs, dtype=torch.int32, device=self.device
            )

    def _load_scene(self, options: dict):
        self._ensure_tracking_tensors()
        super()._load_scene(options)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._ensure_tracking_tensors()
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2), device=self.device) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            xyz[:, 0] += self.cube_spawn_center[0]
            xyz[:, 1] += self.cube_spawn_center[1]
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(
                b, lock_x=True, lock_y=True, device=self.device
            )
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))
            self._initial_cube_heights[env_idx] = xyz[:, 2]
            self._success_hold_counter[env_idx] = 0
            self._ever_grasped[env_idx] = False
            self._ever_lifted[env_idx] = False
            self._dropped_after_lift[env_idx] = False
            self._grasp_hold_counter[env_idx] = 0
            self._lift_hold_counter[env_idx] = 0
            self._max_success_hold_counter[env_idx] = 0
            self._max_consecutive_grasp_steps[env_idx] = 0
            self._max_consecutive_lift_steps[env_idx] = 0

    def _get_obs_extra(self, info: dict):
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp_pose.p,
            )
        return obs

    def _arm_qvel(self) -> torch.Tensor:
        qvel = self.agent.robot.get_qvel()
        return qvel[..., :-2]

    def evaluate(self):
        self._ensure_tracking_tensors()
        is_grasped = self.agent.is_grasping(self.cube)
        self._ever_grasped = torch.logical_or(self._ever_grasped, is_grasped)
        lift_target = self._initial_cube_heights + self.lift_height
        object_height = self.cube.pose.p[:, 2]
        is_lifted = object_height >= lift_target
        self._ever_lifted = torch.logical_or(self._ever_lifted, is_lifted)
        object_speed = torch.linalg.norm(self.cube.linear_velocity, axis=1)
        is_velocity_safe = object_speed <= self.success_velocity_thresh
        success_gate = torch.logical_and(is_grasped, is_velocity_safe)
        hold_condition = torch.logical_and(is_lifted, success_gate)
        self._success_hold_counter = update_success_hold_counter(
            self._success_hold_counter, hold_condition
        )
        self._grasp_hold_counter = update_success_hold_counter(
            self._grasp_hold_counter, is_grasped
        )
        self._lift_hold_counter = update_success_hold_counter(
            self._lift_hold_counter, is_lifted
        )
        self._max_success_hold_counter = torch.maximum(
            self._max_success_hold_counter, self._success_hold_counter
        )
        self._max_consecutive_grasp_steps = torch.maximum(
            self._max_consecutive_grasp_steps, self._grasp_hold_counter
        )
        self._max_consecutive_lift_steps = torch.maximum(
            self._max_consecutive_lift_steps, self._lift_hold_counter
        )
        dropped = torch.logical_and(self._ever_grasped, torch.logical_not(is_grasped))
        dropped_after_lift = torch.logical_and(dropped, self._ever_lifted)
        self._dropped_after_lift = torch.logical_or(
            self._dropped_after_lift, dropped_after_lift
        )
        success = self._success_hold_counter >= self.success_hold_steps
        return {
            "success": success,
            "is_grasped": is_grasped,
            "is_lifted": is_lifted,
            "lift_target_z": lift_target,
            "object_height": object_height,
            "object_speed": object_speed,
            "success_gate": success_gate,
            "success_hold_counter": self._success_hold_counter.clone(),
            "ever_grasped": self._ever_grasped.clone(),
            "ever_lifted": self._ever_lifted.clone(),
            "dropped_after_lift": self._dropped_after_lift.clone(),
            "max_success_hold_counter": self._max_success_hold_counter.clone(),
            "max_consecutive_grasp_steps": self._max_consecutive_grasp_steps.clone(),
            "max_consecutive_lift_steps": self._max_consecutive_lift_steps.clone(),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp_pose.p, axis=1
        )
        z_target_error = torch.abs(info["lift_target_z"] - self.cube.pose.p[:, 2])
        arm_qvel_norm = torch.linalg.norm(self._arm_qvel(), axis=1)
        return compute_lift_dense_reward(
            tcp_to_obj_dist=tcp_to_obj_dist,
            is_grasped=info["is_grasped"],
            z_target_error=z_target_error,
            arm_qvel_norm=arm_qvel_norm,
            is_lifted=info["is_lifted"],
            success=info["success"],
            ever_grasped=info["ever_grasped"],
            ever_lifted=info["ever_lifted"],
            success_hold_counter=info["success_hold_counter"],
        )

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        # Max raw reward: reaching(1) + grasp(1) + lift(4) + static(0.5)
        # + hold progress(1.5) + success(5) = 13.0. Drop penalties only fire
        # when not grasping, so they do not affect the max bound.
        return (
            self.compute_dense_reward(obs=obs, action=action, info=info)
            / LIFT_NORMALIZATION_DENOM
        )


PandaLiftCubeEnv.__doc__ = """
Lift-only variant of PickCube for headless visual RL bring-up.

The cube spawn distribution and task camera are inherited from PickCube-v1.
The place goal is removed. Success is defined as lifting the cube by 0.05m
above its initial table height for 30 consecutive control steps while the
object is grasped and moving slowly (< 0.1 m/s) to avoid throw-and-catch
exploits. Requiring a sustained hold ensures the success metric reflects
real task completion rather than a momentary grab.

The reward includes a capped hold-progress term while the cube is continuously
lifted, grasped, and slow. It also applies a drop penalty (-0.7 per step) once
the cube has been grasped but is no longer being held, plus an additional
post-lift drop penalty (-1.0 per step) if the cube had already been lifted.
"""
