from __future__ import annotations

from typing import Any

EPISODE_INFO_KEYS = (
    "max_success_hold_counter",
    "max_consecutive_grasp_steps",
    "max_consecutive_lift_steps",
    "ever_lifted",
    "ever_grasped",
    "dropped_after_lift",
)


def _copy_episode_info_metrics(info: dict[str, Any]) -> None:
    episode_info = info.get("episode")
    if not episode_info:
        return
    for key in EPISODE_INFO_KEYS:
        if key in info:
            episode_info[key] = info[key]


def _copy_final_episode_info_metrics(info: dict[str, Any]) -> None:
    final_info = info.get("final_info")
    if not final_info:
        return
    _copy_episode_info_metrics(final_info)


def patch_maniskill_episode_metrics() -> None:
    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    if not getattr(ManiSkillVectorEnv, "_rl_dev_episode_metrics_patched", False):
        vector_step = ManiSkillVectorEnv.step

        def step_with_episode_metrics(self, actions):
            obs, rew, terminations, truncations, infos = vector_step(self, actions)
            _copy_episode_info_metrics(infos)
            _copy_final_episode_info_metrics(infos)
            return obs, rew, terminations, truncations, infos

        ManiSkillVectorEnv.step = step_with_episode_metrics
        ManiSkillVectorEnv._rl_dev_episode_metrics_patched = True

    if not getattr(CPUGymWrapper, "_rl_dev_episode_metrics_patched", False):
        cpu_step = CPUGymWrapper.step

        def cpu_step_with_episode_metrics(self, action):
            obs, reward, terminated, truncated, info = cpu_step(self, action)
            _copy_episode_info_metrics(info)
            return obs, reward, terminated, truncated, info

        CPUGymWrapper.step = cpu_step_with_episode_metrics
        CPUGymWrapper._rl_dev_episode_metrics_patched = True
