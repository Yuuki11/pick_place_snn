"""Spiking DDPG (state-based) -- hybrid architecture.

The actor uses a population-coded spiking neural network (LIF neurons with
surrogate gradients) inspired by PopSAN (Tang et al., CoRL 2020).  The critic
remains a standard ANN since it is only used during training.

Key design choices
------------------
* **Population coding** – each action dimension is decoded from a small
  population of spiking neurons via their final membrane voltages, giving
  smooth continuous outputs even though internal communication is spike-based.
* **Surrogate gradients** – the Heaviside non-linearity in the LIF neuron is
  replaced by a fast-sigmoid surrogate during the backward pass so standard
  BPTT works through the spiking dynamics.
* **Membrane-voltage readout** – the output layer is a non-spiking LIF whose
  membrane voltage is linearly decoded, following PopSAN.  This avoids the
  precision issues of spike-count decoding for continuous control.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import os
import random
import signal
import time
from typing import Optional

import gymnasium as gym
import mani_skill.envs
import numpy as np
import snntorch as snn
from snntorch import surrogate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import tyro
from torch.utils.tensorboard import SummaryWriter

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class Args:
    exp_name: Optional[str] = None
    run_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ManiSkill"
    wandb_entity: Optional[str] = None
    wandb_group: str = "SNN-DDPG"
    capture_video: bool = True
    save_trajectory: bool = False
    save_model: bool = True
    evaluate: bool = False
    checkpoint: Optional[str] = None
    log_freq: int = 1_000

    env_id: str = "PickCube-v1"
    env_vectorization: str = "gpu"
    num_envs: int = 16
    num_eval_envs: int = 16
    partial_reset: bool = False
    eval_partial_reset: bool = False
    num_steps: int = 50
    num_eval_steps: int = 50
    reconfiguration_freq: Optional[int] = None
    eval_reconfiguration_freq: Optional[int] = 1
    eval_freq: int = 25
    save_train_video_freq: Optional[int] = None
    control_mode: Optional[str] = "pd_joint_delta_pos"

    total_timesteps: int = 1_000_000
    buffer_size: int = 1_000_000
    buffer_device: str = "cuda"
    gamma: float = 0.8
    tau: float = 0.01
    batch_size: int = 1024
    learning_starts: int = 4_000
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    training_freq: int = 64
    exploration_noise: float = 0.15
    target_noise_clip: float = 0.25
    target_policy_noise: float = 0.2
    max_grad_norm: float = 1.0
    bootstrap_at_done: str = "always"

    # SNN-specific
    snn_timesteps: int = 16
    snn_beta: float = 0.9
    snn_slope: float = 25.0
    snn_hidden: int = 256
    pop_size: int = 10

    grad_steps_per_iteration: int = 0
    steps_per_env: int = 0


# ---------------------------------------------------------------------------
# Replay buffer (identical to ddpg_state.py)
# ---------------------------------------------------------------------------

@dataclass
class ReplayBufferSample:
    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, env, num_envs, buffer_size, storage_device, sample_device):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.per_env_buffer_size = buffer_size // num_envs
        obs_shape = env.single_observation_space.shape
        action_shape = env.single_action_space.shape
        self.obs = torch.zeros((self.per_env_buffer_size, num_envs) + obs_shape, device=storage_device)
        self.next_obs = torch.zeros((self.per_env_buffer_size, num_envs) + obs_shape, device=storage_device)
        self.actions = torch.zeros((self.per_env_buffer_size, num_envs) + action_shape, device=storage_device)
        self.rewards = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.dones = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)

    def add(self, obs, next_obs, action, reward, done):
        if self.storage_device == torch.device("cpu"):
            obs = obs.cpu()
            next_obs = next_obs.cpu()
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSample:
        upper = self.per_env_buffer_size if self.full else self.pos
        batch_inds = torch.randint(0, upper, size=(batch_size,))
        env_inds = torch.randint(0, self.num_envs, size=(batch_size,))
        return ReplayBufferSample(
            obs=self.obs[batch_inds, env_inds].to(self.sample_device),
            next_obs=self.next_obs[batch_inds, env_inds].to(self.sample_device),
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device),
        )


# ---------------------------------------------------------------------------
# Spiking Actor (PopSAN-style)
# ---------------------------------------------------------------------------

class SpikingActor(nn.Module):
    """Population-coded Spiking Actor Network.

    Architecture:
        obs -> FC -> LIF -> FC -> LIF -> FC(pop) -> LIF(output) -> decode

    The output layer projects to ``act_dim * pop_size`` neurons.  The final
    membrane voltages of these neurons are reshaped to
    ``(batch, act_dim, pop_size)`` and linearly decoded to a single scalar
    per action dimension, then squashed through tanh.
    """

    def __init__(self, env, num_steps: int = 16, beta: float = 0.9,
                 slope: float = 25.0, hidden: int = 256, pop_size: int = 10):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        self.act_dim = int(np.prod(env.single_action_space.shape))
        self.num_steps = num_steps
        self.pop_size = pop_size

        spike_grad = surrogate.fast_sigmoid(slope=slope)

        # Spiking hidden layers -- LayerNorm before each LIF ensures input
        # currents are in the right range relative to the spike threshold.
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Population-coded output layer (non-spiking readout via membrane voltage)
        self.fc_pop = nn.Linear(hidden, self.act_dim * pop_size)
        self.ln_pop = nn.LayerNorm(self.act_dim * pop_size)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True)

        # Linear decoder: pop_size membrane voltages -> 1 action value
        self.decoder = nn.Linear(pop_size, 1, bias=False)
        nn.init.uniform_(self.decoder.weight, -0.5, 0.5)

        # Action scaling
        high = env.single_action_space.high
        low = env.single_action_space.low
        self.register_buffer("action_scale",
                             torch.tensor((high - low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias",
                             torch.tensor((high + low) / 2.0, dtype=torch.float32))

    def forward(self, obs: torch.Tensor, return_snn_stats: bool = False) -> torch.Tensor:
        batch = obs.shape[0]

        # Initialise membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk1_count = torch.zeros(batch, device=obs.device)
        spk2_count = torch.zeros(batch, device=obs.device)

        # Simulate for T timesteps (same input each step = rate coding)
        for _ in range(self.num_steps):
            cur1 = self.ln1(self.fc1(obs))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.ln2(self.fc2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            cur_pop = self.ln_pop(self.fc_pop(spk2))
            spk_out, mem_out = self.lif_out(cur_pop, mem_out)

            if return_snn_stats:
                spk1_count += spk1.sum(dim=-1)
                spk2_count += spk2.sum(dim=-1)

        # Decode action from final membrane voltage of output population
        # mem_out: (batch, act_dim * pop_size)
        mem_pop = mem_out.view(batch, self.act_dim, self.pop_size)
        raw_action = self.decoder(mem_pop).squeeze(-1)  # (batch, act_dim)

        action = torch.tanh(raw_action) * self.action_scale + self.action_bias

        if return_snn_stats:
            hidden = self.fc1.out_features
            stats = {
                "snn/spike_rate_lif1": spk1_count.mean() / (self.num_steps * hidden),
                "snn/spike_rate_lif2": spk2_count.mean() / (self.num_steps * hidden),
                "snn/mem_out_mean": mem_out.mean(),
                "snn/mem_out_std": mem_out.std(),
            }
            return action, stats
        return action


# ---------------------------------------------------------------------------
# ANN Critic (unchanged from ddpg_state.py)
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        act_dim = int(np.prod(env.single_action_space.shape))
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=1))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, log_wandb: bool = False,
                 tensorboard: SummaryWriter | None = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.grad_steps_per_iteration = args.training_freq
    args.steps_per_env = args.training_freq // args.num_envs
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
    run_name = args.run_name or f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    termination_state = {"requested": False, "signal": None, "timestamp": None}

    def request_shutdown(signum, _frame):
        signal_name = signal.Signals(signum).name
        if termination_state["requested"]:
            return
        termination_state["requested"] = True
        termination_state["signal"] = signal_name
        termination_state["timestamp"] = time.time()
        print(f"[SNN-DDPG-State] Received {signal_name}. Saving termination checkpoint before exit.")

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    envs = gym.make(
        args.env_id,
        num_envs=args.num_envs if not args.evaluate else 1,
        reconfiguration_freq=args.reconfiguration_freq,
        **env_kwargs,
    )
    eval_envs = gym.make(
        args.env_id,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        human_render_camera_configs=dict(shader_pack="default"),
        **env_kwargs,
    )
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"{run_dir}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(
                envs,
                output_dir=f"{run_dir}/train_videos",
                save_trajectory=False,
                save_video_trigger=save_video_trigger,
                max_steps_per_video=args.num_steps,
                video_fps=30,
            )
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=args.save_trajectory,
            save_video=args.capture_video,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=30,
        )
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box)

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training (Spiking DDPG)")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(
                **env_kwargs,
                num_envs=args.num_envs,
                env_id=args.env_id,
                reward_mode="normalized_dense",
                env_horizon=max_episode_steps,
                partial_reset=args.partial_reset,
            )
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=["snn-ddpg", "state"],
            )
        writer = SummaryWriter(run_dir)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation (Spiking DDPG)")

    # --- Networks ---
    actor = SpikingActor(
        envs,
        num_steps=args.snn_timesteps,
        beta=args.snn_beta,
        slope=args.snn_slope,
        hidden=args.snn_hidden,
        pop_size=args.pop_size,
    ).to(device)
    actor_target = SpikingActor(
        envs,
        num_steps=args.snn_timesteps,
        beta=args.snn_beta,
        slope=args.snn_slope,
        hidden=args.snn_hidden,
        pop_size=args.pop_size,
    ).to(device)
    critic = Critic(envs).to(device)
    critic_target = Critic(envs).to(device)

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.critic_lr)
    max_action = torch.tensor(envs.single_action_space.high, device=device, dtype=torch.float32)
    min_action = torch.tensor(envs.single_action_space.low, device=device, dtype=torch.float32)

    rb = ReplayBuffer(
        env=envs,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        storage_device=torch.device(args.buffer_device),
        sample_device=device,
    )

    # --- Logging SNN architecture info ---
    total_params = sum(p.numel() for p in actor.parameters())
    print(f"[SNN-DDPG] Spiking actor: {total_params:,} params, "
          f"T={args.snn_timesteps} steps, beta={args.snn_beta}, "
          f"pop_size={args.pop_size}")

    obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    global_step = 0
    learning_has_started = False
    cumulative_times = defaultdict(float)
    global_steps_per_iteration = args.num_envs * args.steps_per_env
    pbar = tqdm.tqdm(range(args.total_timesteps))
    actor_loss = torch.tensor(0.0, device=device)
    critic_loss = torch.tensor(0.0, device=device)

    while global_step < args.total_timesteps and not termination_state["requested"]:
        # ---- Evaluation ----
        if args.eval_freq > 0 and (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
            actor.eval()
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            for _ in range(args.num_eval_steps):
                if termination_state["requested"]:
                    break
                with torch.no_grad():
                    eval_action = actor(eval_obs)
                    eval_obs, _, _, _, eval_infos = eval_envs.step(eval_action)
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v[mask])
            eval_metrics_mean = {}
            for k, v in eval_metrics.items():
                mean = torch.cat(v).float().mean()
                eval_metrics_mean[k] = mean
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
            if eval_metrics_mean:
                pbar.set_description(
                    f"success_once: {eval_metrics_mean['success_once']:.2f}, "
                    f"return: {eval_metrics_mean['return']:.2f}"
                )
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                break
            if args.save_model:
                model_path = f"{run_dir}/ckpt_{global_step}.pt"
                torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, model_path)
                print(f"model saved to {model_path}")
            actor.train()
            if termination_state["requested"]:
                break

        # ---- Rollout ----
        rollout_time = time.perf_counter()
        for _ in range(args.steps_per_env):
            if termination_state["requested"]:
                break
            global_step += args.num_envs
            if not learning_has_started:
                actions = 2 * torch.rand(size=envs.action_space.shape, dtype=torch.float32, device=device) - 1
            else:
                with torch.no_grad():
                    actions = actor(obs)
                    noise = torch.randn_like(actions) * args.exploration_noise * actor.action_scale
                    actions = torch.clamp(actions + noise, min_action, max_action)
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            real_next_obs = next_obs.clone()
            if args.bootstrap_at_done == "never":
                need_final_obs = torch.ones_like(terminations, dtype=torch.bool)
                stop_bootstrap = truncations | terminations
            elif args.bootstrap_at_done == "always":
                need_final_obs = truncations | terminations
                stop_bootstrap = torch.zeros_like(terminations, dtype=torch.bool)
            else:
                need_final_obs = truncations & (~terminations)
                stop_bootstrap = terminations
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                real_next_obs[need_final_obs] = infos["final_observation"][need_final_obs].clone()
                if logger is not None:
                    for k, v in final_info["episode"].items():
                        logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
            rb.add(obs, real_next_obs, actions, rewards, stop_bootstrap)
            obs = next_obs
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        pbar.update(global_steps_per_iteration)

        # ---- Gradient updates ----
        if global_step < args.learning_starts:
            continue
        learning_has_started = True
        update_time = time.perf_counter()
        for _ in range(args.grad_steps_per_iteration):
            if termination_state["requested"]:
                break
            data = rb.sample(args.batch_size)

            # Critic update
            with torch.no_grad():
                next_actions = actor_target(data.next_obs)
                noise = (torch.randn_like(next_actions) * args.target_policy_noise).clamp(
                    -args.target_noise_clip, args.target_noise_clip
                )
                next_actions = torch.clamp(next_actions + noise, min_action, max_action)
                next_q = critic_target(data.next_obs, next_actions).view(-1)
                target_q = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * next_q

            current_q = critic(data.obs, data.actions).view(-1)
            critic_loss = F.mse_loss(current_q, target_q)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
            critic_optimizer.step()

            # Actor update (surrogate gradients flow through the spiking network)
            actor_loss = -critic(data.obs, actor(data.obs)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            actor_optimizer.step()

            soft_update(actor, actor_target, args.tau)
            soft_update(critic, critic_target, args.tau)
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time
        if termination_state["requested"]:
            break

        # ---- Logging ----
        if logger is not None and (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
            logger.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
            logger.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            logger.add_scalar("losses/q_values", current_q.mean().item(), global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar("time/rollout_fps", global_steps_per_iteration / rollout_time, global_step)
            for k, v in cumulative_times.items():
                logger.add_scalar(f"time/total_{k}", v, global_step)
            # SNN-specific diagnostics
            with torch.no_grad():
                sample = rb.sample(min(256, args.batch_size))
                _, snn_stats = actor(sample.obs, return_snn_stats=True)
                for k, v in snn_stats.items():
                    logger.add_scalar(k, v.item(), global_step)

    terminated = termination_state["requested"]
    if not args.evaluate and (args.save_model or terminated) and "actor" in locals() and "critic" in locals():
        ckpt_name = "terminated_ckpt.pt" if terminated else "final_ckpt.pt"
        model_path = f"{run_dir}/{ckpt_name}"
        payload = {"actor": actor.state_dict(), "critic": critic.state_dict(), "global_step": global_step}
        if terminated:
            payload["terminated_by"] = termination_state["signal"]
        torch.save(payload, model_path)
        print(f"model saved to {model_path}")
    if terminated:
        termination_path = os.path.join(run_dir, "termination.json")
        with open(termination_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "run_name": run_name,
                    "exp_name": args.exp_name,
                    "global_step": global_step,
                    "signal": termination_state["signal"],
                    "timestamp": termination_state["timestamp"],
                },
                handle,
                indent=2,
                sort_keys=True,
            )
        print(f"termination reason written to {termination_path}")
    if args.track and "wandb" in globals():
        wandb.finish(exit_code=128 + signal.Signals[termination_state["signal"]].value if terminated else 0)
    if not args.evaluate and "writer" in locals():
        writer.close()
    if "envs" in locals():
        envs.close()
    if "eval_envs" in locals():
        eval_envs.close()
