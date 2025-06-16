import numpy as np
import wandb
from typing import Optional, Tuple, List
import torch

from procgen import ProcgenGym3Env
from gym3 import types_np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn


class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        wandb.log({
            "train/avg_reward": self.locals["rewards"].mean(),
            "train/length": self.locals["dones"].sum(),
            "train/step": self.num_timesteps,
        })
        return True


class CoinrunSolvedCallback(BaseCallback):
    def __init__(self, log_every=100, verbose=0):
        super().__init__(verbose)
        self.solved_count = 0
        self.episode_count = 0
        self.log_every = log_every

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for i, done in enumerate(dones):
            if done:
                ep_rew = infos[i].get("episode", {}).get("r", 0)
                self.episode_count += 1
                if ep_rew >= 10:  # Coin collected
                    self.solved_count += 1

        if self.episode_count and self.episode_count % self.log_every == 0:
            wandb.log({
                "train/episodes": self.episode_count,
                "train/levels_solved": self.solved_count,
                "train/solved_pct": 100 * self.solved_count / self.episode_count
            })

        return True


class ProcgenSB3Wrapper(VecEnv):
    def __init__(self, env_name: str = "coinrun", num_envs: int = 8, start_level: int = 0, num_levels: int = 0, distribution_mode: str = "easy"):
        super().__init__(num_envs=num_envs, observation_space=None, action_space=None)
        self.env = ProcgenGym3Env(
            num=num_envs,
            env_name=env_name,
            start_level=start_level,
            num_levels=num_levels,
            distribution_mode=distribution_mode
        )
        self.ob_space = self.env.ob_space
        self.ac_space = self.env.ac_space
        self.actions = None

        self.observation_space = spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(self.ac_space.eltype.n)

    def reset(self) -> VecEnvObs:
        _, obs, _ = self.env.observe()
        return obs["rgb"]

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        self.env.act(self.actions)
        rew, obs, first = self.env.observe()
        done = first.astype(np.bool_)

        infos: List[dict] = [{"episode": {"r": float(r)}} if d else {} for r, d in zip(rew, done)]
        for i, d in enumerate(done):
            if d:
                infos[i]["terminal_observation"] = obs["rgb"][i].copy()

        return obs["rgb"], rew, done, infos

    def close(self) -> None:
        self.env.close()

    def seed(self, seed: Optional[int] = None) -> None:
        pass

    def get_attr(self, attr_name: str, indices: Optional[List[int]] = None) -> List:
        return [getattr(self, attr_name)] * self.num_envs

    def set_attr(self, attr_name: str, value, indices: Optional[List[int]] = None) -> None:
        raise NotImplementedError("set_attr not supported")

    def env_method(self, method_name: str, *method_args, indices: Optional[List[int]] = None, **method_kwargs):
        raise NotImplementedError("env_method not supported")

    def env_is_wrapped(self, wrapper_class, indices: Optional[List[int]] = None) -> List[bool]:
        return [False] * self.num_envs


class ImpalaCNN(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         features_extractor_class=ImpalaFeatureExtractor,
                         features_extractor_kwargs={})


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.res1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        res = self.res1(x)
        return x + res

class ImpalaFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        in_channels = observation_space.shape[0]

        self.net = nn.Sequential(
            ImpalaBlock(in_channels, 16),
            ImpalaBlock(16, 32),
            ImpalaBlock(32, 32),
            nn.Flatten()
        )

        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            n_flatten = self.net(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

        self._features_dim = features_dim 

    def forward(self, x):
        return self.linear(self.net(x))


if __name__ == "__main__":
    wandb.init(
        project="world_models",
        config={
            "env_name": "coinrun",
            "num_envs": 8,
            "total_timesteps": 25_000_000,
            "n_steps": 1024,
            "batch_size": 256,
            "start_level": 0,
            "num_levels": 200,
            "distribution_mode": "easy",
        }
    )
    config = wandb.config

    env = ProcgenSB3Wrapper(env_name=config.env_name, num_envs=config.num_envs,
                            start_level=config.start_level, num_levels=config.num_levels,
                            distribution_mode=config.distribution_mode)
    env = VecMonitor(env)

    model = PPO(ImpalaCNN, env, verbose=1,
                n_steps=config.n_steps, batch_size=config.batch_size, learning_rate=5e-4, ent_coef=0.01)
    wandb.watch(model.policy, log="all", log_freq=100)

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[WandbCallback(), CoinrunSolvedCallback()]
    )

    eval_env = ProcgenSB3Wrapper(env_name=config.env_name, num_envs=1, start_level=500, num_levels=0)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

    wandb.log({"eval/mean_reward": mean_reward, "eval/std_reward": std_reward})
    print(f"Eval Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    model.save("ppo_coinrun_expert")
    wandb.save("ppo_coinrun_expert.zip")
    wandb.finish()