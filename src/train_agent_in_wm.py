import gym
import torch
import wandb
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import torch.nn as nn

from world_model_env import CoinrunWM


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
    

    env = CoinrunWM(tokenizer, dynamics_model, rew_end_model)

    model = PPO(ImpalaCNN, env, verbose=1,
                n_steps=config.n_steps, batch_size=config.batch_size, learning_rate=5e-4, ent_coef=0.01)
    wandb.watch(model.policy, log="all", log_freq=100)

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[WandbCallback(), CoinrunSolvedCallback()]
    )

    eval_env = CoinrunWM(tokenizer, dynamics_model, rew_end_model) #ProcgenSB3Wrapper(env_name=config.env_name, num_envs=1, start_level=500, num_levels=0)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

    wandb.log({"eval/mean_reward": mean_reward, "eval/std_reward": std_reward})
    print(f"Eval Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    model.save("ppo_coinrun_WM")
    wandb.save("ppo_coinrun_WM.zip")
    wandb.finish()