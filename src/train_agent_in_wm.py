import gym
import torch
import wandb
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from world_model_env import WorldModelEnv


class WandbLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.n_calls % 10 == 0:
            wandb.log({
                "train/reward": self.locals["rewards"].mean(),
                "train/episode_length": self.locals["dones"].mean(),
            }, step=self.num_timesteps)
        return True

    def _on_rollout_end(self):
        if hasattr(self.training_env.envs[0], "log_trajectory"):
            traj = self.training_env.envs[0].log_trajectory
            wandb.log({
                "trajectory/images": [wandb.Image(img) for img in traj["obs"]],
                "trajectory/actions": traj["actions"],
                "trajectory/rewards": traj["rewards"]
            }, step=self.num_timesteps)


def evaluate(model, env, n_episodes=10, log_trajectory=False):
    all_rewards = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        images = []
        actions = []
        rewards = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            if log_trajectory:
                images.append(obs[0].transpose(1, 2, 0))  # CHW to HWC
                actions.append(action[0])
                rewards.append(reward[0])
        all_rewards.append(total_reward)
        if log_trajectory:
            wandb.log({
                "eval/trajectory/images": [wandb.Image(img) for img in images],
                "eval/trajectory/actions": actions,
                "eval/trajectory/rewards": rewards
            })
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    wandb.log({
        "eval/mean_reward": mean_reward,
        "eval/std_reward": std_reward
    })
    print(f"Evaluation over {n_episodes} episodes: {mean_reward:.2f} ± {std_reward:.2f}")


if __name__ == "__main__":
    wandb.init(project="world_models", config={
        "algo": "PPO",
        "env": "CoinRun-WorldModel",
        "total_timesteps": 500_000,
    })

    env = DummyVecEnv([lambda: WorldModelEnv(
        tokeniser_ckpt="/homes/53/fpinto/jafar/video_tokeniser_ckpt/tokenizer_1749152837_140000", 
        rew_end_ckpt="/homes/53/fpinto/jafar/rew_end_ckpt/rew_end_ckpt_0999.pt",
    )])

    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_logs")

    model.learn(
        total_timesteps=500_000,
        callback=WandbLoggingCallback()
    )

    model.save("ppo_agent_world_model")

    # Evaluation
    eval_env = DummyVecEnv([lambda: WorldModelEnv(
        tokeniser_ckpt="/homes/53/fpinto/jafar/video_tokeniser_ckpt/tokenizer_1749152837_140000", 
        rew_end_ckpt="/homes/53/fpinto/jafar/rew_end_ckpt/rew_end_ckpt_0999.pt",
    )])
    evaluate(model, eval_env, n_episodes=10, log_trajectory=True)

    wandb.finish()