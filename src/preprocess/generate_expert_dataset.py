"""
Generates a dataset of random-action CoinRun episodes.
Episodes are saved individually as memory-mapped files for efficient loading.
"""

from dataclasses import dataclass
from pathlib import Path

from gym3 import types_np
import numpy as np
from procgen import ProcgenGym3Env
from stable_baselines3 import PPO
import tyro


@dataclass
class Args:
    num_episodes: int = 10000
    output_dir: str = "data/coinrun_episodes"
    min_episode_length: int = 50


args = tyro.cli(Args)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# --- Generate episodes ---
i = 0
metadata = []

# Load your model
model = PPO.load("models/trained_agents/ppo_coinrun_expert.zip", device="cuda")

coins = 0
while i < args.num_episodes:
    seed = np.random.randint(0, 10000)  # Random seed for each episode
    env = ProcgenGym3Env(num=1, env_name="coinrun", start_level=seed)
    rgb_seq = []
    rew_seq = []
    done_seq = []
    act_seq = []

    obs = env.observe()[1]["rgb"]  # Initial observation (shape: (1, 64, 64, 3))

    for j in range(1000):
        # env.act(types_np.sample(env.ac_space, bshape=(env.num,))) # Random action
        obs_rgb = obs.transpose(0, 3, 1, 2).astype(np.float32) / 255.0  # → (1, 3, 64, 64) normalised
        action, _ = model.predict(obs_rgb, deterministic=True)
        env.act(action)
        rew, obs, first = env.observe()

        obs = obs["rgb"]  # Update observation
        
        rgb_seq.append(obs)             # shape (1, 64, 64, 3)
        rew_seq.append(rew.copy())             # shape (1,)
        done_seq.append(first.copy())          # shape (1,)
        act_seq.append(action.copy())          # shape (1,)

        if first:
            break

    if len(rgb_seq) >= args.min_episode_length:
        rgb = np.concatenate(rgb_seq, axis=0)         # (T, 64, 64, 3)
        acts = np.concatenate(act_seq, axis=0)        # (T,)
        rews = np.concatenate(rew_seq, axis=0)        # (T,)
        dones = np.concatenate(done_seq, axis=0)      # (T,)
        

        # Save everything in a dictionary
        episode = {
            "rgb": rgb.astype(np.uint8),
            "actions": acts.astype(np.int64),
            "reward": rews.astype(np.float32),
            "done": dones.astype(np.bool_),
        }

        episode_path = output_dir / f"episode_{i}.npz"
        np.savez_compressed(episode_path, **episode)

        metadata.append({"path": str(episode_path), "length": len(rgb)})
        print(f"Episode {i} completed, length: {len(rgb)}, total reward: {rews.sum()}")
        i += 1
        if rews.sum() > 0:
            coins += 1
    else:
        print(f"Episode too short ({len(rgb_seq)}), resampling...")

# --- Save metadata ---
np.save(output_dir / "metadata.npy", metadata)
print(f"Dataset generated with {len(metadata)} valid episodes, {coins} coins collected.")


