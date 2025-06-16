import gym
import torch
import numpy as np
from gym import spaces
from pathlib import Path
from typing import Any, Dict
from flax.training import orbax_utils
from flax.training.train_state import TrainState
import orbax
import jax
import jax.numpy as jnp

# -----------------------------
# Load tokenizer (Flax/Orbax)
# -----------------------------
def load_tokenizer(ckpt_path: str):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(ckpt_path)
    return restored["model"]

def load_dynamics(ckpt_path: str):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(ckpt_path)
    return restored["model"]

# -----------------------------
# Load reward-end model (PyTorch)
# -----------------------------
def load_rew_end_model(pt_path: str, device: str = "cuda"):
    from models.rew_end_model import RewEndModel, RewEndModelConfig

    model = RewEndModel(RewEndModelConfig(
        lstm_dim=512,
        img_channels=3,
        img_size=64,
        cond_channels=128,
        depths=[2, 2, 2],
        channels=[64, 128, 256],
        attn_depths=[0, 0, 1],
        num_actions=15,
    ))
    checkpoint = torch.load(pt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    return model

# -----------------------------
# World Model Env
# -----------------------------
class CoinrunWM(gym.Env):
    def __init__(self, tokenizer, dynamics_model, rew_end_model, device="cuda", max_steps=1000):
        super().__init__()
        self.tokenizer = tokenizer
        self.dynamics = dynamics_model
        self.rew_end = rew_end_model
        self.device = device
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(15)
        self.observation_space = spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8)

        self.latents = None
        self.state = None
        self.t = 0
        self.done = False

    def reset(self):
        self.latents = self.tokenizer.encode_initial_frame().to(self.device)  # (1, latent_dim)
        obs_tensor = self.tokenizer.decode(self.latents)  # (1, 3, 64, 64)
        obs_tensor = obs_tensor.clamp(0, 1) * 255
        self.state = obs_tensor[0].permute(1, 2, 0).byte().cpu().numpy()  # (64, 64, 3)

        self.t = 0
        self.done = False
        return self.state

    def step(self, action):
        if self.done:
            raise RuntimeError("Cannot call step() on a terminated episode. Call reset() first.")

        action_tensor = torch.tensor([action], device=self.device).unsqueeze(0)  # (1, 1)

        # Step the world model
        self.latents = self.dynamics(self.latents, action_tensor)

        # Decode new frame
        obs_tensor = self.tokenizer.decode(self.latents).clamp(0, 1) * 255  # (1, 3, 64, 64)
        self.state = obs_tensor[0].permute(1, 2, 0).byte().cpu().numpy()  # (64, 64, 3)

        # Compute reward and done using rew_end_model
        with torch.no_grad():
            obs_pair = torch.cat([obs_tensor, obs_tensor], dim=1)  # (1, 6, 64, 64)
            rew_logits, end_logits, _ = self.rew_end.predict_rew_end(
                obs=obs_pair.unsqueeze(0),  # (1, 1, 6, 64, 64)
                act=action_tensor,
                next_obs=obs_pair.unsqueeze(0),
            )
            reward_class = torch.argmax(rew_logits.squeeze(0), dim=-1).item()  # 0 → -1, 1 → 0, 2 → +1
            reward = reward_class - 1
            done_flag = torch.argmax(end_logits.squeeze(0), dim=-1).item()
            self.done = bool(done_flag)

        self.t += 1
        if self.t >= self.max_steps:
            self.done = True

        return self.state, float(reward), self.done, {}

    def render(self, mode='rgb_array'):
        return self.state