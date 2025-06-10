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
class WorldModelEnv(gym.Env):
    def __init__(self, tokeniser_ckpt: str, rew_end_ckpt: str, device="cuda"):
        super().__init__()
        self.device = device

        # Load models
        self.tokenizer: TrainState = load_tokenizer(tokeniser_ckpt)
        self.rew_end_model = load_rew_end_model(rew_end_ckpt, device)

        # State = 64x64x3 RGB images (normalized float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, 64, 64), dtype=np.float32)
        self.action_space = spaces.Discrete(15)  # CoinRun

        # Init state buffer
        self.seq_len = 16
        self.history = []
        self.current_state = self._initial_state()

    def _initial_state(self):
        # Start with random noise (or a specific init frame if needed)
        return np.random.rand(3, 64, 64).astype(np.float32)

    def reset(self):
        self.history = [self.current_state] * self.seq_len
        return self.current_state

    def step(self, action):
        # -----------------------------
        # 1. Encode image sequence with tokenizer
        # -----------------------------
        input_sequence = np.stack(self.history[-self.seq_len:])  # (seq, 3, 64, 64)
        input_sequence = jnp.array(input_sequence).astype(jnp.float32)

        # Predict next state using tokenizer (world model)
        next_state_pred = self.tokenizer.apply_fn(
            {"params": self.tokenizer.params},
            input_sequence[None, ...]  # Add batch dim: (1, seq, 3, 64, 64)
        )
        next_state = np.array(next_state_pred[0, -1])  # Get latest frame, remove batch

        # -----------------------------
        # 2. Predict reward and done
        # -----------------------------
        obs_tensor = torch.tensor(
            np.stack(self.history[-self.seq_len:]), dtype=torch.float32
        ).unsqueeze(0).to(self.device)  # (1, seq, 3, 64, 64)

        act_tensor = torch.tensor(
            [[action] * self.seq_len], dtype=torch.long
        ).to(self.device)  # (1, seq)

        next_obs_tensor = torch.tensor(
            np.stack([next_state] * self.seq_len), dtype=torch.float32
        ).unsqueeze(0).to(self.device)  # (1, seq, 3, 64, 64)

        with torch.no_grad():
            logits_rew, logits_end, _ = self.rew_end_model.predict_rew_end(
                obs_tensor, act_tensor, next_obs_tensor
            )

        # Extract predicted reward {-1, 0, 1}
        pred_reward = torch.argmax(logits_rew[0, -1]).item() - 1

        # Extract predicted terminal flag
        done = torch.argmax(logits_end[0, -1]).item() == 1

        # -----------------------------
        # 3. Update state and return
        # -----------------------------
        self.current_state = next_state.astype(np.float32)
        self.history.append(self.current_state)
        if len(self.history) > self.seq_len:
            self.history.pop(0)

        info: Dict[str, Any] = {}
        return self.current_state, pred_reward, done, info
