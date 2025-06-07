import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from gym3 import types_np
from procgen import ProcgenGym3Env
from tqdm import tqdm
import numpy as np
import os
import wandb

from models.rew_end_model import RewEndModel, RewEndModelConfig

# --- Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_envs = 16
seq_len = 32       # LSTM rollout length
num_updates = 1000
learning_rate = 1e-4

checkpoint_dir = "rew_end_ckpt"
os.makedirs(checkpoint_dir, exist_ok=True)

# wandb.init(
#     project="world_models",
#     config={
#         "env": "coinrun",
#         "seq_len": seq_len,
#         "num_envs": num_envs,
#         "learning_rate": learning_rate,
#         "model": "RewEndModel",
#         "updates": num_updates,
#     }
# )

# --- Environment ---
env = ProcgenGym3Env(num=num_envs, env_name="coinrun", start_level=0, num_levels=0)
ac_space = env.ac_space
ob_space = env.ob_space

# --- Model ---
model = RewEndModel(
    RewEndModelConfig(
        lstm_dim=512,
        img_channels=3,
        img_size=64,
        cond_channels=128,
        depths=[2, 2, 2],
        channels=[64, 128, 256],
        attn_depths=[0, 0, 1],
        num_actions=ac_space.size
    )
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# --- Buffer for online episodes ---
obs_seq = []
next_obs_seq = []
act_seq = []
rew_seq = []
end_seq = []

# --- Step into environment ---
prev_obs = env.observe()[1]["rgb"]

pbar = tqdm(range(num_updates))
for update in pbar:
    # --- Collect seq_len transitions ---
    for t in range(seq_len):
        # Sample random actions
        actions = types_np.sample(ac_space, bshape=(num_envs,))
        env.act(actions)
        rew, obs, first = env.observe()

        obs_tensor = torch.from_numpy(prev_obs).float() / 255.0
        next_obs_tensor = torch.from_numpy(obs["rgb"]) / 255.0

        obs_seq.append(obs_tensor.permute(0, 3, 1, 2))        # (B, C, H, W)
        next_obs_seq.append(next_obs_tensor.permute(0, 3, 1, 2))
        act_seq.append(torch.from_numpy(actions).long())
        rew_seq.append(torch.from_numpy(rew).float())
        end_seq.append(torch.from_numpy(first).float())

        prev_obs = obs["rgb"]

    # --- Convert sequences to tensors ---
    obs_seq_tensor = torch.stack(obs_seq).transpose(0, 1).to(device)         # (B, T, C, H, W)
    next_obs_tensor = torch.stack(next_obs_seq).transpose(0, 1).to(device)   # (B, T, C, H, W)
    act_tensor = torch.stack(act_seq).transpose(0, 1).to(device)             # (B, T)
    rew_tensor = torch.stack(rew_seq).transpose(0, 1).to(device)             # (B, T)
    end_tensor = torch.stack(end_seq).transpose(0, 1).to(device)             # (B, T)

    mask = torch.ones_like(rew_tensor, dtype=torch.bool)

    # Final obs: replace greyed-out frames (only if needed)
    info = [{"final_observation": next_obs_tensor[b, -1]} for b in range(num_envs)]

    # --- Forward through model ---
    logits_rew, logits_end, metrics = model.predict_rew_end(
        obs_seq_tensor,
        act_tensor,
        next_obs_tensor
    )

    # Losses (logits: B, T, *)
    logits_rew = logits_rew[mask]
    logits_end = logits_end[mask]

    target_rew = rew_tensor[mask].sign().long() + 1  # map [-1, 0, 1] to [0, 1, 2]
    target_end = end_tensor[mask].long()

    loss_rew = F.cross_entropy(logits_rew, target_rew)
    loss_end = F.cross_entropy(logits_end, target_end)
    loss = loss_rew + loss_end

    # --- Backprop ---
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    pbar.set_description(f"loss={loss.item():.4f} rew={loss_rew.item():.4f} end={loss_end.item():.4f}")

    # --- Log to wandb ---
    # wandb.log({wandb.log(metrics, step=update)})
    # wandb.log({
    #     "loss": loss.item(),
    #     "loss_rew": loss_rew.item(),
    #     "loss_end": loss_end.item(),
    #     "logits_rew": logits_rew.mean().item(),
    #     "logits_end": logits_end.mean().item(),
    #     "target_rew": target_rew.float().mean().item(),
    #     "target_end": target_end.float().mean().item(),
    # })

    if update % 100 == 0 or update == num_updates - 1:
        # wandb.log({
        #     "input_frame": [wandb.Image(obs_seq_tensor[0, -1].permute(1, 2, 0).cpu().numpy())],
        #     "next_frame": [wandb.Image(next_obs_tensor[0, -1].permute(1, 2, 0).cpu().numpy())],
        # })

        checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": update
        }
        torch.save(checkpoint, checkpoint_dir / f"rew_end_ckpt_{update:04d}.pt")

    # --- Clear buffers for next sequence ---
    obs_seq.clear()
    next_obs_seq.clear()
    act_seq.clear()
    rew_seq.clear()
    end_seq.clear()

# wandb.finish()