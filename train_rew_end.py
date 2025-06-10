import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from gym3 import types_np
from procgen import ProcgenGym3Env
from tqdm import tqdm
import numpy as np
import os
import wandb

from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from models.rew_end_model import RewEndModel, RewEndModelConfig
# seed = np.random.randint(0, 10000)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    torch.distributed.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    num_envs = 16 // world_size
    seq_len = 32
    num_updates = 1000
    learning_rate = 1e-4

    checkpoint_dir = "rew_end_ckpt"
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = ProcgenGym3Env(num=num_envs, env_name="coinrun", start_level=0, num_levels=0)
    ac_space = env.ac_space
    ob_space = env.ob_space

    model = RewEndModel(
        RewEndModelConfig(
            lstm_dim=512,
            img_channels=3,
            img_size=64,
            cond_channels=128,
            depths=[2, 2, 2],
            channels=[64, 128, 256],
            attn_depths=[0, 0, 1],
            num_actions=15
        )
    ).to(device)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    prev_obs = env.observe()[1]["rgb"]


    if rank == 0:
        wandb.init(
        project="world_models",
        config={
            "env": "coinrun",
            "seq_len": seq_len,
            "num_envs": num_envs,
            "learning_rate": learning_rate,
            "model": "RewEndModel",
            "updates": num_updates,
        }
        )

    pbar = range(num_updates) if rank == 0 else range(num_updates)
    for update in pbar:
        obs_seq, next_obs_seq, act_seq, rew_seq, end_seq = [], [], [], [], []

        for t in range(seq_len):
            actions = types_np.sample(ac_space, bshape=(num_envs,))
            env.act(actions)
            rew, obs, first = env.observe()

            obs_tensor = torch.from_numpy(prev_obs).float() / 255.0
            next_obs_tensor = torch.from_numpy(obs["rgb"]).float() / 255.0

            obs_seq.append(obs_tensor.permute(0, 3, 1, 2))
            next_obs_seq.append(next_obs_tensor.permute(0, 3, 1, 2))
            act_seq.append(torch.from_numpy(actions).long())
            rew_seq.append(torch.from_numpy(rew).float())
            end_seq.append(torch.from_numpy(first).float())

            prev_obs = obs["rgb"]

        obs_seq_tensor = torch.stack(obs_seq).transpose(0, 1).to(device)
        next_obs_tensor = torch.stack(next_obs_seq).transpose(0, 1).to(device)
        act_tensor = torch.stack(act_seq).transpose(0, 1).to(device)
        rew_tensor = torch.stack(rew_seq).transpose(0, 1).to(device)
        end_tensor = torch.stack(end_seq).transpose(0, 1).to(device)

        mask = torch.ones_like(rew_tensor, dtype=torch.bool)

        logits_rew, logits_end, _ = model.module.predict_rew_end(
            obs_seq_tensor, act_tensor, next_obs_tensor
        )

        logits_rew = logits_rew[mask]
        logits_end = logits_end[mask]

        target_rew = rew_tensor[mask].sign().long() + 1
        target_end = end_tensor[mask].long()

        loss_rew = F.cross_entropy(logits_rew, target_rew)
        loss_end = F.cross_entropy(logits_end, target_end)
        loss = loss_rew + loss_end

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if rank == 0:
            wandb_metrics = {
                "loss_reward": loss_rew.item(),
                "loss_end": loss_end.item(),
                "loss_total": loss.item()}
            wandb.log(wandb_metrics, step=update)

            print(f"Update {update}, loss: {loss.item():.4f}")

            wandb.log({
                "input_frame": [wandb.Image(obs_seq_tensor[0, -1].permute(1, 2, 0).cpu().numpy())],
                "next_frame": [wandb.Image(next_obs_tensor[0, -1].permute(1, 2, 0).cpu().numpy())],
            })

        if update % 100 == 0 or update == num_updates - 1:
            if rank == 0:
                checkpoint = {
                    "model_state_dict": model.module.state_dict(),  # remove 'module.' prefix
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": update
                }
                checkpoint_path = os.path.join(checkpoint_dir, f"rew_end_ckpt_{update:04d}.pt")
                torch.save(checkpoint, checkpoint_path)


    cleanup()
    if rank==0:
        wandb.finish()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size)