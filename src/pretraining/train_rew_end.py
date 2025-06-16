import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import wandb

from models.rew_end_model import RewEndModel, RewEndModelConfig


class CoinRunEpisodeDataset(Dataset):
    def __init__(self, episode_dir, fraction=0.1):
        all_files = sorted([os.path.join(episode_dir, f) for f in os.listdir(episode_dir) if f.endswith('.npz')])
        keep = int(len(all_files) * fraction)
        self.files = all_files[:keep]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        rgb = data["rgb"]             # (T, 64, 64, 3)
        acts = data["actions"]        # (T,)
        rews = data["reward"]         # (T,)
        dones = data["done"]          # (T,)

        obs = rgb[:-1]
        next_obs = rgb[1:]
        act = acts[:-1]
        rew = rews[:-1]
        done = dones[:-1]

        return {
            "obs": torch.tensor(obs).permute(0, 3, 1, 2) / 255.0,
            "next_obs": torch.tensor(next_obs).permute(0, 3, 1, 2) / 255.0,
            "action": torch.tensor(act),
            "reward": torch.tensor(rew),
            "done": torch.tensor(done),
        }


def run_epoch(model, dataloader, optim=None, device="cuda"):
    model.train() if optim else model.eval()
    total_loss, loss_rew_list, loss_end_list = [], [], []

    with torch.set_grad_enabled(optim is not None):
        for batch in dataloader:
            obs = batch["obs"].to(device)
            next_obs = batch["next_obs"].to(device)
            act = batch["action"].to(device)
            rew = batch["reward"].to(device)
            done = batch["done"].to(device)

            batch_struct = {
                "obs": obs,
                "act": act,
                "rew": rew,
                "end": done,
                "mask_padding": torch.ones_like(done, dtype=torch.bool),
                "info": [{}],
            }

            loss, logs = model(batch_struct)

            if optim:
                optim.zero_grad()
                loss.backward()
                optim.step()

            total_loss.append(loss.item())
            loss_rew_list.append(logs["loss_rew"].item())
            loss_end_list.append(logs["loss_end"].item())

    return {
        "loss_total": np.mean(total_loss),
        "loss_reward": np.mean(loss_rew_list),
        "loss_end": np.mean(loss_end_list),
    }


def train_with_validation(model, train_loader, val_loader, device="cuda", epochs=10):
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float("inf")
    for epoch in tqdm(range(epochs)):
        train_logs = run_epoch(model, train_loader, optim, device)
        val_logs = run_epoch(model, val_loader, None, device)

        wandb.log({
            "train/loss_total": train_logs["loss_total"],
            "train/loss_reward": train_logs["loss_reward"],
            "train/loss_end": train_logs["loss_end"],
            "val/loss_total": val_logs["loss_total"],
            "val/loss_reward": val_logs["loss_reward"],
            "val/loss_end": val_logs["loss_end"],
            "epoch": epoch + 1
        })

        print(f"[Epoch {epoch+1}] Train Loss: {train_logs['loss_total']:.4f} | Val Loss: {val_logs['loss_total']:.4f}")

        if val_logs["loss_total"] < best_val_loss:
            best_val_loss = val_logs["loss_total"]
            torch.save(model.state_dict(), "rew_end_model_best.pth")
            wandb.save("rew_end_model_best.pth")

    return model


def eval_model(model, eval_loader, device="cuda"):
    model.eval()
    eval_logs = run_epoch(model, eval_loader, None, device)

    wandb.log({
        "eval/loss_total": eval_logs["loss_total"],
        "eval/loss_reward": eval_logs["loss_reward"],
        "eval/loss_end": eval_logs["loss_end"],
    })
    print(f"[eval] Loss: {eval_logs['loss_total']:.4f}")


if __name__ == "__main__":
    wandb.init(project="world_models", name="DIAMOND_rew_end_coinrun")

    dataset = CoinRunEpisodeDataset("data/coinrun_episodes")
    n = len(dataset)
    train_set, val_set, eval_set = random_split(dataset, [int(0.7*n), int(0.15*n), n - int(0.7*n) - int(0.15*n)])
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1)
    eval_loader = DataLoader(eval_set, batch_size=1)

    config = RewEndModelConfig(
        lstm_dim=512,
        img_channels=3,
        img_size=64,
        cond_channels=32,
        depths=[2, 2, 2],
        channels=[32, 64, 128],
        attn_depths=[0, 0, 1],
        num_actions=15,
    )
    epochs = 10

    model = RewEndModel(config)
    model = train_with_validation(model, train_loader, val_loader, device="cuda", epochs=epochs)

    torch.save(model.state_dict(), f"rew_end_model_{epochs}.pth")

    eval_model(model, eval_loader, device="cuda")
    wandb.finish()