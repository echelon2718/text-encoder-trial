import torch
import random
import numpy as np
import matplotlib.pyplot as plt

def core_split(X, singlish_and_pn_ratio=0.1):
    eval_size = round(X * singlish_and_pn_ratio)
    test_size = eval_size
    train_size = X - eval_size - test_size
    return train_size, eval_size, test_size

def length_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)

def masked_mean(z: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).float()
    summed = (z * mask_f).sum(dim=-2)
    count = mask_f.sum(dim=-2).clamp(min=eps)
    return summed / count

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        "id": batch["id"],
        "x": batch["x"].to(device, non_blocking=True),
        "texts": batch["texts"],
        "x_lengths": batch["x_lengths"],
        "mask": batch["mask"].to(device, non_blocking=True),
    }

def compare_prior_and_canon_translation_visual(out, batch, view):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].set_title(f"Visualisasi z_{batch},{view}")
    axs[0].imshow(out['z_v'][batch, view].cpu().numpy())
    axs[0].set_xlabel("d_model hasil encoder (prior)")
    axs[0].set_ylabel("L hasil encoder (prior)")
    axs[0].set_ylim(256)
    axs[1].set_title(f"Visualisasi z_c,{batch},{view}")
    axs[1].imshow(out['z_v_canon'][batch, view].cpu().numpy())
    axs[1].set_xlabel("d_model decoder kanonik, expect sama dengan prior")
    axs[1].set_ylabel("L decoder kanonik, expect sama dengan view 0 prior")
    axs[1].set_ylim(256)
    print(out['z_v'][batch, view].shape, out['z_v_canon'][batch, view].shape)

def visualize_views_all(out):
    fig, axs = plt.subplots(out['z_v_canon'].shape[0], out['z_v_canon'].shape[1], figsize=(15, 15))
    for i in range(out['z_v_canon'].shape[0]):
        for j in range(out['z_v_canon'].shape[1]):
            axs[i, j].imshow(out['z_v_canon'][i][j].cpu().numpy())
            axs[i, j].set_title(f"Batch {i}, View {j}")
    plt.tight_layout()
    plt.show()

def visualize_views_across_batch(out, batch):
    fig, axs = plt.subplots(1, out['z_v_canon'].shape[1], figsize=(15, 5))
    for j in range(out['z_v_canon'].shape[1]):
        axs[j].imshow(out['z_v_canon'][batch][j].cpu().numpy())
        axs[j].set_title(f"Batch {batch}, View {j}")
    plt.tight_layout()
    plt.show()

