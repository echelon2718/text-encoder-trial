import torch
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from typing import Optional

from modules.losses import compute_losses, TLeJEPACriterion
from modules.utils import move_batch_to_device, masked_mean, core_split

try:
    from sklearn.manifold import TSNE
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

if _HAS_SKLEARN:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

def train_step(model, optimizer, batch, criterions, n_core, n_negation, device,
               use_amp=True, amp_dtype=torch.bfloat16, grad_clip: Optional[float] = 1.0,
               canon_type: str = "phoneme"):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, losses, out, _ = compute_losses(
        model, batch, criterions, n_core, n_negation, device,
        use_amp=use_amp, amp_dtype=amp_dtype, canon_type=canon_type
    )
    total.backward()

    grad_norm = None
    if grad_clip is not None and grad_clip > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    losses["total"] = total.detach()
    losses["grad_norm"] = grad_norm.detach() if grad_norm is not None else torch.tensor(0.0)
    losses["lr"] = torch.tensor(optimizer.param_groups[0]["lr"])
    return losses, out

@torch.no_grad()
def eval_step(model, batch, criterions, n_core, n_negation, device,
              use_amp=True, amp_dtype=torch.bfloat16, canon_type: str = "phoneme"):
    model.eval()
    total, losses, out, _ = compute_losses(
        model, batch, criterions, n_core, n_negation, device,
        use_amp=use_amp, amp_dtype=amp_dtype, canon_type=canon_type
    )
    losses["total"] = total.detach()
    return losses, out

def save_model(path, model, optimizer, epoch, best_metric, scheduler=None, extra: Optional[dict] = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
    }
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        ckpt["extra"] = extra
    torch.save(ckpt, path)
    return path

def load_model(path, model, optimizer=None, scheduler=None, map_location=None):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("best_metric", float("inf"))

class Trainer:
    def __init__(
        self,
        train_loader,
        val_loader,
        model,
        optimizer,
        criterion: TLeJEPACriterion,
        n_core: int,
        n_singlish: int,
        n_premise_and_negation: int,
        device: Optional[torch.device] = None,
        log_dir: str = "runs/tlejepa",
        ckpt_dir: str = "checkpoints",
        use_amp: bool = True,
        amp_dtype=torch.bfloat16,
        grad_clip: Optional[float] = 1.0,
        canon_type: str = "phoneme",
        log_every_n_steps: int = 20,
        visualize_every_n_epochs: int = 1,
        n_vis_projection_dirs: int = 3,
        n_vis_samples: int = 6,
        seed: int = 42,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion.to(self.device)

        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.grad_clip = grad_clip
        self.canon_type = canon_type
        self.log_every_n_steps = log_every_n_steps
        self.visualize_every_n_epochs = visualize_every_n_epochs
        self.n_vis_samples = n_vis_samples
        self.n_core = n_core
        self.n_singlish = n_singlish
        self.n_pneg = n_premise_and_negation

        self.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0
        self.best_val_loss = float("inf")

        self._vis_batch = next(iter(self.val_loader))

        g = torch.Generator().manual_seed(seed)
        d_model = self.model.d_model
        directions = torch.randn(d_model, n_vis_projection_dirs, generator=g)
        self._fixed_directions = (directions / directions.norm(dim=0, keepdim=True)).to(self.device)

    def train(self, epoch: int, lr_scheduler=None):
        self.model.train()
        running = {}
        n_batches = len(self.train_loader)
        pbar = tqdm(self.train_loader, total=n_batches, desc=f"Epoch {epoch} [train]",
                    colour="green", leave=False, dynamic_ncols=True)

        t_start = time.time()
        n_samples_seen = 0

        for step, batch in enumerate(pbar):
            try:
                losses, _ = train_step(
                    self.model, self.optimizer, batch, self.criterion, self.n_core, self.n_pneg,
                    self.device, use_amp=self.use_amp,
                    amp_dtype=self.amp_dtype, grad_clip=self.grad_clip, canon_type=self.canon_type,
                )
            except Exception as e:
                tqdm.write(f"[Trainer] step {self.global_step} gagal ({e!r}), skip batch ini.")
                self.optimizer.zero_grad(set_to_none=True)
                continue
            if lr_scheduler is not None:
                lr_scheduler.step()

            n_samples_seen += batch["x"].shape[0]
            elapsed = max(time.time() - t_start, 1e-8)
            samples_per_sec = n_samples_seen / elapsed

            for k, v in losses.items():
                running.setdefault(k, []).append(float(v))

            pbar.set_postfix({
                "Loss": f"{float(losses['total']):.4f}",
                "L_syn": f"{float(losses['syntactic']):.3f}",
                "L_sem": f"{float(losses['semantic']):.3f}",
                "L_SIGReg": f"{float(losses['sigreg']):.3f}",
                "L_len": f"{float(losses['canon_len']):.3f}",
                "lr": f"{float(losses['lr']):.2e}",
                "Samples/s": f"{samples_per_sec:.1f}",
            })

            if self.global_step % self.log_every_n_steps == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f"train_step/{k}", float(v), self.global_step)
                self.writer.add_scalar("perf/samples_per_sec", samples_per_sec, self.global_step)

            self.global_step += 1

        pbar.close()
        return {k: float(np.mean(v)) for k, v in running.items()}

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        running = {}
        n_batches = len(self.val_loader)
        pbar = tqdm(self.val_loader, total=n_batches, desc="Validating",
                    colour="cyan", leave=False, dynamic_ncols=True)

        for batch in pbar:
            losses, _ = eval_step(
                self.model, batch, self.criterion, self.n_core, self.n_pneg,
                self.device, use_amp=self.use_amp,
                amp_dtype=self.amp_dtype, canon_type=self.canon_type,
            )
            for k, v in losses.items():
                running.setdefault(k, []).append(float(v))
            pbar.set_postfix({"loss": f"{float(losses['total']):.4f}"})

        pbar.close()
        return {k: float(np.mean(v)) for k, v in running.items()}

    def log_to_tensorboard(self, epoch: int, train_loss: dict, val_loss: dict):
        for k in train_loss:
            if k in ("lr", "grad_norm"):
                continue
            self.writer.add_scalars(f"epoch/{k}", {
                "train": train_loss[k],
                "val": val_loss.get(k, float("nan")),
            }, epoch)
        if "lr" in train_loss:
            self.writer.add_scalar("epoch/lr", train_loss["lr"], epoch)
        if "grad_norm" in train_loss:
            self.writer.add_scalar("epoch/grad_norm", train_loss["grad_norm"], epoch)

        if epoch % self.visualize_every_n_epochs == 0:
            self._log_latent_geometry(epoch)
            self._log_gaussian_diagnostics(epoch)
            self._log_similarity_heatmaps(epoch)

    @torch.no_grad()
    def _forward_vis_batch(self):
        self.model.eval()
        batch = move_batch_to_device(self._vis_batch, self.device)
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                             enabled=self.use_amp and self.device.type == "cuda"):
            out = self.model.train_forward(batch, type=self.canon_type)
        return batch, out

    @torch.no_grad()
    def _log_latent_geometry(self, epoch: int):
        if not _HAS_SKLEARN:
            return
        batch, out = self._forward_vis_batch()

        z_v, masks = out["z_v"], out["masks"]
        z_c, masks_c = out["z_v_canon"], out["masks_v_canon"]

        pooled_prior = masked_mean(z_v, masks).float().cpu().numpy()
        pooled_canon = masked_mean(z_c, masks_c).float().cpu().numpy()

        B, V, d = pooled_prior.shape
        B_show = min(B, self.n_vis_samples)
        texts = batch["texts"]

        prior_flat = pooled_prior[:B_show].reshape(B_show * V, d)
        canon_flat = pooled_canon[:B_show].reshape(B_show * V, d)

        n_pts = B_show * V
        perplexity = max(2, min(30, (n_pts - 1) // 3))

        def project(flat):
            if n_pts <= 3:
                return PCA(n_components=2).fit_transform(flat)
            return TSNE(n_components=2, perplexity=perplexity, init="pca",
                        random_state=42).fit_transform(flat)

        proj_prior = project(prior_flat)
        proj_canon = project(canon_flat)

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        cmap = plt.get_cmap("tab10")
        for i in range(B_show):
            color = cmap(i % 10)
            idxs = [i * V + v for v in range(V)]
            axs[0].scatter(proj_prior[idxs[1:], 0], proj_prior[idxs[1:], 1],
                            color=color, marker="o", s=45, alpha=0.85,
                            label=f"#{i} non-kanonik" if i == 0 else None)
            axs[0].scatter(proj_prior[idxs[0], 0], proj_prior[idxs[0], 1],
                            color=color, marker="*", s=280, edgecolor="black", linewidth=0.8,
                            label=f"#{i} kanonik (anchor)" if i == 0 else None)

            axs[1].scatter(proj_canon[idxs[1:], 0], proj_canon[idxs[1:], 1],
                            color=color, marker="o", s=45, alpha=0.85)
            axs[1].scatter(proj_canon[idxs[0], 0], proj_canon[idxs[0], 1],
                            color=color, marker="*", s=280, edgecolor="black", linewidth=0.8)

        axs[0].set_title(f"Ruang Encoder f_theta (sebelum decode) — Epoch {epoch}")
        axs[1].set_title(f"Ruang Kanonik g_phi (setelah decode) — Epoch {epoch}")
        for ax in axs:
            ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
        axs[0].legend(loc="best", fontsize=8)
        plt.tight_layout()
        self.writer.add_figure("latent_geometry/tsne_prior_vs_canon", fig, global_step=epoch)
        plt.close(fig)

        legend_md = "| Idx | View | Teks |\n|---|---|---|\n"
        for i in range(B_show):
            for v in range(V):
                label = "**KANONIK (anchor)**" if v == 0 else f"non-kanonik #{v}"
                snippet = texts[i][v] if v < len(texts[i]) else ""
                legend_md += f"| {i} | {label} | {snippet} |\n"
        self.writer.add_text("latent_geometry/legend_teks", legend_md, global_step=epoch)

    @torch.no_grad()
    def _log_gaussian_diagnostics(self, epoch: int):
        batch, out = self._forward_vis_batch()
        z_v, masks = out["z_v"], out["masks"]
        pooled = masked_mean(z_v, masks)
        B, V, d = pooled.shape
        flat = pooled.reshape(B * V, d).float()

        proj = flat @ self._fixed_directions
        k = proj.shape[1]

        fig, axs = plt.subplots(1, k, figsize=(4.5 * k, 4))
        if k == 1:
            axs = [axs]
        x_axis = np.linspace(-4, 4, 200)
        gaussian_pdf = (1.0 / math.sqrt(2 * math.pi)) * np.exp(-x_axis ** 2 / 2)
        for i in range(k):
            vals = proj[:, i].detach().cpu().numpy()
            axs[i].hist(vals, bins=30, density=True, alpha=0.6, color="steelblue", label="Proyeksi empiris")
            axs[i].plot(x_axis, gaussian_pdf, color="crimson", linewidth=2, label="Target N(0,1)")
            axs[i].set_title(f"Arah proyeksi acak #{i+1}")
            axs[i].legend(fontsize=8)
        fig.suptitle(f"Distribusi proyeksi 1-D vs Gaussian standar (efek SIGReg) — Epoch {epoch}")
        plt.tight_layout()
        self.writer.add_figure("sigreg_diagnostics/histogram_vs_gaussian", fig, global_step=epoch)
        plt.close(fig)

        for i in range(k):
            self.writer.add_histogram(f"sigreg_diagnostics/projection_dim_{i}", proj[:, i], global_step=epoch)

        var_per_dim = flat.var(dim=0)
        self.writer.add_scalar("sigreg_diagnostics/mean_dim_variance", var_per_dim.mean().item(), epoch)
        self.writer.add_scalar("sigreg_diagnostics/isotropy_std_of_variance", var_per_dim.std().item(), epoch)

    @torch.no_grad()
    def _log_similarity_heatmaps(self, epoch: int):
        batch, out = self._forward_vis_batch()
        z_c, m_c = out["z_v_canon"], out["masks_v_canon"]
        B, V = z_c.shape[0], z_c.shape[1]

        teacher_sim = self.criterion.cossim_fn(batch, n_core=self.n_core, negation_offset=self.n_core - self.n_pneg, n_negation=self.n_pneg)

        pooled = masked_mean(z_c, m_c)
        pooled_norm = F.normalize(pooled, dim=-1)
        flat = pooled_norm.reshape(B * V, -1)
        sim_all = (flat @ flat.t()).view(B, V, B, V).permute(0, 2, 1, 3)
        model_sim = sim_all.mean(dim=(2, 3))

        diff = (model_sim - teacher_sim).abs()

        fig, axs = plt.subplots(1, 3, figsize=(17, 5))
        im0 = axs[0].imshow(teacher_sim.float().cpu().numpy(), cmap="viridis", vmin=-1, vmax=1)
        axs[0].set_title("Guru semantik (s_teacher)")
        plt.colorbar(im0, ax=axs[0], fraction=0.046)

        im1 = axs[1].imshow(model_sim.float().cpu().numpy(), cmap="viridis", vmin=-1, vmax=1)
        axs[1].set_title(f"Model (rata2 semua pasangan view) — Epoch {epoch}")
        plt.colorbar(im1, ax=axs[1], fraction=0.046)

        im2 = axs[2].imshow(diff.float().cpu().numpy(), cmap="inferno", vmin=0)
        axs[2].set_title("|Selisih| (makin gelap makin baik)")
        plt.colorbar(im2, ax=axs[2], fraction=0.046)

        plt.tight_layout()
        self.writer.add_figure("semantic_diagnostics/similarity_heatmaps", fig, global_step=epoch)
        plt.close(fig)
        self.writer.add_scalar("semantic_diagnostics/mean_abs_diff", diff.mean().item(), epoch)

        canon_texts = [t[0] for t in batch["texts"]]
        table = "| Idx | Kalimat kanonik |\n|---|---|\n" + "\n".join(
            f"| {i} | {txt} |" for i, txt in enumerate(canon_texts)
        )
        self.writer.add_text("semantic_diagnostics/legend_teks", table, global_step=epoch)

    def fit(self, num_epochs: int, lr_scheduler=None, start_epoch: int = 1,
            save_every_n_epochs: int = 1):
        epoch_bar = tqdm(range(start_epoch, start_epoch + num_epochs), desc="Total progress",
                          colour="magenta", dynamic_ncols=True)
        for epoch in epoch_bar:
            train_loss = self.train(epoch, lr_scheduler=lr_scheduler)
            val_loss = self.eval()
            self.log_to_tensorboard(epoch, train_loss, val_loss)

            improved = val_loss["total"] < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss["total"]
                save_model(os.path.join(self.ckpt_dir, "best_model.pt"),
                           self.model, self.optimizer, epoch, self.best_val_loss)

            if epoch % save_every_n_epochs == 0:
                save_model(os.path.join(self.ckpt_dir, "latest_model.pt"),
                           self.model, self.optimizer, epoch, self.best_val_loss)

            marker = "\u2605 BEST" if improved else ""
            tqdm.write(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss['total']:.4f} "
                f"(syn={train_loss['syntactic']:.3f} sem={train_loss['semantic']:.3f} "
                f"sig={train_loss['sigreg']:.3f} len={train_loss['canon_len']:.3f}) | "
                f"val_loss={val_loss['total']:.4f} {marker}"
            )
            epoch_bar.set_postfix({"best_val": f"{self.best_val_loss:.4f}"})

        epoch_bar.close()

    def close(self):
        self.writer.flush()
        self.writer.close()
