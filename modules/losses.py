import torch
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import time
import uuid
import datetime
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


# Format nama folder run: "<label>-<random_id 8 hex>-<HH-MM-SS>_<DD-MM-YYYY>"
# random_id selalu 8 karakter hex -> dipakai sebagai jangkar parsing supaya
# label yang mengandung tanda "-" (mis. "no-sigreg") tetap ter-parse benar.
_RUN_DIR_PATTERN = re.compile(
    r"^(?P<label>.+)-(?P<random_id>[0-9a-f]{8})-(?P<ts>\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{4})$"
)


def find_latest_run_dir(ckpt_root: str, label: str) -> Optional[str]:
    """
    Cari folder checkpoint TERBARU untuk `label` tertentu di dalam `ckpt_root`.
    Timestamp di-parse jadi datetime asli (bukan string-sort) supaya urutan
    tanggal/bulan berbeda tetap benar -- format HH-MM-SS_DD-MM-YYYY tidak
    terurut benar kalau cuma dibandingkan sebagai string lintas tanggal.
    Return None kalau tidak ada folder yang cocok.
    """
    if not os.path.isdir(ckpt_root):
        return None

    candidates = []
    for name in os.listdir(ckpt_root):
        full_path = os.path.join(ckpt_root, name)
        if not os.path.isdir(full_path):
            continue
        m = _RUN_DIR_PATTERN.match(name)
        if not m or m.group("label") != label:
            continue
        try:
            ts = datetime.datetime.strptime(m.group("ts"), "%H-%M-%S_%d-%m-%Y")
        except ValueError:
            continue
        candidates.append((ts, full_path))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]

def train_step(model, optimizer, batch, criterions, n_core, n_negation, device,
               use_amp=True, amp_dtype=torch.bfloat16, grad_clip: Optional[float] = 1.0,
               canon_type: str = "phoneme"):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, losses, out, _ = compute_losses(
        model, batch, criterions, n_core, n_negation, device,
        use_amp=use_amp, amp_dtype=amp_dtype, canon_type=canon_type
    )

    if not torch.isfinite(total):
        # Loss NaN/Inf TIDAK boleh di-backward -- optimizer.step() akan
        # menyuntikkan NaN ke bobot model SECARA PERMANEN kalau dipaksa jalan.
        losses["total"] = total.detach()
        losses["grad_norm"] = torch.tensor(0.0)
        losses["lr"] = torch.tensor(optimizer.param_groups[0]["lr"])
        losses["_skipped"] = True
        return losses, out

    total.backward()

    grad_norm = None
    if grad_clip is not None and grad_clip > 0:
        # proj_head IKUT di-clip -- sebelumnya cuma model.parameters(),
        # padahal proj_head ikut dioptimasi optimizer yang sama (lihat
        # train_with_lr_sched.py) dan menerima gradien MSE tak-scale-invariant
        # dari magnitude_loss.
        clip_params = list(model.parameters())
        if criterions is not None and hasattr(criterions, "proj_head"):
            clip_params += list(criterions.proj_head.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)

    if grad_norm is not None and not torch.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        losses["total"] = total.detach()
        losses["grad_norm"] = grad_norm.detach()
        losses["lr"] = torch.tensor(optimizer.param_groups[0]["lr"])
        losses["_skipped"] = True
        return losses, out

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

def save_model(path, model, optimizer, epoch, best_metric, scheduler=None,
                global_step: Optional[int] = None, extra: Optional[dict] = None, criterion = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
    }
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()

    if criterion is not None and hasattr(criterion, "proj_head"):
        ckpt["criterion_extra_state_dict"] = {
            "proj_head": criterion.proj_head.state_dict(),
        }

    merged_extra = dict(extra or {})
    if global_step is not None:
        merged_extra["global_step"] = global_step
    if merged_extra:
        ckpt["extra"] = merged_extra
    torch.save(ckpt, path)
    return path

def load_model(path, model, optimizer=None, scheduler=None, map_location=None, criterion = None):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if criterion is not None and hasattr(criterion, "proj_head"):
        extra_state = ckpt.get("criterion_extra_state_dict", {})
        if "proj_head" in extra_state:
            criterion.proj_head.load_state_dict(extra_state["proj_head"])
    global_step = ckpt.get("extra", {}).get("global_step", 0)
    return ckpt.get("epoch", 0), ckpt.get("best_metric", float("inf")), global_step

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
        save_every_n_steps: int = 1000,
        visualize_every_n_epochs: int = 1,
        visualize_every_n_steps: Optional[int] = 1000,
        n_vis_projection_dirs: int = 3,
        n_vis_samples: int = 6,
        seed: int = 42,
        label: str = "run",
        resume_dir: Optional[str] = None,
        lr_scheduler=None,
        lambda_warmup_steps: Optional[int] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label = label
        # Nilai target lambda_ (bobot SIGReg) diambil dari criterion SEBELUM
        # diapa-apakan -- selama warmup, criterion.lambda_ akan dinaikkan
        # bertahap dari ~0 menuju nilai target ini, supaya sinyal korespondensi
        # (syntactic/semantic) sempat "menang" dulu di awal training sebelum
        # SIGReg (yang TERBUKTI buta thd korespondensi -- lihat losses.py)
        # mulai ikut dominan menarik arah gradien.
        self._lambda_target = criterion.lambda_
        self.lambda_warmup_steps = lambda_warmup_steps

        if resume_dir is not None:
            # --- Lanjutkan run yang sudah ada -- JANGAN bikin folder baru ------
            # Supaya checkpoint & log TensorBoard nyambung di tempat yang sama,
            # bukan folder baru dengan random_id/timestamp baru.
            self._run_id_safe = os.path.basename(os.path.normpath(resume_dir))
            self.run_id = self._run_id_safe
            print(f"[Trainer] Resume dari direktori: {resume_dir}")
        else:
            # --- Identitas unik untuk eksekusi baru --------------------------
            # Supaya beberapa run paralel (parameter beda) tidak saling overwrite
            # checkpoint/log satu sama lain: "<label>-<random_id>-<HH:mm:SS DD/MM/YYYY>"
            self.random_id = uuid.uuid4().hex[:8]
            self._run_timestamp = datetime.datetime.now().strftime("%H:%M:%S %d/%m/%Y")
            self.run_id = f"{self.label}-{self.random_id}-{self._run_timestamp}"
            print(f"Executing training ID: {self.random_id}")
            # ':' dan '/' tidak valid sebagai nama file/folder di banyak filesystem
            # (khususnya Windows), jadi dipakai versi "aman" khusus untuk path.
            self._run_id_safe = (
                self.run_id.replace("/", "-").replace(":", "-").replace(" ", "_")
            )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion.to(self.device)
        # Disimpan di sini (bukan cuma parameter fit()) supaya state scheduler
        # bisa ikut di-load saat resume, dan ikut di-checkpoint tiap save.
        self.lr_scheduler = lr_scheduler

        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.grad_clip = grad_clip
        self.canon_type = canon_type
        self.log_every_n_steps = log_every_n_steps
        self.save_every_n_steps = save_every_n_steps
        self.visualize_every_n_epochs = visualize_every_n_epochs
        self.visualize_every_n_steps = visualize_every_n_steps
        self.n_vis_samples = n_vis_samples
        self.n_core = n_core
        self.n_singlish = n_singlish
        self.n_pneg = n_premise_and_negation

        # Checkpoint tiap run disimpan di subfolder sendiri: <ckpt_dir>/<run_id>/...
        self.ckpt_dir = resume_dir if resume_dir is not None else os.path.join(ckpt_dir, self._run_id_safe)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Log TensorBoard juga dipisah per run -- kalau resume, nama dir sama
        # persis dengan sebelumnya, jadi TensorBoard menyambung di grafik yang
        # sama (bukan grafik baru terpisah).
        run_log_dir = os.path.join(log_dir, self._run_id_safe)
        self.writer = SummaryWriter(log_dir=run_log_dir)

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.resume_epoch = 1  # dipakai sbg default start_epoch di fit()

        if resume_dir is not None:
            self.resume_epoch = self._load_resume_checkpoint(resume_dir)
        else:
            self.writer.add_text("run_info/run_id", self.run_id, global_step=0)

        self._vis_batch = next(iter(self.val_loader))

        g = torch.Generator().manual_seed(seed)
        d_model = self.model.d_model
        directions = torch.randn(d_model, n_vis_projection_dirs, generator=g)
        self._fixed_directions = (directions / directions.norm(dim=0, keepdim=True)).to(self.device)

    def _load_resume_checkpoint(self, resume_dir: str) -> int:
        """
        Prioritas file yang dimuat:
          1. latest_step.pt  -- checkpoint per-1000-step, termasuk progres
             mid-epoch kalau training crash di tengah jalan.
          2. latest_model.pt -- akhir epoch terakhir yang selesai PENUH.
          3. best_model.pt   -- fallback terakhir kalau dua di atas tidak ada.

        Return start_epoch yang seharusnya dipakai:
          - dari latest_step.pt -> epoch YANG SAMA (redo dari awal epoch itu;
            kita tidak menyimpan posisi persis di tengah epoch, jadi paling
            aman ulang epoch itu dari awal dengan bobot model yang sudah ada)
          - dari latest_model.pt / best_model.pt -> epoch + 1 (epoch itu
            sudah selesai penuh, lanjut ke epoch berikutnya)
        """
        step_ckpt = os.path.join(resume_dir, "latest_step.pt")
        epoch_ckpt = os.path.join(resume_dir, "latest_model.pt")
        best_ckpt = os.path.join(resume_dir, "best_model.pt")

        if os.path.exists(step_ckpt):
            path, redo_same_epoch = step_ckpt, True
        elif os.path.exists(epoch_ckpt):
            path, redo_same_epoch = epoch_ckpt, False
        elif os.path.exists(best_ckpt):
            path, redo_same_epoch = best_ckpt, False
        else:
            raise FileNotFoundError(f"Tidak ada checkpoint (.pt) ditemukan di {resume_dir}")

        epoch, best_metric, global_step = load_model(
            path, self.model, optimizer=self.optimizer,
            scheduler=self.lr_scheduler, map_location=self.device,
            criterion=self.criterion,
        )
        self.best_val_loss = best_metric
        self.global_step = global_step
        print(
            f"[Trainer] Dimuat dari {path}: epoch={epoch}, global_step={global_step}, "
            f"best_val_loss={best_metric:.4f}"
        )
        self.writer.add_text(
            "run_info/resumed_from",
            f"Resumed from `{os.path.basename(path)}` at epoch={epoch}, global_step={global_step}",
            global_step=global_step,
        )
        return epoch if redo_same_epoch else epoch + 1

    def train(self, epoch: int, lr_scheduler=None):
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler  # utk kompatibilitas panggilan lama

        self.model.train()
        running = {}
        n_batches = len(self.train_loader)
        pbar = tqdm(self.train_loader, total=n_batches, desc=f"Epoch {epoch} [train]",
                    colour="green", leave=False, dynamic_ncols=True)

        t_start = time.time()
        n_samples_seen = 0

        for step, batch in enumerate(pbar):
            if self.lambda_warmup_steps:
                # Linear warmup: lambda_ mulai dari ~0 di global_step=0, naik
                # linear sampai self._lambda_target persis di step ke-N.
                # SETELAH warmup selesai, lambda_ konstan di nilai target
                # (perilaku sama seperti sebelum patch ini kalau
                # lambda_warmup_steps tidak di-set / None).
                frac = min(1.0, self.global_step / self.lambda_warmup_steps)
                self.criterion.lambda_ = frac * self._lambda_target
            try:
                losses, _ = train_step(
                    self.model, self.optimizer, batch, self.criterion, self.n_core, self.n_pneg,
                    self.device, use_amp=self.use_amp,
                    amp_dtype=self.amp_dtype, grad_clip=self.grad_clip, canon_type=self.canon_type,
                )
            except RuntimeError as e:
                is_oom = "out of memory" in str(e).lower()
                self.optimizer.zero_grad(set_to_none=True)
                if is_oom and torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(self.device) / 1e9
                    reserved = torch.cuda.memory_reserved(self.device) / 1e9
                    tqdm.write(
                        f"[Trainer] OOM di step {self.global_step} "
                        f"(allocated={allocated:.2f}GB reserved={reserved:.2f}GB). "
                        f"Membersihkan cache & skip batch ini."
                    )
                    del e
                    torch.cuda.empty_cache()
                else:
                    tqdm.write(f"[Trainer] RuntimeError di step {self.global_step} ({e!r}), skip batch ini.")
                continue
            except Exception as e:
                tqdm.write(f"[Trainer] step {self.global_step} gagal ({e!r}), skip batch ini.")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            if losses.get("_skipped"):
                tqdm.write(
                    f"[Trainer] Loss/gradien NaN atau Inf di step {self.global_step} -- "
                    f"batch DIBUANG, bobot model TIDAK diupdate."
                )
                continue

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            n_samples_seen += batch["x"].shape[0]
            elapsed = max(time.time() - t_start, 1e-8)
            samples_per_sec = n_samples_seen / elapsed

            for k, v in losses.items():
                running.setdefault(k, []).append(float(v))

            pbar.set_postfix({
                "Loss": f"{float(losses['total']):.4f}",
                "L_syn": f"{float(losses['syntactic']):.3f}",
                "L_sem": f"{float(losses['semantic']):.3f}",
                "L_mag": f"{float(losses['magnitude']):.3f}",
                "L_SIGReg": f"{float(losses['sigreg']):.3f}",
                "L_len": f"{float(losses['canon_len']):.3f}",
                "lambda": f"{self.criterion.lambda_:.3f}",
                "lr": f"{float(losses['lr']):.2e}",
                "Samples/s": f"{samples_per_sec:.1f}",
            })

            if self.global_step % self.log_every_n_steps == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f"train_step/{k}", float(v), self.global_step)
                self.writer.add_scalar("perf/samples_per_sec", samples_per_sec, self.global_step)
                self.writer.add_scalar("train_step/lambda_current", self.criterion.lambda_, self.global_step)

            self.global_step += 1

            # --- Diagnostik visual periodik (BUKAN cuma di akhir epoch) -------
            # Epoch yang butuh berjam-jam/berhari-hari berarti "akhir epoch"
            # bisa jadi TIDAK PERNAH tercapai (crash/OOM/keburu deadline duluan).
            # Jadi diagnostik ini juga dipicu tiap N step, supaya tetap ada
            # sinyal visual di TensorBoard walau training belum lewat 1 epoch.
            if self.visualize_every_n_steps and self.global_step % self.visualize_every_n_steps == 0:
                self._log_all_diagnostics(self.global_step)

            # --- Checkpoint per-iterasi ---------------------------------------
            # Simpan setiap `save_every_n_steps` step (default 1000), terpisah
            # dari checkpoint akhir-epoch. Nama file di-overwrite (bukan
            # per-step unik) supaya tidak membengkak; ini murni jaring pengaman
            # kalau training crash/disconnect di tengah epoch yang panjang.
            if self.save_every_n_steps and self.global_step % self.save_every_n_steps == 0:
                save_model(
                    os.path.join(self.ckpt_dir, "latest_step.pt"),
                    self.model, self.optimizer, epoch, self.best_val_loss,
                    scheduler=self.lr_scheduler,
                    global_step=self.global_step,
                    extra={"run_id": self.run_id},
                    criterion=self.criterion,
                )
                tqdm.write(f"[Trainer] Checkpoint tersimpan di step {self.global_step} (epoch {epoch}).")

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
            self._log_all_diagnostics(self.global_step)

    def _log_all_diagnostics(self, step: int):
        """
        Jalankan ketiga visualisasi diagnostik (t-SNE latent geometry,
        histogram Gaussian SIGReg, heatmap similarity semantik) sekali panggil.

        Dipakai dari DUA tempat: (1) periodik tiap `visualize_every_n_steps`
        di dalam train(), dan (2) di akhir epoch lewat log_to_tensorboard().
        Keduanya pakai `global_step` (BUKAN nomor epoch) sebagai sumbu-x
        TensorBoard, supaya titik-titik dari dua sumber ini nyambung jadi
        satu kurva progres yang sama, bukan saling menimpa di titik yang sama.

        Dibungkus try/except: kalau salah satu visualisasi gagal (mis. TSNE
        error karena batch vis kebetulan terlalu kecil), training TIDAK ikut
        crash -- cuma dicatat & dilewati, karena kegagalan berikut sudah
        cukup mahal (proses ini bisa jalan berjam-jam).
        """
        try:
            self._log_latent_geometry(step)
        except Exception as e:
            tqdm.write(f"[Trainer] _log_latent_geometry gagal di step {step}: {e!r}")
        try:
            self._log_gaussian_diagnostics(step)
        except Exception as e:
            tqdm.write(f"[Trainer] _log_gaussian_diagnostics gagal di step {step}: {e!r}")
        try:
            self._log_similarity_heatmaps(step)
        except Exception as e:
            tqdm.write(f"[Trainer] _log_similarity_heatmaps gagal di step {step}: {e!r}")

    @torch.no_grad()
    def _forward_vis_batch(self):
        self.model.eval()
        batch = move_batch_to_device(self._vis_batch, self.device)
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                             enabled=self.use_amp and self.device.type == "cuda"):
            out = self.model.train_forward(batch, type=self.canon_type)
        return batch, out

    @torch.no_grad()
    def _log_latent_geometry(self, step: int):
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

        axs[0].set_title(f"Ruang Encoder f_theta (sebelum decode) — Step {step}")
        axs[1].set_title(f"Ruang Kanonik g_phi (setelah decode) — Step {step}")
        for ax in axs:
            ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
        axs[0].legend(loc="best", fontsize=8)
        plt.tight_layout()
        self.writer.add_figure("latent_geometry/tsne_prior_vs_canon", fig, global_step=step)
        plt.close(fig)

        legend_md = "| Idx | View | Teks |\n|---|---|---|\n"
        for i in range(B_show):
            for v in range(V):
                label = "**KANONIK (anchor)**" if v == 0 else f"non-kanonik #{v}"
                snippet = texts[i][v] if v < len(texts[i]) else ""
                legend_md += f"| {i} | {label} | {snippet} |\n"
        self.writer.add_text("latent_geometry/legend_teks", legend_md, global_step=step)

    @torch.no_grad()
    def _log_gaussian_diagnostics(self, step: int):
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
        fig.suptitle(f"Distribusi proyeksi 1-D vs Gaussian standar (efek SIGReg) — Step {step}")
        plt.tight_layout()
        self.writer.add_figure("sigreg_diagnostics/histogram_vs_gaussian", fig, global_step=step)
        plt.close(fig)

        for i in range(k):
            self.writer.add_histogram(f"sigreg_diagnostics/projection_dim_{i}", proj[:, i], global_step=step)

        var_per_dim = flat.var(dim=0)
        self.writer.add_scalar("sigreg_diagnostics/mean_dim_variance", var_per_dim.mean().item(), step)
        self.writer.add_scalar("sigreg_diagnostics/isotropy_std_of_variance", var_per_dim.std().item(), step)

    @torch.no_grad()
    def _log_similarity_heatmaps(self, step: int):
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
        axs[1].set_title(f"Model (rata2 semua pasangan view) — Step {step}")
        plt.colorbar(im1, ax=axs[1], fraction=0.046)

        im2 = axs[2].imshow(diff.float().cpu().numpy(), cmap="inferno", vmin=0)
        axs[2].set_title("|Selisih| (makin gelap makin baik)")
        plt.colorbar(im2, ax=axs[2], fraction=0.046)

        plt.tight_layout()
        self.writer.add_figure("semantic_diagnostics/similarity_heatmaps", fig, global_step=step)
        plt.close(fig)
        self.writer.add_scalar("semantic_diagnostics/mean_abs_diff", diff.mean().item(), step)

        canon_texts = [t[0] for t in batch["texts"]]
        table = "| Idx | Kalimat kanonik |\n|---|---|\n" + "\n".join(
            f"| {i} | {txt} |" for i, txt in enumerate(canon_texts)
        )
        self.writer.add_text("semantic_diagnostics/legend_teks", table, global_step=step)

    def fit(self, num_epochs: int, lr_scheduler=None, start_epoch: Optional[int] = None,
            save_every_n_epochs: int = 1):
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler
        if start_epoch is None:
            # Otomatis lanjut dari epoch yang benar kalau resume_dir dipakai
            # di konstruktor; kalau run baru, self.resume_epoch == 1.
            start_epoch = self.resume_epoch

        epoch_bar = tqdm(range(start_epoch, start_epoch + num_epochs), desc="Total progress",
                          colour="magenta", dynamic_ncols=True)
        for epoch in epoch_bar:
            train_loss = self.train(epoch)
            val_loss = self.eval()
            self.log_to_tensorboard(epoch, train_loss, val_loss)

            improved = val_loss["total"] < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss["total"]
                save_model(os.path.join(self.ckpt_dir, "best_model.pt"),
                           self.model, self.optimizer, epoch, self.best_val_loss,
                           scheduler=self.lr_scheduler, global_step=self.global_step,
                           extra={"run_id": self.run_id}, criterion=self.criterion)

            # Selalu simpan checkpoint di akhir tiap epoch (tidak lagi
            # bergantung pada save_every_n_epochs) supaya training bisa
            # di-resume dari epoch manapun kalau tiba-tiba crash/disconnect.
            save_model(os.path.join(self.ckpt_dir, "latest_model.pt"),
                       self.model, self.optimizer, epoch, self.best_val_loss,
                       scheduler=self.lr_scheduler, global_step=self.global_step,
                       extra={"run_id": self.run_id}, criterion=self.criterion)

            marker = "\u2605 BEST" if improved else ""
            tqdm.write(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss['total']:.4f} "
                f"(syn={train_loss['syntactic']:.3f} sem={train_loss['semantic']:.3f} "
                f"mag={train_loss['magnitude']:.3f} "
                f"sig={train_loss['sigreg']:.3f} len={train_loss['canon_len']:.3f}) | "
                f"val_loss={val_loss['total']:.4f} {marker}"
            )
            epoch_bar.set_postfix({"best_val": f"{self.best_val_loss:.4f}"})

        epoch_bar.close()

    def close(self):
        self.writer.flush()
        self.writer.close()
i0002672@xlogin2:~$ cat text-encoder-trial/modules/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

from modules.utils import move_batch_to_device

def canon_len_loss(out: dict) -> torch.Tensor:
    l_v_preds = out['l_v_preds']
    l_v_gts = out['l_v_gts'].detach()
    loss = F.mse_loss(torch.log1p(l_v_preds), torch.log1p(l_v_gts))
    return loss


def syntactical_loss(out: dict) -> torch.Tensor:
    z_v = out["z_v"]
    z_canon = out["z_v_canon"]
    masks = out["masks_v_canon"]

    B, V = z_canon.shape[0], z_canon.shape[1]
    L = min(z_v.size(2), z_canon.size(2), masks.size(2))

    mask = masks[:, 0, :L].unsqueeze(-1).float()

    z_v_trunc = z_v[:, :, :L, :] * mask.unsqueeze(1)
    mu = z_v_trunc.sum(dim=1) / V

    denom = (mask.sum() * z_canon.size(-1) * V).clamp(min=1.0)

    loss = 0.0
    for v in range(V):
        z_canon_v = z_canon[:, v, :L, :] * mask
        loss = loss + F.mse_loss(z_canon_v, mu, reduction='sum')

    return loss / denom

def masked_mean(z: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).float()
    summed = (z * mask_f).sum(dim=-2)
    count = mask_f.sum(dim=-2).clamp(min=eps)
    return summed / count

class SemanticTeacherSimilarity(nn.Module):
    def __init__(self, teacher_model: str = "tum-nlp/NegMPNet"):
        super().__init__()
        self.teacher_semantic_model = SentenceTransformer(teacher_model).eval()

    @torch.no_grad()
    def _get_raw(self, batch: dict) -> torch.Tensor:
        canonical_texts = [t[0] for t in batch['texts']]
        emb = self.teacher_semantic_model.encode(canonical_texts, convert_to_tensor=True)
        return emb

    def forward(
        self,
        batch: dict,
        n_core: int = 64,
        negation_offset: int = 62,
        n_negation: int = 2,
    ) -> torch.Tensor:
        emb = F.normalize(self._get_raw(batch), dim=-1, eps=1e-8)

        emb_A = emb[:n_core]
        emb_B = torch.cat(
            [emb[:negation_offset], emb[n_core:n_core + n_negation]],
            dim=0,
        )
        return emb_A @ emb_B.t()

    def get_raw_embeddings(self, batch: dict) -> torch.Tensor:
        """Dipakai magnitude_loss -- embedding teacher mentah, (B_total, d_teacher)."""
        return self._get_raw(batch)


class CachedSemanticTeacherSimilarity(nn.Module):
    def __init__(self, base_teacher: nn.Module, cache_size: int = 500_000):
        super().__init__()
        self.base_teacher = base_teacher
        for p in self.base_teacher.parameters():
            p.requires_grad_(False)
        self.base_teacher.eval()
        self._cache: dict = {}
        self._cache_size = cache_size

    def train(self, mode: bool = True):
        self.base_teacher.eval()
        return self

    @torch.no_grad()
    def _get_raw(self, batch) -> torch.Tensor:
        """
        Cache sekarang menyimpan embedding MENTAH (belum dinormalisasi) --
        supaya forward() (cosine-sim, normalisasi di sini) dan
        get_raw_embeddings() (magnitude_loss, butuh skala asli) sama-sama
        dilayani dari SATU encode() saja, tidak encode dua kali per teks.
        """
        ids = batch["id"]
        texts = [t[0] for t in batch["texts"]]
        device = (
            next(self.base_teacher.teacher_semantic_model.parameters(), torch.tensor(0.0)).device
            if hasattr(self.base_teacher.teacher_semantic_model, "parameters")
            else torch.device("cpu")
        )

        missing_idx = [i for i, _id in enumerate(ids) if _id not in self._cache]
        local_lookup = {}
        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            new_embs = self.base_teacher.teacher_semantic_model.encode(missing_texts, convert_to_tensor=True)
            for local_i, global_i in enumerate(missing_idx):
                emb_i = new_embs[local_i].detach().cpu()
                local_lookup[ids[global_i]] = emb_i
                if len(self._cache) < self._cache_size:
                    self._cache[ids[global_i]] = emb_i

        emb = torch.stack([
            self._cache.get(_id, local_lookup.get(_id)) for _id in ids
        ]).to(device)
        return emb

    @torch.no_grad()
    def forward(self, batch, n_core=64, negation_offset=62, n_negation=2):
        emb = F.normalize(self._get_raw(batch), dim=-1, eps=1e-8)
        emb_A = emb[:n_core]
        emb_B = torch.cat([emb[:negation_offset], emb[n_core:n_core + n_negation]], dim=0)
        return emb_A @ emb_B.t()

    @torch.no_grad()
    def get_raw_embeddings(self, batch) -> torch.Tensor:
        """Dipakai magnitude_loss -- embedding teacher mentah, (B_total, d_teacher)."""
        return self._get_raw(batch)

    def cache_stats(self) -> dict:
        return {"cached_sentences": len(self._cache)}


def semantic_loss(
    batch: dict,
    out: dict,
    cossim_fn,
    eps: float = 1e-6,
    n_core: int = 64,
    negation_offset: int = 62,
    n_negation: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_c = out['z_v_canon']
    m_c = out['masks_v_canon']

    pooled = masked_mean(z_c, m_c, eps=eps)
    pooled_norm = F.normalize(pooled, dim=-1, eps=eps)

    V, d = pooled_norm.shape[1], pooled_norm.shape[2]

    pooled_A = pooled_norm[:n_core]
    pooled_B = torch.cat(
        [pooled_norm[:negation_offset], pooled_norm[n_core:n_core + n_negation]],
        dim=0,
    )

    flat_A = pooled_A.reshape(n_core * V, d)
    flat_B = pooled_B.reshape(n_core * V, d)

    sim_all = flat_A @ flat_B.t()
    sim_all = sim_all.view(n_core, V, n_core, V).permute(0, 2, 1, 3)

    teacher_sim = cossim_fn(batch, n_core=n_core, negation_offset=negation_offset, n_negation=n_negation)
    assert teacher_sim.shape == (n_core, n_core), (
        f"teacher_sim harus ({n_core},{n_core}), didapat {tuple(teacher_sim.shape)}"
    )
    teacher_sim = teacher_sim.to(sim_all.device)

    D = sim_all - teacher_sim.view(n_core, n_core, 1, 1)
    delta = D.pow(2).sum(dim=(-2, -1))

    neg_block_mask = torch.zeros(n_core, n_core, dtype=torch.bool, device=delta.device)
    if negation_offset < n_core:
        neg_block_mask[negation_offset:n_core, negation_offset:n_core] = True

    n_gen_entries = (~neg_block_mask).sum().clamp(min=1)
    n_neg_entries = neg_block_mask.sum().clamp(min=1)

    l_sem_general = delta[~neg_block_mask].sum() / n_gen_entries
    l_sem_negation = delta[neg_block_mask].sum() / n_neg_entries

    l_sem_negation_contrast = torch.zeros((), device=delta.device, dtype=delta.dtype)
    if negation_offset < n_core and n_core > 1:
        sim_canon = sim_all[:, :, 0, 0]
        premise_rows = torch.arange(negation_offset, n_core, device=delta.device)
        denom = float(n_core - 1)

        teacher_diag = teacher_sim[premise_rows, premise_rows]
        teacher_baseline = (teacher_sim[premise_rows].sum(dim=1) - teacher_diag) / denom
        teacher_contrast = (teacher_diag - teacher_baseline).detach()

        student_diag = sim_canon[premise_rows, premise_rows]
        student_baseline = (sim_canon[premise_rows].sum(dim=1) - student_diag) / denom
        student_contrast = student_diag - student_baseline

        l_sem_negation_contrast = F.mse_loss(student_contrast, teacher_contrast)

    return l_sem_general, l_sem_negation, l_sem_negation_contrast


class TeacherMagnitudeProjection(nn.Module):
    """
    Proyeksi linear student (d_model) -> ruang embedding teacher (d_teacher).
    Modul ini SCAFFOLDING MURNI untuk magnitude_loss:
      - Parameternya WAJIB didaftarkan ke optimizer yang sama dengan model
        utama (lihat contoh wiring di bawah). Kalau tidak, dia diam di
        inisialisasi acak dan magnitude_loss justru MERUSAK training --
        mendorong z_v_canon mengejar output acak yang tidak berarti.
      - Modul ini BUKAN bagian dari TLeJEPA dan tidak dipakai saat inference.
        Setelah training selesai, buang saja state_dict-nya (jangan ikut
        di-load ke model final).
    """
    def __init__(self, d_model: int, d_teacher: int = 768):
        super().__init__()
        self.proj = nn.Linear(d_model, d_teacher)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def magnitude_loss(
    batch: dict,
    out: dict,
    cossim_fn,
    proj_head: "TeacherMagnitudeProjection",
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Regresi MSE langsung per-kalimat (bukan matriks N x N seperti
    semantic_loss) antara embedding teacher MENTAH (belum dinormalisasi --
    magnitude-nya masih ada) dan pooled decoder output KANONIK (v=0) student,
    setelah diproyeksikan ke dimensi teacher lewat proj_head.

    Bedanya dari semantic_loss: semantic_loss pakai cosine similarity (murni
    sudut, F.normalize dulu di kedua sisi) -- di sini MSE mentah otomatis ikut
    menghukum selisih PANJANG vektor, bukan cuma selisih arah.
    """
    if not hasattr(cossim_fn, "get_raw_embeddings"):
        raise AttributeError(
            "cossim_fn butuh method get_raw_embeddings(batch) -> (B_total, d_teacher) "
            "untuk magnitude_loss. Pastikan pakai SemanticTeacherSimilarity / "
            "CachedSemanticTeacherSimilarity versi terbaru di file ini."
        )

    teacher_raw = cossim_fn.get_raw_embeddings(batch).detach()

    z_c = out['z_v_canon']
    m_c = out['masks_v_canon']

    pooled = masked_mean(z_c[:, 0, :, :], m_c[:, 0, :], eps=eps)
    pooled_proj = proj_head(pooled)

    teacher_centered = teacher_raw - teacher_raw.mean(dim=0, keepdim=True)
    student_centered = pooled_proj - pooled_proj.mean(dim=0, keepdim=True)

    return F.mse_loss(student_centered, teacher_centered.to(student_centered.device))


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17, num_slices: int = 256, t_max: float = 3.0):
        super().__init__()
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.num_slices = num_slices
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        A = torch.randn(pooled.size(-1), self.num_slices, device=pooled.device)
        A = A / A.norm(p=2, dim=0)
        x_t = (pooled @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * pooled.size(-2)
        return statistic.mean()

def sigreg_loss(out: dict, sigreg_fn: "SIGReg", key: str = "z_v", mask_key: str = "masks") -> torch.Tensor:
    """
    Digeneralisasi supaya bisa dipanggil ke ruang embedding MANA PUN --
    dipakai dua kali di compute_losses: sekali utk 'z_v' (encoder), sekali
    lagi utk 'z_v_canon' (decoder). Ini SESUAI Algorithm 2 LeJEPA: SIGReg
    harus menargetkan embedding yang SAMA dengan yang dipakai loss prediktif
    (di sini: syntactic_loss/semantic_loss/magnitude_loss, yang semuanya
    beroperasi di z_v_canon) -- bukan cuma z_v seperti sebelumnya.
    """
    z, masks = out[key], out[mask_key]
    pooled = masked_mean(z, masks)
    return sigreg_fn(pooled.transpose(0, 1))

def compute_losses(model, batch, criterion, n_core, n_negation, device, use_amp: bool = True, amp_dtype=torch.bfloat16, canon_type: str = "phoneme"):
    batch = move_batch_to_device(batch, device)

    autocast_enabled = use_amp and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
        out = model.train_forward(batch, type=canon_type)

        l_syn = syntactical_loss(out)
        l_sem_general, l_sem_negation, l_sem_negation_contrast = semantic_loss(
            batch, out, criterion.cossim_fn,
            n_core=n_core, negation_offset=n_core - n_negation, n_negation=n_negation,
        )
        l_mag = magnitude_loss(batch, out, criterion.cossim_fn, criterion.proj_head)

        l_sig_prior = sigreg_loss(out, criterion.sigreg_fn, key="z_v", mask_key="masks")
        l_sig_canon = sigreg_loss(out, criterion.sigreg_fn, key="z_v_canon", mask_key="masks_v_canon")
        l_sig = l_sig_prior + criterion.sigreg_canon_weight * l_sig_canon
        l_len = canon_len_loss(out)

        l_sem_negation_combined = l_sem_negation + l_sem_negation_contrast

        canonical_embedding_obj = (
            criterion.zeta_1 * l_syn
            + criterion.zeta_2 * l_sem_general
            + criterion.zeta_2_neg * l_sem_negation_combined
            + criterion.zeta_mag * l_mag
        )
        total = (1 - criterion.lambda_) * canonical_embedding_obj \
            + criterion.lambda_ * l_sig \
            + criterion.canon_len_weight * l_len

    losses = {
        "total": total,
        "syntactic": l_syn.detach(),
        "semantic": (l_sem_general + l_sem_negation_combined).detach(),
        "semantic_general": l_sem_general.detach(),
        "semantic_negation": l_sem_negation.detach(),
        "semantic_negation_contrast": l_sem_negation_contrast.detach(),
        "magnitude": l_mag.detach(),
        "sigreg": l_sig.detach(),
        "sigreg_prior": l_sig_prior.detach(),
        "sigreg_canon": l_sig_canon.detach(),
        "canon_len": l_len.detach(),
    }
    return total, losses, out, batch

@dataclass
class TLeJEPACriterion:
    sigreg_fn: SIGReg
    cossim_fn: nn.Module
    proj_head: "TeacherMagnitudeProjection"
    lambda_: float = 0.5
    zeta_1: float = 1.0
    zeta_2: float = 1.0
    zeta_2_neg: float = 1.0
    zeta_mag: float = 1.0
    canon_len_weight: float = 1.0
    sigreg_canon_weight: float = 1.0

    def to(self, device):
        self.sigreg_fn = self.sigreg_fn.to(device)
        self.cossim_fn = self.cossim_fn.to(device)
        self.proj_head = self.proj_head.to(device)
        return self
