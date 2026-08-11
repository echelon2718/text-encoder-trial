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
from modules import research_metrics as RM
from modules.utils import move_batch_to_device, masked_mean, core_split
try:
    from modules.grad_debug import GradientDebugger
except ImportError:  # modul debug opsional; belum ter-commit di sebagian branch
    GradientDebugger = None

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


class GradNormMonitor:
    """
    Penjaga spike berbasis ambang RELATIF.

    train_step lama hanya menolak grad_norm non-finite. Nilai seperti 1e9 itu
    finite, jadi lolos: clip_grad_norm_ menormalkannya ke `grad_clip` dengan
    ARAH TETAP, lalu Adam menormalisasi per-parameter sehingga langkahnya
    kembali berukuran ~lr. Artinya gradien yang arahnya sudah rusak tetap
    dieksekusi penuh. Beberapa langkah seperti itu cukup untuk merusak
    representasi -- persis pola "grad meledak dulu, collapse menyusul".
    """

    def __init__(self, ratio: float = 6.0, warmup: int = 50, beta: float = 0.98):
        self.ratio, self.warmup, self.beta = ratio, warmup, beta
        self.ema = None
        self.n_seen = 0
        self.n_skipped = 0
        self.n_ema_breach = 0

    def state_dict(self) -> dict:  # n_ema_breach ikut supaya resume tidak reset counter
        return {"ema": self.ema, "n_seen": self.n_seen, "n_skipped": self.n_skipped,
                "n_ema_breach": self.n_ema_breach}

    def load_state_dict(self, d: dict):
        self.ema = d.get("ema", None)
        self.n_seen = d.get("n_seen", 0)
        self.n_skipped = d.get("n_skipped", 0)
        self.n_ema_breach = d.get("n_ema_breach", 0)

    def ema_value(self) -> float:
        return float(self.ema) if self.ema is not None else 0.0

    def check_persistent_explosion(self, threshold: float, n_required: int) -> bool:
        """
        Ledakan PERSISTEN != spike tunggal.

        Spike tunggal sudah ditangani is_spike() (step-nya dibuang, bobot tidak
        diupdate) dan tidak perlu diberitahukan ke manusia. Yang perlu dialertkan
        adalah kalau EMA grad-norm ITU SENDIRI berada di atas ambang selama
        `n_required` pengecekan berturut-turut, karena artinya training sudah
        divergen secara persisten, bukan kena satu batch aneh.

        Counter di-reset begitu EMA turun kembali di bawah ambang, jadi alert
        hanya menyala untuk kondisi yang benar-benar bertahan.
        """
        if self.ema is None or self.n_seen < self.warmup:
            return False
        if float(self.ema) > threshold:
            self.n_ema_breach += 1
        else:
            self.n_ema_breach = 0
        return self.n_ema_breach >= n_required

    def is_spike(self, g: float) -> bool:
        self.n_seen += 1
        if self.ema is None:
            self.ema = g
            return False
        spike = (self.n_seen > self.warmup) and (g > self.ratio * self.ema)
        if spike:
            self.n_skipped += 1
        else:  # EMA hanya diperbarui dari step yang sehat
            self.ema = self.beta * self.ema + (1 - self.beta) * g
        return spike


def train_step(model, optimizer, batch, criterions, n_core, n_negation, device,
               use_amp=True, amp_dtype=torch.bfloat16, grad_clip: Optional[float] = 1.0,
               canon_type: str = "phoneme", grad_debugger: Optional[GradientDebugger] = None,
               step: int = 0, log_every_n_steps: int = 20,
               grad_monitor: Optional["GradNormMonitor"] = None,
               compute_rank: bool = False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, losses, out, _ = compute_losses(
        model, batch, criterions, n_core, n_negation, device,
        use_amp=use_amp, amp_dtype=amp_dtype, canon_type=canon_type,
        compute_rank=compute_rank,
    )

    if not torch.isfinite(total):
        # Loss NaN/Inf TIDAK boleh di-backward -- optimizer.step() akan
        # menyuntikkan NaN ke bobot model SECARA PERMANEN kalau dipaksa jalan.
        losses["total"] = total.detach()
        losses["grad_norm"] = torch.tensor(0.0)
        losses["lr"] = torch.tensor(optimizer.param_groups[0]["lr"])
        losses["_skipped"] = True
        if grad_debugger is not None:
            # Loss itu sendiri sudah NaN/Inf SEBELUM backward() -- akar
            # masalahnya ada di FORWARD (bukan gradien meledak). Dilaporkan
            # terpisah supaya tidak salah didiagnosis sebagai exploding grad.
            grad_debugger.report_forward_anomaly(step, losses, batch)
        return losses, out

    total.backward()

    if grad_debugger is not None:
        # SEBELUM clip_grad_norm_ -- angka yang dicatat harus murni magnitude
        # gradien ASLI, belum diskalakan turun oleh clipping. Versi "full=False"
        # ini sengaja diminimalkan sinkronisasi GPU->CPU-nya (lihat grad_debug.py)
        # supaya boleh dipanggil TIAP step tanpa membebani training yang lama.
        total_norm_raw, per_group_norm, _, per_group_weight_norm = grad_debugger.compute_grad_norms(full=False)
        if step % max(1, log_every_n_steps) == 0:
            grad_debugger.log_lightweight(step, total_norm_raw, per_group_norm, per_group_weight_norm)
        # observe() sendiri yang memutuskan apakah ini "anomali" (non-finite
        # atau melonjak jauh di atas EMA historis) -- kalau ya, breakdown
        # per-parameter yang lebih mahal baru dihitung DI DALAM observe(),
        # jadi biaya besar itu HANYA muncul saat benar-benar dibutuhkan.
        grad_debugger.observe(step, total_norm_raw, per_group_norm, per_group_weight_norm, losses, batch)

    grad_norm = None
    if grad_clip is not None and grad_clip > 0:
        # Sejak magnitude_loss dihapus (paper 5.8) tidak ada lagi proj_head,
        # jadi seluruh parameter yang dioptimasi memang hanya model.parameters().
        grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), grad_clip)

    if grad_norm is not None:
        gn = float(grad_norm)
        bad = (not torch.isfinite(grad_norm)) or (
            grad_monitor is not None and grad_monitor.is_spike(gn))
        if bad:
            # Buang step ini sepenuhnya: gradiennya tidak dipercaya.
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
        debug_gradients: bool = True,
        debug_track_activations: bool = True,
        debug_spike_ratio: float = 6.0,
        debug_spike_zscore: float = 6.0,
        debug_ema_decay: float = 0.98,
        debug_warmup_steps: int = 20,
        debug_max_report_params: int = 25,
        debug_report_cooldown_steps: int = 5,
        debug_max_dumps: int = 20,
        grad_ema_threshold: float = 50.0,
        grad_breach_required: int = 5,
        max_steps: Optional[int] = None,
        research_every_n_steps: Optional[int] = 2000,
        augmenter=None,
        n_aug_1: int = 3,
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

        # --- Debug exploding-gradient ---------------------------------------
        # Dipasang SEKALI di sini (bukan dibuat ulang tiap step) karena hook
        # aktivasi butuh nempel ke instance model yang sama sepanjang training.
        # Set debug_gradients=False kalau root cause sudah ketemu & mau kembali
        # ke kecepatan penuh tanpa overhead breakdown per-grup tiap step.
        self.debug_gradients = debug_gradients
        self.grad_debugger: Optional[GradientDebugger] = None
        self.grad_monitor = GradNormMonitor(ratio=debug_spike_ratio, warmup=debug_warmup_steps)
        self._n_clipped = 0
        self._n_steps_done = 0
        if debug_gradients and GradientDebugger is not None:
            self.grad_debugger = GradientDebugger(
                model=self.model,
                criterion=self.criterion,
                writer=self.writer,
                ckpt_dir=self.ckpt_dir,
                spike_ratio=debug_spike_ratio,
                spike_zscore=debug_spike_zscore,
                ema_decay=debug_ema_decay,
                warmup_steps=debug_warmup_steps,
                max_report_params=debug_max_report_params,
                track_activations=debug_track_activations,
                report_cooldown_steps=debug_report_cooldown_steps,
                max_dumps=debug_max_dumps,
            )

        # Ambang dipasang pada EMA grad-norm (default 50), BUKAN grad-norm sesaat:
        # spike tunggal sudah dibuang otomatis dan tidak perlu dicatat.
        self.grad_ema_threshold = grad_ema_threshold
        self.grad_breach_required = grad_breach_required
        # Metrik riset dihitung berkala supaya konfigurasi yang jelas gagal bisa
        # dihentikan lebih awal, bukan ditunggu sampai run selesai.
        self.research_every_n_steps = research_every_n_steps
        self.augmenter = augmenter
        self.n_aug_1 = n_aug_1
        self._probe_texts = None
        self._current_epoch = 0
        self._death_handled = False
        # Jumlah batch yang harus dilewati pada epoch pertama setelah resume.
        self._resume_batch_offset = 0
        # Batas langkah global. Dipakai untuk protokol token-matched antar
        # ukuran model (paper 6.2): dengan batch size dan urutan data yang
        # sama, jumlah langkah yang sama berarti jumlah token yang sama.
        self.max_steps = max_steps

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.resume_epoch = 1  # dipakai sbg default start_epoch di fit()

        if resume_dir is not None:
            self.resume_epoch = self._load_resume_checkpoint(resume_dir)
        else:
            self.writer.add_text("run_info/run_id", self.run_id, global_step=0)

        self._vis_batch = next(iter(self.val_loader))

        # SLURM mengirim SIGTERM saat time-limit/scancel dan SIGUSR1 kalau job
        # disubmit dengan --signal=B:USR1@<detik>. Keduanya memberi jendela
        # beberapa detik untuk menyimpan checkpoint sebelum SIGKILL menyusul.
        self._install_signal_handlers()

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
        self._restore_trainer_extra(path)
        print(
            f"[Trainer] Dimuat dari {path}: epoch={epoch}, global_step={global_step}, "
            f"best_val_loss={best_metric:.4f}"
        )
        self.writer.add_text(
            "run_info/resumed_from",
            f"Resumed from `{os.path.basename(path)}` at epoch={epoch}, global_step={global_step}",
            global_step=global_step,
        )
        # Berapa batch dari epoch ini yang SUDAH dikonsumsi. Tanpa ini, resume
        # memuat bobot yang benar tapi mengulang epoch dari batch pertama --
        # bobotnya benar, tapi data yang sama diproses dua kali dan sisa epoch
        # tidak pernah dilihat. Untuk run berhari-hari itu pemborosan besar.
        n_per_epoch = max(1, len(self.train_loader))
        self._resume_batch_offset = self.global_step % n_per_epoch
        if self._resume_batch_offset:
            print(f"[Trainer] Melanjutkan di tengah epoch: {self._resume_batch_offset}"
                  f"/{n_per_epoch} batch sudah dikonsumsi, akan dilewati.")
        # Epoch yang benar dihitung dari global_step, bukan dari nilai tersimpan,
        # supaya konsisten walau checkpoint diambil di tengah epoch.
        derived_epoch = self.global_step // n_per_epoch + 1
        if self._resume_batch_offset:
            return derived_epoch
        return epoch if redo_same_epoch else epoch + 1

    def _save_ckpt(self, filename: str, epoch: int) -> str:
        """Satu pintu penyimpanan supaya nama file & isi extra selalu konsisten."""
        path = os.path.join(self.ckpt_dir, filename)
        save_model(
            path, self.model, self.optimizer, epoch, self.best_val_loss,
            scheduler=self.lr_scheduler, global_step=self.global_step,
            extra={"run_id": self.run_id, **self._trainer_extra()},
            criterion=self.criterion,
        )
        return path

    def _record(self, event: str, detail: str = ""):
        """
        Catat kejadian penting ke <ckpt_dir>/events.log. Sengaja file biasa, bukan
        email: watchdog di slurm/watchdog.sh yang membaca berkas ini dan menyalakan
        ulang job, jadi tidak perlu ada manusia di tengah jalur pemulihan.
        """
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{event}\tstep={self.global_step}\tepoch={self._current_epoch}\t{detail}\n"
        try:
            with open(os.path.join(self.ckpt_dir, "events.log"), "a") as f:
                f.write(line)
        except Exception:
            pass
        print(f"[Trainer] {event} {detail}", flush=True)

    def _on_death(self, signame: str):
        """
        Handler sinyal. Tugasnya hanya menyelamatkan bobot lalu mencatat; watchdog
        yang menyalakan ulang. Dibungkus try/except karena proses sedang sekarat --
        kegagalan di sini tidak boleh menutupi penyebab aslinya.
        """
        if self._death_handled:
            return
        self._death_handled = True
        try:
            path = self._save_ckpt("latest_step.pt", self._current_epoch)
            self._record("SIGNAL_SAVE", f"sinyal={signame} ckpt={path}")
        except Exception as e:
            self._record("SIGNAL_SAVE_FAILED", f"sinyal={signame} err={e!r}")
        try:
            self.writer.flush()
        except Exception:
            pass

    def _install_signal_handlers(self):
        import signal

        def _handler(signum, frame):
            self._on_death(signal.Signals(signum).name)
            raise SystemExit(128 + signum)

        for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                pass

    def _check_grad_health(self, losses: dict):
        """
        Ledakan PERSISTEN dicatat, bukan dikirim lewat email. Spike tunggal sudah
        dibuang otomatis oleh GradNormMonitor dan tidak perlu dicatat.
        """
        if self.grad_monitor.check_persistent_explosion(
            self.grad_ema_threshold, self.grad_breach_required
        ):
            self._record(
                "GRAD_EXPLOSION",
                f"ema={self.grad_monitor.ema_value():.3f} ambang={self.grad_ema_threshold} "
                f"raw={float(losses.get('grad_norm', 0.0)):.3f} "
                f"berturut={self.grad_monitor.n_ema_breach}",
            )

    def _get_probe_texts(self):
        """Batch teks tetap sepanjang run supaya kurvanya dapat diperbandingkan."""
        if self._probe_texts is None:
            b = self._vis_batch
            self._probe_texts = [t[1] for t in b["texts"]][:64]   # indeks 1 = grafem kanonik
        return self._probe_texts

    @torch.no_grad()
    def _log_research_metrics(self, out: dict):
        """
        Bukti SEMENTARA untuk tiap hipotesis, ditulis ke TensorBoard dengan tag
        berawalan nomor hipotesis supaya panelnya langsung terbaca.
        """
        self.model.eval()
        metrics = {}
        try:
            metrics.update(RM.rank_diagnostics(out))
            metrics.update(RM.length_diagnostics(out, tau_l=self.criterion.tau_l))
            metrics.update(RM.reconstruction_probe_readout(out))
            metrics.update(RM.boundary_detection_accuracy(out, self.model.empty_norm_ratio))
            metrics.update(RM.length_error_by_view_group(out, n_easy=self.n_aug_1))
            texts = self._get_probe_texts()
            if texts:
                metrics.update(RM.word_order_discrimination(self.model, texts, self.device))
                metrics.update(RM.homophone_discrimination(self.model, self.device))
                metrics.update(RM.holdout_corruption_recall(self.model, texts, self.device))
                metrics.update(RM.length_error_sensitivity(self.model, texts[:16], self.device))
                if self.augmenter is not None:
                    metrics.update(RM.progressive_corruption_recall(
                        self.model, self.augmenter, texts, self.device))
        except Exception as e:
            self._record("RESEARCH_METRIC_FAILED", repr(e))
        finally:
            self.model.train()

        for k, v in metrics.items():
            self.writer.add_scalar(k, v, self.global_step)
        return metrics

    def _trainer_extra(self) -> dict:
        """State non-parameter yang harus ikut selamat lintas crash/resume.
        Tanpa ini, GradNormMonitor mulai dari nol setelah resume -- artinya
        `warmup` step pertama TIDAK terlindungi dari spike, tepat di titik
        paling rawan (bobot baru dimuat, momen Adam baru dipulihkan)."""
        return {
            "grad_monitor": self.grad_monitor.state_dict(),
            "n_clipped": self._n_clipped,
            "n_steps_done": self._n_steps_done,
        }

    def _restore_trainer_extra(self, path):
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return
        ex = ck.get("extra", {}) or {}
        if "grad_monitor" in ex:
            self.grad_monitor.load_state_dict(ex["grad_monitor"])
            print(f"[Trainer] GradNormMonitor dipulihkan: ema={self.grad_monitor.ema}, "
                  f"seen={self.grad_monitor.n_seen}, skipped={self.grad_monitor.n_skipped}")
        self._n_clipped = ex.get("n_clipped", 0)
        self._n_steps_done = ex.get("n_steps_done", 0)

    def train(self, epoch: int, lr_scheduler=None):
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler  # utk kompatibilitas panggilan lama

        self.model.train()
        self._current_epoch = epoch
        running = {}
        n_batches = len(self.train_loader)
        pbar = tqdm(self.train_loader, total=n_batches, desc=f"Epoch {epoch} [train]",
                    colour="green", leave=False, dynamic_ncols=True)

        t_start = time.time()
        n_samples_seen = 0

        # Lewati batch yang sudah dikonsumsi sebelum crash. Dikosongkan setelah
        # epoch pertama supaya epoch berikutnya berjalan penuh.
        skip = self._resume_batch_offset
        self._resume_batch_offset = 0
        if skip:
            self._record("RESUME_SKIP", f"melewati {skip} batch pada epoch {epoch}")

        for step, batch in enumerate(pbar):
            if step < skip:
                continue
            # Model perlu tahu global_step supaya jadwal p_TF (mixed-length
            # canvas, paper 5.7 Tahap 7) bergerak sesuai progres training.
            self.model.set_global_step(self.global_step)
            # RankMe adalah instrumen penentu H5/H6/H8. Dihitung berkala karena
            # SVD mahal kalau tiap step.
            want_rank = bool(self.research_every_n_steps
                             and self.global_step % self.research_every_n_steps == 0)
            if self.lambda_warmup_steps:
                # Linear warmup: lambda_ mulai dari ~0 di global_step=0, naik
                # linear sampai self._lambda_target persis di step ke-N.
                # SETELAH warmup selesai, lambda_ konstan di nilai target
                # (perilaku sama seperti sebelum patch ini kalau
                # lambda_warmup_steps tidak di-set / None).
                frac = min(1.0, self.global_step / self.lambda_warmup_steps)
                self.criterion.lambda_ = frac * self._lambda_target
            try:
                losses, out = train_step(
                    self.model, self.optimizer, batch, self.criterion, self.n_core, self.n_pneg,
                    self.device, use_amp=self.use_amp,
                    amp_dtype=self.amp_dtype, grad_clip=self.grad_clip, canon_type=self.canon_type,
                    grad_debugger=self.grad_debugger, step=self.global_step,
                    log_every_n_steps=self.log_every_n_steps,
                    grad_monitor=self.grad_monitor,
                    compute_rank=want_rank,
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

            # Cek ledakan persisten SETELAH step yang sah (bukan yang dibuang),
            # supaya EMA yang dibaca mencerminkan gradien yang benar-benar dipakai.
            self._check_grad_health(losses)

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
                # Kurva loss yang mulus bisa menyembunyikan clipping yang aktif TIAP step.
                self._n_steps_done += 1
                if self.grad_clip and float(losses.get("grad_norm", 0.0)) > self.grad_clip:
                    self._n_clipped += 1
                self.writer.add_scalar("train_step/clip_frac",
                    self._n_clipped / max(1, self._n_steps_done), self.global_step)
                self.writer.add_scalar("train_step/spike_skip_frac",
                    self.grad_monitor.n_skipped / max(1, self.grad_monitor.n_seen), self.global_step)
                self.writer.add_scalar("train_step/grad_norm_ema",
                    self.grad_monitor.ema_value(), self.global_step)
                # gamma LayerNorm terakhir: knob skala yang ditekan naik oleh SIGReg.
                # Kalau kurva ini merangkak naik monoton dan titik baliknya berimpit
                # dengan titik balik train_step/syntactic, mekanisme skala terkonfirmasi.
                with torch.no_grad():
                    self.writer.add_scalar("scale/gamma_enc_last",
                        float(self.model.encoder.layers[-1].norm2.weight.norm()), self.global_step)
                    self.writer.add_scalar("scale/gamma_dec_last",
                        float(self.model.decoder.layers[-1].norm3.weight.norm()), self.global_step)

            self.global_step += 1

            if self.max_steps is not None and self.global_step >= self.max_steps:
                self._save_ckpt("latest_step.pt", epoch)
                tqdm.write(f"[Trainer] max_steps={self.max_steps} tercapai; menghentikan epoch.")
                break

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
            if (self.research_every_n_steps
                    and self.global_step % self.research_every_n_steps == 0):
                self._log_research_metrics(out)

            if self.save_every_n_steps and self.global_step % self.save_every_n_steps == 0:
                self._save_ckpt("latest_step.pt", epoch)
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
        # m_content, bukan m_canvas: wilayah kosong tidak boleh ikut pooling.
        pooled_canon = masked_mean(z_c, out["masks_v_content"]).float().cpu().numpy()

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

        pooled = masked_mean(z_c, out["masks_v_content"])
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
            if self.max_steps is not None and self.global_step >= self.max_steps:
                tqdm.write(f"[Trainer] max_steps={self.max_steps} tercapai; berhenti.")
                break
            train_loss = self.train(epoch)
            val_loss = self.eval()
            self.log_to_tensorboard(epoch, train_loss, val_loss)

            improved = val_loss["total"] < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss["total"]
                self._save_ckpt("best_model.pt", epoch)

            # Selalu simpan checkpoint di akhir tiap epoch (tidak lagi
            # bergantung pada save_every_n_epochs) supaya training bisa
            # di-resume dari epoch manapun kalau tiba-tiba crash/disconnect.
            self._save_ckpt("latest_model.pt", epoch)
            # latest_step.pt juga disegarkan di akhir epoch supaya --resume
            # selalu menemukan state paling baru apa pun titik matinya.
            self._save_ckpt("latest_step.pt", epoch)

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
        self._record("FINISHED", f"best_val={self.best_val_loss:.6f}")

    def close(self):
        if self.grad_debugger is not None:
            summary = self.grad_debugger.summary()
            tqdm.write(
                f"[Trainer] Ringkasan GradientDebugger: "
                f"{summary['n_anomalies_total']} anomali terdeteksi sepanjang run, "
                f"{summary['n_dumps_written']} dump detail tersimpan di "
                f"{os.path.join(self.ckpt_dir, 'anomaly_dumps')}, "
                f"anomali terakhir di step={summary['last_anomaly_step']}."
            )
            self.grad_debugger.remove_hooks()
        self.writer.flush()
        self.writer.close()
