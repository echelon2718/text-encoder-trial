from __future__ import annotations

import argparse
import csv                          # ── NEW ──
import json                         # ── NEW ──
import logging
import math
import os                           # ── NEW ──
from collections import defaultdict, deque
from dataclasses import dataclass, asdict   # ── NEW: asdict ──
from datetime import datetime               # ── NEW ──
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# ── project imports (adjust paths to your package layout) ──────────────────
from data.dataset import AugmentDataset, collate_fn
from modules.transformer import SwaraJEPA
from modules.losses import compute_phase1_loss


# ---------------------------------------------------------------------------
# Logging: route all records through tqdm.write so bars stay intact
# ---------------------------------------------------------------------------

class _TqdmHandler(logging.StreamHandler):
    """Emits log records via tqdm.write() — prevents progress bars from being
    overwritten by interleaved log lines."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
            self.flush()
        except Exception:
            self.handleError(record)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[_TqdmHandler()],
)
log = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class TrainConfig:
    # ── Data ──────────────────────────────────────────────────────────────
    lexicon_path:   str   = "./lexicon/abbrev-lexicon.json"
    dataset_path:   str   = "./data/complete_corpus.csv"
    val_fraction:   float = 0.05
    canon_type:     str   = "phoneme"          # "phoneme" or "text"

    # ── Model ─────────────────────────────────────────────────────────────
    n_vocab_text:   int   = 256                # byte-level vocabulary
    d_model:        int   = 128
    n_heads:        int   = 8
    enc_layers:     int   = 6
    can_dec_layers: int   = 4

    # ── Training ──────────────────────────────────────────────────────────
    num_epochs:     int   = 50
    batch_size:     int   = 8
    lr:             float = 1e-4
    weight_decay:   float = 1e-2
    warmup_epochs:  float = 3.0                # linear warmup (may be fractional)
    max_grad_norm:  float = 1.0
    device:         str   = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Loss weights ──────────────────────────────────────────────────────
    lam:            float = 0.05               # λ: SIGReg + JEPA block weight
    beta:           float = 0.5               # β: SIGReg vs JEPA balance
    eps:            float = 1.0               # ε: L_canon weight
    zeta:           float = 0.01              # ζ: L_canonlen weight
    num_slices:     int   = 1024              # |A| for SIGReg directions

    # ── Logging & checkpointing ────────────────────────────────────────────
    eval_every:     int   = 1                  # run validation every N epochs
    save_every:     int   = 5                  # save checkpoint every N epochs
    log_window:     int   = 50                 # rolling window (batches) for tqdm postfix
    run_dir:        str   = "runs/phase1"

    # ━━ NEW: HuggingFace Hub ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Set push_to_hub=True and provide hf_repo_id to enable auto-upload.
    # Token priority: hf_token field → HF_TOKEN env var → cached `huggingface-cli login`
    push_to_hub:    bool  = False              # auto-upload after training finishes
    hf_repo_id:     str   = ""                # e.g. "username/swara-jepa-phase1"
    hf_token:       str   = ""                # explicit token (optional)
    hf_private:     bool  = False             # create as private repo
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RunningMean:
    """Sliding-window mean — keeps the tqdm postfix smooth across batches."""

    def __init__(self, window: int = 50) -> None:
        self._bufs: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    def update(self, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            self._bufs[k].append(v)

    def means(self) -> dict[str, float]:
        return {k: sum(v) / len(v) for k, v in self._bufs.items() if v}

    def reset(self) -> None:
        self._bufs.clear()


def warmup_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Linear warmup over `warmup_steps`, then cosine decay to zero."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Trainer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Trainer:
    """
    Epoch-based Phase-1 pre-training of SwaraJEPA.

    Features
    --------
    - Epoch loop with a persistent outer tqdm bar (green)
    - Per-batch inner tqdm bar (yellow) with live smoothed-loss postfix
    - Validation bar (cyan) that replaces the inner bar during eval
    - All log lines routed through tqdm.write — no bar corruption
    - TensorBoard: per-step scalars + per-epoch train/val averages
      + side-by-side train vs val L1 comparison chart
    - [NEW] Training history saved to JSON + CSV (training_history_*.csv)
    - [NEW] Auto-upload to HuggingFace Hub after training (if push_to_hub=True)

    Typical lifecycle
    -----------------
        cfg     = TrainConfig(num_epochs=50, push_to_hub=True,
                              hf_repo_id="myname/swara-jepa")
        trainer = Trainer(cfg)
        trainer.train()
    """

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg         = cfg
        self.epoch       = 0       # last completed epoch
        self.global_step = 0       # total optimiser steps taken
        self.run_dir     = Path(cfg.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._build_data()
        self._build_model()
        self._build_optimiser()

        self.writer  = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        self._smooth = RunningMean(window=cfg.log_window)

        # ── NEW: training history ─────────────────────────────────────────
        # Populated during _epoch_loop; persisted via _save_history()
        self.history: dict = {
            "metadata": {
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "config":     asdict(cfg),
            },
            "train": [],   # list of {epoch, L1, L_sigreg, L_jepa, L_canon, ...}
            "val":   [],   # list of {epoch, L1, L_sigreg, L_jepa, L_canon, ...}
        }
        # ─────────────────────────────────────────────────────────────────

    # ── Setup helpers ─────────────────────────────────────────────────────

    def _build_data(self) -> None:
        cfg = self.cfg
        full_ds = AugmentDataset(
            lexicon_path=cfg.lexicon_path,
            dataset_path=cfg.dataset_path,
            mode=cfg.dataset_mode
        )
        n_val   = max(1, int(len(full_ds) * cfg.val_fraction))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        self.train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size,
            shuffle=True, collate_fn=collate_fn, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size,
            shuffle=False, collate_fn=collate_fn, drop_last=False,
        )
        self._phoneme_vocab   = full_ds.phoneme_tokenizer.vocab_size
        self._steps_per_epoch = len(self.train_loader)
        log.info(
            "Data  %d train / %d val  |  %d steps/epoch  |  phoneme vocab %d",
            n_train, n_val, self._steps_per_epoch, self._phoneme_vocab,
        )

    def _build_model(self) -> None:
        cfg = self.cfg
        self.model = SwaraJEPA(
            n_vocab_text=cfg.n_vocab_text,
            n_vocab_phoneme=self._phoneme_vocab,
            d_model=cfg.d_model,
            n_attn_heads=cfg.n_heads,
            enc_layers=cfg.enc_layers,
            can_dec_layers=cfg.can_dec_layers,
        ).to(cfg.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info("Model %s  |  params %s", type(self.model).__name__, f"{n_params:,}")

    def _build_optimiser(self) -> None:
        cfg          = self.cfg
        total_steps  = cfg.num_epochs * self._steps_per_epoch
        warmup_steps = int(cfg.warmup_epochs * self._steps_per_epoch)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98),
        )
        self.scheduler = warmup_cosine_schedule(self.optimizer, warmup_steps, total_steps)
        log.info(
            "AdamW lr=%.2e  |  warmup %d steps (%.1f ep)  |  total %d steps",
            cfg.lr, warmup_steps, cfg.warmup_epochs, total_steps,
        )

    # ── Single training step ──────────────────────────────────────────────

    def _train_step(self, batch: dict) -> dict[str, float]:
        """Forward + backward + clip + step. Returns Python-float metrics."""
        self.model.train()
        batch = {
            k: v.to(self.cfg.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        out       = self.model(batch, type=self.cfg.canon_type, mode="training")
        L1, parts = compute_phase1_loss(
            out, batch,
            global_step=self.global_step,
            lam=self.cfg.lam, beta=self.cfg.beta,
            eps=self.cfg.eps,  zeta=self.cfg.zeta,
            num_slices=self.cfg.num_slices,
        )
        self.optimizer.zero_grad(set_to_none=True)
        L1.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        return {k: v.item() for k, v in parts.items()} | {
            "grad_norm": grad_norm.item(),
            "lr":        self.scheduler.get_last_lr()[0],
        }

    # ── Evaluation ────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """
        Run the full validation set.
        Returns per-component averages — directly comparable to training logs.
        """
        self.model.eval()
        accum: dict[str, float] = defaultdict(float)
        n = 0

        for batch in tqdm(
            self.val_loader,
            desc="  Val   ",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            position=1,
            colour="cyan",
        ):
            batch = {
                k: v.to(self.cfg.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out      = self.model(batch, type=self.cfg.canon_type, mode="training")
            _, parts = compute_phase1_loss(
                out, batch,
                global_step=self.global_step,
                lam=self.cfg.lam, beta=self.cfg.beta,
                eps=self.cfg.eps,  zeta=self.cfg.zeta,
                num_slices=self.cfg.num_slices,
            )
            for k, v in parts.items():
                accum[k] += v.item()
            n += 1

        return {k: v / n for k, v in accum.items()} if n else {}

    # ── Checkpointing ─────────────────────────────────────────────────────

    def save_checkpoint(self, tag: str | None = None) -> Path:
        fname = f"ckpt_{tag or f'epoch{self.epoch:04d}'}.pt"
        path  = self.run_dir / fname
        torch.save({
            "epoch":       self.epoch,
            "global_step": self.global_step,
            "model":       self.model.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict(),
            "config":      self.cfg,
        }, path)
        log.info("Checkpoint → %s", path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.cfg.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        # Backwards-compat: old checkpoints stored "step" not "global_step"
        self.epoch       = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", ckpt.get("step", 0))
        log.info("Resumed → epoch %d  global step %d", self.epoch, self.global_step)

    # ━━ NEW: History persistence ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _save_history(self) -> None:
        """
        Write self.history to:
          • training_history.json        — full structured record
          • training_history_train.csv   — train metrics, one row per epoch
          • training_history_val.csv     — val   metrics, one row per eval epoch
        """
        # ── JSON ──────────────────────────────────────────────────────────
        json_path = self.run_dir / "training_history.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        log.info("History (JSON) → %s", json_path)

        # ── CSV (one file per split) ───────────────────────────────────────
        for split in ("train", "val"):
            rows = self.history.get(split, [])
            if not rows:
                continue
            csv_path = self.run_dir / f"training_history_{split}.csv"
            fieldnames = list(rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            log.info("History (CSV)  → %s", csv_path)

    # ━━ NEW: Model card ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _save_model_card(self) -> None:
        """
        Generate a README.md inside run_dir that will serve as the
        HuggingFace model card.
        """
        cfg = self.cfg

        train_rows = self.history.get("train", [])
        val_rows   = self.history.get("val",   [])
        final_train = train_rows[-1] if train_rows else {}
        best_val    = (
            min(val_rows, key=lambda r: r.get("L1", float("inf")))
            if val_rows else {}
        )

        nan = float("nan")

        def fmt(d: dict, key: str) -> str:
            v = d.get(key, nan)
            return f"{v:.4f}" if v == v else "—"   # NaN check

        started_at  = self.history["metadata"].get("started_at", "unknown")
        finished_at = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        card = f"""\
---
tags:
  - swara-jepa
  - audio
  - speech
  - self-supervised-learning
  - phase1
license: apache-2.0
---

# SwaraJEPA — Phase-1 Pre-trained Model

Pre-trained with the **SwaraJEPA Phase-1** self-supervised objective
(SIGReg + JEPA block prediction + canonical-form decoder).

| | |
|---|---|
| **Started**  | `{started_at}` |
| **Finished** | `{finished_at}` |
| **Epochs**   | {cfg.num_epochs} |
| **Device**   | `{cfg.device}` |

---

## Training Configuration

### Architecture

| Parameter | Value |
|-----------|-------|
| `d_model` | {cfg.d_model} |
| `n_heads` | {cfg.n_heads} |
| `enc_layers` | {cfg.enc_layers} |
| `can_dec_layers` | {cfg.can_dec_layers} |
| `n_vocab_text` | {cfg.n_vocab_text} |
| `canon_type` | `{cfg.canon_type}` |

### Optimiser

| Parameter | Value |
|-----------|-------|
| `batch_size` | {cfg.batch_size} |
| `lr` | {cfg.lr} |
| `weight_decay` | {cfg.weight_decay} |
| `warmup_epochs` | {cfg.warmup_epochs} |
| `max_grad_norm` | {cfg.max_grad_norm} |

### Loss Weights

| Symbol | Parameter | Value |
|--------|-----------|-------|
| λ | `lam` (SIGReg + JEPA block weight) | {cfg.lam} |
| β | `beta` (SIGReg vs JEPA balance) | {cfg.beta} |
| ε | `eps` (L_canon weight) | {cfg.eps} |
| ζ | `zeta` (L_canonlen weight) | {cfg.zeta} |
| — | `num_slices` (SIGReg directions) | {cfg.num_slices} |

---

## Final Metrics

### Training — last epoch ({final_train.get("epoch", "?")})

| Metric | Value |
|--------|-------|
| **L1** | **{fmt(final_train, "L1")}** |
| L_sigreg | {fmt(final_train, "L_sigreg")} |
| L_jepa | {fmt(final_train, "L_jepa")} |
| L_canon | {fmt(final_train, "L_canon")} |
| L_canonlen | {fmt(final_train, "L_canonlen")} |

### Best Validation (epoch {best_val.get("epoch", "?")})

| Metric | Value |
|--------|-------|
| **L1** | **{fmt(best_val, "L1")}** |
| L_sigreg | {fmt(best_val, "L_sigreg")} |
| L_jepa | {fmt(best_val, "L_jepa")} |
| L_canon | {fmt(best_val, "L_canon")} |
| L_canonlen | {fmt(best_val, "L_canonlen")} |

---

## Repository Contents

| File / Folder | Description |
|---------------|-------------|
| `ckpt_final.pt` | Final checkpoint — model weights + optimiser + scheduler state |
| `ckpt_epoch*.pt` | Periodic checkpoints (every {cfg.save_every} epochs) |
| `train_config.json` | Full `TrainConfig` serialised to JSON |
| `training_history.json` | Per-epoch train + val metrics (full structured record) |
| `training_history_train.csv` | Train metrics per epoch (easy to open in pandas / Excel) |
| `training_history_val.csv` | Validation metrics per eval epoch |
| `tb/` | TensorBoard event files |

---

## TensorBoard

After downloading the `tb/` folder:

```bash
pip install tensorboard
tensorboard --logdir tb/
```

Available dashboards:
- `step/*` — per-step loss and learning rate
- `epoch/train_*` — per-epoch training averages
- `epoch/val_*` — per-epoch validation averages
- `compare/L1` — train vs val L1 on a single chart

---

## Usage

```python
import torch
from modules.transformer import SwaraJEPA

ckpt  = torch.load("ckpt_final.pt", map_location="cpu")
cfg   = ckpt["config"]              # TrainConfig dataclass

model = SwaraJEPA(
    n_vocab_text    = cfg.n_vocab_text,
    n_vocab_phoneme = ...,           # match your phoneme tokenizer
    d_model         = cfg.d_model,
    n_attn_heads    = cfg.n_heads,
    enc_layers      = cfg.enc_layers,
    can_dec_layers  = cfg.can_dec_layers,
)
model.load_state_dict(ckpt["model"])
model.eval()
```
"""
        card_path = self.run_dir / "README.md"
        card_path.write_text(card, encoding="utf-8")
        log.info("Model card → %s", card_path)

    # ━━ NEW: HuggingFace Hub upload ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def push_to_hub(self) -> None:
        """
        Upload the entire run directory to HuggingFace Hub.

        Uploads (everything inside run_dir/):
          ckpt_final.pt, ckpt_epoch*.pt
          train_config.json
          training_history.json / *.csv
          tb/   (TensorBoard event files)
          README.md (model card)

        Token resolution order:
          1. cfg.hf_token
          2. HF_TOKEN  environment variable
          3. Cached credentials from `huggingface-cli login`
        """
        try:
            from huggingface_hub import HfApi, create_repo
        except ImportError:
            log.error(
                "huggingface_hub is not installed. "
                "Run:  pip install huggingface_hub"
            )
            return

        cfg = self.cfg
        if not cfg.hf_repo_id:
            log.error(
                "push_to_hub=True but hf_repo_id is empty. "
                "Set --hf_repo_id username/repo-name and retry."
            )
            return

        # ── Resolve token ─────────────────────────────────────────────────
        token: str | None = (
            cfg.hf_token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or None          # fall back to cached CLI credentials
        )

        api = HfApi(token=token)

        # ── Create repo if it doesn't exist ───────────────────────────────
        try:
            create_repo(
                repo_id=cfg.hf_repo_id,
                repo_type="model",
                private=cfg.hf_private,
                token=token,
                exist_ok=True,
            )
            log.info("Hub repo ready: %s", cfg.hf_repo_id)
        except Exception as exc:
            log.error("Failed to create/access repo '%s': %s", cfg.hf_repo_id, exc)
            return

        # ── Persist artefacts that live alongside the checkpoints ─────────
        self._save_history()
        self._save_model_card()

        # ── Save train_config.json ─────────────────────────────────────────
        config_path = self.run_dir / "train_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2)
        log.info("Config → %s", config_path)

        # ── Upload everything ─────────────────────────────────────────────
        log.info(
            "Uploading %s → https://huggingface.co/%s …",
            self.run_dir, cfg.hf_repo_id,
        )
        try:
            api.upload_folder(
                folder_path=str(self.run_dir),
                repo_id=cfg.hf_repo_id,
                repo_type="model",
                token=token,
                commit_message=(
                    f"SwaraJEPA phase-1 — epoch {self.epoch} "
                    f"| {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
                ),
            )
            log.info(
                "✓ Upload complete → https://huggingface.co/%s",
                cfg.hf_repo_id,
            )
        except Exception as exc:
            log.error("Hub upload failed: %s", exc)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # ── Public entry-point ────────────────────────────────────────────────

    def train(self, resume: str | None = None) -> None:
        if resume:
            self.load_checkpoint(resume)

        cfg = self.cfg
        log.info(
            "Phase-1  |  device %s  |  epochs %d  |  steps/ep %d  |  batch %d",
            cfg.device, cfg.num_epochs, self._steps_per_epoch, cfg.batch_size,
        )
        log.info(
            "λ=%.3f  β=%.3f  ε=%.3f  ζ=%.4f  |  slices %d",
            cfg.lam, cfg.beta, cfg.eps, cfg.zeta, cfg.num_slices,
        )

        try:
            self._epoch_loop()
        finally:
            self.writer.close()   # always flush TensorBoard, even on crash

        # ── NEW: push to Hub after a clean finish ─────────────────────────
        if cfg.push_to_hub:
            self.push_to_hub()
        # ─────────────────────────────────────────────────────────────────

    # ── Internal epoch loop ───────────────────────────────────────────────

    def _epoch_loop(self) -> None:
        cfg      = self.cfg
        last_val: dict[str, float] = {}        # latest known val metrics (for outer bar)

        # ── outer bar: one tick per epoch ─────────────────────────────────
        epoch_bar = tqdm(
            range(self.epoch, cfg.num_epochs),
            desc="Training",
            unit="epoch",
            initial=self.epoch,
            total=cfg.num_epochs,
            dynamic_ncols=True,
            position=0,
            colour="green",
        )

        for epoch in epoch_bar:
            self.epoch = epoch
            self._smooth.reset()
            epoch_acc: dict[str, list[float]] = defaultdict(list)

            # ── inner bar: one tick per batch ─────────────────────────────
            batch_bar = tqdm(
                self.train_loader,
                desc=f"  Ep {epoch + 1:>3d}/{cfg.num_epochs}",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
                position=1,
                colour="yellow",
            )

            for batch in batch_bar:
                metrics = self._train_step(batch)
                self.global_step += 1

                # Per-step TensorBoard scalars
                for k, v in metrics.items():
                    self.writer.add_scalar(f"step/{k}", v, self.global_step)

                # Accumulate for epoch-level averages
                for k, v in metrics.items():
                    epoch_acc[k].append(v)

                # Live smoothed postfix on the batch bar
                self._smooth.update(metrics)
                s = self._smooth.means()
                batch_bar.set_postfix({
                    "L1":    f"{s.get('L1',         float('nan')):.3f}",
                    "SIG":   f"{s.get('L_sigreg',   float('nan')):.3f}",
                    "JEPA":  f"{s.get('L_jepa',     float('nan')):.4f}",
                    "canon": f"{s.get('L_canon',    float('nan')):.3f}",
                    "lr":    f"{s.get('lr',         float('nan')):.2e}",
                }, refresh=False)

            # ── epoch-end: compute & log averages ─────────────────────────
            ep = {k: sum(v) / len(v) for k, v in epoch_acc.items() if v}

            for k, v in ep.items():
                self.writer.add_scalar(f"epoch/train_{k}", v, epoch + 1)

            log.info(
                "Ep %3d/%d  L1 %.3f  SIG %.3f  JEPA %.4f  "
                "canon %.3f  canonlen %.3f  ‖g‖ %.3f  lr %.2e",
                epoch + 1, cfg.num_epochs,
                ep.get("L1",         float("nan")),
                ep.get("L_sigreg",   float("nan")),
                ep.get("L_jepa",     float("nan")),
                ep.get("L_canon",    float("nan")),
                ep.get("L_canonlen", float("nan")),
                ep.get("grad_norm",  float("nan")),
                ep.get("lr",         float("nan")),
            )

            # ── NEW: record train history ──────────────────────────────────
            self.history["train"].append({"epoch": epoch + 1, **ep})
            # ─────────────────────────────────────────────────────────────

            # ── periodic evaluation ───────────────────────────────────────
            if (epoch + 1) % cfg.eval_every == 0:
                last_val = self.evaluate()
                if last_val:
                    for k, v in last_val.items():
                        self.writer.add_scalar(f"epoch/val_{k}", v, epoch + 1)

                    # Side-by-side train vs val L1 on one TensorBoard chart
                    self.writer.add_scalars("compare/L1", {
                        "train": ep.get("L1", float("nan")),
                        "val":   last_val.get("L1", float("nan")),
                    }, epoch + 1)

                    log.info(
                        "  ↳ val  L1 %.3f  SIG %.3f  JEPA %.4f  "
                        "canon %.3f  canonlen %.3f",
                        last_val.get("L1",         float("nan")),
                        last_val.get("L_sigreg",   float("nan")),
                        last_val.get("L_jepa",     float("nan")),
                        last_val.get("L_canon",    float("nan")),
                        last_val.get("L_canonlen", float("nan")),
                    )

                    # ── NEW: record val history ────────────────────────────
                    self.history["val"].append({"epoch": epoch + 1, **last_val})
                    # ─────────────────────────────────────────────────────

            # ── periodic checkpoint ───────────────────────────────────────
            if (epoch + 1) % cfg.save_every == 0:
                self.save_checkpoint()

            # ── outer bar postfix (always shows latest val if available) ──
            pf: dict[str, str] = {"train_L1": f"{ep.get('L1', float('nan')):.3f}"}
            if last_val:
                pf["val_L1"] = f"{last_val.get('L1', float('nan')):.3f}"
            epoch_bar.set_postfix(pf)

        # ── final checkpoint + eval ───────────────────────────────────────
        self.epoch = cfg.num_epochs
        self.save_checkpoint(tag="final")
        last_val = self.evaluate()
        if last_val:
            log.info(
                "Done  L1 %.3f  SIG %.3f  JEPA %.4f  canon %.3f  canonlen %.3f",
                last_val.get("L1",         float("nan")),
                last_val.get("L_sigreg",   float("nan")),
                last_val.get("L_jepa",     float("nan")),
                last_val.get("L_canon",    float("nan")),
                last_val.get("L_canonlen", float("nan")),
            )
            # ── NEW: record final val ─────────────────────────────────────
            self.history["val"].append({"epoch": cfg.num_epochs, **last_val})
            # ─────────────────────────────────────────────────────────────


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI entry-point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-1 SwaraJEPA pre-training")

    # Training schedule
    p.add_argument("--num_epochs",    type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=float, default=3.0,
                   help="Linear warmup duration in epochs (may be fractional)")
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")

    # Loss weights
    p.add_argument("--lam",           type=float, default=0.05)
    p.add_argument("--beta",          type=float, default=0.5)
    p.add_argument("--eps",           type=float, default=1.0)
    p.add_argument("--zeta",          type=float, default=0.01)
    p.add_argument("--num_slices",    type=int,   default=1024)

    # Paths
    p.add_argument("--lexicon_path",  type=str,   default="./lexicon/abbrev-lexicon.json")
    p.add_argument("--dataset_path",  type=str,   default="./data/complete_corpus.csv")
    p.add_argument("--run_dir",       type=str,   default="runs/phase1")
    p.add_argument("--resume",        type=str,   default=None,
                   help="Path to a checkpoint to resume from")

    # Logging & saving
    p.add_argument("--eval_every",    type=int,   default=1,
                   help="Run validation every N epochs")
    p.add_argument("--save_every",    type=int,   default=5,
                   help="Save a checkpoint every N epochs")

    # ── NEW: HuggingFace Hub ──────────────────────────────────────────────
    p.add_argument(
        "--push_to_hub",
        action="store_true",
        default=False,
        help="Upload run_dir to HuggingFace Hub after training finishes",
    )
    p.add_argument(
        "--hf_repo_id",
        type=str,
        default="",
        help='Hub repository, e.g. "username/swara-jepa-phase1"',
    )
    p.add_argument(
        "--hf_token",
        type=str,
        default="",
        help="HuggingFace token (falls back to HF_TOKEN env var or cached login)",
    )
    p.add_argument(
        "--hf_private",
        action="store_true",
        default=False,
        help="Create the Hub repository as private",
    )
    # ─────────────────────────────────────────────────────────────────────

    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    cfg     = TrainConfig(**{k: v for k, v in vars(args).items() if k != "resume"})
    trainer = Trainer(cfg)
    trainer.train(resume=args.resume)
