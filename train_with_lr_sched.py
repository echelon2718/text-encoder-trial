import argparse
import math
import torch
from torch.utils.data import DataLoader
from functools import partial
from data.dataset import AugmentDataset, BatchSampler, collate_fn
from modules.tlejepa import TLeJEPA
from modules.training import Trainer, find_latest_run_dir
from modules.losses import (
    TLeJEPACriterion,
    SIGReg,
    SemanticTeacherSimilarity,
    CachedSemanticTeacherSimilarity,
    TeacherMagnitudeProjection,
)

from modules.utils import core_split


def get_args():
    parser = argparse.ArgumentParser(description="Train TLeJEPA")
    parser.add_argument("--lexicon-path", type=str, default="./lexicon/abbrev-lexicon.json")
    parser.add_argument("--dataset-path", type=str, default="./data/english_singlish_g2p.csv")
    parser.add_argument("--dataset-mode", type=str, default="huggingface", choices=["local", "huggingface"])
    parser.add_argument("--run-label", type=str, default="run")
    parser.add_argument("--ckpt-root", type=str, default="checkpoints")
    parser.add_argument("--resume", action="store_true",
                         help="Lanjutkan dari checkpoint TERBARU utk --run-label ini di --ckpt-root "
                              "(cari otomatis: latest_step.pt > latest_model.pt > best_model.pt)")

    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--n_singlish", type=int, default=2)
    parser.add_argument("--n_premise_and_negation", type=int, default=2)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--enc-layers", type=int, default=6)
    parser.add_argument("--dec-layers", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--num-slices", type=int, default=1024)
    parser.add_argument("--lambda_", type=float, default=0.6)
    parser.add_argument("--zeta1", type=float, default=1)
    parser.add_argument("--zeta2", type=float, default=0.1)
    parser.add_argument("--zeta2-neg", type=float, default=1.0,
                         help="Bobot (l_sem_negation + l_sem_negation_contrast) -- blok negasi murni "
                              "dari semantic_loss. Ini sinyal negasi UTAMA, jangan dikecilkan.")
    parser.add_argument("--zeta-mag", type=float, default=0.1,
                         help="Bobot magnitude_loss (MSE thd embedding teacher MENTAH lewat proj_head). "
                              "Sengaja kecil (default 0.1, jauh di bawah --zeta2-neg): magnitude_loss "
                              "berpotensi tarik-menarik dengan SIGReg (keduanya menekan skala/statistik "
                              "populasi embedding), jadi ini pelengkap sinyal negasi, bukan penentu utama. "
                              "Pantau losses['magnitude'] mentah dulu sebelum menaikkan ini.")
    parser.add_argument("--d-teacher", type=int, default=768,
                         help="Dimensi embedding teacher (NegMPNet = 768) -- target proj_head.")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)

    # --- Learning rate scheduler: linear warmup -> cosine decay ---
    parser.add_argument("--warmup-ratio", type=float, default=0.05,
                         help="Fraksi total training step yang dipakai untuk linear warmup (0-1)")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1,
                         help="Lantai LR sebagai rasio dari lr puncak, dicapai di akhir cosine decay")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize-every-n-steps", type=int, default=1000,
                         help="Log diagnostik visual (t-SNE, histogram SIGReg, heatmap semantik) "
                              "ke TensorBoard tiap N step -- TIDAK menunggu 1 epoch selesai. "
                              "Set 0 untuk matikan (cuma logging di akhir epoch, perilaku lama).")

    return parser.parse_args()


def build_lr_scheduler(optimizer, total_steps: int, warmup_ratio: float = 0.05, min_lr_ratio: float = 0.1):
    """
    Linear warmup diikuti cosine decay ke `min_lr_ratio * lr_puncak`.
    Dipanggil per-step (bukan per-epoch) — cocok dengan Trainer.train() yang
    memanggil lr_scheduler.step() di setiap batch.
    """
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main(args):
    resume_dir = None
    if args.resume:
        resume_dir = find_latest_run_dir(args.ckpt_root, args.run_label)
        if resume_dir is None:
            raise FileNotFoundError(
                f"--resume dipakai tapi tidak ada folder checkpoint utk --run-label='{args.run_label}' "
                f"di dalam '{args.ckpt_root}'. Cek lagi nama label atau path --ckpt-root."
            )
        print(f"[main] Ditemukan checkpoint terbaru utk '{args.run_label}': {resume_dir}")
        print(
            "[main] PENTING: pastikan semua argumen arsitektur model "
            "(--d-model, --heads, --enc-layers, --dec-layers, dst) dan --epochs "
            "SAMA PERSIS dengan run aslinya, kalau tidak state_dict/LR schedule "
            "tidak akan cocok."
        )

    full_dataset = AugmentDataset(
        lexicon_path=args.lexicon_path,
        dataset_path=args.dataset_path,
        mode=args.dataset_mode,
    )

    train_size = int(args.train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(full_dataset), generator=g).tolist()

    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    # `.select()` (bukan pandas `.iloc`) -- tetap Arrow-backed & di-cache ke
    # disk oleh library `datasets`, jadi aman di-share lintas worker DataLoader
    # tanpa memori dobel-dobel.
    train_hf = full_dataset.dataset.select(train_idx)
    val_hf = full_dataset.dataset.select(val_idx)
    shared_tokenizer = full_dataset.phoneme_tokenizer

    # full_dataset (dataset penuh) sudah tidak dibutuhkan lagi setelah ini --
    # dilepas eksplisit supaya tidak nyimpen copy dataset yang nganggur
    # sepanjang training (sebelumnya ini jadi salah satu sumber pemakaian
    # RAM host yang tidak perlu).
    del full_dataset

    train_dataset = AugmentDataset(
        lexicon_path=args.lexicon_path,
        hf_dataset=train_hf,
        tokenizer=shared_tokenizer,
    )

    val_dataset = AugmentDataset(
        lexicon_path=args.lexicon_path,
        hf_dataset=val_hf,
        tokenizer=shared_tokenizer,
    )

    train_sampler = BatchSampler(
        dataset=train_dataset.dataset,
        batch_size=args.batch_size,
        n_singlish_per_batch=args.n_singlish,
        n_negation_per_batch=args.n_premise_and_negation,
        seed=args.seed,
        drop_last=True,
    )

    val_sampler = BatchSampler(
        dataset=val_dataset.dataset,
        batch_size=args.batch_size,
        n_singlish_per_batch=args.n_singlish,
        n_negation_per_batch=args.n_premise_and_negation,
        seed=args.seed,
        drop_last=False,
    )

    loader_extra_kwargs = (
        {"persistent_workers": True, "prefetch_factor": args.prefetch_factor}
        if args.num_workers > 0 else {}
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=partial(collate_fn, max_seq_len=args.max_length),
        pin_memory=True,
        **loader_extra_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=partial(collate_fn, max_seq_len=args.max_length),
        pin_memory=True,
        **loader_extra_kwargs,
    )

    model = TLeJEPA(
        n_vocab_text=256,
        n_vocab_phoneme=shared_tokenizer.vocab_size,
        d_model=args.d_model,
        n_attn_heads=args.heads,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        max_length=args.max_length,
        dropout=args.dropout,
    )
    model.set_gradient_checkpointing(True)

    criterion = TLeJEPACriterion(
        sigreg_fn=SIGReg(num_slices=args.num_slices),
        cossim_fn=CachedSemanticTeacherSimilarity(
            SemanticTeacherSimilarity()
        ),
        proj_head=TeacherMagnitudeProjection(d_model=args.d_model, d_teacher=args.d_teacher),
        lambda_=args.lambda_,
        zeta_1=args.zeta1,
        zeta_2=args.zeta2,
        zeta_2_neg=args.zeta2_neg,
        zeta_mag=args.zeta_mag,
    )

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.proj_head.parameters()),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )

    # len(train_loader) == train_sampler.n_batches karena DataLoader dipakai dengan batch_sampler
    total_steps = args.epochs * len(train_loader)
    lr_scheduler = build_lr_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
    )

    trainer = Trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        n_core=args.batch_size,
        n_singlish=args.n_singlish,
        n_premise_and_negation=args.n_premise_and_negation,
        device=torch.device(args.device),
        label=args.run_label,
        ckpt_dir=args.ckpt_root,
        resume_dir=resume_dir,
        lr_scheduler=lr_scheduler,
        visualize_every_n_steps=(args.visualize_every_n_steps or None),
    )

    # --epochs = TOTAL epoch yang ditarget (bukan "epoch tambahan di atas yang
    # sudah jalan"). trainer.resume_epoch == 1 utk run baru, jadi rumus ini
    # otomatis benar utk kedua kasus (baru maupun resume).
    remaining_epochs = max(0, args.epochs - (trainer.resume_epoch - 1))
    if remaining_epochs == 0:
        print(f"[main] --epochs={args.epochs} sudah tercapai (resume_epoch={trainer.resume_epoch}). Tidak ada yang perlu dijalankan.")
        return

    trainer.fit(
        num_epochs=remaining_epochs,
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
