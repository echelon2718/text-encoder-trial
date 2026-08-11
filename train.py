"""
Entry point pelatihan TLeJEPA.

Penamaan bobot mengikuti paper Persamaan (Bagian 5.8.6):

    L_total = (1 - lambda) * (zeta_syn L_syn + zeta_sem L_sem) / sum(zeta)
              + lambda * L_SIGReg
              + gamma_len * L_len

`lambda` KHUSUS trade-off SIGReg; bobot komponen prediktif memakai `zeta`.
Keduanya tidak boleh dipertukarkan.
"""

import argparse
import math

import torch
from torch.utils.data import DataLoader
from functools import partial

from data.dataset import BatchSampler, collate_fn
from data.dataset import AugmentDataset
from modules.tlejepa import TLeJEPA
from modules.training import Trainer, find_latest_run_dir
from modules.losses import (
    TLeJEPACriterion,
    SIGReg,
    SemanticTeacherSimilarity,
    CachedSemanticTeacherSimilarity,
)


def get_args():
    p = argparse.ArgumentParser(description="Train TLeJEPA")

    # --- Data ---
    p.add_argument("--lexicon-path", type=str, default="./lexicon/abbrev-lexicon.json")
    p.add_argument("--dataset-path", type=str, default="./data/english_singlish_g2p.csv")
    p.add_argument("--dataset-mode", type=str, default="huggingface", choices=["local", "huggingface"])
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--n_singlish", type=int, default=3)
    p.add_argument("--n_premise_and_negation", type=int, default=6,
                   help="K = jumlah pasangan negasi per batch. Ini juga menentukan "
                        "negation_offset = batch_size - K pada semantic_loss.")

    # --- Run ---
    p.add_argument("--run-label", type=str, default="run")
    p.add_argument("--ckpt-root", type=str, default="checkpoints")
    p.add_argument("--log-dir", type=str, default="runs/tlejepa")
    p.add_argument("--resume", action="store_true",
                   help="Lanjutkan dari checkpoint TERBARU utk --run-label ini di --ckpt-root "
                        "(urutan pencarian: latest_step.pt > latest_model.pt > best_model.pt)")

    # --- Arsitektur (paper 6.2) ---
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--heads", type=int, default=8,
                   help="d_h = d_model / heads dipertahankan 64 di semua ukuran model, "
                        "supaya penskalaan tidak mencampur efek granularitas attention.")
    p.add_argument("--enc-layers", type=int, default=8)
    p.add_argument("--dec-layers", type=int, default=6)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-deepnorm", action="store_true")
    p.add_argument("--encoder-only", action="store_true",
                   help="R11: buang decoder, proyektor kanvas, dan prediktor panjang. "
                        "Objektif bekerja pada Pool(H). Setara LeJEPA polos pada teks.")
    p.add_argument("--sigreg-space", type=str, default="encoder", choices=["encoder", "decoder"],
                   help="Ruang tempat SIGReg bekerja. 'decoder' adalah ablasi R10.")
    p.add_argument("--empty-norm-ratio", type=float, default=0.35,
                   help="Ambang relatif deteksi batas konten saat inferensi. Bertumpu pada "
                        "L_empty yang menarik wilayah kosong ke norma nol; menggantikan "
                        "m_content yang tidak tersedia tanpa L^A.")
    p.add_argument("--c-max", type=float, default=2.0,
                   help="Pagar ekspansi kanvas relatif terhadap L^A. Membatasi routing saja; "
                        "tidak mengubah gradien karena jalur rho->L_hat sudah diskrit.")
    p.add_argument("--tau-r", type=float, default=0.3,
                   help="Temperatur Gaussian temporal resampling (paper 5.5). Makin kecil, "
                        "makin tajam bobot resampling ke satu posisi sumber.")

    # --- Anchor & objektif (paper 5.8) ---
    p.add_argument("--canon-type", type=str, default="phoneme", choices=["phoneme", "text"],
                   help="Modalitas anchor. Mengatur SEKALIGUS view 0 dataset dan "
                        "tabel embedding yang dipakai model -- keduanya harus konsisten.")
    p.add_argument("--w-canon", type=float, default=8.0,
                   help="Bobot view kanonik pada target rerata berbobot mu (paper 5.8.1). "
                        "w_canon=1.0 -> rerata rata (ablasi M6); w_canon -> inf -> hard anchor.")
    p.add_argument("--eta-empty", type=float, default=1.0,
                   help="Bobot objektif wilayah kosong. Hanya aktif kalau mixed-length menyala.")
    p.add_argument("--lambda_", type=float, default=0.5,
                   help="Trade-off SIGReg. JANGAN dipakai untuk bobot syn/sem.")
    p.add_argument("--lambda-warmup-steps", type=int, default=0)
    p.add_argument("--zeta-syn", type=float, default=1.0)
    p.add_argument("--zeta-sem", type=float, default=1.0)
    p.add_argument("--zeta-sem-neg", type=float, default=1.0)
    p.add_argument("--gamma-len", type=float, default=1.0)
    p.add_argument("--tau-l", type=float, default=0.9,
                   help="Kuantil pinball loss panjang (paper 5.8.5). 0.9 -> underestimate "
                        "dihukum 9x lebih berat, jadi prediktor bias ke atas terkendali.")
    p.add_argument("--pure-latent-anchor", action="store_true",
                   help="Ablasi M7: target L_syn = keluaran tabel embedding anchor "
                        "(dibekukan), bukan rerata berbobot lintas view. Ini objektif "
                        "REKONSTRUKSI, bukan JEPA -- dipakai justru untuk mengujinya.")
    p.add_argument("--stopgrad-canon", action="store_true",
                   help="Detach view kanonik pada L_syn. Opsi ablasi; paper default TIDAK detach.")
    p.add_argument("--no-normalize-obj-weights", action="store_true",
                   help="Matikan normalisasi bobot cabang prediktif. Tanpa normalisasi, "
                        "mematikan satu suku saat ablasi diam-diam menaikkan bobot relatif "
                        "SIGReg -- dua variabel berubah sekaligus.")

    # --- SIGReg (paper 5.8.3: ruang encoder, level kalimat) ---
    p.add_argument("--num-slices", type=int, default=256)
    p.add_argument("--sigreg-knots", type=int, default=17)
    p.add_argument("--sigreg-t-max", type=float, default=3.0)

    # --- Mixed-length canvas (paper 5.7 Tahap 7) ---
    p.add_argument("--tf-warmup-frac", type=float, default=0.3,
                   help="Fraksi total step yang seluruhnya teacher-forced (p_TF = 1).")
    p.add_argument("--tf-end-frac", type=float, default=0.6,
                   help="Fraksi total step saat p_TF mencapai --p-tf-min.")
    p.add_argument("--p-tf-min", type=float, default=0.5,
                   help="p_TF minimum. Default 1.0 = mixed-length NONAKTIF (kanvas selalu "
                        "panjang anchor). Set mis. 0.5 untuk mengaktifkannya.")

    # --- Optimisasi ---
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.98)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=0,
                   help="Batas langkah global (0 = tidak dibatasi). Dipakai untuk protokol "
                        "token-matched antar ukuran model: batch dan urutan data sama, "
                        "jadi langkah sama berarti token sama.")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--visualize-every-n-steps", type=int, default=1000)
    p.add_argument("--save-every-n-steps", type=int, default=1000)

    # --- Pemantauan (dicatat ke events.log, bukan email) ---
    p.add_argument("--grad-ema-threshold", type=float, default=50.0,
                   help="Ambang EMA grad-norm. Dicatat kalau EMA (bukan grad-norm sesaat) "
                        "di atas ambang selama N cek berturut-turut.")
    p.add_argument("--grad-breach-required", type=int, default=5)
    p.add_argument("--research-every-n-steps", type=int, default=2000,
                   help="Interval perhitungan metrik riset (H1/H3/H7/H9 + RankMe) "
                        "yang dikirim ke TensorBoard. 0 untuk mematikan.")

    # --- Debug ---
    p.add_argument("--no-debug-gradients", action="store_true")
    p.add_argument("--no-debug-activations", action="store_true")
    p.add_argument("--debug-spike-ratio", type=float, default=6.0)
    p.add_argument("--debug-spike-zscore", type=float, default=6.0)
    p.add_argument("--debug-warmup-steps", type=int, default=20)
    p.add_argument("--debug-max-dumps", type=int, default=20)

    return p.parse_args()


def build_lr_scheduler(optimizer, total_steps: int, warmup_ratio: float = 0.05,
                       min_lr_ratio: float = 0.1):
    """Linear warmup lalu cosine decay ke `min_lr_ratio * lr_puncak`, per-step."""
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def estimate_rho_init(loader, max_batches: int = 20) -> float:
    """
    Rerata log-rasio ekspansi dataset, dipakai sebagai bias awal head panjang
    (paper 5.6.4). Dengan ini model bermula sebagai prediktor rasio rerata
    global, bukan keluaran acak.

    Dihitung dari mask saja (tanpa forward model), jadi murah.
    """
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        mask = batch["mask"]                       # [B,V,L]
        src_len = mask.sum(dim=2).float().clamp(min=1.0)      # [B,V]
        l_anchor = mask[:, 0, :].sum(dim=1).float().clamp(min=1.0)  # [B]
        rho = torch.log(l_anchor).unsqueeze(1) - torch.log(src_len)
        total += float(rho.sum())
        count += rho.numel()
    if count == 0:
        return 0.0
    return total / count


def main(args):
    resume_dir = None
    if args.resume:
        resume_dir = find_latest_run_dir(args.ckpt_root, args.run_label)
        if resume_dir is None:
            raise FileNotFoundError(
                f"--resume dipakai tapi tidak ada folder checkpoint utk "
                f"--run-label='{args.run_label}' di dalam '{args.ckpt_root}'."
            )
        print(f"[main] Ditemukan checkpoint terbaru utk '{args.run_label}': {resume_dir}")
        print("[main] PENTING: pastikan semua argumen arsitektur dan --epochs "
              "SAMA PERSIS dengan run aslinya.")

    if args.d_model % args.heads != 0:
        raise ValueError(f"d_model={args.d_model} tidak habis dibagi heads={args.heads}")
    d_head = args.d_model // args.heads
    if d_head != 64:
        print(f"[main] PERINGATAN: d_head={d_head} (bukan 64). Paper 6.2 menetapkan "
              f"d_h konstan 64 di seluruh ukuran model supaya penskalaan bersih.")

    ds_kwargs = dict(lexicon_path=args.lexicon_path, canon_mode=args.canon_type)
    full_dataset = AugmentDataset(
        dataset_path=args.dataset_path,
        mode=args.dataset_mode,
        **ds_kwargs,
    )

    train_size = int(args.train_ratio * len(full_dataset))
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(full_dataset), generator=g).tolist()
    train_idx, val_idx = perm[:train_size], perm[train_size:]

    # `.select()` (bukan pandas `.iloc`): tetap Arrow-backed & di-cache ke disk
    # oleh `datasets`, jadi aman di-share lintas worker tanpa memori dobel.
    train_hf = full_dataset.dataset.select(train_idx)
    val_hf = full_dataset.dataset.select(val_idx)
    shared_tokenizer = full_dataset.phoneme_tokenizer
    del full_dataset

    train_dataset = AugmentDataset(hf_dataset=train_hf, tokenizer=shared_tokenizer, **ds_kwargs)
    val_dataset = AugmentDataset(hf_dataset=val_hf, tokenizer=shared_tokenizer, **ds_kwargs)

    sampler_kwargs = dict(
        batch_size=args.batch_size,
        n_singlish_per_batch=args.n_singlish,
        n_negation_per_batch=args.n_premise_and_negation,
        seed=args.seed,
    )
    train_sampler = BatchSampler(dataset=train_dataset.dataset, drop_last=True, **sampler_kwargs)
    val_sampler = BatchSampler(dataset=val_dataset.dataset, drop_last=False, **sampler_kwargs)

    loader_extra = ({"persistent_workers": True, "prefetch_factor": args.prefetch_factor}
                    if args.num_workers > 0 else {})
    make_loader = lambda ds, smp: DataLoader(
        ds, batch_sampler=smp, num_workers=args.num_workers,
        collate_fn=partial(collate_fn, max_seq_len=args.max_length),
        pin_memory=True, **loader_extra)

    train_loader = make_loader(train_dataset, train_sampler)
    val_loader = make_loader(val_dataset, val_sampler)

    # Bias awal head panjang. Dihitung SEBELUM model dibangun supaya bisa
    # langsung dipasang di konstruktor.
    rho_init = estimate_rho_init(val_loader, max_batches=10)
    print(f"[main] rho_init (rerata log-rasio ekspansi dataset) = {rho_init:.4f} "
          f"-> rasio ekspansi rerata ~{math.exp(rho_init):.3f}x")

    model = TLeJEPA(
        n_vocab_text=256,
        n_vocab_phoneme=shared_tokenizer.vocab_size,
        d_model=args.d_model,
        n_attn_heads=args.heads,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        max_length=args.max_length,
        dropout=args.dropout,
        use_deepnorm=not args.no_deepnorm,
        tau_r=args.tau_r,
        rho_init=rho_init,
        pure_latent_anchor=args.pure_latent_anchor,
        encoder_only=args.encoder_only,
        c_max=args.c_max,
        empty_norm_ratio=args.empty_norm_ratio,
    )
    model.set_gradient_checkpointing(True)
    c = model.deepnorm_coeffs
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[main] DeepNorm={'ON' if not args.no_deepnorm else 'OFF'} "
          f"(N={args.enc_layers}, M={args.dec_layers}) "
          f"enc alpha={c['enc_alpha']:.4f} beta={c['enc_beta']:.4f} | "
          f"dec alpha={c['dec_alpha']:.4f} beta={c['dec_beta']:.4f}")
    print(f"[main] Parameter: {n_params/1e6:.2f} M | d_head={d_head} | anchor={args.canon_type}")

    criterion = TLeJEPACriterion(
        sigreg_fn=SIGReg(knots=args.sigreg_knots, num_slices=args.num_slices,
                         t_max=args.sigreg_t_max),
        cossim_fn=CachedSemanticTeacherSimilarity(SemanticTeacherSimilarity()),
        lambda_=args.lambda_,
        zeta_syn=args.zeta_syn,
        zeta_sem=args.zeta_sem,
        zeta_sem_neg=args.zeta_sem_neg,
        gamma_len=args.gamma_len,
        w_canon=args.w_canon,
        eta_empty=args.eta_empty,
        tau_l=args.tau_l,
        stopgrad_canon=args.stopgrad_canon,
        sigreg_space=args.sigreg_space,
        normalize_obj_weights=not args.no_normalize_obj_weights,
    )

    # Weight decay TIDAK dikenakan ke parameter ber-ndim < 2 (bias dan bobot
    # LayerNorm). Kalau gamma LayerNorm ikut di-decay, terjadi tarik-menarik
    # tepat pada parameter yang jadi pusat masalah skala.
    named = list(model.named_parameters())
    decay = [p for _, p in named if p.requires_grad and p.ndim >= 2]
    no_decay = [p for _, p in named if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(args.beta1, args.beta2),
    )
    print(f"[main] AdamW: {len(decay)} tensor ber-decay, {len(no_decay)} tanpa decay")

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    lr_scheduler = build_lr_scheduler(
        optimizer, total_steps=total_steps,
        warmup_ratio=args.warmup_ratio, min_lr_ratio=args.min_lr_ratio)

    # Jadwal mixed-length canvas dinyatakan sebagai fraksi total step supaya
    # tidak perlu disetel ulang tiap kali panjang training berubah.
    if args.encoder_only:
        print("[main] Mode encoder-only (R11): tanpa decoder, tanpa prediktor panjang.")
    else:
        model.set_length_schedule(
            warmup_steps=int(args.tf_warmup_frac * total_steps),
            end_steps=int(args.tf_end_frac * total_steps),
            p_min=args.p_tf_min,
        )
    print(f"[main] total_steps={total_steps} | mixed-length: "
          f"{'NONAKTIF' if args.p_tf_min >= 1.0 else f'p_TF 1.0 -> {args.p_tf_min}'}")

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
        log_dir=args.log_dir,
        ckpt_dir=args.ckpt_root,
        resume_dir=resume_dir,
        lr_scheduler=lr_scheduler,
        canon_type=args.canon_type,
        grad_clip=args.grad_clip,
        save_every_n_steps=args.save_every_n_steps,
        visualize_every_n_steps=(args.visualize_every_n_steps or None),
        lambda_warmup_steps=(args.lambda_warmup_steps or None),
        max_steps=(args.max_steps or None),
        grad_ema_threshold=args.grad_ema_threshold,
        grad_breach_required=args.grad_breach_required,
        research_every_n_steps=(args.research_every_n_steps or None),
        augmenter=train_dataset.augmenter,
        n_aug_1=train_dataset.n_aug_1,
        debug_gradients=not args.no_debug_gradients,
        debug_track_activations=not args.no_debug_activations,
        debug_spike_ratio=args.debug_spike_ratio,
        debug_spike_zscore=args.debug_spike_zscore,
        debug_warmup_steps=args.debug_warmup_steps,
        debug_max_dumps=args.debug_max_dumps,
    )

    # --epochs = TOTAL epoch yang ditarget (bukan tambahan di atas yang sudah
    # jalan). trainer.resume_epoch == 1 utk run baru, jadi rumus ini benar
    # untuk kedua kasus.
    remaining = max(0, args.epochs - (trainer.resume_epoch - 1))
    if remaining == 0:
        print(f"[main] --epochs={args.epochs} sudah tercapai "
              f"(resume_epoch={trainer.resume_epoch}).")
        return

    try:
        trainer.fit(num_epochs=remaining)
    except BaseException as e:
        # Termasuk SystemExit dari signal handler. _on_death sudah menyimpan
        # checkpoint; di sini hanya mencatat lalu melempar ulang supaya exit code
        # non-nol terbaca oleh watchdog di slurm/watchdog.sh.
        import traceback
        if not isinstance(e, SystemExit):
            trainer._on_death(f"exception:{type(e).__name__}")
            trainer._record("CRASH", traceback.format_exc().replace("\n", " | ")[:800])
        raise
    finally:
        trainer.close()


if __name__ == "__main__":
    main(get_args())
