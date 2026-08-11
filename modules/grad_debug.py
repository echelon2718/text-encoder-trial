"""
Instrumentasi debugging untuk melacak akar penyebab exploding gradient di
TLeJEPA. Dipakai dari modules/training.py (Trainer + train_step).

FILOSOFI (biar overhead-nya tidak numpuk di training yang lama):

  1. MURAH & SELALU JALAN tiap step (dipanggil dari train_step SETELAH
     backward(), SEBELUM clip_grad_norm_):
       - grad-norm per GRUP modul (embedding teks/fonem, tiap layer encoder,
         tiap layer decoder, length_predictor) + rasio
         grad-norm/weight-norm per grup -> dicatat ke TensorBoard.
       - Sinkronisasi GPU->CPU diminimalkan: nilai per-parameter dijumlahkan
         dulu SEBAGAI TENSOR (belum di-.item()), baru di-stack & di-transfer
         SEKALI per grup (bukan sekali per parameter) -- kalau tidak, ratusan
         panggilan .item() per step (satu per parameter) akan jadi titik
         sinkronisasi CPU<->GPU yang serius dan memperlambat training.

  2. MAHAL & CUMA JALAN SAAT ANOMALI TERDETEKSI (grad_norm non-finite ATAU
     melonjak jauh di atas rata-rata bergerak/EMA historisnya):
       - breakdown grad-norm PER PARAMETER individual (top-K terbesar),
       - statistik aktivasi forward TERAKHIR per layer (max|x|, mean|x|, std,
         apakah semuanya finite) -- direkam lewat forward hook yang jalan tiap
         forward pass, jadi begitu ada NaN/ledakan kita tahu PERSIS layer mana
         yang PERTAMA kali menghasilkan nilai non-finite/aneh, tanpa perlu
         mengulang training buat reproduksi,
       - ringkasan batch penyebab (id kalimat, panjang, cuplikan teks) supaya
         bisa dicek apakah pemicunya data spesifik (mis. kalimat yang sangat
         panjang/aneh setelah augmentasi),
       - semuanya ditulis ke: (a) console/tqdm, (b) TensorBoard (teks +
         scalar), dan (c) file .pt terpisah di <ckpt_dir>/anomaly_dumps/,
         supaya bisa dianalisis offline tanpa harus nunggu training selesai
         atau mengulang run.
     Ada cooldown + batas jumlah dump supaya kalau training BENAR-BENAR
     divergen (anomali beruntun ribuan step), disk tidak kebanjiran file dan
     console tidak banjir log.
"""
import os
import math
import time
from typing import Optional

import torch
import torch.nn as nn


def _module_group_name(param_name: str) -> str:
    """
    Kelompokkan nama parameter (dari named_parameters()) jadi grup yang
    manusiawi buat dibaca di laporan/TensorBoard, misal:
      "encoder.layers.3.self_attn.in_proj_weight" -> "encoder.layer_3"
      "decoder.layers.1.ffn.0.weight"              -> "decoder.layer_1"
      "text_embedding.weight"                      -> "text_embedding"
      "embed_norm.weight"                          -> "embed_norm"
      "length_predictor.head.0.weight"             -> "length_predictor"
      "decoder.input_norm.weight"                  -> "decoder.input_norm"
      "decoder.canonical_query_library.weight"     -> "decoder.canonical_query_library"
      "length_predictor.head.0.weight"             -> "length_predictor"
    """
    parts = param_name.split(".")
    if parts[0] in ("encoder", "decoder") and len(parts) >= 2:
        if parts[1] == "layers" and len(parts) >= 3:
            return f"{parts[0]}.layer_{parts[2]}"
        # Sub-modul langsung di encoder/decoder yang bukan stack layer (mis.
        # input_norm, canonical_query_library) -- dikasih grup sendiri-sendiri
        # supaya tidak tercampur jadi satu bucket generik "encoder"/"decoder"
        # yang sulit dibaca saat salah satu di antaranya jadi biang masalah.
        return f"{parts[0]}.{parts[1]}"
    if parts[0] == "criterion" and len(parts) >= 2:
        return ".".join(parts[:2])
    return parts[0]


class GradientDebugger:
    def __init__(
        self,
        model: nn.Module,
        criterion,
        writer,
        ckpt_dir: str,
        spike_ratio: float = 6.0,
        spike_zscore: float = 6.0,
        ema_decay: float = 0.98,
        warmup_steps: int = 20,
        max_report_params: int = 25,
        track_activations: bool = True,
        report_cooldown_steps: int = 5,
        max_dumps: int = 20,
        dump_dir_name: str = "anomaly_dumps",
    ):
        """
        spike_ratio / spike_zscore: step dianggap ANOMALI kalau
            total_grad_norm >= spike_ratio * EMA_historis  ATAU
            (total_grad_norm - EMA) / std_historis >= spike_zscore
            (yang mana pun lebih dulu terpenuhi), ATAU total_grad_norm
            non-finite (NaN/Inf) -- ini selalu dianggap anomali apa pun
            threshold-nya.
        warmup_steps: jumlah step pertama yang dipakai HANYA untuk membangun
            baseline EMA/std, belum dipakai untuk deteksi spike (supaya
            grad_norm yang wajar besar di awal training -- sebelum LR warmup
            selesai -- tidak langsung dicap "anomali").
        report_cooldown_steps: kalau anomali terjadi BERUNTUN, laporan detail
            (+ dump file) hanya ditulis ulang setiap N step -- di antaranya
            cukup dicatat ringkas ke TensorBoard supaya console/disk tidak
            banjir.
        max_dumps: batas jumlah file dump .pt yang disimpan ke disk per run.
        """
        self.model = model
        self.criterion = criterion
        self.writer = writer
        self.spike_ratio = spike_ratio
        self.spike_zscore = spike_zscore
        self.ema_decay = ema_decay
        self.warmup_steps = warmup_steps
        self.max_report_params = max_report_params
        self.track_activations = track_activations
        self.report_cooldown_steps = report_cooldown_steps
        self.max_dumps = max_dumps
        self.dump_dir = os.path.join(ckpt_dir, dump_dir_name)

        self._ema: Optional[float] = None
        self._ema_sq: Optional[float] = None
        self._n_seen = 0
        self.last_anomaly_step = -1
        self.n_anomalies_total = 0
        self._n_dumps_written = 0

        self._activation_stats: dict = {}
        self._activation_order: list = []
        self._hook_handles = []
        if track_activations:
            self._register_hooks()

    # ------------------------------------------------------------- hooks --
    def _register_hooks(self):
        named = dict(self.model.named_modules())
        targets = []
        for name in named:
            if name.startswith("encoder.layers.") and name.count(".") == 2:
                targets.append(name)
            elif name.startswith("decoder.layers.") and name.count(".") == 2:
                targets.append(name)
            elif name in ("text_embedding", "phoneme_embedding", "length_predictor"):
                targets.append(name)
        # Urutkan supaya laporan mengikuti urutan eksekusi forward yang wajar
        # (embedding -> encoder layer 0..N -> length_predictor -> decoder layer 0..N).
        order_key = {"text_embedding": -2, "phoneme_embedding": -2, "length_predictor": 998}
        def sort_key(n):
            if n in order_key:
                return (order_key[n], n)
            prefix, _, idx = n.rpartition(".")
            base = 0 if n.startswith("encoder.") else 500
            return (base + int(idx), n)
        targets.sort(key=sort_key)

        for name in targets:
            handle = named[name].register_forward_hook(self._make_hook(name))
            self._hook_handles.append(handle)


    def _make_hook(self, name: str):
        if name not in self._activation_order:
            self._activation_order.append(name)

        def hook(module, inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(t):
                return
            with torch.no_grad():
                t_f = t.detach()
                finite_mask = torch.isfinite(t_f)
                all_finite = bool(finite_mask.all())
                if all_finite:
                    max_abs = t_f.abs().max().item()
                    mean_abs = t_f.abs().mean().item()
                    std = t_f.std().item() if t_f.numel() > 1 else 0.0
                else:
                    finite_vals = t_f[finite_mask]
                    n_valid = finite_vals.numel()
                    max_abs = finite_vals.abs().max().item() if n_valid else float("nan")
                    mean_abs = finite_vals.abs().mean().item() if n_valid else float("nan")
                    std = finite_vals.std().item() if n_valid > 1 else float("nan")
                self._activation_stats[name] = {
                    "max_abs": max_abs,
                    "mean_abs": mean_abs,
                    "std": std,
                    "all_finite": all_finite,
                    "n_nonfinite": int((~finite_mask).sum().item()),
                    "shape": tuple(t_f.shape),
                }
        return hook

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    # -------------------------------------------------------- grad norms --
    def compute_grad_norms(self, full: bool = False):
        """
        full=False (dipanggil TIAP step): cuma total_norm + per-grup, dengan
        MINIMAL sinkronisasi GPU->CPU (satu .cpu()/.tolist() untuk SEMUA grup
        sekaligus, bukan satu per parameter).
        full=True (HANYA dipanggil saat anomali terdeteksi, jadi jarang):
        tambahan breakdown per-parameter individual untuk laporan detail.

        Return: (total_norm: float, per_group_norm: dict, per_param_norm:
        dict | None, per_group_weight_norm: dict)
        """
        group_sq: dict = {}
        weight_group_sq: dict = {}
        per_param_sq = {} if full else None

        # criterion tidak lagi punya parameter sendiri: proj_head hilang bersama
        # magnitude_loss (paper 5.8), jadi seluruh parameter ada di model.
        all_named = list(self.model.named_parameters())

        for name, p in all_named:
            if p.grad is None:
                continue
            grp = _module_group_name(name)
            g_sq = p.grad.detach().float().pow(2).sum()  # tetap tensor GPU, belum sync
            w_sq = p.detach().float().pow(2).sum()
            group_sq[grp] = g_sq if grp not in group_sq else group_sq[grp] + g_sq
            weight_group_sq[grp] = w_sq if grp not in weight_group_sq else weight_group_sq[grp] + w_sq
            if full:
                per_param_sq[name] = g_sq

        if not group_sq:
            return 0.0, {}, ({} if full else None), {}

        group_names = list(group_sq.keys())
        group_norms = torch.stack([group_sq[g] for g in group_names]).sqrt().cpu().tolist()
        weight_norms = torch.stack([weight_group_sq[g] for g in group_names]).sqrt().cpu().tolist()
        per_group_norm = dict(zip(group_names, group_norms))
        per_group_weight_norm = dict(zip(group_names, weight_norms))
        total_norm = sum(v * v for v in group_norms) ** 0.5

        per_param_norm = None
        if full and per_param_sq:
            p_names = list(per_param_sq.keys())
            p_norms = torch.stack([per_param_sq[n] for n in p_names]).sqrt().cpu().tolist()
            per_param_norm = dict(zip(p_names, p_norms))

        return total_norm, per_group_norm, per_param_norm, per_group_weight_norm

    # ------------------------------------------------------------ EMA/spike
    def _update_ema(self, value: float):
        self._n_seen += 1
        if self._ema is None:
            self._ema = value
            self._ema_sq = value * value
        else:
            self._ema = self.ema_decay * self._ema + (1 - self.ema_decay) * value
            self._ema_sq = self.ema_decay * self._ema_sq + (1 - self.ema_decay) * (value * value)

    def _current_std(self) -> float:
        if self._ema is None or self._ema_sq is None:
            return 0.0
        return max(self._ema_sq - self._ema ** 2, 0.0) ** 0.5

    def is_spike(self, total_norm: float) -> bool:
        if not math.isfinite(total_norm):
            return True
        if self._n_seen < self.warmup_steps or self._ema is None or self._ema <= 0:
            return False
        ratio = total_norm / max(self._ema, 1e-12)
        std = self._current_std()
        zscore = (total_norm - self._ema) / max(std, 1e-12)
        return ratio >= self.spike_ratio or zscore >= self.spike_zscore

    # ------------------------------------------------------------- logging
    def log_lightweight(self, step: int, total_norm: float, per_group_norm: dict,
                         per_group_weight_norm: dict):
        """Dipanggil sesuai cadence logging biasa (mis. tiap log_every_n_steps)."""
        for grp, gnorm in per_group_norm.items():
            self.writer.add_scalar(f"grad_debug/norm/{grp}", gnorm, step)
            wnorm = per_group_weight_norm.get(grp, 0.0)
            if wnorm > 0:
                self.writer.add_scalar(f"grad_debug/grad_to_weight_ratio/{grp}", gnorm / wnorm, step)
        self.writer.add_scalar("grad_debug/total_grad_norm_raw", total_norm, step)
        self.writer.add_scalar("grad_debug/ema_grad_norm", self._ema or 0.0, step)
        self.writer.add_scalar("grad_debug/ema_std", self._current_std(), step)

    def observe(self, step: int, total_norm: float, per_group_norm: dict,
                per_group_weight_norm: dict, losses: dict, batch: dict) -> bool:
        """
        Dipanggil TIAP step setelah backward(), SEBELUM clip_grad_norm_.
        Return True kalau step ini terdeteksi anomali.
        """
        if not self.is_spike(total_norm):
            self._update_ema(total_norm)
            return False

        self.n_anomalies_total += 1
        in_cooldown = (
            self.last_anomaly_step >= 0
            and (step - self.last_anomaly_step) < self.report_cooldown_steps
        )
        self.last_anomaly_step = step
        # EMA SENGAJA TIDAK di-update dengan nilai anomali -- supaya baseline
        # "normal" tidak ikut tercemar oleh lonjakan yang justru mau dideteksi.

        if in_cooldown or self._n_dumps_written >= self.max_dumps:
            reason = "cooldown" if in_cooldown else f"sudah {self.max_dumps} dump"
            self._emit(
                f"[GradientDebugger] step {step}: grad_norm={total_norm:.4e} masih anomali "
                f"(EMA={self._ema if self._ema else float('nan'):.4e}) -- laporan detail "
                f"dilewati ({reason}), cuma dicatat ringkas."
            )
            self.writer.add_scalar(
                "anomaly/total_grad_norm", total_norm if math.isfinite(total_norm) else -1.0, step
            )
            return True

        _, _, per_param_norm, _ = self.compute_grad_norms(full=True)
        self._write_full_report(
            step, total_norm, per_group_norm, per_param_norm or {}, per_group_weight_norm,
            losses, batch, header="ANOMALI grad_norm (SETELAH backward)",
        )
        return True

    def report_forward_anomaly(self, step: int, losses: dict, batch: dict):
        """
        Loss itu sendiri sudah NaN/Inf SEBELUM backward() dipanggil -- akar
        masalahnya ada di FORWARD (bukan gradien meledak). Dilaporkan
        terpisah dari observe() supaya jelas bedanya di laporan.

        Pakai proteksi cooldown/max_dumps yang SAMA dengan observe() --
        kalau training sudah benar-benar collapse dan forward menghasilkan
        NaN di ribuan step berturut-turut, kita tidak mau disk kebanjiran
        file dump ataupun console banjir laporan panjang tiap step.
        """
        self.n_anomalies_total += 1
        in_cooldown = (
            self.last_anomaly_step >= 0
            and (step - self.last_anomaly_step) < self.report_cooldown_steps
        )
        self.last_anomaly_step = step

        if in_cooldown or self._n_dumps_written >= self.max_dumps:
            reason = "cooldown" if in_cooldown else f"sudah {self.max_dumps} dump"
            self._emit(
                f"[GradientDebugger] step {step}: loss NaN/Inf lagi (forward) -- "
                f"laporan detail dilewati ({reason}), cuma dicatat ringkas."
            )
            self.writer.add_scalar("anomaly/total_grad_norm", -1.0, step)
            return

        self._write_full_report(
            step, float("nan"), {}, {}, {}, losses, batch,
            header="LOSS NaN/Inf SEBELUM backward (masalah di FORWARD, bukan gradien)",
        )

    # ------------------------------------------------------------- report --
    def _loss_lines(self, losses: dict) -> list:
        lines = ["-- Loss komponen (nilai MENTAH, sebelum bobot zeta/lambda) --"]
        for k, v in losses.items():
            if k == "_skipped":
                continue
            try:
                # compute_losses() mengembalikan "total" TANPA di-.detach()
                # (masih terhubung ke graph) -- detach dulu di sini supaya
                # tidak memicu warning/autograd overhead cuma buat logging.
                if torch.is_tensor(v):
                    v = v.detach()
                lines.append(f"  {k:28s} = {float(v):.6e}")
            except (TypeError, ValueError):
                pass
        return lines

    def _activation_lines(self) -> tuple:
        lines = ["-- Statistik aktivasi forward TERAKHIR per layer (urutan eksekusi) --"]
        first_bad = None
        for name in self._activation_order:
            stat = self._activation_stats.get(name)
            if stat is None:
                continue
            flag = "" if stat["all_finite"] else "  <-- NON-FINITE DI SINI"
            if not stat["all_finite"] and first_bad is None:
                first_bad = name
            lines.append(
                f"  {name:28s} shape={stat['shape']} max|x|={stat['max_abs']:.4e} "
                f"mean|x|={stat['mean_abs']:.4e} std={stat['std']:.4e}{flag}"
            )
        if first_bad is not None:
            lines.append(f"  >>> Kandidat titik PERTAMA nilai non-finite muncul (forward): '{first_bad}' <<<")
        elif not lines[1:]:
            lines.append("  (tidak ada hook aktivasi terpasang / belum ada forward pass tercatat)")
        return lines, first_bad

    def _describe_batch(self, batch: dict, max_samples: Optional[int] = None) -> list:
        """
        max_samples=None (default): tampilkan SEMUA sample di batch, BUKAN cuma
        10 pertama -- dan tampilkan SEMUA view (kanonik + tiap augmentasi) secara
        UTUH (tidak dipotong), supaya kamu bisa langsung baca kalimat mana yang
        memicu ledakan, bukan cuma tahu nama layernya.

        Diurutkan dari sekuens TERPANJANG di tiap sample -- karena secara empiris
        sekuens terpanjang dalam batch adalah kandidat pertama yang paling masuk
        akal (attention/FFN kerja paling berat di situ, dan L2-norm gradien FFN
        biasanya naik ~proporsional dengan jumlah token yang diproses).
        """
        lines = ["-- Info batch penyebab (SEMUA sample & SEMUA view teks, diurutkan dari sekuens TERPANJANG) --"]
        try:
            ids = batch.get("id", []) or []
            texts = batch.get("texts", []) or []
            x_lengths = batch.get("x_lengths", []) or []
            n = len(ids)
            lines.append(f"  batch_size={n}")

            all_lens = [l for lengths in x_lengths for l in (lengths or [])]
            if all_lens:
                lines.append(
                    f"  panjang sekuens gabungan semua view/sample: "
                    f"max={max(all_lens)}  mean={sum(all_lens) / len(all_lens):.1f}  min={min(all_lens)}"
                )

            order = sorted(
                range(n),
                key=lambda i: max(x_lengths[i]) if i < len(x_lengths) and x_lengths[i] else 0,
                reverse=True,
            )
            limit = n if max_samples is None else min(max_samples, n)
            for rank, i in enumerate(order[:limit]):
                length_info = x_lengths[i] if i < len(x_lengths) else []
                sample_texts = texts[i] if i < len(texts) else []
                max_len = max(length_info) if length_info else 0
                lines.append(f"  [{rank + 1:>3}] id={ids[i]!r}  panjang_maks_sample_ini={max_len}")
                for v, l in enumerate(length_info):
                    t = sample_texts[v] if v < len(sample_texts) else "?"
                    tag = "canonical (view_0, ini yg dipakai canon_len_loss)" if v == 0 else f"aug_view_{v}"
                    lines.append(f"         view_{v:<2} len={l:<5} [{tag}]: {t!r}")
            if limit < n:
                lines.append(
                    f"    ... ({n - limit} sample lain tidak ditampilkan di console, "
                    f"tapi TETAP LENGKAP ADA di file dump .pt)"
                )
        except Exception as e:
            lines.append(f"  (gagal ekstrak info batch: {e!r})")
        return lines

    def _write_full_report(self, step, total_norm, per_group_norm, per_param_norm,
                            per_group_weight_norm, losses, batch, header: str):
        ema_str = f"{self._ema:.4e}" if self._ema is not None else "n/a (masih warmup)"
        std_str = f"{self._current_std():.4e}" if self._ema is not None else "n/a"
        lines = [
            f"\n{'=' * 78}",
            f"[GradientDebugger] {header} -- step {step}",
            f"  total_grad_norm={total_norm:.4e} | EMA_sebelumnya={ema_str} | std={std_str} | "
            f"anomali_ke-{self.n_anomalies_total}",
            f"{'=' * 78}",
        ]
        lines += self._loss_lines(losses)

        if per_group_norm:
            lines.append("-- Grad-norm per grup modul (urut terbesar -> paling mencurigakan) --")
            for grp, gnorm in sorted(per_group_norm.items(), key=lambda x: -x[1]):
                wnorm = per_group_weight_norm.get(grp, 0.0)
                ratio = gnorm / wnorm if wnorm > 0 else float("nan")
                finite = "" if math.isfinite(gnorm) else "  <-- NON-FINITE"
                lines.append(
                    f"  {grp:28s} grad_norm={gnorm:.4e}  weight_norm={wnorm:.4e}  "
                    f"grad/weight={ratio:.4e}{finite}"
                )

        if per_param_norm:
            lines.append(f"-- Top {self.max_report_params} parameter individual by grad-norm --")
            ranked = sorted(
                per_param_norm.items(),
                key=lambda x: (-x[1] if math.isfinite(x[1]) else float("-inf")),
            )
            for name, gnorm in ranked[: self.max_report_params]:
                finite = "OK" if math.isfinite(gnorm) else "NON-FINITE"
                lines.append(f"  [{finite:11s}] {name:60s} grad_norm={gnorm:.4e}")

        if self.track_activations:
            act_lines, first_bad = self._activation_lines()
            lines += act_lines
        else:
            first_bad = None

        lines += self._describe_batch(batch)

        report = "\n".join(lines)
        self._emit(report)
        self.writer.add_text("anomaly/report", "```\n" + report + "\n```", global_step=step)
        self.writer.add_scalar(
            "anomaly/total_grad_norm", total_norm if math.isfinite(total_norm) else -1.0, step
        )
        self._dump_to_disk(step, total_norm, per_group_norm, per_param_norm, losses, batch, first_bad)

    def _emit(self, text: str):
        try:
            from tqdm.auto import tqdm
            tqdm.write(text)
        except Exception:
            print(text)

    def _dump_to_disk(self, step, total_norm, per_group_norm, per_param_norm, losses, batch, first_bad_layer):
        try:
            os.makedirs(self.dump_dir, exist_ok=True)
            payload = {
                "step": step,
                "total_grad_norm": total_norm,
                "per_group_grad_norm": per_group_norm,
                "per_param_grad_norm": per_param_norm,
                "activation_stats": dict(self._activation_stats),
                "first_nonfinite_layer": first_bad_layer,
                "losses": {
                    k: (float(v) if not isinstance(v, bool) else v)
                    for k, v in losses.items() if k != "_skipped"
                },
                "batch_ids": batch.get("id", None),
                "batch_texts": batch.get("texts", None),
                "batch_x_lengths": batch.get("x_lengths", None),
                "timestamp": time.time(),
            }
            path = os.path.join(self.dump_dir, f"anomaly_step_{step}.pt")
            torch.save(payload, path)
            self._n_dumps_written += 1
            self._emit(f"[GradientDebugger] Detail lengkap disimpan ke: {path}")
        except Exception as e:
            self._emit(f"[GradientDebugger] Gagal menyimpan dump anomali: {e!r}")

    def summary(self) -> dict:
        return {
            "n_anomalies_total": self.n_anomalies_total,
            "n_dumps_written": self._n_dumps_written,
            "last_anomaly_step": self.last_anomaly_step,
            "final_ema_grad_norm": self._ema,
        }
