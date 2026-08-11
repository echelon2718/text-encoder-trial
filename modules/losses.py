"""
Objektif TLeJEPA -- implementasi ketat paper Bagian 5.8.

    L_total = (1 - lambda) * (zeta_syn L_syn + zeta_sem L_sem) / (zeta_syn + zeta_sem)
              + lambda * L_SIGReg
              + gamma_len * L_len

Perubahan pokok terhadap versi lama:

  * magnitude_loss DIHAPUS beserta TeacherMagnitudeProjection (paper 5.8).
  * SIGReg pindah ke RUANG ENCODER dan hanya level KALIMAT (paper 5.8.3).
    Varian token-level dan varian pada ruang decoder dihapus.
  * syntactical_loss kembali ke MSE dengan target rerata berbobot mu
    (paper 5.8.1). Bentuk NMSE/rasio dari versi lama dibuang.
  * canon_len_loss jadi asymmetric quantile (pinball) atas LOG-RASIO
    (paper 5.8.5), bukan MSE simetris atas log1p panjang absolut.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import move_batch_to_device


# ---------------------------------------------------------------------------
# Utilitas numerik
# ---------------------------------------------------------------------------
def masked_mean(z: torch.Tensor, mask: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Pembagi di sini adalah CACAHAN token, jadi lantainya 1.0 (bukan 1e-6)."""
    mask_f = mask.unsqueeze(-1).to(z.dtype)
    summed = (z * mask_f).sum(dim=-2)
    count = mask_f.sum(dim=-2).clamp(min=eps)
    return summed / count


def safe_normalize(x: torch.Tensor, dim: int = -1, floor: float = 1e-3) -> torch.Tensor:
    """
    F.normalize(eps=1e-6) memberi gradien sebesar 1/||x|| yang bisa mencapai 1e6
    kalau ||x|| runtuh. Lantai aditif membatasi penguatan ke 1/floor = 1e3 dan
    tidak punya diskontinuitas seperti clamp.
    """
    return x / (x.norm(p=2, dim=dim, keepdim=True) + floor)


def effective_rank(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    erank(X) = exp(-sum_k p_k log p_k), p_k = sigma_k / sum sigma  (Roy & Vetterli).
    Dipakai sebagai diagnostik RankMe (paper 5.9.6): dilaporkan TERPISAH untuk
    H (ruang ter-SIGReg) dan Z (ruang decoder, di luar regularisasi).
    """
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    try:
        sigma = torch.linalg.svdvals(x)
    except Exception:
        return torch.tensor(float("nan"), device=x.device)
    p = sigma / sigma.sum().clamp(min=eps)
    p = p.clamp(min=eps)
    return torch.exp(-(p * p.log()).sum())


# ---------------------------------------------------------------------------
# 5.8.1  Canonical / syntactical prediction loss
# ---------------------------------------------------------------------------
def syntactical_loss(out: dict, w_canon: float = 8.0,
                     eta_empty: float = 1.0,
                     stopgrad_canon: bool = False,
                     tau_r: float = 0.3) -> torch.Tensor:
    """
    Paper 5.8.1.

    Target adalah rerata BERBOBOT lintas view, dihitung pada KOORDINAT KANONIK
    berpanjang L^A:

        mu_{n,i} = (w_canon Z^ref_{n,0,i} + sum_{v>=1} Z^ref_{n,v,i})
                   / (w_canon + V' - 1)

    Z^ref adalah keluaran decoder yang sudah diselaraskan dari kanvasnya
    masing-masing ke L^A. Penyelarasan itu WAJIB: pada fase mixed-length setiap
    view punya M_{n,v} sendiri, dan indeks kanvas yang sama merujuk konten
    sumber yang berbeda karena kappa = L_src/M berbeda. Tanpa penyelarasan,
    rerata lintas view tidak terdefinisi -- dan pada implementasi sebelumnya
    w_canon menjadi tidak berpengaruh sama sekali begitu kanvas tidak seragam,
    sehingga ablasi w_canon diam-diam mengukur nol.

    Setelah mu terbentuk pada koordinat kanonik, ia diadaptasi kembali ke kanvas
    tiap view (zero-pad bila kanvas lebih panjang, resample bila lebih pendek)
    sebelum galat dihitung.

    w_canon HANYA membentuk target, tidak dipakai lagi sebagai bobot agregasi;
    pembobotan ganda membuat nilai objektif meluruh ~1/w_canon.
    """
    from modules.tlejepa import adapt_to_canvas

    z_c = out["z_v_canon"]                     # [B,V,M,D]
    m_content = out["masks_v_content"]
    m_canvas = out["masks_v_canon"]
    B, V, M, d = z_c.shape

    if "anchor_latent" in out:
        # Ablasi target laten murni: target tetap, tidak dibentuk dari view.
        mu_v = out["anchor_latent"].unsqueeze(1).expand(B, V, M, d)
        content = m_content.to(z_c.dtype).unsqueeze(-1)
    else:
        z_ref = out["z_ref"]                   # [B,V,M,D] pada koordinat kanonik
        ref_mask = out["ref_mask"]
        la = out["l_anchor_long"]              # [B]
        canvas = out["canvas_clamped"]         # [B,V]

        w = torch.ones(V, device=z_c.device, dtype=z_c.dtype)
        w[0] = w_canon
        w_view = w.view(1, V, 1, 1)

        # Hanya posisi yang valid pada SELURUH view yang boleh membentuk mu.
        common = ref_mask.all(dim=1, keepdim=True).to(z_c.dtype).unsqueeze(-1)
        z_ref_m = z_ref * common
        if stopgrad_canon:
            z_ref_m = torch.cat([z_ref_m[:, :1].detach(), z_ref_m[:, 1:]], dim=1)

        mu_ref = (w_view * z_ref_m).sum(dim=1) / w.sum()        # [B,M,D]

        # Adaptasi mu kembali ke kanvas tiap view.
        mu_v, tgt_mask = adapt_to_canvas(
            mu_ref.unsqueeze(1).expand(B, V, M, d).reshape(B * V, M, d),
            la.repeat_interleave(V),
            canvas.reshape(B * V),
            M, tau_r=tau_r,
        )
        mu_v = mu_v.view(B, V, M, d)
        content = (m_content & tgt_mask.view(B, V, M)).to(z_c.dtype).unsqueeze(-1)

    sq = ((z_c - mu_v) ** 2) * content
    l_content = sq.sum() / (content.sum() * d * V).clamp(min=1.0)

    # --- Wilayah kosong ditarik ke nol -------------------------------------
    # Ini yang membuat batas konten dapat dibaca kembali dari profil norma saat
    # inferensi, ketika L^A tidak tersedia. Tanpa suku ini decoder bebas
    # mengisi kanvas kosong dengan fitur arbitrer yang mencemari pooling.
    m_empty = (m_canvas & ~m_content).to(z_c.dtype).unsqueeze(-1)
    n_empty = m_empty.sum()
    if float(n_empty) > 0:
        l_empty = ((z_c ** 2) * m_empty).sum() / (n_empty * d).clamp(min=1.0)
    else:
        l_empty = torch.zeros((), device=z_c.device, dtype=z_c.dtype)

    return l_content + eta_empty * l_empty


# ---------------------------------------------------------------------------
# Teacher semantik (dibekukan)
# ---------------------------------------------------------------------------
class SemanticTeacherSimilarity(nn.Module):
    def __init__(self, teacher_model: str = "tum-nlp/NegMPNet"):
        super().__init__()
        # import lazy: sentence_transformers adalah dependensi berat dan hanya
        # dibutuhkan di jalur ini.
        from sentence_transformers import SentenceTransformer
        self.teacher_semantic_model = SentenceTransformer(teacher_model).eval()

    @torch.no_grad()
    def _get_raw(self, batch: dict) -> torch.Tensor:
        canonical_texts = [t[0] for t in batch["texts"]]
        return self.teacher_semantic_model.encode(canonical_texts, convert_to_tensor=True)

    def forward(self, batch, n_core=64, negation_offset=62, n_negation=2):
        emb = safe_normalize(self._get_raw(batch).float())
        emb_A = emb[:n_core]
        emb_B = torch.cat([emb[:negation_offset], emb[n_core:n_core + n_negation]], dim=0)
        return emb_A @ emb_B.t()

    def get_raw_embeddings(self, batch) -> torch.Tensor:
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
        ids = batch["id"]
        texts = [t[0] for t in batch["texts"]]
        device = next(self.base_teacher.teacher_semantic_model.parameters(),
                      torch.tensor(0.0)).device

        missing_idx = [i for i, _id in enumerate(ids) if _id not in self._cache]
        local_lookup = {}
        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            new_embs = self.base_teacher.teacher_semantic_model.encode(
                missing_texts, convert_to_tensor=True)
            for local_i, global_i in enumerate(missing_idx):
                emb_i = new_embs[local_i].detach().cpu()
                local_lookup[ids[global_i]] = emb_i
                if len(self._cache) < self._cache_size:
                    self._cache[ids[global_i]] = emb_i

        return torch.stack(
            [self._cache.get(_id, local_lookup.get(_id)) for _id in ids]
        ).to(device)

    @torch.no_grad()
    def forward(self, batch, n_core=64, negation_offset=62, n_negation=2):
        emb = safe_normalize(self._get_raw(batch).float())
        emb_A = emb[:n_core]
        emb_B = torch.cat([emb[:negation_offset], emb[n_core:n_core + n_negation]], dim=0)
        return emb_A @ emb_B.t()

    @torch.no_grad()
    def get_raw_embeddings(self, batch) -> torch.Tensor:
        return self._get_raw(batch)

    def cache_stats(self) -> dict:
        return {"cached_sentences": len(self._cache)}


# ---------------------------------------------------------------------------
# 5.8.2  Semantic distillation (distilasi RELASIONAL)
# ---------------------------------------------------------------------------
def encoder_only_syntactical_loss(out, w_canon: float = 8.0,
                                  stopgrad_canon: bool = False) -> torch.Tensor:
    """
    R11 (paper 6.3). Tanpa decoder, kesepakatan antar-view ditegakkan langsung pada
    representasi tingkat kalimat Pool(H). Tidak ada dimensi temporal, jadi tidak ada
    kanonikalisasi maupun wilayah kosong -- ini LeJEPA polos pada teks.
    """
    pooled = masked_mean(out["z_v"], out["masks"])          # [B,V,D]
    B, V, d = pooled.shape
    w = torch.ones(V, device=pooled.device, dtype=pooled.dtype)
    w[0] = w_canon
    p = pooled
    if stopgrad_canon:
        p = torch.cat([p[:, :1].detach(), p[:, 1:]], dim=1)
    mu = (w.view(1, V, 1) * p).sum(dim=1, keepdim=True) / w.sum()
    return ((p - mu) ** 2).sum() / (B * V * d)


def semantic_loss(batch, out, cossim_fn, n_core=64,
                  negation_offset=62, n_negation=2, encoder_only: bool = False):
    """
    Paper 5.8.2. Teacher hanya menyediakan SATU target per pasangan sampel,
    sedangkan student menghasilkan V^2 kemiripan untuk pasangan yang sama.
    Memaksa seluruh kombinasi view menuju satu target yang sama menegakkan
    geometri semantik dan invariansi surface form sekaligus.

    Pooling memakai m_content (bukan m_canvas): wilayah kosong tidak boleh
    ikut membentuk representasi tingkat kalimat.
    """
    if encoder_only:
        pooled = masked_mean(out["z_v"], out["masks"])
    else:
        pooled = masked_mean(out["z_v_canon"], out["masks_v_content"])
    pooled_norm = safe_normalize(pooled)

    V, d = pooled_norm.shape[1], pooled_norm.shape[2]
    pooled_A = pooled_norm[:n_core]
    pooled_B = torch.cat(
        [pooled_norm[:negation_offset], pooled_norm[n_core:n_core + n_negation]], dim=0)

    sim_all = (pooled_A.reshape(n_core * V, d) @ pooled_B.reshape(n_core * V, d).t())
    sim_all = sim_all.view(n_core, V, n_core, V).permute(0, 2, 1, 3)

    teacher_sim = cossim_fn(batch, n_core=n_core, negation_offset=negation_offset,
                            n_negation=n_negation)
    assert teacher_sim.shape == (n_core, n_core), (
        f"teacher_sim harus ({n_core},{n_core}), didapat {tuple(teacher_sim.shape)}")
    teacher_sim = teacher_sim.to(sim_all.device).to(sim_all.dtype)

    # Delta = JUMLAH atas V^2 pasangan view (bukan rerata); normalisasi hanya
    # dilakukan pada tingkat pasangan sampel.
    delta = (sim_all - teacher_sim.view(n_core, n_core, 1, 1)).pow(2).mean(dim=(-2, -1))

    neg_block_mask = torch.zeros(n_core, n_core, dtype=torch.bool, device=delta.device)
    if negation_offset < n_core:
        neg_block_mask[negation_offset:n_core, negation_offset:n_core] = True

    l_sem_general = delta[~neg_block_mask].sum() / (~neg_block_mask).sum().clamp(min=1)
    l_sem_negation = delta[neg_block_mask].sum() / neg_block_mask.sum().clamp(min=1)

    # --- Kontras negasi (view kanonik saja, u = w = 0) ------------------
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
        l_sem_negation_contrast = F.mse_loss(student_diag - student_baseline, teacher_contrast)

    return l_sem_general, l_sem_negation, l_sem_negation_contrast


# ---------------------------------------------------------------------------
# 5.8.3  SIGReg -- ruang ENCODER, level KALIMAT
# ---------------------------------------------------------------------------
class SIGReg(nn.Module):
    """
    Uji Epps-Pulley ter-sketch terhadap N(0, I), lewat proyeksi 1-D acak
    (Cramer-Wold).

    forward(x)
        x : (..., N, d), N = ukuran POPULASI yang diuji distribusinya

    Statistik TIDAK dikalikan N. Penskalaan itu konvensi uji hipotesis supaya
    statistik punya distribusi asimtotik tertentu di bawah H0; sebagai objektif
    optimisasi ia hanya membuat nilai loss bergantung ukuran batch dan
    menyulitkan perbandingan antar-konfigurasi ablasi.
    """

    def __init__(self, knots: int = 17, num_slices: int = 256, t_max: float = 3.0,
                 slice_chunk: int = 256):
        super().__init__()
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.num_slices = num_slices
        self.slice_chunk = slice_chunk
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fp32 wajib: err adalah selisih dua bilangan sama-sama dekat 1, jadi
        # bf16 (8 bit mantissa) menghancurkannya lewat catastrophic cancellation.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            d = x.size(-1)

            A = torch.randn(d, self.num_slices, device=x.device, dtype=torch.float32)
            A = A / A.norm(p=2, dim=0, keepdim=True)

            stats = []
            for s0 in range(0, self.num_slices, self.slice_chunk):
                x_t = (x @ A[:, s0:s0 + self.slice_chunk]).unsqueeze(-1) * self.t
                cos_m, sin_m = x_t.cos().mean(dim=-3), x_t.sin().mean(dim=-3)
                err = (cos_m - self.phi).square() + sin_m.square()
                stats.append(err @ self.weights)

            return torch.cat(stats, dim=-1).mean()


def sigreg_loss(out, sigreg_fn, space: str = "both",
                w_encoder: float = 0.5, w_decoder: float = 0.5):
    """
    space="both"    (baku) -> Pool(H) DAN Pool(Z_c), masing-masing berbobot
    space="encoder"        -> hanya Pool(H) dengan m_src
    space="decoder"        -> hanya Pool(Z_c) dengan m_content

    Ruang decoder disertakan karena eksperimen menunjukkan H6 tidak berlaku:
    ruang encoder tetap sehat (SIGReg 0,039 dan norma pooled 21,6) sementara
    ruang decoder menyusut hingga norma minimum 3,26 dan kosinus antar-sampel
    acak mencapai 0,9999. SIGReg pada Pool(H) tidak merambat ke Z_c karena
    keduanya dipisahkan oleh decoder, dan tidak ada suku lain yang mengatur
    distribusi Z_c: L_sem murni kosinus sehingga buta terhadap skala, dan
    L_syn hanya menuntut kesepakatan antar-view yang dapat dipenuhi pada
    subruang berdimensi rendah.

    Mengembalikan (total, komponen_encoder, komponen_decoder) supaya keduanya
    dapat dipantau terpisah di TensorBoard.
    """
    zero = torch.zeros((), device=out["z_v"].device)
    l_enc, l_dec = zero, zero

    if space in ("encoder", "both"):
        l_enc = sigreg_fn(masked_mean(out["z_v"], out["masks"]).transpose(0, 1))
    if space in ("decoder", "both") and not out.get("encoder_only", False):
        l_dec = sigreg_fn(
            masked_mean(out["z_v_canon"], out["masks_v_content"]).transpose(0, 1)
        )
    if space not in ("encoder", "decoder", "both"):
        raise ValueError(f"space tidak dikenal: {space}")

    if space == "encoder":
        return l_enc, l_enc, zero
    if space == "decoder":
        return l_dec, zero, l_dec
    return w_encoder * l_enc + w_decoder * l_dec, l_enc, l_dec


def sigreg_encoder_loss(out, sigreg_fn) -> torch.Tensor:
    """
    Paper 5.8.3. Populasi = B representasi tingkat kalimat pada ruang ENCODER,
    per view. Pooling memakai m_src (mask padding sumber), BUKAN m_content
    yang berlaku pada kanvas decoder -- keduanya beda panjang dan beda makna.
    """
    h = out["z_v"]           # [B,V,L,D]
    m_src = out["masks"]     # [B,V,L]
    pooled = masked_mean(h, m_src)              # [B,V,D]
    return sigreg_fn(pooled.transpose(0, 1))    # populasi di dim -2 = B


# ---------------------------------------------------------------------------
# 5.8.5  Length prediction loss -- pinball atas log-rasio
# ---------------------------------------------------------------------------
def canon_len_loss(out: dict, tau_l: float = 0.9) -> torch.Tensor:
    """
    Paper 5.8.5. e = rho* - rho_hat.
        e > 0  => underestimate  (dihukum tau_L)
        e < 0  => overestimate   (dihukum 1 - tau_L)

    tau_L = 0.9 berarti underestimation dihukum sembilan kali lebih berat,
    sehingga rho_hat mengestimasi kuantil ke-0.9 dari distribusi bersyarat --
    overestimate yang terkendali, bukan rerata.

    Perbandingan dilakukan pada skala log biasa, bukan log1p: log1p hanya
    relevan untuk panjang absolut yang bisa mendekati nol.
    """
    # View ANCHOR dikecualikan. Pada view anchor berlaku L_src = L^A menurut
    # definisi, sehingga rho* = log(L^A / L^A) = 0 secara identik. Melatih
    # prediktor pada target konstan trivial itu mengencerkan sinyal kuantil,
    # menghabiskan kapasitas untuk memetakan masukan fonem ke nol, dan melatih
    # tugas yang tidak pernah dipakai saat inferensi (masukan inferensi selalu
    # grafem). N_len = B * (V' - 1).
    e = (out["rho_star"].detach() - out["rho_pred"])[:, 1:]
    return (tau_l * e.clamp(min=0) + (1.0 - tau_l) * (-e).clamp(min=0)).mean()


# ---------------------------------------------------------------------------
# Kriteria & agregasi
# ---------------------------------------------------------------------------
@dataclass
class TLeJEPACriterion:
    sigreg_fn: SIGReg
    cossim_fn: nn.Module

    lambda_: float = 0.5          # trade-off SIGReg
    zeta_syn: float = 1.0         # bobot L_syn
    zeta_sem: float = 1.0         # bobot L_sem (umum)
    zeta_sem_neg: float = 1.0     # bobot (L_sem_neg + L_sem_contrast)
    gamma_len: float = 1.0        # bobot L_len
    w_canon: float = 8.0
    eta_empty: float = 1.0
    tau_l: float = 0.9
    stopgrad_canon: bool = False
    # "both" (baku), "encoder", atau "decoder". Bobot masing-masing ruang
    # berjumlah 1,0 supaya lambda_ tetap menyatakan trade-off total terhadap
    # cabang prediktif, tidak berubah artinya ketika ruang ditambah.
    sigreg_space: str = "both"
    sigreg_w_encoder: float = 0.5
    sigreg_w_decoder: float = 0.5

    # Normalisasi bobot cabang prediktif. Tanpa ini, mematikan satu suku saat
    # ablasi juga menurunkan magnitudo cabang prediktif, sehingga bobot relatif
    # SIGReg naik diam-diam dan dua variabel berubah sekaligus.
    normalize_obj_weights: bool = True

    def to(self, device):
        self.sigreg_fn = self.sigreg_fn.to(device)
        self.cossim_fn = self.cossim_fn.to(device)
        return self


def compute_losses(model, batch, criterion, n_core, n_negation, device,
                   use_amp: bool = True, amp_dtype=torch.bfloat16,
                   canon_type: str = "phoneme",
                   compute_rank: bool = False):
    batch = move_batch_to_device(batch, device)
    autocast_enabled = use_amp and device.type == "cuda"

    # Forward model boleh bf16; blok LOSS tidak. Semua suku di bawah berbentuk
    # selisih-lalu-kuadrat, dan bf16 menghancurkan presisinya.
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
        out = model.train_forward(batch, type=canon_type)

    with torch.autocast(device_type=device.type, enabled=False):
        out = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
               for k, v in out.items()}

        enc_only = bool(out.get("encoder_only", False))
        if enc_only:
            # R11: tidak ada decoder. Objektif bekerja pada Pool(H); tidak ada
            # kanonikalisasi sekuensial dan tidak ada prediksi panjang.
            l_syn = encoder_only_syntactical_loss(out, w_canon=criterion.w_canon,
                                                  stopgrad_canon=criterion.stopgrad_canon)
            l_len = torch.zeros((), device=device)
        else:
            l_syn = syntactical_loss(
                out, w_canon=criterion.w_canon,
                eta_empty=criterion.eta_empty,
                stopgrad_canon=criterion.stopgrad_canon,
                tau_r=model.decoder.tau_r,
            )
            l_len = canon_len_loss(out, tau_l=criterion.tau_l)

        l_sem_general, l_sem_negation, l_sem_contrast = semantic_loss(
            batch, out, criterion.cossim_fn,
            n_core=n_core, negation_offset=n_core - n_negation, n_negation=n_negation,
            encoder_only=enc_only)
        l_sig, l_sig_enc, l_sig_dec = sigreg_loss(
            out, criterion.sigreg_fn,
            space="encoder" if enc_only else criterion.sigreg_space,
            w_encoder=criterion.sigreg_w_encoder,
            w_decoder=criterion.sigreg_w_decoder,
        )

        l_sem_neg_combined = l_sem_negation + l_sem_contrast

        obj_terms = [
            (criterion.zeta_syn, l_syn),
            (criterion.zeta_sem, l_sem_general),
            (criterion.zeta_sem_neg, l_sem_neg_combined),
        ]
        obj = sum(w * v for w, v in obj_terms)
        if criterion.normalize_obj_weights:
            w_total = sum(w for w, _ in obj_terms)
            obj = obj / max(w_total, 1e-8)

        total = ((1 - criterion.lambda_) * obj
                 + criterion.lambda_ * l_sig
                 + criterion.gamma_len * l_len)

        # --- Diagnostik (tanpa gradien) ---------------------------------
        with torch.no_grad():
            pooled_h = masked_mean(out["z_v"], out["masks"])
            diag = {"pooled_norm_h_mean": pooled_h.norm(dim=-1).mean()}
            if compute_rank:
                diag["rank_encoder"] = effective_rank(pooled_h[:, 1, :])
            if not enc_only:
                pooled_canon = masked_mean(out["z_v_canon"], out["masks_v_content"])
                pn = pooled_canon.norm(dim=-1)
                # Kecualikan anchor: e[:,0] selalu nol menurut definisi.
                e = (out["rho_star"] - out["rho_pred"])[:, 1:]
                diag.update({
                    "pooled_norm_min": pn.min(),
                    "pooled_norm_mean": pn.mean(),
                    # Proporsi view yang TIDAK ter-underestimate. Untuk tau_L=0.9,
                    # angka ini semestinya merangkak menuju ~0.9.
                    "len_over_frac": (e <= 0).float().mean(),
                    "len_mae_log": e.abs().mean(),
                    "p_tf": out["p_tf"],
                    "empty_frac": (out["masks_v_canon"] & ~out["masks_v_content"]).float().mean(),
                })
                if compute_rank:
                    diag["rank_decoder"] = effective_rank(pooled_canon[:, 1, :])

    losses = {
        "total": total,
        "syntactic": l_syn.detach(),
        "semantic": (l_sem_general + l_sem_neg_combined).detach(),
        "semantic_general": l_sem_general.detach(),
        "semantic_negation": l_sem_negation.detach(),
        "semantic_negation_contrast": l_sem_contrast.detach(),
        "sigreg": l_sig.detach(),
        "sigreg_encoder": l_sig_enc.detach(),
        "sigreg_decoder": l_sig_dec.detach(),
        "canon_len": l_len.detach(),
        **{k: v.detach() for k, v in diag.items()},
    }
    return total, losses, out, batch
