"""
TLeJEPA -- implementasi ketat mengikuti paper (Bagian 5.4 s.d. 5.7).

Perubahan pokok terhadap versi lama:

  * canonical_query_library DIHAPUS (paper 5.5). Kueri awal decoder sekarang
    Q0 = Norm_Q(R_M(H) + PE_tgt) dengan R_M = soft Gaussian temporal
    resampling atas KELUARAN ENCODER H. Versi lama memakai embedding posisi
    yang dipelajari dan identik untuk semua sampel, sehingga decoder bisa
    membentuk representasi yang bergantung posisi saja sebelum informasi
    sumber masuk.

  * LengthPredictor memprediksi LOG-RASIO rho = log(L_anchor / L_src), bukan
    panjang absolut (paper 5.6.3). Arsitekturnya sendiri tidak berubah.

  * train_forward mendukung mixed-length canvas (paper 5.7 Tahap 7): setelah
    warmup, panjang kanvas tiap view diundi Bernoulli antara panjang anchor
    dan panjang prediksi. Ini yang memunculkan wilayah kosong, sehingga
    m_content dan m_canvas benar-benar berbeda.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint
from modules.utils import length_to_mask


# ---------------------------------------------------------------------------
# DeepNorm (DeepNet, arXiv:2203.00555) -- varian encoder-decoder
# ---------------------------------------------------------------------------
def deepnorm_coeffs(n_enc: int, n_dec: int) -> dict:
    """
    N = jumlah layer encoder, M = jumlah layer decoder.
      encoder: alpha = 0.81 (N^4 M)^(1/16)   beta = 0.87 (N^4 M)^(-1/16)
      decoder: alpha = (3M)^(1/4)            beta = (12M)^(-1/4)

    Ini formula DeepNet untuk arsitektur ENCODER-DECODER. Paper versi awal
    mengutip formula encoder-only ((2N)^(1/4), (8N)^(-1/4)); yang benar untuk
    TLeJEPA adalah varian di bawah, dan paper sudah dikoreksi mengikuti ini.
    """
    p = math.pow(math.pow(n_enc, 4) * n_dec, 1.0 / 16.0)
    return {
        "enc_alpha": 0.81 * p,
        "enc_beta": 0.87 / p,
        "dec_alpha": math.pow(3.0 * n_dec, 0.25),
        "dec_beta": math.pow(12.0 * n_dec, -0.25),
    }


def _deepnorm_init_attention(attn: nn.MultiheadAttention, beta: float):
    """
    nn.MultiheadAttention menyatukan q/k/v dalam SATU in_proj_weight, jadi
    penskalaan berbasis nama parameter ala torchscale tidak bisa dipakai --
    harus di-slice. Hanya v dan out_proj yang diskalakan beta; q dan k tidak,
    karena keduanya hanya memengaruhi bobot attention (pasca-softmax), bukan
    magnitudo output.
    """
    d = attn.embed_dim
    w = attn.in_proj_weight
    nn.init.xavier_uniform_(w[:d])                            # q
    nn.init.xavier_uniform_(w[d:2 * d])                       # k
    nn.init.xavier_uniform_(w[2 * d:], gain=beta)             # v   <- diskalakan
    nn.init.xavier_uniform_(attn.out_proj.weight, gain=beta)  # <- diskalakan
    if attn.in_proj_bias is not None:
        nn.init.zeros_(attn.in_proj_bias)
    if attn.out_proj.bias is not None:
        nn.init.zeros_(attn.out_proj.bias)


def _deepnorm_init_ffn(ffn: nn.Sequential, beta: float):
    for m in ffn:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=beta)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def sinusoidal_PE(length: int, d_model: int, device=None,
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(length, d_model, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe.to(dtype=dtype)


# ---------------------------------------------------------------------------
# Operator R_M: soft Gaussian temporal resampling (paper 5.5)
# ---------------------------------------------------------------------------
def gaussian_resample(
    h: torch.Tensor,
    source_mask: torch.Tensor,
    target_lengths: torch.Tensor,
    tau_r: float = 0.3,
    target_max: Optional[int] = None,
):
    """
    C_t = sum_i omega_{t,i} H_i,  omega = softmax_i( -(i - kappa t)^2 / tau_R )
    dengan kappa = L_src / M.

    Berbeda dari uniform-copy Gu et al. yang memakai gather diskrit, operator
    ini kontinu terhadap koordinat sumber sehingga gradien mengalir ke SELURUH
    posisi encoder yang berbobot, bukan hanya ke satu posisi nearest-neighbour.
    Tidak ada detach di sini: paper mensyaratkan jalur H -> R_M -> Q0 tetap
    membawa gradien.

    h             : [B, L_src, D]  -- KELUARAN ENCODER (bukan embedding mentah)
    source_mask   : [B, L_src] bool
    target_lengths: [B] long
    """
    if h.dim() != 3:
        raise ValueError(f"h harus [B, L_src, D], dapat {tuple(h.shape)}")
    B, L_src, D = h.shape
    device = h.device

    src_len = source_mask.sum(dim=1).clamp(min=1).to(torch.float32)          # [B]
    tgt_len = target_lengths.to(device=device, dtype=torch.float32).clamp(min=1.0)
    M = int(target_max if target_max is not None else int(target_lengths.max().item()))
    M = max(M, 1)

    # kappa = L_src / M  ->  koordinat sumber kontinu c_t = kappa * t
    kappa = (src_len / tgt_len).view(B, 1, 1)
    t = torch.arange(M, device=device, dtype=torch.float32).view(1, M, 1)
    i = torch.arange(L_src, device=device, dtype=torch.float32).view(1, 1, L_src)

    c_t = kappa * t
    psi = -((i - c_t) ** 2) / float(tau_r)                                   # [B,M,L_src]

    # Padding sumber tidak ikut normalisasi softmax.
    psi = psi.masked_fill(~source_mask.view(B, 1, L_src), float("-inf"))
    omega = torch.softmax(psi, dim=-1)
    omega = torch.nan_to_num(omega, nan=0.0)

    c = torch.bmm(omega.to(h.dtype), h)                                      # [B,M,D]
    canvas_mask = length_to_mask(target_lengths.to(device), M)
    c = c * canvas_mask.unsqueeze(-1).to(c.dtype)
    return c, canvas_mask


def detect_content_boundary(zc, canvas_mask, ratio: float = 0.35):
    """
    Menentukan di mana konten kanonik berakhir SAAT INFERENSI, ketika L^A tidak
    tersedia sehingga m_content tidak bisa dihitung seperti saat pelatihan.

    Bertumpu pada L_empty: wilayah kosong dilatih menuju norma nol, jadi batas
    dapat dibaca kembali dari profil norma. Ambang bersifat relatif terhadap
    norma median posisi kanvas, bukan absolut, supaya tidak bergantung pada
    skala representasi.

    Tanpa mekanisme ini, pooling inferensi akan mencakup seluruh kanvas termasuk
    wilayah kosong -- persis train/test mismatch pada objek yang dievaluasi.

    zc          : [B, M, D]
    canvas_mask : [B, M]
    keluaran    : [B] panjang konten terdeteksi (minimal 1)
    """
    norms = zc.norm(dim=-1) * canvas_mask.to(zc.dtype)          # [B,M]
    B, M = norms.shape
    ref = torch.where(canvas_mask, norms, torch.zeros_like(norms))
    med = ref.sum(dim=1) / canvas_mask.sum(dim=1).clamp(min=1)  # rerata posisi valid
    thr = (ratio * med).unsqueeze(1)
    is_content = (norms > thr) & canvas_mask
    # Batas = posisi konten TERAKHIR + 1, bukan cacahan posisi di atas ambang:
    # satu posisi lemah di tengah kalimat tidak boleh memotong sufiksnya.
    pos = torch.arange(M, device=zc.device).unsqueeze(0).expand(B, M)
    last = torch.where(is_content, pos, torch.full_like(pos, -1)).max(dim=1).values
    return (last + 1).clamp(min=1)


def adapt_to_canvas(src, src_len, canvas_len, M_max, tau_r: float = 0.3):
    """
    Sesuaikan tensor `src` (panjang valid `src_len`) ke panjang kanvas
    `canvas_len`. Paper 5.7 Tahap 9, arah: kanvas menentukan, target menyesuaikan.

        canvas > src_len  -> zero-pad; selisihnya jadi wilayah kosong
        canvas < src_len  -> RESAMPLE laten BESERTA masknya
        canvas = src_len  -> apa adanya

    src        : [N, M_max, D]
    src_len    : [N] panjang valid isi src
    canvas_len : [N] panjang kanvas yang dituju
    keluaran   : [N, M_max, D], [N, M_max] (mask konten hasil adaptasi)
    """
    N, _, D = src.shape
    device = src.device
    src_mask = length_to_mask(src_len, M_max)

    out = src.clone()
    out_mask = length_to_mask(torch.minimum(canvas_len, src_len), M_max)

    shrink = canvas_len < src_len
    if bool(shrink.any()):
        idx = shrink.nonzero(as_tuple=False).squeeze(-1)
        res, res_mask = gaussian_resample(
            src[idx], src_mask[idx], canvas_len[idx], tau_r=tau_r, target_max=M_max
        )
        out[idx] = res
        out_mask[idx] = res_mask

    canvas_mask = length_to_mask(canvas_len, M_max)
    out = out * canvas_mask.unsqueeze(-1).to(out.dtype)
    return out, (out_mask & canvas_mask)


# ---------------------------------------------------------------------------
# Blok Transformer
# ---------------------------------------------------------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float,
                 alpha: float = 1.0, beta: float = None):
        super().__init__()
        self.alpha = alpha
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        if beta is not None:
            _deepnorm_init_attention(self.self_attn, beta)
            _deepnorm_init_ffn(self.ffn, beta)

    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    def forward(self, h: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        key_padding_mask = ~mask if mask is not None else None
        attn_out, _ = self.self_attn(
            query=h, key=h, value=h,
            key_padding_mask=key_padding_mask, need_weights=False,
        )
        h = self.norm1(self.residual_connection(self.dropout1(attn_out), h))
        ffn_out = self.ffn(h)
        h = self.norm2(self.residual_connection(self.dropout2(ffn_out), h))
        return h


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout,
                 alpha: float = 1.0, beta: float = None):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, alpha=alpha, beta=beta)
            for _ in range(n_layers)
        ])
        self.gradient_checkpointing = False

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and h.requires_grad:
                h = checkpoint(layer, h, mask, use_reentrant=False)
            else:
                h = layer(h, mask=mask)
        return h


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float,
                 alpha: float = 1.0, beta: float = None):
        super().__init__()
        self.alpha = alpha
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

        if beta is not None:
            _deepnorm_init_attention(self.self_attn, beta)
            _deepnorm_init_attention(self.cross_attn, beta)
            _deepnorm_init_ffn(self.ffn, beta)

    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    def forward(self, z, q_canon, query_mask, source_mask):
        self_kpm = ~query_mask if query_mask is not None else None
        attn_out, _ = self.self_attn(q_canon, q_canon, q_canon,
                                     key_padding_mask=self_kpm, need_weights=False)
        q_canon = self.norm1(self.residual_connection(self.dropout1(attn_out), q_canon))

        cross_kpm = ~source_mask if source_mask is not None else None
        cross_out, _ = self.cross_attn(q_canon, z, z,
                                       key_padding_mask=cross_kpm, need_weights=False)
        q_canon = self.norm2(self.residual_connection(self.dropout2(cross_out), q_canon))

        ffn_out = self.ffn(q_canon)
        q_canon = self.norm3(self.residual_connection(self.dropout3(ffn_out), q_canon))
        return q_canon


class TransformerDecoder(nn.Module):
    """
    Paper 5.5. Tidak ada learned query library. Kueri awal dibentuk dari
    keluaran encoder yang di-resample ke panjang kanvas, ditambah PE target.
    """

    def __init__(self, d_model, n_heads, d_ff, n_layers, max_length=2048, dropout=0.1,
                 alpha: float = 1.0, beta: float = None, tau_r: float = 0.3):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.tau_r = tau_r
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout, alpha=alpha, beta=beta)
            for _ in range(n_layers)
        ])
        # Norm_Q pada jalur kueri -- di luar cabang residual, tidak diskalakan DeepNorm.
        self.input_norm = nn.LayerNorm(d_model)
        self.gradient_checkpointing = False

    def forward(self, z, l_star, source_mask):
        """
        z           : [B, L_src, D] keluaran encoder (jadi K, V sekaligus sumber kueri)
        l_star      : [B] panjang kanvas yang diminta
        source_mask : [B, L_src]
        """
        device = z.device
        l_star = l_star.to(device=device, dtype=torch.long).clamp(min=1, max=self.max_length)
        l_star_max = int(l_star.max().item())

        # --- Q0 = Norm_Q(R_M(H) + PE_tgt) ---------------------------------
        c, query_mask = gaussian_resample(
            z, source_mask, l_star, tau_r=self.tau_r, target_max=l_star_max
        )
        pe = sinusoidal_PE(l_star_max, self.d_model, device=device, dtype=c.dtype)
        query_canonical = self.input_norm(c + pe.unsqueeze(0))
        query_canonical = query_canonical * query_mask.unsqueeze(-1).to(query_canonical.dtype)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training and query_canonical.requires_grad:
                query_canonical = checkpoint(
                    layer, z, query_canonical, query_mask, source_mask, use_reentrant=False
                )
            else:
                query_canonical = layer(z, query_canonical,
                                        query_mask=query_mask, source_mask=source_mask)

        return query_canonical, query_mask


# ---------------------------------------------------------------------------
# Prediktor panjang (paper 5.6)
# ---------------------------------------------------------------------------
class MaskedAttentionPooling(nn.Module):
    def __init__(self, d_model, hidden: int = 128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(z).squeeze(-1)
        # -1e4 (bukan -inf): pelatihan berjalan pada presisi campuran, dan -inf
        # pada baris yang seluruhnya termask menghasilkan NaN di softmax.
        logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (z * weights).sum(dim=1)


class LengthPredictor(nn.Module):
    """
    Paper 5.6. Arsitektur SAMA dengan versi lama (pooling attention + fitur
    log L_src + MLP tiga lapis); yang berubah hanya parametrisasi keluaran:
    head mengeluarkan rho_hat = log(L_anchor / L_src) secara identitas, tanpa
    softplus.

    Alasan (paper 5.6.3): pada parametrisasi panjang absolut, gradien terhadap
    keluaran mentah head mengandung faktor 1/(1+L) sehingga sekuens panjang
    menerima gradien jauh lebih kecil untuk galat log yang sama besar.
    """

    def __init__(self, d_model: int, hidden: int = 256, dropout: float = 0.1,
                 rho_min: float = -2.5, rho_max: float = 2.5,
                 rho_init: float = 0.0):
        super().__init__()
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.pool = MaskedAttentionPooling(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model + 1, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        # Paper 5.6.4: W_last = 0, b_last = rerata log-rasio dataset. Model
        # bermula sebagai prediktor rasio rerata global, bukan keluaran acak.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, float(rho_init))

    def set_rho_init(self, rho_init: float):
        with torch.no_grad():
            self.head[-1].bias.fill_(float(rho_init))

    def forward(self, z, mask, detach_input: bool = True):
        """
        Mengembalikan (rho_hat, l_hat_float) dengan
        l_hat_float = L_src * exp(rho_use). Pembulatan ke atas dilakukan
        pemanggil supaya operasi diskrit tidak masuk graph.
        """
        if detach_input:
            z = z.detach()
        pooled = self.pool(z, mask)
        lengths = mask.sum(dim=1).float().clamp(min=1.0)
        log_len = torch.log(lengths).unsqueeze(-1)
        feat = torch.cat([pooled, log_len], dim=-1)
        rho = self.head(feat).squeeze(-1)

        # Paper 5.6.5: clamp HANYA saat inferensi. Saat pelatihan, clamping
        # meniadakan gradien pada sampel yang justru paling perlu dikoreksi.
        # Diikat ke torch.is_grad_enabled(), BUKAN ke self.training: kalau pemanggil
        # lupa model.eval(), clamp tetap berlaku di jalur inferensi. Saat pelatihan
        # rho sengaja tidak dibatasi supaya gradien tetap hidup pada sampel yang
        # justru paling perlu dikoreksi.
        training_mode = self.training and torch.is_grad_enabled()
        rho_use = rho if training_mode else rho.clamp(self.rho_min, self.rho_max)
        l_hat = lengths * torch.exp(rho_use)
        return rho, l_hat


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TLeJEPA(nn.Module):
    def __init__(self, n_vocab_text, n_vocab_phoneme, d_model=256, n_attn_heads=8,
                 enc_layers=6, dec_layers=4, max_length=4096, dropout=0.1,
                 use_deepnorm: bool = True, tau_r: float = 0.3,
                 rho_init: float = 0.0, pure_latent_anchor: bool = False,
                 encoder_only: bool = False, c_max: float = 2.0,
                 empty_norm_ratio: float = 0.35):
        super().__init__()
        self.d_model = d_model
        self.use_deepnorm = use_deepnorm
        self.max_length = max_length
        # Ablasi M7 (paper 6.3): target L_syn diganti keluaran tabel embedding
        # anchor secara langsung. Ini BUKAN JEPA lagi melainkan objektif
        # rekonstruksi pada ruang embedding, karena targetnya tetap dan tidak
        # dipelajari bersama. Justru itu gunanya: menguji klaim bahwa prediksi
        # laten lebih menguntungkan daripada rekonstruksi.
        self.pure_latent_anchor = pure_latent_anchor

        # R11 (paper 6.3): buang decoder, proyektor kanvas, dan prediktor panjang
        # sepenuhnya. Setara LeJEPA polos pada teks -- objektif bekerja pada
        # Pool(H) dan tidak ada kanonikalisasi apa pun.
        self.encoder_only = encoder_only

        # Pagar ekspansi kanvas relatif terhadap L^A tiap sampel (paper 5.6, Pers. pagar).
        # Relatif, bukan absolut, supaya sekuens pendek dan panjang memperoleh ruang
        # overestimate yang sebanding.
        self.c_max = float(c_max)

        # Ambang deteksi batas konten saat inferensi, relatif terhadap norma
        # median posisi kanvas. L_empty melatih wilayah kosong menuju norma nol,
        # sehingga batas dapat dibaca kembali dari norma -- inilah yang
        # menggantikan m_content yang tidak tersedia saat inferensi.
        self.empty_norm_ratio = float(empty_norm_ratio)

        c = deepnorm_coeffs(enc_layers, dec_layers)
        if use_deepnorm:
            enc_alpha, enc_beta = c["enc_alpha"], c["enc_beta"]
            dec_alpha, dec_beta = c["dec_alpha"], c["dec_beta"]
        else:
            enc_alpha = dec_alpha = 1.0
            enc_beta = dec_beta = None
        self.deepnorm_coeffs = c

        # padding_idx SENGAJA TIDAK dipakai: collate_fn membungkus tiap sekuens
        # dengan pad_value=0 sebagai penanda BOS/EOS dan memasukkannya ke dalam
        # mask, jadi baris 0 adalah token KONTEN yang harus tetap dilatih.
        self.text_embedding = nn.Embedding(n_vocab_text, d_model)
        self.phoneme_embedding = nn.Embedding(n_vocab_phoneme, d_model)
        nn.init.normal_(self.text_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.phoneme_embedding.weight, mean=0.0, std=0.02)

        self.embed_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(
            d_model=d_model, n_heads=n_attn_heads, d_ff=4 * d_model,
            n_layers=enc_layers, dropout=dropout, alpha=enc_alpha, beta=enc_beta)
        if encoder_only:
            self.decoder = None
            self.length_predictor = None
        else:
            self.decoder = TransformerDecoder(
                d_model=d_model, n_heads=n_attn_heads, d_ff=4 * d_model,
                n_layers=dec_layers, max_length=max_length, dropout=dropout,
                alpha=dec_alpha, beta=dec_beta, tau_r=tau_r)
            self.length_predictor = LengthPredictor(d_model=d_model, rho_init=rho_init)

        # --- Jadwal mixed-length canvas (paper 5.7 Tahap 7) ---------------
        # Diatur dari luar oleh Trainer lewat set_length_schedule().
        self.tf_warmup_steps = 0     # s_warm : sebelum ini, selalu teacher-forced
        self.tf_end_steps = 1        # s_end  : setelah ini, p_TF = p_min
        self.tf_p_min = 1.0          # p_min  : 1.0 => mixed-length nonaktif
        self._global_step = 0

    def set_length_schedule(self, warmup_steps: int, end_steps: int, p_min: float):
        self.tf_warmup_steps = int(warmup_steps)
        self.tf_end_steps = int(max(end_steps, warmup_steps + 1))
        self.tf_p_min = float(p_min)

    def set_global_step(self, step: int):
        self._global_step = int(step)

    def teacher_forcing_prob(self) -> float:
        s = self._global_step
        if s <= self.tf_warmup_steps or self.tf_p_min >= 1.0:
            return 1.0
        frac = (s - self.tf_warmup_steps) / max(1, self.tf_end_steps - self.tf_warmup_steps)
        return max(self.tf_p_min, 1.0 - frac)

    def set_gradient_checkpointing(self, enabled: bool = True):
        self.encoder.gradient_checkpointing = enabled
        if self.decoder is not None:
            self.decoder.gradient_checkpointing = enabled
        return self

    def _embed(self, ids, use_phoneme: bool):
        emb = self.phoneme_embedding(ids) if use_phoneme else self.text_embedding(ids)
        pe = sinusoidal_PE(ids.shape[1], self.d_model, device=ids.device, dtype=emb.dtype)
        return self.embed_norm(emb + pe.unsqueeze(0))

    # ------------------------------------------------------------------ train
    def train_forward(self, x: dict, type: str = "phoneme") -> dict:
        """
        Susunan view (paper 5.2.1), V' = V + 1:
            v = 0        anchor  -- fonem, atau SALINAN grafem bila type="text"
            v = 1        grafem kanonik, tidak pernah jadi anchor
            v = 2..V     augmentasi

        Anchor selalu didekode pada L_T natural: ia alat ukur, bukan peserta.
        """
        assert type in ("phoneme", "text")
        use_phoneme = (type == "phoneme")

        B, V, L = x["x"].shape          # V di sini sudah V' = V+1
        masks = x["mask"]

        # --- Tahap 3: pengodean seluruh view lewat encoder bersama ----------
        h0_anchor = self._embed(x["x"][:, 0, :], use_phoneme=use_phoneme)
        if V > 1:
            rest = x["x"][:, 1:, :].reshape(B * (V - 1), L)
            h0_rest = self._embed(rest, use_phoneme=False).view(B, V - 1, L, self.d_model)
            h0 = torch.cat([h0_anchor.unsqueeze(1), h0_rest], dim=1)
        else:
            h0 = h0_anchor.unsqueeze(1)

        h0_flat = h0.reshape(B * V, L, self.d_model)
        mask_flat = masks.reshape(B * V, L)
        z_flat = self.encoder(h=h0_flat, mask=mask_flat)          # H
        z_v = z_flat.view(B, V, L, self.d_model)

        src_len = mask_flat.sum(dim=1).float().clamp(min=1.0).view(B, V)
        l_anchor = masks[:, 0, :].sum(dim=1).float().clamp(min=1.0)          # L^A
        rho_star = torch.log(l_anchor).unsqueeze(1) - torch.log(src_len)     # [B,V]

        out = {
            "z_v": z_v,
            "masks": masks,
            "rho_star": rho_star.detach(),
            "l_anchor": l_anchor,
            "src_len": src_len,
        }

        # --- R11: berhenti di sini, tidak ada decoder maupun prediktor panjang ---
        if self.encoder_only:
            out["encoder_only"] = True
            return out

        # --- Tahap 5: prediksi panjang dari sg(H) ----------------------------
        rho_flat, l_hat_flat = self.length_predictor(z_flat, mask_flat, detach_input=True)
        out["rho_pred"] = rho_flat.view(B, V)

        # --- Tahap 7: panjang kanvas per view --------------------------------
        l_anchor_long = l_anchor.round().long().clamp(min=1, max=self.max_length)

        # Pagar RELATIF terhadap L^A sampel itu (paper 5.6). Tidak menyentuh
        # gradien: jalur rho -> L_hat sudah detach + ceil, jadi diskrit.
        cap = torch.ceil(self.c_max * l_anchor).long().clamp(min=1, max=self.max_length)
        l_hat_long = torch.ceil(l_hat_flat.detach()).long().view(B, V)
        l_hat_long = torch.minimum(l_hat_long, cap.unsqueeze(1)).clamp(min=1, max=self.max_length)

        p_tf = self.teacher_forcing_prob() if self.training else 1.0
        if p_tf >= 1.0:
            canvas_len = l_anchor_long.unsqueeze(1).expand(B, V).contiguous()
        else:
            b_tf = (torch.rand(B, V, device=z_flat.device) < p_tf)
            b_tf[:, 0] = True                    # anchor SELALU panjang natural
            canvas_len = torch.where(b_tf, l_anchor_long.unsqueeze(1), l_hat_long)
        canvas_len[:, 0] = l_anchor_long          # ditegaskan ulang

        # --- Tahap 6 & 8: dekode seluruh view --------------------------------
        canvas_flat = canvas_len.reshape(B * V)
        z_canon_flat, mask_canvas_flat = self.decoder(
            z=z_flat, l_star=canvas_flat, source_mask=mask_flat
        )
        M_max = z_canon_flat.shape[1]

        # --- Tahap 9: adaptasi TARGET mengikuti kanvas ------------------------
        # Arahnya: prediksi menentukan kanvas, target menyesuaikan diri.
        z_canon = z_canon_flat.view(B, V, M_max, self.d_model)
        mask_canvas = mask_canvas_flat.view(B, V, M_max)

        canvas_clamped = canvas_len.clamp(max=M_max)
        la = l_anchor_long.clamp(max=M_max)                            # [B]

        # Seluruh view lebih dulu diselaraskan ke KOORDINAT KANONIK berpanjang
        # L^A. Tanpa langkah ini, rerata berbobot mu tidak terdefinisi ketika
        # kanvas berbeda antar-view: indeks yang sama merujuk konten yang
        # berbeda karena kappa = L_src/M berbeda. Menyelaraskan dulu membuat
        # w_canon tetap bermakna pada fase mixed-length.
        z_ref, ref_mask = adapt_to_canvas(
            z_canon.reshape(B * V, M_max, self.d_model),
            canvas_clamped.reshape(B * V),
            la.repeat_interleave(V),
            M_max, tau_r=self.decoder.tau_r,
        )
        out["z_ref"] = z_ref.view(B, V, M_max, self.d_model)
        out["ref_mask"] = ref_mask.view(B, V, M_max)
        out["l_anchor_long"] = la
        out["canvas_clamped"] = canvas_clamped

        # m_content: posisi berisi konten kanonik, yaitu min(M, L^A)
        content_len = torch.minimum(canvas_clamped, l_anchor_long.unsqueeze(1).clamp(max=M_max))
        mask_content = length_to_mask(content_len.reshape(B * V), M_max).view(B, V, M_max)

        out.update({
            "z_v_canon": z_canon,
            "masks_v_canon": mask_canvas,
            "masks_v_content": mask_content,
            "canvas_len": canvas_len,
            "p_tf": torch.tensor(float(p_tf), device=z_flat.device),
        })

        if self.pure_latent_anchor:
            emb_fn = self.phoneme_embedding if use_phoneme else self.text_embedding
            raw = emb_fn(x["x"][:, 0, :])
            if raw.shape[1] < M_max:
                raw = torch.nn.functional.pad(raw, (0, 0, 0, M_max - raw.shape[1]))
            raw = raw[:, :M_max, :]
            # Standardisasi wajib: Z_c melewati LayerNorm sehingga bernorma ~sqrt(d),
            # sedangkan embedding mentah ~0.02*sqrt(d). Tanpa ini target berada di luar
            # himpunan nilai yang bisa dihasilkan decoder dan perbandingan jadi timpang.
            raw = torch.nn.functional.layer_norm(raw, (self.d_model,))
            out["anchor_latent"] = raw.detach()

        return out

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def forward(self, text, use_phoneme: bool = False):
        if self.encoder_only:
            raise RuntimeError(
                "Model encoder_only tidak punya decoder; pakai .encode() untuk readout H."
            )

        def _one(t):
            t = t.unsqueeze(0) if t.dim() == 1 else t
            mask = torch.ones(t.shape, dtype=torch.bool, device=t.device)
            h = self._embed(t, use_phoneme=use_phoneme)
            z = self.encoder(h, mask=mask)
            _, l_hat = self.length_predictor(z, mask, detach_input=True)
            # Pembulatan KE ATAS, konsisten dengan arah bias upper-quantile.
            l_star = torch.ceil(l_hat).long().clamp(min=1, max=self.max_length)
            zc, canvas_mask = self.decoder(z, l_star, source_mask=mask)
            l_content = detect_content_boundary(zc, canvas_mask, self.empty_norm_ratio)
            return zc, l_star, l_content

        if isinstance(text, list):
            outs = [_one(t) for t in text]
            return [o[0] for o in outs], [o[1] for o in outs], [o[2] for o in outs]
        return _one(text)

    @torch.no_grad()
    def encode(self, ids, mask=None, use_phoneme: bool = False):
        """Readout ENCODER (ruang yang diregularisasi SIGReg) untuk evaluasi."""
        ids = ids.unsqueeze(0) if ids.dim() == 1 else ids
        if mask is None:
            mask = torch.ones(ids.shape, dtype=torch.bool, device=ids.device)
        h = self._embed(ids, use_phoneme=use_phoneme)
        return self.encoder(h, mask=mask), mask
