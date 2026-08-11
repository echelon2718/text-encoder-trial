"""
Metrik riset yang dijalankan berkala saat pelatihan, lalu dikirim ke TensorBoard.

Tujuannya bukan menggantikan evaluasi akhir, melainkan memberi bukti SEMENTARA
untuk tiap hipotesis sementara run berjalan -- supaya konfigurasi yang jelas
gagal bisa dihentikan lebih awal alih-alih menunggu tiga hari.

Semua metrik dihitung pada model yang dibekukan, memakai batch validasi yang
sama sepanjang run supaya kurvanya dapat diperbandingkan antar-step. Tiap metrik
ditandai hipotesis yang diuji supaya panel TensorBoard langsung terbaca.

    tag TensorBoard                     hipotesis
    -----------------------------------------------------------------
    H1_invariance/*                     H1  modalitas anchor
    H2_dual_use/*                       H2  kegunaan ganda
    H3_word_order/*                     H3  dimensi sekuensial
    H5_H6_H8_rank/*                     H5, H6, H8  peringkat efektif
    H7_negation/*                       H7  supervisi semantik
    H9_length/*                         H9  prediktor panjang
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from modules.losses import masked_mean, safe_normalize, effective_rank


# ---------------------------------------------------------------------------
# Utilitas
# ---------------------------------------------------------------------------
def _encode_text_batch(model, texts, device, max_len: int = 512):
    """Byte UTF-8 -> Pool(H) dan Pool(Z_c). Dua readout dilaporkan terpisah."""
    ids, masks = [], []
    L = min(max(len(t.encode("utf-8")) for t in texts) + 2, max_len)
    for t in texts:
        b = list(t.encode("utf-8"))[: L - 2]
        seq = [0] + b + [0]
        m = [True] * len(seq)
        seq = seq + [0] * (L - len(seq))
        m = m + [False] * (L - len(m))
        ids.append(seq)
        masks.append(m)
    ids = torch.tensor(ids, dtype=torch.long, device=device)
    masks = torch.tensor(masks, dtype=torch.bool, device=device)

    h, _ = model.encode(ids, mask=masks, use_phoneme=False)
    r_enc = masked_mean(h.unsqueeze(1), masks.unsqueeze(1)).squeeze(1)

    r_dec = None
    if not model.encoder_only:
        _, l_hat = model.length_predictor(h, masks, detach_input=True)
        l_star = torch.ceil(l_hat).long().clamp(min=1, max=model.max_length)
        zc, canvas_mask = model.decoder(h, l_star, source_mask=masks)
        r_dec = masked_mean(zc.unsqueeze(1), canvas_mask.unsqueeze(1)).squeeze(1)
    return r_enc, r_dec


def _cos(a, b):
    return (safe_normalize(a) * safe_normalize(b)).sum(-1)


# ---------------------------------------------------------------------------
# H1 -- invariansi terhadap korupsi permukaan
# ---------------------------------------------------------------------------
@torch.no_grad()
def progressive_corruption_recall(model, augmenter, texts, device, levels=(1, 2, 3)):
    """
    Versi ringan dari Progressive Corruption Recall untuk pemantauan.

    Korpus = `texts` bersih. Kueri = versi terkorupsi pada beberapa tingkat
    (jumlah penerapan augmenter). Diukur Recall@1 dan MRR terhadap kalimat
    asalnya. Bentuk KURVA yang dicari, bukan angka tunggal: representasi kanonik
    yang baik menurun landai, kecocokan permukaan menurun tajam.
    """
    out = {}
    r_enc_clean, r_dec_clean = _encode_text_batch(model, texts, device)
    for lv in levels:
        q = []
        for t in texts:
            c = t
            for _ in range(lv):
                c = augmenter.augment_easy(c) if lv <= 2 else augmenter.augment_hard_surface(c)
            q.append(c)
        r_enc_q, r_dec_q = _encode_text_batch(model, q, device)
        for tag, base, qry in (("enc", r_enc_clean, r_enc_q), ("dec", r_dec_clean, r_dec_q)):
            if base is None or qry is None:
                continue
            sim = safe_normalize(qry) @ safe_normalize(base).t()
            gold = torch.arange(len(texts), device=device)
            rank = (sim > sim.gather(1, gold[:, None])).sum(1) + 1
            out[f"H1_invariance/recall@1_{tag}_lv{lv}"] = float((rank == 1).float().mean())
            out[f"H1_invariance/mrr_{tag}_lv{lv}"] = float((1.0 / rank.float()).mean())
    return out


# ---------------------------------------------------------------------------
# H3 -- sensitivitas urutan kata
# ---------------------------------------------------------------------------
@torch.no_grad()
def word_order_discrimination(model, texts, device):
    """
    Menguji contoh pembuka paper: "man eats dog" versus "dog eats man".

    Untuk tiap kalimat dibentuk varian PERMUTASI (dua kata ditukar, komposisi
    token nyaris identik tetapi relasi berubah) dan varian PARAFRASE ringan
    (kata diganti sinonim kasar, komposisi berbeda tetapi makna dipertahankan).
    Model dinilai apakah memberi kemiripan lebih tinggi pada parafrase.

    Model yang tidak peka urutan akan memberi skor ~0,5 karena permutasi terlihat
    sama dengan aslinya.
    """
    import random
    rng = random.Random(0)
    base, perm, para = [], [], []
    syn = {"big": "large", "small": "tiny", "quick": "fast", "happy": "glad",
           "man": "guy", "dog": "hound", "house": "home", "good": "fine"}
    for t in texts:
        w = t.split()
        if len(w) < 4:
            continue
        i, j = rng.sample(range(len(w)), 2)
        pw = w[:]
        pw[i], pw[j] = pw[j], pw[i]
        aw = [syn.get(x, x) for x in w]
        if aw == w:
            aw = w[:] + ["indeed"]
        base.append(t); perm.append(" ".join(pw)); para.append(" ".join(aw))
    if not base:
        return {}

    out = {}
    rb = _encode_text_batch(model, base, device)
    rp = _encode_text_batch(model, perm, device)
    ra = _encode_text_batch(model, para, device)
    for k, tag in ((0, "enc"), (1, "dec")):
        if rb[k] is None:
            continue
        s_perm = _cos(rb[k], rp[k])
        s_para = _cos(rb[k], ra[k])
        out[f"H3_word_order/acc_{tag}"] = float((s_para > s_perm).float().mean())
        out[f"H3_word_order/margin_{tag}"] = float((s_para - s_perm).mean())
        # Kemiripan terhadap permutasi: makin RENDAH makin peka urutan.
        out[f"H3_word_order/sim_permuted_{tag}"] = float(s_perm.mean())
    return out


# ---------------------------------------------------------------------------
# H7 -- pemisahan pasangan negasi
# ---------------------------------------------------------------------------
@torch.no_grad()
def negation_separation(model, premises, negations, paraphrases, device):
    """
    delta_cos = cos(premis, parafrase) - cos(premis, negasi).
    Nilai > 0 berarti model membedakan negasi dari parafrase.
    """
    if not premises:
        return {}
    rp = _encode_text_batch(model, premises, device)
    rn = _encode_text_batch(model, negations, device)
    ra = _encode_text_batch(model, paraphrases, device)
    out = {}
    for k, tag in ((0, "enc"), (1, "dec")):
        if rp[k] is None:
            continue
        d = _cos(rp[k], ra[k]) - _cos(rp[k], rn[k])
        out[f"H7_negation/delta_cos_{tag}"] = float(d.mean())
        out[f"H7_negation/acc_{tag}"] = float((d > 0).float().mean())
    return out


# ---------------------------------------------------------------------------
# H5, H6, H8 -- peringkat efektif
# ---------------------------------------------------------------------------
@torch.no_grad()
def rank_diagnostics(out: dict):
    """
    Peringkat efektif dilaporkan TERPISAH untuk H dan Z_c.

    H berada di dalam ruang yang diregularisasi SIGReg, jadi peringkatnya adalah
    pemeriksaan bahwa regularisasi bekerja. Z_c berada di luarnya, jadi
    peringkatnya adalah pengujian empiris apakah regularisasi ruang encoder sudah
    memadai -- inti H6, dan instrumen yang menangkap collapse pada H8.
    """
    res = {}
    pooled_h = masked_mean(out["z_v"], out["masks"])
    res["H5_H6_H8_rank/encoder"] = float(effective_rank(pooled_h[:, 1, :]))
    res["H5_H6_H8_rank/encoder_ratio"] = res["H5_H6_H8_rank/encoder"] / pooled_h.shape[-1]
    if not out.get("encoder_only", False):
        pooled_z = masked_mean(out["z_v_canon"], out["masks_v_content"])
        res["H5_H6_H8_rank/decoder"] = float(effective_rank(pooled_z[:, 1, :]))
        res["H5_H6_H8_rank/decoder_ratio"] = res["H5_H6_H8_rank/decoder"] / pooled_z.shape[-1]
    return res


# ---------------------------------------------------------------------------
# H9 -- kualitas prediktor panjang
# ---------------------------------------------------------------------------
@torch.no_grad()
def length_diagnostics(out: dict, tau_l: float = 0.9):
    """
    Anchor dikecualikan karena rho*_{n,0} = 0 menurut definisi.
    `over_frac` semestinya merangkak menuju tau_L: itulah tafsiran operasional
    kuantil pinball, yaitu proporsi sampel yang tidak ter-underestimate.
    """
    if out.get("encoder_only", False):
        return {}
    e = (out["rho_star"] - out["rho_pred"])[:, 1:]
    ratio_pred = torch.exp(out["rho_pred"][:, 1:])
    return {
        "H9_length/mae_log_ratio": float(e.abs().mean()),
        "H9_length/over_frac": float((e <= 0).float().mean()),
        "H9_length/over_frac_target": tau_l,
        "H9_length/mean_expansion": float(ratio_pred.mean()),
        "H9_length/canvas_waste": float(
            (out["masks_v_canon"] & ~out["masks_v_content"]).float().mean()
        ),
    }


# ---------------------------------------------------------------------------
# H2 -- probing rekonstruksi ringan
# ---------------------------------------------------------------------------
@torch.no_grad()
def reconstruction_probe_readout(out: dict, max_pos: int = 4096):
    """
    Proksi murah untuk PER/CER tanpa melatih probe: seberapa mudah posisi kanonik
    dibedakan satu sama lain pada Z_c. Kalau seluruh posisi runtuh ke vektor yang
    sama, rekonstruksi per posisi mustahil berapa pun kapasitas probe-nya.

    Tidak berlaku untuk R11: keluarannya tidak punya dimensi temporal yang
    selaras panjang. Ketidakberlakuan itu HASIL, bukan keterbatasan pengukuran.
    """
    if out.get("encoder_only", False):
        return {}
    z = out["z_v_canon"][:, 1, :, :]
    m = out["masks_v_content"][:, 1, :]
    v = z[m]
    if v.shape[0] < 4:
        return {}
    v = v[:max_pos]
    return {
        "H2_dual_use/token_rank": float(effective_rank(v)),
        "H2_dual_use/token_rank_ratio": float(effective_rank(v)) / v.shape[-1],
    }


# ---------------------------------------------------------------------------
# Q2 -- batas konten saat inferensi, dan sensitivitas terhadap galat panjang
# ---------------------------------------------------------------------------
@torch.no_grad()
def boundary_detection_accuracy(out: dict, ratio: float = 0.35):
    """
    Saat pelatihan m_content diketahui dari L^A. Saat inferensi L^A tidak ada,
    sehingga batas harus dibaca kembali dari profil norma -- itulah yang dilatih
    L_empty. Metrik ini mengukur seberapa akurat pembacaan itu, memakai L^A
    sejati sebagai kebenaran dasar.

    Kalau MAE-nya besar, pooling inferensi akan mencakup wilayah kosong dan
    seluruh evaluasi hilir tercemar.
    """
    if out.get("encoder_only", False):
        return {}
    from modules.tlejepa import detect_content_boundary
    z = out["z_v_canon"]
    B, V, M, _ = z.shape
    det = detect_content_boundary(
        z.reshape(B * V, M, z.shape[-1]),
        out["masks_v_canon"].reshape(B * V, M),
        ratio,
    ).view(B, V).float()
    true_len = out["masks_v_content"].sum(dim=-1).float()
    err = det - true_len
    return {
        "Q2_boundary/mae": float(err.abs().mean()),
        "Q2_boundary/rel_mae": float((err.abs() / true_len.clamp(min=1)).mean()),
        "Q2_boundary/over_frac": float((err > 0).float().mean()),
        "Q2_boundary/exact_frac": float((err.abs() < 0.5).float().mean()),
    }


@torch.no_grad()
def length_error_sensitivity(model, texts, device, deltas=(-0.2, -0.1, 0.1, 0.2)):
    """
    Seberapa besar representasi bergeser ketika panjang kanvas meleset?

    Diukur dengan memaksa kanvas ke (1+delta) kali panjang prediksi, lalu
    membandingkan representasi teragregasi terhadap kanvas prediksi asli. Kalau
    kemiripannya runtuh pada delta kecil, seluruh sistem rapuh terhadap galat
    prediktor panjang -- dan itu tidak pernah terukur di matriks eksperimen.
    """
    if model.encoder_only or not texts:
        return {}
    from modules.tlejepa import detect_content_boundary
    ids, masks = [], []
    L = min(max(len(t.encode("utf-8")) for t in texts) + 2, 512)
    for t in texts:
        b = list(t.encode("utf-8"))[: L - 2]
        seq = [0] + b + [0]
        m = [True] * len(seq) + [False] * (L - len(seq))
        ids.append(seq + [0] * (L - len(seq)))
        masks.append(m)
    ids = torch.tensor(ids, dtype=torch.long, device=device)
    masks = torch.tensor(masks, dtype=torch.bool, device=device)

    h, _ = model.encode(ids, mask=masks, use_phoneme=False)
    _, l_hat = model.length_predictor(h, masks, detach_input=True)
    base_len = torch.ceil(l_hat).long().clamp(min=1, max=model.max_length)

    def pooled_for(lens):
        zc, cm = model.decoder(h, lens, source_mask=masks)
        cl = detect_content_boundary(zc, cm, model.empty_norm_ratio)
        from modules.utils import length_to_mask
        cmask = length_to_mask(cl, zc.shape[1])
        return masked_mean(zc.unsqueeze(1), cmask.unsqueeze(1)).squeeze(1)

    base = pooled_for(base_len)
    out = {}
    for d in deltas:
        pert = torch.ceil(base_len.float() * (1 + d)).long().clamp(min=1, max=model.max_length)
        out[f"Q2_sensitivity/cos_delta{d:+.0%}".replace("%", "pct")] = float(
            _cos(base, pooled_for(pert)).mean()
        )
    return out


# ---------------------------------------------------------------------------
# Q5 -- homofon: prediksi H1 yang dapat dipalsukan
# ---------------------------------------------------------------------------
HOMOPHONES = [
    ("their", "there"), ("to", "too"), ("write", "right"), ("flour", "flower"),
    ("bare", "bear"), ("knight", "night"), ("piece", "peace"), ("sea", "see"),
    ("weak", "week"), ("male", "mail"), ("plain", "plane"), ("sail", "sale"),
]
HOMOPHONE_FRAMES = [
    "i saw the {} yesterday", "this is a {} for you", "we talked about the {}",
]


@torch.no_grad()
def homophone_discrimination(model, device):
    """
    Uji yang secara khusus dapat MEMALSUKAN H1.

    Anchor fonem memetakan pasangan homofon ("their"/"there") ke fonem kanonik
    yang identik, sehingga L_syn menarik keduanya ke titik yang sama sementara
    L_sem menariknya menjauh. Prediksi yang jujur karenanya berbentuk dua sisi:
    anchor fonem LEBIH BAIK pada korupsi permukaan tetapi LEBIH BURUK pada
    homofon. Tanpa metrik ini, klaim H1 hanya melaporkan sisi yang menguntungkan.

    Dilaporkan kemiripan pasangan homofon dalam bingkai kalimat yang sama.
    Nilai TINGGI berarti model meruntuhkan perbedaan makna.
    """
    a_txt, b_txt = [], []
    for w1, w2 in HOMOPHONES:
        for f in HOMOPHONE_FRAMES:
            a_txt.append(f.format(w1))
            b_txt.append(f.format(w2))
    ra = _encode_text_batch(model, a_txt, device)
    rb = _encode_text_batch(model, b_txt, device)
    res = {}
    for k, tag in ((0, "enc"), (1, "dec")):
        if ra[k] is None:
            continue
        sim = _cos(ra[k], rb[k])
        res[f"Q5_homophone/sim_{tag}"] = float(sim.mean())
        # Pembanding: pasangan acak non-homofon dalam bingkai yang sama.
        perm = torch.randperm(rb[k].shape[0], device=device)
        res[f"Q5_homophone/sim_random_{tag}"] = float(_cos(ra[k], rb[k][perm]).mean())
        res[f"Q5_homophone/collapse_margin_{tag}"] = (
            res[f"Q5_homophone/sim_{tag}"] - res[f"Q5_homophone/sim_random_{tag}"]
        )
    return res


# ---------------------------------------------------------------------------
# Q7 -- rasio global versus ekspansi lokal
# ---------------------------------------------------------------------------
@torch.no_grad()
def length_error_by_view_group(out: dict, n_easy: int = 3):
    """
    Galat prediksi panjang dipisah antara view tingkat EASY dan HARD.

    Korupsi hard (word concatenation, random letter space) menghasilkan ekspansi
    LOKAL yang non-uniform, sedangkan prediktor hanya mengeluarkan satu rasio
    global. Kalau galat pada kelompok hard jauh lebih besar, itu bukti langsung
    keterbatasan parametrisasi global -- keberatan yang memang diangkat
    literatur NAT lewat fertility per token.
    """
    if out.get("encoder_only", False):
        return {}
    e = (out["rho_star"] - out["rho_pred"])
    # v=0 anchor (dikecualikan), v=1 grafem kanonik, v=2.. augmentasi
    easy = e[:, 2:2 + n_easy]
    hard = e[:, 2 + n_easy:]
    res = {"Q7_local/mae_clean": float(e[:, 1].abs().mean())}
    if easy.numel():
        res["Q7_local/mae_easy"] = float(easy.abs().mean())
    if hard.numel():
        res["Q7_local/mae_hard"] = float(hard.abs().mean())
    if easy.numel() and hard.numel():
        res["Q7_local/hard_over_easy"] = (
            res["Q7_local/mae_hard"] / max(res["Q7_local/mae_easy"], 1e-8)
        )
    return res


# ---------------------------------------------------------------------------
# Q4 -- PCR pada korupsi HELD-OUT
# ---------------------------------------------------------------------------
@torch.no_grad()
def holdout_corruption_recall(model, texts, device, kinds=("homoglyph", "ocr", "leet")):
    """
    PCR memakai jenis korupsi yang TIDAK ADA di augmenter pelatihan.

    Menjawab keberatan bahwa PCR biasa tidak dapat membedakan kanonikalisasi
    sejati dari hafalan distribusi augmenter. Selisih antara metrik ini dan
    `progressive_corruption_recall` adalah ukuran langsung besarnya overfitting
    terhadap augmenter sendiri.
    """
    from modules.holdout_corruption import corrupt
    base_enc, base_dec = _encode_text_batch(model, texts, device)
    res = {}
    for kind in kinds:
        q = [corrupt(t, kind, seed=i) for i, t in enumerate(texts)]
        q_enc, q_dec = _encode_text_batch(model, q, device)
        for tag, base, qry in (("enc", base_enc, q_enc), ("dec", base_dec, q_dec)):
            if base is None or qry is None:
                continue
            sim = safe_normalize(qry) @ safe_normalize(base).t()
            gold = torch.arange(len(texts), device=device)
            rank = (sim > sim.gather(1, gold[:, None])).sum(1) + 1
            res[f"Q4_holdout/recall@1_{tag}_{kind}"] = float((rank == 1).float().mean())
    return res
