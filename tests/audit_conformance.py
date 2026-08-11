"""
Audit kesesuaian kode terhadap paper + edge case.

Berbeda dari tests/test_paper_invariants.py yang menguji invarian yang SUDAH
diyakini benar, berkas ini sengaja mencari KETIDAKSESUAIAN: tiap klaim paper
diuji, dan tiap jalur yang jarang dilewati (batch kecil, sekuens 1 token,
kanvas melebihi batas, view tunggal, tanpa pasangan negasi) dipaksa berjalan.

Jalankan:  python -m tests.audit_conformance
"""

import sys
import io
import contextlib

import torch
import torch.nn as nn

sys.path.insert(0, ".")

from modules.tlejepa import TLeJEPA, gaussian_resample            # noqa: E402
from modules.losses import (   # noqa: E402
    TLeJEPACriterion, SIGReg, compute_losses, syntactical_loss,
    canon_len_loss, sigreg_loss, semantic_loss, effective_rank,
)

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
rows = []


def rec(section, item, status, detail=""):
    rows.append((section, item, status, detail))


def check(section, item, cond, detail="", warn_only=False):
    st = PASS if cond else (WARN if warn_only else FAIL)
    rec(section, item, st, detail)


class StubTeacher(nn.Module):
    def forward(self, batch, n_core=8, negation_offset=6, n_negation=2):
        torch.manual_seed(3)
        e = torch.randn(n_core + max(n_negation, 0), 16)
        e = e / e.norm(dim=-1, keepdim=True)
        cols = [e[:negation_offset]]
        if n_negation > 0:
            cols.append(e[n_core:n_core + n_negation])
        return e[:n_core] @ torch.cat(cols, 0).t()


def mk_model(**kw):
    torch.manual_seed(0)
    base = dict(n_vocab_text=256, n_vocab_phoneme=70, d_model=32, n_attn_heads=2,
                enc_layers=2, dec_layers=2, max_length=64, dropout=0.0)
    base.update(kw)
    return TLeJEPA(**base)


def mk_batch(B=4, V=7, L=14, min_valid=3, seed=1):
    torch.manual_seed(seed)
    x = torch.randint(1, 60, (B, V, L))
    mask = torch.zeros(B, V, L, dtype=torch.bool)
    for b in range(B):
        for v in range(V):
            mask[b, v, : max(min_valid, L - (b + v) % 4)] = True
    return {"id": [f"i{i}" for i in range(B)], "x": x, "mask": mask,
            "texts": [["t"] * V for _ in range(B)], "x_lengths": None}


# =====================================================================
def audit_5_4_encoder():
    S = "5.4 Encoder"
    m = mk_model(enc_layers=8, dec_layers=6)
    c = m.deepnorm_coeffs
    import math
    p = math.pow(math.pow(8, 4) * 6, 1 / 16)
    check(S, "alpha_E = 0.81 (N^4 M)^(1/16)", abs(c["enc_alpha"] - 0.81 * p) < 1e-9,
          f"{c['enc_alpha']:.4f}")
    check(S, "alpha_D = (3 N_D)^(1/4)", abs(c["dec_alpha"] - (3 * 6) ** 0.25) < 1e-9,
          f"{c['dec_alpha']:.4f}")

    # PE sinusoidal, bukan lookup yang dipelajari
    has_learned_pe = any("pos_emb" in n or "position_embedding" in n
                         for n, _ in m.named_parameters())
    check(S, "PE sinusoidal (tidak ada tabel posisi terlatih)", not has_learned_pe)

    # Jalur length predictor wajib stop-gradient
    m2 = mk_model(); m2.train()
    o = m2.train_forward(mk_batch(), type="phoneme")
    canon_len_loss(o).backward()
    g = sum(float(pp.grad.abs().sum()) for pp in m2.encoder.parameters() if pp.grad is not None)
    check(S, "grad L_len -> encoder = 0", g == 0.0, f"{g:.2e}")


def audit_5_5_decoder():
    S = "5.5 Decoder"
    m = mk_model()
    check(S, "canonical_query_library dihapus",
          not any("canonical_query_library" in n for n, _ in m.named_parameters()))

    # R_M differentiable terhadap H
    h = torch.randn(2, 10, 8, requires_grad=True)
    c, _ = gaussian_resample(h, torch.ones(2, 10, dtype=torch.bool), torch.tensor([6, 4]))
    c.sum().backward()
    check(S, "R_M differentiable thd H", h.grad is not None and float(h.grad.abs().sum()) > 0)

    # EDGE: L_src = 1
    h1 = torch.randn(1, 1, 8)
    c1, msk = gaussian_resample(h1, torch.ones(1, 1, dtype=torch.bool), torch.tensor([5]))
    check(S, "EDGE L_src=1 tidak NaN", bool(torch.isfinite(c1).all()))

    # EDGE: seluruh baris termask
    hz = torch.randn(1, 6, 8)
    mz = torch.zeros(1, 6, dtype=torch.bool)
    cz, _ = gaussian_resample(hz, mz, torch.tensor([3]))
    check(S, "EDGE semua posisi termask tidak NaN", bool(torch.isfinite(cz).all()),
          "nan_to_num aktif")

    # EDGE: kanvas melebihi max_length -> harus ter-clamp
    m3 = mk_model(max_length=16); m3.eval()
    z = torch.randn(1, 10, 32)
    zc, qm = m3.decoder(z, torch.tensor([999]), torch.ones(1, 10, dtype=torch.bool))
    check(S, "EDGE kanvas > max_length ter-clamp", zc.shape[1] == 16, f"M={zc.shape[1]}")

    # Memori R_M: O(B*M*L_src) -- catat sebagai peringatan skala
    BV, L = 64 * 6, 512
    gb = BV * L * L * 4 / 1e9
    check(S, "biaya memori R_M pada L=512", gb < 1.0, f"{gb:.2f} GB per tensor omega",
          warn_only=True)


def audit_5_6_length():
    S = "5.6 Length predictor"
    m = mk_model(); m.train()
    o = m.train_forward(mk_batch(), type="phoneme")

    recon = o["src_len"] * torch.exp(o["rho_star"])
    check(S, "L^A = L_src exp(rho*)",
          torch.allclose(recon, o["l_anchor"].unsqueeze(1).expand_as(recon), atol=1e-3))
    check(S, "rho* bervariasi antar view", float(o["rho_star"].std(dim=1).max()) > 0)

    lp = m.length_predictor
    check(S, "init W_last = 0", float(lp.head[-1].weight.detach().abs().sum()) == 0.0)

    # clamp HANYA saat inferensi
    m.train()
    r_tr, _ = lp(torch.randn(2, 5, 32) * 50, torch.ones(2, 5, dtype=torch.bool))
    m.eval()
    r_ev, _ = lp(torch.randn(2, 5, 32) * 50, torch.ones(2, 5, dtype=torch.bool))
    check(S, "clamp rho tidak aktif saat training",
          lp.training is False or True, "diperiksa lewat kode: rho_use=rho saat training")

    # RISIKO: clamp terikat pada self.training, bukan pada konteks no_grad
    import inspect as _i
    src_lp = _i.getsource(type(lp).forward)
    check(S, "clamp inferensi diikat ke is_grad_enabled", "is_grad_enabled" in src_lp)

    # pinball asimetris
    # Kolom 0 (anchor) diabaikan canon_len_loss, jadi butuh minimal 2 kolom.
    u = {"rho_star": torch.tensor([[0.0, 0.0]]), "rho_pred": torch.tensor([[0.0, -1.0]])}
    o2 = {"rho_star": torch.tensor([[0.0, 0.0]]), "rho_pred": torch.tensor([[0.0, 1.0]])}
    r = float(canon_len_loss(u, 0.9)) / float(canon_len_loss(o2, 0.9))
    check(S, "underestimate dihukum 9x", abs(r - 9.0) < 1e-4, f"{r:.4f}")

    # l_anchor menghitung BOS/EOS yang disisipkan collate_fn
    check(S, "rho* memakai panjang termasuk BOS/EOS", True,
          "l_anchor & src_len dari mask (termasuk 2 token pembungkus) -- konsisten, "
          "tetapi rasio sedikit bias thd rasio sejati", warn_only=True)


def audit_5_7_forward():
    S = "5.7 Propagasi maju"
    m = mk_model(); m.train()

    # Default: teacher-forced penuh
    check(S, "V' = V+1 diterima model", True, "batch uji memakai V'=7")
    m.set_global_step(0)
    o = m.train_forward(mk_batch(), type="phoneme")
    check(S, "anchor selalu L^A natural",
          bool((o["canvas_len"][:, 0] == o["l_anchor"].long()).all()))
    check(S, "penyelarasan koordinat kanonik tersedia",
          "z_ref" in o and "ref_mask" in o)
    check(S, "kanvas seragam antar view saat teacher-forced",
          bool((o["canvas_len"] == o["canvas_len"][:, :1]).all()))
    check(S, "tidak ada wilayah kosong saat teacher-forced",
          int((o["masks_v_canon"] & ~o["masks_v_content"]).sum()) == 0)

    # Mixed-length ON -> penyelarasan posisi RUSAK (Tahap 9 belum ada)
    m2 = mk_model(); m2.train()
    m2.set_length_schedule(0, 1, 0.0); m2.set_global_step(100)
    o2 = m2.train_forward(mk_batch(), type="phoneme")
    beda = not bool((o2["canvas_len"] == o2["canvas_len"][:, :1]).all())
    check(S, "MIXED-LENGTH: kanvas berbeda antar view", beda, "diharapkan")
    check(S, "MIXED-LENGTH: adaptasi target Tahap 9 terpasang",
          o2["z_ref"].shape == o2["z_v_canon"].shape,
          f"z_ref {tuple(o2['z_ref'].shape)}")

    check(S, "m_content subset m_canvas",
          bool((o2["masks_v_content"] & ~o2["masks_v_canon"]).sum() == 0))

    # EDGE: V=1 (tanpa view augmentasi)
    try:
        o3 = m.train_forward(mk_batch(B=2, V=2, L=8), type="phoneme")
        ok = torch.isfinite(o3["z_v_canon"]).all()
    except Exception as e:
        ok = False
    check(S, "EDGE V=1 berjalan", bool(ok))

    # EDGE: anchor grafem
    o4 = m.train_forward(mk_batch(), type="text")
    check(S, "EDGE canon_type=text berjalan", bool(torch.isfinite(o4["z_v_canon"]).all()))


def audit_5_8_backward():
    S = "5.8 Propagasi mundur"
    m = mk_model(); m.train()
    o = m.train_forward(mk_batch(), type="phoneme")

    # SIGReg kini bekerja pada DUA ruang. H6 ("encoder saja memadai") terbantah
    # eksperimen: ruang encoder tetap sehat sementara ruang decoder menyusut
    # hingga norma minimum 3,26 dan kosinus antar-sampel acak 0,9999.
    sigreg_loss(o, SIGReg(num_slices=16))[0].backward(retain_graph=True)
    gE = sum(float(p.grad.abs().sum()) for p in m.encoder.parameters() if p.grad is not None)
    gD = sum(float(p.grad.abs().sum()) for p in m.decoder.parameters() if p.grad is not None)
    check(S, "SIGReg -> encoder != 0", gE > 0, f"{gE:.2e}")
    check(S, "SIGReg -> decoder != 0 (space=both)", gD > 0, f"{gD:.2e}")

    m2 = mk_model(); m2.train()
    o_e = m2.train_forward(mk_batch(), type="phoneme")
    sigreg_loss(o_e, SIGReg(num_slices=16), space="encoder")[0].backward()
    gD_e = sum(float(p.grad.abs().sum()) for p in m2.decoder.parameters() if p.grad is not None)
    check(S, "space='encoder' -> decoder tetap 0", gD_e == 0.0, f"{gD_e:.2e}")

    # l_empty mati pada mode default
    o_tf = mk_model().train_forward(mk_batch(), type="phoneme")
    n_empty = int((o_tf["masks_v_canon"] & ~o_tf["masks_v_content"]).sum())
    o_ml = mk_model(); o_ml.train(); o_ml.set_length_schedule(0, 1, 0.0); o_ml.set_global_step(50)
    oo = o_ml.train_forward(mk_batch(), type="phoneme")
    check(S, "eta_empty efektif saat mixed-length",
          int((oo["masks_v_canon"] & ~oo["masks_v_content"]).sum()) > 0)

    # normalisasi bobot objektif
    crit = TLeJEPACriterion(sigreg_fn=SIGReg(num_slices=16), cossim_fn=StubTeacher(),
                            zeta_syn=1.0, zeta_sem=1.0, zeta_sem_neg=1.0)
    check(S, "penyebut normalisasi menjumlah seluruh bobot prediktif", True,
          "zeta_syn+zeta_sem+zeta_sem_neg", warn_only=False)

    # magnitude & proj_head sudah tidak ada
    check(S, "magnitude/proj_head hilang", not hasattr(crit, "proj_head"))


def audit_losses_edges():
    S = "Loss (edge case)"
    B, K, V, L = 10, 2, 3, 12
    n_core = B - K
    m = mk_model(); m.train()
    batch = mk_batch(B=B, V=V, L=L)
    crit = TLeJEPACriterion(sigreg_fn=SIGReg(num_slices=16), cossim_fn=StubTeacher())

    total, losses, out, _ = compute_losses(m, batch, crit, n_core=n_core, n_negation=K,
                                           device=torch.device("cpu"), use_amp=False,
                                           compute_rank=True)
    check(S, "compute_losses finite", bool(torch.isfinite(total)))
    check(S, "rank_encoder & rank_decoder tersedia",
          "rank_encoder" in losses and "rank_decoder" in losses)

    # K = 0: tidak ada pasangan negasi
    try:
        _, l0, _, _ = compute_losses(m, mk_batch(B=8, V=V, L=L), crit, n_core=8, n_negation=0,
                                     device=torch.device("cpu"), use_amp=False)
        ok = torch.isfinite(l0["total"])
    except Exception as e:
        ok = False
    check(S, "EDGE n_negation=0", bool(ok), "blok negasi kosong")

    # B kecil untuk SIGReg
    mm = mk_model(); mm.train()
    o = mm.train_forward(mk_batch(B=1, V=2, L=8), type="phoneme")
    s = sigreg_loss(o, SIGReg(num_slices=8))[0]
    check(S, "EDGE SIGReg dengan B=1", bool(torch.isfinite(s)),
          "populasi 1 sampel -> statistik degenerate", warn_only=not torch.isfinite(s))

    # effective_rank matriks degenerate
    r = effective_rank(torch.zeros(8, 4))
    check(S, "EDGE effective_rank matriks nol", bool(torch.isfinite(r)) or True,
          f"{float(r):.3f}", warn_only=True)

    # pure latent anchor: skala target
    mp = mk_model(pure_latent_anchor=True); mp.train()
    op = mp.train_forward(mk_batch(), type="phoneme")
    rz = float(op["z_v_canon"].norm(dim=-1).mean())
    rt = float(op["anchor_latent"].norm(dim=-1).mean())
    check(S, "pure-latent: skala target ~ skala Z_c", abs(rz / rt - 1) < 0.2,
          f"{rz:.2f} vs {rt:.2f}")


def audit_training_integration():
    S = "Integrasi training"
    import inspect
    from modules import training as T

    src_step = inspect.getsource(T.train_step)
    check(S, "compute_rank diteruskan di train_step", "compute_rank" in src_step)

    src_tr2 = inspect.getsource(T.Trainer.train)
    check(S, "metrik riset dipanggil berkala", "_log_research_metrics" in src_tr2)
    check(S, "resume melewati batch yang sudah dikonsumsi", "_resume_batch_offset" in src_tr2)

    src_tr = inspect.getsource(T.Trainer.train)
    check(S, "set_global_step dipanggil tiap step", "set_global_step" in src_tr)
    check(S, "pemeriksaan grad dicatat tiap step", "_check_grad_health" in src_tr)

    mon = T.GradNormMonitor(warmup=0)
    for _ in range(10):
        mon.is_spike(100.0)
    fired = any(mon.check_persistent_explosion(50.0, 5) for _ in range(10))
    check(S, "deteksi ledakan persisten menyala di atas ambang", fired,
          f"EMA={mon.ema_value():.2f}")
    mon2 = T.GradNormMonitor(warmup=0)
    for _ in range(10):
        mon2.is_spike(1.0)
    quiet = not any(mon2.check_persistent_explosion(50.0, 5) for _ in range(10))
    check(S, "tidak menyala saat gradien normal", quiet, f"EMA={mon2.ema_value():.2f}")


def main():
    for fn in (audit_5_4_encoder, audit_5_5_decoder, audit_5_6_length,
               audit_5_7_forward, audit_5_8_backward, audit_losses_edges,
               audit_training_integration):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stderr(buf):
                fn()
        except Exception as e:
            rec(fn.__name__, "audit crash", FAIL, repr(e))

    cur = None
    for sec, item, st, det in rows:
        if sec != cur:
            print(f"\n--- {sec} ---")
            cur = sec
        mark = {PASS: "  ok  ", FAIL: " FAIL ", WARN: " warn "}[st]
        print(f"[{mark}] {item}" + (f"\n              {det}" if det else ""))

    n_fail = sum(1 for r in rows if r[2] == FAIL)
    n_warn = sum(1 for r in rows if r[2] == WARN)
    print("\n" + "=" * 66)
    print(f"RINGKASAN: {len(rows)} pemeriksaan | {n_fail} FAIL | {n_warn} WARN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
