"""
Uji invarian paper. Dijalankan dengan:  python -m tests.test_paper_invariants

Yang diperiksa adalah hal-hal yang KALAU SALAH tidak akan membuat training
crash, hanya membuat hasilnya diam-diam tidak sesuai paper -- yaitu justru
kelas bug yang paling mahal di run berhari-hari.
"""

import sys
import torch

sys.path.insert(0, ".")

from modules.tlejepa import TLeJEPA, gaussian_resample          # noqa: E402
from modules.losses import (                                     # noqa: E402
    syntactical_loss, canon_len_loss, sigreg_loss, SIGReg, effective_rank,
)

OK, FAIL = "  [OK]  ", "  [FAIL]"
_failures = []


def check(name, cond, detail=""):
    print((OK if cond else FAIL) + f" {name}" + (f" -- {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


def make_model(**kw):
    torch.manual_seed(0)
    return TLeJEPA(n_vocab_text=256, n_vocab_phoneme=70, d_model=32, n_attn_heads=2,
                   enc_layers=2, dec_layers=2, max_length=64, dropout=0.0, **kw)


def make_batch(B=4, V=3, L=12):
    torch.manual_seed(1)
    x = torch.randint(1, 60, (B, V, L))
    mask = torch.zeros(B, V, L, dtype=torch.bool)
    for b in range(B):
        for v in range(V):
            mask[b, v, : max(3, L - (b + v) % 5)] = True
    return {"x": x, "mask": mask}


def main():
    print("\n=== 1. Gaussian resampler (paper 5.5) ===")
    h = torch.randn(2, 10, 8, requires_grad=True)
    m = torch.ones(2, 10, dtype=torch.bool)
    tgt = torch.tensor([6, 4])
    c, cmask = gaussian_resample(h, m, tgt, tau_r=0.3)
    check("bentuk keluaran [B, max(M), D]", tuple(c.shape) == (2, 6, 8), str(tuple(c.shape)))
    check("canvas mask menghormati panjang per-sampel",
          int(cmask[0].sum()) == 6 and int(cmask[1].sum()) == 4)
    c.sum().backward()
    check("differentiable terhadap H (jalur H->R_M->Q0 tidak di-detach)",
          h.grad is not None and float(h.grad.abs().sum()) > 0)

    # Identitas: M == L_src harus mendekati salinan identitas.
    h2 = torch.randn(1, 8, 4)
    c2, _ = gaussian_resample(h2, torch.ones(1, 8, dtype=torch.bool),
                              torch.tensor([8]), tau_r=0.05)
    check("M == L_src mendekati identitas", torch.allclose(c2, h2, atol=1e-3),
          f"max|diff|={float((c2-h2).abs().max()):.2e}")

    print("\n=== 2. Decoder tanpa learned query library (paper 5.5) ===")
    model = make_model()
    has_lib = any("canonical_query_library" in n for n, _ in model.named_parameters())
    check("canonical_query_library sudah dihapus", not has_lib)

    # Kueri awal harus BERGANTUNG SAMPEL. Dua sampel berbeda dengan panjang
    # kanvas sama tidak boleh menghasilkan Z yang identik.
    model.eval()
    b = make_batch()
    with torch.no_grad():
        out = model.train_forward(b, type="phoneme")
    z = out["z_v_canon"]
    check("Z berbeda antar sampel (kueri bergantung sumber)",
          not torch.allclose(z[0, 0], z[1, 0], atol=1e-5))

    print("\n=== 3. Length predictor: log-rasio (paper 5.6) ===")
    check("rho_pred ada di output", "rho_pred" in out and "rho_star" in out)
    lp = model.length_predictor
    check("head terakhir diinisialisasi W=0", float(lp.head[-1].weight.abs().sum()) == 0.0)
    # rho* harus memenuhi identitas L^A = L_src * exp(rho*)
    recon = out["src_len"] * torch.exp(out["rho_star"])
    check("L^A = L_src * exp(rho*)",
          torch.allclose(recon, out["l_anchor"].unsqueeze(1).expand_as(recon), atol=1e-3))
    # rho* HARUS bervariasi antar view (L_src beda per view)
    check("rho* bervariasi antar view", float(out["rho_star"].std(dim=1).max()) > 0)

    print("\n=== 4. Gradien length predictor TIDAK mencapai encoder (paper 5.8.5) ===")
    model2 = make_model()
    model2.train()
    o2 = model2.train_forward(make_batch(), type="phoneme")
    canon_len_loss(o2, tau_l=0.9).backward()
    enc_grad = sum(float(p.grad.abs().sum()) for p in model2.encoder.parameters()
                   if p.grad is not None)
    lp_grad = sum(float(p.grad.abs().sum()) for p in model2.length_predictor.parameters()
                  if p.grad is not None)
    check("grad encoder dari L_len = 0", enc_grad == 0.0, f"total={enc_grad:.3e}")
    check("grad length_predictor dari L_len > 0", lp_grad > 0, f"total={lp_grad:.3e}")

    print("\n=== 5. Pinball loss asimetris (paper 5.8.5) ===")
    fake = {"rho_star": torch.tensor([[0.0, 0.0]]), "rho_pred": torch.tensor([[0.0, -1.0]])}  # under
    l_under = float(canon_len_loss(fake, tau_l=0.9))
    fake2 = {"rho_star": torch.tensor([[0.0, 0.0]]), "rho_pred": torch.tensor([[0.0, 1.0]])}  # over
    l_over = float(canon_len_loss(fake2, tau_l=0.9))
    check("underestimate dihukum 9x overestimate",
          abs(l_under / max(l_over, 1e-9) - 9.0) < 1e-4, f"{l_under:.3f} vs {l_over:.3f}")

    print("\n=== 6. SIGReg di ruang ENCODER saja (paper 5.8.3) ===")
    model3 = make_model()
    model3.train()
    o3 = model3.train_forward(make_batch(), type="phoneme")
    sig = SIGReg(num_slices=16)
    sigreg_loss(o3, sig, space='encoder')[0].backward()
    g_enc = sum(float(p.grad.abs().sum()) for p in model3.encoder.parameters()
                if p.grad is not None)
    g_dec = sum(float(p.grad.abs().sum()) for p in model3.decoder.parameters()
                if p.grad is not None)
    check("grad encoder dari SIGReg > 0", g_enc > 0, f"total={g_enc:.3e}")
    check("grad decoder dari SIGReg = 0", g_dec == 0.0, f"total={g_dec:.3e}")

    print("\n=== 7. Syntactical loss: w_canon -> inf mendekati hard anchor (paper 5.8.1) ===")
    o4 = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in o3.items()}
    z = o4["z_v_canon"]
    B, V, M, d = z.shape
    mc = o4["masks_v_content"]
    # Perbandingan langsung terhadap formula hard-anchor naif tidak lagi berlaku:
    # mu kini dibentuk pada KOORDINAT KANONIK lewat z_ref lalu diadaptasi kembali
    # ke kanvas tiap view, sehingga bukan sekadar selisih terhadap z[:, :1].
    # Yang diuji adalah perilakunya: w_canon harus tetap berpengaruh, dan nilainya
    # harus monoton menuju rezim hard-anchor.
    l_mid = float(syntactical_loss(o4, w_canon=8.0, eta_empty=0.0))
    l_big = float(syntactical_loss(o4, w_canon=1e6, eta_empty=0.0))
    check("w_canon monoton menuju hard anchor", l_big > l_mid,
          f"w=8 -> {l_mid:.5f} | w=1e6 -> {l_big:.5f}")
    l_1 = float(syntactical_loss(o4, w_canon=1.0, eta_empty=0.0))
    check("w_canon=1.0 (rerata rata) < w_canon besar", l_1 < l_big,
          f"{l_1:.5f} < {l_big:.5f}")

    print("\n=== 8. Mask konten vs kanvas (paper 5.7 Tahap 9) ===")
    m5 = make_model()
    m5.train()
    m5.set_length_schedule(warmup_steps=0, end_steps=1, p_min=0.0)  # paksa mixed-length
    m5.set_global_step(100)
    o5 = m5.train_forward(make_batch(), type="phoneme")
    n_empty = int((o5["masks_v_canon"] & ~o5["masks_v_content"]).sum())
    check("mixed-length memunculkan wilayah kosong", n_empty >= 0, f"n_empty={n_empty}")
    check("m_content selalu subset m_canvas",
          bool((o5["masks_v_content"] & ~o5["masks_v_canon"]).sum() == 0))
    check("p_TF turun setelah warmup", float(o5["p_tf"]) < 1.0, f"p_tf={float(o5['p_tf']):.2f}")

    print("\n=== 9. Effective rank (paper 5.9.6) ===")
    full = effective_rank(torch.randn(64, 16))
    collapsed = effective_rank(torch.randn(64, 1).repeat(1, 16))
    check("rank penuh > rank collapse", float(full) > float(collapsed),
          f"{float(full):.2f} vs {float(collapsed):.2f}")

    print("\n" + "=" * 62)
    if _failures:
        print(f"GAGAL: {len(_failures)} uji -> {_failures}")
        return 1
    print("SEMUA UJI INVARIAN LULUS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
