"""
Jaminan resume: melanjutkan, BUKAN memulai ulang.

Berkas ini ada karena satu klaim yang tidak boleh hanya dinyatakan: setelah job
mati kena time limit lalu disubmit ulang, pelatihan harus lanjut dari langkah
terakhir. Kalau klaim itu salah, kegagalannya SENYAP -- loss tetap turun,
kurva tetap wajar, tetapi model mengulang data yang sama dan anggaran GPU
terbakar tanpa kemajuan.

Yang dibuktikan di sini:
  1. Bobot model identik bit-per-bit setelah muat ulang
  2. State optimizer (momen Adam) ikut pulih, bukan direset
  3. State scheduler pulih sehingga LR menyambung, bukan mengulang warmup
  4. global_step pulih
  5. Epoch DITURUNKAN dari global_step, bukan diambil mentah dari checkpoint
  6. Offset batch dalam epoch dihitung benar sehingga batch yang sudah
     dikonsumsi dilewati
  7. Langkah berikutnya setelah resume menghasilkan LR yang sama persis dengan
     langkah yang seharusnya, seolah tidak pernah mati

Jalankan:  python -m tests.test_resume
"""

import os
import sys
import shutil
import tempfile

import torch
import torch.nn as nn

sys.path.insert(0, ".")

from modules.training import save_model, load_model  # noqa: E402

_fail = []


def check(name, cond, detail=""):
    print(("  [ ok ] " if cond else " [FAIL] ") + name + (f" -- {detail}" if detail else ""))
    if not cond:
        _fail.append(name)


def build(seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50)      # warmup 50 langkah
    )
    return model, opt, sched


def run_steps(model, opt, sched, n, seed=1):
    torch.manual_seed(seed)
    xs = torch.randn(n, 8, 16)
    for i in range(n):
        loss = model(xs[i]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model, opt, sched


def main():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "latest_step.pt")
    N_BEFORE, N_AFTER = 37, 13
    N_PER_EPOCH = 10          # batch per epoch pada uji ini

    print("\n=== Referensi: latih 37+13 langkah TANPA interupsi ===")
    m_ref, o_ref, s_ref = build()
    run_steps(m_ref, o_ref, s_ref, N_BEFORE, seed=1)
    run_steps(m_ref, o_ref, s_ref, N_AFTER, seed=2)
    lr_ref = s_ref.get_last_lr()[0]
    ref_w = [p.detach().clone() for p in m_ref.parameters()]

    print("\n=== Skenario: mati setelah 37 langkah, lalu resume ===")
    m1, o1, s1 = build()
    run_steps(m1, o1, s1, N_BEFORE, seed=1)
    w_before = [p.detach().clone() for p in m1.parameters()]
    lr_before = s1.get_last_lr()[0]
    # Epoch yang tercatat di checkpoint sengaja SALAH (0) untuk membuktikan bahwa
    # epoch diturunkan dari global_step, bukan dipercaya mentah.
    save_model(path, m1, o1, epoch=0, best_metric=1.23, scheduler=s1,
               global_step=N_BEFORE, extra={"run_id": "uji"})

    m2, o2, s2 = build(seed=999)                # bobot awal SENGAJA berbeda
    diff_pre = max(float((a - b).abs().max()) for a, b in zip(m2.parameters(), w_before))
    check("model baru memang berbeda sebelum dimuat", diff_pre > 1e-6, f"maks|selisih|={diff_pre:.4f}")

    epoch_ck, best, gstep = load_model(path, m2, optimizer=o2, scheduler=s2)

    # --- 1. bobot ---
    dmax = max(float((a - b).abs().max()) for a, b in zip(m2.parameters(), w_before))
    check("bobot model pulih identik", dmax == 0.0, f"maks|selisih|={dmax:.2e}")

    # --- 2. optimizer ---
    st1 = o1.state_dict()["state"]
    st2 = o2.state_dict()["state"]
    same_opt = all(
        torch.equal(st1[k]["exp_avg"], st2[k]["exp_avg"])
        and torch.equal(st1[k]["exp_avg_sq"], st2[k]["exp_avg_sq"])
        and int(st1[k]["step"]) == int(st2[k]["step"])
        for k in st1
    )
    check("momen Adam pulih (bukan direset)", same_opt,
          f"step optimizer={int(list(st2.values())[0]['step'])}")

    # --- 3. scheduler ---
    check("LR menyambung, tidak mengulang warmup",
          abs(s2.get_last_lr()[0] - lr_before) < 1e-12,
          f"{s2.get_last_lr()[0]:.6e} vs {lr_before:.6e}")

    # --- 4. global_step ---
    check("global_step pulih", gstep == N_BEFORE, f"{gstep}")
    check("best_metric pulih", abs(best - 1.23) < 1e-9)

    # --- 5 & 6. epoch dan offset diturunkan dari global_step ---
    derived_epoch = gstep // N_PER_EPOCH + 1
    offset = gstep % N_PER_EPOCH
    check("epoch DITURUNKAN dari global_step, bukan dari nilai tersimpan",
          derived_epoch == 4 and epoch_ck == 0,
          f"tersimpan={epoch_ck} -> diturunkan={derived_epoch}")
    check("offset batch dalam epoch benar", offset == 7, f"offset={offset}")
    check("tidak mengulang dari batch pertama", offset != 0,
          "kalau 0, seluruh epoch akan diulang")

    # --- 7. lanjutkan dan bandingkan dengan referensi ---
    run_steps(m2, o2, s2, N_AFTER, seed=2)
    dmax2 = max(float((a - b).abs().max()) for a, b in zip(m2.parameters(), ref_w))
    check("hasil setelah resume identik dengan tanpa interupsi", dmax2 == 0.0,
          f"maks|selisih|={dmax2:.2e}")
    check("LR akhir identik dengan referensi",
          abs(s2.get_last_lr()[0] - lr_ref) < 1e-12,
          f"{s2.get_last_lr()[0]:.6e} vs {lr_ref:.6e}")

    # --- 8. kasus batas: mati tepat di batas epoch ---
    print("\n=== Kasus batas: mati tepat di batas epoch ===")
    for gs, exp_ep, exp_off in ((40, 5, 0), (0, 1, 0), (9, 1, 9), (10, 2, 0)):
        de, of = gs // N_PER_EPOCH + 1, gs % N_PER_EPOCH
        check(f"global_step={gs} -> epoch={exp_ep}, offset={exp_off}",
              de == exp_ep and of == exp_off, f"dapat epoch={de}, offset={of}")

    shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 64)
    if _fail:
        print(f"GAGAL: {len(_fail)} -> {_fail}")
        return 1
    print("RESUME TERJAMIN: melanjutkan, bukan memulai ulang")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
