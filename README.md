# TLeJEPA

Kode diselaraskan ketat dengan paper (`tlejepa_revisi.tex`) Bagian 5.2–5.9 dan 6.

Verifikasi: `python -m tests.test_paper_invariants` (9 kelompok) dan
`python -m tests.audit_conformance` (45 pemeriksaan, termasuk edge case).

## Perubahan pokok

| Komponen | Sebelum | Sekarang |
|---|---|---|
| Jumlah view | $V$ | **$V' = V+1$**: anchor ditambahkan, tidak menggantikan |
| Grafem kanonik | tidak terjamin hadir | selalu di indeks 1, tak pernah jadi anchor |
| Mode anchor teks | tanpa view fonem | indeks 0 dan 1 identik, redundansi disengaja |
| Adaptasi target | hanya irisan mask | zero-pad / **resample laten + mask** (Tahap 9) |
| $\mathcal{L}_\text{len}$ | seluruh view | **kecualikan anchor** ($\rho^*_{n,0}=0$ trivial) |
| Pagar kanvas | tidak ada | $\hat L \le \lceil c_\text{max} L^A \rceil$, routing saja |
| Mixed-length | mati (`p_tf_min=1`) | **aktif** (`p_tf_min=0.5`) |
| SIGReg | encoder saja | encoder, `--sigreg-space decoder` untuk R10 |
| Encoder-only | tidak ada | `--encoder-only` untuk R11 |
| Alerting | email SMTP | **dihapus**; dicatat ke `events.log`, watchdog memulihkan |
| Resume | ulang epoch dari awal | **lanjut di tengah epoch** |
| TensorBoard | loss saja | metrik riset per hipotesis |

## Batas cluster

| Batas | Nilai | Diterapkan |
|---|---|---|
| Memori maksimum | 200G | tertinggi R7 = 200G |
| `gpu-long` maksimum | 3 hari | seluruh `--time=3-00:00:00` |
| Partisi | hanya `gpu-long` | TensorBoard & watchdog ikut ke sini |
| `--num-workers` | 4 | diturunkan dari 8 mengikuti memori 160G |

TensorBoard dan watchdog **tidak meminta `--gres`**, jadi keduanya tidak menahan
kartu GPU — hanya memakai core CPU pada node GPU.

## Pemulihan otomatis saat kena time limit

Matriks ini butuh lebih dari 3 hari, jadi setiap run pasti kena time limit
sedikitnya sekali. Ada **dua jalur** pemulihan yang saling menutup:

**Jalur 1 — self-resubmit.** `--signal=B:USR1@300` membuat SLURM mengirim
SIGUSR1 300 detik sebelum SIGKILL. Trainer memakai jendela itu untuk menulis
`latest_step.pt`, lalu trap di skrip run men-submit dirinya sendiri kembali.
Butuh izin `sbatch` dari dalam job.

**Jalur 2 — watchdog.** Memantau antrian tiap 5 menit; run yang hilang tetapi
belum menulis `FINISHED` disubmit ulang. Kalau cluster melarang `sbatch` dari
dalam job, watchdog dijalankan di login node tanpa SLURM sama sekali:

```bash
nohup bash slurm/watchdog.slurm > logs/watchdog.log 2>&1 &
```

Kedua jalur aman berjalan bersamaan: watchdog hanya men-submit run yang **tidak
ada** di antrian, dan keduanya berhenti begitu `FINISHED` tertulis. Dibatasi 40
kali self-resubmit dan 20 kali oleh watchdog.

Untuk menguji apakah cluster mengizinkan `sbatch` dari dalam job:

```bash
sbatch --wrap='sbatch --wrap=hostname'
```

Kalau berhasil, jalur 1 aktif dan watchdog cukup jadi cadangan. Kalau ditolak,
jalankan watchdog di login node.

## Jaminan resume

```bash
python -m tests.test_resume
```

Membuktikan bahwa resume **melanjutkan**, bukan memulai ulang — klaim yang
kalau salah gagalnya senyap: loss tetap turun, kurva tetap wajar, tetapi data
yang sama diulang dan anggaran GPU terbakar tanpa kemajuan.

Yang dibuktikan: bobot pulih identik bit-per-bit; momen Adam pulih (bukan
direset); LR menyambung tanpa mengulang warmup; `global_step` pulih; epoch
**diturunkan dari `global_step`**, bukan dipercaya mentah dari checkpoint;
offset batch benar sehingga batch yang sudah dikonsumsi dilewati. Uji terakhir
paling menentukan: melatih 37 langkah, mati, resume, lanjut 13 langkah — hasilnya
**identik bit-per-bit** (selisih 0.00e+00) dengan melatih 50 langkah tanpa
interupsi.

## Setup

```bash
conda activate avalonai
pip install torch sentence-transformers tensorboard scikit-learn matplotlib tqdm datasets
echo 'export NGROK_AUTHTOKEN="..."' >> ~/.bashrc   # untuk TensorBoard publik
```

## Menjalankan

```bash
bash slurm/submit_all.sh              # 12 run + TensorBoard + watchdog
bash slurm/submit_all.sh R1 R2        # subset
TLEJEPA_SEED=43 bash slurm/submit_all.sh R1 R2
```

| Run | GPU | Menguji |
|---|---|---|
| R1 | h100-96 | referensi |
| R2 | a100-80 | H1 modalitas anchor |
| R3 / R4 | a100-80 / h100-96 | H4 rekonstruksi vs prediksi laten |
| R5 | a100-80 | H5 asimetri anchor |
| R6 / R7 | a100-80 / h100-96 | penskalaan |
| R8 | h100-96 | H8 objektif sintaktik |
| R9 | a100-80 | H7 supervisi semantik |
| R10 | h100-96 | H6 ruang SIGReg |
| R11a / R11b | a100-80 / h100-96 | H3 dimensi sekuensial |

Enam A100-80 dan enam H100-96 — beban merata, tidak menumpuk di satu tipe.

## Watchdog

Menggantikan alerting email. Memberi tahu manusia bahwa job mati tidak
memulihkan apa pun, dan jeda sampai email dibaca itu sendiri sudah mahal pada
run berhari-hari.

Setiap 5 menit watchdog memeriksa antrian. Run yang tidak ada di antrian tetapi
belum menulis penanda `FINISHED` dianggap mati lalu disubmit ulang; skrip run
otomatis menambahkan `--resume` kalau `latest_step.pt` sudah ada. Dibatasi 20
percobaan per run supaya kegagalan permanen tidak jadi loop tak berujung.

```bash
tail -f checkpoints/*/events.log     # SIGNAL_SAVE, GRAD_EXPLOSION, RESUME_SKIP, CRASH
cat logs/watchdog_state.tsv          # jumlah percobaan ulang per run
```

## Resume di tengah epoch

Sebelumnya `--resume` memuat bobot yang benar tapi mengulang epoch dari batch
pertama: data yang sama diproses dua kali dan sisa epoch tak pernah dilihat.
Sekarang epoch diturunkan dari `global_step`, dan batch yang sudah dikonsumsi
dilewati:

```
_resume_batch_offset = global_step % len(train_loader)
derived_epoch        = global_step // len(train_loader) + 1
```

Dicatat sebagai `RESUME_SKIP` di `events.log`.

## TensorBoard

Satu instance untuk seluruh matriks, di partisi CPU, diekspos lewat ngrok. URL
ditulis ke `logs/tensorboard_url.txt` karena hostname gratis berubah tiap restart.

Panel dikelompokkan per hipotesis:

| Tag | Menguji | Arah yang diharapkan |
|---|---|---|
| `H1_invariance/recall@1_*_lv*` | H1 | kurva landai terhadap tingkat korupsi |
| `H2_dual_use/token_rank` | H2 | tinggi; runtuh berarti rekonstruksi mustahil |
| `H3_word_order/acc_*` | H3 | jauh di atas 0,5 |
| `H5_H6_H8_rank/encoder` vs `/decoder` | H5, H6, H8 | keduanya sehat |
| `H7_negation/delta_cos_*` | H7 | positif |
| `H9_length/over_frac` | H9 | merangkak ke $\tau_L = 0{,}9$ |
| `train_step/grad_norm_ema` | stabilitas | di bawah 50 |

`H5_H6_H8_rank/decoder` adalah kurva paling penting di awal: kalau ia runtuh
sementara `encoder` sehat, berarti regularisasi ruang encoder saja **tidak**
memadai dan H6 gugur.
