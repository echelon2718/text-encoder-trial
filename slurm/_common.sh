#!/bin/bash
# Konfigurasi bersama seluruh job TLeJEPA. Ubah di SINI saja.

# ---------------------------------------------------------------------------
# NAMA SPESIFIK CLUSTER -- periksa dengan `bash slurm/preflight.sh` sebelum
# submit. Diubah di SINI saja; submit_all.sh meneruskannya sebagai argumen
# baris perintah sbatch, yang menimpa direktif #SBATCH di dalam skrip run.
# (Direktif #SBATCH tidak bisa memakai variabel shell, jadi override adalah
# satu-satunya cara memusatkan konfigurasi ini.)
export TLEJEPA_GPU_PARTITION="${TLEJEPA_GPU_PARTITION:-gpu-long}"
# TensorBoard dan watchdog ikut gpu-long karena tidak ada partisi CPU terpisah.
# Keduanya TIDAK meminta --gres, jadi tidak menahan kartu GPU: mereka hanya
# memakai core CPU pada node GPU.
export TLEJEPA_CPU_PARTITION="${TLEJEPA_CPU_PARTITION:-gpu-long}"
export TLEJEPA_GPU_A="${TLEJEPA_GPU_A:-h100-96}"     # tipe GPU pertama
export TLEJEPA_GPU_B="${TLEJEPA_GPU_B:-a100-80}"     # tipe GPU kedua
export TLEJEPA_TIME_LIMIT="${TLEJEPA_TIME_LIMIT:-3-00:00:00}"
export TLEJEPA_WD_TIME_LIMIT="${TLEJEPA_WD_TIME_LIMIT:-3-00:00:00}"
# Batas keras cluster: memori maksimum 200G, gpu-long maksimum 3 hari.
# Seluruh --mem di skrip run sudah <= 200G; jangan dinaikkan tanpa
# memeriksa ulang batas partisi.
# ---------------------------------------------------------------------------

export TLEJEPA_ROOT="${TLEJEPA_ROOT:-$HOME/tlejepa-revised}"
export TLEJEPA_CONDA_ENV="${TLEJEPA_CONDA_ENV:-avalonai}"

# SEMUA run menulis ke root TensorBoard yang sama supaya satu instance bisa
# membandingkan seluruh matriks dalam satu grafik.
export TLEJEPA_TB_ROOT="${TLEJEPA_TB_ROOT:-$TLEJEPA_ROOT/runs/tlejepa}"
export TLEJEPA_CKPT_ROOT="${TLEJEPA_CKPT_ROOT:-$TLEJEPA_ROOT/checkpoints}"

# Hyperparameter yang SAMA di seluruh matriks. Yang berubah hanya faktor yang
# sedang diablasi, ditulis eksplisit di tiap skrip run.
export COMMON_ARGS="\
--dataset-mode huggingface \
--batch-size 64 \
--num-workers 4 \
--n_singlish 3 \
--n_premise_and_negation 6 \
--max-length 2048 \
--epochs 5 \
--max-steps 60000 \
--lr 1e-4 \
--warmup-ratio 0.05 \
--min-lr-ratio 0.1 \
--lambda_ 0.5 \
--zeta-syn 1.0 \
--zeta-sem 1.0 \
--zeta-sem-neg 1.0 \
--gamma-len 1.0 \
--tau-l 0.9 \
--tau-r 0.3 \
--c-max 2.0 \
--p-tf-min 0.5 \
--tf-warmup-frac 0.3 \
--tf-end-frac 0.6 \
--num-slices 256 \
--visualize-every-n-steps 2000 \
--research-every-n-steps 2000 \
--save-every-n-steps 500 \
--ckpt-root $TLEJEPA_CKPT_ROOT \
--log-dir $TLEJEPA_TB_ROOT"

# Preset ukuran. d_h = 64 konstan di seluruh ukuran.
export SIZE_SMALL="--enc-layers 4  --dec-layers 2 --d-model 256 --heads 4"
export SIZE_BASE="--enc-layers 8  --dec-layers 6 --d-model 512 --heads 8"
export SIZE_LARGE="--enc-layers 12 --dec-layers 8 --d-model 768 --heads 12"
# R11: encoder-only. Enc-8 lebih ringan, Enc-16 setara parameter Base (-0.5%).
export SIZE_ENC8="--enc-layers 8  --dec-layers 1 --d-model 512 --heads 8 --encoder-only"
export SIZE_ENC16="--enc-layers 16 --dec-layers 1 --d-model 512 --heads 8 --encoder-only"

# Bersihkan variabel SLURM warisan sebelum apa pun dijalankan.
#
# `--export=ALL` meneruskan SELURUH environment penyubmit, termasuk
# SLURM_CPUS_PER_TASK milik alokasi lain (mis. ketika job disubmit dari dalam
# job lain oleh watchdog, atau dari sesi salloc). Slurm >= 22.05 juga menetapkan
# SLURM_TRES_PER_TASK dari --cpus-per-task job yang baru. Kalau keduanya
# berbeda, srun berhenti dengan:
#
#   srun: fatal: cpus-per-task set by two different environment variables
#         SLURM_CPUS_PER_TASK=20 != SLURM_TRES_PER_TASK=cpu=16
#
# Nilai yang benar adalah milik alokasi job INI, yang dapat dibaca dari
# SLURM_JOB_CPUS_PER_NODE. Variabel warisan dibuang, bukan ditimpa, supaya tidak
# ada dua sumber kebenaran yang tersisa.
tlejepa_sanitize_slurm_env() {
    unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK SRUN_CPUS_PER_TASK
    unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE
    unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_GPUS_ON_NODE
    unset SLURM_NTASKS SLURM_NPROCS SLURM_TASKS_PER_NODE
}

tlejepa_setup_env() {
    tlejepa_sanitize_slurm_env
    export MALLOC_ARENA_MAX=2
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate "$TLEJEPA_CONDA_ENV"
    cd "$TLEJEPA_ROOT" || exit 1
    mkdir -p logs "$TLEJEPA_TB_ROOT" "$TLEJEPA_CKPT_ROOT"
}

# Apakah run ini sudah selesai normal? Trainer menulis FINISHED ke events.log
# saat fit() selesai. Dipakai watchdog DAN self-resubmit supaya run yang sudah
# tuntas tidak dinyalakan ulang tanpa henti.
tlejepa_is_finished() {
    local label="$1"
    compgen -G "$TLEJEPA_CKPT_ROOT/*${label}*/events.log" > /dev/null || return 1
    grep -qh "FINISHED" "$TLEJEPA_CKPT_ROOT"/*"${label}"*/events.log 2>/dev/null
}

# Submit ulang skrip run yang sama. Dipanggil dari trap SIGUSR1 ketika time
# limit hampir tercapai. Dibatasi MAX_SELF_RESUBMIT supaya kegagalan permanen
# tidak jadi loop tak berujung; hitungannya disimpan di berkas, bukan di memori,
# karena tiap submit ulang adalah proses baru.
tlejepa_resubmit_self() {
    local label="$1" script="$2"
    local max="${MAX_SELF_RESUBMIT:-40}"
    local f="$TLEJEPA_ROOT/logs/resubmit_${label}.count"
    mkdir -p "$TLEJEPA_ROOT/logs"
    local c=0; [ -f "$f" ] && c=$(cat "$f")
    if [ "$c" -ge "$max" ]; then
        echo "[$label] sudah $c kali submit ulang; berhenti."
        return 1
    fi
    echo $((c + 1)) > "$f"
    local slot="A"
    case "$label" in *R2_*|*R3_*|*R5_*|*R6_*|*R9_*|*R11a_*) slot="B";; esac
    local gpu="$TLEJEPA_GPU_A"; [ "$slot" = "B" ] && gpu="$TLEJEPA_GPU_B"
    local jid
    jid=$(sbatch --parsable \
          --partition="$TLEJEPA_GPU_PARTITION" \
          --gres="gpu:${gpu}:1" \
          --time="$TLEJEPA_TIME_LIMIT" \
          --export=ALL,TLEJEPA_SEED="${TLEJEPA_SEED:-42}" "$script" 2>&1)
    if [[ "$jid" =~ ^[0-9]+$ ]]; then
        echo "[$label] submit ulang ke-$((c + 1)) berhasil, job=$jid"
    else
        echo "[$label] submit ulang GAGAL: $jid"
        echo "[$label] kemungkinan cluster melarang sbatch dari dalam job;"
        echo "[$label] watchdog akan menanganinya dari luar."
    fi
}

# Menjalankan train.py dengan --resume otomatis kalau checkpoint sudah ada.
# Watchdog memanggil ini; percobaan pertama tanpa --resume, berikutnya dengan.
tlejepa_run() {
    local label="$1"; shift
    local resume=""
    if compgen -G "$TLEJEPA_CKPT_ROOT/*${label}*/latest_step.pt" > /dev/null; then
        resume="--resume"
        echo "[run] Checkpoint ditemukan untuk $label -> melanjutkan."
    fi
    # TANPA `srun`. Job ini satu task dan satu GPU, jadi srun tidak menambah
    # apa pun selain satu lapisan yang justru menjadi sumber konflik
    # cpus-per-task di atas.
    #
    # Dijalankan di BACKGROUND lalu ditunggu, bukan di foreground. Bash menunda
    # penanganan trap selama sebuah perintah foreground berjalan, sehingga trap
    # SIGUSR1 baru dieksekusi setelah python selesai -- yaitu setelah SIGKILL
    # tiba, ketika sudah terlambat untuk menyimpan checkpoint maupun submit
    # ulang. Dengan `&` dan `wait`, trap dieksekusi seketika.
    python train.py $COMMON_ARGS --run-label "$label" \
           --seed "${TLEJEPA_SEED:-42}" $resume "$@" &
    TLEJEPA_TRAIN_PID=$!
    export TLEJEPA_TRAIN_PID

    # `wait` dikembalikan dengan kode >128 ketika tersela sinyal; selama proses
    # latih masih hidup, tunggu lagi.
    local rc=0
    while true; do
        wait "$TLEJEPA_TRAIN_PID"; rc=$?
        if [ "$rc" -gt 128 ] && kill -0 "$TLEJEPA_TRAIN_PID" 2>/dev/null; then
            continue
        fi
        break
    done
    return "$rc"
}
