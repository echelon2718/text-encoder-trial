#!/bin/bash
#SBATCH --job-name=tlejepa-R11a
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#
# R11a: LeJEPA polos, ringan (H3)
#
# --requeue + --signal=B:USR1@300 : SLURM memberi jendela 300 detik sebelum
# SIGKILL sehingga Trainer sempat menyimpan latest_step.pt, lalu job dijadwalkan
# ulang otomatis. Bersama --resume di tlejepa_run, run melanjutkan dari langkah
# terakhir DAN dari posisi batch di tengah epoch, bukan mengulang epoch.
#
# GPU diseling antara a100-80 dan h100-96 agar beban tidak menumpuk pada satu
# tipe kartu; lihat header masing-masing skrip run.

set -uo pipefail
# source "$(dirname "$0")/_common.sh"
# tlejepa_setup_env

export TLEJEPA_ROOT="${SLURM_SUBMIT_DIR:-$HOME/tlejepa-revised}"
source "$TLEJEPA_ROOT/slurm/_common.sh"
tlejepa_setup_env

# Time limit gpu-long adalah 3 hari, sedangkan matriks ini butuh lebih lama.
# SIGUSR1 tiba 300 detik sebelum SIGKILL; Trainer memakai jendela itu untuk
# menulis latest_step.pt, lalu skrip ini men-submit dirinya sendiri kembali.
# tlejepa_run otomatis menambahkan --resume karena checkpoint sudah ada.
#
# Jalur ini butuh izin sbatch dari dalam job. Kalau cluster melarangnya,
# watchdog menangani hal yang sama dari luar -- lihat catatan di
# slurm/watchdog.slurm. Kedua jalur aman dijalankan bersamaan: watchdog hanya
# men-submit run yang TIDAK ada di antrian.
_resubmit_on_timeout() {
    echo "[$SLURM_JOB_NAME] SIGUSR1 diterima pada $(date +%FT%T)."
    # Teruskan sinyal ke proses latih: handler di Trainer yang menulis
    # latest_step.pt. Tanpa penerusan ini, python tidak pernah tahu job akan
    # dihentikan karena --signal=B:USR1 hanya mengirim ke skrip batch.
    if [ -n "${TLEJEPA_TRAIN_PID:-}" ] && kill -0 "$TLEJEPA_TRAIN_PID" 2>/dev/null; then
        kill -USR1 "$TLEJEPA_TRAIN_PID" 2>/dev/null
    fi
    sleep 120
    if tlejepa_is_finished "$1"; then
        echo "[$SLURM_JOB_NAME] sudah FINISHED, tidak disubmit ulang."
        return
    fi
    tlejepa_resubmit_self "$1" "$0"
}
trap '_resubmit_on_timeout R11a_enc8' USR1

tlejepa_run R11a_enc8 $SIZE_ENC8 --canon-type phoneme --w-canon 8.0
