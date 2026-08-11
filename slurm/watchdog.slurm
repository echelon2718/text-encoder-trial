#!/bin/bash
#SBATCH --job-name=tlejepa-watchdog
#SBATCH --partition=gpu-long
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
#
# Watchdog: memantau seluruh run dan menyalakan ulang yang mati.
#
# Menggantikan mekanisme email. Alasannya sederhana: memberitahu manusia bahwa
# job mati tidak memulihkan apa pun, dan pada run berhari-hari jeda sampai
# seseorang membaca email itu sendiri sudah mahal. Yang dibutuhkan adalah
# pemulihan tanpa manusia di tengah jalur.
#
# Cara kerjanya:
#   1. Setiap SCAN_INTERVAL, periksa antrian SLURM.
#   2. Run yang TIDAK ada di antrian tetapi belum menulis penanda FINISHED
#      dianggap mati, lalu disubmit ulang.
#   3. Skrip run memanggil tlejepa_run yang otomatis menambahkan --resume kalau
#      latest_step.pt sudah ada, sehingga melanjutkan dari langkah terakhir.
#   4. Percobaan ulang dibatasi MAX_RETRY per run supaya kegagalan permanen
#      (misalnya salah argumen) tidak jadi loop tak berujung.

#   5. Kalau cluster MELARANG submit dari dalam job, skrip ini juga bisa
#      dijalankan langsung di login node tanpa SLURM sama sekali:
#          nohup bash slurm/watchdog.slurm > logs/watchdog.log 2>&1 &
#      Tidak ada perintah di sini yang butuh hak admin.

set -uo pipefail
cd "$(dirname "$0")/.." 2>/dev/null || true
source "$(dirname "$0")/_common.sh" 2>/dev/null || source slurm/_common.sh
tlejepa_setup_env

SCAN_INTERVAL="${SCAN_INTERVAL:-300}"
MAX_RETRY="${MAX_RETRY:-20}"
STATE="$TLEJEPA_ROOT/logs/watchdog_state.tsv"
RUNS=(${WATCH_RUNS:-R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11a R11b})

touch "$STATE"

retry_count() { awk -v r="$1" '$1==r {print $2}' "$STATE" | tail -1; }

set_retry() {
    local r="$1" c="$2"
    grep -v -P "^$r\t" "$STATE" > "$STATE.tmp" 2>/dev/null || true
    printf "%s\t%s\t%s\n" "$r" "$c" "$(date +%FT%T)" >> "$STATE.tmp"
    mv "$STATE.tmp" "$STATE"
}

is_running() {
    squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qx "tlejepa-$1"
}

is_finished() { tlejepa_is_finished "$1"; }

echo "[watchdog] memantau: ${RUNS[*]}"
echo "[watchdog] interval ${SCAN_INTERVAL}s, maksimum ${MAX_RETRY} percobaan ulang per run"

while true; do
    alive=0
    for r in "${RUNS[@]}"; do
        script="slurm/train_${r}.slurm"
        [ -f "$script" ] || continue

        if is_running "$r"; then
            alive=$((alive + 1))
            continue
        fi
        if is_finished "$r"; then
            continue
        fi

        c=$(retry_count "$r"); c=${c:-0}
        if [ "$c" -ge "$MAX_RETRY" ]; then
            echo "[watchdog] $r sudah $c kali gagal; TIDAK disubmit ulang lagi."
            continue
        fi

        c=$((c + 1))
        set_retry "$r" "$c"
        slot="A"
        case "$r" in R2|R3|R5|R6|R9|R11a) slot="B";; esac
        if [ "$slot" = "A" ]; then gpu="$TLEJEPA_GPU_A"; else gpu="$TLEJEPA_GPU_B"; fi
        jid=$(sbatch --parsable \
              --partition="$TLEJEPA_GPU_PARTITION" \
              --gres="gpu:${gpu}:1" \
              --time="$TLEJEPA_TIME_LIMIT" \
              --export=ALL,TLEJEPA_SEED="${TLEJEPA_SEED:-42}" "$script" 2>&1)
        echo "[watchdog] $(date +%FT%T) $r mati -> submit ulang (percobaan $c) job=$jid"
        alive=$((alive + 1))
        sleep 5
    done

    if [ "$alive" -eq 0 ]; then
        echo "[watchdog] tidak ada run aktif dan semua selesai atau menyerah. Keluar."
        break
    fi
    sleep "$SCAN_INTERVAL"
done
