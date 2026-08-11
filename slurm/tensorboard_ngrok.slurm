#!/bin/bash
#SBATCH --job-name=tlejepa-tensorboard
#SBATCH --partition=gpu-long
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#
# Satu TensorBoard untuk SELURUH matriks, diekspos lewat ngrok.
#
# Semua run menulis ke $TLEJEPA_TB_ROOT yang sama, jadi tiap run muncul sebagai
# kurva berlabel run-id-nya dan bisa dibandingkan langsung. Di partisi CPU
# karena tidak butuh GPU; menaruhnya di partisi GPU berarti membuang satu kartu
# selama seminggu.
#
# Panel yang paling berguna, dikelompokkan per hipotesis:
#   H1_invariance/*     recall@1 dan MRR terhadap tingkat korupsi
#   H2_dual_use/*       peringkat token pada Z_c
#   H3_word_order/*     akurasi & margin parafrase vs permutasi
#   H5_H6_H8_rank/*     peringkat efektif encoder DAN decoder, terpisah
#   H7_negation/*       delta cosine premis-negasi
#   H9_length/*         over_frac harus merangkak ke tau_L = 0.9
#   train_step/*        grad_norm_ema, clip_frac, spike_skip_frac
#   epoch/*             train vs val berdampingan
#
# URL ngrok ditulis ke logs/tensorboard_url.txt karena hostname gratis berubah
# setiap job restart.

set -uo pipefail
source "$(dirname "$0")/_common.sh"
tlejepa_setup_env

TB_PORT="${TB_PORT:-$((6006 + RANDOM % 1000))}"
# 0.0.0.0, BUKAN 127.0.0.1. Dengan 127.0.0.1 TensorBoard hanya menerima koneksi
# dari dalam node itu sendiri, sehingga `ssh -L port:<node>:port user@login`
# selalu ditolak: login node membuka koneksi TCP biasa ke compute node, bukan
# loopback. ngrok tetap bekerja karena ia berjalan pada node yang sama.
TB_HOST="${TB_HOST:-0.0.0.0}"
NGROK_BIN="${NGROK_BIN:-ngrok}"

cleanup() {
    [ -n "${TB_PID:-}" ] && kill "$TB_PID" 2>/dev/null
    [ -n "${NGROK_PID:-}" ] && kill "$NGROK_PID" 2>/dev/null
}
trap cleanup EXIT

echo "[tb] logdir : $TLEJEPA_TB_ROOT"
echo "[tb] node   : ${SLURMD_NODENAME:-?}"
echo "[tb] port   : $TB_PORT"
echo "[tb] bind   : $TB_HOST"
echo "[tb] SSH forward dari mesin lokal:"
echo "[tb]   ssh -J <user>@stujump.comp.nus.edu.sg -L ${TB_PORT}:${SLURMD_NODENAME:-NODE}:${TB_PORT} <user>@xlogin.comp.nus.edu.sg"

# --reload_multifile wajib: beberapa run menulis serentak ke subfolder berbeda,
# dan tanpa flag ini TensorBoard hanya menyegarkan berkas terbaru per direktori
# sehingga run lain terlihat membeku.
tensorboard --logdir "$TLEJEPA_TB_ROOT" --port "$TB_PORT" --host "$TB_HOST" \
            --reload_multifile true --reload_interval 30 &
TB_PID=$!
sleep 20

if [ -z "${NGROK_AUTHTOKEN:-}" ]; then
    echo "[tb] NGROK_AUTHTOKEN kosong; tunnel dilewati."
    echo "[tb] Akses lewat SSH port-forward:"
    echo "[tb]   ssh -J <user>@stujump.comp.nus.edu.sg -L ${TB_PORT}:${SLURMD_NODENAME}:${TB_PORT} <user>@xlogin.comp.nus.edu.sg"
    wait "$TB_PID"; exit 0
fi

"$NGROK_BIN" config add-authtoken "$NGROK_AUTHTOKEN" >/dev/null 2>&1
"$NGROK_BIN" http "$TB_PORT" --log stdout > "logs/ngrok-$SLURM_JOB_ID.log" 2>&1 &
NGROK_PID=$!

PUBLIC_URL=""
for _ in $(seq 1 30); do
    sleep 2
    PUBLIC_URL=$(curl -s http://127.0.0.1:4040/api/tunnels | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    print(next((t['public_url'] for t in d.get('tunnels',[]) if t['public_url'].startswith('https')),''))
except Exception:
    print('')" 2>/dev/null)
    [ -n "$PUBLIC_URL" ] && break
done

if [ -n "$PUBLIC_URL" ]; then
    echo "[tb] URL publik: $PUBLIC_URL"
    echo "$PUBLIC_URL" > "$TLEJEPA_ROOT/logs/tensorboard_url.txt"
else
    echo "[tb] ngrok gagal; cek logs/ngrok-$SLURM_JOB_ID.log"
fi

wait "$TB_PID"
