#!/bin/bash
# Submit seluruh matriks + TensorBoard + watchdog.
#
#   bash slurm/submit_all.sh                    # semua, seed 42
#   bash slurm/submit_all.sh R1 R2              # subset
#   TLEJEPA_SEED=43 bash slurm/submit_all.sh R1 R2
#   SKIP_TB=1 SKIP_WD=1 bash slurm/submit_all.sh
#
# Watchdog menggantikan alerting email: ia memantau antrian dan menyalakan ulang
# run yang mati, memakai --resume sehingga melanjutkan dari langkah terakhir.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source slurm/_common.sh

ALL=(R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11a R11b)
RUNS=("$@"); [ ${#RUNS[@]} -eq 0 ] && RUNS=("${ALL[@]}")
export TLEJEPA_SEED="${TLEJEPA_SEED:-42}"
mkdir -p logs

if ! command -v sbatch >/dev/null; then
    echo "sbatch tidak ada. Jalankan di login node cluster."; exit 1
fi
echo "==================================================================="
echo " Matriks TLeJEPA | seed=$TLEJEPA_SEED | run: ${RUNS[*]}"
echo " TB   : $TLEJEPA_TB_ROOT"
echo " ckpt : $TLEJEPA_CKPT_ROOT"
echo "==================================================================="

if [ "${SKIP_TB:-0}" != "1" ]; then
    echo "  [TB ] job $(sbatch --parsable --partition="$TLEJEPA_CPU_PARTITION" \
        --time="$TLEJEPA_WD_TIME_LIMIT" slurm/tensorboard_ngrok.slurm)"
fi

# Tipe GPU diseling A dan B supaya beban tidak menumpuk pada satu tipe kartu.
declare -A GPU_OF=(
  [R1]=A [R2]=B [R3]=B [R4]=A [R5]=B [R6]=B
  [R7]=A [R8]=A [R9]=B [R10]=A [R11a]=B [R11b]=A
)

for r in "${RUNS[@]}"; do
    s="slurm/train_${r}.slurm"
    [ -f "$s" ] || { echo "  [!!] $s tidak ada"; continue; }
    slot="${GPU_OF[$r]:-A}"
    if [ "$slot" = "A" ]; then gpu="$TLEJEPA_GPU_A"; else gpu="$TLEJEPA_GPU_B"; fi
    jid=$(sbatch --parsable \
        --partition="$TLEJEPA_GPU_PARTITION" \
        --gres="gpu:${gpu}:1" \
        --time="$TLEJEPA_TIME_LIMIT" \
        --export=ALL,TLEJEPA_SEED="$TLEJEPA_SEED" "$s")
    echo "  [$r] job $jid  gpu=$gpu"
done

if [ "${SKIP_WD:-0}" != "1" ]; then
    export WATCH_RUNS="${RUNS[*]}"
    echo "  [WD ] job $(sbatch --parsable --partition="$TLEJEPA_CPU_PARTITION" \
        --time="$TLEJEPA_WD_TIME_LIMIT" \
        --export=ALL,WATCH_RUNS="${RUNS[*]}",TLEJEPA_SEED="$TLEJEPA_SEED" slurm/watchdog.slurm)"
fi

echo "-------------------------------------------------------------------"
echo "Pantau : squeue -u \$USER"
echo "URL TB : cat logs/tensorboard_url.txt"
echo "Event  : tail -f checkpoints/*/events.log"
echo "Retry  : cat logs/watchdog_state.tsv"
echo "==================================================================="
