#!/bin/bash
#SBATCH --job-name=tlejepa-kevin-full-revised-2048
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/%j.out

export MALLOC_ARENA_MAX=2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate avalonai

cd ~/text-encoder-trial

python train_with_lr_sched_continue.py \
  --dataset-mode huggingface \
  --batch-size 28 \
  --num-workers 4 \
  --n_singlish 3 \
  --n_premise_and_negation 3 \
  --d-model 512 \
  --heads 8 \
  --enc-layers 8 \
  --dec-layers 6 \
  --max-length 2048 \
  --epochs 1 \
  --zeta1 1.0 \
  --zeta2 1.0 \
  --run-label base_full_revised_2048 \
