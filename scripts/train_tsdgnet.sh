#!/usr/bin/env bash
set -e
DATA_PATH=${1:-data/gait1_preprocessed.npz}
mkdir -p checkpoints
python train_tsdgnet.py \
  --npz_path "$DATA_PATH" \
  --checkpoint_path checkpoints/best_tsdgnet.pt \
  --epochs 400 \
  --batch_size 64 \
  --seq_len 2048
