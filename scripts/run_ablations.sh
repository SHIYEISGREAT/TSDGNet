#!/usr/bin/env bash
set -e

DATA_PATH=${1:-data/processed/gait1_preprocessed.npz}
mkdir -p checkpoints

python ablations/train_tsdgnet_without_temporal.py --npz_path "$DATA_PATH" --checkpoint_path checkpoints/best_tsdgnet_without_temporal.pt
python ablations/train_tsdgnet_without_graph.py --npz_path "$DATA_PATH" --checkpoint_path checkpoints/best_tsdgnet_without_graph.pt
python ablations/train_tsdgnet_without_imbalance_optimization.py --npz_path "$DATA_PATH" --checkpoint_path checkpoints/best_tsdgnet_without_imbalance_optimization.pt
