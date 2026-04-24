#!/usr/bin/env bash
set -e
DATA_PATH=${1:-data/gait1_preprocessed.npz}
mkdir -p checkpoints
python ablations/train_tsdgnet_graph_only.py --npz_path "$DATA_PATH" --checkpoint_path checkpoints/best_tsdgnet_graph_only.pt
python ablations/train_tsdgnet_temporal_only.py --npz_path "$DATA_PATH" --checkpoint_path checkpoints/best_tsdgnet_temporal_only.pt
python ablations/train_tsdgnet_no_imbalance.py --npz_path "$DATA_PATH" --checkpoint_path checkpoints/best_tsdgnet_no_imbalance.pt
