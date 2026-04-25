#!/usr/bin/env bash
set -e

DATA_PATH=${1:-data/processed/gait1_preprocessed.npz}
mkdir -p checkpoints

python baselines/train_xgboost.py --npz_path "$DATA_PATH" --model_path checkpoints/best_xgboost.json
python baselines/train_resnet1d.py --npz_path "$DATA_PATH" --checkpoint_path checkpoints/best_resnet1d.pt
python baselines/train_tcn.py --npz_path "$DATA_PATH" --checkpoint_path checkpoints/best_tcn.pt
python baselines/train_cnn_lstm.py --npz_path "$DATA_PATH" --checkpoint_path checkpoints/best_cnn_lstm.pt
python baselines/train_transformer_encoder.py --npz_path "$DATA_PATH" --checkpoint_path checkpoints/best_transformer_encoder.pt
