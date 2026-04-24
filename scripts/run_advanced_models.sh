#!/usr/bin/env bash
set -e
DATA_PATH=${1:-data/gait1_preprocessed.npz}
mkdir -p checkpoints
python advanced_models/train_aicare_cnn_svm.py --npz_path "$DATA_PATH" --checkpoint_path checkpoints/aicare_cnn_svm.pt
python advanced_models/train_cmafnet.py --npz_path "$DATA_PATH" --checkpoint_path checkpoints/cmafnet.pt
python advanced_models/train_conv1d_bigru.py --npz_path "$DATA_PATH" --checkpoint_path checkpoints/conv1d_bigru.pt
python advanced_models/train_cp_dualbranch.py --npz_path "$DATA_PATH" --checkpoint_path checkpoints/cp_dualbranch.pt
python advanced_models/train_gaitsegnet.py --npz_path "$DATA_PATH" --checkpoint_path checkpoints/gaitsegnet.pt
python advanced_models/train_osconv_dualpath.py --npz_path "$DATA_PATH" --checkpoint_path checkpoints/osconv_dualpath.pt
python advanced_models/train_sgat.py --npz_path "$DATA_PATH" --checkpoint_path checkpoints/sgat.pt
python advanced_models/train_wctnet.py --npz_path "$DATA_PATH" --checkpoint_path checkpoints/wctnet.pt
