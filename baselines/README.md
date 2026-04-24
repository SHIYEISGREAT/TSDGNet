# Baseline Models

This directory contains conventional baseline models used for comparison experiments.

Included scripts:

```text
train_xgboost.py
train_resnet1d.py
train_tcn.py
train_cnn_lstm.py
train_transformer_encoder.py
```

Each script follows the same preprocessed data format and evaluation protocol as the proposed TSDGNet model.

Example:

```bash
python baselines/train_resnet1d.py --npz_path data/gait1_preprocessed.npz --checkpoint_path checkpoints/best_resnet1d.pt
```
