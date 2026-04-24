# Ablation Models

This directory contains ablation variants of TSDGNet.

Included scripts:

```text
train_tsdgnet_graph_only.py
train_tsdgnet_temporal_only.py
train_tsdgnet_no_imbalance.py
```

The variants are used to evaluate the contribution of structural graph learning, temporal modeling, and class imbalance optimization.

Example:

```bash
python ablations/train_tsdgnet_temporal_only.py --npz_path data/gait1_preprocessed.npz --checkpoint_path checkpoints/best_tsdgnet_temporal_only.pt
```
