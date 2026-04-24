# Ablation Models

This directory contains ablation variants of TSDGNet used to evaluate the contribution of the main components in the proposed framework.

Included scripts:

```text
train_tsdgnet_without_temporal.py
train_tsdgnet_without_graph.py
train_tsdgnet_without_imbalance_optimization.py
```

The three variants correspond to removing the temporal branch, removing the graph branch, and removing the class imbalance optimization strategy, respectively. They follow the same data format and evaluation protocol as the main TSDGNet training script.

Examples:

```bash
python ablations/train_tsdgnet_without_temporal.py \
  --npz_path data/processed/gait1_preprocessed.npz \
  --checkpoint_path checkpoints/best_tsdgnet_without_temporal.pth

python ablations/train_tsdgnet_without_graph.py \
  --npz_path data/processed/gait1_preprocessed.npz \
  --checkpoint_path checkpoints/best_tsdgnet_without_graph.pth

python ablations/train_tsdgnet_without_imbalance_optimization.py \
  --npz_path data/processed/gait1_preprocessed.npz \
  --checkpoint_path checkpoints/best_tsdgnet_without_imbalance_optimization.pth
```
