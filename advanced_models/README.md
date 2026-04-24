# Advanced Comparison Models

This directory contains advanced comparison models adapted to the same IMU input format and evaluation protocol as TSDGNet.

Included scripts:

```text
train_aicare_cnn_svm.py
train_cmafnet.py
train_conv1d_bigru.py
train_cp_dualbranch.py
train_gaitsegnet.py
train_osconv_dualpath.py
train_sgat.py
train_wctnet.py
```

Example:

```bash
python advanced_models/train_cmafnet.py --npz_path data/gait1_preprocessed.npz --checkpoint_path checkpoints/cmafnet.pt
```
