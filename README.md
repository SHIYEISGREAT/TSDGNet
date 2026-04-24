 TSDGNet

PyTorch implementation of TSDGNet for multi-class lower limb disease gait recognition using wearable inertial measurement unit signals.

TSDGNet integrates multi-scale temporal feature extraction, dual-graph structural refinement, and class-balanced optimization for lower limb disease recognition from multi-node IMU gait signals.

 Repository Structure

text
TSDGNet/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ requirements.txt
├─ train_tsdgnet.py
├─ data/
│  └─ README.md
├─ baselines/
│  ├─ README.md
│  ├─ train_xgboost.py
│  ├─ train_resnet1d.py
│  ├─ train_tcn.py
│  ├─ train_cnn_lstm.py
│  └─ train_transformer_encoder.py
├─ advanced_models/
│  ├─ README.md
│  ├─ train_aicare_cnn_svm.py
│  ├─ train_cmafnet.py
│  ├─ train_conv1d_bigru.py
│  ├─ train_cp_dualbranch.py
│  ├─ train_gaitsegnet.py
│  ├─ train_osconv_dualpath.py
│  ├─ train_sgat.py
│  └─ train_wctnet.py
├─ ablations/
│  ├─ README.md
│  ├─ train_tsdgnet_graph_only.py
│  ├─ train_tsdgnet_temporal_only.py
│  └─ train_tsdgnet_no_imbalance.py
├─ scripts/
│  ├─ train_tsdgnet.sh
│  ├─ run_baselines.sh
│  ├─ run_advanced_models.sh
│  └─ run_ablations.sh
├─ checkpoints/
│  └─ README.md
└─ results/
   └─ README.md

Environment

Install dependencies with:

bash
pip install -r requirements.txt

A CUDA-enabled PyTorch installation is recommended for GPU training. Install the PyTorch build that matches your CUDA environment if the default installation does not match your system.

Data Format

The training scripts expect a preprocessed `.npz` file. The default file name used in examples is:

text
data/gait1_preprocessed.npz

Required arrays:

text
X:          shape = (N, T, 4, 6)
lengths:    shape = (N,)
pathology:  shape = (N,)
subject_id: shape = (N,)

Each sample is cropped or padded to `seq_len` and reshaped to `(24, seq_len)` before model input. Private clinical data, participant identifiers, or non-public raw data should not be uploaded to GitHub.

Train TSDGNet

bash
bash scripts/train_tsdgnet.sh data/gait1_preprocessed.npz

Equivalent direct command:

bash
python train_tsdgnet.py \
  --npz_path data/gait1_preprocessed.npz \
  --checkpoint_path checkpoints/best_tsdgnet.pt \
  --epochs 400 \
  --batch_size 64 \
  --seq_len 2048


Run Baseline Models

bash
bash scripts/run_baselines.sh data/gait1_preprocessed.npz

This runs XGBoost, ResNet1D, TCN, CNN-LSTM, and Transformer Encoder comparison models.

Run Advanced Comparison Models

bash
bash scripts/run_advanced_models.sh data/gait1_preprocessed.npz

This runs the advanced comparison models included in the repository.

Run Ablation Experiments

bash
bash scripts/run_ablations.sh data/gait1_preprocessed.npz

This runs the graph-only, temporal-only, and no-imbalance-optimization variants.

Main Evaluation Metrics

The scripts report the following metrics on the test set:

text
Accuracy
Macro-Precision
Macro-Recall
Macro-F1

Checkpoints and Results

Model weights are saved under `checkpoints/` when the provided shell scripts are used. Checkpoint files and generated results are ignored by Git by default. Large trained weights should be uploaded through GitHub Releases instead of being committed directly to the repository.

Citation

The manuscript information is intentionally omitted while the work is under review. After acceptance or publication, update `CITATION.cff` with the final author list, venue, DOI, and repository URL.

License

This repository uses the MIT License for code. Data are not covered by the code license and should follow the access and redistribution policy of the original dataset or clinical protocol.
