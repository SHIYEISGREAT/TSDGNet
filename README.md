# TSDGNet

PyTorch implementation of **TSDGNet** for multi-class lower limb disease gait recognition using wearable inertial measurement unit (IMU) signals.

TSDGNet integrates multi-scale temporal feature extraction, dual-graph structural refinement, and class-balanced optimization for lower limb disease recognition from multi-node IMU gait signals.

The manuscript information is intentionally omitted while the work is under review. Formal citation details will be updated after publication.

## Repository Structure

```text
TSDGNet/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ requirements.txt
├─ .gitignore
├─ train_tsdgnet.py
├─ datasets/
│  └─ preprocess_gait1.py
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
│  ├─ train_tsdgnet_without_temporal.py
│  ├─ train_tsdgnet_without_graph.py
│  └─ train_tsdgnet_without_imbalance_optimization.py
├─ scripts/
│  ├─ train_tsdgnet.sh
│  ├─ run_baselines.sh
│  ├─ run_advanced_models.sh
│  └─ run_ablations.sh
├─ checkpoints/
│  └─ README.md
└─ results/
   └─ README.md
```

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

A CUDA-enabled PyTorch installation is recommended for GPU training. Install the PyTorch build that matches your CUDA environment if the default installation does not match your system.

## Data Preparation

The preprocessed `.npz` file is not included in this repository. Please download the original public gait dataset from its official source and generate the processed file locally.

Place the downloaded dataset in a local directory, then run:

```bash
python datasets/preprocess_gait1.py \
  --dataset_root /path/to/downloaded/dataset \
  --output_dir data/processed
```

This command generates:

```text
data/processed/gait1_preprocessed.npz
```

The generated file should contain the following arrays:

```text
X:          shape = (N, T, 4, 6)
lengths:    shape = (N,)
pathology:  shape = (N,)
subject_id: shape = (N,)
```

Each sample contains four IMU sensor nodes and six inertial channels per node. During training, each sample is cropped or padded to `seq_len` and reshaped to `(24, seq_len)` before model input.

Do not upload `.npz` data files, clinical records, participant identifiers, trained weights, or generated result files directly to the repository.

## Train TSDGNet

Using the shell script:

```bash
bash scripts/train_tsdgnet.sh data/processed/gait1_preprocessed.npz
```

Equivalent direct command:

```bash
python train_tsdgnet.py \
  --npz_path data/processed/gait1_preprocessed.npz \
  --checkpoint_path checkpoints/best_tsdgnet.pt \
  --epochs 400 \
  --batch_size 64 \
  --seq_len 2048
```

## Run Baseline Models

```bash
bash scripts/run_baselines.sh data/processed/gait1_preprocessed.npz
```

This runs XGBoost, ResNet1D, TCN, CNN-LSTM, and Transformer Encoder comparison models.

## Run Advanced Comparison Models

```bash
bash scripts/run_advanced_models.sh data/processed/gait1_preprocessed.npz
```

This runs the advanced comparison models included in the repository, including CMAFNet, Conv1D-BiGRU, CP Dual-Branch, GaitSegNet, OSConv Dual-Path, SGAT, WCT-Net, and AiCare CNN-SVM.

## Run Ablation Experiments

```bash
bash scripts/run_ablations.sh data/processed/gait1_preprocessed.npz
```

The ablation scripts correspond to the following settings:

```text
train_tsdgnet_without_temporal.py                  Remove the temporal classification branch
train_tsdgnet_without_graph.py                     Remove the graph structural refinement branch
train_tsdgnet_without_imbalance_optimization.py    Remove class imbalance optimization
```

## Main Evaluation Metrics

The scripts report the following test metrics:

```text
Accuracy
Macro-Precision
Macro-Recall
Macro-F1
```

## Checkpoints and Results

Model weights are saved under `checkpoints/` when the provided shell scripts are used. Checkpoint files and generated results are ignored by Git by default. Large trained weights should be uploaded through GitHub Releases instead of being committed directly to the repository.

## Citation

If you use this code, please cite this repository. The formal manuscript citation will be added after acceptance or publication.

The `CITATION.cff` file currently provides repository-level citation metadata only. Update it after publication with the final author list, venue, DOI, and repository URL.

## License

This repository uses the MIT License for code. Data are not covered by the code license and should follow the access and redistribution policy of the original public dataset.
