# Dataset Preprocessing

This directory contains the preprocessing script used to generate the input NPZ file required by the training scripts.

The preprocessed data file is not included in this repository. Please download the original public gait dataset from its official source and run:

```bash
python datasets/preprocess_gait1.py \
  --dataset_root /path/to/downloaded/dataset \
  --output_dir data/processed
```

This will generate:

```text
data/processed/gait1_preprocessed.npz
```

The generated NPZ file should contain the following keys:

```text
X
lengths
pathology
subject_id
cohort
trial_id
target_fs
```

The main training script and all comparison scripts use `X`, `lengths`, `pathology`, and `subject_id` as the required fields.

The expected shape of `X` is:

```text
(N, T, 4, 6)
```

where `N` is the number of gait trials, `T` is the padded sequence length, `4` denotes the four IMU sensor locations, and `6` denotes the inertial channels of each sensor.
