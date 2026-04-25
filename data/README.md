# Data Directory

This directory is used for local data storage. The processed NPZ file is not included in this repository.

After downloading the original public gait dataset, generate the processed file with:

```bash
python datasets/preprocess_gait1.py \
  --dataset_root /path/to/downloaded/dataset \
  --output_dir data/processed
```

The expected generated file is:

```text
data/processed/gait1_preprocessed.npz
```

Do not commit raw datasets or processed NPZ files to this repository.
