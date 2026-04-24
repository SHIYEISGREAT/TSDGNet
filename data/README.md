# Data Preparation

Place the preprocessed `.npz` gait data file in this directory.

Default example path:

```text
data/gait1_preprocessed.npz
```

Required keys:

```text
X
lengths
pathology
subject_id
```

Expected shapes:

```text
X:          (N, T, 4, 6)
lengths:    (N,)
pathology:  (N,)
subject_id: (N,)
```

`X` stores the gait sequence. The four nodes correspond to the wearable IMU locations used in the experiment, and each node contains six inertial channels. During training, each sample is converted to `(24, seq_len)` after center cropping or zero padding.

Do not upload private clinical data, personal identifiers, or non-public raw data to GitHub. If the dataset cannot be redistributed, provide only the data format description, preprocessing procedure, and access instructions.
