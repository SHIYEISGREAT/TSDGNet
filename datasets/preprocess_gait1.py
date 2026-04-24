import os
import json
import argparse
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.signal import butter, filtfilt
from tqdm import tqdm

def load_numeric_table(path: str) -> np.ndarray:
    lines_num = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            try:
                [float(x) for x in parts]
                lines_num.append(" ".join(parts) + "\n")
            except ValueError:
                continue
    if not lines_num:
        raise ValueError(f"No numeric data lines found in {path}")
    from io import StringIO
    arr = np.loadtxt(StringIO("".join(lines_num)))
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr

def discover_trials(data_root: str) -> List[Tuple[str, str, str]]:
    cohorts = ["healthy", "neuro", "ortho"]
    trials = []
    for cohort in cohorts:
        cohort_dir = os.path.join(data_root, cohort)
        if not os.path.isdir(cohort_dir):
            continue
        for root, _, files in os.walk(cohort_dir):
            for file_name in files:
                if file_name.endswith("_meta.json"):
                    trial_id = file_name.replace("_meta.json", "")
                    trials.append((cohort, root, trial_id))
    return trials

def bandpass_filter(data: np.ndarray, fs: float, low: float = 0.5, high: float = 20.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq
    b, a = butter(order, [low_n, high_n], btype="band")
    orig_shape = data.shape
    flat = data.reshape(-1, orig_shape[-1])
    for c in range(flat.shape[1]):
        flat[:, c] = filtfilt(b, a, flat[:, c])
    return flat.reshape(orig_shape)

def load_imus_for_trial(trial_dir: str, trial_id: str) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    meta_path = os.path.join(trial_dir, f"{trial_id}_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    fs = meta.get("freq") or meta.get("fs") or meta.get("sampling_frequency") or meta.get("sampling_rate") or 100.0
    sensors = ["HE", "LB", "LF", "RF"]
    sensor_data: List[np.ndarray] = []
    has_all_raw = True
    for sensor in sensors:
        raw_path = os.path.join(trial_dir, f"{trial_id}_raw_data_{sensor}.txt")
        alt_path = os.path.join(trial_dir, f"{trial_id}_{sensor}.txt")
        if os.path.isfile(raw_path):
            path = raw_path
        elif os.path.isfile(alt_path):
            path = alt_path
        else:
            has_all_raw = False
            break
        arr = load_numeric_table(path)
        if arr.shape[1] < 6:
            raise ValueError(f"{path} has fewer than 6 numeric columns: {arr.shape[1]}")
        sensor_data.append(arr[:, -6:])
    if has_all_raw:
        min_len = min(s.shape[0] for s in sensor_data)
        sensor_data = [s[:min_len] for s in sensor_data]
        imu_data = np.stack(sensor_data, axis=1).astype(np.float32)
        return imu_data, float(fs), meta
    proc_path = os.path.join(trial_dir, f"{trial_id}_processed_data.txt")
    if not os.path.isfile(proc_path):
        raise FileNotFoundError(f"Missing raw_data_* and processed_data for trial {trial_id} in {trial_dir}")
    arr = load_numeric_table(proc_path)
    ncol = arr.shape[1]
    usable = (ncol // 6) * 6
    if usable < 24:
        raise ValueError(f"{proc_path} has {ncol} numeric columns and cannot form 4x6 IMU channels")
    imu_flat = arr[:, ncol - usable:ncol]
    n_sensor = usable // 6
    if n_sensor > 4:
        imu_flat = imu_flat[:, :4 * 6]
        n_sensor = 4
    imu_data = imu_flat.reshape(arr.shape[0], n_sensor, 6)
    if n_sensor < 4:
        T = imu_data.shape[0]
        tmp = np.zeros((T, 4, 6), dtype=imu_data.dtype)
        tmp[:, :n_sensor, :] = imu_data
        imu_data = tmp
    return imu_data.astype(np.float32), float(fs), meta

def trim_and_resample(imu_data: np.ndarray, fs: float, trim_sec: float = 1.0, target_fs: float = 50.0) -> Tuple[np.ndarray, float]:
    T = imu_data.shape[0]
    trim_n = int(trim_sec * fs)
    start = trim_n
    end = max(T - trim_n, start + 10)
    imu_trim = imu_data[start:end]
    factor = max(int(round(fs / target_fs)), 1)
    imu_ds = imu_trim[::factor]
    new_fs = fs / factor
    return imu_ds, new_fs

def normalize_per_trial(imu_data: np.ndarray) -> np.ndarray:
    mean = imu_data.mean(axis=0, keepdims=True)
    std = imu_data.std(axis=0, keepdims=True) + 1e-6
    return (imu_data - mean) / std

def parse_labels_from_meta(meta: Dict[str, Any], cohort_folder: str) -> Tuple[str, str, str]:
    subject_id = str(meta.get("subject") or meta.get("subject_id") or meta.get("id") or "unknown")
    pathology = str(meta.get("pathologyKey") or meta.get("pathology") or cohort_folder)
    cohort = str(meta.get("group") or cohort_folder)
    return subject_id, pathology, cohort

def main(dataset_root: str, output_dir: str, output_name: str = "gait1_preprocessed.npz", trim_sec: float = 1.0, target_fs: float = 50.0) -> None:
    data_root = os.path.join(dataset_root, "data")
    if not os.path.isdir(data_root):
        raise RuntimeError(f"The data folder was not found under: {dataset_root}")
    trials = discover_trials(data_root)
    print(f"Discovered trials: {len(trials)}")
    all_signals: List[np.ndarray] = []
    all_lengths: List[int] = []
    all_cohort: List[str] = []
    all_pathology: List[str] = []
    all_subject: List[str] = []
    all_trial_id: List[str] = []
    all_meta: List[Dict[str, Any]] = []
    max_len = 0
    skipped = 0
    for cohort, trial_dir, trial_id in tqdm(trials, desc="Processing trials"):
        try:
            imu_raw, fs, meta = load_imus_for_trial(trial_dir, trial_id)
        except FileNotFoundError as e:
            print(f"[WARN] Skipping trial {trial_id}: {e}")
            skipped += 1
            continue
        imu_filt = bandpass_filter(imu_raw, fs=fs, low=0.5, high=20.0, order=4)
        imu_ds, _ = trim_and_resample(imu_filt, fs=fs, trim_sec=trim_sec, target_fs=target_fs)
        imu_norm = normalize_per_trial(imu_ds)
        Tprime = imu_norm.shape[0]
        max_len = max(max_len, Tprime)
        all_signals.append(imu_norm)
        all_lengths.append(Tprime)
        subject_id, pathology, cohort_label = parse_labels_from_meta(meta, cohort)
        all_subject.append(subject_id)
        all_pathology.append(pathology)
        all_cohort.append(cohort_label)
        all_trial_id.append(trial_id)
        all_meta.append(meta)
    N = len(all_signals)
    print(f"Processed trials: {N}, max_len: {max_len}, skipped: {skipped}")
    if N == 0:
        raise RuntimeError("No trials were successfully processed. Please check the dataset path and file names.")
    X = np.zeros((N, max_len, 4, 6), dtype=np.float32)
    lengths = np.array(all_lengths, dtype=np.int32)
    for i, sig in enumerate(all_signals):
        T = sig.shape[0]
        X[i, :T, :, :] = sig
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_name)
    np.savez_compressed(out_path, X=X, lengths=lengths, cohort=np.array(all_cohort), pathology=np.array(all_pathology), subject_id=np.array(all_subject), trial_id=np.array(all_trial_id), meta_raw=np.array(all_meta, dtype=object), target_fs=np.array(target_fs, dtype=np.float32))
    print(f"Saved preprocessed data to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the public clinical gait dataset into an NPZ file for TSDGNet.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory of the downloaded public gait dataset. This directory should contain the data folder.")
    parser.add_argument("--output_dir", type=str, default=os.path.join("data", "processed"), help="Directory used to save the generated NPZ file.")
    parser.add_argument("--output_name", type=str, default="gait1_preprocessed.npz", help="Name of the generated NPZ file.")
    parser.add_argument("--trim_sec", type=float, default=1.0, help="Seconds removed from both the beginning and the end of each trial.")
    parser.add_argument("--target_fs", type=float, default=50.0, help="Target sampling frequency in Hz.")
    args = parser.parse_args()
    main(args.dataset_root, args.output_dir, args.output_name, args.trim_sec, args.target_fs)
