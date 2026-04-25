#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
from scipy.signal import butter, filtfilt
from tqdm import tqdm

# ======= 你的数据集根目录（下面要能看到 data/ quick_start/） =======
DATASET_ROOT = r"D:\硕士项目\下肢机器人\公开数据集\A Dataset of Clinical Gait Signals with Wearable Sensors from Healthy, Neurological, and Orthopedic Cohorts\dataset\dataset"


# ----------------------------------------------------------------------
# 公共：只保留“全是数字”的行，再用 np.loadtxt 读取
# ----------------------------------------------------------------------
def load_numeric_table(path: str) -> np.ndarray:
    """
    从 txt 文件中读取纯数值表格：
      - 自动跳过表头（如包含 PacketCounter 等字符串）
      - 自动跳过空行和注释行

    返回二维数组 (T, D)
    """
    lines_num = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            # 注释行
            if stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            try:
                [float(x) for x in parts]
                lines_num.append(" ".join(parts) + "\n")
            except ValueError:
                # 有非数字内容，当成表头/说明，丢弃
                continue

    if not lines_num:
        raise ValueError(f"No numeric data lines found in {path}")

    from io import StringIO
    buf = StringIO("".join(lines_num))
    arr = np.loadtxt(buf)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


# ----------------------------------------------------------------------
# 发现所有 trial（根据 *_meta.json）
# ----------------------------------------------------------------------
def discover_trials(data_root: str) -> List[Tuple[str, str, str]]:
    """
    遍历 data/healthy, data/neuro, data/ortho，找到所有 trial 的 meta.json

    返回列表元素 (cohort_folder, trial_dir, trial_id)：
        cohort_folder : 'healthy' / 'neuro' / 'ortho' （按文件夹名）
        trial_dir     : 该 trial 所在文件夹
        trial_id      : 比如 'HS_2_2'（从 *_meta.json 去掉后缀）
    """
    cohorts = ["healthy", "neuro", "ortho"]
    trials = []

    for cohort in cohorts:
        cohort_dir = os.path.join(data_root, cohort)
        if not os.path.isdir(cohort_dir):
            continue

        for root, _, files in os.walk(cohort_dir):
            for f in files:
                if f.endswith("_meta.json"):
                    trial_id = f.replace("_meta.json", "")
                    trials.append((cohort, root, trial_id))

    return trials


# ----------------------------------------------------------------------
# 带通滤波：0.5–20 Hz
# ----------------------------------------------------------------------
def bandpass_filter(
    data: np.ndarray,
    fs: float,
    low: float = 0.5,
    high: float = 20.0,
    order: int = 4,
) -> np.ndarray:
    """
    对最后一维的每个通道做带通滤波。
    data: (..., C)
    """
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq
    b, a = butter(order, [low_n, high_n], btype="band")

    orig_shape = data.shape
    flat = data.reshape(-1, orig_shape[-1])

    for c in range(flat.shape[1]):
        flat[:, c] = filtfilt(b, a, flat[:, c])

    return flat.reshape(orig_shape)


# ----------------------------------------------------------------------
# 读取 4 个 raw_data_* 或退回 processed_data
# ----------------------------------------------------------------------
def load_imus_for_trial(
    trial_dir: str,
    trial_id: str,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    优先使用 4 个 raw_data_*_HE/LB/LF/RF.txt，
    若缺少则退回 <trial_id>_processed_data.txt。

    返回:
        imu_data : (T, 4, 6)  [时间, 传感器, 通道]
        fs       : 采样频率
        meta     : meta.json 字典
    """
    # ---- meta ----
    meta_path = os.path.join(trial_dir, f"{trial_id}_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 你发的 json 里采样率字段是 "freq"
    fs = (
        meta.get("freq")
        or meta.get("fs")
        or meta.get("sampling_frequency")
        or meta.get("sampling_rate")
        or 100.0
    )

    sensors = ["HE", "LB", "LF", "RF"]
    sensor_data: List[np.ndarray] = []

    # ---- 优先读 raw_data_* ----
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

        # 这里用 load_numeric_table 跳过 PacketCounter 等表头
        arr = load_numeric_table(path)

        if arr.shape[1] < 6:
            raise ValueError(f"{path} has < 6 numeric columns, got {arr.shape[1]}.")

        # 默认最后 6 列是 [ax, ay, az, gx, gy, gz]
        imu = arr[:, -6:]
        sensor_data.append(imu)

    if has_all_raw:
        # 对齐 4 个传感器长度
        min_len = min(s.shape[0] for s in sensor_data)
        sensor_data = [s[:min_len] for s in sensor_data]  # 每个 (T, 6)

        imu_data = np.stack(sensor_data, axis=1).astype(np.float32)  # (T, 4, 6)
        return imu_data, float(fs), meta

    # ---- raw_data 不全，退回 processed_data ----
    proc_path = os.path.join(trial_dir, f"{trial_id}_processed_data.txt")
    if not os.path.isfile(proc_path):
        raise FileNotFoundError(
            f"Missing raw_data_* and processed_data for trial {trial_id} in {trial_dir}"
        )

    arr = load_numeric_table(proc_path)

    ncol = arr.shape[1]
    usable = (ncol // 6) * 6
    if usable < 24:
        raise ValueError(
            f"{proc_path} has {ncol} numeric columns, cannot even form 4x6 IMU channels."
        )

    imu_flat = arr[:, ncol - usable : ncol]  # (T, usable)
    n_sensor = usable // 6

    if n_sensor > 4:
        imu_flat = imu_flat[:, : 4 * 6]
        n_sensor = 4

    imu_data = imu_flat.reshape(arr.shape[0], n_sensor, 6)

    if n_sensor < 4:
        T = imu_data.shape[0]
        tmp = np.zeros((T, 4, 6), dtype=imu_data.dtype)
        tmp[:, :n_sensor, :] = imu_data
        imu_data = tmp

    imu_data = imu_data.astype(np.float32)
    return imu_data, float(fs), meta


# ----------------------------------------------------------------------
# 去掉首尾静止段 + 简单下采样
# ----------------------------------------------------------------------
def trim_and_resample(
    imu_data: np.ndarray,
    fs: float,
    trim_sec: float = 1.0,
    target_fs: float = 50.0,
) -> Tuple[np.ndarray, float]:
    """
    去掉首尾 trim_sec 秒，并整数下采样到 target_fs

    imu_data: (T, 4, 6)
    """
    T = imu_data.shape[0]
    trim_n = int(trim_sec * fs)

    start = trim_n
    end = max(T - trim_n, start + 10)  # 至少保留一点数据
    imu_trim = imu_data[start:end]

    factor = max(int(round(fs / target_fs)), 1)
    imu_ds = imu_trim[::factor]
    new_fs = fs / factor

    return imu_ds, new_fs


# ----------------------------------------------------------------------
# trial 内 z-score 标准化
# ----------------------------------------------------------------------
def normalize_per_trial(imu_data: np.ndarray) -> np.ndarray:
    """
    imu_data: (T, 4, 6)
    """
    mean = imu_data.mean(axis=0, keepdims=True)
    std = imu_data.std(axis=0, keepdims=True) + 1e-6
    return (imu_data - mean) / std


# ----------------------------------------------------------------------
# 从 meta 里解析标签
# ----------------------------------------------------------------------
def parse_labels_from_meta(
    meta: Dict[str, Any],
    cohort_folder: str,
) -> Tuple[str, str, str]:
    """
    subject_id : meta['subject']  (例如 'HS_2')
    pathology  : meta['pathologyKey'] (例如 'HS','PD','CVA' 等)
    cohort     : meta['group'] （'healthy','neuro','ortho'），若不存在用文件夹名顶上
    """
    subject_id = str(
        meta.get("subject")
        or meta.get("subject_id")
        or meta.get("id")
        or "unknown"
    )

    pathology = str(
        meta.get("pathologyKey")
        or meta.get("pathology")
        or cohort_folder
    )

    cohort = str(meta.get("group") or cohort_folder)

    return subject_id, pathology, cohort


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main(
    dataset_root: str,
    output_dir: str,
    trim_sec: float = 1.0,
    target_fs: float = 50.0,
) -> None:
    data_root = os.path.join(dataset_root, "data")
    if not os.path.isdir(data_root):
        raise RuntimeError(f"'data' folder not found under: {dataset_root}")

    trials = discover_trials(data_root)
    print("发现 trial 数量：", len(trials))

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
            print(f"[WARN] Skip trial {trial_id}: {e}")
            skipped += 1
            continue

        # 0.5–20 Hz 带通滤波
        imu_filt = bandpass_filter(imu_raw, fs=fs, low=0.5, high=20.0, order=4)

        # 去掉首尾静止 + 下采样
        imu_ds, new_fs = trim_and_resample(
            imu_filt, fs=fs, trim_sec=trim_sec, target_fs=target_fs
        )

        # trial 内标准化
        imu_norm = normalize_per_trial(imu_ds)  # (T', 4, 6)

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
    print(f"统计完成，N={N}, max_len={max_len}, skipped={skipped}")

    if N == 0:
        raise RuntimeError("没有成功处理任何 trial，请检查文件命名或路径。")

    # padding 到统一长度： (N, max_len, 4, 6)
    X = np.zeros((N, max_len, 4, 6), dtype=np.float32)
    lengths = np.array(all_lengths, dtype=np.int32)

    for i, sig in enumerate(all_signals):
        T = sig.shape[0]
        X[i, :T, :, :] = sig

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "gait1_preprocessed.npz")

    np.savez_compressed(
        out_path,
        X=X,
        lengths=lengths,
        cohort=np.array(all_cohort),
        pathology=np.array(all_pathology),
        subject_id=np.array(all_subject),
        trial_id=np.array(all_trial_id),
        meta_raw=np.array(all_meta, dtype=object),
        target_fs=np.array(target_fs, dtype=np.float32),
    )

    print("已保存预处理结果到：", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess dataset_gait_1 into deep-learning ready npz."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=DATASET_ROOT,
        help="包含 data/ quick_start/ 的根目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="预处理结果保存目录",
    )
    parser.add_argument(
        "--trim_sec",
        type=float,
        default=1.0,
        help="裁掉首尾静止时长（秒）",
    )
    parser.add_argument(
        "--target_fs",
        type=float,
        default=50.0,
        help="目标采样率（Hz），简单整数下采样",
    )

    args = parser.parse_args()
    main(args.dataset_root, args.output_dir, args.trim_sec, args.target_fs)
