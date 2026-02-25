#!/usr/bin/env python3
"""
先切片、再分 split：对全部样本切段后，再将片段按比例划分为 train/test。
1) 每段单独 npz：segment/{id}.npz
2) 合并大 npz：segment_train.npz / segment_test.npz（按 --train_ratio 划分片段）

用法:
  python scripts/beat_segment_to_npz.py --dataset_dir ./data/BEAT_v2 --id_list train.txt test.txt
  python scripts/beat_segment_to_npz.py --dataset_dir ./data/BEAT_v2 --train_ratio 0.8 --seed 42
"""
import os
import argparse
import random
import numpy as np
from os.path import join as pjoin
from tqdm import tqdm

WHISPER_PER_MOTION = 50
MOTION_PER_WHISPER = 60
BEAT_MOTION_SEGMENT_LEN = 300
BEAT_WHISPER_SEGMENT_LEN = 250


def _segment_id(original_id, seg_idx):
    """将 1/1_wayne_0_100_100 + seg_idx 转为文件名友好 id：1_1_wayne_0_100_100_seg000"""
    base = original_id.replace("/", "_")
    return f"{base}_seg{seg_idx:03d}"


def load_caption(data_root, name):
    path = pjoin(data_root, os.path.dirname(name), os.path.basename(name) + "_whisper_features.txt")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip().replace("\n", " ")


def run(data_root, id_list_files, segment_dir, motion_key="qpos", train_ratio=0.8, seed=42):
    data_root = os.path.abspath(data_root)
    segment_dir = os.path.abspath(segment_dir)
    os.makedirs(segment_dir, exist_ok=True)

    # 收集全部样本 id（不区分 train/test）
    all_ids = []
    for f in id_list_files:
        full = pjoin(data_root, f) if not os.path.isabs(f) else f
        if not os.path.exists(full):
            print(f"Skip missing: {full}")
            continue
        with open(full, "r") as fp:
            for line in fp:
                id_ = line.strip()
                if id_ and id_ not in all_ids:
                    all_ids.append(id_)

    # 先切片：得到全部片段，再不做划分
    motion_list = []
    whisper_list = []
    caption_list = []
    original_id_list = []
    seg_idx_list = []
    segment_id_list = []
    n_total = 0
    for name in tqdm(all_ids, desc="Segmenting"):
        motion_path = pjoin(data_root, name + ".npz")
        whisper_path = pjoin(data_root, os.path.dirname(name), os.path.basename(name) + "_whisper_features.npy")
        if not os.path.exists(motion_path) or not os.path.exists(whisper_path):
            continue
        motion_data = np.load(motion_path)
        motion = motion_data[motion_key] if motion_key in motion_data else motion_data[list(motion_data.keys())[0]]
        if len(motion.shape) == 1:
            motion = motion.reshape(-1, 1)
        whisper = np.load(whisper_path).astype(np.float32)
        T_m, dim_m = motion.shape
        T_w, dim_w = whisper.shape
        if T_m < BEAT_MOTION_SEGMENT_LEN:
            continue
        caption = load_caption(data_root, name)
        n_seg = T_m // BEAT_MOTION_SEGMENT_LEN
        for k in range(n_seg):
            start_m = k * BEAT_MOTION_SEGMENT_LEN
            end_m = start_m + BEAT_MOTION_SEGMENT_LEN
            start_w = int(start_m * WHISPER_PER_MOTION / MOTION_PER_WHISPER)
            end_w = int(end_m * WHISPER_PER_MOTION / MOTION_PER_WHISPER)
            if end_w > T_w:
                break
            motion_seg = motion[start_m:end_m].astype(np.float32)
            whisper_seg = whisper[start_w:end_w]
            if motion_seg.shape[0] != BEAT_MOTION_SEGMENT_LEN or whisper_seg.shape[0] != BEAT_WHISPER_SEGMENT_LEN:
                if whisper_seg.shape[0] < BEAT_WHISPER_SEGMENT_LEN:
                    pad = np.zeros((BEAT_WHISPER_SEGMENT_LEN - whisper_seg.shape[0], dim_w), dtype=np.float32)
                    whisper_seg = np.concatenate([whisper_seg, pad], axis=0)
                else:
                    whisper_seg = whisper_seg[:BEAT_WHISPER_SEGMENT_LEN].copy()
            whisper_seg = whisper_seg.astype(np.float32)
            seg_id = _segment_id(name, k)
            out_path = pjoin(segment_dir, seg_id + ".npz")
            np.savez(
                out_path,
                motion=motion_seg,
                whisper=whisper_seg,
                caption=np.array(caption, dtype=object),
                original_id=np.array(name, dtype=object),
                seg_idx=np.int32(k),
            )
            n_total += 1
            motion_list.append(motion_seg)
            whisper_list.append(whisper_seg)
            caption_list.append(caption)
            original_id_list.append(name)
            seg_idx_list.append(k)
            segment_id_list.append(seg_id)

    if n_total == 0:
        print("No segments produced.")
        return 0

    # 再分 split：对片段打乱后按比例划分 train / test
    indices = list(range(n_total))
    random.seed(seed)
    random.shuffle(indices)
    n_train = int(n_total * train_ratio)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    for split_name, idx in [("train", train_idx), ("test", test_idx)]:
        if not idx:
            continue
        ids_txt = [segment_id_list[i] for i in idx]
        out_txt = pjoin(segment_dir, f"segment_{split_name}.txt")
        with open(out_txt, "w") as f:
            f.write("\n".join(ids_txt) + "\n")
        print(f"Wrote {len(ids_txt)} segment ids to {out_txt}")

        merged_path = pjoin(segment_dir, f"segment_{split_name}.npz")
        np.savez(
            merged_path,
            motion=np.stack([motion_list[i] for i in idx], axis=0),
            whisper=np.stack([whisper_list[i] for i in idx], axis=0),
            caption=np.array([caption_list[i] for i in idx], dtype=object),
            original_id=np.array([original_id_list[i] for i in idx], dtype=object),
            seg_idx=np.array([seg_idx_list[i] for i in idx], dtype=np.int32),
            segment_id=np.array([segment_id_list[i] for i in idx], dtype=object),
        )
        print(f"Wrote merged {merged_path} (N={len(idx)})")

    print(f"Saved {n_total} segments under {segment_dir} (train {len(train_idx)}, test {len(test_idx)})")
    return n_total


def main():
    parser = argparse.ArgumentParser(description="BEAT 先切片、再分 split：切段后按比例划分 train/test")
    parser.add_argument("--dataset_dir", type=str, required=True, help="BEAT_v2 根目录")
    parser.add_argument("--segment_dir", type=str, default=None, help="输出目录，默认 dataset_dir/segment")
    parser.add_argument("--id_list", type=str, nargs="+", default=None,
                        help="样本 id 列表文件（可多个，合并去重），如 train.txt test.txt；默认 train.txt test.txt")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="片段中作为 train 的比例，其余为 test")
    parser.add_argument("--seed", type=int, default=42, help="划分前打乱片段的随机种子")
    parser.add_argument("--motion_key", type=str, default="qpos")
    args = parser.parse_args()

    segment_dir = args.segment_dir or pjoin(args.dataset_dir, "segment")
    id_list_files = args.id_list or ["train.txt", "test.txt"]
    run(args.dataset_dir, id_list_files, segment_dir, args.motion_key, args.train_ratio, args.seed)


if __name__ == "__main__":
    main()
