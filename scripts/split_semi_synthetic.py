#!/usr/bin/env python3
"""
为 semi_synthetic 段数据生成 train.txt / val.txt（及可选 test.txt）。
目录下每段为：{segment_id}_motion.npz、{segment_id}_clip_description.npy。
只保留同时存在 motion 与 clip 的 segment_id。
"""
import os
import argparse
import random
from os.path import join as pjoin


def main():
    parser = argparse.ArgumentParser(description="Split semi_synthetic segment dir into train/val/test .txt")
    parser.add_argument("--segment_dir", type=str, required=True,
                        help="段数据目录，内含 *_motion.npz 与 *_clip_description.npy")
    parser.add_argument("--clip_dir", type=str, default=None,
                        help="CLIP 特征目录（默认与 segment_dir 相同；可设为 v1 路径）")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.0, help="测试集比例，0 则不写 test.txt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    segment_dir = os.path.abspath(args.segment_dir)
    clip_dir = os.path.abspath(args.clip_dir) if args.clip_dir else segment_dir
    if not os.path.isdir(segment_dir):
        raise FileNotFoundError(f"segment_dir not found: {segment_dir}")

    ids = []
    for f in os.listdir(segment_dir):
        if f.endswith("_motion.npz"):
            seg_id = f[: -len("_motion.npz")]
            clip_path = pjoin(clip_dir, seg_id + "_clip_description.npy")
            if os.path.exists(clip_path):
                ids.append(seg_id)
    ids = sorted(ids)
    if not ids:
        raise RuntimeError(
            f"No segments with both *_motion.npz and *_clip_description.npy in {segment_dir}"
            + (f" (clip from {clip_dir})" if clip_dir != segment_dir else "")
        )

    total = args.train_ratio + args.val_ratio + args.test_ratio
    if total <= 0:
        total = 1.0
    tr = args.train_ratio / total
    va = args.val_ratio / total
    te = args.test_ratio / total
    random.seed(args.seed)
    random.shuffle(ids)
    n = len(ids)
    n_train = max(0, int(n * tr))
    n_val = max(0, int(n * va))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train - n_test

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]

    def write_split(name, id_list):
        path = pjoin(segment_dir, name)
        with open(path, "w") as out:
            out.write("\n".join(id_list) + ("\n" if id_list else ""))
        print(f"  {path}  ({len(id_list)} samples)")

    print(f"Total segments (motion+clip): {n}  ->  train {len(train_ids)}, val {len(val_ids)}, test {len(test_ids)}")
    write_split("train.txt", train_ids)
    write_split("val.txt", val_ids)
    if test_ids:
        write_split("test.txt", test_ids)
    print("Done.")


if __name__ == "__main__":
    main()
