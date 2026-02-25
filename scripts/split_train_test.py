#!/usr/bin/env python3
"""
根据数据集目录中的 motion（及可选 text）文件，生成 train / val / test 划分文件。
用法示例：
  python scripts/split_train_test.py --dataset_dir ./data/BEAT_v2
  python scripts/split_train_test.py --dataset_dir ./data/G1ML3D_v1 --train_ratio 0.8 --test_ratio 0.2 --no_val
"""
import os
import argparse
import random
from os.path import join as pjoin


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test .txt files")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="数据集根目录，将在此目录下生成 train.txt, val.txt, test.txt")
    parser.add_argument("--motion_dir", type=str, default="joints_npz",
                        help="运动数据子目录名（相对 dataset_dir），默认 joints_npz")
    parser.add_argument("--motion_ext", type=str, default=".npz",
                        help="运动文件扩展名，如 .npz 或 .npy")
    parser.add_argument("--text_dir", type=str, default="texts",
                        help="文本子目录名；若存在则只保留同时有 motion 和 text 的样本，默认 texts（nested 布局下不按此过滤）")
    parser.add_argument("--layout", type=str, default="auto", choices=("flat", "nested", "auto"),
                        help="flat=单层 motion_dir；nested=按子目录递归扫描（如 BEAT_v2 的 1/ 2/ ... 30/）；auto=先试 flat，不存在则用 nested")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--no_val", action="store_true",
                        help="不生成 val.txt，仅 train + test（train_ratio 与 test_ratio 归一化）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    assert args.train_ratio >= 0 and args.test_ratio >= 0 and args.val_ratio >= 0
    if args.no_val:
        total = args.train_ratio + args.test_ratio
        args.train_ratio /= total
        args.test_ratio /= total
        args.val_ratio = 0.0
    else:
        total = args.train_ratio + args.val_ratio + args.test_ratio
        args.train_ratio /= total
        args.val_ratio /= total
        args.test_ratio /= total

    data_root = os.path.abspath(args.dataset_dir)
    motion_dir = pjoin(data_root, args.motion_dir)
    text_dir = pjoin(data_root, args.text_dir)
    has_text_dir = os.path.isdir(text_dir)

    use_nested = False
    if args.layout == "nested":
        use_nested = True
    elif args.layout == "auto" and not os.path.isdir(motion_dir):
        use_nested = True
        print(f"Motion dir not found: {motion_dir}, using nested scan under {data_root}")

    ids_from_motion = set()
    if use_nested:
        # 递归扫描 dataset_dir 下所有 *motion_ext，id = 相对 data_root 的路径（不含扩展名），如 1/1_wayne_0_100_100
        for root, dirs, files in os.walk(data_root):
            rel_root = os.path.relpath(root, data_root)
            if rel_root == ".":
                rel_root = ""
            for f in files:
                if f.endswith(args.motion_ext):
                    base = f[: -len(args.motion_ext)]
                    id_ = pjoin(rel_root, base).replace("\\", "/") if rel_root else base
                    ids_from_motion.add(id_)
        # nested 布局下不按顶层 text_dir 过滤（文本常在各自子目录且命名可能不同）
        ids = sorted(ids_from_motion)
    else:
        if not os.path.isdir(motion_dir):
            raise FileNotFoundError(
                f"Motion directory not found: {motion_dir}\n"
                f"若数据在子目录中（如 BEAT_v2 的 1/ 2/ ...），请使用:  --layout nested"
            )
        for f in os.listdir(motion_dir):
            if f.endswith(args.motion_ext):
                ids_from_motion.add(f[: -len(args.motion_ext)])
        if has_text_dir:
            ids = [iid for iid in ids_from_motion if os.path.exists(pjoin(text_dir, iid + ".txt"))]
        else:
            ids = list(ids_from_motion)
        ids = sorted(ids)

    random.seed(args.seed)
    random.shuffle(ids)
    n = len(ids)
    if n == 0:
        raise RuntimeError(f"No valid samples found in {motion_dir}" + (
            f" (with matching {text_dir})" if has_text_dir else ""
        ))

    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train - n_test

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]

    def write_split(name, id_list):
        path = pjoin(data_root, name)
        with open(path, "w") as f:
            f.write("\n".join(id_list) + ("\n" if id_list else ""))
        print(f"  {path}  ({len(id_list)} samples)")

    print(f"Total samples: {n}  ->  train {len(train_ids)}, val {len(val_ids)}, test {len(test_ids)}")
    write_split("train.txt", train_ids)
    if not args.no_val and val_ids:
        write_split("val.txt", val_ids)
    write_split("test.txt", test_ids)
    print("Done.")


if __name__ == "__main__":
    main()
