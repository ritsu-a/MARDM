#!/usr/bin/env python3
"""
从 semi_synthetic_v1_segments 中仅保留「5秒片段内与动作时间重叠 > 1.5 秒」的 segment，
并复制到 segment_v2 文件夹。
剩余动作时长 = segment 时间范围与 motion 时间范围的重叠长度。
"""

import json
import os
import shutil
from pathlib import Path


def motion_overlap_in_segment(seg_start, seg_end, motion_start, motion_end):
    """计算 segment 时间窗 [seg_start, seg_end] 与动作时间 [motion_start, motion_end] 的重叠时长（秒）。"""
    overlap_start = max(seg_start, motion_start)
    overlap_end = min(seg_end, motion_end)
    return max(0.0, overlap_end - overlap_start)


def main():
    data_root = Path(__file__).resolve().parents[1] / "data"
    src_dir = data_root / "semi_synthetic_v1_segments"
    dst_dir = data_root / "semi_synthetic_v2_segments"
    min_overlap = 1.5  # 秒

    summary_path = src_dir / "summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data["segments"]
    kept = []
    for seg in segments:
        overlap = motion_overlap_in_segment(
            seg["segment_start_time"],
            seg["segment_end_time"],
            seg["motion_start_time"],
            seg["motion_end_time"],
        )
        if overlap > min_overlap:
            seg["motion_overlap_seconds"] = round(overlap, 4)
            kept.append(seg)

    # 每个 segment 对应的文件后缀
    suffixes = [
        "_metadata.json",
        "_audio.npy",
        "_clip_description.npy",
        "_clip_name.npy",
        "_clip_semantic.npy",
        "_motion.npz",
    ]

    os.makedirs(dst_dir, exist_ok=True)
    missing = []
    for seg in kept:
        name = seg["segment_name"]
        for suf in suffixes:
            src = src_dir / (name + suf)
            if not src.exists():
                missing.append(str(src))
                continue
            shutil.copy2(src, dst_dir / (name + suf))

    if missing:
        print("Warning: some files not found (skipped):")
        for m in missing[:20]:
            print(" ", m)
        if len(missing) > 20:
            print(" ... and", len(missing) - 20, "more")

    # 写入新的 summary.json（不保留 motion_overlap_seconds 也可，仅便于检查）
    out_summary = {
        "total_segments": len(kept),
        "min_motion_overlap_seconds": min_overlap,
        "segments": kept,
    }
    with open(dst_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(out_summary, f, ensure_ascii=False, indent=2)

    print(f"Filtered: {len(segments)} -> {len(kept)} segments (motion overlap > {min_overlap}s)")
    print(f"Saved to: {dst_dir}")


if __name__ == "__main__":
    main()
