#!/usr/bin/env python3
"""
预处理semi_synthetic_v1数据集：
1. 将audio feature和motion npz切分成5秒片段（audio 250帧，motion 300帧）
2. 确保后4秒包含一个手势动作
3. 从motion_stat读取动作语义和描述，使用CLIP提取特征并保存
"""

import os
import json
import numpy as np
import torch
import clip
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple, Optional

# 帧率设置
AUDIO_FPS = 50  # audio feature帧率
MOTION_FPS = 60  # motion帧率
SEGMENT_DURATION = 5.0  # 5秒片段
AUDIO_FRAMES_PER_SEGMENT = int(SEGMENT_DURATION * AUDIO_FPS)  # 250帧
MOTION_FRAMES_PER_SEGMENT = int(SEGMENT_DURATION * MOTION_FPS)  # 300帧
REQUIRED_MOTION_START = 1.0  # 后4秒从第1秒开始


def load_motion_stat(motion_stat_dir: str) -> Dict[str, Dict[str, str]]:
    """
    加载motion_stat目录下的所有动作描述文件
    
    Returns:
        Dict[motion_name, Dict['name', 'description', 'semantic']]
    """
    motion_stat_path = Path(motion_stat_dir)
    motion_dict = {}
    
    for txt_file in motion_stat_path.glob("*.txt"):
        motion_name = txt_file.stem
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析文件内容
            lines = content.strip().split('\n')
            name_cn = ""
            description_cn = ""
            semantic_cn = ""
            
            for line in lines:
                if line.startswith("动作名称:"):
                    name_cn = line.replace("动作名称:", "").strip()
                elif line.startswith("描述:"):
                    description_cn = line.replace("描述:", "").strip()
                elif line.startswith("语义信息:"):
                    semantic_cn = line.replace("语义信息:", "").strip()
            
            motion_dict[motion_name] = {
                'name_cn': name_cn,
                'description_cn': description_cn,
                'semantic_cn': semantic_cn
            }
        except Exception as e:
            print(f"Warning: Failed to load {txt_file}: {e}")
    
    return motion_dict


def load_clip_model(device='cpu', clip_version='ViT-B/32'):
    """
    加载CLIP模型
    """
    clip_model, _ = clip.load(clip_version, device=device, jit=False)
    if torch.cuda.is_available() and device != 'cpu':
        clip.model.convert_weights(clip_model)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    return clip_model


def encode_text_with_clip(clip_model, text: str, device='cpu'):
    """
    使用CLIP对文本进行编码
    
    Args:
        clip_model: CLIP模型
        text: 要编码的文本
        device: 设备
    
    Returns:
        CLIP特征向量 (numpy array, shape: [512])
    """
    if not text or text.strip() == "":
        return np.zeros(512, dtype=np.float32)
    
    try:
        with torch.no_grad():
            text_tokens = clip.tokenize([text], truncate=True).to(device)
            text_features = clip_model.encode_text(text_tokens).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 归一化
            return text_features.cpu().numpy()[0]
    except Exception as e:
        print(f"Warning: CLIP encoding failed for '{text[:50]}...': {e}")
        return np.zeros(512, dtype=np.float32)


def calculate_segment_window(actual_start: float, actual_end: float, 
                            total_duration: float) -> Optional[Tuple[float, float]]:
    """
    计算5秒片段的窗口，使得后4秒（1-5秒）包含手势动作
    
    要求：在5秒片段[segment_start, segment_start+5]中，后4秒[segment_start+1, segment_start+5]
          必须与手势动作[actual_start, actual_end]有交集
    
    即需要满足：actual_start < segment_start + 5.0 且 actual_end > segment_start + 1.0
    
    Args:
        actual_start: 手势动作实际开始时间
        actual_end: 手势动作实际结束时间
        total_duration: 总时长
    
    Returns:
        (segment_start, segment_end) 或 None（如果无法满足条件）
    """
    # 手势动作时长
    motion_duration = actual_end - actual_start
    
    # 如果手势动作时长超过4秒，尝试将其放在后4秒的中间位置
    if motion_duration > 4.0:
        # 将手势动作的中心放在后4秒的中心（即segment_start + 3.0）
        motion_center = (actual_start + actual_end) / 2.0
        ideal_segment_start = motion_center - 3.0
        segment_start = max(0.0, ideal_segment_start)
        segment_end = min(total_duration, segment_start + SEGMENT_DURATION)
        
        # 如果窗口被截断，调整
        if segment_end - segment_start < SEGMENT_DURATION:
            segment_start = max(0.0, segment_end - SEGMENT_DURATION)
        
        # 验证是否有交集
        window_motion_start = segment_start + REQUIRED_MOTION_START
        window_motion_end = segment_start + SEGMENT_DURATION
        if actual_start < window_motion_end and actual_end > window_motion_start:
            return (segment_start, segment_end)
        return None
    
    # 计算使得手势动作与后4秒有交集的窗口起始位置
    # 需要满足：actual_start < segment_start + 5.0 且 actual_end > segment_start + 1.0
    # 即：segment_start > actual_start - 5.0 且 segment_start < actual_end - 1.0
    
    min_segment_start = actual_start - SEGMENT_DURATION  # segment_start > actual_start - 5.0
    max_segment_start = actual_end - REQUIRED_MOTION_START  # segment_start < actual_end - 1.0
    
    # 检查是否有有效的窗口位置
    if min_segment_start >= max_segment_start:
        return None
    
    # 选择窗口起始位置（优先选择较小的，尽量包含更多上下文）
    # 但不能小于0
    segment_start = max(0.0, min_segment_start + 0.01)  # 稍微大于min_segment_start
    segment_end = segment_start + SEGMENT_DURATION
    
    # 检查是否超出总时长
    if segment_end > total_duration:
        segment_end = total_duration
        segment_start = max(0.0, segment_end - SEGMENT_DURATION)
        
        # 重新检查是否有交集
        window_motion_start = segment_start + REQUIRED_MOTION_START
        window_motion_end = segment_start + SEGMENT_DURATION
        if actual_start >= window_motion_end or actual_end <= window_motion_start:
            return None
        return (segment_start, segment_end)
    
    # 验证是否有交集
    window_motion_start = segment_start + REQUIRED_MOTION_START
    window_motion_end = segment_start + SEGMENT_DURATION
    
    if actual_start < window_motion_end and actual_end > window_motion_start:
        return (segment_start, segment_end)
    
    return None


def extract_segment(motion: np.ndarray, audio_features: np.ndarray,
                   segment_start: float, segment_end: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    从motion和audio_features中提取指定时间段的片段
    
    Args:
        motion: motion数据，shape [frames, dim]
        audio_features: audio特征，shape [frames, dim]
        segment_start: 片段开始时间（秒）
        segment_end: 片段结束时间（秒）
    
    Returns:
        (motion_segment, audio_segment)
    """
    # 计算帧索引
    motion_start_frame = int(segment_start * MOTION_FPS)
    motion_end_frame = int(segment_end * MOTION_FPS)
    audio_start_frame = int(segment_start * AUDIO_FPS)
    audio_end_frame = int(segment_end * AUDIO_FPS)
    
    # 提取片段
    motion_segment = motion[motion_start_frame:motion_end_frame]
    audio_segment = audio_features[audio_start_frame:audio_end_frame]
    
    # 检查长度，如果不足则padding
    if motion_segment.shape[0] < MOTION_FRAMES_PER_SEGMENT:
        padding = np.zeros((MOTION_FRAMES_PER_SEGMENT - motion_segment.shape[0], motion_segment.shape[1]))
        motion_segment = np.concatenate([motion_segment, padding], axis=0)
    elif motion_segment.shape[0] > MOTION_FRAMES_PER_SEGMENT:
        motion_segment = motion_segment[:MOTION_FRAMES_PER_SEGMENT]
    
    if audio_segment.shape[0] < AUDIO_FRAMES_PER_SEGMENT:
        padding = np.zeros((AUDIO_FRAMES_PER_SEGMENT - audio_segment.shape[0], audio_segment.shape[1]))
        audio_segment = np.concatenate([audio_segment, padding], axis=0)
    elif audio_segment.shape[0] > AUDIO_FRAMES_PER_SEGMENT:
        audio_segment = audio_segment[:AUDIO_FRAMES_PER_SEGMENT]
    
    return motion_segment, audio_segment


def process_single_file(json_path: str, motion_stat_dict: Dict[str, Dict[str, str]],
                       output_dir: str, clip_model=None, device='cpu') -> List[Dict]:
    """
    处理单个JSON文件
    
    Returns:
        处理结果列表，每个元素包含片段信息和元数据
    """
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    json_stem = Path(json_path).stem
    data_dir = Path(json_path).parent
    
    # 加载motion和audio features
    motion_path = data_dir / f"{json_stem}.npz"
    audio_path = data_dir / f"{json_stem}_whisper_features.npy"
    
    if not motion_path.exists():
        print(f"Warning: Motion file not found: {motion_path}")
        return []
    
    if not audio_path.exists():
        print(f"Warning: Audio features not found: {audio_path}")
        return []
    
    # 加载数据
    motion_data = np.load(motion_path)
    if 'qpos' in motion_data:
        motion = motion_data['qpos']
    else:
        keys = list(motion_data.keys())
        if len(keys) > 0:
            motion = motion_data[keys[0]]
        else:
            print(f"Warning: No valid motion data in {motion_path}")
            return []
    
    audio_features = np.load(audio_path)
    
    # 获取总时长
    total_duration = data.get('total_duration', 0.0)
    
    # 处理每个手势动作
    results = []
    blended_timeline = data.get('blended_timeline', [])
    
    for idx, motion_item in enumerate(blended_timeline):
        motion_name = motion_item['motion']
        actual_start = motion_item['actual_start_time']
        actual_end = motion_item['actual_end_time']
        
        # 计算5秒窗口
        window = calculate_segment_window(actual_start, actual_end, total_duration)
        if window is None:
            print(f"Warning: Cannot create valid segment for {motion_name} in {json_stem}")
            continue
        
        segment_start, segment_end = window
        
        # 检查数据长度是否足够
        motion_frames_needed = int(segment_end * MOTION_FPS)
        audio_frames_needed = int(segment_end * AUDIO_FPS)
        
        if motion.shape[0] < motion_frames_needed or audio_features.shape[0] < audio_frames_needed:
            print(f"Warning: Insufficient data for {motion_name} in {json_stem}")
            continue
        
        # 提取片段
        motion_segment, audio_segment = extract_segment(
            motion, audio_features, segment_start, segment_end
        )
        
        # 获取动作的语义和描述（直接使用中文）
        motion_info = motion_stat_dict.get(motion_name, {})
        name_cn = motion_info.get('name_cn', '')
        description_cn = motion_info.get('description_cn', '')
        semantic_cn = motion_info.get('semantic_cn', '')
        
        # 使用CLIP提取特征
        clip_name_feature = None
        clip_description_feature = None
        clip_semantic_feature = None
        
        if clip_model is not None:
            if name_cn:
                clip_name_feature = encode_text_with_clip(clip_model, name_cn, device)
            if description_cn:
                clip_description_feature = encode_text_with_clip(clip_model, description_cn, device)
            if semantic_cn:
                clip_semantic_feature = encode_text_with_clip(clip_model, semantic_cn, device)
        
        # 保存片段
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名：{json_stem}_{motion_name}_{idx}
        segment_name = f"{json_stem}_{motion_name}_{idx}"
        
        # 保存motion片段
        motion_output_path = output_path / f"{segment_name}_motion.npz"
        np.savez_compressed(motion_output_path, qpos=motion_segment)
        
        # 保存audio片段
        audio_output_path = output_path / f"{segment_name}_audio.npy"
        np.save(audio_output_path, audio_segment)
        
        # 保存CLIP特征
        if clip_name_feature is not None:
            clip_name_path = output_path / f"{segment_name}_clip_name.npy"
            np.save(clip_name_path, clip_name_feature)
        
        if clip_description_feature is not None:
            clip_description_path = output_path / f"{segment_name}_clip_description.npy"
            np.save(clip_description_path, clip_description_feature)
        
        if clip_semantic_feature is not None:
            clip_semantic_path = output_path / f"{segment_name}_clip_semantic.npy"
            np.save(clip_semantic_path, clip_semantic_feature)
        
        # 保存元数据（直接保存中文）
        metadata = {
            'segment_name': segment_name,
            'source_file': json_stem,
            'motion_name': motion_name,
            'segment_start_time': float(segment_start),
            'segment_end_time': float(segment_end),
            'motion_start_time': float(actual_start),
            'motion_end_time': float(actual_end),
            'motion_name': name_cn,
            'motion_description': description_cn,
            'motion_semantic': semantic_cn,
            'audio_frames': int(audio_segment.shape[0]),
            'motion_frames': int(motion_segment.shape[0]),
            'has_clip_features': clip_model is not None
        }
        
        metadata_path = output_path / f"{segment_name}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        results.append(metadata)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='预处理semi_synthetic_v1数据集')
    parser.add_argument('--data_dir', type=str, 
                       default='/root/workspace/MARDM/data/semi_synthetic_v1',
                       help='semi_synthetic_v1数据目录')
    parser.add_argument('--motion_stat_dir', type=str,
                       default='/root/workspace/MARDM/data/motion_stat',
                       help='motion_stat目录')
    parser.add_argument('--output_dir', type=str,
                       default='/root/workspace/MARDM/data/semi_synthetic_v1_segments',
                       help='输出目录')
    parser.add_argument('--clip_version', type=str,
                       default='ViT-B/32',
                       help='CLIP模型版本')
    parser.add_argument('--device', type=str,
                       default=None,
                       help='设备 (cuda/cpu)，默认自动选择')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # 加载CLIP模型
    print(f"Loading CLIP model ({args.clip_version}) on {device}...")
    clip_model = load_clip_model(device=device, clip_version=args.clip_version)
    print("CLIP model loaded successfully")
    
    # 加载motion_stat
    print("Loading motion_stat...")
    motion_stat_dict = load_motion_stat(args.motion_stat_dir)
    print(f"Loaded {len(motion_stat_dict)} motion descriptions")
    
    # 查找所有JSON文件
    data_dir = Path(args.data_dir)
    json_files = sorted(data_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    # 处理每个JSON文件
    all_results = []
    for json_path in tqdm(json_files, desc="Processing files"):
        try:
            results = process_single_file(str(json_path), motion_stat_dict, args.output_dir, 
                                         clip_model=clip_model, device=device)
            all_results.extend(results)
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存汇总信息
    summary_path = Path(args.output_dir) / "summary.json"
    summary = {
        'total_segments': len(all_results),
        'segments': all_results
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Total segments created: {len(all_results)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()

