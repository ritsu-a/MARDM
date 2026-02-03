import os
import torch
import numpy as np
import whisper
import argparse
from pathlib import Path


def extract_whisper_features(audio_path, model_name="base", device=None, return_embedding=True, 
                             segment_length=30.0, overlap=0.0):
    """
    使用 OpenAI Whisper 提取音频特征（支持长音频）
    
    Args:
        audio_path: 音频文件路径
        model_name: Whisper 模型名称 (tiny, base, small, medium, large)
        device: 设备 ('cuda' 或 'cpu')，如果为 None 则自动选择
        return_embedding: 是否返回编码器嵌入特征
        segment_length: 分段长度（秒），默认 30 秒
        overlap: 分段重叠长度（秒），默认 0 秒
    
    Returns:
        features: 音频特征 (numpy array)，形状为 [时间帧数, 特征维度]
        text: 识别的文本
        info: 其他信息（语言、概率等）
    
    特征长度与音频长度的关系：
    ============================
    1. Whisper 固定采样率: 16000 Hz
    2. 长音频会被分段处理，每段 segment_length 秒
    3. Mel Spectrogram:
       - hop_length = 160 (每帧对应 160 个采样点)
       - 时间分辨率 = 160 / 16000 = 0.01 秒 = 10 毫秒/帧
    4. 编码器特征:
       - 编码器通常有 2x 下采样
       - 时间分辨率 ≈ 20 毫秒/帧
       - 特征维度: 取决于模型 (base 模型通常是 512 维)
    
    总结:
    - 特征时间维度与音频长度成正比
    - 每帧特征对应约 20 毫秒的音频
    - 特征维度 (features.shape[1]) 取决于模型大小
    """
    # 设置设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"使用设备: {device}")
    print(f"加载 Whisper 模型: {model_name}")
    
    # 加载 Whisper 模型
    model = whisper.load_model(model_name, device=device)
    
    # 加载音频（不进行 pad_or_trim）
    print(f"加载音频文件: {audio_path}")
    audio = whisper.load_audio(audio_path)  # 重采样到 16000 Hz
    audio_duration = len(audio) / 16000.0  # 音频时长（秒）
    print(f"音频时长: {audio_duration:.2f} 秒 ({len(audio)} 采样点)")
    
    # 使用 transcribe 进行语音识别（自动处理长音频）
    print("进行语音识别...")
    result = model.transcribe(audio_path)
    text = result["text"]
    info = {
        "language": result.get("language", "unknown"),
        "language_prob": result.get("language_prob", 0.0)
    }
    
    # 提取编码器特征
    print("提取编码器特征...")
    
    # 如果音频长度 <= segment_length，直接处理
    if audio_duration <= segment_length:
        # 对短音频进行 pad_or_trim
        audio_segment = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio_segment).to(device)
        
        with torch.no_grad():
            encoder_output = model.encoder(mel.unsqueeze(0))
            features = encoder_output.squeeze(0).cpu().numpy()
            
            # 根据实际音频长度截取特征
            hop_length = 160  # Whisper 的 hop_length
            mel_frames = len(audio) // hop_length
            feature_frames = mel_frames // 2  # 编码器 2x 下采样
            features = features[:feature_frames]
    else:
        # 长音频分段处理
        hop_length = 160  # Whisper 的 hop_length
        segment_samples = int(segment_length * 16000)  # 每段采样点数
        overlap_samples = int(overlap * 16000)  # 重叠采样点数
        step_samples = segment_samples - overlap_samples  # 每步采样点数
        
        feature_list = []
        total_samples = len(audio)
        
        print(f"分段处理: 每段 {segment_length} 秒, 重叠 {overlap} 秒")
        
        start_idx = 0
        segment_idx = 0
        
        while start_idx < total_samples:
            end_idx = min(start_idx + segment_samples, total_samples)
            audio_segment = audio[start_idx:end_idx]
            
            # 对每段进行 pad_or_trim（确保长度一致）
            audio_segment = whisper.pad_or_trim(audio_segment)
            mel = whisper.log_mel_spectrogram(audio_segment).to(device)
            
            with torch.no_grad():
                encoder_output = model.encoder(mel.unsqueeze(0))
                segment_features = encoder_output.squeeze(0).cpu().numpy()
            
            # 计算实际有效特征长度
            actual_segment_samples = end_idx - start_idx
            mel_frames = actual_segment_samples // hop_length
            feature_frames = mel_frames // 2  # 编码器 2x 下采样
            
            # 截取有效特征（去除 padding）
            segment_features = segment_features[:feature_frames]
            feature_list.append(segment_features)
            
            segment_idx += 1
            if segment_idx % 10 == 0:
                print(f"  已处理 {segment_idx} 段...")
            
            # 移动到下一段（考虑重叠）
            if end_idx >= total_samples:
                break
            start_idx += step_samples
        
        # 拼接所有段的特征
        features = np.concatenate(feature_list, axis=0)
        print(f"  共处理 {segment_idx} 段，拼接后特征长度: {features.shape[0]} 帧")
    
    print(f"识别文本: {text[:100]}..." if len(text) > 100 else f"识别文本: {text}")
    print(f"语言: {info['language']}")
    print(f"特征形状: {features.shape}")
    print(f"特征时间维度: {features.shape[0]} 帧")
    print(f"特征空间维度: {features.shape[1]} 维")
    
    # 计算时间分辨率
    time_per_frame = audio_duration / features.shape[0]
    print(f"实际音频时长: {audio_duration:.2f} 秒")
    print(f"特征时间分辨率: {time_per_frame * 1000:.2f} 毫秒/帧")
    print(f"每帧特征对应约 {time_per_frame * 1000:.1f} 毫秒的音频")
    
    return features, text, info


def save_features(features, output_path, text=None, info=None):
    """
    保存特征到文件
    
    Args:
        features: 特征数组
        output_path: 输出文件路径
        text: 识别的文本（可选）
        info: 其他信息（可选）
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # 保存特征
    np.save(output_path, features)
    print(f"特征已保存到: {output_path}")
    
    # 如果提供了文本，保存到文本文件
    if text is not None:
        txt_path = output_path.replace('.npy', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"识别文本: {text}\n")
            if info:
                f.write(f"语言: {info.get('language', 'unknown')}\n")
                f.write(f"语言概率: {info.get('language_prob', 0.0):.4f}\n")
        print(f"文本信息已保存到: {txt_path}")


def main(args):
    # 检查输入文件是否存在
    if not os.path.exists(args.audio_path):
        raise FileNotFoundError(f"音频文件不存在: {args.audio_path}")
    
    # 提取特征
    features, text, info = extract_whisper_features(
        audio_path=args.audio_path,
        model_name=args.model_name,
        device=args.device,
        return_embedding=args.return_embedding,
        segment_length=args.segment_length,
        overlap=args.overlap
    )
    
    # 确定输出路径
    if args.output_path:
        output_path = args.output_path
    else:
        # 默认输出路径：与音频文件同目录，文件名相同但扩展名为 .npy
        audio_dir = os.path.dirname(args.audio_path)
        audio_name = Path(args.audio_path).stem
        output_path = os.path.join(audio_dir, f"{audio_name}_whisper_features.npy")
    
    # 保存特征
    save_features(features, output_path, text, info)
    
    # 计算实际音频时长
    audio = whisper.load_audio(args.audio_path)
    actual_duration = len(audio) / 16000.0
    
    print(f"\n处理完成！")
    print(f"输入音频: {args.audio_path}")
    print(f"输出特征: {output_path}")
    print(f"特征形状: {features.shape}")
    print(f"特征维度: {features.shape[0]} 帧 x {features.shape[1]} 维")
    print(f"\n特征长度说明:")
    print(f"  - 实际音频时长: {actual_duration:.2f} 秒")
    print(f"  - 时间帧数 (features.shape[0]): {features.shape[0]} 帧")
    print(f"  - 时间分辨率: {actual_duration / features.shape[0] * 1000:.2f} 毫秒/帧")
    print(f"  - 特征维度 (features.shape[1]): {features.shape[1]} 维")
    print(f"\n注意: 使用 transcribe 方法处理长音频，特征长度与音频长度成正比")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 OpenAI Whisper 提取音频特征")
    parser.add_argument(
        '--audio_path',
        type=str,
        default='/root/workspace/MARDM/data/BEAT_v2/1/1_wayne_0_1_1.wav',
        help='音频文件路径'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper 模型名称 (tiny/base/small/medium/large)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='输出特征文件路径（默认为音频文件同目录）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='使用的设备（默认自动选择）'
    )
    parser.add_argument(
        '--return_embedding',
        action='store_true',
        default=True,
        help='返回编码器嵌入特征（默认 True）'
    )
    parser.add_argument(
        '--segment_length',
        type=float,
        default=30.0,
        help='分段长度（秒），用于处理长音频（默认 30.0 秒）'
    )
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.0,
        help='分段重叠长度（秒），默认 0 秒'
    )
    
    args = parser.parse_args()
    main(args)

