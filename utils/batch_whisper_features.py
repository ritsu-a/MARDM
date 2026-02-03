import os
import torch
import numpy as np
import whisper
import argparse
from pathlib import Path
from tqdm import tqdm
from whisper_audio_feature import extract_whisper_features, save_features


def find_wav_files(root_dir):
    """
    递归查找所有 wav 文件
    
    Args:
        root_dir: 根目录路径
    
    Returns:
        wav_files: wav 文件路径列表
    """
    wav_files = []
    root_path = Path(root_dir)
    
    # 递归查找所有 .wav 文件
    for wav_file in root_path.rglob("*.wav"):
        wav_files.append(str(wav_file))
    
    return sorted(wav_files)


def extract_features_with_model(audio_path, model, device, segment_length=30.0, overlap=0.0):
    """
    使用已加载的模型提取特征（避免重复加载模型）
    """
    # 加载音频（不进行 pad_or_trim）
    audio = whisper.load_audio(audio_path)
    audio_duration = len(audio) / 16000.0
    
    # 使用 transcribe 进行语音识别（自动处理长音频）
    result = model.transcribe(audio_path)
    text = result["text"]
    info = {
        "language": result.get("language", "unknown"),
        "language_prob": result.get("language_prob", 0.0)
    }
    
    # 提取编码器特征
    hop_length = 160
    
    # 如果音频长度 <= segment_length，直接处理
    if audio_duration <= segment_length:
        audio_segment = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio_segment).to(device)
        
        with torch.no_grad():
            encoder_output = model.encoder(mel.unsqueeze(0))
            features = encoder_output.squeeze(0).cpu().numpy()
            
            # 根据实际音频长度截取特征
            mel_frames = len(audio) // hop_length
            feature_frames = mel_frames // 2
            features = features[:feature_frames]
    else:
        # 长音频分段处理
        segment_samples = int(segment_length * 16000)
        overlap_samples = int(overlap * 16000)
        step_samples = segment_samples - overlap_samples
        
        feature_list = []
        total_samples = len(audio)
        
        start_idx = 0
        segment_idx = 0
        
        while start_idx < total_samples:
            end_idx = min(start_idx + segment_samples, total_samples)
            audio_segment = audio[start_idx:end_idx]
            
            audio_segment = whisper.pad_or_trim(audio_segment)
            mel = whisper.log_mel_spectrogram(audio_segment).to(device)
            
            with torch.no_grad():
                encoder_output = model.encoder(mel.unsqueeze(0))
                segment_features = encoder_output.squeeze(0).cpu().numpy()
            
            actual_segment_samples = end_idx - start_idx
            mel_frames = actual_segment_samples // hop_length
            feature_frames = mel_frames // 2
            
            segment_features = segment_features[:feature_frames]
            feature_list.append(segment_features)
            
            segment_idx += 1
            
            if end_idx >= total_samples:
                break
            start_idx += step_samples
        
        features = np.concatenate(feature_list, axis=0)
    
    return features, text, info


def batch_process_whisper_features(
    input_dir,
    output_dir=None,
    model_name="base",
    device=None,
    segment_length=30.0,
    overlap=0.0,
    skip_existing=True,
    num_workers=1
):
    """
    批量处理音频文件，提取 Whisper 特征
    
    Args:
        input_dir: 输入目录（包含 wav 文件的根目录）
        output_dir: 输出目录（如果为 None，则在输入文件同目录保存）
        model_name: Whisper 模型名称
        device: 设备 ('cuda' 或 'cpu')
        segment_length: 分段长度（秒）
        overlap: 分段重叠长度（秒）
        skip_existing: 是否跳过已存在的特征文件
        num_workers: 并行处理数量（当前版本为 1，顺序处理）
    
    Returns:
        processed_count: 成功处理的文件数
        failed_count: 失败的文件数
        skipped_count: 跳过的文件数
    """
    # 查找所有 wav 文件
    print(f"正在查找 {input_dir} 目录下的所有 wav 文件...")
    wav_files = find_wav_files(input_dir)
    
    if len(wav_files) == 0:
        print(f"未找到任何 wav 文件！")
        return 0, 0, 0
    
    print(f"找到 {len(wav_files)} 个 wav 文件")
    
    # 加载模型（只加载一次）
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"使用设备: {device}")
    print(f"加载 Whisper 模型: {model_name}...")
    model = whisper.load_model(model_name, device=device)
    print("模型加载完成！")
    
    # 统计信息
    processed_count = 0
    failed_count = 0
    skipped_count = 0
    failed_files = []
    
    # 处理每个文件
    for wav_file in tqdm(wav_files, desc="处理进度"):
        try:
            # 确定输出路径
            wav_path = Path(wav_file)
            if output_dir:
                # 保持相对路径结构
                rel_path = wav_path.relative_to(Path(input_dir))
                output_path = Path(output_dir) / rel_path.parent / f"{wav_path.stem}_whisper_features.npy"
            else:
                # 保存在同目录
                output_path = wav_path.parent / f"{wav_path.stem}_whisper_features.npy"
            
            # 检查是否已存在
            if skip_existing and output_path.exists():
                skipped_count += 1
                continue
            
            # 创建输出目录
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 提取特征（使用已加载的模型）
            features, text, info = extract_features_with_model(
                audio_path=wav_file,
                model=model,
                device=device,
                segment_length=segment_length,
                overlap=overlap
            )
            
            # 保存特征（不打印详细信息）
            np.save(str(output_path), features)
            
            # 保存文本信息（可选）
            txt_path = str(output_path).replace('.npy', '.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"识别文本: {text}\n")
                if info:
                    f.write(f"语言: {info.get('language', 'unknown')}\n")
                    f.write(f"语言概率: {info.get('language_prob', 0.0):.4f}\n")
            
            processed_count += 1
            
        except Exception as e:
            failed_count += 1
            failed_files.append((wav_file, str(e)))
            print(f"\n处理失败: {wav_file}")
            print(f"错误信息: {e}")
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("批量处理完成！")
    print("=" * 60)
    print(f"总文件数: {len(wav_files)}")
    print(f"成功处理: {processed_count}")
    print(f"跳过文件: {skipped_count}")
    print(f"失败文件: {failed_count}")
    
    if failed_files:
        print("\n失败文件列表:")
        for wav_file, error in failed_files[:10]:  # 只显示前10个
            print(f"  - {wav_file}: {error}")
        if len(failed_files) > 10:
            print(f"  ... 还有 {len(failed_files) - 10} 个失败文件")
    
    return processed_count, failed_count, skipped_count


def main(args):
    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"输入目录不存在: {args.input_dir}")
    
    # 创建输出目录（如果指定）
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"输出目录: {args.output_dir}")
    else:
        print("输出目录: 与输入文件同目录")
    
    # 批量处理
    processed, failed, skipped = batch_process_whisper_features(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        device=args.device,
        segment_length=args.segment_length,
        overlap=args.overlap,
        skip_existing=args.skip_existing
    )
    
    print(f"\n处理完成！成功: {processed}, 失败: {failed}, 跳过: {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量处理音频文件，提取 Whisper 特征")
    parser.add_argument(
        '--input_dir',
        type=str,
        default='/root/workspace/MARDM/data/BEAT_v2',
        help='输入目录（包含 wav 文件的根目录）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录（如果为 None，则在输入文件同目录保存）'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper 模型名称 (tiny/base/small/medium/large)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='使用的设备（默认自动选择）'
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
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        default=True,
        help='跳过已存在的特征文件（默认 True）'
    )
    parser.add_argument(
        '--no_skip_existing',
        dest='skip_existing',
        action='store_false',
        help='不跳过已存在的特征文件（重新处理所有文件）'
    )
    
    args = parser.parse_args()
    main(args)

