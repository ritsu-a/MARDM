import os
import torch
import numpy as np
import whisper
import argparse
from pathlib import Path
from tqdm import tqdm
from whisper_audio_feature import extract_whisper_features, save_features
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from threading import Lock
import queue
import time


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


def process_single_file(args_tuple):
    """
    处理单个文件的包装函数，用于多进程/多线程
    """
    wav_file, input_dir, output_dir, model_name, device, segment_length, overlap, skip_existing = args_tuple
    
    try:
        # 确定输出路径
        wav_path = Path(wav_file)
        if output_dir:
            rel_path = wav_path.relative_to(Path(input_dir))
            output_path = Path(output_dir) / rel_path.parent / f"{wav_path.stem}_whisper_features.npy"
        else:
            output_path = wav_path.parent / f"{wav_path.stem}_whisper_features.npy"
        
        # 检查是否已存在
        if skip_existing and output_path.exists():
            return ('skipped', wav_file, None)
        
        # 创建输出目录
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 加载模型（每个进程/线程需要自己的模型实例）
        model = whisper.load_model(model_name, device=device)
        
        # 提取特征
        features, text, info = extract_features_with_model(
            audio_path=wav_file,
            model=model,
            device=device,
            segment_length=segment_length,
            overlap=overlap
        )
        
        # 保存特征
        np.save(str(output_path), features)
        
        # 保存文本信息
        txt_path = str(output_path).replace('.npy', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"识别文本: {text}\n")
            if info:
                f.write(f"语言: {info.get('language', 'unknown')}\n")
                f.write(f"语言概率: {info.get('language_prob', 0.0):.4f}\n")
        
        return ('success', wav_file, None)
        
    except Exception as e:
        return ('failed', wav_file, str(e))


def worker_process(worker_id, gpu_id, file_queue, result_queue, input_dir, output_dir, 
                   model_name, segment_length, overlap, skip_existing):
    """
    工作进程函数，在每个GPU上运行
    """
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    
    # 加载模型（每个进程加载一次）
    model = whisper.load_model(model_name, device=device)
    
    processed = 0
    failed = 0
    skipped = 0
    
    while True:
        try:
            # 从队列获取文件
            wav_file = file_queue.get(timeout=1)
            if wav_file is None:  # 结束信号
                break
            
            try:
                # 确定输出路径
                wav_path = Path(wav_file)
                if output_dir:
                    rel_path = wav_path.relative_to(Path(input_dir))
                    output_path = Path(output_dir) / rel_path.parent / f"{wav_path.stem}_whisper_features.npy"
                else:
                    output_path = wav_path.parent / f"{wav_path.stem}_whisper_features.npy"
                
                # 检查是否已存在
                if skip_existing and output_path.exists():
                    skipped += 1
                    result_queue.put(('skipped', wav_file, None))
                    continue
                
                # 创建输出目录
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 提取特征
                features, text, info = extract_features_with_model(
                    audio_path=wav_file,
                    model=model,
                    device=device,
                    segment_length=segment_length,
                    overlap=overlap
                )
                
                # 保存特征
                np.save(str(output_path), features)
                
                # 保存文本信息
                txt_path = str(output_path).replace('.npy', '.txt')
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"识别文本: {text}\n")
                    if info:
                        f.write(f"语言: {info.get('language', 'unknown')}\n")
                        f.write(f"语言概率: {info.get('language_prob', 0.0):.4f}\n")
                
                processed += 1
                result_queue.put(('success', wav_file, None))
                
            except Exception as e:
                failed += 1
                result_queue.put(('failed', wav_file, str(e)))
                
        except:
            # multiprocessing.Queue 的 Empty 异常处理
            time.sleep(0.1)
            continue
    
    result_queue.put(('worker_done', worker_id, (processed, failed, skipped)))


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
    num_workers=1,
    num_gpus=None,
    use_multiprocessing=True
):
    """
    批量处理音频文件，提取 Whisper 特征
    
    Args:
        input_dir: 输入目录（包含 wav 文件的根目录）
        output_dir: 输出目录（如果为 None，则在输入文件同目录保存）
        model_name: Whisper 模型名称
        device: 设备 ('cuda' 或 'cpu')，多GPU时会被忽略
        segment_length: 分段长度（秒）
        overlap: 分段重叠长度（秒）
        skip_existing: 是否跳过已存在的特征文件
        num_workers: 每个GPU的线程/进程数（默认 1）
        num_gpus: 使用的GPU数量（None表示使用所有可用GPU）
        use_multiprocessing: 是否使用多进程（True）或多线程（False）
    
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
    
    # 检测GPU数量
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        if num_gpus is None:
            num_gpus = available_gpus
        else:
            num_gpus = min(num_gpus, available_gpus)
        print(f"检测到 {available_gpus} 个GPU，使用 {num_gpus} 个GPU")
    else:
        num_gpus = 1
        print("未检测到GPU，使用CPU")
    
    # 如果只有一个GPU或CPU，使用单进程多线程模式
    if num_gpus == 1 and not use_multiprocessing:
        return _process_single_device_multithread(
            wav_files, input_dir, output_dir, model_name, device, 
            segment_length, overlap, skip_existing, num_workers
        )
    
    # 多GPU或多进程模式
    if num_gpus > 1:
        return _process_multigpu(
            wav_files, input_dir, output_dir, model_name, 
            segment_length, overlap, skip_existing, num_gpus, num_workers
        )
    else:
        # 单GPU多进程模式
        return _process_single_device_multiprocess(
            wav_files, input_dir, output_dir, model_name, device,
            segment_length, overlap, skip_existing, num_workers
        )


def _process_single_device_multithread(
    wav_files, input_dir, output_dir, model_name, device,
    segment_length, overlap, skip_existing, num_workers
):
    """单设备多线程处理（共享模型）"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"使用设备: {device}, 线程数: {num_workers}")
    print(f"加载 Whisper 模型: {model_name}...")
    
    # 在主线程加载模型（线程安全）
    model = whisper.load_model(model_name, device=device)
    print("模型加载完成！")
    
    # 准备任务参数（包含模型）
    tasks = []
    for wav_file in wav_files:
        tasks.append((
            wav_file, input_dir, output_dir, model, device,
            segment_length, overlap, skip_existing
        ))
    
    # 使用线程池处理
    processed_count = 0
    failed_count = 0
    skipped_count = 0
    failed_files = []
    
    def process_with_shared_model(args_tuple):
        """使用共享模型处理单个文件"""
        wav_file, input_dir, output_dir, model, device, segment_length, overlap, skip_existing = args_tuple
        try:
            # 确定输出路径
            wav_path = Path(wav_file)
            if output_dir:
                rel_path = wav_path.relative_to(Path(input_dir))
                output_path = Path(output_dir) / rel_path.parent / f"{wav_path.stem}_whisper_features.npy"
            else:
                output_path = wav_path.parent / f"{wav_path.stem}_whisper_features.npy"
            
            # 检查是否已存在
            if skip_existing and output_path.exists():
                return ('skipped', wav_file, None)
            
            # 创建输出目录
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 提取特征（使用共享模型）
            features, text, info = extract_features_with_model(
                audio_path=wav_file,
                model=model,
                device=device,
                segment_length=segment_length,
                overlap=overlap
            )
            
            # 保存特征
            np.save(str(output_path), features)
            
            # 保存文本信息
            txt_path = str(output_path).replace('.npy', '.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"识别文本: {text}\n")
                if info:
                    f.write(f"语言: {info.get('language', 'unknown')}\n")
                    f.write(f"语言概率: {info.get('language_prob', 0.0):.4f}\n")
            
            return ('success', wav_file, None)
            
        except Exception as e:
            return ('failed', wav_file, str(e))
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_with_shared_model, task): task[0] for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理进度"):
            status, wav_file, error = future.result()
            if status == 'success':
                processed_count += 1
            elif status == 'skipped':
                skipped_count += 1
            else:
                failed_count += 1
                failed_files.append((wav_file, error))
                if error:
                    print(f"\n处理失败: {wav_file}")
                    print(f"错误信息: {error}")
    
    _print_summary(len(wav_files), processed_count, skipped_count, failed_count, failed_files)
    return processed_count, failed_count, skipped_count


def _process_single_device_multiprocess(
    wav_files, input_dir, output_dir, model_name, device,
    segment_length, overlap, skip_existing, num_workers
):
    """单设备多进程处理"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"使用设备: {device}, 进程数: {num_workers}")
    
    # 准备任务参数
    tasks = []
    for wav_file in wav_files:
        tasks.append((
            wav_file, input_dir, output_dir, model_name, device,
            segment_length, overlap, skip_existing
        ))
    
    # 使用进程池处理
    processed_count = 0
    failed_count = 0
    skipped_count = 0
    failed_files = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_file, task): task[0] for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理进度"):
            status, wav_file, error = future.result()
            if status == 'success':
                processed_count += 1
            elif status == 'skipped':
                skipped_count += 1
            else:
                failed_count += 1
                failed_files.append((wav_file, error))
                if error:
                    print(f"\n处理失败: {wav_file}")
                    print(f"错误信息: {error}")
    
    _print_summary(len(wav_files), processed_count, skipped_count, failed_count, failed_files)
    return processed_count, failed_count, skipped_count


def _process_multigpu(
    wav_files, input_dir, output_dir, model_name,
    segment_length, overlap, skip_existing, num_gpus, num_workers_per_gpu
):
    """多GPU处理（每个GPU一个进程，进程内多线程）"""
    print(f"使用 {num_gpus} 个GPU，每个GPU {num_workers_per_gpu} 个线程")
    
    # 将文件分配到不同的GPU
    files_per_gpu = len(wav_files) // num_gpus
    file_chunks = []
    for i in range(num_gpus):
        start_idx = i * files_per_gpu
        if i == num_gpus - 1:
            end_idx = len(wav_files)
        else:
            end_idx = (i + 1) * files_per_gpu
        file_chunks.append(wav_files[start_idx:end_idx])
    
    # 为每个GPU创建进程
    processes = []
    result_queue = mp.Queue()
    file_queues = [mp.Queue() for _ in range(num_gpus)]
    
    # 将文件放入队列
    for i, chunk in enumerate(file_chunks):
        for wav_file in chunk:
            file_queues[i].put(wav_file)
        # 添加结束信号
        for _ in range(num_workers_per_gpu):
            file_queues[i].put(None)
    
    # 启动工作进程
    for gpu_id in range(num_gpus):
        for worker_id in range(num_workers_per_gpu):
            p = mp.Process(
                target=worker_process,
                args=(
                    worker_id, gpu_id, file_queues[gpu_id], result_queue,
                    input_dir, output_dir, model_name, segment_length,
                    overlap, skip_existing
                )
            )
            p.start()
            processes.append(p)
    
    # 收集结果
    processed_count = 0
    failed_count = 0
    skipped_count = 0
    failed_files = []
    worker_done_count = 0
    completed_files = 0
    
    with tqdm(total=len(wav_files), desc="处理进度") as pbar:
        while completed_files < len(wav_files) or worker_done_count < len(processes):
            try:
                status, wav_file, error = result_queue.get(timeout=2)
                if status == 'success':
                    processed_count += 1
                    completed_files += 1
                    pbar.update(1)
                elif status == 'skipped':
                    skipped_count += 1
                    completed_files += 1
                    pbar.update(1)
                elif status == 'failed':
                    failed_count += 1
                    completed_files += 1
                    failed_files.append((wav_file, error))
                    pbar.update(1)
                    if error:
                        print(f"\n处理失败: {wav_file}")
                        print(f"错误信息: {error}")
                elif status == 'worker_done':
                    worker_done_count += 1
            except:
                # multiprocessing.Queue 超时或其他异常
                if all(not p.is_alive() for p in processes) and completed_files >= len(wav_files):
                    break
                continue
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    _print_summary(len(wav_files), processed_count, skipped_count, failed_count, failed_files)
    return processed_count, failed_count, skipped_count


def _print_summary(total, processed, skipped, failed, failed_files):
    """打印统计信息"""
    print("\n" + "=" * 60)
    print("批量处理完成！")
    print("=" * 60)
    print(f"总文件数: {total}")
    print(f"成功处理: {processed}")
    print(f"跳过文件: {skipped}")
    print(f"失败文件: {failed}")
    
    if failed_files:
        print("\n失败文件列表:")
        for wav_file, error in failed_files[:10]:  # 只显示前10个
            print(f"  - {wav_file}: {error}")
        if len(failed_files) > 10:
            print(f"  ... 还有 {len(failed_files) - 10} 个失败文件")


def main(args):
    # 设置多进程启动方法（仅在需要多进程时）
    if args.num_gpus is None or args.num_gpus > 1 or args.use_multiprocessing:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # 已经设置过了
    
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
        skip_existing=args.skip_existing,
        num_workers=args.num_workers,
        num_gpus=args.num_gpus,
        use_multiprocessing=args.use_multiprocessing
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
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='每个GPU的线程/进程数（默认 4）'
    )
    parser.add_argument(
        '--num_gpus',
        type=int,
        default=None,
        help='使用的GPU数量（None表示使用所有可用GPU，默认 None）'
    )
    parser.add_argument(
        '--use_multiprocessing',
        action='store_true',
        default=True,
        help='使用多进程模式（默认 True，多GPU时自动启用）'
    )
    parser.add_argument(
        '--use_multithreading',
        dest='use_multiprocessing',
        action='store_false',
        help='使用多线程模式（单GPU时推荐）'
    )
    
    args = parser.parse_args()
    main(args)

