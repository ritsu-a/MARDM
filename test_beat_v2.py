import os
os.environ["MUJOCO_GL"] = "egl"

from os.path import join as pjoin
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from models.AE import AE_models
from models.MARDM import MARDM_models
from utils.datasets import BEAT_v2Audio2MotionDataset, collate_fn
from utils.whisper_audio_feature import extract_whisper_features
from general_motion_retargeting import RobotMotionViewer
import argparse
from tqdm import tqdm
import tempfile
import imageio
from skimage.transform import resize
from moviepy.editor import VideoFileClip, AudioFileClip
import shutil


def vis_npz_motion(motion_npz_path, output_path, robot_type="g1_brainco", rate_limit=False, motion_fps=30, label_text=""):
    """Visualize motion from npz file (BEAT_v2 qpos format)"""
    data = np.load(motion_npz_path)
    
    if 'qpos' in data:
        motion_csv = data['qpos']
    elif 'qpos_original' in data:
        motion_csv = data['qpos_original']
    elif 'original' in data:
        motion_csv = data['original']
    elif 'reconstructed' in data:
        motion_csv = data['reconstructed']
    else:
        keys = list(data.keys())
        if len(keys) > 0:
            motion_csv = data[keys[0]]
        else:
            raise ValueError(f"No valid motion data found in {motion_npz_path}")
    
    data_frames = motion_csv.shape[0]
    
    robot_motion_viewer = RobotMotionViewer(robot_type=robot_type,
                                            motion_fps=motion_fps,
                                            transparent_robot=0,
                                            record_video=True,
                                            video_path=output_path)
    
    pbar = tqdm(total=data_frames, desc=f"Visualizing {label_text}" if label_text else "Visualizing", leave=False)
    
    i = 0
    while i < data_frames:
        pbar.update(1)
        qpos = motion_csv[i]
        quat_wxyz = qpos[3:7]
        root_pos = qpos[:3]
        
        robot_motion_viewer.step(
            root_pos=root_pos,
            root_rot=quat_wxyz,
            dof_pos=qpos[7:],
            rate_limit=rate_limit,
        )
        i += 1
    
    pbar.close()
    robot_motion_viewer.close()
    del robot_motion_viewer


def concatenate_videos_horizontally(video1_path, video2_path, output_path):
    """将两个视频左右拼接"""
    reader1 = imageio.get_reader(video1_path)
    reader2 = imageio.get_reader(video2_path)
    
    # 获取视频属性
    fps = reader1.get_meta_data()['fps']
    width1, height1 = reader1.get_meta_data()['size']
    width2, height2 = reader2.get_meta_data()['size']
    
    # 确保两个视频高度相同，如果不相同则调整
    target_height = max(height1, height2)
    target_width = width1 + width2
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 使用imageio写入视频
    writer = imageio.get_writer(output_path, fps=fps)
    
    print(f"  Concatenating videos horizontally...")
    pbar = tqdm(desc="  Concatenating", leave=False)
    
    frame_count = 0
    try:
        # 使用迭代器方式读取，直到任一视频结束
        for frame1, frame2 in zip(reader1, reader2):
            # 调整两个视频到相同高度
            if frame1.shape[0] != target_height:
                frame1 = resize(frame1, (target_height, width1), preserve_range=True, anti_aliasing=True).astype(frame1.dtype)
            if frame2.shape[0] != target_height:
                frame2 = resize(frame2, (target_height, width2), preserve_range=True, anti_aliasing=True).astype(frame2.dtype)
            
            # 左右拼接
            combined_frame = np.hstack([frame1, frame2])
            writer.append_data(combined_frame)
            frame_count += 1
            pbar.update(1)
    except (StopIteration, IndexError):
        # 视频读取结束
        pass
    
    pbar.close()
    reader1.close()
    reader2.close()
    writer.close()
    print(f"  Concatenated {frame_count} frames")


def combine_video_with_audio(video_path, audio_path, output_path):
    """
    将视频和音频合并
    
    参数:
    video_path: 视频文件路径
    audio_path: 音频文件路径（可选，为None或空字符串时只复制视频）
    output_path: 输出文件路径
    """
    # 如果没有音频路径，直接复制视频文件
    if audio_path is None or not os.path.exists(audio_path):
        if os.path.abspath(video_path) != os.path.abspath(output_path):
            shutil.copy2(video_path, output_path)
        return
    
    # 如果源文件路径和目标文件路径相同，需要先将文件复制到临时位置
    temp_video = video_path
    if os.path.abspath(video_path) == os.path.abspath(output_path):
        temp_video = video_path + ".temp.mp4"
        shutil.copy2(video_path, temp_video)
    
    try:
        # 加载视频文件
        final_clip = VideoFileClip(temp_video)
        
        # 加载音频文件
        audio = AudioFileClip(audio_path)
        
        # 设置视频的音频
        # 如果音频长度超过视频长度，截取音频；如果短于视频，循环音频
        if audio.duration > final_clip.duration:
            audio = audio.subclip(0, final_clip.duration)
        elif audio.duration < final_clip.duration:
            # 如果音频短于视频，循环音频
            num_loops = int(np.ceil(final_clip.duration / audio.duration))
            audio = audio.loop(n=num_loops).subclip(0, final_clip.duration)
        
        # 设置音频
        final_clip = final_clip.set_audio(audio)
        
        # 导出最终视频
        print(f"  Combining video with audio...")
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            verbose=False,
            logger=None
        )
        
        # 关闭所有剪辑以释放资源
        audio.close()
        final_clip.close()
    finally:
        # 如果使用了临时文件，删除它
        if temp_video != video_path and os.path.exists(temp_video):
            os.remove(temp_video)


def test_on_testset(args):
    """在test set上展示效果"""
    print("=" * 80)
    print("Testing on test set...")
    print("=" * 80)
    
    # 设置随机种子
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 加载数据
    data_root = '/root/workspace/MARDM/data/BEAT_v2'
    mean = np.load(pjoin(data_root, 'Mean.npy'))
    std = np.load(pjoin(data_root, 'Std.npy'))
    dim_pose = mean.shape[0]
    
    # 使用val split作为test set（因为数据集没有单独的test split）
    test_dataset = BEAT_v2Audio2MotionDataset(mean, std, data_root, args.unit_length, 
                                               args.max_motion_length, split='train')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False, 
                             num_workers=args.num_workers, shuffle=False)
    
    # 加载模型
    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')
    ae = AE_models[args.ae_model](input_width=dim_pose)
    ckpt = torch.load(pjoin(args.checkpoints_dir, args.dataset_name, args.ae_name, 'model',
                            'latest.tar'), map_location='cpu')
    ae.load_state_dict(ckpt['ae'])
    
    # 加载MARDM模型
    cond_mode = 'audio'
    audio_dim = 512
    ema_mardm = MARDM_models[args.model](ae_dim=ae.output_emb_width, cond_mode=cond_mode, audio_dim=audio_dim)
    checkpoint_path = pjoin(model_dir, 'latest.tar')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    ema_mardm.load_state_dict(checkpoint['ema_mardm'], strict=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae.to(device)
    ema_mardm.to(device)
    ae.eval()
    ema_mardm.eval()
    
    # 创建输出目录
    result_dir = pjoin('./test_results', args.name, 'test_set')
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"Test set size: {len(test_dataset)}")
    print(f"Output directory: {result_dir}")
    
    # 测试
    num_samples = min(args.num_test_samples, len(test_dataset))
    
    with torch.no_grad():
        for idx, (whisper_features, motion_gt, m_lens) in enumerate(tqdm(test_loader, desc="Testing batches")):
            if idx * args.batch_size >= num_samples:
                break
            
            # 移动到设备
            whisper_features = whisper_features.to(device).float()  # [B, 250, 512]
            # m_lens from dataset is in motion frame space (300), need to convert to latent space (300/4 = 75)
            m_lens_latent = (m_lens // 4).to(device).long()  # Convert to latent space length
            
            # 生成motion（带进度条）
            # whisper_features shape: [B, 250, 512]
            # m_lens_latent: [B] - motion lengths in latent space (300/4 = 75)
            with tqdm(total=args.time_steps, desc=f"  Generating batch {idx+1}", leave=False) as pbar:
                pred_latents = ema_mardm.generate(
                    conds=whisper_features,  # [B, 250, 512]
                    m_lens=m_lens_latent,  # [B] - latent lengths (300)
                    timesteps=args.time_steps,
                    cond_scale=args.cfg,
                    temperature=args.temperature,
                    progress_callback=lambda step: pbar.update(1)
                )
            # 解码motion
            # pred_latents shape: [B, ae_dim, L] where L is latent sequence length
            # ae.decode expects [B, C, T] format and outputs [B, T, dim_pose] (decoder has permute at the end)
            pred_motions = ae.decode(pred_latents)  # [B, T, dim_pose] where T = L * 4
            pred_motions = pred_motions.detach().cpu().numpy()
            
            # Debug: 检查形状（仅第一个batch）
            if idx == 0:
                print(f"Debug: pred_latents.shape = {pred_latents.shape}")
                print(f"Debug: pred_motions.shape = {pred_motions.shape}")
                print(f"Debug: m_lens (motion frames) = {m_lens.cpu().numpy()}")
                print(f"Debug: m_lens_latent (latent frames) = {m_lens_latent.cpu().numpy()}")
                print(f"Debug: expected motion length = {m_lens[0].item()}")
            
            pred_motions_denorm = pred_motions * std + mean
            
            # 处理ground truth
            motion_gt = motion_gt.detach().cpu().numpy()
            motion_gt_denorm = motion_gt * std + mean
            
            # 保存结果
            batch_size = whisper_features.size(0)
            for b in range(batch_size):
                sample_idx = idx * args.batch_size + b
                if sample_idx >= num_samples:
                    break
                
                # 创建样本目录
                sample_dir = pjoin(result_dir, f'sample_{sample_idx:04d}')
                os.makedirs(sample_dir, exist_ok=True)
                
                # 预测的motion (qpos format: [frames, qpos_dim])
                # m_lens[b] is motion length (300), pred_motions should be 300 frames
                actual_motion_len = m_lens[b].item()  # 300
                pred_motion = pred_motions_denorm[b][:actual_motion_len]  # [300, dim_pose]
                
                # 保存为npz格式
                pred_npz_path = pjoin(sample_dir, 'prediction.npz')
                np.savez(pred_npz_path, qpos=pred_motion)
                
                # 可视化预测motion（带进度条）
                pred_video_path = pjoin(sample_dir, 'prediction.mp4')
                print(f"  Rendering prediction video for sample {sample_idx}...")
                vis_npz_motion(pred_npz_path, pred_video_path, 
                              robot_type=args.robot_type, 
                              rate_limit=args.rate_limit, 
                              motion_fps=args.motion_fps, 
                              label_text="Predicted")
                
                # Ground truth motion
                gt_motion = motion_gt_denorm[b][:m_lens[b].item()]  # [300, dim_pose]
                
                # 保存为npz格式
                gt_npz_path = pjoin(sample_dir, 'ground_truth.npz')
                np.savez(gt_npz_path, qpos=gt_motion)
                
                # 可视化ground truth motion（带进度条）
                print(f"  Rendering ground truth video for sample {sample_idx}...")
                gt_video_path = pjoin(sample_dir, 'ground_truth.mp4')
                vis_npz_motion(gt_npz_path, gt_video_path, 
                              robot_type=args.robot_type, 
                              rate_limit=args.rate_limit, 
                              motion_fps=args.motion_fps, 
                              label_text="Ground Truth")
                
                # 拼接gt和pred视频（左右拼接）
                comparison_video_path = pjoin(sample_dir, 'comparison.mp4')
                print(f"  Creating comparison video (GT left, Pred right) for sample {sample_idx}...")
                concatenate_videos_horizontally(gt_video_path, pred_video_path, comparison_video_path)
                
                print(f"Saved sample {sample_idx} to {sample_dir}")
    
    print(f"\nTest completed! Results saved to {result_dir}")


def generate_from_audio_wav(args):
    """从给定的audio wav文件生成motion并渲染成视频"""
    print("=" * 80)
    print("Generating motion from audio wav file...")
    print("=" * 80)
    
    # 检查音频文件是否存在
    if not os.path.exists(args.audio_path):
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")
    
    # 设置随机种子
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 加载数据统计信息
    data_root = '/root/workspace/MARDM/data/BEAT_v2'
    mean = np.load(pjoin(data_root, 'Mean.npy'))
    std = np.load(pjoin(data_root, 'Std.npy'))
    dim_pose = mean.shape[0]
    
    # 提取whisper特征
    print(f"Extracting whisper features from {args.audio_path}...")
    features, text, info = extract_whisper_features(
        audio_path=args.audio_path,
        model_name=args.whisper_model,
        device=args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"),
        return_embedding=True,
        segment_length=30.0,
        overlap=0.0
    )
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Recognized text: {text[:100]}..." if len(text) > 100 else f"Recognized text: {text}")
    
    # 处理音频特征：需要250帧（对应300帧motion）
    # Frame alignment: 50 audio frames = 60 motion frames
    # Ratio: audio_frames / motion_frames = 50/60 = 5/6
    # For 300 motion frames, we need 300 * 5/6 = 250 audio frames
    target_audio_frames = 250
    target_motion_frames = 300
    target_latent_frames = target_motion_frames // 4  # 75
    overlap_ratio = 0.5  # 50%重叠
    
    # 如果音频特征长度 < 250，填充到250帧
    if features.shape[0] < target_audio_frames:
        padding = np.zeros((target_audio_frames - features.shape[0], features.shape[1]))
        features = np.concatenate([features, padding], axis=0)
        print(f"Padded audio features from {features.shape[0]} to {target_audio_frames} frames")
    
    # 加载模型
    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')
    ae = AE_models[args.ae_model](input_width=dim_pose)
    ckpt = torch.load(pjoin(args.checkpoints_dir, args.dataset_name, args.ae_name, 'model',
                            'latest.tar'), map_location='cpu')
    ae.load_state_dict(ckpt['ae'])
    
    # 加载MARDM模型
    cond_mode = 'audio'
    audio_dim = 512
    ema_mardm = MARDM_models[args.model](ae_dim=ae.output_emb_width, cond_mode=cond_mode, audio_dim=audio_dim)
    checkpoint_path = pjoin(model_dir, 'latest.tar')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    ema_mardm.load_state_dict(checkpoint['ema_mardm'], strict=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae.to(device)
    ema_mardm.to(device)
    ae.eval()
    ema_mardm.eval()
    
    # 处理长音频：如果音频长度 > 250帧，分成带重叠的片段
    total_audio_frames = features.shape[0]
    
    if total_audio_frames <= target_audio_frames:
        # 短音频：直接处理
        print(f"Audio is short ({total_audio_frames} frames), processing directly...")
        audio_segments = [(0, total_audio_frames)]
    else:
        # 长音频：分成带重叠的片段
        print(f"Audio is long ({total_audio_frames} frames), splitting into overlapping segments...")
        step_size = int(target_audio_frames * (1 - overlap_ratio))  # 每个片段前进的步长（50%重叠）
        audio_segments = []
        start = 0
        while start < total_audio_frames:
            end = min(start + target_audio_frames, total_audio_frames)
            audio_segments.append((start, end))
            if end >= total_audio_frames:
                break
            start += step_size
        print(f"Split into {len(audio_segments)} segments with {overlap_ratio*100:.0f}% overlap")
    
    # 生成每个片段的motion
    motion_segments = []
    print("Generating motion for each segment...")
    with torch.no_grad():
        for seg_idx, (start_audio, end_audio) in enumerate(tqdm(audio_segments, desc="Processing segments")):
            # 提取音频片段
            audio_segment = features[start_audio:end_audio]
            
            # 如果片段长度不足250帧，填充
            if audio_segment.shape[0] < target_audio_frames:
                padding = np.zeros((target_audio_frames - audio_segment.shape[0], audio_segment.shape[1]))
                audio_segment = np.concatenate([audio_segment, padding], axis=0)
            
            # 准备输入
            audio_features_tensor = torch.from_numpy(audio_segment).unsqueeze(0).to(device).float()  # [1, 250, 512]
            m_lens = torch.tensor([target_latent_frames], dtype=torch.long).to(device)  # [1] = [75]
            
            # 生成motion（带进度条）
            with tqdm(total=args.time_steps, desc=f"  Segment {seg_idx+1}/{len(audio_segments)}", leave=False) as pbar:
                pred_latents = ema_mardm.generate(
                    conds=audio_features_tensor,  # [1, 250, 512]
                    m_lens=m_lens,  # [1] = [75]
                    timesteps=args.time_steps,
                    cond_scale=args.cfg,
                    temperature=args.temperature,
                    progress_callback=lambda step: pbar.update(step)
                )
            
            # 解码motion
            pred_motions = ae.decode(pred_latents)  # [1, 300, dim_pose]
            pred_motions = pred_motions.detach().cpu().numpy()[0]  # [300, dim_pose]
            pred_motions_denorm = pred_motions * std + mean
            
            motion_segments.append(pred_motions_denorm)
    
    # 拼接motion片段（带插值）
    print("Concatenating motion segments with interpolation...")
    if len(motion_segments) == 1:
        # 只有一个片段，直接使用
        pred_motions_denorm = motion_segments[0]
    else:
        # 多个片段，需要拼接
        # 计算每个片段对应的motion帧范围
        # audio到motion的比例：250 audio frames -> 300 motion frames
        audio_to_motion_ratio = target_motion_frames / target_audio_frames  # 300/250 = 1.2
        
        # 计算每个音频片段对应的motion帧范围
        motion_segment_ranges = []
        for start_audio, end_audio in audio_segments:
            start_motion = int(start_audio * audio_to_motion_ratio)
            end_motion = int(end_audio * audio_to_motion_ratio)
            motion_segment_ranges.append((start_motion, end_motion))
        
        # 计算总motion长度
        total_motion_frames = int(total_audio_frames * audio_to_motion_ratio)
        pred_motions_denorm = np.zeros((total_motion_frames, dim_pose))
        
        # 计算每个位置的权重（用于重叠区域的插值）
        weights = np.zeros(total_motion_frames)
        
        # 将每个片段添加到对应位置
        for seg_idx, (motion_seg, (start_motion, end_motion)) in enumerate(zip(motion_segments, motion_segment_ranges)):
            actual_seg_len = min(motion_seg.shape[0], end_motion - start_motion)
            actual_end = min(start_motion + actual_seg_len, total_motion_frames)
            
            # 计算重叠区域的插值权重
            # 使用cosine插值（更平滑）：cosine插值在边界处有零导数，过渡更自然
            overlap_size = int(target_motion_frames * overlap_ratio)  # 重叠的motion帧数
            
            def cosine_interpolate(start, end, n):
                """Cosine插值：从start到end，共n个点"""
                t = np.linspace(0, np.pi, n)
                return start + (end - start) * (1 - np.cos(t)) / 2
            
            if seg_idx == 0:
                # 第一个片段：前半部分权重为1，后半部分（重叠区域）权重从1平滑减少到0
                seg_weights = np.ones(actual_seg_len)
                if actual_seg_len > overlap_size:
                    # 重叠区域权重cosine减少（更平滑）
                    fade_out = cosine_interpolate(1.0, 0.0, overlap_size)
                    seg_weights[-overlap_size:] = fade_out
            elif seg_idx == len(motion_segments) - 1:
                # 最后一个片段：前半部分（重叠区域）权重从0平滑增加到1，后半部分权重为1
                seg_weights = np.ones(actual_seg_len)
                if actual_seg_len > overlap_size:
                    # 重叠区域权重cosine增加（更平滑）
                    fade_in = cosine_interpolate(0.0, 1.0, overlap_size)
                    seg_weights[:overlap_size] = fade_in
            else:
                # 中间片段：前半部分（重叠区域）权重从0平滑增加到1，后半部分（重叠区域）权重从1平滑减少到0
                seg_weights = np.ones(actual_seg_len)
                if actual_seg_len > 2 * overlap_size:
                    # 前半部分重叠区域权重cosine增加（更平滑）
                    fade_in = cosine_interpolate(0.0, 1.0, overlap_size)
                    seg_weights[:overlap_size] = fade_in
                    # 后半部分重叠区域权重cosine减少（更平滑）
                    fade_out = cosine_interpolate(1.0, 0.0, overlap_size)
                    seg_weights[-overlap_size:] = fade_out
                elif actual_seg_len > overlap_size:
                    # 如果片段较短，只处理前半部分
                    fade_in = cosine_interpolate(0.0, 1.0, overlap_size)
                    seg_weights[:overlap_size] = fade_in
            
            # 加权累加
            pred_motions_denorm[start_motion:actual_end] += motion_seg[:actual_end-start_motion] * seg_weights[:actual_end-start_motion, None]
            weights[start_motion:actual_end] += seg_weights[:actual_end-start_motion]
        
        # 归一化（避免除零）
        weights[weights == 0] = 1.0
        pred_motions_denorm = pred_motions_denorm / weights[:, None]
        
        print(f"Concatenated {len(motion_segments)} segments into {total_motion_frames} frames")
    
    # 创建输出目录
    audio_name = os.path.splitext(os.path.basename(args.audio_path))[0]
    result_dir = pjoin('./test_results', args.name, 'audio_generation', audio_name)
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存motion数据为npz格式 (qpos format)
    motion_npz_path = pjoin(result_dir, 'generated_motion.npz')
    np.savez(motion_npz_path, qpos=pred_motions_denorm)
    np.save(pjoin(result_dir, 'audio_features.npy'), features)
    
    # 渲染视频
    print("Rendering video...")
    video_path = pjoin(result_dir, 'generated_motion.mp4')
    vis_npz_motion(motion_npz_path, video_path, 
                  robot_type=args.robot_type, 
                  rate_limit=args.rate_limit, 
                  motion_fps=args.motion_fps, 
                  label_text=f"Generated from {os.path.basename(args.audio_path)}")
    
    # 合并视频和音频
    video_with_audio_path = pjoin(result_dir, 'generated_motion_with_audio.mp4')
    print("Combining video with original audio...")
    combine_video_with_audio(video_path, args.audio_path, video_with_audio_path)
    
    # 保存识别文本
    with open(pjoin(result_dir, 'recognized_text.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Audio file: {args.audio_path}\n")
        f.write(f"Recognized text: {text}\n")
        f.write(f"Language: {info.get('language', 'unknown')}\n")
        f.write(f"Language probability: {info.get('language_prob', 0.0):.4f}\n")
        f.write(f"Audio features shape: {features.shape}\n")
        f.write(f"Generated motion shape: {pred_motions_denorm.shape}\n")
        f.write(f"Number of segments: {len(motion_segments) if 'motion_segments' in locals() else 1}\n")
    
    print(f"\nGeneration completed!")
    print(f"Video saved to: {video_path}")
    print(f"Video with audio saved to: {video_with_audio_path}")
    print(f"Motion data saved to: {motion_npz_path}")
    print(f"Results directory: {result_dir}")


def main(args):
    if args.mode == 'testset':
        test_on_testset(args)
    elif args.mode == 'audio':
        generate_from_audio_wav(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Choose 'testset' or 'audio'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MARDM model for beat_v2 dataset")
    
    # 基本参数
    parser.add_argument('--mode', type=str, default='testset', choices=['testset', 'audio'],
                       help='Test mode: "testset" for testing on test set, "audio" for generating from audio wav')
    parser.add_argument('--name', type=str, default='MARDM_SiT_XL_beat_v2',
                       help='Model name')
    parser.add_argument('--ae_name', type=str, default="AE",
                       help='AE model name')
    parser.add_argument('--ae_model', type=str, default='AE_Model',
                       help='AE model type')
    parser.add_argument('--model', type=str, default='MARDM-SiT-XL',
                       help='MARDM model type')
    parser.add_argument('--dataset_name', type=str, default='beat_v2',
                       help='Dataset name')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                       help='Checkpoints directory')
    
    # 测试集相关参数
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for test set evaluation')
    parser.add_argument('--num_test_samples', type=int, default=100,
                       help='Number of samples to test on test set')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--max_motion_length', type=int, default=300,
                       help='Maximum motion length')
    parser.add_argument('--unit_length', type=int, default=4,
                       help='Unit length for motion')
    
    # 音频生成相关参数
    parser.add_argument('--audio_path', type=str, default='',
                       help='Path to audio wav file (required for audio mode)')
    parser.add_argument('--whisper_model', type=str, default='base',
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model name')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='Device for whisper feature extraction')
    
    # 生成参数
    parser.add_argument('--time_steps', type=int, default=18,
                       help='Number of diffusion timesteps')
    parser.add_argument('--cfg', type=float, default=4.5,
                       help='Classifier-free guidance scale')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for sampling')
    parser.add_argument('--seed', type=int, default=3407,
                       help='Random seed')
    
    # Visualization parameters
    parser.add_argument('--robot_type', type=str, default='g1_brainco',
                       help='Robot type for visualization (g1_brainco, unitree_g1, etc.)')
    parser.add_argument('--rate_limit', action='store_true',
                       help='Rate limit for visualization')
    parser.add_argument('--motion_fps', type=int, default=60,
                       help='Motion FPS for visualization')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.mode == 'audio' and not args.audio_path:
        raise ValueError("--audio_path is required when mode is 'audio'")
    
    main(args)

