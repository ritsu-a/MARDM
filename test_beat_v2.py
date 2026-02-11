import os
os.environ["MUJOCO_GL"] = "egl"

from os.path import join as pjoin
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from models.AE import AE_models
from models.MARDM import MARDM_models
from utils.datasets import BEAT_v2Audio2MotionDataset, MixedAudioTextDataset, SemiSyntheticAudioTextDataset, collate_fn
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
    
    # 根据数据集类型加载不同的数据集
    if args.dataset_name == 'mixed':
        beat_v2_root = '/root/workspace/MARDM/data/BEAT_v2'
        semi_synthetic_root = '/root/workspace/MARDM/data/semi_synthetic_v1_segments'
        test_dataset = MixedAudioTextDataset(mean, std, beat_v2_root, semi_synthetic_root, 
                                             args.unit_length, args.max_motion_length, split='val')
    elif args.dataset_name == 'semi_synthetic':
        semi_synthetic_root = '/root/workspace/MARDM/data/semi_synthetic_v1_segments'
        test_dataset = SemiSyntheticAudioTextDataset(mean, std, semi_synthetic_root, 
                                                      args.unit_length, args.max_motion_length, split='val')
    else:
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
    if args.dataset_name == 'mixed':
        cond_mode = 'mixed'
    else:
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
        for idx, batch_data in enumerate(tqdm(test_loader, desc="Testing batches")):
            if idx * args.batch_size >= num_samples:
                break
            
            # 处理不同数据集格式
            if args.dataset_name == 'mixed' or args.dataset_name == 'semi_synthetic':
                whisper_features, clip_features, motion_condition, motion_target, m_lens = batch_data
            else:
                whisper_features, motion_condition, motion_target, m_lens = batch_data
                clip_features = None
            
            # 移动到设备
            whisper_features = whisper_features.to(device).float()  # [B, 250, 512]
            motion_condition = motion_condition.to(device).float()  # [B, 60, dim_pose]
            motion_target = motion_target.to(device).float()  # [B, 240, dim_pose]
            m_lens = m_lens.to(device).long()  # Total length: 300
            
            # Encode condition motion to latent
            motion_condition_latent = ae.encode(motion_condition)  # [B, ae_dim, 15] (60/4=15)
            
            # Target length in latent space: 60 (for 240 frames)
            m_lens_target = torch.tensor([240 // 4] * whisper_features.size(0), device=device).long()  # [B] = [60]
            
            # 准备text condition（mixed模式）
            text_condition_tensor = None
            if cond_mode == 'mixed':
                if clip_features is not None:
                    text_condition_tensor = clip_features.to(device).float()  # [B, 512]
                else:
                    # 如果没有提供CLIP特征，使用零向量padding
                    batch_size = whisper_features.size(0)
                    clip_feature = np.zeros(512, dtype=np.float32)
                    text_condition_tensor = torch.from_numpy(clip_feature).unsqueeze(0).expand(batch_size, -1).to(device).float()  # [B, 512]
            
            # 生成motion（带进度条）
            # whisper_features shape: [B, 250, 512]
            # m_lens_target: [B] - target motion lengths in latent space (240/4 = 60)
            # motion_condition_latent: [B, ae_dim, 15] - condition motion latents (60/4 = 15)
            with tqdm(total=args.time_steps, desc=f"  Generating batch {idx+1}", leave=False) as pbar:
                pred_latents = ema_mardm.generate(
                    conds=whisper_features,  # [B, 250, 512]
                    m_lens=m_lens_target,  # [B] - latent lengths (60 for 240 frames)
                    timesteps=args.time_steps,
                    cond_scale=args.cfg,
                    temperature=args.temperature,
                    progress_callback=lambda step: pbar.update(1),
                    motion_condition_latent=motion_condition_latent,  # [B, ae_dim, 15]
                    text_condition=text_condition_tensor  # [B, 512] for mixed mode
                )
            # 解码motion
            # pred_latents shape: [B, ae_dim, 60] where 60 is latent sequence length for 240 frames
            # ae.decode expects [B, C, T] format and outputs [B, T, dim_pose] (decoder has permute at the end)
            pred_motions = ae.decode(pred_latents)  # [B, 240, dim_pose]
            pred_motions = pred_motions.detach().cpu().numpy()
            
            # Concatenate condition + predicted motion
            motion_condition_np = motion_condition.detach().cpu().numpy()
            pred_motions_full = np.concatenate([motion_condition_np, pred_motions], axis=1)  # [B, 300, dim_pose]
            
            # Debug: 检查形状（仅第一个batch）
            if idx == 0:
                print(f"Debug: motion_condition.shape = {motion_condition.shape}")
                print(f"Debug: motion_condition_latent.shape = {motion_condition_latent.shape}")
                print(f"Debug: pred_latents.shape = {pred_latents.shape}")
                print(f"Debug: pred_motions.shape = {pred_motions.shape}")
                print(f"Debug: pred_motions_full.shape = {pred_motions_full.shape}")
                print(f"Debug: m_lens (motion frames) = {m_lens.cpu().numpy()}")
                print(f"Debug: m_lens_target (latent frames) = {m_lens_target.cpu().numpy()}")
            
            pred_motions_denorm = pred_motions_full * std + mean
            
            # 处理ground truth: concatenate condition + target
            motion_target_np = motion_target.detach().cpu().numpy()
            motion_gt_full = np.concatenate([motion_condition_np, motion_target_np], axis=1)  # [B, 300, dim_pose]
            motion_gt_denorm = motion_gt_full * std + mean
            
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
                # Full motion: 300 frames (60 condition + 240 predicted)
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
    
    # 处理音频特征：需要250帧（对应240帧motion生成 + 60帧条件）
    # Frame alignment: 50 audio frames = 60 motion frames
    # Ratio: audio_frames / motion_frames = 50/60 = 5/6
    # For 240 motion frames (target), we need 240 * 5/6 = 200 audio frames
    # But we use 250 audio frames to match training (250 audio -> 240 motion target + 60 condition)
    target_audio_frames = 250
    target_motion_frames_generate = 240  # Frames to generate (target)
    condition_motion_frames = 60  # Condition frames
    target_latent_frames = target_motion_frames_generate // 4  # 60 (for 240 frames)
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
    if args.dataset_name == 'mixed' or args.dataset_name == 'semi_synthetic':
        cond_mode = 'mixed'
    else:
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
    
    # 对于mixed模式，需要准备CLIP特征（从音频生成时使用零向量）
    clip_feature = None
    if cond_mode == 'mixed':
        # 从音频生成时没有text description，使用零向量padding
        clip_feature = np.zeros(512, dtype=np.float32)  # CLIP feature dimension is 512
        print("Using zero-padded CLIP feature for audio-only generation (mixed mode)")
    
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
    
    # 创建初始条件motion（60帧）：使用均值姿势（rest pose）
    condition_motion_frames = 60
    condition_motion = np.zeros((condition_motion_frames, dim_pose))  # [60, dim_pose]
    # 使用均值作为初始姿势（已经是归一化后的，所以均值是0）
    # 如果需要，可以使用mean（但mean在归一化后是0）
    condition_motion_tensor = torch.from_numpy(condition_motion).unsqueeze(0).to(device).float()  # [1, 60, dim_pose]
    condition_motion_latent = ae.encode(condition_motion_tensor)  # [1, ae_dim, 15] (60/4=15)
    
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
            m_lens = torch.tensor([target_latent_frames], dtype=torch.long).to(device)  # [1] = [60]
            
            # 对于第一个片段，使用初始条件motion
            # 对于后续片段，使用前一个片段生成的后60帧作为条件（实现连续性）
            if seg_idx == 0:
                current_condition_latent = condition_motion_latent
            else:
                # 使用前一个片段生成的后60帧作为条件
                # 前一个片段生成的是240帧，取最后60帧
                prev_motion = motion_segments[-1]  # [240, dim_pose]
                prev_condition_motion = prev_motion[-condition_motion_frames:]  # [60, dim_pose]
                prev_condition_motion_tensor = torch.from_numpy(prev_condition_motion).unsqueeze(0).to(device).float()
                current_condition_latent = ae.encode(prev_condition_motion_tensor)  # [1, ae_dim, 15]
            
            # 准备text condition（mixed模式）
            text_condition_tensor = None
            if cond_mode == 'mixed':
                text_condition_tensor = torch.from_numpy(clip_feature).unsqueeze(0).to(device).float()  # [1, 512]
            
            # 生成motion（带进度条）
            with tqdm(total=args.time_steps, desc=f"  Segment {seg_idx+1}/{len(audio_segments)}", leave=False) as pbar:
                pred_latents = ema_mardm.generate(
                    conds=audio_features_tensor,  # [1, 250, 512]
                    m_lens=m_lens,  # [1] = [60] for 240 frames
                    timesteps=args.time_steps,
                    cond_scale=args.cfg,
                    temperature=args.temperature,
                    progress_callback=lambda step: pbar.update(step),
                    motion_condition_latent=current_condition_latent,  # [1, ae_dim, 15]
                    text_condition=text_condition_tensor  # [1, 512] for mixed mode
                )
            
            # 解码motion
            pred_motions = ae.decode(pred_latents)  # [1, 240, dim_pose]
            pred_motions = pred_motions.detach().cpu().numpy()[0]  # [240, dim_pose]
            pred_motions_denorm = pred_motions * std + mean
            
            motion_segments.append(pred_motions_denorm)
    
    # 拼接motion片段
    # 每个片段生成240帧，需要拼接
    print("Concatenating motion segments...")
    if len(motion_segments) == 1:
        # 只有一个片段，直接使用（240帧）
        pred_motions_denorm = motion_segments[0]
    else:
        # 多个片段，直接拼接（每个片段240帧）
        # 注意：由于每个片段使用前一个片段的后60帧作为条件，所以片段之间是连续的
        pred_motions_denorm = np.concatenate(motion_segments, axis=0)  # [N*240, dim_pose]
        
        print(f"Concatenated {len(motion_segments)} segments into {pred_motions_denorm.shape[0]} frames")
    
    # 在开头添加初始条件motion（60帧）
    condition_motion_denorm = condition_motion * std + mean  # [60, dim_pose]
    pred_motions_denorm = np.concatenate([condition_motion_denorm, pred_motions_denorm], axis=0)  # [60 + N*240, dim_pose]
    
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
    parser.add_argument('--name', type=str, default='MARDM_SiT_XL_mixed',
                       help='Model name')
    parser.add_argument('--ae_name', type=str, default="AE",
                       help='AE model name')
    parser.add_argument('--ae_model', type=str, default='AE_Model',
                       help='AE model type')
    parser.add_argument('--model', type=str, default='MARDM-SiT-XL',
                       help='MARDM model type')
    parser.add_argument('--dataset_name', type=str, default='mixed',
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

