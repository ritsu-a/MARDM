import os
os.environ["MUJOCO_GL"] = "egl"

from os.path import join as pjoin
import torch
import numpy as np
import random
import json
from torch.utils.data import DataLoader
from models.AE import AE_models
from utils.datasets import BEAT_v2Dataset, MixedDataset
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset
import tempfile
import shutil
import imageio
from skimage.transform import resize
from PIL import Image, ImageDraw, ImageFont
from general_motion_retargeting import RobotMotionViewer
import time
from moviepy.editor import VideoFileClip, AudioFileClip

class SelectedSamplesDataset(Dataset):
    """
    只加载选中样本的数据集类
    """
    def __init__(self, base_dataset, selected_indices, mean, std):
        """
        Args:
            base_dataset: 基础数据集（BEAT_v2Dataset或MixedDataset）
            selected_indices: 选中的样本索引列表
            mean: 均值（用于反归一化）
            std: 标准差（用于反归一化）
        """
        self.base_dataset = base_dataset
        self.selected_indices = selected_indices
        self.mean = mean
        self.std = std
    
    def __len__(self):
        return len(self.selected_indices)
    
    def __getitem__(self, idx):
        # 获取在base_dataset中的实际索引
        actual_idx = self.selected_indices[idx]
        return self.base_dataset[actual_idx]

def vis_npz_motion(motion_npz_path, output_path, robot_type="g1_branco", rate_limit=False, motion_fps=30, label_text=""):
    """Visualize motion from npz file"""
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

def add_text_to_video(video_path, output_path, text, position='top-left', font_size=40):
    """Add text label to video frames"""
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']
    width, height = reader.get_meta_data()['size']
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    writer = imageio.get_writer(output_path, fps=fps)
    
    if position == 'top-left':
        text_pos = (20, 20)
    elif position == 'top-right':
        text_pos = (width - 200, 20)
    elif position == 'bottom-left':
        text_pos = (20, height - 60)
    elif position == 'bottom-right':
        text_pos = (width - 200, height - 60)
    else:
        text_pos = (width // 2 - 100, height // 2)
    
    for frame in reader:
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        
        # Draw outline (black)
        for adj in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            draw.text((text_pos[0] + adj[0], text_pos[1] + adj[1]), text, font=font, fill='black')
        # Draw text (white)
        draw.text(text_pos, text, font=font, fill='white')
        
        frame_with_text = np.array(img)
        writer.append_data(frame_with_text)
    
    reader.close()
    writer.close()

def concatenate_videos_horizontally(video1_path, video2_path, output_path):
    """Concatenate two videos horizontally"""
    reader1 = imageio.get_reader(video1_path)
    reader2 = imageio.get_reader(video2_path)
    
    fps = reader1.get_meta_data()['fps']
    width1, height1 = reader1.get_meta_data()['size']
    width2, height2 = reader2.get_meta_data()['size']
    
    target_height = max(height1, height2)
    target_width = width1 + width2
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    writer = imageio.get_writer(output_path, fps=fps)
    
    frame_count = 0
    try:
        for frame1, frame2 in zip(reader1, reader2):
            if frame1.shape[0] != target_height:
                frame1 = resize(frame1, (target_height, width1), preserve_range=True, anti_aliasing=True).astype(frame1.dtype)
            if frame2.shape[0] != target_height:
                frame2 = resize(frame2, (target_height, width2), preserve_range=True, anti_aliasing=True).astype(frame2.dtype)
            
            combined_frame = np.hstack([frame1, frame2])
            writer.append_data(combined_frame)
            frame_count += 1
    except (StopIteration, IndexError):
        pass
    
    reader1.close()
    reader2.close()
    writer.close()

def combine_video_with_audio(video_path, audio_path, output_path):
    """
    将视频和音频合并
    """
    if audio_path is None or not os.path.exists(audio_path):
        if os.path.abspath(video_path) != os.path.abspath(output_path):
            shutil.copy2(video_path, output_path)
        return
    
    temp_video = video_path
    if os.path.abspath(video_path) == os.path.abspath(output_path):
        temp_video = video_path + ".temp.mp4"
        shutil.copy2(video_path, temp_video)
    
    try:
        final_clip = VideoFileClip(temp_video)
        audio = AudioFileClip(audio_path)
        
        if audio.duration > final_clip.duration:
            audio = audio.subclip(0, final_clip.duration)
        elif audio.duration < final_clip.duration:
            num_loops = int(np.ceil(final_clip.duration / audio.duration))
            audio = audio.loop(n=num_loops).subclip(0, final_clip.duration)
        
        final_clip = final_clip.set_audio(audio)
        
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            verbose=False,
            logger=None
        )
        
        audio.close()
        final_clip.close()
    finally:
        if temp_video != video_path and os.path.exists(temp_video):
            os.remove(temp_video)

def extract_audio_segment(audio_path, start_time, end_time, output_path):
    """
    从音频文件中提取指定时间段的音频
    """
    try:
        audio = AudioFileClip(audio_path)
        segment = audio.subclip(start_time, end_time)
        segment.write_audiofile(output_path, verbose=False, logger=None)
        audio.close()
        segment.close()
        return True
    except Exception as e:
        print(f"Error extracting audio segment: {e}")
        return False

def find_audio_path_for_sample(sample_info, beat_v2_root, semi_synthetic_root, temp_dir=None, beat_v2_npz_files=None):
    """
    根据样本信息找到对应的音频文件路径
    对于semi_synthetic，返回临时提取的音频文件路径
    """
    dataset_type = sample_info.get('dataset_type', 'beat_v2')
    
    if dataset_type == 'beat_v2':
        # BEAT_v2: 音频文件路径是npz文件路径替换.npz为.wav
        npz_path = sample_info.get('source_npz_path', '')
        if npz_path:
            audio_path = npz_path.replace('.npz', '.wav')
            if os.path.exists(audio_path):
                return audio_path
    elif dataset_type == 'semi_synthetic':
        # semi_synthetic: 从metadata读取source_file，找到对应的.wav文件并提取时间段
        source_file = sample_info.get('source_file', '')
        segment_start = sample_info.get('segment_start_time', 0.0)
        segment_end = sample_info.get('segment_end_time', 5.0)
        sample_id = sample_info.get('sample_id', 0)
        
        if source_file and temp_dir:
            # 查找原始wav文件
            wav_path = pjoin('/root/workspace/MARDM/data/semi_synthetic_v1', f'{source_file}.wav')
            if os.path.exists(wav_path):
                # 提取对应时间段的音频
                temp_audio_path = os.path.join(temp_dir, f'audio_segment_{sample_id}.wav')
                if extract_audio_segment(wav_path, segment_start, segment_end, temp_audio_path):
                    return temp_audio_path
    
    return None

def main(args):
    torch.backends.cudnn.benchmark = False
    os.environ["OMP_NUM_THREADS"] = "1"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load datasets
    beat_v2_root = '/root/workspace/MARDM/data/BEAT_v2'
    semi_synthetic_root = '/root/workspace/MARDM/data/semi_synthetic_v1_segments'
    
    mean = np.load(pjoin(beat_v2_root, 'Mean.npy'))
    std = np.load(pjoin(beat_v2_root, 'Std.npy'))
    dim_pose = mean.shape[0]
    joints_num = dim_pose
    
    # Create base datasets (needed to get dataset size and sample from)
    beat_v2_base_dataset = BEAT_v2Dataset(mean, std, beat_v2_root, args.window_size, split='val')
    semi_synthetic_base_dataset = MixedDataset(mean, std, beat_v2_root, semi_synthetic_root, args.window_size, split='val')
    
    # Sample from each dataset
    num_samples_per_dataset = args.num_samples_per_dataset
    
    # Sample from BEAT_v2 - select exactly num_samples_per_dataset different samples
    beat_v2_size = len(beat_v2_base_dataset)
    if num_samples_per_dataset > beat_v2_size:
        print(f"Warning: Requested {num_samples_per_dataset} samples but BEAT_v2 only has {beat_v2_size}. Using all samples.")
        beat_v2_indices = list(range(beat_v2_size))
    else:
        if args.sample_strategy == 'diverse':
            spacing = max(1, beat_v2_size // num_samples_per_dataset)
            max_start = max(0, beat_v2_size - num_samples_per_dataset * spacing)
            start_idx = random.randint(0, max_start) if max_start > 0 else 0
            beat_v2_indices = [start_idx + i * spacing for i in range(num_samples_per_dataset)]
            beat_v2_indices = [idx for idx in beat_v2_indices if idx < beat_v2_size]
            # Ensure we have enough samples
            while len(beat_v2_indices) < num_samples_per_dataset and len(beat_v2_indices) < beat_v2_size:
                new_idx = random.randint(0, beat_v2_size - 1)
                if new_idx not in beat_v2_indices:
                    beat_v2_indices.append(new_idx)
            beat_v2_indices = sorted(beat_v2_indices[:num_samples_per_dataset])
        elif args.sample_strategy == 'sequential':
            beat_v2_indices = list(range(num_samples_per_dataset))
        else:  # random
            beat_v2_indices = sorted(random.sample(range(beat_v2_size), num_samples_per_dataset))
    
    print(f"Selected {len(beat_v2_indices)} samples from BEAT_v2 (total: {beat_v2_size})")
    
    # Create selected samples dataset for BEAT_v2 (only loads selected samples)
    beat_v2_dataset = SelectedSamplesDataset(beat_v2_base_dataset, beat_v2_indices, mean, std)
    
    # Sample from semi_synthetic - need to find indices in MixedDataset that correspond to semi_synthetic samples
    # For simplicity, we'll sample from the entire mixed dataset
    semi_synthetic_size = len(semi_synthetic_base_dataset)
    if num_samples_per_dataset > semi_synthetic_size:
        print(f"Warning: Requested {num_samples_per_dataset} samples but semi_synthetic only has {semi_synthetic_size}. Using all samples.")
        semi_synthetic_indices = list(range(semi_synthetic_size))
    else:
        if args.sample_strategy == 'diverse':
            spacing = max(1, semi_synthetic_size // num_samples_per_dataset)
            max_start = max(0, semi_synthetic_size - num_samples_per_dataset * spacing)
            start_idx = random.randint(0, max_start) if max_start > 0 else 0
            semi_synthetic_indices = [start_idx + i * spacing for i in range(num_samples_per_dataset)]
            semi_synthetic_indices = [idx for idx in semi_synthetic_indices if idx < semi_synthetic_size]
            # Ensure we have enough samples
            while len(semi_synthetic_indices) < num_samples_per_dataset and len(semi_synthetic_indices) < semi_synthetic_size:
                new_idx = random.randint(0, semi_synthetic_size - 1)
                if new_idx not in semi_synthetic_indices:
                    semi_synthetic_indices.append(new_idx)
            semi_synthetic_indices = sorted(semi_synthetic_indices[:num_samples_per_dataset])
        elif args.sample_strategy == 'sequential':
            semi_synthetic_indices = list(range(num_samples_per_dataset))
        else:  # random
            semi_synthetic_indices = sorted(random.sample(range(semi_synthetic_size), num_samples_per_dataset))
    
    print(f"Selected {len(semi_synthetic_indices)} samples from semi_synthetic_v1_segments (total: {semi_synthetic_size})")
    
    # Create selected samples dataset for semi_synthetic (only loads selected samples)
    semi_synthetic_dataset = SelectedSamplesDataset(semi_synthetic_base_dataset, semi_synthetic_indices, mean, std)
    
    print(f"\nSampling configuration:")
    print(f"  - BEAT_v2: {len(beat_v2_indices)} samples")
    print(f"  - semi_synthetic_v1_segments: {len(semi_synthetic_indices)} samples")
    print(f"  - Total: {len(beat_v2_indices) + len(semi_synthetic_indices)} samples")
    
    # Load model
    model_dir = pjoin(args.checkpoints_dir, 'mixed', args.name, 'model')
    checkpoint_path = os.path.join(model_dir, 'latest.tar')
    
    print(f"Loading model from {checkpoint_path}")
    ae = AE_models[args.model](input_width=dim_pose)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    ae.load_state_dict(checkpoint['ae'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae.eval()
    ae.to(device)
    
    # Output directory
    output_dir = pjoin(args.checkpoints_dir, 'mixed', args.name, 'test_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process BEAT_v2 samples
    sample_info_list = []
    sample_count = 0
    
    # Find npz file paths for BEAT_v2 (for audio matching)
    # Note: BEAT_v2Dataset uses shuffle, so we need to track the mapping
    beat_v2_npz_files = []
    for root, dirs, files in os.walk(beat_v2_root):
        for file in files:
            if file.endswith('.npz') and not file.endswith('_whisper_features.npy'):
                beat_v2_npz_files.append(os.path.join(root, file))
    beat_v2_npz_files.sort()
    
    # Process only selected BEAT_v2 samples (dataset only contains selected samples)
    print(f"Processing {len(beat_v2_dataset)} selected BEAT_v2 samples...")
    with torch.no_grad():
        for dataset_idx in tqdm(range(len(beat_v2_dataset)), desc="Processing BEAT_v2"):
            # Get the sample from selected dataset (only contains selected samples)
            motion_normalized = beat_v2_dataset[dataset_idx]  # Already normalized
            motion_tensor = torch.from_numpy(motion_normalized).unsqueeze(0).to(device).float()
            
            # Denormalize original motion
            original = motion_normalized * std + mean
            
            # Reconstruct
            reconstructed_tensor = ae(motion_tensor)
            reconstructed = reconstructed_tensor.detach().cpu().numpy()[0] * std + mean
            
            # Get the actual index in base dataset
            actual_idx = beat_v2_indices[dataset_idx]
            # Try to find corresponding npz file (may not be accurate due to shuffle)
            npz_path = None
            if actual_idx < len(beat_v2_npz_files):
                npz_path = beat_v2_npz_files[actual_idx]
            
            save_path = pjoin(output_dir, f'beat_v2_sample_{sample_count:04d}.npz')
            np.savez(save_path,
                    qpos_original=original,
                    qpos_reconstructed=reconstructed,
                    original=original,
                    reconstructed=reconstructed,
                    dataset_type='beat_v2',
                    sample_id=sample_count)
            
            sample_info_list.append({
                'sample_id': sample_count,
                'dataset_type': 'beat_v2',
                'source_npz_path': npz_path,
                'save_path': save_path
            })
            
            sample_count += 1
    
    # Process semi_synthetic samples
    # Find all semi_synthetic motion files (for metadata and audio matching)
    semi_synthetic_npz_files = []
    for root, dirs, files in os.walk(semi_synthetic_root):
        for file in files:
            if file.endswith('_motion.npz'):
                semi_synthetic_npz_files.append(os.path.join(root, file))
    semi_synthetic_npz_files.sort()
    
    # Process only selected semi_synthetic samples (dataset only contains selected samples)
    print(f"Processing {len(semi_synthetic_dataset)} selected semi_synthetic samples...")
    with torch.no_grad():
        for dataset_idx in tqdm(range(len(semi_synthetic_dataset)), desc="Processing semi_synthetic"):
            # Get the sample from selected dataset (only contains selected samples)
            motion_normalized = semi_synthetic_dataset[dataset_idx]  # Already normalized
            motion_tensor = torch.from_numpy(motion_normalized).unsqueeze(0).to(device).float()
            
            # Denormalize original motion
            original = motion_normalized * std + mean
            
            # Reconstruct
            reconstructed_tensor = ae(motion_tensor)
            reconstructed = reconstructed_tensor.detach().cpu().numpy()[0] * std + mean
            
            # Get the actual index in base dataset to find npz file
            actual_idx = semi_synthetic_indices[dataset_idx]
            npz_path = None
            if actual_idx < len(semi_synthetic_npz_files):
                npz_path = semi_synthetic_npz_files[actual_idx]
            
            # Load metadata
            metadata = {}
            if npz_path:
                metadata_path = npz_path.replace('_motion.npz', '_metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
            
            save_path = pjoin(output_dir, f'semi_synthetic_sample_{sample_count:04d}.npz')
            np.savez(save_path,
                    qpos_original=original,
                    qpos_reconstructed=reconstructed,
                    original=original,
                    reconstructed=reconstructed,
                    dataset_type='semi_synthetic',
                    sample_id=sample_count)
            
            sample_info_list.append({
                'sample_id': sample_count,
                'dataset_type': 'semi_synthetic',
                'source_npz_path': npz_path,
                'save_path': save_path,
                'source_file': metadata.get('source_file', ''),
                'segment_start_time': metadata.get('segment_start_time', 0.0),
                'segment_end_time': metadata.get('segment_end_time', 5.0)
            })
            
            sample_count += 1
    
    print(f"\nSaved {sample_count} samples to {output_dir}")
    
    # Generate videos with audio
    if args.generate_videos:
        print(f"\nGenerating comparison videos with audio...")
        video_output_dir = pjoin(output_dir, 'videos')
        os.makedirs(video_output_dir, exist_ok=True)
        
        temp_dir = tempfile.mkdtemp(prefix="test_AE_videos_")
        try:
            for sample_info in tqdm(sample_info_list[:args.num_video_samples * 2], desc="Generating videos"):
                sample_id = sample_info['sample_id']
                dataset_type = sample_info['dataset_type']
                npz_path = sample_info['save_path']
                
                # Find audio path (for semi_synthetic, extract segment from original audio)
                audio_path = find_audio_path_for_sample(sample_info, beat_v2_root, semi_synthetic_root, temp_dir, beat_v2_npz_files)
                
                # Save original and reconstructed to temporary npz files
                data = np.load(npz_path)
                original_npz = os.path.join(temp_dir, f"original_{sample_id}.npz")
                reconstructed_npz = os.path.join(temp_dir, f"reconstructed_{sample_id}.npz")
                
                np.savez(original_npz, qpos=data['qpos_original'])
                np.savez(reconstructed_npz, qpos=data['qpos_reconstructed'])
                
                # Render videos
                original_video = os.path.join(temp_dir, f"original_{sample_id}.mp4")
                reconstructed_video = os.path.join(temp_dir, f"reconstructed_{sample_id}.mp4")
                
                dataset_label = "BEAT_v2" if dataset_type == 'beat_v2' else "Semi-Synthetic"
                vis_npz_motion(original_npz, original_video, args.robot_type, args.rate_limit, args.motion_fps, f"Original ({dataset_label})")
                vis_npz_motion(reconstructed_npz, reconstructed_video, args.robot_type, args.rate_limit, args.motion_fps, f"Reconstructed ({dataset_label})")
                
                # Add text labels
                original_video_labeled = os.path.join(temp_dir, f"original_labeled_{sample_id}.mp4")
                reconstructed_video_labeled = os.path.join(temp_dir, f"reconstructed_labeled_{sample_id}.mp4")
                
                add_text_to_video(original_video, original_video_labeled, f"Original ({dataset_label})", position='top-left')
                add_text_to_video(reconstructed_video, reconstructed_video_labeled, f"VAE Reconstructed ({dataset_label})", position='top-left')
                
                # Concatenate horizontally
                comparison_video = pjoin(video_output_dir, f'{dataset_type}_sample_{sample_id:04d}_comparison.mp4')
                concatenate_videos_horizontally(original_video_labeled, reconstructed_video_labeled, comparison_video)
                
                # Combine with audio
                if audio_path:
                    final_video = pjoin(video_output_dir, f'{dataset_type}_sample_{sample_id:04d}_with_audio.mp4')
                    combine_video_with_audio(comparison_video, audio_path, final_video)
                    print(f"  Added audio to {os.path.basename(final_video)}")
            
            print(f"\nGenerated {len(sample_info_list[:args.num_video_samples * 2])} comparison videos in {video_output_dir}")
        finally:
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='AE')
    parser.add_argument('--model', type=str, default='AE_Model')
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_samples_per_dataset', type=int, default=10, help='Number of samples per dataset (BEAT_v2 and semi_synthetic)')
    parser.add_argument('--sample_strategy', type=str, default='diverse', choices=['sequential', 'diverse', 'random'])
    
    parser.add_argument('--generate_videos', action='store_true')
    parser.add_argument('--num_video_samples', type=int, default=10, help='Number of video samples per dataset')
    parser.add_argument('--robot_type', type=str, default='g1_branco')
    parser.add_argument('--motion_fps', type=int, default=60)
    parser.add_argument('--rate_limit', action='store_true')
    
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    
    arg = parser.parse_args()
    main(arg)

