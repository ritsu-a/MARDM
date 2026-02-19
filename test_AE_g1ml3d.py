import os
os.environ["MUJOCO_GL"] = "egl"

from os.path import join as pjoin
import torch
import numpy as np
import random
import json
from torch.utils.data import DataLoader
from models.AE import AE_models
from utils.datasets import G1ML3D_v1Dataset
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
            base_dataset: 基础数据集（G1ML3D_v1Dataset）
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

def vis_npz_motion(motion_npz_path, output_path, robot_type="g1_brainco", rate_limit=False, motion_fps=60, label_text=""):
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
    writer = imageio.get_writer(output_path, fps=fps)
    
    try:
        # Try to load a font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    for frame in reader:
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        
        # Calculate text position
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        if position == 'top-left':
            x, y = 10, 10
        elif position == 'top-right':
            x, y = img.width - text_width - 10, 10
        elif position == 'bottom-left':
            x, y = 10, img.height - text_height - 10
        elif position == 'bottom-right':
            x, y = img.width - text_width - 10, img.height - text_height - 10
        else:
            x, y = 10, 10
        
        # Draw text with background
        padding = 5
        draw.rectangle([x - padding, y - padding, x + text_width + padding, y + text_height + padding], 
                      fill=(0, 0, 0, 200))
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        
        writer.append_data(np.array(img))
    
    reader.close()
    writer.close()

def main(args):
    #################################################################################
    #                                      Seed                                     #
    #################################################################################
    torch.backends.cudnn.benchmark = False
    os.environ["OMP_NUM_THREADS"] = "1"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    #################################################################################
    #                                    Test Data                                  #
    #################################################################################
    g1ml3d_root = '/root/workspace/MARDM/data/G1ML3D_v1/joints_npz'
    
    mean = np.load(pjoin('/root/workspace/MARDM/data/G1ML3D_v1', 'Mean.npy'))
    std = np.load(pjoin('/root/workspace/MARDM/data/G1ML3D_v1', 'Std.npy'))
    dim_pose = mean.shape[0]
    
    # Load test dataset
    test_dataset = G1ML3D_v1Dataset(mean, std, g1ml3d_root, args.window_size, split='val')
    
    # Select samples based on strategy
    total_samples = len(test_dataset)
    if args.sample_strategy == 'sequential':
        # Sequential sampling
        step = max(1, total_samples // args.num_samples_per_dataset)
        selected_indices = list(range(0, total_samples, step))[:args.num_samples_per_dataset]
    elif args.sample_strategy == 'diverse':
        # Diverse sampling (spaced out)
        if args.num_samples_per_dataset >= total_samples:
            selected_indices = list(range(total_samples))
        else:
            step = total_samples // args.num_samples_per_dataset
            selected_indices = [i * step for i in range(args.num_samples_per_dataset)]
    else:  # random
        # Random sampling
        selected_indices = sorted(random.sample(range(total_samples), 
                                                min(args.num_samples_per_dataset, total_samples)))
    
    print(f"Selected {len(selected_indices)} samples from {total_samples} total samples")
    
    # Create selected dataset
    selected_dataset = SelectedSamplesDataset(test_dataset, selected_indices, mean, std)
    test_loader = DataLoader(selected_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    #################################################################################
    #                                      Models                                   #
    #################################################################################
    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')
    ae = AE_models[args.model](input_width=dim_pose)
    checkpoint_path = os.path.join(model_dir, 'latest.tar')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    ae.load_state_dict(checkpoint['ae'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae.eval()
    ae.to(device)
    
    #################################################################################
    #                                  Test Loop                                    #
    #################################################################################
    output_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'test_results')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Testing on {len(selected_dataset)} samples...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Testing")):
            motions = batch_data.detach().to(device).float()
            pred_motion = ae(motions)
            
            # Denormalize
            motions_denorm = motions.cpu().numpy()[0] * std + mean
            pred_motion_denorm = pred_motion.cpu().numpy()[0] * std + mean
            
            # Save results
            sample_idx = selected_indices[batch_idx]
            output_path = pjoin(output_dir, f'sample_{sample_idx:04d}.npz')
            np.savez(output_path,
                    original=motions_denorm,
                    reconstructed=pred_motion_denorm,
                    qpos_original=motions_denorm,
                    qpos_reconstructed=pred_motion_denorm)
    
    print(f"Test results saved to {output_dir}")
    
    #################################################################################
    #                              Generate Videos                                   #
    #################################################################################
    if args.generate_videos:
        print(f"\nGenerating comparison videos for {args.num_video_samples} samples...")
        video_output_dir = pjoin(output_dir, 'videos')
        os.makedirs(video_output_dir, exist_ok=True)
        
        num_videos = min(args.num_video_samples, len(selected_indices))
        selected_for_video = selected_indices[:num_videos]
        
        for idx, sample_idx in enumerate(tqdm(selected_for_video, desc="Generating videos")):
            npz_path = pjoin(output_dir, f'sample_{sample_idx:04d}.npz')
            
            # Original video
            orig_video_path = pjoin(video_output_dir, f'sample_{sample_idx:04d}_original.mp4')
            vis_npz_motion(npz_path, orig_video_path, args.robot_type, args.rate_limit, 
                          args.motion_fps, label_text="Original")
            
            # Reconstructed video
            recon_video_path = pjoin(video_output_dir, f'sample_{sample_idx:04d}_reconstructed.mp4')
            # Create temporary npz with reconstructed data
            data = np.load(npz_path)
            temp_npz = tempfile.NamedTemporaryFile(delete=False, suffix='.npz')
            np.savez(temp_npz.name, qpos=data['reconstructed'])
            temp_npz.close()
            
            vis_npz_motion(temp_npz.name, recon_video_path, args.robot_type, args.rate_limit,
                          args.motion_fps, label_text="Reconstructed")
            os.unlink(temp_npz.name)
            
            # Add text labels
            orig_labeled = pjoin(video_output_dir, f'sample_{sample_idx:04d}_original_labeled.mp4')
            recon_labeled = pjoin(video_output_dir, f'sample_{sample_idx:04d}_reconstructed_labeled.mp4')
            
            add_text_to_video(orig_video_path, orig_labeled, "Original", 'top-left')
            add_text_to_video(recon_video_path, recon_labeled, "Reconstructed", 'top-left')
            
            # Side-by-side comparison (optional)
            # This would require additional video processing
        
        print(f"Videos saved to {video_output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='AE', help='Name of the model')
    parser.add_argument('--model', type=str, default='AE_Model', help='Model type')
    parser.add_argument('--dataset_name', type=str, default='g1ml3d', help='Dataset name')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='Checkpoints directory')
    parser.add_argument('--window_size', type=int, default=180, help='Window size')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num_samples_per_dataset', type=int, default=10, 
                       help='Number of samples to test per dataset')
    parser.add_argument('--sample_strategy', type=str, default='diverse', 
                       choices=['sequential', 'diverse', 'random'],
                       help='Sampling strategy: sequential, diverse (spaced), or random')
    parser.add_argument('--generate_videos', action='store_true', 
                       help='Generate comparison videos')
    parser.add_argument('--num_video_samples', type=int, default=10,
                       help='Number of samples to generate videos for')
    parser.add_argument('--robot_type', type=str, default='g1_brainco',
                       help='Robot type for visualization')
    parser.add_argument('--motion_fps', type=int, default=60,
                       help='Motion frame rate')
    parser.add_argument('--rate_limit', action='store_true',
                       help='Rate limit visualization')
    
    args = parser.parse_args()
    main(args)
