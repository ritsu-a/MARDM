import os
os.environ["MUJOCO_GL"] = "egl"

from os.path import join as pjoin
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from models.AE import AE_models
from utils.datasets import BEAT_v2Dataset, AEDataset, Text2MotionDataset
import argparse
from tqdm import tqdm
import tempfile
import shutil
import imageio
from skimage.transform import resize
from PIL import Image, ImageDraw, ImageFont
from general_motion_retargeting import RobotMotionViewer
import time

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
    if args.dataset_name == "beat_v2":
        # BEAT_v2 dataset
        data_root = '/root/workspace/MARDM/data/BEAT_v2'
        mean = np.load(pjoin(data_root, 'Mean.npy'))
        std = np.load(pjoin(data_root, 'Std.npy'))
        dim_pose = mean.shape[0]
        joints_num = dim_pose
        
        # Use validation split for testing
        test_dataset = BEAT_v2Dataset(mean, std, data_root, args.window_size, split='val')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False, 
                                num_workers=args.num_workers, shuffle=False, pin_memory=True)
        
    elif args.dataset_name == "t2m":
        data_root = f'{args.dataset_dir}/HumanML3D/'
        joints_num = 22
        dim_pose = 67
        motion_dir = pjoin(data_root, 'new_joint_vecs')
        text_dir = pjoin(data_root, 'texts')
        max_motion_length = 196
        mean = np.load(pjoin(data_root, 'Mean.npy'))
        std = np.load(pjoin(data_root, 'Std.npy'))
        eval_mean = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_mean.npy')
        eval_std = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_std.npy')
        split_file = pjoin(data_root, 'test.txt')
        test_dataset = Text2MotionDataset(eval_mean, eval_std, split_file, args.dataset_name, motion_dir, text_dir,
                                          4, max_motion_length, 20, evaluation=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False, 
                                num_workers=args.num_workers, shuffle=False)
    else:
        data_root = f'{args.dataset_dir}/KIT-ML/'
        joints_num = 21
        dim_pose = 64
        motion_dir = pjoin(data_root, 'new_joint_vecs')
        text_dir = pjoin(data_root, 'texts')
        max_motion_length = 196
        mean = np.load(pjoin(data_root, 'Mean.npy'))
        std = np.load(pjoin(data_root, 'Std.npy'))
        eval_mean = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_mean.npy')
        eval_std = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_std.npy')
        split_file = pjoin(data_root, 'test.txt')
        test_dataset = Text2MotionDataset(eval_mean, eval_std, split_file, args.dataset_name, motion_dir, text_dir,
                                          4, max_motion_length, 20, evaluation=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False, 
                                num_workers=args.num_workers, shuffle=False)
    
    #################################################################################
    #                                      Models                                   #
    #################################################################################
    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')
    checkpoint_path = os.path.join(model_dir, 'latest.tar' if args.dataset_name == 't2m' or args.dataset_name == 'beat_v2' else 'net_best_fid.tar')
    
    print(f"Loading model from {checkpoint_path}")
    ae = AE_models[args.model](input_width=dim_pose)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    ae.load_state_dict(checkpoint['ae'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae.eval()
    ae.to(device)
    
    #################################################################################
    #                                  Save Results                                 #
    #################################################################################
    output_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'test_results')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving results to {output_dir}")
    print(f"Processing {args.num_samples} samples...")
    
    # For BEAT_v2, select sample indices based on strategy to ensure diversity
    if args.dataset_name == "beat_v2":
        dataset_size = len(test_dataset)
        if args.num_samples > dataset_size:
            print(f"Warning: Requested {args.num_samples} samples but dataset only has {dataset_size}. Using all samples.")
            selected_indices = list(range(dataset_size))
        else:
            if args.sample_strategy == 'sequential':
                # Sequential sampling (original behavior)
                selected_indices = list(range(args.num_samples))
            elif args.sample_strategy == 'diverse':
                # Spaced sampling to ensure different motions
                spacing = max(1, dataset_size // args.num_samples)
                # Start from random position and sample with spacing
                max_start = max(0, dataset_size - args.num_samples * spacing)
                start_idx = random.randint(0, max_start) if max_start > 0 else 0
                selected_indices = [start_idx + i * spacing for i in range(args.num_samples)]
                selected_indices = [idx for idx in selected_indices if idx < dataset_size]
                # Ensure we have enough samples
                while len(selected_indices) < args.num_samples and len(selected_indices) < dataset_size:
                    # Add random samples that aren't already selected
                    new_idx = random.randint(0, dataset_size - 1)
                    if new_idx not in selected_indices:
                        selected_indices.append(new_idx)
                selected_indices = sorted(selected_indices[:args.num_samples])
            else:  # random
                # Completely random sampling
                selected_indices = sorted(random.sample(range(dataset_size), args.num_samples))
        print(f"Selected {len(selected_indices)} samples using '{args.sample_strategy}' strategy from {dataset_size} total samples")
    else:
        selected_indices = None
    
    sample_count = 0
    processed_indices = set()
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Processing")):
            if args.dataset_name == "beat_v2":
                # BEAT_v2: batch_data is just motion
                motions = batch_data.detach().to(device).float()
                # Denormalize original motion
                original_motions = motions.detach().cpu().numpy() * std + mean
            else:
                # t2m/kit: batch_data contains (caption, motion, m_length)
                _, motions, m_lengths = batch_data
                motions = motions.to(device).float()
                # Denormalize original motion
                original_motions = test_dataset.inv_transform(motions.detach().cpu().numpy(), mean, std)
            
            # Reconstruct through VAE
            reconstructed_motions = ae(motions)
            
            # Denormalize reconstructed motion
            if args.dataset_name == "beat_v2":
                reconstructed_motions_denorm = reconstructed_motions.detach().cpu().numpy() * std + mean
            else:
                reconstructed_motions_denorm = test_dataset.inv_transform(reconstructed_motions.detach().cpu().numpy(), mean, std)
            
            batch_size = motions.shape[0]
            
            # Calculate global sample index
            global_start_idx = batch_idx * args.batch_size
            
            for i in range(batch_size):
                if sample_count >= args.num_samples:
                    break
                
                global_idx = global_start_idx + i
                
                # For BEAT_v2, only process selected indices
                if args.dataset_name == "beat_v2" and selected_indices is not None:
                    if global_idx not in selected_indices:
                        continue
                    if global_idx in processed_indices:
                        continue
                    processed_indices.add(global_idx)
                
                # Get original and reconstructed motion
                if args.dataset_name == "beat_v2":
                    original = original_motions[i]  # (window_size, dim)
                    reconstructed = reconstructed_motions_denorm[i]  # (window_size, dim)
                else:
                    # For t2m/kit, use actual motion length
                    m_len = m_lengths[i].item()
                    original = original_motions[i, :m_len]  # (m_len, dim)
                    reconstructed = reconstructed_motions_denorm[i, :m_len]  # (m_len, dim)
                
                # Save as npz
                save_path = pjoin(output_dir, f'sample_{sample_count:04d}.npz')
                if args.dataset_name == "beat_v2":
                    # For BEAT_v2, save as qpos format for compatibility with visualization tools
                    np.savez(save_path,
                            qpos_original=original,
                            qpos_reconstructed=reconstructed,
                            original=original,
                            reconstructed=reconstructed,
                            dataset_name=args.dataset_name,
                            sample_id=sample_count)
                else:
                    np.savez(save_path,
                            original=original,
                            reconstructed=reconstructed,
                            dataset_name=args.dataset_name,
                            sample_id=sample_count)
                
                sample_count += 1
            
            if sample_count >= args.num_samples:
                break
    
    print(f"\nSaved {sample_count} samples to {output_dir}")
    print(f"Each npz file contains:")
    print(f"  - 'original': original motion data (before VAE)")
    print(f"  - 'reconstructed': reconstructed motion data (after VAE)")
    print(f"  - 'dataset_name': name of the dataset")
    print(f"  - 'sample_id': sample index")
    
    #################################################################################
    #                              Generate Comparison Videos                       #
    #################################################################################
    if args.generate_videos:
        print(f"\nGenerating comparison videos...")
        video_output_dir = pjoin(output_dir, 'videos')
        os.makedirs(video_output_dir, exist_ok=True)
        
        temp_dir = tempfile.mkdtemp(prefix="test_AE_videos_")
        try:
            for sample_id in tqdm(range(min(sample_count, args.num_video_samples)), desc="Generating videos"):
                npz_path = pjoin(output_dir, f'sample_{sample_id:04d}.npz')
                
                # Save original and reconstructed to temporary npz files
                data = np.load(npz_path)
                original_npz = os.path.join(temp_dir, f"original_{sample_id}.npz")
                reconstructed_npz = os.path.join(temp_dir, f"reconstructed_{sample_id}.npz")
                
                if 'qpos_original' in data and 'qpos_reconstructed' in data:
                    np.savez(original_npz, qpos=data['qpos_original'])
                    np.savez(reconstructed_npz, qpos=data['qpos_reconstructed'])
                elif 'original' in data and 'reconstructed' in data:
                    np.savez(original_npz, qpos=data['original'])
                    np.savez(reconstructed_npz, qpos=data['reconstructed'])
                else:
                    print(f"Warning: Skipping sample {sample_id}, missing required keys")
                    continue
                
                # Render videos
                original_video = os.path.join(temp_dir, f"original_{sample_id}.mp4")
                reconstructed_video = os.path.join(temp_dir, f"reconstructed_{sample_id}.mp4")
                
                vis_npz_motion(original_npz, original_video, args.robot_type, args.rate_limit, args.motion_fps, "Original")
                vis_npz_motion(reconstructed_npz, reconstructed_video, args.robot_type, args.rate_limit, args.motion_fps, "Reconstructed")
                
                # Add text labels
                original_video_labeled = os.path.join(temp_dir, f"original_labeled_{sample_id}.mp4")
                reconstructed_video_labeled = os.path.join(temp_dir, f"reconstructed_labeled_{sample_id}.mp4")
                
                add_text_to_video(original_video, original_video_labeled, "Original", position='top-left')
                add_text_to_video(reconstructed_video, reconstructed_video_labeled, "VAE Reconstructed", position='top-left')
                
                # Concatenate horizontally
                final_video = pjoin(video_output_dir, f'sample_{sample_id:04d}_comparison.mp4')
                concatenate_videos_horizontally(original_video_labeled, reconstructed_video_labeled, final_video)
            
            print(f"\nGenerated {min(sample_count, args.num_video_samples)} comparison videos in {video_output_dir}")
        finally:
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='AE')
    parser.add_argument('--model', type=str, default='AE_Model')
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument('--dataset_name', type=str, default='beat_v2')
    parser.add_argument('--window_size', type=int, default=64, help='Window size for BEAT_v2 dataset')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for testing')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to save')
    parser.add_argument('--sample_strategy', type=str, default='diverse', 
                       choices=['sequential', 'diverse', 'random'],
                       help='Sampling strategy: sequential (order), diverse (spaced), random')
    
    # Video generation arguments
    parser.add_argument('--generate_videos', action='store_true', help='Generate comparison videos after saving npz files')
    parser.add_argument('--num_video_samples', type=int, default=10, help='Number of samples to generate videos for')
    parser.add_argument('--robot_type', type=str, default='g1_branco', 
                       choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", 
                               "fourier_n1", "engineai_pm01", "g1_branco", "g1_brainco"],
                       help='Robot type for visualization')
    parser.add_argument('--motion_fps', type=int, default=60, help='Motion FPS for video generation')
    parser.add_argument('--rate_limit', action='store_true', help='Limit rendering rate')
    
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    
    arg = parser.parse_args()
    main(arg)

