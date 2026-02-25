import os
from os.path import join as pjoin
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from models.AE import AE_models
from utils.evaluators import Evaluators
from utils.datasets import G1ML3DText2MotionDataset, BeatV2Text2MotionDataset, BeatSegmentDataset, collate_fn
from utils.eval_utils import evaluation_ae
import warnings
warnings.filterwarnings('ignore')
import argparse
import subprocess
import tempfile
import shutil
import imageio
from skimage.transform import resize
from tqdm import tqdm

def vis_npz_motion(motion_npz_path, output_path, robot_type="g1_brainco", rate_limit=False, motion_fps=60):
    """渲染单个npz文件为视频"""
    vis_script = "/root/workspace/MARDM/external/GMR/scripts/vis_npz_motion.py"
    cmd = [
        "python", vis_script,
        "--npz_path", motion_npz_path,
        "--video_path", output_path,
        "--robot", robot_type,
        "--motion_fps", str(motion_fps)
    ]
    if rate_limit:
        cmd.append("--rate_limit")
    
    subprocess.run(cmd, check=True, capture_output=True)


def concatenate_videos_horizontally(video1_path, video2_path, output_path):
    """将两个视频左右拼接"""
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
    
    print(f"Concatenating videos horizontally...")
    pbar = tqdm(desc="Concatenating", leave=False)
    
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
            pbar.update(1)
    except (StopIteration, IndexError):
        pass
    
    pbar.close()
    reader1.close()
    reader2.close()
    writer.close()
    print(f"Concatenated {frame_count} frames")


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
    #                                    Eval Data                                  #
    #################################################################################
    data_root = args.dataset_dir
    max_motion_length = args.max_motion_length
    use_beat = getattr(args, 'dataset_type', 'g1ml3d') == 'beat'
    if use_beat:
        motion_dir = data_root  # BEAT_v2: 运动在 data_root/{id}.npz，id 如 1/1_wayne_0_100_100
        text_dir = data_root
    else:
        motion_dir = pjoin(data_root, 'joints_npz')
        text_dir = pjoin(data_root, 'texts')
    
    # Load mean and std
    mean_path = pjoin(data_root, 'Mean.npy')
    std_path = pjoin(data_root, 'Std.npy')
    
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        raise FileNotFoundError(f"Mean.npy or Std.npy not found in {data_root}. Please run training first to generate these files.")
    
    mean = np.load(mean_path)
    std = np.load(std_path)
    
    # Use same mean/std for evaluation
    eval_mean = mean
    eval_std = std
    
    use_segment = use_beat and getattr(args, 'use_segment', False)
    if use_segment:
        segment_dir = pjoin(data_root, 'segment')
        segment_split_name = os.path.basename(args.split_file) if 'segment_' in args.split_file else 'segment_test.txt'
        segment_split = pjoin(segment_dir, segment_split_name)
        segment_base = os.path.splitext(segment_split_name)[0]
        merged_npz = pjoin(segment_dir, segment_base + '.npz')
        if not os.path.exists(segment_split) and not os.path.exists(merged_npz):
            raise FileNotFoundError(
                f"Segment split not found: {segment_split} 或 {merged_npz}\n"
                f"请先运行 scripts/beat_segment_to_npz.py 生成 segment 目录及 segment_test.npz / segment_test.txt"
            )
        split_file = segment_split
    else:
        if os.path.isabs(args.split_file) and os.path.exists(args.split_file):
            split_file = args.split_file
        else:
            split_file = pjoin(data_root, args.split_file)
        if not os.path.exists(split_file):
            fallback = pjoin(data_root, 'test.txt') if 'val' in args.split_file else pjoin(data_root, 'val.txt')
            if os.path.exists(fallback):
                split_file = fallback
                print(f"Using split file: {split_file}")
            else:
                raise FileNotFoundError(
                    f"Split file not found: {split_file}\n"
                    f"请在数据集目录下放置 val.txt 或 test.txt，每行一个样本 id（与 motion/text 文件名对应）。"
                )

    if use_beat:
        if use_segment:
            eval_dataset = BeatSegmentDataset(segment_dir, segment_split, eval_mean, eval_std,
                                              max_motion_length, 20, evaluation=True)
        else:
            eval_dataset = BeatV2Text2MotionDataset(eval_mean, eval_std, split_file, data_root,
                                                    4, max_motion_length, 20, evaluation=True)
    else:
        eval_dataset = G1ML3DText2MotionDataset(eval_mean, eval_std, split_file, 'g1ml3d', motion_dir, text_dir,
                                              4, max_motion_length, 20, evaluation=True)
    if len(eval_dataset) == 0:
        raise RuntimeError("Eval dataset is empty. Check dataset_dir, split_file, and motion/text paths (BEAT_v2: use --dataset_type beat).")
    eval_loader = DataLoader(eval_dataset, batch_size=32, num_workers=args.num_workers, drop_last=True,
                            collate_fn=collate_fn, shuffle=True)
    #################################################################################
    #                                      Models                                   #
    #################################################################################
    model_dir = getattr(args, 'model_dir', None) or pjoin(args.checkpoints_dir, 'g1ml3d', args.name, 'model')

    # Get motion dimension from mean shape
    dim_pose = mean.shape[0]
    # Estimate joints_num (adjust based on your data)
    joints_num = args.joints_num if hasattr(args, 'joints_num') else 22
    
    ae = AE_models[args.model](input_width=dim_pose)
    
    # Load checkpoint
    checkpoint_path = pjoin(model_dir, args.checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    ae.load_state_dict(checkpoint['ae'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Note: Evaluators expects dim_pose=67 (t2m) or 64 (kit), but G1ML3D has different dimensions
    # Skip evaluator for G1ML3D as it requires retraining with G1ML3D-specific dimensions
    eval_wrapper = None
    print("Note: Skipping evaluator metrics for G1ML3D (dimension mismatch). Using simplified evaluation.")
    #################################################################################
    #                                  Evaluation Loop                              #
    #################################################################################
    out_dir = pjoin(args.checkpoints_dir, 'g1ml3d', args.name, 'eval')
    os.makedirs(out_dir, exist_ok=True)
    f = open(pjoin(out_dir, 'eval.log'), 'w')
    
    # Create directory for visualization outputs
    vis_dir = pjoin(out_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory for rendering: {temp_dir}")

    ae.eval()
    ae.to(device)
    
    # Collect samples for visualization
    vis_samples = []
    sample_count = 0

    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    mae = []
    repeat_time = args.repeat_time
    for i in range(repeat_time):
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, mpjpe = 1000, 0, 0, 0, 0, 100, 100
        
        if eval_wrapper is not None:
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, mpjpe, writer = evaluation_ae(
                checkpoint_path, eval_loader, ae, None, i, device=device, num_joint=joints_num, best_fid=best_fid,
                best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                train_mean=mean, train_std=std, best_matching=best_matching, eval_wrapper=eval_wrapper,
                save=False, draw=False)
            
            # Collect samples for visualization (only in first iteration)
            if i == 0 and sample_count < args.num_vis_samples:
                print(f"Collecting samples for visualization...")
                with torch.no_grad():
                    for batch_idx, batch in enumerate(eval_loader):
                        if sample_count >= args.num_vis_samples:
                            break
                        if len(batch) == 8:
                            word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, whisper_feat = batch
                        else:
                            word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch
                            whisper_feat = None
                        motion = motion.to(device)
                        pred_motion = ae(motion)
                        
                        batch_size = motion.shape[0]
                        for j in range(min(batch_size, args.num_vis_samples - sample_count)):
                            gt_motion = motion[j].detach().cpu().numpy()
                            pred_motion_np = pred_motion[j].detach().cpu().numpy()
                            m_len = m_length[j].item()
                            
                            # Align lengths for visualization
                            min_len = min(pred_motion_np.shape[0], gt_motion.shape[0], m_len)
                            gt_motion_aligned = gt_motion[:min_len]
                            pred_motion_aligned = pred_motion_np[:min_len]
                            
                            # Denormalize motions
                            gt_motion_denorm = eval_dataset.inv_transform(gt_motion_aligned)
                            pred_motion_denorm = eval_dataset.inv_transform(pred_motion_aligned)
                            
                            # Get caption text
                            cap_text = caption[j] if isinstance(caption, list) else str(caption[j])
                            
                            vis_samples.append({
                                'gt': gt_motion_denorm,
                                'pred': pred_motion_denorm,
                                'name': f'sample_{sample_count:03d}',
                                'caption': cap_text
                            })
                            sample_count += 1
                            if sample_count >= args.num_vis_samples:
                                break
        else:
            # Simplified evaluation without evaluator
            print(f"Running simplified evaluation (iteration {i+1}/{repeat_time})...")
            ae.eval()
            total_loss = 0
            total_samples = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(eval_loader):
                    if len(batch) == 8:
                        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, whisper_feat = batch
                    else:
                        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch
                        whisper_feat = None
                    motion = motion.to(device)
                    pred_motion = ae(motion)
                    
                    # Align lengths: VAE may output different length due to downsampling
                    min_len = min(pred_motion.shape[1], motion.shape[1])
                    pred_motion_aligned = pred_motion[:, :min_len, :]
                    motion_aligned = motion[:, :min_len, :]
                    
                    loss = torch.nn.functional.mse_loss(pred_motion_aligned, motion_aligned)
                    total_loss += loss.item() * motion.shape[0]
                    total_samples += motion.shape[0]
                    
                    # Collect samples for visualization (only in first iteration)
                    if i == 0 and sample_count < args.num_vis_samples:
                        batch_size = motion.shape[0]
                        for j in range(min(batch_size, args.num_vis_samples - sample_count)):
                            gt_motion = motion[j].detach().cpu().numpy()
                            pred_motion_np = pred_motion[j].detach().cpu().numpy()
                            m_len = m_length[j].item()
                            
                            # Denormalize motions
                            gt_motion_denorm = eval_dataset.inv_transform(gt_motion[:m_len])
                            pred_motion_denorm = eval_dataset.inv_transform(pred_motion_np[:m_len])
                            
                            # Get caption text
                            cap_text = caption[j] if isinstance(caption, list) else str(caption[j])
                            
                            vis_samples.append({
                                'gt': gt_motion_denorm,
                                'pred': pred_motion_denorm,
                                'name': f'sample_{sample_count:03d}',
                                'caption': cap_text
                            })
                            sample_count += 1
                            if sample_count >= args.num_vis_samples:
                                break
            avg_loss = total_loss / total_samples
            print(f"Average reconstruction loss: {avg_loss:.6f}")
            mpjpe = avg_loss  # Use reconstruction loss as proxy
            best_fid = avg_loss * 1000  # Scale for compatibility
            best_div = 0  # Not available without evaluator
            best_top1 = 0
            best_top2 = 0
            best_top3 = 0
            best_matching = 100
        
        fid.append(best_fid)
        div.append(best_div)
        top1.append(best_top1)
        top2.append(best_top2)
        top3.append(best_top3)
        matching.append(best_matching)
        mae.append(mpjpe)

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    mae = np.array(mae)

    print(f'final result')
    print(f'final result', file=f, flush=True)

    if eval_wrapper is not None:
        msg_final = f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tMAE:{np.mean(mae):.3f}, conf.{np.std(mae) * 1.96 / np.sqrt(repeat_time):.3f}\n\n"
    else:
        msg_final = f"\tReconstruction Loss (proxy for FID): {np.mean(fid):.6f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.6f}\n" \
                    f"\tMAE:{np.mean(mae):.6f}, conf.{np.std(mae) * 1.96 / np.sqrt(repeat_time):.6f}\n\n"

    print(msg_final)
    print(msg_final, file=f, flush=True)
    f.close()
    
    #################################################################################
    #                              Visualization                                    #
    #################################################################################
    if len(vis_samples) > 0:
        print(f"\nGenerating visualization videos for {len(vis_samples)} samples...")
        combined_videos = []
        
        try:
            for idx, sample in enumerate(vis_samples):
                print(f"\nProcessing sample {idx+1}/{len(vis_samples)}: {sample['name']}")
                
                # Save GT and Pred motions as npz files
                gt_npz_path = pjoin(temp_dir, f"{sample['name']}_gt.npz")
                pred_npz_path = pjoin(temp_dir, f"{sample['name']}_pred.npz")
                
                np.savez(gt_npz_path, qpos=sample['gt'])
                np.savez(pred_npz_path, qpos=sample['pred'])
                
                # Render videos
                gt_video_path = pjoin(temp_dir, f"{sample['name']}_gt.mp4")
                pred_video_path = pjoin(temp_dir, f"{sample['name']}_pred.mp4")
                combined_video_path = pjoin(temp_dir, f"{sample['name']}_combined.mp4")
                
                print(f"  Rendering GT motion...")
                vis_npz_motion(gt_npz_path, gt_video_path, args.vis_robot, False, args.motion_fps)
                
                print(f"  Rendering Pred motion...")
                vis_npz_motion(pred_npz_path, pred_video_path, args.vis_robot, False, args.motion_fps)
                
                # Concatenate horizontally
                print(f"  Concatenating videos...")
                concatenate_videos_horizontally(gt_video_path, pred_video_path, combined_video_path)
                
                # Copy to final output directory
                final_video_path = pjoin(vis_dir, f"{sample['name']}_comparison.mp4")
                shutil.copy2(combined_video_path, final_video_path)
                combined_videos.append(final_video_path)
                
                print(f"  Saved to: {final_video_path}")
            
            # Create a combined video with all samples
            if len(combined_videos) > 1:
                final_combined_path = pjoin(vis_dir, 'all_samples_comparison.mp4')
                print(f"\nCreating final combined video...")
                
                # Use imageio to concatenate vertically
                readers = [imageio.get_reader(v) for v in combined_videos]
                fps = readers[0].get_meta_data()['fps']
                width, height = readers[0].get_meta_data()['size']
                
                writer = imageio.get_writer(final_combined_path, fps=fps)
                for reader in readers:
                    for frame in reader:
                        if frame.shape[0] != height or frame.shape[1] != width:
                            frame = resize(frame, (height, width), preserve_range=True, anti_aliasing=True).astype(frame.dtype)
                        writer.append_data(frame)
                    reader.close()
                writer.close()
                
                print(f"Final combined video saved to: {final_combined_path}")
            
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary directory: {temp_dir}")
        
        print(f"\nVisualization complete! Videos saved to: {vis_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='AE')
    parser.add_argument('--model', type=str, default='AE_Model')
    parser.add_argument('--dataset_type', type=str, default='g1ml3d', choices=('g1ml3d', 'beat'),
                        help='g1ml3d: joints_npz+texts 平铺；beat: BEAT_v2 嵌套目录+_whisper_features.txt')
    parser.add_argument('--use_segment', action='store_true',
                        help='BEAT 时从 dataset_dir/segment 读切好的片段（segment_test.npz 或 segment_test.txt）')
    parser.add_argument('--dataset_dir', type=str, default='./data/G1ML3D_v1',
                        help='Root directory of G1ML3D dataset')
    parser.add_argument('--split_file', type=str, default='test.txt',
                        help='Split file name (test.txt, val.txt, etc.)')
    parser.add_argument('--max_motion_length', type=int, default=196,
                        help='Maximum motion length for evaluation')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='AE 模型目录，默认 checkpoints_dir/g1ml3d/name/model；mixed 可用 ./checkpoints/mixed/ae/model')
    parser.add_argument('--checkpoint_name', type=str, default='latest.tar',
                        help='Checkpoint file name (latest.tar or net_best_fid.tar)')
    parser.add_argument('--joints_num', type=int, default=22,
                        help='Number of joints (adjust based on G1ML3D structure)')
    parser.add_argument('--repeat_time', type=int, default=20,
                        help='Number of evaluation runs for statistics')
    parser.add_argument('--num_vis_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--vis_robot', type=str, default='g1_brainco',
                        help='Robot type for visualization')
    parser.add_argument('--motion_fps', type=int, default=60,
                        help='Motion FPS for visualization')

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')

    arg = parser.parse_args()
    main(arg)
