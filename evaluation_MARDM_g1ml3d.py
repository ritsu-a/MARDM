"""
Evaluate G1ML3D MARDM (DDPM) checkpoint: generate motion from text, compute MSE vs GT, save npz and optional videos.
Checkpoint example: /root/workspace/MARDM/checkpoints/g1ml3d/MARDM_g1ml3d/model/checkpoint_epoch_600.tar
"""
import os
from os.path import join as pjoin
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from models.AE import AE_models
from models.MARDM import MARDM_models
from utils.datasets import G1ML3DText2MotionDataset, collate_fn
import argparse
import subprocess
import tempfile
import shutil
import imageio
from skimage.transform import resize
from tqdm import tqdm

# VAE downsampling factor (same as in train_MARDM_g1ml3d)
VAE_DOWNSAMPLE_FACTOR = 16

# Max length for prompt in filename to avoid too long paths
PROMPT_FILENAME_MAX_LEN = 80


def prompt_to_filename(prompt, max_len=PROMPT_FILENAME_MAX_LEN):
    """Turn text prompt into a safe filename (strip invalid chars, truncate)."""
    if not prompt or not isinstance(prompt, str):
        return "unnamed"
    # Replace invalid path chars
    for c in r'\/:*?"<>|':
        prompt = prompt.replace(c, "_")
    # Collapse spaces and underscores, strip
    s = "_".join(prompt.split()).strip("_")
    if not s:
        return "unnamed"
    # Truncate and ensure no trailing dot (Windows)
    s = s[:max_len].rstrip(".")
    return s or "unnamed"


def vis_npz_motion(motion_npz_path, output_path, robot_type="g1_brainco", rate_limit=False, motion_fps=60):
    """Render single npz motion to video."""
    vis_script = pjoin(os.path.dirname(__file__), "external", "GMR", "scripts", "vis_npz_motion.py")
    if not os.path.exists(vis_script):
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
    """Concatenate two videos side by side."""
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
    for frame1, frame2 in zip(reader1, reader2):
        if frame1.shape[0] != target_height:
            frame1 = resize(frame1, (target_height, width1), preserve_range=True, anti_aliasing=True).astype(frame1.dtype)
        if frame2.shape[0] != target_height:
            frame2 = resize(frame2, (target_height, width2), preserve_range=True, anti_aliasing=True).astype(frame2.dtype)
        combined_frame = np.hstack([frame1, frame2])
        writer.append_data(combined_frame)
    reader1.close()
    reader2.close()
    writer.close()


def main(args):
    torch.backends.cudnn.benchmark = False
    os.environ["OMP_NUM_THREADS"] = "1"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Paths
    data_root = args.dataset_dir
    motion_dir = pjoin(data_root, 'joints_npz')
    text_dir = pjoin(data_root, 'texts')
    mean_path = pjoin(data_root, 'Mean.npy')
    std_path = pjoin(data_root, 'Std.npy')
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        raise FileNotFoundError(f"Mean.npy or Std.npy not found in {data_root}.")
    mean = np.load(mean_path)
    std = np.load(std_path)

    split_file = pjoin(data_root, args.split_file)
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    # MARDM checkpoint: allow full path or name under model dir
    if os.path.isabs(args.checkpoint_name) and os.path.isfile(args.checkpoint_name):
        mardm_ckpt_path = args.checkpoint_name
    else:
        mardm_model_dir = pjoin(args.checkpoints_dir, 'g1ml3d', args.name, 'model')
        mardm_ckpt_path = pjoin(mardm_model_dir, args.checkpoint_name)
    if not os.path.isfile(mardm_ckpt_path):
        raise FileNotFoundError(f"MARDM checkpoint not found: {mardm_ckpt_path}")

    # AE checkpoint
    ae_ckpt_path = pjoin(args.checkpoints_dir, 'g1ml3d', args.ae_name, 'model', args.ae_checkpoint_name)
    if not os.path.isfile(ae_ckpt_path):
        raise FileNotFoundError(f"AE checkpoint not found: {ae_ckpt_path}")

    dim_pose = mean.shape[0]
    max_motion_length = args.max_motion_length

    # Datasets & loader
    eval_dataset = G1ML3DText2MotionDataset(
        mean, std, split_file, 'g1ml3d', motion_dir, text_dir,
        4, max_motion_length, 20, evaluation=True
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        drop_last=False, collate_fn=collate_fn, shuffle=False
    )

    # Models
    ae = AE_models[args.ae_model](input_width=dim_pose)
    ckpt_ae = torch.load(ae_ckpt_path, map_location='cpu')
    ae.load_state_dict(ckpt_ae['ae'])

    ema_mardm = MARDM_models[args.model](ae_dim=ae.output_emb_width, cond_mode='text')
    ckpt_mardm = torch.load(mardm_ckpt_path, map_location='cpu')
    if 'ema_mardm' not in ckpt_mardm:
        raise KeyError("Checkpoint must contain 'ema_mardm' key.")
    missing, unexpected = ema_mardm.load_state_dict(ckpt_mardm['ema_mardm'], strict=False)
    assert len(unexpected) == 0
    assert all(k.startswith('clip_model.') for k in missing)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae.eval()
    ae.to(device)
    ema_mardm.eval()
    ema_mardm.to(device)

    out_dir = pjoin(args.checkpoints_dir, 'g1ml3d', args.name, 'eval')
    os.makedirs(out_dir, exist_ok=True)
    npz_dir = pjoin(out_dir, 'generated_npz')
    os.makedirs(npz_dir, exist_ok=True)

    log_path = pjoin(out_dir, 'eval.log')
    f_log = open(log_path, 'w')

    total_mse = 0.0
    total_count = 0
    vis_samples = []
    sample_count = 0

    for i, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
        word_embeddings, pos_one_hots, caption, sent_len, motion_gt, m_length, _ = batch
        motion_gt = motion_gt.float().to(device)
        m_length = m_length.long().to(device)

        # Captions: batch may be tuple of strings
        if isinstance(caption, (list, tuple)):
            captions = list(caption)
        else:
            captions = [caption]

        # Latent length for G1ML3D VAE (16x downsampling); ensure at least 1 (tensor on device for lengths_to_mask)
        m_lens = (m_length // VAE_DOWNSAMPLE_FACTOR).long().to(device).clamp(min=1)

        with torch.no_grad():
            pred_latents = ema_mardm.generate(
                captions, m_lens, args.time_steps, args.cfg,
                temperature=args.temperature, hard_pseudo_reorder=args.hard_pseudo_reorder
            )
            pred_motions = ae.decode(pred_latents)
            pred_motions = pred_motions.detach().cpu().numpy()
            pred_motions = eval_dataset.inv_transform(pred_motions, mean, std)

        motion_gt_np = eval_dataset.inv_transform(motion_gt.detach().cpu().numpy(), mean, std)
        bs = motion_gt_np.shape[0]

        for j in range(bs):
            len_j = int(m_length[j].item())
            gt_j = motion_gt_np[j, :len_j]
            pred_j = pred_motions[j]
            min_len = min(gt_j.shape[0], pred_j.shape[0], len_j)
            if min_len <= 0:
                continue
            gt_j = gt_j[:min_len]
            pred_j = pred_j[:min_len]
            mse_j = np.mean((gt_j - pred_j) ** 2)
            total_mse += mse_j
            total_count += 1

            # Base name from text prompt (safe for filename)
            caption_j = captions[j] if j < len(captions) else ""
            prompt_base = prompt_to_filename(caption_j)
            base_name = f"{prompt_base}_b{i:04d}_j{j:02d}"

            # Save npz with prompt-based name
            npz_name = f"{base_name}.npz"
            np.savez(pjoin(npz_dir, npz_name), qpos=pred_j)

            # Collect samples for visualization
            if sample_count < args.num_vis_samples:
                vis_samples.append({
                    'name': base_name,
                    'gt': gt_j,
                    'pred': pred_j,
                    'caption': caption_j,
                })
                sample_count += 1

    avg_mse = total_mse / total_count if total_count else float('nan')
    msg = f"Evaluated {total_count} samples. Mean MSE: {avg_mse:.6f}\n"
    print(msg)
    f_log.write(msg)
    f_log.flush()

    # Visualization
    if vis_samples and args.num_vis_samples > 0:
        vis_dir = pjoin(out_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        temp_dir = tempfile.mkdtemp()
        try:
            for idx, sample in enumerate(vis_samples):
                # name already is prompt_base_bXXXX_jXX
                base = sample['name']
                gt_npz = pjoin(temp_dir, f"{base}_gt.npz")
                pred_npz = pjoin(temp_dir, f"{base}_pred.npz")
                np.savez(gt_npz, qpos=sample['gt'])
                np.savez(pred_npz, qpos=sample['pred'])
                gt_video = pjoin(temp_dir, f"{base}_gt.mp4")
                pred_video = pjoin(temp_dir, f"{base}_pred.mp4")
                combined_path = pjoin(temp_dir, f"{base}_combined.mp4")
                vis_npz_motion(gt_npz, gt_video, args.vis_robot, False, args.motion_fps)
                vis_npz_motion(pred_npz, pred_video, args.vis_robot, False, args.motion_fps)
                concatenate_videos_horizontally(gt_video, pred_video, combined_path)
                shutil.copy2(combined_path, pjoin(vis_dir, f"{base}_comparison.mp4"))
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Visualization error: {e}")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"Videos saved to {vis_dir}")

    # Render generated npz to videos (optional)
    if args.num_render_npz != 0:
        video_dir = pjoin(out_dir, 'generated_videos')
        os.makedirs(video_dir, exist_ok=True)
        npz_files = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
        if args.num_render_npz > 0:
            npz_files = npz_files[: args.num_render_npz]
        print(f"Rendering {len(npz_files)} npz to videos in {video_dir}...")
        for fname in tqdm(npz_files, desc="Render npz"):
            npz_path = pjoin(npz_dir, fname)
            video_path = pjoin(video_dir, fname.replace('.npz', '.mp4'))
            try:
                vis_npz_motion(npz_path, video_path, args.vis_robot, False, args.motion_fps)
            except Exception as e:
                print(f"  Skip {fname}: {e}")
        print(f"Generated videos saved to {video_dir}")

    f_log.close()
    print(f"Log saved to {log_path}")
    print(f"Generated npz saved to {npz_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='MARDM_g1ml3d')
    parser.add_argument('--ae_name', type=str, default='AE_g1ml3d')
    parser.add_argument('--ae_model', type=str, default='AE_Model')
    parser.add_argument('--model', type=str, default='MARDM-SiT-XL')
    parser.add_argument('--dataset_dir', type=str, default='./data/G1ML3D_v1')
    parser.add_argument('--split_file', type=str, default='val.txt')
    parser.add_argument('--max_motion_length', type=int, default=196)
    parser.add_argument('--checkpoint_name', type=str, default='checkpoint_epoch_600.tar',
                        help='Checkpoint file name or full path, e.g. checkpoint_epoch_600.tar')
    parser.add_argument('--ae_checkpoint_name', type=str, default='latest.tar')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--time_steps', type=int, default=18)
    parser.add_argument('--cfg', type=float, default=4.5)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--hard_pseudo_reorder', action='store_true')
    parser.add_argument('--num_vis_samples', type=int, default=5,
                        help='Number of GT vs Pred comparison videos')
    parser.add_argument('--num_render_npz', type=int, default=-1,
                        help='Render generated npz to video: -1=all, 0=none, N=first N')
    parser.add_argument('--vis_robot', type=str, default='g1_brainco')
    parser.add_argument('--motion_fps', type=int, default=60)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    arg = parser.parse_args()
    main(arg)
