"""
从训练集（BEAT segment_train）取 segment，逐段生成并每生成一段立即可视化 GT vs Pred。
用于快速查看在训练集上的重建/生成效果。
用法（BEAT segment + whisper 条件）:
  python evaluation_MARDM_g1ml3d_train_vis.py \\
      --dataset_dir ./data/BEAT_v2 --use_segment --name MARDM_beat \\
      --ae_checkpoint_dir ./checkpoints/mixed/AE/model \\
      --checkpoint_name checkpoint_epoch_400.tar --checkpoints_dir ./checkpoints \\
      --num_segments 20 --vis_robot g1_brainco --motion_fps 60
"""
import os
from os.path import join as pjoin
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from models.AE import AE_models
from models.MARDM import MARDM_models
from utils.datasets import BeatSegmentDataset, collate_fn
import argparse
import subprocess
import tempfile
import shutil
import imageio
from skimage.transform import resize
from tqdm import tqdm

VAE_DOWNSAMPLE_FACTOR = 16
PROMPT_FILENAME_MAX_LEN = 80


def prompt_to_filename(prompt, max_len=PROMPT_FILENAME_MAX_LEN):
    if not prompt or not isinstance(prompt, str):
        return "seg"
    for c in r'\/:*?"<>|':
        prompt = prompt.replace(c, "_")
    s = "_".join(prompt.split()).strip("_")
    if not s:
        return "seg"
    return s[:max_len].rstrip(".") or "seg"


def vis_npz_motion(motion_npz_path, output_path, robot_type="g1_brainco", rate_limit=False, motion_fps=60):
    vis_script = pjoin(os.path.dirname(__file__), "external", "GMR", "scripts", "vis_npz_motion.py")
    if not os.path.exists(vis_script):
        vis_script = pjoin(os.path.dirname(__file__), "external", "GMR", "scripts", "vis_npz_motion.py")
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

    data_root = args.dataset_dir
    segment_dir = pjoin(data_root, 'segment')
    # 默认用训练集
    full_split = pjoin(segment_dir, args.split_file)
    if not os.path.exists(full_split):
        raise FileNotFoundError(f"Split not found: {full_split}")

    with open(full_split, 'r') as f:
        all_ids = [line.strip() for line in f if line.strip()]
    num_segments = min(args.num_segments, len(all_ids))
    if args.shuffle_train:
        random.shuffle(all_ids)
    selected_ids = all_ids[:num_segments]
    # 只读这 num_segments 个 id，写临时列表，让 BeatSegmentDataset 按列表读单段 npz（不加载整体 merged npz）
    tmp_split = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', dir=segment_dir, delete=False)
    tmp_split.write('\n'.join(selected_ids) + '\n')
    tmp_split.close()
    split_file = tmp_split.name

    mean_path = pjoin(data_root, 'Mean.npy')
    std_path = pjoin(data_root, 'Std.npy')
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        raise FileNotFoundError(f"Mean.npy or Std.npy not found in {data_root}.")
    mean = np.load(mean_path)
    std = np.load(std_path)

    if os.path.isabs(args.checkpoint_name) and os.path.isfile(args.checkpoint_name):
        mardm_ckpt_path = args.checkpoint_name
    else:
        mardm_model_dir = pjoin(args.checkpoints_dir, 'g1ml3d', args.name, 'model')
        mardm_ckpt_path = pjoin(mardm_model_dir, args.checkpoint_name)
    if not os.path.isfile(mardm_ckpt_path):
        raise FileNotFoundError(f"MARDM checkpoint not found: {mardm_ckpt_path}")

    ae_checkpoint_dir = getattr(args, 'ae_checkpoint_dir', None)
    if ae_checkpoint_dir:
        ae_ckpt_path = pjoin(ae_checkpoint_dir, args.ae_checkpoint_name)
    else:
        ae_ckpt_path = pjoin(args.checkpoints_dir, 'g1ml3d', args.ae_name, 'model', args.ae_checkpoint_name)
    if not os.path.isfile(ae_ckpt_path):
        raise FileNotFoundError(f"AE checkpoint not found: {ae_ckpt_path}")

    dim_pose = mean.shape[0]
    max_motion_length = args.max_motion_length if args.max_motion_length is not None else 300

    eval_dataset = BeatSegmentDataset(
        segment_dir, split_file, mean, std, max_motion_length, 20, evaluation=True
    )
    # 仅 20 段，batch_size=1，每段生成完立即可视化
    eval_loader = DataLoader(
        eval_dataset, batch_size=1, num_workers=0, drop_last=False, collate_fn=collate_fn, shuffle=False
    )

    ae = AE_models[args.ae_model](input_width=dim_pose)
    ckpt_ae = torch.load(ae_ckpt_path, map_location='cpu')
    ae.load_state_dict(ckpt_ae['ae'])

    sample0 = eval_dataset[0]
    whisper_dim = sample0[7].shape[-1] if isinstance(sample0, (list, tuple)) and len(sample0) >= 8 else 512
    cond_mode = getattr(args, 'cond_mode', None) or 'whisper'
    print(f'Whisper condition dim: {whisper_dim}, cond_mode: {cond_mode}')

    ema_mardm = MARDM_models[args.model](
        ae_dim=ae.output_emb_width,
        cond_mode=cond_mode,
        whisper_dim=whisper_dim
    )
    ckpt_mardm = torch.load(mardm_ckpt_path, map_location='cpu')
    if 'ema_mardm' not in ckpt_mardm:
        raise KeyError("Checkpoint must contain 'ema_mardm' key.")
    missing, unexpected = ema_mardm.load_state_dict(ckpt_mardm['ema_mardm'], strict=False)
    if cond_mode != 'text':
        assert all(k.startswith('clip_model.') for k in (unexpected or [])), f"unexpected: {unexpected}"
    else:
        assert len(unexpected) == 0
    assert all(k.startswith('clip_model.') for k in (missing or []))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae.eval()
    ae.to(device)
    ema_mardm.eval()
    ema_mardm.to(device)

    out_subdir = getattr(args, 'out_subdir', None) or ('eval_test' if 'test' in args.split_file else 'eval_train')
    out_dir = pjoin(args.checkpoints_dir, 'g1ml3d', args.name, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    npz_dir = pjoin(out_dir, 'generated_npz')
    vis_dir = pjoin(out_dir, 'visualizations')
    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    total_mse = 0.0
    total_count = 0
    temp_dir = tempfile.mkdtemp()

    try:
        for idx, batch in enumerate(tqdm(eval_loader, desc="Generate & vis", total=len(eval_dataset))):
            word_embeddings, pos_one_hots, caption, sent_len, motion_gt, m_length, _, whisper_feat = batch[:8]
            motion_gt = motion_gt.float().to(device)
            m_length = m_length.long().to(device)
            captions = list(caption) if isinstance(caption, (list, tuple)) else [caption]
            m_lens = (m_length // VAE_DOWNSAMPLE_FACTOR).long().to(device).clamp(min=1)

            with torch.no_grad():
                conds_in = whisper_feat.to(device).float()
                pred_latents = ema_mardm.generate(
                    conds_in, m_lens, args.time_steps, args.cfg,
                    temperature=args.temperature, hard_pseudo_reorder=args.hard_pseudo_reorder
                )
                pred_motions = ae.decode(pred_latents)
                pred_motions = pred_motions.detach().cpu().numpy()
                pred_motions = eval_dataset.inv_transform(pred_motions, mean, std)

            motion_gt_np = eval_dataset.inv_transform(motion_gt.detach().cpu().numpy(), mean, std)
            len_j = int(m_length[0].item())
            gt_j = motion_gt_np[0, :len_j]
            pred_j = pred_motions[0]
            min_len = min(gt_j.shape[0], pred_j.shape[0], len_j)
            if min_len <= 0:
                continue
            gt_j = gt_j[:min_len]
            pred_j = pred_j[:min_len]
            mse_j = np.mean((gt_j - pred_j) ** 2)
            total_mse += mse_j
            total_count += 1

            caption_j = captions[0] if captions else ""
            prompt_base = prompt_to_filename(caption_j)
            base_name = f"seg_{idx:05d}_{prompt_base}"

            np.savez(pjoin(npz_dir, f"{base_name}.npz"), qpos=pred_j)

            # 每段立即可视化
            gt_npz = pjoin(temp_dir, f"{base_name}_gt.npz")
            pred_npz = pjoin(temp_dir, f"{base_name}_pred.npz")
            gt_video = pjoin(temp_dir, f"{base_name}_gt.mp4")
            pred_video = pjoin(temp_dir, f"{base_name}_pred.mp4")
            combined_path = pjoin(vis_dir, f"{base_name}_comparison.mp4")
            np.savez(gt_npz, qpos=gt_j)
            np.savez(pred_npz, qpos=pred_j)
            try:
                vis_npz_motion(gt_npz, gt_video, args.vis_robot, False, args.motion_fps)
                vis_npz_motion(pred_npz, pred_video, args.vis_robot, False, args.motion_fps)
                concatenate_videos_horizontally(gt_video, pred_video, combined_path)
            except Exception as e:
                print(f"  Vis error seg {idx}: {e}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    avg_mse = total_mse / total_count if total_count else float('nan')
    print(f"Evaluated {total_count} segments. Mean MSE: {avg_mse:.6f}")
    print(f"Videos: {vis_dir}")
    print(f"NPZ: {npz_dir}")
    if os.path.exists(split_file):
        try:
            os.unlink(split_file)
        except OSError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从训练集取 segment，逐段生成并立即可视化")
    parser.add_argument('--name', type=str, default='MARDM_beat')
    parser.add_argument('--ae_name', type=str, default='AE_g1ml3d')
    parser.add_argument('--ae_model', type=str, default='AE_Model')
    parser.add_argument('--model', type=str, default='MARDM-SiT-XL')
    parser.add_argument('--dataset_dir', type=str, default='./data/BEAT_v2')
    parser.add_argument('--split_file', type=str, default='segment_train.txt',
                        help='segment 划分文件：segment_train.txt / segment_test.txt')
    parser.add_argument('--out_subdir', type=str, default=None,
                        help='输出子目录名，默认根据 split_file 推断：train->eval_train, test->eval_test')
    parser.add_argument('--checkpoint_name', type=str, default='checkpoint_epoch_400.tar')
    parser.add_argument('--ae_checkpoint_name', type=str, default='latest.tar')
    parser.add_argument('--ae_checkpoint_dir', type=str, default=None)
    parser.add_argument('--cond_mode', type=str, default='whisper', choices=['text', 'whisper'])
    parser.add_argument('--max_motion_length', type=int, default=300)
    parser.add_argument('--num_segments', type=int, default=20, help='生成并可视化的 segment 数量')
    parser.add_argument('--shuffle_train', action='store_true', help='从训练集中随机取 segment')
    parser.add_argument('--time_steps', type=int, default=18)
    parser.add_argument('--cfg', type=float, default=4.5)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--hard_pseudo_reorder', action='store_true')
    parser.add_argument('--vis_robot', type=str, default='g1_brainco')
    parser.add_argument('--motion_fps', type=int, default=60)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    arg = parser.parse_args()
    main(arg)
