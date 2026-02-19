"""
Test MARDM model for G1ML3D_v1 (text2motion, text prompt only).
对应训练命令: readme 169-179 (MARDM_SiT_XL_g1ml3d_text_only)
"""
import os
os.environ["MUJOCO_GL"] = "egl"

from os.path import join as pjoin
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
import clip
from models.AE import AE_models
from models.MARDM import MARDM_models
from utils.datasets import G1ML3DText2MotionDataset
from general_motion_retargeting import RobotMotionViewer
import argparse
from tqdm import tqdm
import imageio
from skimage.transform import resize


def vis_npz_motion(motion_npz_path, output_path, robot_type="g1_brainco", rate_limit=False, motion_fps=60, label_text=""):
    """Visualize motion from npz file (qpos format)."""
    data = np.load(motion_npz_path)
    if 'qpos' in data:
        motion_csv = data['qpos']
    else:
        keys = list(data.keys())
        motion_csv = data[keys[0]] if keys else None
    if motion_csv is None:
        raise ValueError(f"No valid motion data in {motion_npz_path}")
    data_frames = motion_csv.shape[0]
    robot_motion_viewer = RobotMotionViewer(
        robot_type=robot_type, motion_fps=motion_fps,
        transparent_robot=0, record_video=True, video_path=output_path)
    pbar = tqdm(total=data_frames, desc=label_text or "Visualizing", leave=False)
    for i in range(data_frames):
        pbar.update(1)
        qpos = motion_csv[i]
        quat_wxyz = qpos[3:7]
        root_pos = qpos[:3]
        robot_motion_viewer.step(root_pos=root_pos, root_rot=quat_wxyz, dof_pos=qpos[7:], rate_limit=rate_limit)
    pbar.close()
    robot_motion_viewer.close()
    del robot_motion_viewer


def concatenate_videos_horizontally(video1_path, video2_path, output_path):
    """将两个视频左右拼接"""
    reader1 = imageio.get_reader(video1_path)
    reader2 = imageio.get_reader(video2_path)
    fps = reader1.get_meta_data()['fps']
    width1, height1 = reader1.get_meta_data()['size']
    width2, height2 = reader2.get_meta_data()['size']
    target_height = max(height1, height2)
    target_width = width1 + width2
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps)
    for frame1, frame2 in zip(reader1, reader2):
        if frame1.shape[0] != target_height:
            frame1 = resize(frame1, (target_height, width1), preserve_range=True, anti_aliasing=True).astype(frame1.dtype)
        if frame2.shape[0] != target_height:
            frame2 = resize(frame2, (target_height, width2), preserve_range=True, anti_aliasing=True).astype(frame2.dtype)
        writer.append_data(np.hstack([frame1, frame2]))
    reader1.close()
    reader2.close()
    writer.close()


def test_on_testset(args):
    """在验证集上测试：给定文本+前60帧，生成后240帧，保存预测与真值对比。"""
    print("=" * 80)
    print("Testing on G1ML3D_v1 validation set...")
    print("=" * 80)

    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = '/root/workspace/MARDM/data/G1ML3D_v1'
    g1ml3d_root = pjoin(data_root, 'joints_npz')
    text_dir = pjoin(data_root, 'texts')
    val_split_file = pjoin(data_root, 'val.txt')
    mean = np.load(pjoin(data_root, 'Mean.npy'))
    std = np.load(pjoin(data_root, 'Std.npy'))
    dim_pose = mean.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = G1ML3DText2MotionDataset(
        mean, std, g1ml3d_root, text_dir, val_split_file,
        args.unit_length, args.max_motion_length, split='val',
        clip_version='ViT-B/32', device=device)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, drop_last=False,
        num_workers=args.num_workers, shuffle=False)

    model_dir = pjoin(args.checkpoints_dir, 'g1ml3d', args.name, 'model')
    ae = AE_models[args.ae_model](input_width=dim_pose)
    ckpt = torch.load(pjoin(args.checkpoints_dir, 'g1ml3d', args.ae_name, 'model', 'latest.tar'), map_location='cpu')
    ae.load_state_dict(ckpt['ae'])

    cond_mode = 'mixed'
    audio_dim = 512
    ema_mardm = MARDM_models[args.model](
        ae_dim=ae.output_emb_width, cond_mode=cond_mode, audio_dim=audio_dim,
        use_cross_attn=False)
    checkpoint = torch.load(pjoin(model_dir, 'latest.tar'), map_location='cpu')
    ema_mardm.load_state_dict(checkpoint['ema_mardm'], strict=False)

    ae.to(device)
    ema_mardm.to(device)
    ae.eval()
    ema_mardm.eval()

    result_dir = pjoin('./test_results', 'g1ml3d', args.name, 'test_set')
    os.makedirs(result_dir, exist_ok=True)
    num_samples = min(args.num_test_samples, len(test_dataset))
    print(f"Test set size: {len(test_dataset)}, evaluating on {num_samples} samples")
    print(f"Output directory: {result_dir}")

    with torch.no_grad():
        for idx, batch_data in enumerate(tqdm(test_loader, desc="Testing batches")):
            if idx * args.batch_size >= num_samples:
                break
            _, clip_features, motion_condition, motion_target, m_lens = batch_data
            clip_features = clip_features.to(device).float()
            motion_condition = motion_condition.to(device).float()
            motion_target = motion_target.to(device).float()
            m_lens = m_lens.to(device).long()

            motion_condition_latent = ae.encode(motion_condition)
            m_lens_target = torch.tensor([240 // 4] * clip_features.size(0), device=device).long()

            pred_latents = ema_mardm.generate(
                conds=None,
                m_lens=m_lens_target,
                timesteps=args.time_steps,
                cond_scale=args.cfg,
                temperature=args.temperature,
                progress_callback=lambda step: None,
                motion_condition_latent=motion_condition_latent,
                text_condition=clip_features,
            )
            pred_motions = ae.decode(pred_latents)
            pred_motions = pred_motions.detach().cpu().numpy()
            motion_condition_np = motion_condition.detach().cpu().numpy()
            pred_motions_full = np.concatenate([motion_condition_np, pred_motions], axis=1)
            pred_motions_denorm = pred_motions_full * std + mean

            motion_target_np = motion_target.detach().cpu().numpy()
            motion_gt_full = np.concatenate([motion_condition_np, motion_target_np], axis=1)
            motion_gt_denorm = motion_gt_full * std + mean

            for b in range(clip_features.size(0)):
                sample_idx = idx * args.batch_size + b
                if sample_idx >= num_samples:
                    break
                sample_dir = pjoin(result_dir, f'sample_{sample_idx:04d}')
                os.makedirs(sample_dir, exist_ok=True)
                actual_len = m_lens[b].item()
                pred_motion = pred_motions_denorm[b][:actual_len]
                gt_motion = motion_gt_denorm[b][:actual_len]

                pred_npz_path = pjoin(sample_dir, 'prediction.npz')
                np.savez(pred_npz_path, qpos=pred_motion)
                gt_npz_path = pjoin(sample_dir, 'ground_truth.npz')
                np.savez(gt_npz_path, qpos=gt_motion)

                if args.generate_videos:
                    pred_video_path = pjoin(sample_dir, 'prediction.mp4')
                    vis_npz_motion(pred_npz_path, pred_video_path,
                                   robot_type=args.robot_type, rate_limit=args.rate_limit,
                                   motion_fps=args.motion_fps, label_text="Pred")
                    gt_video_path = pjoin(sample_dir, 'ground_truth.mp4')
                    vis_npz_motion(gt_npz_path, gt_video_path,
                                   robot_type=args.robot_type, rate_limit=args.rate_limit,
                                   motion_fps=args.motion_fps, label_text="GT")
                    comparison_path = pjoin(sample_dir, 'comparison.mp4')
                    concatenate_videos_horizontally(gt_video_path, pred_video_path, comparison_path)
    print(f"\nTest completed! Results saved to {result_dir}")


def generate_from_text_prompt(args):
    """从单个文本描述生成动作：前60帧用零姿态作为条件，生成后240帧。"""
    print("=" * 80)
    print("Generating motion from text prompt...")
    print("=" * 80)

    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = '/root/workspace/MARDM/data/G1ML3D_v1'
    mean = np.load(pjoin(data_root, 'Mean.npy'))
    std = np.load(pjoin(data_root, 'Std.npy'))
    dim_pose = mean.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CLIP 编码文本
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()
    with torch.no_grad():
        text_tokens = clip.tokenize([args.text_prompt], truncate=True).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        clip_feature = text_features.cpu().numpy().astype(np.float32)
    del clip_model

    model_dir = pjoin(args.checkpoints_dir, 'g1ml3d', args.name, 'model')
    ae = AE_models[args.ae_model](input_width=dim_pose)
    ckpt = torch.load(pjoin(args.checkpoints_dir, 'g1ml3d', args.ae_name, 'model', 'latest.tar'), map_location='cpu')
    ae.load_state_dict(ckpt['ae'])

    ema_mardm = MARDM_models[args.model](
        ae_dim=ae.output_emb_width, cond_mode='mixed', audio_dim=512,
        use_cross_attn=False)
    checkpoint = torch.load(pjoin(model_dir, 'latest.tar'), map_location='cpu')
    ema_mardm.load_state_dict(checkpoint['ema_mardm'], strict=False)

    ae.to(device)
    ema_mardm.to(device)
    ae.eval()
    ema_mardm.eval()

    # 条件：前60帧使用零向量（归一化后即 mean）
    condition_motion = np.zeros((60, dim_pose), dtype=np.float32)
    condition_tensor = torch.from_numpy(condition_motion).unsqueeze(0).to(device).float()
    condition_latent = ae.encode(condition_tensor)
    text_condition_tensor = torch.from_numpy(clip_feature).unsqueeze(0).to(device).float()
    m_lens = torch.tensor([240 // 4], dtype=torch.long, device=device)

    with torch.no_grad():
        pred_latents = ema_mardm.generate(
            conds=None,
            m_lens=m_lens,
            timesteps=args.time_steps,
            cond_scale=args.cfg,
            temperature=args.temperature,
            motion_condition_latent=condition_latent,
            text_condition=text_condition_tensor,
        )
        pred_motions = ae.decode(pred_latents)
        pred_motions = pred_motions[0].cpu().numpy()
    # 反归一化：condition 为零即 mean，pred 乘 std 加 mean
    condition_denorm = condition_motion * std + mean
    pred_denorm = pred_motions * std + mean
    full_motion = np.concatenate([condition_denorm, pred_denorm], axis=0)

    result_dir = pjoin('./test_results', 'g1ml3d', args.name, 'text_generation')
    os.makedirs(result_dir, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in args.text_prompt[:50])
    npz_path = pjoin(result_dir, f'{safe_name}.npz')
    np.savez(npz_path, qpos=full_motion)
    with open(pjoin(result_dir, f'{safe_name}.txt'), 'w', encoding='utf-8') as f:
        f.write(args.text_prompt)

    if args.generate_videos:
        video_path = pjoin(result_dir, f'{safe_name}.mp4')
        vis_npz_motion(npz_path, video_path,
                       robot_type=args.robot_type, rate_limit=args.rate_limit,
                       motion_fps=args.motion_fps, label_text="Text2Motion")
        print(f"Video saved to: {video_path}")
    print(f"Motion saved to: {npz_path}")
    print(f"Results directory: {result_dir}")


def main(args):
    if args.mode == 'testset':
        test_on_testset(args)
    elif args.mode == 'text':
        generate_from_text_prompt(args)
    else:
        raise ValueError("--mode must be 'testset' or 'text'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MARDM (g1ml3d text2motion)")
    parser.add_argument('--mode', type=str, default='testset', choices=['testset', 'text'],
                        help='testset: 验证集评测; text: 从 --text_prompt 生成')
    parser.add_argument('--name', type=str, default='MARDM_SiT_XL_g1ml3d_text_only', help='Model name')
    parser.add_argument('--ae_name', type=str, default='AE')
    parser.add_argument('--ae_model', type=str, default='AE_Model')
    parser.add_argument('--model', type=str, default='MARDM-SiT-XL')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_test_samples', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_motion_length', type=int, default=300)
    parser.add_argument('--unit_length', type=int, default=4)

    parser.add_argument('--text_prompt', type=str, default='',
                        help='Text description (required when mode=text)')
    parser.add_argument('--time_steps', type=int, default=18)
    parser.add_argument('--cfg', type=float, default=4.5)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=3407)

    parser.add_argument('--generate_videos', action='store_true', help='生成 npz 的同时渲染对比/生成视频')
    parser.add_argument('--robot_type', type=str, default='g1_brainco')
    parser.add_argument('--rate_limit', action='store_true')
    parser.add_argument('--motion_fps', type=int, default=60)

    args = parser.parse_args()
    if args.mode == 'text' and not args.text_prompt.strip():
        raise ValueError("--text_prompt is required when --mode=text")
    main(args)
