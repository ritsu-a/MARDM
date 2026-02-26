import os
from os.path import join as pjoin
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from models.AE import AE_models
from models.MARDM import MARDM_models
from utils.datasets import G1ML3DText2MotionDataset, BeatSegmentDataset, collate_fn
import time
import copy
from collections import OrderedDict, defaultdict
from utils.train_utils import update_lr_warm_up, def_value, save, print_current_loss, update_ema
import argparse


def main(args):
    #################################################################################
    #                                      Seed                                     #
    #################################################################################
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)
    # setting this to true significantly increase training and sampling speed
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    #################################################################################
    #                                    Train Data                                 #
    #################################################################################
    data_root = args.dataset_dir
    use_segment = getattr(args, 'use_segment', False)
    max_motion_length = args.max_motion_length if args.max_motion_length is not None else (300 if use_segment else 196)

    mean_path = pjoin(data_root, 'Mean.npy')
    std_path = pjoin(data_root, 'Std.npy')
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        raise FileNotFoundError(f"Mean.npy or Std.npy not found in {data_root}. Please run VAE training first.")
    mean = np.load(mean_path)
    std = np.load(std_path)

    if use_segment:
        segment_dir = pjoin(data_root, 'segment')
        train_split = pjoin(segment_dir, 'segment_train.txt')
        val_split = pjoin(segment_dir, 'segment_test.txt')
        if not os.path.exists(train_split) and not os.path.exists(pjoin(segment_dir, 'segment_train.npz')):
            raise FileNotFoundError(f"BEAT segment not found: {train_split} or segment_train.npz. Run scripts/beat_segment_to_npz.py first.")
        train_dataset = BeatSegmentDataset(segment_dir, train_split, mean, std, max_motion_length, 20, evaluation=False)
        val_dataset = BeatSegmentDataset(segment_dir, val_split, mean, std, max_motion_length, 20, evaluation=False)
    else:
        motion_dir = pjoin(data_root, 'joints_npz')
        text_dir = pjoin(data_root, 'texts')
        train_split_file = pjoin(data_root, 'train.txt')
        val_split_file = pjoin(data_root, 'val.txt')
        train_dataset = G1ML3DText2MotionDataset(mean, std, train_split_file, 'g1ml3d', motion_dir, text_dir,
                                                 args.unit_length, max_motion_length, 20, evaluation=False)
        val_dataset = G1ML3DText2MotionDataset(mean, std, val_split_file, 'g1ml3d', motion_dir, text_dir,
                                               args.unit_length, max_motion_length, 20, evaluation=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                            shuffle=True)

    #################################################################################
    #                                    Eval Data                                  #
    #################################################################################
    eval_loader = None
    if args.need_evaluation:
        eval_mean, eval_std = mean, std
        if use_segment:
            split_file = pjoin(segment_dir, 'segment_test.txt')
            eval_dataset = BeatSegmentDataset(segment_dir, split_file, eval_mean, eval_std, max_motion_length, 20, evaluation=True)
        else:
            split_file = pjoin(data_root, 'val.txt')
            eval_dataset = G1ML3DText2MotionDataset(eval_mean, eval_std, split_file, 'g1ml3d', motion_dir, text_dir,
                                                   4, max_motion_length, 20, evaluation=True)
        eval_loader = DataLoader(eval_dataset, batch_size=32, num_workers=args.num_workers, drop_last=True,
                                 collate_fn=collate_fn, shuffle=True)
    #################################################################################
    #                                      Models                                   #
    #################################################################################
    model_dir = pjoin(args.checkpoints_dir, 'g1ml3d', args.name, 'model')
    os.makedirs(model_dir, exist_ok=True)

    # Get motion dimension from mean shape
    dim_pose = mean.shape[0]
    
    # Load VAE (AE) model
    ae = AE_models[args.ae_model](input_width=dim_pose)
    ae_checkpoint_path = getattr(args, 'ae_checkpoint_dir', None)
    if ae_checkpoint_path:
        ae_checkpoint_path = pjoin(ae_checkpoint_path, args.ae_checkpoint_name)
    else:
        ae_checkpoint_path = pjoin(args.checkpoints_dir, 'g1ml3d', args.ae_name, 'model', args.ae_checkpoint_name)
    if not os.path.exists(ae_checkpoint_path):
        raise FileNotFoundError(f"AE checkpoint not found: {ae_checkpoint_path}")
    
    ckpt = torch.load(ae_checkpoint_path, map_location='cpu')
    model_key = 'ae'
    ae.load_state_dict(ckpt[model_key])
    print(f"Loaded VAE from {ae_checkpoint_path}")

    # Create MARDM model（use_segment 时用 whisper 条件，否则 text）
    cond_mode = getattr(args, 'cond_mode', None) or ('whisper' if use_segment else 'text')
    whisper_dim = None
    if cond_mode == 'whisper':
        # 从数据推断 whisper 特征维度（segment 可能为 512 或 1024）
        sample_whisper = train_dataset[0][0]
        whisper_dim = sample_whisper.shape[-1]
        print(f'Whisper condition dim (from data): {whisper_dim}')
    mardm = MARDM_models[args.model](ae_dim=ae.output_emb_width, cond_mode=cond_mode, whisper_dim=whisper_dim or 512)
    ema_mardm = copy.deepcopy(mardm)
    ema_mardm.eval()
    for param in ema_mardm.parameters():
        param.requires_grad_(False)

    all_params = 0
    exclude_prefix = 'clip_model.' if cond_mode == 'text' else ''
    pc_transformer = sum(param.numel() for param in
                         [p for name, p in mardm.named_parameters() if not name.startswith(exclude_prefix)])
    all_params += pc_transformer
    print('Total parameters of all models: {:.2f}M'.format(all_params / 1000_000))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Note: Evaluators not available for G1ML3D due to dimension mismatch
    eval_wrapper = None
    #################################################################################
    #                                    Training Loop                              #
    #################################################################################
    logger = SummaryWriter(model_dir)
    ae.eval()
    ae.to(device)
    mardm.to(device)
    ema_mardm.to(device)

    optimizer = optim.AdamW(mardm.parameters(), betas=(0.9, 0.99), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_decay)

    epoch = 0
    it = 0
    if args.is_continue:
        checkpoint_path = pjoin(model_dir, 'latest.tar')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            missing_keys, unexpected_keys = mardm.load_state_dict(checkpoint['mardm'], strict=False)
            missing_keys2, unexpected_keys2 = ema_mardm.load_state_dict(checkpoint['ema_mardm'], strict=False)
            if cond_mode == 'text':
                assert len(unexpected_keys) == 0 and len(unexpected_keys2) == 0
            else:
                # whisper 等模式：允许 ckpt 中多出 clip_model（从 text 切到 whisper 时忽略）
                assert all([k.startswith('clip_model.') for k in unexpected_keys])
                assert all([k.startswith('clip_model.') for k in unexpected_keys2])
            assert all([k.startswith('clip_model.') for k in missing_keys])
            assert all([k.startswith('clip_model.') for k in missing_keys2])
            optimizer.load_state_dict(checkpoint['opt_mardm'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            epoch, it = checkpoint['ep'], checkpoint['total_it']
            print("Load model epoch:%d iterations:%d" % (epoch, it))
        else:
            print(f"Checkpoint not found at {checkpoint_path}, starting from scratch")

    start_time = time.time()
    total_iters = args.epoch * len(train_loader)
    print(f'Total Epochs: {args.epoch}, Total Iters: {total_iters}')
    print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))

    logs = defaultdict(def_value, OrderedDict())

    best_fid, best_div, best_top1, best_top2, best_top3, best_matching, clip_score = 1000, 0, 0, 0, 0, 100, -1
    worst_loss = 100

    # VAE downsampling factor: down_t=4 means 2^4=16x downsampling
    # Note: Original VAE had down_t=2 (4x), but we changed to down_t=4 (16x)
    vae_downsample_factor = 16

    while epoch < args.epoch:
        ae.eval()
        mardm.train()

        for i, batch_data in enumerate(train_loader):
            it += 1
            if it < args.warm_up_iter:
                update_lr_warm_up(it, args.warm_up_iter, optimizer, args.lr)

            conds, motion, m_lens = batch_data
            motion = motion.detach().float().to(device)
            m_lens = m_lens.detach().long().to(device)

            latent = ae.encode(motion)
            # VAE downsampling: divide by 16 (was 4 for original VAE)
            m_lens = m_lens // vae_downsample_factor

            conds = conds.to(device).float() if torch.is_tensor(conds) else conds

            loss = mardm.forward_loss(latent, conds, m_lens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            logs['loss'] += loss.item()
            logs['lr'] += optimizer.param_groups[0]['lr']
            update_ema(mardm, ema_mardm, 0.9999)

            if it % args.log_every == 0:
                mean_loss = OrderedDict()
                for tag, value in logs.items():
                    logger.add_scalar('Train/%s' % tag, value / args.log_every, it)
                    mean_loss[tag] = value / args.log_every
                logs = defaultdict(def_value, OrderedDict())
                print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

        save(pjoin(model_dir, 'latest.tar'), epoch, mardm, optimizer, scheduler,
             it, 'mardm', ema_mardm=ema_mardm)
        
        # Save checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            checkpoint_name = f'checkpoint_epoch_{epoch+1}.tar'
            save(pjoin(model_dir, checkpoint_name), epoch, mardm, optimizer, scheduler,
                 it, 'mardm', ema_mardm=ema_mardm)
            print(f"Saved checkpoint: {checkpoint_name}")
        
        epoch += 1
        #################################################################################
        #                                      Eval Loop                                #
        #################################################################################
        print('Validation time:')
        ae.eval()
        mardm.eval()
        val_loss = []
        with torch.no_grad():
            for i, batch_data in enumerate(val_loader):
                conds, motion, m_lens = batch_data
                motion = motion.detach().float().to(device)
                m_lens = m_lens.detach().long().to(device)

                latent = ae.encode(motion)
                m_lens = m_lens // vae_downsample_factor

                conds = conds.to(device).float() if torch.is_tensor(conds) else conds

                loss = mardm.forward_loss(latent, conds, m_lens)
                val_loss.append(loss.item())

        print(f"Validation loss:{np.mean(val_loss):.3f}")
        logger.add_scalar('Val/loss', np.mean(val_loss), epoch)
        if np.mean(val_loss) < worst_loss:
            print(f"Improved loss from {worst_loss:.02f} to {np.mean(val_loss)}!!!")
            worst_loss = np.mean(val_loss)
            save(pjoin(model_dir, 'net_best_loss.tar'), epoch-1, mardm, optimizer, scheduler,
                 it, 'mardm', ema_mardm=ema_mardm)
        
        # Note: Evaluation metrics (FID, etc.) not available for G1ML3D due to dimension mismatch
        # If needed, implement G1ML3D-specific evaluator or use simplified metrics
        if args.need_evaluation and eval_loader is not None:
            print("Note: Full evaluation metrics not available for G1ML3D. Skipping detailed evaluation.")
            # Could add simplified evaluation here if needed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='MARDM')
    parser.add_argument('--ae_name', type=str, default="AE_g1ml3d")
    parser.add_argument('--ae_model', type=str, default='AE_Model')
    parser.add_argument('--ae_checkpoint_name', type=str, default='latest.tar',
                        help='AE checkpoint file name (latest.tar or net_best_fid.tar)')
    parser.add_argument('--ae_checkpoint_dir', type=str, default=None,
                        help='AE 模型目录（含 checkpoint 文件）；BEAT mixed 可用 ./checkpoints/mixed/ae/model')
    parser.add_argument('--use_segment', action='store_true',
                        help='使用 BEAT segment（dataset_dir/segment/segment_train.npz 等）')
    parser.add_argument('--cond_mode', type=str, default=None, choices=['text', 'whisper'],
                        help='条件类型：text=CLIP 文本；whisper=Whisper 音频特征。use_segment 时默认 whisper')
    parser.add_argument('--model', type=str, default='MARDM-SiT-XL')
    parser.add_argument('--dataset_dir', type=str, default='./data/G1ML3D_v1',
                        help='Root directory of G1ML3D dataset')
    parser.add_argument("--max_motion_length", type=int, default=None,
                        help='Motion 最大帧数；use_segment 时默认 300')
    parser.add_argument("--unit_length", type=int, default=4)
    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--warm_up_iter', default=2000, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--milestones', default=[50_000], nargs="+", type=int)
    parser.add_argument('--lr_decay', default=0.1, type=float)

    parser.add_argument('--diffmlps_batch_mul', type=int, default=4)
    parser.add_argument('--need_evaluation', action="store_true",
                        help='Enable evaluation (note: full metrics not available for G1ML3D)')

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--is_continue', action="store_true")
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')

    parser.add_argument('--log_every', default=50, type=int)

    arg = parser.parse_args()
    main(arg)
