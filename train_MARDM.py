import os
from os.path import join as pjoin
import torch
import torch.distributed as dist
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from models.AE import AE_models
from models.MARDM import MARDM_models
from utils.evaluators import Evaluators
from utils.datasets import Text2MotionDataset, BEAT_v2Audio2MotionDataset, collate_fn
import time
import copy
from collections import OrderedDict, defaultdict
from utils.train_utils import update_lr_warm_up, def_value, save, print_current_loss, update_ema
from utils.eval_utils import evaluation_mardm
import argparse


def main(args):
    #################################################################################
    #                           Distributed Training Setup                         #
    #################################################################################
    if args.distributed:
        # Initialize distributed training
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ['RANK'])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.local_rank = int(os.environ['LOCAL_RANK'])
        else:
            print('Not using distributed mode')
            args.distributed = False
            args.rank = 0
            args.world_size = 1
            args.local_rank = 0
        
        if args.distributed:
            # Set master port if specified
            if args.master_port is not None:
                os.environ['MASTER_PORT'] = str(args.master_port)
            elif 'MASTER_PORT' not in os.environ:
                # Use a default port if not set (avoid 29500)
                os.environ['MASTER_PORT'] = '29501'
            
            torch.cuda.set_device(args.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
            torch.distributed.barrier()
            is_main_process = (args.rank == 0)
            
            if is_main_process:
                print(f'Distributed training initialized on port {os.environ["MASTER_PORT"]}')
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        is_main_process = True
    
    #################################################################################
    #                                      Seed                                     #
    #################################################################################
    torch.backends.cudnn.benchmark = False
    os.environ["OMP_NUM_THREADS"] = "1"
    random.seed(args.seed + args.rank)
    np.random.seed(args.seed + args.rank)
    torch.manual_seed(args.seed + args.rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + args.rank)
    torch.autograd.set_detect_anomaly(True)
    # setting this to true significantly increase training and sampling speed
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    #################################################################################
    #                                    Train Data                                 #
    #################################################################################
    if args.dataset_name == "beat_v2":
        # BEAT_v2 dataset (audio-to-motion)
        data_root = '/root/workspace/MARDM/data/BEAT_v2'
        mean = np.load(pjoin(data_root, 'Mean.npy'))
        std = np.load(pjoin(data_root, 'Std.npy'))
        dim_pose = mean.shape[0]
        
        train_dataset = BEAT_v2Audio2MotionDataset(mean, std, data_root, args.unit_length, 
                                                     args.max_motion_length, split='train')
        val_dataset = BEAT_v2Audio2MotionDataset(mean, std, data_root, args.unit_length, 
                                                   args.max_motion_length, split='val')
        
        # Setup distributed sampler if using distributed training
        if args.distributed:
            train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
            shuffle = False  # Shuffle is handled by DistributedSampler
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, 
                                  num_workers=args.num_workers, shuffle=shuffle, pin_memory=True, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, 
                                num_workers=args.num_workers, shuffle=False, pin_memory=True, sampler=val_sampler)
    elif args.dataset_name == "t2m":
        data_root = f'{args.dataset_dir}/HumanML3D/'
        dim_pose = 67
        motion_dir = pjoin(data_root, 'new_joint_vecs')
        text_dir = pjoin(data_root, 'texts')
        mean = np.load(pjoin(data_root, 'Mean.npy'))
        std = np.load(pjoin(data_root, 'Std.npy'))
        train_split_file = pjoin(data_root, 'train.txt')
        val_split_file = pjoin(data_root, 'val.txt')

        train_dataset = Text2MotionDataset(mean, std, train_split_file, args.dataset_name, motion_dir, text_dir,
                                              args.unit_length, args.max_motion_length, 20, evaluation=False)
        val_dataset = Text2MotionDataset(mean, std, val_split_file, args.dataset_name, motion_dir, text_dir,
                                          args.unit_length, args.max_motion_length, 20, evaluation=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                                  shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                                shuffle=True)
    else:
        data_root = f'{args.dataset_dir}/KIT-ML/'
        dim_pose = 64
        motion_dir = pjoin(data_root, 'new_joint_vecs')
        text_dir = pjoin(data_root, 'texts')
        mean = np.load(pjoin(data_root, 'Mean.npy'))
        std = np.load(pjoin(data_root, 'Std.npy'))
        train_split_file = pjoin(data_root, 'train.txt')
        val_split_file = pjoin(data_root, 'val.txt')

        train_dataset = Text2MotionDataset(mean, std, train_split_file, args.dataset_name, motion_dir, text_dir,
                                              args.unit_length, args.max_motion_length, 20, evaluation=False)
        val_dataset = Text2MotionDataset(mean, std, val_split_file, args.dataset_name, motion_dir, text_dir,
                                          args.unit_length, args.max_motion_length, 20, evaluation=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                                  shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                                shuffle=True)

    #################################################################################
    #                                    Eval Data                                  #
    #################################################################################
    if args.need_evaluation:
        if args.dataset_name == "beat_v2":
            # For beat_v2, use validation dataset for evaluation
            eval_dataset = BEAT_v2Audio2MotionDataset(mean, std, data_root, 4, 196, split='val')
            eval_loader = DataLoader(eval_dataset, batch_size=32, num_workers=args.num_workers, drop_last=True,
                                     shuffle=True)
        else:
            eval_mean = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_mean.npy')
            eval_std = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_std.npy')
            split_file = pjoin(data_root, 'val.txt')
            eval_dataset = Text2MotionDataset(eval_mean, eval_std, split_file, args.dataset_name, motion_dir, text_dir,
                                              4, 196, 20, evaluation=True)
            eval_loader = DataLoader(eval_dataset, batch_size=32, num_workers=args.num_workers, drop_last=True,
                                     collate_fn=collate_fn, shuffle=True)
    #################################################################################
    #                                      Models                                   #
    #################################################################################
    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')
    os.makedirs(model_dir, exist_ok=True)

    ae = AE_models[args.ae_model](input_width=dim_pose)
    ckpt = torch.load(pjoin(args.checkpoints_dir, args.dataset_name, args.ae_name, 'model',
                            'latest.tar'), map_location='cpu')
    model_key = 'ae'
    ae.load_state_dict(ckpt[model_key])

    # Set condition mode based on dataset
    if args.dataset_name == "beat_v2":
        cond_mode = 'audio'
        # Whisper base model feature dimension is 512
        audio_dim = 512
        mardm = MARDM_models[args.model](ae_dim=ae.output_emb_width, cond_mode=cond_mode, audio_dim=audio_dim)
    else:
        cond_mode = 'text'
        mardm = MARDM_models[args.model](ae_dim=ae.output_emb_width, cond_mode=cond_mode)
    ema_mardm = copy.deepcopy(mardm)
    ema_mardm.eval()
    for param in ema_mardm.parameters():
        param.requires_grad_(False)

    all_params = 0
    pc_transformer = sum(param.numel() for param in
                         [p for name, p in mardm.named_parameters() if not name.startswith('clip_model.')])
    all_params += pc_transformer
    print('Total parameters of all models: {:.2f}M'.format(all_params / 1000_000))

    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    if args.need_evaluation and is_main_process:
        eval_wrapper = Evaluators(args.dataset_name, device=device)
    else:
        eval_wrapper = None
    #################################################################################
    #                                    Training Loop                              #
    #################################################################################
    if is_main_process:
        logger = SummaryWriter(model_dir)
    else:
        logger = None
    ae.eval()
    ae.to(device)
    mardm.to(device)
    ema_mardm.to(device)
    
    # Wrap model with DDP if using distributed training
    if args.distributed:
        mardm = DDP(mardm, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)

    optimizer = optim.AdamW(mardm.parameters(), betas=(0.9, 0.99), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_decay)

    epoch = 0
    it = 0
    if args.is_continue:
        model_dir = pjoin(model_dir, 'latest.tar')
        checkpoint = torch.load(model_dir, map_location=device)
        if args.distributed:
            missing_keys, unexpected_keys = mardm.module.load_state_dict(checkpoint['mardm'], strict=False)
        else:
            missing_keys, unexpected_keys = mardm.load_state_dict(checkpoint['mardm'], strict=False)
        missing_keys2, unexpected_keys2 = ema_mardm.load_state_dict(checkpoint['ema_mardm'], strict=False)
        assert len(unexpected_keys) == 0
        assert len(unexpected_keys2) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])
        assert all([k.startswith('clip_model.') for k in missing_keys2])
        optimizer.load_state_dict(checkpoint['opt_mardm'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch, it = checkpoint['ep'], checkpoint['total_it']
        if is_main_process:
            print("Load model epoch:%d iterations:%d" % (epoch, it))

    start_time = time.time()
    total_iters = args.epoch * len(train_loader)
    if is_main_process:
        print(f'Total Epochs: {args.epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        if args.distributed:
            print(f'Using distributed training with {args.world_size} GPUs')

    logs = defaultdict(def_value, OrderedDict())

    best_fid, best_div, best_top1, best_top2, best_top3, best_matching, clip_score = 1000, 0, 0, 0, 0, 100, -1
    worst_loss = 100

    while epoch < args.epoch:
        if args.distributed:
            train_sampler.set_epoch(epoch)
        ae.eval()
        if args.distributed:
            mardm.module.train()
        else:
            mardm.train()

        for i, batch_data in enumerate(train_loader):
            it += 1
            if it < args.warm_up_iter:
                update_lr_warm_up(it, args.warm_up_iter, optimizer, args.lr)

            conds, motion, m_lens = batch_data
            motion = motion.detach().float().to(device)
            m_lens = m_lens.detach().long().to(device)

            latent = ae.encode(motion)
            m_lens = m_lens // 4

            # For beat_v2, conds is whisper features sequence [batch_size, audio_frames, feature_dim]
            # For text datasets, conds is text strings
            if args.dataset_name == "beat_v2":
                if isinstance(conds, np.ndarray):
                    conds = torch.from_numpy(conds).to(device).float()
                else:
                    conds = conds.to(device).float()
                # Pass full sequence for cross-attention (model will handle it)
                # Shape: [batch_size, audio_frames, feature_dim]
            else:
                conds = conds.to(device).float() if torch.is_tensor(conds) else conds

            if args.distributed:
                loss = mardm.module.forward_loss(latent, conds, m_lens)
            else:
                loss = mardm.forward_loss(latent, conds, m_lens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            logs['loss'] += loss.item()
            logs['lr'] += optimizer.param_groups[0]['lr']
            if args.distributed:
                update_ema(mardm.module, ema_mardm, 0.9999)
            else:
                update_ema(mardm, ema_mardm, 0.9999)

            if it % args.log_every == 0 and is_main_process:
                mean_loss = OrderedDict()
                for tag, value in logs.items():
                    if logger is not None:
                        logger.add_scalar('Train/%s' % tag, value / args.log_every, it)
                    mean_loss[tag] = value / args.log_every
                logs = defaultdict(def_value, OrderedDict())
                print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

        if is_main_process:
            if args.distributed:
                save(pjoin(model_dir, 'latest.tar'), epoch, mardm.module, optimizer, scheduler,
                     it, 'mardm', ema_mardm=ema_mardm)
            else:
                save(pjoin(model_dir, 'latest.tar'), epoch, mardm, optimizer, scheduler,
                     it, 'mardm', ema_mardm=ema_mardm)
        epoch += 1
        #################################################################################
        #                                      Eval Loop                                #
        #################################################################################
        if is_main_process:
            print('Validation time:')
        ae.eval()
        if args.distributed:
            mardm.module.eval()
        else:
            mardm.eval()
        val_loss = []
        with torch.no_grad():
            for i, batch_data in enumerate(val_loader):
                conds, motion, m_lens = batch_data
                motion = motion.detach().float().to(device)
                m_lens = m_lens.detach().long().to(device)

                latent = ae.encode(motion)
                m_lens = m_lens // 4

                # For beat_v2, conds is whisper features sequence [batch_size, audio_frames, feature_dim]
                # For text datasets, conds is text strings
                if args.dataset_name == "beat_v2":
                    if isinstance(conds, np.ndarray):
                        conds = torch.from_numpy(conds).to(device).float()
                    else:
                        conds = conds.to(device).float()
                    # Pass full sequence for cross-attention (model will handle it)
                    # Shape: [batch_size, audio_frames, feature_dim]
                else:
                    conds = conds.to(device).float() if torch.is_tensor(conds) else conds

                if args.distributed:
                    loss = mardm.module.forward_loss(latent, conds, m_lens)
                else:
                    loss = mardm.forward_loss(latent, conds, m_lens)
                val_loss.append(loss.item())

        if is_main_process:
            print(f"Validation loss:{np.mean(val_loss):.3f}")
            if logger is not None:
                logger.add_scalar('Val/loss', np.mean(val_loss), epoch)
            if np.mean(val_loss) < worst_loss:
                print(f"Improved loss from {worst_loss:.02f} to {np.mean(val_loss)}!!!")
                worst_loss = np.mean(val_loss)
            if args.need_evaluation:
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching, _, clip_score, writer, save_now= evaluation_mardm(
                    model_dir, eval_loader, ema_mardm, ae, logger, epoch-1, best_fid=best_fid, clip_score_old=clip_score,
                    best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                    best_matching=best_matching, eval_wrapper=eval_wrapper, device=device, train_mean=mean, train_std=std)
                if save_now:
                    if args.distributed:
                        save(pjoin(model_dir, 'net_best_fid.tar'), epoch-1, mardm.module, optimizer, scheduler,
                             it, 'mardm', ema_mardm=ema_mardm)
                    else:
                        save(pjoin(model_dir, 'net_best_fid.tar'), epoch-1, mardm, optimizer, scheduler,
                             it, 'mardm', ema_mardm=ema_mardm)
    
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='MARDM')
    parser.add_argument('--ae_name', type=str, default="AE")
    parser.add_argument('--ae_model', type=str, default='AE_Model')
    parser.add_argument('--model', type=str, default='MARDM-SiT-XL')
    parser.add_argument('--dataset_name', type=str, default='t2m')
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument("--max_motion_length", type=int, default=196)
    parser.add_argument("--unit_length", type=int, default=4)
    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--warm_up_iter', default=2000, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--milestones', default=[50_000], nargs="+", type=int)
    parser.add_argument('--lr_decay', default=0.1, type=float)

    parser.add_argument('--diffmlps_batch_mul', type=int, default=4)
    parser.add_argument('--need_evaluation', action="store_true" )

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--is_continue', action="store_true")
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')

    parser.add_argument('--log_every', default=50, type=int)
    
    # Distributed training arguments
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--master_port', type=int, default=None, help='Master port for distributed training (default: 29501 or from env)')

    arg = parser.parse_args()
    main(arg)