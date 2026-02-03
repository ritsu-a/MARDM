import os
from os.path import join as pjoin
import torch
import torch.distributed as dist
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from models.AE import AE_models
from utils.evaluators import Evaluators
from utils.datasets import AEDataset, Text2MotionDataset, BEAT_v2Dataset, collate_fn
import time
from collections import OrderedDict, defaultdict
from utils.train_utils import update_lr_warm_up, def_value, save, print_current_loss
from utils.eval_utils import evaluation_ae
from tqdm import tqdm
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
    #################################################################################
    #                                    Train Data                                 #
    #################################################################################
    if args.dataset_name == "beat_v2":
        # BEAT_v2 dataset
        data_root = '/root/workspace/MARDM/data/BEAT_v2'
        
        # Load pre-computed mean and std
        mean_path = pjoin(data_root, 'Mean.npy')
        std_path = pjoin(data_root, 'Std.npy')
        
        if os.path.exists(mean_path) and os.path.exists(std_path):
            mean = np.load(mean_path)
            std = np.load(std_path)
            print(f"Loaded mean and std from {mean_path} and {std_path}")
            print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
        else:
            raise FileNotFoundError(
                f"Mean and std files not found for BEAT_v2 dataset.\n"
                f"Please run: python utils/cal_mean_std.py\n"
                f"Expected files: {mean_path} and {std_path}"
            )
        
        dim_pose = mean.shape[0]
        joints_num = dim_pose  # For BEAT_v2, we use the full dimension
        
        train_dataset = BEAT_v2Dataset(mean, std, data_root, args.window_size, split='train')
        val_dataset = BEAT_v2Dataset(mean, std, data_root, args.window_size, split='val')
        
        # For evaluation, we still need Text2MotionDataset but it won't be used for BEAT_v2
        max_motion_length = 180
        eval_mean = mean
        eval_std = std
        
    elif args.dataset_name == "t2m":
        data_root = f'{args.dataset_dir}/HumanML3D/'
        joints_num = 22
        dim_pose = 67
        motion_dir = pjoin(data_root, 'new_joint_vecs')
        text_dir = pjoin(data_root, 'texts')
        max_motion_length = 196
        mean = np.load(pjoin(data_root, 'Mean.npy'))
        std = np.load(pjoin(data_root, 'Std.npy'))
        train_split_file = pjoin(data_root, 'train.txt')
        val_split_file = pjoin(data_root, 'val.txt')

        train_dataset = AEDataset(mean, std, motion_dir, args.window_size, train_split_file)
        val_dataset = AEDataset(mean, std, motion_dir, args.window_size, val_split_file)
        eval_mean = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_mean.npy')
        eval_std = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_std.npy')
    else:
        data_root = f'{args.dataset_dir}/KIT-ML/'
        joints_num = 21
        dim_pose = 64
        motion_dir = pjoin(data_root, 'new_joint_vecs')
        text_dir = pjoin(data_root, 'texts')
        max_motion_length = 196
        mean = np.load(pjoin(data_root, 'Mean.npy'))
        std = np.load(pjoin(data_root, 'Std.npy'))
        train_split_file = pjoin(data_root, 'train.txt')
        val_split_file = pjoin(data_root, 'val.txt')

        train_dataset = AEDataset(mean, std, motion_dir, args.window_size, train_split_file)
        val_dataset = AEDataset(mean, std, motion_dir, args.window_size, val_split_file)
        eval_mean = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_mean.npy')
        eval_std = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_std.npy')

    # Setup distributed sampler if using distributed training
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
        shuffle = False  # Shuffle is handled by DistributedSampler
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                              shuffle=shuffle, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                            shuffle=False, pin_memory=True, sampler=val_sampler)
    #################################################################################
    #                                    Eval Data                                  #
    #################################################################################
    if args.dataset_name == "beat_v2":
        # For BEAT_v2, we skip the text-based evaluation dataset
        eval_loader = None
        eval_wrapper = None
    else:
        split_file = pjoin(data_root, 'val.txt')
        eval_dataset = Text2MotionDataset(eval_mean, eval_std, split_file, args.dataset_name, motion_dir, text_dir,
                                          4, max_motion_length, 20, evaluation=True)
        eval_loader = DataLoader(eval_dataset, batch_size=32, num_workers=args.num_workers, drop_last=True,
                                collate_fn=collate_fn, shuffle=True)
    #################################################################################
    #                                      Models                                   #
    #################################################################################
    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')
    if is_main_process:
        os.makedirs(model_dir, exist_ok=True)

    ae = AE_models[args.model](input_width=dim_pose)

    if is_main_process:
        print(ae)
        pc_vae = sum(param.numel() for param in ae.parameters())
        print('Total parameters of all models: {}M'.format(pc_vae / 1000_000))

    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    ae.to(device)
    
    # Wrap model with DDP if using distributed training
    if args.distributed:
        ae = DDP(ae, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
        ae_model = ae.module  # Access the underlying model
    else:
        ae_model = ae
    
    if args.dataset_name != "beat_v2":
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
    
    if args.recons_loss == 'l1_smooth':
       criterion = torch.nn.SmoothL1Loss()
    else:
        criterion = torch.nn.MSELoss()

    optimizer = optim.AdamW(ae_model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_decay)
    epoch = 0
    it = 0
    if args.is_continue:
        checkpoint_path = pjoin(model_dir, 'latest.tar')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if args.distributed:
            ae_model.load_state_dict(checkpoint['ae'])
        else:
            ae.load_state_dict(checkpoint['ae'])
        optimizer.load_state_dict(checkpoint[f'opt_ae'])
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

    current_lr = args.lr
    logs = defaultdict(def_value, OrderedDict())

    best_fid, best_div, best_top1, best_top2, best_top3, best_matching, mpjpe = 1000, 0, 0, 0, 0, 100, 100

    while epoch < args.epoch:
        if args.distributed:
            train_sampler.set_epoch(epoch)
        ae_model.train()
        for i, batch_data in enumerate(train_loader):
            it += 1
            if it < args.warm_up_iter:
                current_lr = update_lr_warm_up(it, args.warm_up_iter, optimizer, args.lr)

            motions = batch_data.detach().to(device).float()
            pred_motion = ae(motions)

            loss_rec = criterion(pred_motion, motions)
            
            # For BEAT_v2, we use a simpler loss (only reconstruction)
            # For other datasets, we use the original joint-based loss
            if args.dataset_name == "beat_v2":
                loss = loss_rec
                loss_explicit = torch.tensor(0.0)
            else:
                pred_local_pos = pred_motion[..., 4: (joints_num - 1) * 3 + 4]
                local_pos = motions[..., 4: (joints_num - 1) * 3 + 4]
                loss_explicit = criterion(pred_local_pos, local_pos)
                loss = loss_rec + args.aux_loss_joints * loss_explicit

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if it >= args.warm_up_iter:
                scheduler.step()

            logs['loss'] += loss.item()
            logs['loss_rec'] += loss_rec.item()
            logs['loss_vel'] += loss_explicit.item()
            logs['lr'] += optimizer.param_groups[0]['lr']

            if it % args.log_every == 0 and is_main_process:
                mean_loss = OrderedDict()
                for tag, value in logs.items():
                    if logger is not None:
                        logger.add_scalar('Train/%s' % tag, value / args.log_every, it)
                    mean_loss[tag] = value / args.log_every
                logs = defaultdict(def_value, OrderedDict())
                print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

        if is_main_process:
            save(pjoin(model_dir, 'latest.tar'), epoch, ae_model, optimizer, scheduler, it, 'ae')
        
        if args.distributed:
            dist.barrier()  # Wait for all processes to finish epoch

        epoch += 1
        #################################################################################
        #                                      Eval Loop                                #
        #################################################################################
        if is_main_process:
            print('Validation time:')
        ae_model.eval()
        val_loss_rec = []
        val_loss_vel = []
        val_loss = []
        with torch.no_grad():
            for i, batch_data in enumerate(val_loader):
                motions = batch_data.detach().to(device).float()
                pred_motion = ae(motions)

                loss_rec = criterion(pred_motion, motions)
                
                # For BEAT_v2, we use a simpler loss (only reconstruction)
                if args.dataset_name == "beat_v2":
                    loss = loss_rec
                    loss_explicit = torch.tensor(0.0)
                else:
                    pred_local_pos = pred_motion[..., 4: (joints_num - 1) * 3 + 4]
                    local_pos = motions[..., 4: (joints_num - 1) * 3 + 4]
                    loss_explicit = criterion(pred_local_pos, local_pos)
                    loss = loss_rec + args.aux_loss_joints * loss_explicit

                val_loss.append(loss.item())
                val_loss_rec.append(loss_rec.item())
                val_loss_vel.append(loss_explicit.item())

        if is_main_process:
            if logger is not None:
                logger.add_scalar('Val/loss', sum(val_loss) / len(val_loss), epoch)
                logger.add_scalar('Val/loss_rec', sum(val_loss_rec) / len(val_loss_rec), epoch)
                logger.add_scalar('Val/loss_vel', sum(val_loss_vel) / len(val_loss_vel), epoch)
            print('Validation Loss: %.5f, Reconstruction: %.5f, Velocity: %.5f,' %
                  (sum(val_loss) / len(val_loss), sum(val_loss_rec) / len(val_loss), sum(val_loss_vel) / len(val_loss)))

        if args.dataset_name != "beat_v2" and eval_loader is not None and is_main_process:
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, mpjpe, writer = evaluation_ae(
                model_dir, eval_loader, ae_model, logger, epoch-1, device=device, num_joint=joints_num, best_fid=best_fid,
                best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                train_mean=mean, train_std=std, best_matching=best_matching, eval_wrapper=eval_wrapper)
            print(f'best fid {best_fid}')
        elif args.dataset_name == "beat_v2" and is_main_process:
            print("Skipping evaluation for BEAT_v2 dataset")
    
    if args.distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='AE')
    parser.add_argument('--model', type=str, default='AE_Model')
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument('--dataset_name', type=str, default='t2m')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--window_size', type=int, default=64)

    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--warm_up_iter', default=2000, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--milestones', default=[150000, 250000], nargs="+", type=int)
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--aux_loss_joints', type=float, default=1)
    parser.add_argument('--recons_loss', type=str, default='l1_smooth')

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--is_continue', action="store_true")
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')

    parser.add_argument('--log_every', default=10, type=int)
    
    # Distributed training arguments
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--master_port', type=int, default=None, help='Master port for distributed training (default: 29501 or from env)')

    arg = parser.parse_args()
    main(arg)