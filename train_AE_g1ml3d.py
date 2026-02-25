import os
from os.path import join as pjoin
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from models.AE import AE_models
from utils.evaluators import Evaluators
from utils.datasets import G1ML3DAEDataset, G1ML3DText2MotionDataset, collate_fn
import time
from collections import OrderedDict, defaultdict
from utils.train_utils import update_lr_warm_up, def_value, save, print_current_loss
from utils.eval_utils import evaluation_ae
import argparse

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
    #                                    Train Data                                 #
    #################################################################################
    # G1ML3D dataset configuration
    data_root = args.dataset_dir
    motion_dir = pjoin(data_root, 'joints_npz')
    text_dir = pjoin(data_root, 'texts')
    max_motion_length = args.max_motion_length
    
    # Load mean and std (you need to compute these for G1ML3D dataset)
    # If not available, compute from training data
    mean_path = pjoin(data_root, 'Mean.npy')
    std_path = pjoin(data_root, 'Std.npy')
    
    mean = np.load(mean_path)
    std = np.load(std_path)
   
    train_split_file = pjoin(data_root, 'train.txt')
    val_split_file = pjoin(data_root, 'val.txt')

    train_dataset = G1ML3DAEDataset(mean, std, motion_dir, args.window_size, train_split_file)
    val_dataset = G1ML3DAEDataset(mean, std, motion_dir, args.window_size, val_split_file)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                            shuffle=True, pin_memory=True)
    #################################################################################
    #                                    Eval Data                                  #
    #################################################################################
    # For evaluation, use G1ML3DText2MotionDataset
    eval_mean = mean  # Use training mean/std for evaluation
    eval_std = std
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
    # Estimate joints_num (approximate, may need adjustment based on your data)
    # For G1ML3D, joints_num might be different, adjust as needed
    joints_num = args.joints_num if hasattr(args, 'joints_num') else 22
    
    ae = AE_models[args.model](input_width=dim_pose)

    print(ae)
    pc_vae = sum(param.numel() for param in ae.parameters())
    print('Total parameters of all models: {}M'.format(pc_vae / 1000_000))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Note: Evaluators might need adjustment for G1ML3D
    # For now, use a placeholder or create G1ML3D-specific evaluator
    eval_wrapper = None  # Will skip evaluation metrics that require evaluator
    #################################################################################
    #                                    Training Loop                              #
    #################################################################################
    logger = SummaryWriter(model_dir)
    if args.recons_loss == 'l1_smooth':
       criterion = torch.nn.SmoothL1Loss()
    else:
        criterion = torch.nn.MSELoss()

    ae.to(device)
    optimizer = optim.AdamW(ae.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_decay)
    epoch = 0
    it = 0
    if args.is_continue:
        model_path = pjoin(model_dir, 'latest.tar')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            ae.load_state_dict(checkpoint['ae'])
            optimizer.load_state_dict(checkpoint[f'opt_ae'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            epoch, it = checkpoint['ep'], checkpoint['total_it']
            print("Load model epoch:%d iterations:%d" % (epoch, it))
        else:
            print(f"Checkpoint not found at {model_path}, starting from scratch")

    start_time = time.time()
    total_iters = args.epoch * len(train_loader)
    print(f'Total Epochs: {args.epoch}, Total Iters: {total_iters}')
    print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))

    current_lr = args.lr
    logs = defaultdict(def_value, OrderedDict())

    best_fid, best_div, best_top1, best_top2, best_top3, best_matching, mpjpe = 1000, 0, 0, 0, 0, 100, 100

    while epoch < args.epoch:
        ae.train()
        for i, batch_data in enumerate(train_loader):
            it += 1
            if it < args.warm_up_iter:
                current_lr = update_lr_warm_up(it, args.warm_up_iter, optimizer, args.lr)

            motions = batch_data.detach().to(device).float()
            pred_motion = ae(motions)

            loss_rec = criterion(pred_motion, motions)
            # For G1ML3D, adjust joint position indices if needed
            # Assuming similar structure: [root_pos(3), root_rot(1), joints(...)]
            if dim_pose > 4:
                pred_local_pos = pred_motion[..., 4:]
                local_pos = motions[..., 4:]
                loss_explicit = criterion(pred_local_pos, local_pos)
            else:
                loss_explicit = torch.tensor(0.0).to(device)
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

            if it % args.log_every == 0:
                mean_loss = OrderedDict()
                for tag, value in logs.items():
                    logger.add_scalar('Train/%s' % tag, value / args.log_every, it)
                    mean_loss[tag] = value / args.log_every
                logs = defaultdict(def_value, OrderedDict())
                print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

        save(pjoin(model_dir, 'latest.tar'), epoch, ae, optimizer, scheduler, it, 'ae')
        
        # Save checkpoint every 500 epochs
        if (epoch + 1) % 500 == 0:
            checkpoint_name = f'checkpoint_epoch_{epoch+1}.tar'
            save(pjoin(model_dir, checkpoint_name), epoch, ae, optimizer, scheduler, it, 'ae')
            print(f"Saved checkpoint: {checkpoint_name}")

        epoch += 1
        #################################################################################
        #                                      Eval Loop                                #
        #################################################################################
        print('Validation time:')
        ae.eval()
        val_loss_rec = []
        val_loss_vel = []
        val_loss = []
        with torch.no_grad():
            for i, batch_data in enumerate(val_loader):
                motions = batch_data.detach().to(device).float()
                pred_motion = ae(motions)

                loss_rec = criterion(pred_motion, motions)
                if dim_pose > 4:
                    pred_local_pos = pred_motion[..., 4:]
                    local_pos = motions[..., 4:]
                    loss_explicit = criterion(pred_local_pos, local_pos)
                else:
                    loss_explicit = torch.tensor(0.0).to(device)
                loss = loss_rec + args.aux_loss_joints * loss_explicit

                val_loss.append(loss.item())
                val_loss_rec.append(loss_rec.item())
                val_loss_vel.append(loss_explicit.item())

        logger.add_scalar('Val/loss', sum(val_loss) / len(val_loss), epoch)
        logger.add_scalar('Val/loss_rec', sum(val_loss_rec) / len(val_loss_rec), epoch)
        logger.add_scalar('Val/loss_vel', sum(val_loss_vel) / len(val_loss_vel), epoch)
        print('Validation Loss: %.5f, Reconstruction: %.5f, Velocity: %.5f,' %
              (sum(val_loss) / len(val_loss), sum(val_loss_rec) / len(val_loss), sum(val_loss_vel) / len(val_loss)))

        # Evaluation with text-to-motion dataset
        if eval_wrapper is not None:
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, mpjpe, writer = evaluation_ae(
                model_dir, eval_loader, ae, logger, epoch-1, device=device, num_joint=joints_num, best_fid=best_fid,
                best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                train_mean=mean, train_std=std, best_matching=best_matching, eval_wrapper=eval_wrapper)
            print(f'best fid {best_fid}')
        else:
            print("Skipping evaluation metrics (eval_wrapper not available)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='AE')
    parser.add_argument('--model', type=str, default='AE_Model')
    parser.add_argument('--dataset_dir', type=str, default='./data/G1ML3D_v1', 
                        help='Root directory of G1ML3D dataset')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--max_motion_length', type=int, default=196, 
                        help='Maximum motion length for evaluation')

    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--warm_up_iter', default=2000, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--milestones', default=[150000, 250000], nargs="+", type=int)
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--aux_loss_joints', type=float, default=1)
    parser.add_argument('--recons_loss', type=str, default='l1_smooth')
    parser.add_argument('--joints_num', type=int, default=22, 
                        help='Number of joints (adjust based on G1ML3D structure)')

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--is_continue', action="store_true")
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')

    parser.add_argument('--log_every', default=10, type=int)

    arg = parser.parse_args()
    main(arg)
