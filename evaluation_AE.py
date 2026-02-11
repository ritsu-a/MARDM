import os
from os.path import join as pjoin
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from models.AE import AE_models
from utils.evaluators import Evaluators
from utils.datasets import Text2MotionDataset, BEAT_v2Dataset, MixedDataset, collate_fn
from utils.eval_utils import evaluation_ae
import warnings
warnings.filterwarnings('ignore')
import argparse
from tqdm import tqdm

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
    if args.dataset_name == "beat_v2" or args.dataset_name == "mixed":
        # Mixed dataset (BEAT_v2 + semi_synthetic_v1_segments) or BEAT_v2 only
        beat_v2_root = '/root/workspace/MARDM/data/BEAT_v2'
        semi_synthetic_root = '/root/workspace/MARDM/data/semi_synthetic_v1_segments'
        
        mean = np.load(pjoin(beat_v2_root, 'Mean.npy'))
        std = np.load(pjoin(beat_v2_root, 'Std.npy'))
        dim_pose = mean.shape[0]
        joints_num = dim_pose
        
        if args.dataset_name == "mixed":
            eval_dataset = MixedDataset(mean, std, beat_v2_root, semi_synthetic_root, args.window_size, split='val')
        else:
            eval_dataset = BEAT_v2Dataset(mean, std, beat_v2_root, args.window_size, split='val')
        
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                                shuffle=False, pin_memory=True)
        
        eval_wrapper = None  # No text-based evaluation for BEAT_v2/mixed
        
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
        eval_dataset = Text2MotionDataset(eval_mean, eval_std, split_file, args.dataset_name, motion_dir, text_dir,
                                          4, max_motion_length, 20, evaluation=True)
        eval_loader = DataLoader(eval_dataset, batch_size=32, num_workers=args.num_workers, drop_last=True,
                                collate_fn=collate_fn, shuffle=True)
        eval_wrapper = Evaluators(args.dataset_name, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
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
        eval_dataset = Text2MotionDataset(eval_mean, eval_std, split_file, args.dataset_name, motion_dir, text_dir,
                                          4, max_motion_length, 20, evaluation=True)
        eval_loader = DataLoader(eval_dataset, batch_size=32, num_workers=args.num_workers, drop_last=True,
                                collate_fn=collate_fn, shuffle=True)
        eval_wrapper = Evaluators(args.dataset_name, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    #################################################################################
    #                                      Models                                   #
    #################################################################################
    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')

    ae = AE_models[args.model](input_width=dim_pose)
    checkpoint_path = os.path.join(model_dir, 'latest.tar' if args.dataset_name in ['t2m', 'beat_v2', 'mixed'] else 'net_best_fid.tar')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    ae.load_state_dict(checkpoint['ae'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #################################################################################
    #                                  Evaluation Loop                              #
    #################################################################################
    out_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'eval')
    os.makedirs(out_dir, exist_ok=True)
    f = open(pjoin(out_dir, 'eval.log'), 'w')

    ae.eval()
    ae.to(device)

    if args.dataset_name == "beat_v2" or args.dataset_name == "mixed":
        # Simplified evaluation for BEAT_v2/mixed (no text-based metrics)
        fid = []
        div = []
        top1 = []
        top2 = []
        top3 = []
        matching = []
        mae = []
        reconstruction_losses = []
        repeat_time = 20
        
        criterion = torch.nn.MSELoss()
        criterion_l1 = torch.nn.L1Loss()
        
        for i in range(repeat_time):
            print(f"Evaluation run {i+1}/{repeat_time}")
            total_loss = 0.0
            total_l1_loss = 0.0
            total_samples = 0
            
            with torch.no_grad():
                for batch_data in tqdm(eval_loader, desc=f"Eval run {i+1}"):
                    motions = batch_data.detach().to(device).float()
                    pred_motion = ae(motions)
                    
                    loss = criterion(pred_motion, motions)
                    l1_loss = criterion_l1(pred_motion, motions)
                    
                    total_loss += loss.item() * motions.shape[0]
                    total_l1_loss += l1_loss.item() * motions.shape[0]
                    total_samples += motions.shape[0]
            
            avg_loss = total_loss / total_samples
            avg_l1_loss = total_l1_loss / total_samples
            
            reconstruction_losses.append(avg_loss)
            # For BEAT_v2, we don't have text-based metrics, so set dummy values
            fid.append(0.0)
            div.append(0.0)
            top1.append(0.0)
            top2.append(0.0)
            top3.append(0.0)
            matching.append(0.0)
            mae.append(avg_l1_loss)
    else:
        # Original evaluation for t2m and kit datasets
        fid = []
        div = []
        top1 = []
        top2 = []
        top3 = []
        matching = []
        mae = []
        repeat_time = 20
        for i in range(repeat_time):
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, mpjpe = 1000, 0, 0, 0, 0, 100, 100
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, mpjpe, writer = evaluation_ae(
                model_dir, eval_loader, ae, None, i, device=device, num_joint=joints_num, best_fid=best_fid,
                best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                train_mean=mean, train_std=std, best_matching=best_matching, eval_wrapper=eval_wrapper,
                save=False, draw=False)
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

    if args.dataset_name == "beat_v2" or args.dataset_name == "mixed":
        reconstruction_losses = np.array(reconstruction_losses)
        msg_final = f"\tReconstruction Loss (MSE): {np.mean(reconstruction_losses):.6f}, conf. {np.std(reconstruction_losses) * 1.96 / np.sqrt(repeat_time):.6f}\n" \
                    f"\tReconstruction Loss (L1/MAE): {np.mean(mae):.6f}, conf. {np.std(mae) * 1.96 / np.sqrt(repeat_time):.6f}\n" \
                    f"\tNote: BEAT_v2 dataset does not have text annotations, so text-based metrics (FID, Diversity, R-precision) are not available.\n\n"
    else:
        msg_final = f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tMAE:{np.mean(mae):.3f}, conf.{np.std(mae) * 1.96 / np.sqrt(repeat_time):.3f}\n\n"

    print(msg_final)
    print(msg_final, file=f, flush=True)
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='AE')
    parser.add_argument('--model', type=str, default='AE_Model')
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument('--dataset_name', type=str, default='t2m')
    parser.add_argument('--window_size', type=int, default=64, help='Window size for BEAT_v2 dataset')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for evaluation')

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')

    arg = parser.parse_args()
    main(arg)