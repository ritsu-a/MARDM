import numpy as np
import os
from os.path import join as pjoin
from tqdm import tqdm

#################################################################################
#                                Calculate Mean Std                             #
#################################################################################
def mean_variance(data_dir, save_dir, joints_num):
    file_list = os.listdir(data_dir)
    data_list = []

    for file in tqdm(file_list):
        data = np.load(pjoin(data_dir, file))
        if np.isnan(data).any():
            print(file)
            continue
        data_list.append(data[:, :4+(joints_num-1)*3])

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    Std[0:1] = Std[0:1].mean() / 1.0
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    Std[4: 4+(joints_num - 1) * 3] = Std[4: 4+(joints_num - 1) * 3].mean() / 1.0

    np.save(pjoin(save_dir, 'Mean.npy'), Mean)
    np.save(pjoin(save_dir, 'Std.npy'), Std)

    return Mean, Std


def mean_variance_beat_v2(data_root, save_dir):
    """
    Calculate mean and std for BEAT_v2 dataset
    Args:
        data_root: root directory of BEAT_v2 data (e.g., '/root/workspace/MARDM/data/BEAT_v2')
        save_dir: directory to save Mean.npy and Std.npy
    """
    # Find all npz files
    npz_files = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.npz'):
                npz_files.append(os.path.join(root, file))
    
    npz_files.sort()  # Ensure consistent ordering
    print(f"Found {len(npz_files)} npz files in {data_root}")
    
    all_motions = []
    for npz_path in tqdm(npz_files, desc="Loading BEAT_v2 data"):
        try:
            data = np.load(npz_path)
            if 'qpos' in data:
                motion = data['qpos']
            else:
                # Try to get the first array if qpos doesn't exist
                keys = list(data.keys())
                if len(keys) > 0:
                    motion = data[keys[0]]
                else:
                    continue
            
            # Ensure motion is 2D
            if len(motion.shape) == 1:
                motion = motion.reshape(-1, 1)
            
            if np.isnan(motion).any():
                print(f"Warning: NaN values found in {npz_path}, skipping...")
                continue
            
            all_motions.append(motion)
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            continue
    
    if len(all_motions) == 0:
        raise ValueError("No valid motion data found in BEAT_v2 dataset")
    
    # Concatenate all motions
    data = np.concatenate(all_motions, axis=0)
    print(f"Total motion data shape: {data.shape}")
    
    # Calculate mean and std
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    
    # Avoid division by zero
    Std = np.where(Std < 1e-8, 1.0, Std)
    
    # Save mean and std
    os.makedirs(save_dir, exist_ok=True)
    mean_path = pjoin(save_dir, 'Mean.npy')
    std_path = pjoin(save_dir, 'Std.npy')
    
    np.save(mean_path, Mean)
    np.save(std_path, Std)
    
    print(f"Saved Mean to {mean_path}, shape: {Mean.shape}")
    print(f"Saved Std to {std_path}, shape: {Std.shape}")
    print(f"Mean range: [{Mean.min():.4f}, {Mean.max():.4f}]")
    print(f"Std range: [{Std.min():.4f}, {Std.max():.4f}]")
    
    return Mean, Std


if __name__ == '__main__':
    # Calculate for HumanML3D
    # data_dir1 = 'datasets/HumanML3D/new_joint_vecs/'
    # save_dir1 = 'datasets/HumanML3D/'
    # if os.path.exists(data_dir1):
    #     mean, std = mean_variance(data_dir1, save_dir1, 22)
    
    # # Calculate for KIT-ML
    # data_dir2 = 'datasets/KIT-ML/new_joint_vecs/'
    # save_dir2 = 'datasets/KIT-ML/'
    # if os.path.exists(data_dir2):
    #     mean2, std2 = mean_variance(data_dir2, save_dir2, 21)
    
    # Calculate for BEAT_v2
    data_root_beat = '/root/workspace/MARDM/data/BEAT_v2'
    save_dir_beat = '/root/workspace/MARDM/data/BEAT_v2'
    if os.path.exists(data_root_beat):
        print("\n" + "="*50)
        print("Calculating mean and std for BEAT_v2 dataset...")
        print("="*50)
        mean_beat, std_beat = mean_variance_beat_v2(data_root_beat, save_dir_beat)
        print("BEAT_v2 mean and std calculation completed!")
    else:
        print(f"BEAT_v2 data directory not found: {data_root_beat}")