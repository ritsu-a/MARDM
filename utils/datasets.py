from os.path import join as pjoin
from torch.utils import data
import numpy as np
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import random
import codecs as cs
import os
import glob
from utils.glove import GloVe

#################################################################################
#                                  Collate Function                             #
#################################################################################
def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

#################################################################################
#                                      Datasets                                 #
#################################################################################
class AEDataset(data.Dataset):
    def __init__(self, mean, std, motion_dir, window_size, split_file):
        self.data = []
        self.lengths = []
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(motion_dir, name + '.npy'))
                if motion.shape[0] < window_size:
                    continue
                self.lengths.append(motion.shape[0] - window_size)
                self.data.append(motion)
            except Exception as e:
                pass
        self.cumsum = np.cumsum([0] + self.lengths)
        self.window_size = window_size

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.window_size]
        "Z Normalization"
        motion = motion[:, :self.mean.shape[0]]
        motion = (motion - self.mean) / self.std

        return motion


class BEAT_v2Dataset(data.Dataset):
    def __init__(self, mean, std, data_root, window_size, split='train', train_ratio=0.9):
        """
        BEAT_v2 Dataset for VAE training
        Args:
            mean: mean for normalization
            std: std for normalization
            data_root: root directory of BEAT_v2 data (e.g., '/root/workspace/MARDM/data/BEAT_v2')
            window_size: window size for training
            split: 'train' or 'val'
            train_ratio: ratio of training data
        """
        self.data = []
        self.lengths = []
        self.window_size = window_size
        self.mean = mean
        self.std = std
        
        # Find all npz files
        npz_files = []
        for root, dirs, files in os.walk(data_root):
            for file in files:
                if file.endswith('.npz'):
                    npz_files.append(os.path.join(root, file))
        
        npz_files.sort()  # Ensure consistent ordering
        random.seed(42)  # Fixed seed for reproducibility
        random.shuffle(npz_files)
        
        # Split train/val
        split_idx = int(len(npz_files) * train_ratio)
        if split == 'train':
            npz_files = npz_files[:split_idx]
        else:
            npz_files = npz_files[split_idx:]
        
        print(f"Loading {split} data from {len(npz_files)} files...")
        
        for npz_path in tqdm(npz_files):
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
                
                if motion.shape[0] < window_size:
                    continue
                
                # Ensure motion is 2D
                if len(motion.shape) == 1:
                    motion = motion.reshape(-1, 1)
                
                self.lengths.append(motion.shape[0] - window_size)
                self.data.append(motion)
            except Exception as e:
                print(f"Error loading {npz_path}: {e}")
                continue
        
        self.cumsum = np.cumsum([0] + self.lengths)
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))
    
    def __len__(self):
        return self.cumsum[-1]
    
    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        
        motion = self.data[motion_id][idx:idx + self.window_size]
        
        # Z Normalization
        # Handle dimension mismatch
        min_dim = min(motion.shape[1], self.mean.shape[0])
        motion = motion[:, :min_dim]
        mean_subset = self.mean[:min_dim]
        std_subset = self.std[:min_dim]
        
        # Avoid division by zero
        std_subset = np.where(std_subset < 1e-8, 1.0, std_subset)
        
        motion = (motion - mean_subset) / std_subset
        
        # Pad if necessary
        if motion.shape[1] < self.mean.shape[0]:
            padding = np.zeros((motion.shape[0], self.mean.shape[0] - motion.shape[1]))
            motion = np.concatenate([motion, padding], axis=1)
        
        return motion


class MixedDataset(data.Dataset):
    """
    Mixed dataset combining BEAT_v2 and semi_synthetic_v1_segments datasets
    """
    def __init__(self, mean, std, beat_v2_root, semi_synthetic_root, window_size, split='train', train_ratio=0.9):
        """
        Mixed Dataset for VAE training
        Args:
            mean: mean for normalization (should be computed from combined datasets)
            std: std for normalization (should be computed from combined datasets)
            beat_v2_root: root directory of BEAT_v2 data
            semi_synthetic_root: root directory of semi_synthetic_v1_segments data
            window_size: window size for training
            split: 'train' or 'val'
            train_ratio: ratio of training data
        """
        self.data = []
        self.lengths = []
        self.window_size = window_size
        self.mean = mean
        self.std = std
        
        # Collect npz files from both datasets
        npz_files = []
        
        # From BEAT_v2
        if os.path.exists(beat_v2_root):
            for root, dirs, files in os.walk(beat_v2_root):
                for file in files:
                    if file.endswith('.npz') and not file.endswith('_whisper_features.npy'):
                        npz_files.append(('beat_v2', os.path.join(root, file)))
        
        # From semi_synthetic_v1_segments (only *_motion.npz files)
        if os.path.exists(semi_synthetic_root):
            for root, dirs, files in os.walk(semi_synthetic_root):
                for file in files:
                    if file.endswith('_motion.npz'):
                        npz_files.append(('semi_synthetic', os.path.join(root, file)))
        
        npz_files.sort(key=lambda x: x[1])  # Sort by path for reproducibility
        random.seed(42)  # Fixed seed for reproducibility
        random.shuffle(npz_files)
        
        # Split train/val
        split_idx = int(len(npz_files) * train_ratio)
        if split == 'train':
            npz_files = npz_files[:split_idx]
        else:
            npz_files = npz_files[split_idx:]
        
        print(f"Loading {split} data from {len(npz_files)} files (BEAT_v2 + semi_synthetic_v1_segments)...")
        
        beat_v2_count = sum(1 for dataset_type, path in npz_files if dataset_type == 'beat_v2')
        semi_synthetic_count = len(npz_files) - beat_v2_count
        print(f"  - BEAT_v2: {beat_v2_count} files")
        print(f"  - semi_synthetic_v1_segments: {semi_synthetic_count} files")
        
        for dataset_type, npz_path in tqdm(npz_files, desc=f"Loading {split} data"):
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
                
                if motion.shape[0] < window_size:
                    continue
                
                # Ensure motion is 2D
                if len(motion.shape) == 1:
                    motion = motion.reshape(-1, 1)
                
                self.lengths.append(motion.shape[0] - window_size)
                self.data.append(motion)
            except Exception as e:
                print(f"Error loading {npz_path}: {e}")
                continue
        
        self.cumsum = np.cumsum([0] + self.lengths)
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))
    
    def __len__(self):
        return self.cumsum[-1]
    
    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        
        motion = self.data[motion_id][idx:idx + self.window_size]
        
        # Z Normalization
        # Handle dimension mismatch
        min_dim = min(motion.shape[1], self.mean.shape[0])
        motion = motion[:, :min_dim]
        mean_subset = self.mean[:min_dim]
        std_subset = self.std[:min_dim]
        
        # Avoid division by zero
        std_subset = np.where(std_subset < 1e-8, 1.0, std_subset)
        
        motion = (motion - mean_subset) / std_subset
        
        # Pad if necessary
        if motion.shape[1] < self.mean.shape[0]:
            padding = np.zeros((motion.shape[0], self.mean.shape[0] - motion.shape[1]))
            motion = np.concatenate([motion, padding], axis=1)
        
        return motion


class BEAT_v2Audio2MotionDataset(data.Dataset):
    def __init__(self, mean, std, data_root, unit_length, max_motion_length, split='train', train_ratio=0.9):
        """
        BEAT_v2 Dataset for Audio-to-Motion training (MARDM)
        Args:
            mean: mean for normalization
            std: std for normalization
            data_root: root directory of BEAT_v2 data (e.g., '/root/workspace/MARDM/data/BEAT_v2')
            unit_length: unit length for motion (default 4)
            max_motion_length: maximum motion length (should be 300)
            split: 'train' or 'val'
            train_ratio: ratio of training data
        """
        self.mean = mean
        self.std = std
        self.unit_length = unit_length
        self.max_motion_length = max_motion_length  # Should be 300
        
        # Frame alignment: 50 audio frames = 60 motion frames
        # Ratio: audio_frames / motion_frames = 50/60 = 5/6
        # For 300 motion frames, we need 300 * 5/6 = 250 audio frames
        # New logic: 250 audio frames + 60 motion frames (condition) -> 240 motion frames (target)
        self.target_motion_frames = max_motion_length  # 300 frames total
        self.condition_motion_frames = 60  # First 60 frames as condition
        self.target_motion_frames_generate = self.target_motion_frames - self.condition_motion_frames  # 240 frames to generate
        self.target_audio_frames = int(self.target_motion_frames * 5 / 6)  # 250 for 300 motion frames
        self.audio_to_motion_ratio = 5.0 / 6.0  # 50/60
        
        # Minimum length requirement: need at least target frames
        min_motion_len = self.target_motion_frames
        max_motion_len = 50000  # BEAT_v2 sequences can be very long
        
        # Find all npz files (motion data)
        npz_files = []
        for root, dirs, files in os.walk(data_root):
            for file in files:
                if file.endswith('.npz') and not file.endswith('_whisper_features.npy'):
                    npz_files.append(os.path.join(root, file))
        
        npz_files.sort()  # Ensure consistent ordering
        random.seed(42)  # Fixed seed for reproducibility
        random.shuffle(npz_files)
        
        # Split train/val
        split_idx = int(len(npz_files) * train_ratio)
        if split == 'train':
            npz_files = npz_files[:split_idx]
        else:
            npz_files = npz_files[split_idx:]
        
        print(f"Loading {split} data from {len(npz_files)} files...")
        
        self.data_dict = {}
        self.name_list = []
        self.length_list = []
        
        skipped_no_whisper = 0
        skipped_length = 0
        skipped_error = 0
        
        for npz_path in tqdm(npz_files):
            try:
                # Load motion data
                motion_data = np.load(npz_path)
                if 'qpos' in motion_data:
                    motion = motion_data['qpos']
                else:
                    keys = list(motion_data.keys())
                    if len(keys) > 0:
                        motion = motion_data[keys[0]]
                    else:
                        skipped_error += 1
                        continue
                
                if len(motion.shape) == 1:
                    motion = motion.reshape(-1, 1)
                
                motion_len = motion.shape[0]
                if motion_len < min_motion_len or motion_len >= max_motion_len:
                    skipped_length += 1
                    continue
                
                # Find corresponding whisper feature file
                # npz file: /path/to/1_wayne_0_1_1.npz
                # whisper file: /path/to/1_wayne_0_1_1_whisper_features.npy
                npz_dir = os.path.dirname(npz_path)
                npz_basename = os.path.basename(npz_path)
                npz_stem = os.path.splitext(npz_basename)[0]
                
                # Try to find whisper feature file
                whisper_path = os.path.join(npz_dir, f"{npz_stem}_whisper_features.npy")
                
                if not os.path.exists(whisper_path):
                    skipped_no_whisper += 1
                    if skipped_no_whisper <= 5:  # Only print first 5 warnings
                        print(f"Warning: No whisper feature found for {npz_path}, expected: {whisper_path}")
                    continue
                
                # Load whisper features
                try:
                    whisper_features = np.load(whisper_path)
                    # whisper_features shape: [time_frames, feature_dim]
                    if len(whisper_features.shape) != 2:
                        skipped_error += 1
                        if skipped_error <= 5:
                            print(f"Warning: Invalid whisper feature shape {whisper_features.shape} for {whisper_path}")
                        continue
                except Exception as e:
                    skipped_error += 1
                    if skipped_error <= 5:
                        print(f"Error loading whisper features from {whisper_path}: {e}")
                    continue
                
                # Verify alignment: check if audio and motion lengths match the expected ratio
                motion_len = motion.shape[0]
                audio_len = whisper_features.shape[0]
                expected_audio_len = int(motion_len * self.audio_to_motion_ratio)
                
                # Allow some tolerance (within 5% difference)
                tolerance = 0.05
                if abs(audio_len - expected_audio_len) / max(expected_audio_len, 1) > tolerance:
                    skipped_error += 1
                    if skipped_error <= 5:
                        print(f"Warning: Audio-motion length mismatch for {npz_path}: "
                              f"motion={motion_len}, audio={audio_len}, expected_audio={expected_audio_len:.0f}")
                    continue
                
                # Check if we have enough frames
                # Need at least target_motion_frames (300) for motion and target_audio_frames (250) for audio
                if motion_len < self.target_motion_frames or audio_len < self.target_audio_frames:
                    skipped_length += 1
                    continue
                
                # Generate multiple samples from long sequences by sliding window
                # For sequences longer than target, we create multiple samples
                num_samples = (motion_len - self.target_motion_frames) // (self.target_motion_frames // 2) + 1
                num_samples = min(num_samples, 10)  # Limit to 10 samples per sequence to avoid too many duplicates
                
                for sample_idx in range(num_samples):
                    # Calculate start position
                    if num_samples == 1:
                        start_motion = 0
                    else:
                        max_start = motion_len - self.target_motion_frames
                        start_motion = int(sample_idx * max_start / (num_samples - 1))
                    
                    start_audio = int(start_motion * self.audio_to_motion_ratio)
                    
                    # Extract fixed-length segments
                    end_motion = start_motion + self.target_motion_frames
                    end_audio = start_audio + self.target_audio_frames
                    
                    if end_motion > motion_len or end_audio > audio_len:
                        continue
                    
                    motion_segment = motion[start_motion:end_motion]  # [300, dim]
                    audio_segment = whisper_features[start_audio:end_audio]  # [250, feature_dim]
                    
                    # Normalize motion
                    motion_segment = motion_segment[:, :self.mean.shape[0]]
                    motion_segment = (motion_segment - self.mean) / self.std
                    
                    # Pad motion if necessary (shouldn't happen, but just in case)
                    if motion_segment.shape[0] < self.target_motion_frames:
                        padding = np.zeros((self.target_motion_frames - motion_segment.shape[0], motion_segment.shape[1]))
                        motion_segment = np.concatenate([motion_segment, padding], axis=0)
                    
                    # Split motion into condition (first 60 frames) and target (last 240 frames)
                    motion_condition = motion_segment[:self.condition_motion_frames]  # [60, dim]
                    motion_target = motion_segment[self.condition_motion_frames:]  # [240, dim]
                    
                    # Store data
                    name = f"{npz_stem}_sample{sample_idx}"
                    self.data_dict[name] = {
                        'motion': motion_segment.astype(np.float32),  # Full 300 frames for compatibility
                        'motion_condition': motion_condition.astype(np.float32),  # First 60 frames as condition
                        'motion_target': motion_target.astype(np.float32),  # Last 240 frames as target
                        'whisper': audio_segment.astype(np.float32),
                        'length': self.target_motion_frames  # Total length: 300
                    }
                    self.name_list.append(name)
                    self.length_list.append(self.target_motion_frames)
                
            except Exception as e:
                skipped_error += 1
                if skipped_error <= 5:
                    print(f"Error loading {npz_path}: {e}")
                continue
        
        self.length_arr = np.array(self.length_list)
        print(f"Total number of samples: {len(self.data_dict)}")
        print(f"Skipped: {skipped_no_whisper} (no whisper), {skipped_length} (length), {skipped_error} (error)")
    
    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, item):
        """
        Returns:
            whisper_features: [target_audio_frames, feature_dim] - fixed size audio features sequence (250 frames)
            motion_condition: [condition_motion_frames, motion_dim] - first 60 frames as condition
            motion_target: [target_motion_frames_generate, motion_dim] - last 240 frames as target
            m_length: int - total motion length (300)
        """
        name = self.name_list[item]
        data = self.data_dict[name]
        motion_condition = data['motion_condition'].copy()  # [60, dim]
        motion_target = data['motion_target'].copy()  # [240, dim]
        whisper_features = data['whisper'].copy()  # [250, feature_dim]
        m_length = data['length']  # Always 300
        
        # Return: audio features, condition motion (60 frames), target motion (240 frames), total length
        # Shape: whisper_features [250, feature_dim], motion_condition [60, motion_dim], motion_target [240, motion_dim]
        return whisper_features, motion_condition, motion_target, m_length


class MixedAudioTextDataset(data.Dataset):
    """
    Mixed dataset combining BEAT_v2 (audio-to-motion) and semi_synthetic_v1_segments (text-to-motion)
    For BEAT_v2: uses audio (whisper) features, CLIP features are padded with zeros
    For semi-synthetic: uses text CLIP features, audio features are also available
    """
    def __init__(self, mean, std, beat_v2_root, semi_synthetic_root, unit_length, max_motion_length, split='train', train_ratio=0.9):
        """
        Mixed Dataset for Audio+Text-to-Motion training (MARDM)
        Args:
            mean: mean for normalization
            std: std for normalization
            beat_v2_root: root directory of BEAT_v2 data
            semi_synthetic_root: root directory of semi_synthetic_v1_segments data
            unit_length: unit length for motion (default 4)
            max_motion_length: maximum motion length (should be 300)
            split: 'train' or 'val'
            train_ratio: ratio of training data
        """
        self.mean = mean
        self.std = std
        self.unit_length = unit_length
        self.max_motion_length = max_motion_length  # Should be 300
        self.clip_dim = 512  # CLIP feature dimension
        
        # Frame alignment: same as BEAT_v2Audio2MotionDataset
        self.target_motion_frames = max_motion_length  # 300 frames total
        self.condition_motion_frames = 60  # First 60 frames as condition
        self.target_motion_frames_generate = self.target_motion_frames - self.condition_motion_frames  # 240 frames to generate
        self.target_audio_frames = int(self.target_motion_frames * 5 / 6)  # 250 for 300 motion frames
        self.audio_to_motion_ratio = 5.0 / 6.0  # 50/60
        
        min_motion_len = self.target_motion_frames
        max_motion_len = 50000
        
        # Collect data from both datasets
        self.data_dict = {}
        self.name_list = []
        self.length_list = []
        self.dataset_type_list = []  # Track which dataset each sample comes from
        
        skipped_no_whisper = 0
        skipped_length = 0
        skipped_error = 0
        
        # Load BEAT_v2 data
        beat_v2_npz_files = []
        if os.path.exists(beat_v2_root):
            for root, dirs, files in os.walk(beat_v2_root):
                for file in files:
                    if file.endswith('.npz') and not file.endswith('_whisper_features.npy'):
                        beat_v2_npz_files.append(os.path.join(root, file))
        
        beat_v2_npz_files.sort()
        random.seed(42)
        random.shuffle(beat_v2_npz_files)
        
        split_idx = int(len(beat_v2_npz_files) * train_ratio)
        if split == 'train':
            beat_v2_npz_files = beat_v2_npz_files[:split_idx]
        else:
            beat_v2_npz_files = beat_v2_npz_files[split_idx:]
        
        print(f"Loading BEAT_v2 {split} data from {len(beat_v2_npz_files)} files...")
        
        # Process BEAT_v2 files (same as BEAT_v2Audio2MotionDataset)
        for npz_path in tqdm(beat_v2_npz_files, desc="Loading BEAT_v2"):
            try:
                motion_data = np.load(npz_path)
                if 'qpos' in motion_data:
                    motion = motion_data['qpos']
                else:
                    keys = list(motion_data.keys())
                    if len(keys) > 0:
                        motion = motion_data[keys[0]]
                    else:
                        skipped_error += 1
                        continue
                
                if len(motion.shape) == 1:
                    motion = motion.reshape(-1, 1)
                
                motion_len = motion.shape[0]
                if motion_len < min_motion_len or motion_len >= max_motion_len:
                    skipped_length += 1
                    continue
                
                # Find whisper feature file
                npz_dir = os.path.dirname(npz_path)
                npz_basename = os.path.basename(npz_path)
                npz_stem = os.path.splitext(npz_basename)[0]
                whisper_path = os.path.join(npz_dir, f"{npz_stem}_whisper_features.npy")
                
                if not os.path.exists(whisper_path):
                    skipped_no_whisper += 1
                    continue
                
                try:
                    whisper_features = np.load(whisper_path)
                    if len(whisper_features.shape) != 2:
                        skipped_error += 1
                        continue
                except Exception as e:
                    skipped_error += 1
                    continue
                
                audio_len = whisper_features.shape[0]
                expected_audio_len = int(motion_len * self.audio_to_motion_ratio)
                tolerance = 0.05
                if abs(audio_len - expected_audio_len) / max(expected_audio_len, 1) > tolerance:
                    skipped_error += 1
                    continue
                
                if motion_len < self.target_motion_frames or audio_len < self.target_audio_frames:
                    skipped_length += 1
                    continue
                
                # Generate samples
                num_samples = (motion_len - self.target_motion_frames) // (self.target_motion_frames // 2) + 1
                num_samples = min(num_samples, 10)
                
                for sample_idx in range(num_samples):
                    if num_samples == 1:
                        start_motion = 0
                    else:
                        max_start = motion_len - self.target_motion_frames
                        start_motion = int(sample_idx * max_start / (num_samples - 1))
                    
                    start_audio = int(start_motion * self.audio_to_motion_ratio)
                    end_motion = start_motion + self.target_motion_frames
                    end_audio = start_audio + self.target_audio_frames
                    
                    if end_motion > motion_len or end_audio > audio_len:
                        continue
                    
                    motion_segment = motion[start_motion:end_motion]
                    audio_segment = whisper_features[start_audio:end_audio]
                    
                    # Normalize motion
                    motion_segment = motion_segment[:, :self.mean.shape[0]]
                    motion_segment = (motion_segment - self.mean) / self.std
                    
                    if motion_segment.shape[0] < self.target_motion_frames:
                        padding = np.zeros((self.target_motion_frames - motion_segment.shape[0], motion_segment.shape[1]))
                        motion_segment = np.concatenate([motion_segment, padding], axis=0)
                    
                    motion_condition = motion_segment[:self.condition_motion_frames]
                    motion_target = motion_segment[self.condition_motion_frames:]
                    
                    # BEAT_v2 has no text description, so CLIP feature is zero-padded
                    clip_feature = np.zeros(self.clip_dim, dtype=np.float32)
                    
                    name = f"beat_v2_{npz_stem}_sample{sample_idx}"
                    self.data_dict[name] = {
                        'motion': motion_segment.astype(np.float32),
                        'motion_condition': motion_condition.astype(np.float32),
                        'motion_target': motion_target.astype(np.float32),
                        'whisper': audio_segment.astype(np.float32),
                        'clip_feature': clip_feature,
                        'length': self.target_motion_frames,
                        'dataset_type': 'beat_v2'
                    }
                    self.name_list.append(name)
                    self.length_list.append(self.target_motion_frames)
                    self.dataset_type_list.append('beat_v2')
                
            except Exception as e:
                skipped_error += 1
                continue
        
        print(f"BEAT_v2: Loaded {sum(1 for dt in self.dataset_type_list if dt == 'beat_v2')} samples")
        print(f"BEAT_v2 Skipped: {skipped_no_whisper} (no whisper), {skipped_length} (length), {skipped_error} (error)")
        
        # Load semi-synthetic data
        semi_synthetic_npz_files = []
        if os.path.exists(semi_synthetic_root):
            for root, dirs, files in os.walk(semi_synthetic_root):
                for file in files:
                    if file.endswith('_motion.npz'):
                        semi_synthetic_npz_files.append(os.path.join(root, file))
        
        semi_synthetic_npz_files.sort()
        random.seed(42)
        random.shuffle(semi_synthetic_npz_files)
        
        split_idx = int(len(semi_synthetic_npz_files) * train_ratio)
        if split == 'train':
            semi_synthetic_npz_files = semi_synthetic_npz_files[:split_idx]
        else:
            semi_synthetic_npz_files = semi_synthetic_npz_files[split_idx:]
        
        print(f"Loading semi-synthetic {split} data from {len(semi_synthetic_npz_files)} files...")
        
        skipped_semi = 0
        
        for npz_path in tqdm(semi_synthetic_npz_files, desc="Loading semi-synthetic"):
            try:
                # Load motion
                motion_data = np.load(npz_path)
                if 'qpos' in motion_data:
                    motion = motion_data['qpos']
                else:
                    keys = list(motion_data.keys())
                    if len(keys) > 0:
                        motion = motion_data[keys[0]]
                    else:
                        skipped_semi += 1
                        continue
                
                if len(motion.shape) == 1:
                    motion = motion.reshape(-1, 1)
                
                motion_len = motion.shape[0]
                if motion_len < self.target_motion_frames:
                    skipped_semi += 1
                    continue
                
                # Get base name for finding related files
                npz_dir = os.path.dirname(npz_path)
                npz_basename = os.path.basename(npz_path)
                segment_name = npz_basename.replace('_motion.npz', '')
                
                # Load audio features
                audio_path = os.path.join(npz_dir, f"{segment_name}_audio.npy")
                if not os.path.exists(audio_path):
                    skipped_semi += 1
                    continue
                
                try:
                    audio_features = np.load(audio_path)
                    if len(audio_features.shape) != 2:
                        skipped_semi += 1
                        continue
                except Exception as e:
                    skipped_semi += 1
                    continue
                
                # Load CLIP feature (prefer description, fallback to name or semantic)
                clip_feature = None
                clip_paths = [
                    os.path.join(npz_dir, f"{segment_name}_clip_description.npy"),
                    os.path.join(npz_dir, f"{segment_name}_clip_name.npy"),
                    os.path.join(npz_dir, f"{segment_name}_clip_semantic.npy")
                ]
                
                for clip_path in clip_paths:
                    if os.path.exists(clip_path):
                        try:
                            clip_feature = np.load(clip_path)
                            if len(clip_feature.shape) == 1 and clip_feature.shape[0] == self.clip_dim:
                                break
                        except Exception as e:
                            continue
                
                if clip_feature is None:
                    # If no CLIP feature found, use zero padding
                    clip_feature = np.zeros(self.clip_dim, dtype=np.float32)
                
                # Extract fixed-length segments
                if motion_len >= self.target_motion_frames:
                    motion_segment = motion[:self.target_motion_frames]
                else:
                    padding = np.zeros((self.target_motion_frames - motion_len, motion.shape[1]))
                    motion_segment = np.concatenate([motion, padding], axis=0)
                
                # Normalize motion
                motion_segment = motion_segment[:, :self.mean.shape[0]]
                motion_segment = (motion_segment - self.mean) / self.std
                
                motion_condition = motion_segment[:self.condition_motion_frames]
                motion_target = motion_segment[self.condition_motion_frames:]
                
                # Process audio features (should be 250 frames)
                if audio_features.shape[0] >= self.target_audio_frames:
                    audio_segment = audio_features[:self.target_audio_frames]
                else:
                    padding = np.zeros((self.target_audio_frames - audio_features.shape[0], audio_features.shape[1]))
                    audio_segment = np.concatenate([audio_features, padding], axis=0)
                
                name = f"semi_synthetic_{segment_name}"
                self.data_dict[name] = {
                    'motion': motion_segment.astype(np.float32),
                    'motion_condition': motion_condition.astype(np.float32),
                    'motion_target': motion_target.astype(np.float32),
                    'whisper': audio_segment.astype(np.float32),
                    'clip_feature': clip_feature.astype(np.float32),
                    'length': self.target_motion_frames,
                    'dataset_type': 'semi_synthetic'
                }
                self.name_list.append(name)
                self.length_list.append(self.target_motion_frames)
                self.dataset_type_list.append('semi_synthetic')
                
            except Exception as e:
                skipped_semi += 1
                continue
        
        self.length_arr = np.array(self.length_list)
        print(f"Semi-synthetic: Loaded {sum(1 for dt in self.dataset_type_list if dt == 'semi_synthetic')} samples")
        print(f"Semi-synthetic Skipped: {skipped_semi} (error/missing)")
        print(f"Total samples: {len(self.data_dict)} (BEAT_v2: {sum(1 for dt in self.dataset_type_list if dt == 'beat_v2')}, Semi-synthetic: {sum(1 for dt in self.dataset_type_list if dt == 'semi_synthetic')})")
    
    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, item):
        """
        Returns:
            whisper_features: [target_audio_frames, feature_dim] - audio features (250 frames)
            clip_feature: [clip_dim] - CLIP text feature (512 dim)
            motion_condition: [condition_motion_frames, motion_dim] - first 60 frames as condition
            motion_target: [target_motion_frames_generate, motion_dim] - last 240 frames as target
            m_length: int - total motion length (300)
        """
        name = self.name_list[item]
        data = self.data_dict[name]
        motion_condition = data['motion_condition'].copy()
        motion_target = data['motion_target'].copy()
        whisper_features = data['whisper'].copy()
        clip_feature = data['clip_feature'].copy()
        m_length = data['length']
        
        return whisper_features, clip_feature, motion_condition, motion_target, m_length


class SemiSyntheticAudioTextDataset(data.Dataset):
    """
    Dataset for semi_synthetic_v1_segments only (text-to-motion with audio)
    Uses text CLIP features and audio features
    """
    def __init__(self, mean, std, semi_synthetic_root, unit_length, max_motion_length, split='train', train_ratio=0.9):
        """
        Semi-Synthetic Dataset for Audio+Text-to-Motion training (MARDM)
        Args:
            mean: mean for normalization
            std: std for normalization
            semi_synthetic_root: root directory of semi_synthetic_v1_segments data
            unit_length: unit length for motion (default 4)
            max_motion_length: maximum motion length (should be 300)
            split: 'train' or 'val'
            train_ratio: ratio of training data
        """
        self.mean = mean
        self.std = std
        self.unit_length = unit_length
        self.max_motion_length = max_motion_length  # Should be 300
        self.clip_dim = 512  # CLIP feature dimension
        
        # Frame alignment: same as BEAT_v2Audio2MotionDataset
        self.target_motion_frames = max_motion_length  # 300 frames total
        self.condition_motion_frames = 60  # First 60 frames as condition
        self.target_motion_frames_generate = self.target_motion_frames - self.condition_motion_frames  # 240 frames to generate
        self.target_audio_frames = int(self.target_motion_frames * 5 / 6)  # 250 for 300 motion frames
        self.audio_to_motion_ratio = 5.0 / 6.0  # 50/60
        
        min_motion_len = self.target_motion_frames
        max_motion_len = 50000
        
        # Collect data from semi-synthetic dataset
        self.data_dict = {}
        self.name_list = []
        self.length_list = []
        
        skipped_semi = 0
        
        # Load semi-synthetic data
        semi_synthetic_npz_files = []
        if os.path.exists(semi_synthetic_root):
            for root, dirs, files in os.walk(semi_synthetic_root):
                for file in files:
                    if file.endswith('_motion.npz'):
                        semi_synthetic_npz_files.append(os.path.join(root, file))
        
        semi_synthetic_npz_files.sort()
        random.seed(42)
        random.shuffle(semi_synthetic_npz_files)
        
        split_idx = int(len(semi_synthetic_npz_files) * train_ratio)
        if split == 'train':
            semi_synthetic_npz_files = semi_synthetic_npz_files[:split_idx]
        else:
            semi_synthetic_npz_files = semi_synthetic_npz_files[split_idx:]
        
        print(f"Loading semi-synthetic {split} data from {len(semi_synthetic_npz_files)} files...")
        
        for npz_path in tqdm(semi_synthetic_npz_files, desc="Loading semi-synthetic"):
            try:
                # Load motion
                motion_data = np.load(npz_path)
                if 'qpos' in motion_data:
                    motion = motion_data['qpos']
                else:
                    keys = list(motion_data.keys())
                    if len(keys) > 0:
                        motion = motion_data[keys[0]]
                    else:
                        skipped_semi += 1
                        continue
                
                if len(motion.shape) == 1:
                    motion = motion.reshape(-1, 1)
                
                motion_len = motion.shape[0]
                if motion_len < self.target_motion_frames:
                    skipped_semi += 1
                    continue
                
                # Get base name for finding related files
                npz_dir = os.path.dirname(npz_path)
                npz_basename = os.path.basename(npz_path)
                segment_name = npz_basename.replace('_motion.npz', '')
                
                # Load audio features
                audio_path = os.path.join(npz_dir, f"{segment_name}_audio.npy")
                if not os.path.exists(audio_path):
                    skipped_semi += 1
                    continue
                
                try:
                    audio_features = np.load(audio_path)
                    if len(audio_features.shape) != 2:
                        skipped_semi += 1
                        continue
                except Exception as e:
                    skipped_semi += 1
                    continue
                
                # Load CLIP feature (prefer description, fallback to name or semantic)
                clip_feature = None
                clip_paths = [
                    os.path.join(npz_dir, f"{segment_name}_clip_description.npy"),
                    os.path.join(npz_dir, f"{segment_name}_clip_name.npy"),
                    os.path.join(npz_dir, f"{segment_name}_clip_semantic.npy")
                ]
                
                for clip_path in clip_paths:
                    if os.path.exists(clip_path):
                        try:
                            clip_feature = np.load(clip_path)
                            if len(clip_feature.shape) == 1 and clip_feature.shape[0] == self.clip_dim:
                                break
                        except Exception as e:
                            continue
                
                if clip_feature is None:
                    # If no CLIP feature found, use zero padding
                    clip_feature = np.zeros(self.clip_dim, dtype=np.float32)
                
                # Extract fixed-length segments
                if motion_len >= self.target_motion_frames:
                    motion_segment = motion[:self.target_motion_frames]
                else:
                    padding = np.zeros((self.target_motion_frames - motion_len, motion.shape[1]))
                    motion_segment = np.concatenate([motion, padding], axis=0)
                
                # Normalize motion
                motion_segment = motion_segment[:, :self.mean.shape[0]]
                motion_segment = (motion_segment - self.mean) / self.std
                
                motion_condition = motion_segment[:self.condition_motion_frames]
                motion_target = motion_segment[self.condition_motion_frames:]
                
                # Process audio features (should be 250 frames)
                if audio_features.shape[0] >= self.target_audio_frames:
                    audio_segment = audio_features[:self.target_audio_frames]
                else:
                    padding = np.zeros((self.target_audio_frames - audio_features.shape[0], audio_features.shape[1]))
                    audio_segment = np.concatenate([audio_features, padding], axis=0)
                
                name = f"semi_synthetic_{segment_name}"
                self.data_dict[name] = {
                    'motion': motion_segment.astype(np.float32),
                    'motion_condition': motion_condition.astype(np.float32),
                    'motion_target': motion_target.astype(np.float32),
                    'whisper': audio_segment.astype(np.float32),
                    'clip_feature': clip_feature.astype(np.float32),
                    'length': self.target_motion_frames,
                    'dataset_type': 'semi_synthetic'
                }
                self.name_list.append(name)
                self.length_list.append(self.target_motion_frames)
                
            except Exception as e:
                skipped_semi += 1
                continue
        
        self.length_arr = np.array(self.length_list)
        print(f"Semi-synthetic: Loaded {len(self.data_dict)} samples")
        print(f"Semi-synthetic Skipped: {skipped_semi} (error/missing)")
    
    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, item):
        """
        Returns:
            whisper_features: [target_audio_frames, feature_dim] - audio features (250 frames)
            clip_feature: [clip_dim] - CLIP text feature (512 dim)
            motion_condition: [condition_motion_frames, motion_dim] - first 60 frames as condition
            motion_target: [target_motion_frames_generate, motion_dim] - last 240 frames as target
            m_length: int - total motion length (300)
        """
        name = self.name_list[item]
        data = self.data_dict[name]
        motion_condition = data['motion_condition'].copy()
        motion_target = data['motion_target'].copy()
        whisper_features = data['whisper'].copy()
        clip_feature = data['clip_feature'].copy()
        m_length = data['length']
        
        return whisper_features, clip_feature, motion_condition, motion_target, m_length


class Text2MotionDataset(data.Dataset):
    def __init__(self, mean, std, split_file, dataset_name, motion_dir, text_dir, unit_length, max_motion_length,
                 max_text_length, evaluation=False):
        self.evaluation = evaluation
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.max_text_len = max_text_length
        self.unit_length = unit_length
        min_motion_len = 40 if dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass
        if self.evaluation:
            self.w_vectorizer = GloVe('./glove', 'our_vab')
            name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        else:
            name_list, length_list = new_name_list, length_list
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        if self.evaluation:
            self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return (data - self.mean) / self.std
        else:
            return (data - mean) / std

    def inv_transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return data * self.std + self.mean
        else:
            return data * std + mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if self.evaluation:
            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = motion[:, :self.mean.shape[0]]
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        elif m_length > self.max_motion_length:
            if not self.evaluation:
                idx = random.randint(0, self.max_motion_length - m_length)
                motion = motion[idx:idx + self.max_motion_length]

        if self.evaluation:
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)
        else:
            return caption, motion, m_length