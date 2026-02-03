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