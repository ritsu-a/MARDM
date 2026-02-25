from os.path import join as pjoin
from torch.utils import data
import numpy as np
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import random
import codecs as cs
import os
import torch
import clip
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


class G1ML3DAEDataset(data.Dataset):
    """
    AEDataset for G1ML3D_v1 dataset
    Uses .npz format instead of .npy
    """
    def __init__(self, mean, std, motion_dir, window_size, split_file):
        self.data = []
        self.lengths = []
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                # Load motion data (.npz format for G1ML3D)
                motion_path = pjoin(motion_dir, name + '.npz')
                if not os.path.exists(motion_path):
                    continue
                
                motion_data = np.load(motion_path)
                if 'qpos' in motion_data:
                    motion = motion_data['qpos']
                else:
                    keys = list(motion_data.keys())
                    if len(keys) > 0:
                        motion = motion_data[keys[0]]
                    else:
                        continue
                
                if len(motion.shape) == 1:
                    motion = motion.reshape(-1, 1)
                
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
        # Copy motion to ensure writable array (fixes "Trying to resize storage that is not resizable" error)
        motion = motion.copy()
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
                # Copy to ensure writable arrays (fixes "Trying to resize storage that is not resizable" error)
                pos_one_hots.append(pos_oh[None, :].copy())
                word_embeddings.append(word_emb[None, :].copy())
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)
            # Ensure arrays are writable and contiguous
            pos_one_hots = np.ascontiguousarray(pos_one_hots)
            word_embeddings = np.ascontiguousarray(word_embeddings)

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
            # In evaluation mode, always crop to max_motion_length for consistent batch sizes
            # In training mode, randomly crop
            if self.evaluation:
                motion = motion[:self.max_motion_length]
                m_length = self.max_motion_length
            else:
                idx = random.randint(0, m_length - self.max_motion_length)
                motion = motion[idx:idx + self.max_motion_length]
                m_length = self.max_motion_length

        # Ensure motion array is writable and contiguous before returning
        motion = np.ascontiguousarray(motion)

        if self.evaluation:
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)
        else:
            return caption, motion, m_length


class G1ML3DText2MotionDataset(data.Dataset):
    """
    Dataset for G1ML3D_v1 text-to-motion training
    Compatible with Text2MotionDataset interface
    Note: G1ML3D uses 60 fps, while Text2MotionDataset uses 20 fps
    """
    def __init__(self, mean, std, split_file, dataset_name, motion_dir, text_dir, unit_length, max_motion_length,
                 max_text_length, evaluation=False):
        """
        G1ML3D Dataset for Text-to-Motion training (compatible with Text2MotionDataset)
        Args:
            mean: mean for normalization
            std: std for normalization
            split_file: path to split file (train.txt or val.txt)
            dataset_name: dataset name (for compatibility, not used for G1ML3D)
            motion_dir: root directory of G1ML3D_v1 motion data (e.g., '/root/workspace/MARDM/data/G1ML3D_v1/joints_npz')
            text_dir: root directory of text files (e.g., '/root/workspace/MARDM/data/G1ML3D_v1/texts')
            unit_length: unit length for motion (default 4)
            max_motion_length: maximum motion length
            max_text_length: maximum text length
            evaluation: whether in evaluation mode
        """
        self.evaluation = evaluation
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.max_text_len = max_text_length
        self.unit_length = unit_length
        # G1ML3D minimum motion length (equivalent to 40 frames at 20fps = 120 frames at 60fps)
        # Using 24 frames at 20fps as reference = 72 frames at 60fps, but we use a bit more conservative
        min_motion_len = 120  # Conservative minimum for 60fps data
        
        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
                

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                # Load motion data (.npz format for G1ML3D)
                motion_path = pjoin(motion_dir, name + '.npz')
                if not os.path.exists(motion_path):
                    continue
                
                motion_data = np.load(motion_path)
                if 'qpos' in motion_data:
                    motion = motion_data['qpos']
                else:
                    keys = list(motion_data.keys())
                    if len(keys) > 0:
                        motion = motion_data[keys[0]]
                    else:
                        continue
                
                if len(motion.shape) == 1:
                    motion = motion.reshape(-1, 1)
                
                motion_len = motion.shape[0]
                if (motion_len < min_motion_len) or (motion_len >= 600):
                    continue
                
                # Load text file
                text_path = pjoin(text_dir, name + '.txt')
                if not os.path.exists(text_path):
                    continue
                
                text_data = []
                flag = False
                with cs.open(text_path, 'r') as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        
                        # Parse tokens if available (format: "word1/OTHER word2/OTHER ...")
                        if len(line_split) >= 2 and line_split[1].strip():
                            tokens = line_split[1].split(' ')
                        else:
                            # If no tokens provided, split caption into words and add '/OTHER' tag
                            # to match Text2MotionDataset format
                            tokens = [word + '/OTHER' if '/' not in word else word for word in caption.split()]
                        
                        # Parse time tags if available (G1ML3D uses 60fps, not 20fps)
                        if len(line_split) >= 4:
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        else:
                            f_tag = 0.0
                            to_tag = 0.0

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                # G1ML3D uses 60fps, so multiply by 60 instead of 20
                                n_motion = motion[int(f_tag*60) : int(to_tag*60)]
                                if (len(n_motion) < min_motion_len) or (len(n_motion) >= 600):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                # Copy motion to ensure writable array
                                data_dict[new_name] = {'motion': n_motion.copy(),
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2] if len(line_split) > 2 else 'N/A', 
                                      line_split[3] if len(line_split) > 3 else 'N/A', 
                                      f_tag, to_tag, name)

                if flag:
                    # Copy motion to ensure writable array
                    data_dict[name] = {'motion': motion.copy(),
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
        # Copy motion to ensure writable array (fixes "Trying to resize storage that is not resizable" error)
        motion = motion.copy()
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
                # Copy to ensure writable arrays (fixes "Trying to resize storage that is not resizable" error)
                pos_one_hots.append(pos_oh[None, :].copy())
                word_embeddings.append(word_emb[None, :].copy())
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)
            # Ensure arrays are writable and contiguous
            pos_one_hots = np.ascontiguousarray(pos_one_hots)
            word_embeddings = np.ascontiguousarray(word_embeddings)

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
            # In evaluation mode, always crop to max_motion_length for consistent batch sizes
            # In training mode, randomly crop
            if self.evaluation:
                motion = motion[:self.max_motion_length]
                m_length = self.max_motion_length
            else:
                idx = random.randint(0, m_length - self.max_motion_length)
                motion = motion[idx:idx + self.max_motion_length]
                m_length = self.max_motion_length

        # Ensure motion array is writable and contiguous before returning
        motion = np.ascontiguousarray(motion)

        if self.evaluation:
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)
        else:
            return caption, motion, m_length

