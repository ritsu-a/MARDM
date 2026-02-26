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
            if new_name_list:
                name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
            else:
                name_list, length_list = [], []
        else:
            name_list, length_list = new_name_list, length_list
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = list(name_list)
        if self.evaluation and len(self.name_list) > 0:
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
            if new_name_list:
                name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
            else:
                name_list, length_list = [], []
        else:
            name_list, length_list = new_name_list, length_list
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = list(name_list)
        if self.evaluation and len(self.name_list) > 0:
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


# BEAT: 50 帧 whisper 对应 60 帧 motion；每段固定 300 帧 motion、250 帧 whisper
WHISPER_PER_MOTION = 50
MOTION_PER_WHISPER = 60
BEAT_MOTION_SEGMENT_LEN = 300   # 每段 motion 300 帧
BEAT_WHISPER_SEGMENT_LEN = 250  # 每段 whisper 250 帧（50:60）


class BeatV2Text2MotionDataset(data.Dataset):
    """
    BEAT_v2 数据集：按子目录组织，加载 motion + whisper_features.npy + whisper_features.txt。
    对齐比例：50 帧 whisper 对应 60 帧 motion。每段固定 300 帧 motion、250 帧 whisper。
    """
    def __init__(self, mean, std, split_file, data_root, unit_length, max_motion_length,
                 max_text_length, evaluation=False, motion_key='qpos', min_motion_len=None):
        self.evaluation = evaluation
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.max_text_len = max_text_length
        self.unit_length = unit_length
        self.data_root = os.path.abspath(data_root)
        self.motion_key = motion_key
        self.min_motion_len = min_motion_len if min_motion_len is not None else BEAT_MOTION_SEGMENT_LEN
        self.max_whisper_len = BEAT_WHISPER_SEGMENT_LEN  # 固定 250

        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        self.data_dict = {}
        for name in tqdm(id_list):
            if not name:
                continue
            try:
                motion_path = pjoin(self.data_root, name + '.npz')
                if not os.path.exists(motion_path):
                    continue
                motion_data = np.load(motion_path)
                if self.motion_key and self.motion_key in motion_data:
                    motion = motion_data[self.motion_key]
                else:
                    keys = list(motion_data.keys())
                    if len(keys) == 0:
                        continue
                    motion = motion_data[keys[0]]
                if len(motion.shape) == 1:
                    motion = motion.reshape(-1, 1)
                motion_len = motion.shape[0]
                if motion_len < self.min_motion_len or motion_len >= 6000:  # 至少 300 帧才能切一段
                    continue
                text_path = pjoin(self.data_root, os.path.dirname(name), os.path.basename(name) + '_whisper_features.txt')
                if not os.path.exists(text_path):
                    continue
                whisper_path = pjoin(self.data_root, os.path.dirname(name), os.path.basename(name) + '_whisper_features.npy')
                if not os.path.exists(whisper_path):
                    continue
                whisper_feat = np.load(whisper_path).astype(np.float32)  # (T_whisper, 512)
                with cs.open(text_path, 'r') as f:
                    caption = f.read().strip().replace('\n', ' ')
                if not caption:
                    continue
                tokens = [w + '/OTHER' if '/' not in w else w for w in caption.split()]
                new_name_list.append(name)
                length_list.append(motion_len)
                self.data_dict[name] = {
                    'motion': motion.copy(), 'length': motion_len, 'text': [{'caption': caption, 'tokens': tokens}],
                    'whisper_feat': whisper_feat,
                }
            except Exception:
                pass

        if self.evaluation:
            self.w_vectorizer = GloVe('./glove', 'our_vab')
            if new_name_list:
                name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
            else:
                name_list, length_list = [], []
        else:
            name_list, length_list = new_name_list, length_list
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list) if length_list else np.array([])
        self.name_list = list(name_list) if name_list else []
        if self.evaluation and len(self.name_list) > 0:
            self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return (data - self.mean) / self.std
        return (data - mean) / std

    def inv_transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return data * self.std + self.mean
        return data * std + mean

    def __len__(self):
        return max(0, len(self.data_dict) - self.pointer)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        motion = motion.copy()
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if self.evaluation:
            if len(tokens) < self.max_text_len:
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
            else:
                tokens = tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :].copy())
                word_embeddings.append(word_emb[None, :].copy())
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)
            pos_one_hots = np.ascontiguousarray(pos_one_hots)
            word_embeddings = np.ascontiguousarray(word_embeddings)

        # 每段固定 300 帧 motion、250 帧 whisper（50:60）
        L = BEAT_MOTION_SEGMENT_LEN
        if len(motion) < L:
            L = len(motion)
        idx_start = random.randint(0, max(0, len(motion) - L))
        m_length = L
        motion = motion[idx_start:idx_start + m_length]

        motion = motion[:, :self.mean.shape[0]]
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion, np.zeros((self.max_motion_length - m_length, motion.shape[1]))], axis=0)
        elif m_length > self.max_motion_length:
            motion = motion[:self.max_motion_length]
            m_length = self.max_motion_length

        motion = np.ascontiguousarray(motion)
        # 50 帧 whisper 对应 60 帧 motion：motion [idx_start, idx_start+L] -> whisper [idx_start*50/60, (idx_start+L)*50/60]
        whisper_feat = data['whisper_feat']  # (T_w, 512)
        T_w, w_dim = whisper_feat.shape
        w_start = int(idx_start * WHISPER_PER_MOTION / MOTION_PER_WHISPER)
        w_end = min(int((idx_start + m_length) * WHISPER_PER_MOTION / MOTION_PER_WHISPER), T_w)
        w_slice = whisper_feat[w_start:w_end]  # (L_whisper, 512), L_whisper ≈ L*50/60
        if w_slice.shape[0] < self.max_whisper_len:
            whisper_out = np.concatenate([w_slice, np.zeros((self.max_whisper_len - w_slice.shape[0], w_dim), dtype=np.float32)], axis=0)
        else:
            whisper_out = w_slice[:self.max_whisper_len].copy()
        whisper_out = np.ascontiguousarray(whisper_out)
        if self.evaluation:
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), whisper_out
        return caption, motion, m_length


class BeatSegmentDataset(data.Dataset):
    """
    从 BEAT segment 目录加载：优先读合并 npz（如 segment_test.npz），否则按 id 列表读单段 npz。
    返回格式与 BeatV2Text2MotionDataset eval 一致（含 whisper），便于 VAE 测试。
    """
    def __init__(self, segment_dir, split_file, mean, std, max_motion_length, max_text_length, evaluation=True):
        self.segment_dir = os.path.abspath(segment_dir)
        self.mean = mean
        self.std = std
        self.max_motion_length = max_motion_length
        self.max_text_len = max_text_length
        self.evaluation = evaluation
        split_base = os.path.splitext(os.path.basename(split_file))[0]
        merged_npz = pjoin(self.segment_dir, split_base + ".npz")
        if os.path.exists(merged_npz):
            data = np.load(merged_npz, allow_pickle=True)
            self.motion = data["motion"]
            self.whisper = data["whisper"]
            self.caption = data["caption"]
            self.original_id = data["original_id"] if "original_id" in data else np.array([""] * len(self.motion), dtype=object)
            self.seg_idx = data["seg_idx"] if "seg_idx" in data else np.arange(len(self.motion), dtype=np.int32)
            self.segment_id = data["segment_id"] if "segment_id" in data else np.array([f"seg_{i}" for i in range(len(self.motion))], dtype=object)
            self.use_merged = True
            self.name_list = None
        else:
            self.use_merged = False
            split_path = split_file if os.path.isabs(split_file) else pjoin(self.segment_dir, os.path.basename(split_file))
            if not os.path.exists(split_path):
                split_path = split_file
            with open(split_path, "r") as f:
                self.name_list = [line.strip() for line in f if line.strip()]
            self.motion = self.whisper = self.caption = self.original_id = self.seg_idx = self.segment_id = None
        if self.evaluation:
            self.w_vectorizer = GloVe('./glove', 'our_vab')

    def __len__(self):
        if self.use_merged:
            return len(self.motion)
        return len(self.name_list)

    def __getitem__(self, item):
        if self.use_merged:
            motion = self.motion[item].copy()
            whisper_out = self.whisper[item].copy()
            caption = str(self.caption[item]) if self.caption[item] is not None else ""
        else:
            npz_path = pjoin(self.segment_dir, self.name_list[item] + ".npz")
            data = np.load(npz_path, allow_pickle=True)
            motion = data["motion"].copy()
            whisper_out = data["whisper"].copy()
            caption = str(data["caption"].item()) if "caption" in data and data["caption"].item() else ""
        dim_m = motion.shape[1]
        m_length = motion.shape[0]
        if m_length > self.max_motion_length:
            motion = motion[:self.max_motion_length]
            m_length = self.max_motion_length
        elif m_length < self.max_motion_length:
            motion = np.concatenate([motion, np.zeros((self.max_motion_length - m_length, dim_m), dtype=np.float32)], axis=0)
        motion = motion[:, :self.mean.shape[0]]
        motion = (motion - self.mean) / self.std
        motion = np.ascontiguousarray(motion)
        tokens = [w + '/OTHER' if '/' not in w else w for w in caption.split()]
        if self.evaluation:
            if len(tokens) < self.max_text_len:
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
            else:
                tokens = tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :].copy())
                word_embeddings.append(word_emb[None, :].copy())
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)
            pos_one_hots = np.ascontiguousarray(pos_one_hots)
            word_embeddings = np.ascontiguousarray(word_embeddings)
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), whisper_out
        # 训练时返回 whisper 作为条件（与 caption 同位置），便于 MARDM cond_mode='whisper'
        return whisper_out, motion, m_length

    def transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return (data - self.mean) / self.std
        return (data - mean) / std

    def inv_transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return data * self.std + self.mean
        return data * std + mean
