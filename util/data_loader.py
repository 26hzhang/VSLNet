import math
import random
import numpy as np
from util.data_util import pad_seq, pad_char_seq, pad_video_seq


class TrainLoader:
    def __init__(self, dataset, visual_features, configs):
        super(TrainLoader, self).__init__()
        self.dataset = dataset
        self.visual_feats = visual_features
        self.extend = configs.extend
        self.batch_size = configs.batch_size

    def set_extend(self, extend):
        self.extend = extend

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def num_samples(self):
        return len(self.dataset)

    def num_batches(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def batch_iter(self):
        random.shuffle(self.dataset)  # shuffle the train set first
        for index in range(0, len(self.dataset), self.batch_size):
            batch_data = self.dataset[index:(index + self.batch_size)]
            vfeats, vfeat_lens, word_ids, char_ids, s_labels, e_labels, h_labels = self.process_batch(batch_data)
            yield batch_data, vfeats, vfeat_lens, word_ids, char_ids, s_labels, e_labels, h_labels

    def process_batch(self, batch_data):
        vfeats, word_ids, char_ids, s_inds, e_inds = [], [], [], [], []
        for data in batch_data:
            vfeat = self.visual_feats[data['vid']]
            vfeats.append(vfeat)
            word_ids.append(data['w_ids'])
            char_ids.append(data['c_ids'])
            s_inds.append(data['s_ind'])
            e_inds.append(data['e_ind'])
        batch_size = len(batch_data)
        # process word ids
        word_ids, _ = pad_seq(word_ids)
        word_ids = np.asarray(word_ids, dtype=np.int32)  # (batch_size, w_seq_len)
        # process char ids
        char_ids, _ = pad_char_seq(char_ids)
        char_ids = np.asarray(char_ids, dtype=np.int32)  # (batch_size, w_seq_len, c_seq_len)
        # process video features
        vfeats, vfeat_lens = pad_video_seq(vfeats)
        vfeats = np.asarray(vfeats, dtype=np.float32)  # (batch_size, v_seq_len, v_dim)
        vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (batch_size, )
        # process labels
        max_len = np.max(vfeat_lens)
        s_labels = np.zeros(shape=[batch_size, max_len], dtype=np.int32)
        e_labels = np.zeros(shape=[batch_size, max_len], dtype=np.int32)
        h_labels = np.zeros(shape=[batch_size, max_len], dtype=np.int32)
        for idx in range(batch_size):
            st, et = s_inds[idx], e_inds[idx]
            s_labels[idx][st] = 1
            e_labels[idx][et] = 1
            cur_max_len = vfeat_lens[idx]
            extend_len = round(self.extend * float(et - st + 1))
            if extend_len > 0:
                st_ = max(0, st - extend_len)
                et_ = min(et + extend_len, cur_max_len - 1)
                h_labels[idx][st_:(et_ + 1)] = 1
            else:
                h_labels[idx][st:(et + 1)] = 1
        return vfeats, vfeat_lens, word_ids, char_ids, s_labels, e_labels, h_labels


class TestLoader:
    def __init__(self, datasets, visual_features, configs):
        self.visual_feats = visual_features
        self.val_set = None if datasets['val_set'] is None else datasets['val_set']
        self.test_set = datasets['test_set']
        self.batch_size = configs.batch_size

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def num_samples(self, mode='test'):
        if mode == 'val':
            if self.val_set is None:
                return 0
            return len(self.val_set)
        elif mode == 'test':
            return len(self.test_set)
        else:
            raise ValueError('Unknown mode!!! Only support [val | test | test_iid | test_ood].')

    def num_batches(self, mode='test'):
        if mode == 'val':
            if self.val_set is None:
                return 0
            return math.ceil(len(self.val_set) / self.batch_size)
        elif mode == 'test':
            return math.ceil(len(self.test_set) / self.batch_size)
        else:
            raise ValueError('Unknown mode!!! Only support [val | test].')

    def test_iter(self, mode='test'):
        if mode not in ['val', 'test']:
            raise ValueError('Unknown mode!!! Only support [val | test].')
        test_sets = {'val': self.val_set, 'test': self.test_set}
        dataset = test_sets[mode]
        if mode == 'val' and dataset is None:
            raise ValueError('val set is not available!!!')
        for index in range(0, len(dataset), self.batch_size):
            batch_data = dataset[index:(index + self.batch_size)]
            vfeats, vfeat_lens, word_ids, char_ids = self.process_batch(batch_data)
            yield batch_data, vfeats, vfeat_lens, word_ids, char_ids

    def process_batch(self, batch_data):
        vfeats, word_ids, char_ids, s_inds, e_inds = [], [], [], [], []
        for data in batch_data:
            vfeats.append(self.visual_feats[data['vid']])
            word_ids.append(data['w_ids'])
            char_ids.append(data['c_ids'])
            s_inds.append(data['s_ind'])
            e_inds.append(data['e_ind'])
        # process word ids
        word_ids, _ = pad_seq(word_ids)
        word_ids = np.asarray(word_ids, dtype=np.int32)  # (batch_size, w_seq_len)
        # process char ids
        char_ids, _ = pad_char_seq(char_ids)
        char_ids = np.asarray(char_ids, dtype=np.int32)  # (batch_size, w_seq_len, c_seq_len)
        # process video features
        vfeats, vfeat_lens = pad_video_seq(vfeats)
        vfeats = np.asarray(vfeats, dtype=np.float32)  # (batch_size, v_seq_len, v_dim)
        vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (batch_size, )
        return vfeats, vfeat_lens, word_ids, char_ids
