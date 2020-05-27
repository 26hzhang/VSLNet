import os
import glob
import json
import random
import codecs
import numpy as np
from tqdm import tqdm

glove_sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
PAD, UNK = "<PAD>", "<UNK>"


def load_glove(glove_path, dim):
    vocab = list()
    with codecs.open(glove_path, mode="r", encoding="utf-8") as f:
        total = glove_sizes[glove_path.split(".")[-3]]

        for line in tqdm(f, total=total, desc="load glove vocabulary"):
            line = line.lstrip().rstrip().split(" ")

            if len(line) == 2 or len(line) != dim + 1:
                continue

            word = line[0]
            vocab.append(word)

    return set(vocab)


def filter_glove_embedding(word_dict, glove_path, dim):
    vectors = np.zeros(shape=[len(word_dict), dim], dtype=np.float32)

    with codecs.open(glove_path, mode="r", encoding="utf-8") as f:
        total = glove_sizes[glove_path.split(".")[-3]]

        for line in tqdm(f, total=total, desc="load glove embeddings"):
            line = line.lstrip().rstrip().split(" ")

            if len(line) == 2 or len(line) != dim + 1:
                continue

            word = line[0]

            if word in word_dict:
                vector = [float(x) for x in line[1:]]
                word_index = word_dict[word]
                vectors[word_index] = np.asarray(vector)

    return np.asarray(vectors)


def load_video_features(root, max_position_length):
    video_features = dict()
    filenames = glob.glob(os.path.join(root, "*.npy"))

    for filename in tqdm(filenames, total=len(filenames), desc="load video features"):
        video_id = filename.split("/")[-1].split(".")[0]
        feature = np.load(filename)

        if max_position_length is None:
            video_features[video_id] = feature

        else:
            new_feature = visual_feature_sampling(feature, max_num_clips=max_position_length)
            video_features[video_id] = new_feature

    return video_features


def visual_feature_sampling(visual_feature, max_num_clips):
    num_clips = visual_feature.shape[0]

    if num_clips <= max_num_clips:
        return visual_feature

    idxs = np.arange(0, max_num_clips + 1, 1.0) / max_num_clips * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1

    new_visual_feature = []
    for i in range(max_num_clips):
        s_idx, e_idx = idxs[i], idxs[i + 1]

        if s_idx < e_idx:
            new_visual_feature.append(np.mean(visual_feature[s_idx:e_idx], axis=0))

        else:
            new_visual_feature.append(visual_feature[s_idx])

    new_visual_feature = np.asarray(new_visual_feature)

    return new_visual_feature


def iou(pred, gt):  # require pred and gt is numpy
    assert isinstance(pred, list) and isinstance(gt, list)

    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)

    if not pred_is_list:
        pred = [pred]

    if not gt_is_list:
        gt = [gt]

    pred, gt = np.array(pred), np.array(gt)

    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)

    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(1e-12, union_right - union_left)

    overlap = 1.0 * inter / union

    if not gt_is_list:
        overlap = overlap[:, 0]

    if not pred_is_list:
        overlap = overlap[0]

    return overlap


def time_to_index(start_time, end_time, feature_shape, duration):
    s_times = np.arange(0, feature_shape).astype(np.float32) * duration / float(feature_shape)
    e_times = np.arange(1, feature_shape + 1).astype(np.float32) * duration / float(feature_shape)

    candidates = np.stack([np.repeat(s_times[:, None], repeats=feature_shape, axis=1),
                           np.repeat(e_times[None, :], repeats=feature_shape, axis=0)], axis=2).reshape((-1, 2))

    overlaps = iou(candidates.tolist(), [start_time, end_time]).reshape(feature_shape, feature_shape)
    start_index = np.argmax(overlaps) // feature_shape
    end_index = np.argmax(overlaps) % feature_shape

    return start_index, end_index


def load_video_ids(root):
    video_ids = []
    filenames = glob.glob(os.path.join(root, "*.npy"))

    for filename in filenames:
        basename = os.path.basename(filename)
        vid = basename[0:-4]
        video_ids.append(vid)

    return video_ids


def write_json(dataset, save_path, pretty=False):
    with codecs.open(filename=save_path, mode="w", encoding="utf-8") as f:
        if pretty:
            json.dump(dataset, f, indent=4, sort_keys=True)
        else:
            json.dump(dataset, f)


def load_json(filename):
    with codecs.open(filename=filename, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def word_convert(word, word_lower=True, char_lower=True):
    if char_lower:
        chars = [c for c in word.lower()]
    else:
        chars = [c for c in word]

    if word_lower:
        word = word.lower()

    return word, chars


def create_vocabularies(configs, word_counter, char_counter):
    # generate word dict and vectors
    emb_vocab = load_glove(configs.wordvec_path, configs.word_dim)

    word_vocab = list()
    for word, _ in word_counter.most_common():
        if word in emb_vocab:
            word_vocab.append(word)

    tmp_word_dict = dict([(word, index) for index, word in enumerate(word_vocab)])
    vectors = filter_glove_embedding(tmp_word_dict, configs.wordvec_path, configs.word_dim)

    word_vocab = [PAD, UNK] + word_vocab
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])

    # generate character dict
    char_vocab = [PAD, UNK] + [char for char, count in char_counter.most_common() if count >= 5]
    char_dict = dict([(char, idx) for idx, char in enumerate(char_vocab)])

    return word_dict, char_dict, vectors


def boolean_string(bool_str):
    bool_str = bool_str.lower()

    if bool_str not in {"false", "true"}:
        raise ValueError("Not a valid boolean string!!!")

    return bool_str == "true"


def pad_sequences(sequences, pad_tok=None, max_length=None):
    if pad_tok is None:
        pad_tok = 0  # 0: "PAD" for words and chars, "PAD" for tags

    if max_length is None:
        max_length = max([len(seq) for seq in sequences])

    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded.append(seq_)
        sequence_length.append(min(len(seq), max_length))

    return sequence_padded, sequence_length


def pad_char_sequences(sequences, max_length=None, max_length_2=None):
    sequence_padded, sequence_length = [], []

    if max_length is None:
        max_length = max(map(lambda x: len(x), sequences))

    if max_length_2 is None:
        max_length_2 = max([max(map(lambda x: len(x), seq)) for seq in sequences])

    for seq in sequences:
        sp, sl = pad_sequences(seq, max_length=max_length_2)
        sequence_padded.append(sp)
        sequence_length.append(sl)

    sequence_padded, _ = pad_sequences(sequence_padded, pad_tok=[0] * max_length_2, max_length=max_length)
    sequence_length, _ = pad_sequences(sequence_length, max_length=max_length)

    return sequence_padded, sequence_length


def pad_video_sequence(sequences, max_length=None):
    if max_length is None:
        max_length = max([vfeat.shape[0] for vfeat in sequences])

    feature_length = sequences[0].shape[1]
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        add_length = max_length - seq.shape[0]
        sequence_length.append(seq.shape[0])

        if add_length > 0:
            add_feature = np.zeros(shape=[add_length, feature_length], dtype=np.float32)
            seq_ = np.concatenate([seq, add_feature], axis=0)

        else:
            seq_ = seq

        sequence_padded.append(seq_)

    return sequence_padded, sequence_length


def batch_iter(dataset, all_video_features, batch_size, extend=0.2, shuffle=False):
    if shuffle:
        random.shuffle(dataset)

    for index in range(0, len(dataset), batch_size):
        batch_data = dataset[index:(index + batch_size)]
        video_ids, word_ids, char_ids, start_indexes, end_indexes = [], [], [], [], []

        for data in batch_data:
            video_ids.append(data["video_id"])
            word_ids.append(data["word_ids"])
            char_ids.append(data["char_ids"])
            start_indexes.append(data["start_index"])
            end_indexes.append(data["end_index"])

        true_batch_size = len(batch_data)

        # process word ids
        word_ids, _ = pad_sequences(word_ids)
        word_ids = np.asarray(word_ids, dtype=np.int32)

        # process char ids
        char_ids, _ = pad_char_sequences(char_ids)
        char_ids = np.asarray(char_ids, dtype=np.int32)

        # process video features
        video_features = [all_video_features[video_id] for video_id in video_ids]
        max_length = max([vfeat.shape[0] for vfeat in video_features])
        video_features, video_seq_length = pad_video_sequence(video_features, max_length=max_length)
        video_features = np.asarray(video_features, dtype=np.float32)
        video_seq_length = np.asarray(video_seq_length, dtype=np.int32)

        # generate labels
        start_label = np.zeros(shape=[true_batch_size, max_length], dtype=np.int32)
        end_label = np.zeros(shape=[true_batch_size, max_length], dtype=np.int32)
        highlight_labels = np.zeros(shape=[true_batch_size, max_length], dtype=np.int32)

        for idx in range(true_batch_size):
            st, et = start_indexes[idx], end_indexes[idx]
            start_label[idx][st] = 1
            end_label[idx][et] = 1

            extend_len = round(extend * float(et - st + 1))
            if extend_len > 0:
                st_ = max(0, st - extend_len)
                et_ = min(et + extend_len, max_length - 1)
                highlight_labels[idx][st_:(et_ + 1)] = 1
            else:
                highlight_labels[idx][st:(et + 1)] = 1

        yield (batch_data, video_features, word_ids, char_ids, video_seq_length, start_label, end_label,
               highlight_labels)
