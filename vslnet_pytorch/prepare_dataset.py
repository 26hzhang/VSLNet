import os
import codecs
import json
import pickle
import numpy as np
from tqdm import tqdm
from collections import Counter
from nltk.tokenize import word_tokenize

PAD, UNK = "<pad>", "<unk>"
glove_path = os.path.join('data', 'features', 'glove.840B.300d.txt')


def compute_overlap(pred, gt):
    # check format
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    pred = pred if pred_is_list else [pred]
    gt = gt if gt_is_list else [gt]
    # compute overlap
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(1e-12, union_right - union_left)
    overlap = 1.0 * inter / union
    # reformat output
    overlap = overlap if gt_is_list else overlap[:, 0]
    overlap = overlap if pred_is_list else overlap[0]
    return overlap


def time_to_index(start_time, end_time, num_units, duration):
    s_times = np.arange(0, num_units).astype(np.float32) * duration / float(num_units)
    e_times = np.arange(1, num_units + 1).astype(np.float32) * duration / float(num_units)
    candidates = np.stack([np.repeat(s_times[:, None], repeats=num_units, axis=1),
                           np.repeat(e_times[None, :], repeats=num_units, axis=0)], axis=2).reshape((-1, 2))
    overlaps = compute_overlap(candidates.tolist(), [start_time, end_time]).reshape(num_units, num_units)
    start_index = np.argmax(overlaps) // num_units  # row index
    end_index = np.argmax(overlaps) % num_units  # column index
    return start_index, end_index


def index_to_time(start_index, end_index, num_units, duration):
    s_times = np.arange(0, num_units).astype(np.float32) * duration / float(num_units)
    e_times = np.arange(1, num_units + 1).astype(np.float32) * duration / float(num_units)
    start_time = s_times[start_index]
    end_time = e_times[end_index]
    return start_time, end_time


def load_glove():
    vocab = list()
    with codecs.open(glove_path, mode="r", encoding="utf-8") as f:
        for line in tqdm(f, total=2196018, desc="load vocabulary from glove.840B.300d.txt"):
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2 or len(line) != 301:
                continue
            word = line[0]
            vocab.append(word)
    return set(vocab)


def filter_glove_embedding(word_dict):
    vectors = np.zeros(shape=[len(word_dict), 300], dtype=np.float32)  # 0: <pad>, 1: <unk>
    vectors[1] = np.random.uniform(-0.01, 0.01, size=(300, ))  # initialize the <unk> token
    with codecs.open(glove_path, mode="r", encoding="utf-8") as f:
        for line in tqdm(f, total=2196018, desc="load embeddings from glove.840B.300d.txt"):
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2 or len(line) != 301:
                continue
            word = line[0]
            if word in word_dict:
                vector = [float(x) for x in line[1:]]
                word_index = word_dict[word]
                vectors[word_index] = np.asarray(vector, dtype=np.float32)
    return np.asarray(vectors)


def create_vocabularies(datasets):
    # word and char counter
    word_counter, char_counter = Counter(), Counter()
    for data in datasets:
        for record in data:
            words = record[-1]
            for word in words:
                word_counter[word] += 1
                for char in list(word):
                    char_counter[char] += 1
    # generate word dict and vectors
    emb_vocab = load_glove()
    word_vocab = list()
    for word, _ in word_counter.most_common():
        if word in emb_vocab:
            word_vocab.append(word)
    tmp_word_dict = dict([(word, index) for index, word in enumerate(word_vocab)])
    vectors = filter_glove_embedding(tmp_word_dict)
    word_vocab = [PAD, UNK] + word_vocab
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    # generate character dict
    char_vocab = [PAD, UNK] + [char for char, count in char_counter.most_common() if count >= 5]
    char_dict = dict([(char, idx) for idx, char in enumerate(char_vocab)])
    return word_dict, char_dict, vectors


def load_data(filename, domain):
    with codecs.open(filename=filename, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    results = []
    for vid, records in data.items():
        if domain == 'tacos':
            duration = float(records["num_frames"]) / float(records["fps"])
        else:
            duration = float(records['duration'])
        for timestamp, sentence in zip(records["timestamps"], records["sentences"]):
            if domain == 'tacos':
                start_time = max(0.0, float(timestamp[0]) / float(records["fps"]))
                end_time = min(float(timestamp[1]) / float(records["fps"]), duration)
            else:
                start_time = max(0.0, float(timestamp[0]))
                end_time = min(float(timestamp[1]), duration)
            words = word_tokenize(sentence.strip().lower(), language="english")
            results.append((vid, start_time, end_time, duration, words))
    return results


def generate_dataset(data, feature_length, word_dict, char_dict, max_pos_len, scope):
    dataset = list()
    for record in tqdm(data, total=len(data), desc="process {} data".format(scope)):
        video_id, start_time, end_time, duration, words = record
        num_units = feature_length[video_id]
        # compute best start and end indices
        start_index, end_index = time_to_index(start_time, end_time, num_units, duration)
        # convert words and characters
        word_ids, char_ids = list(), list()
        words = words[0:max_pos_len]  # truncate words
        for word in words:
            word_id = word_dict[word] if word in word_dict else word_dict[UNK]
            char_id = [char_dict[char] if char in char_dict else char_dict[UNK] for char in word]
            word_ids.append(word_id)
            char_ids.append(char_id)
        example = {"vid": str(video_id), "s_time": float(start_time), "e_time": float(end_time),
                   "s_ind": int(start_index), "e_ind": int(end_index), "duration": float(duration),
                   "num_units": int(num_units), "word_ids": word_ids, "char_ids": char_ids}
        dataset.append(example)
    return dataset


def prepare_datasets(configs):
    if not os.path.exists(configs.save_dir):
        os.makedirs(configs.save_dir)
    # directory path
    data_dir = os.path.join('data', 'dataset', configs.task)
    feature_dir = os.path.join('data', 'features', configs.task)
    # load datasets
    train_data = load_data(os.path.join(data_dir, 'train.json'), configs.task)
    val_data = None if configs.task == 'charades' else load_data(os.path.join(data_dir, 'val.json'), configs.task)
    test_data = load_data(os.path.join(data_dir, 'test.json'), configs.task)
    # load feature lengths and down-sample if possible
    feature_lengths = dict()
    with codecs.open(filename=os.path.join(feature_dir, 'feature_shapes.json'), mode="r", encoding="utf-8") as f:
        feature_shapes = json.load(f)
    for vid, length in feature_shapes.items():
        length = configs.max_pos_len if length > configs.max_pos_len else length
        feature_lengths[vid] = length
    # generate token dicts and load pre-trained vectors
    datasets = [train_data, test_data] if val_data is None else [train_data, val_data, test_data]
    word_dict, char_dict, vectors = create_vocabularies(datasets)
    # generate datasets
    train_set = generate_dataset(train_data, feature_lengths, word_dict, char_dict, configs.max_pos_len, "train")
    val_set = None if val_data is None else generate_dataset(val_data, feature_lengths, word_dict, char_dict,
                                                             configs.max_pos_len, "val")
    test_set = generate_dataset(test_data, feature_lengths, word_dict, char_dict, configs.max_pos_len, "test")
    # save datasets
    dataset = {'train_set': train_set, 'val_set': val_set, 'test_set': test_set, 'word_dict': word_dict,
               'char_dict': char_dict, 'word_vector': vectors}
    save_filename = os.path.join(configs.save_dir, '_'.join([configs.task, str(configs.max_pos_len)]) + '.pkl')
    with open(save_filename, 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
