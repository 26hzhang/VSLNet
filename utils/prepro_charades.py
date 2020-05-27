import os
import json
import numpy as np
from tqdm import tqdm
from collections import Counter
from nltk.tokenize import word_tokenize
from utils.data_utils import create_vocabularies, load_json, write_json, UNK, time_to_index


def read_charades_data(charades_dir, max_sentence_length):
    # load charades json file
    with open(os.path.join(charades_dir, "charades.json"), mode="r", encoding="utf-8") as f:
        charades = json.load(f)

    def load_information(charades_sta_dir):
        with open(charades_sta_dir, mode="r", encoding="utf-8") as f_sta:
            vids, data = [], []

            for line in f_sta:
                line = line.lstrip().rstrip()

                if len(line) == 0:
                    continue

                video_info, sentence = line.split("##")
                vid, start_time, end_time = video_info.split(" ")
                words = word_tokenize(sentence.lower(), language="english")

                if max_sentence_length is not None:
                    words = words[0:max_sentence_length]

                duration = float(charades[vid]["duration"])
                start_time = max(0.0, float(start_time))
                end_time = min(float(end_time), duration)

                vids.append(vid)
                data.append((vid, start_time, end_time, duration, words))

            return vids, data

    # load train and test dataset
    train_vids, train_data = load_information(os.path.join(charades_dir, "charades_sta_train.txt"))
    test_vids, test_data = load_information(os.path.join(charades_dir, "charades_sta_test.txt"))

    video_ids = list(set(train_vids + test_vids))

    return train_data, test_data, video_ids


def generate_dataset(data, feature_shapes, word_dict, char_dict, scope):
    dataset = list()
    for record in tqdm(data, total=len(data), desc="process {} data".format(scope)):
        video_id, start_time, end_time, duration, words = record
        feature_shape = feature_shapes[video_id]

        # compute best start and end indices
        start_index, end_index = time_to_index(start_time, end_time, feature_shape, duration)

        # convert words and characters
        word_indices, char_indices = list(), list()
        for word in words:
            word_index = word_dict[word] if word in word_dict else word_dict[UNK]
            char_index = [char_dict[char] if char in char_dict else char_dict[UNK] for char in word]
            word_indices.append(word_index)
            char_indices.append(char_index)

        example = {"video_id": str(video_id), "start_time": float(start_time), "end_time": float(end_time),
                   "duration": float(duration), "start_index": int(start_index), "end_index": int(end_index),
                   "feature_shape": int(feature_shape), "word_ids": word_indices, "char_ids": char_indices}
        dataset.append(example)

    return dataset


def prepro_charades(configs):

    if not os.path.exists(configs.save_dir):
        os.makedirs(configs.save_dir)

    # train/test data format: (video_id, start_time, end_time, duration, words)
    train_data, test_data, _ = read_charades_data(configs.root, configs.max_position_length)

    # load features and sample feature shapes if possible
    features_path = os.path.join(configs.root, "charades_features_{}/feature_shapes.json".format(configs.feature))
    feature_shapes = dict()
    for vid, length in load_json(features_path).items():
        if configs.max_position_length is not None and length > configs.max_position_length:
            length = configs.max_position_length
        feature_shapes[vid] = length

    # generate token dicts and load pre-trained vectors
    word_counter, char_counter = Counter(), Counter()
    for data in [train_data, test_data]:
        for record in data:
            words = record[-1]
            for word in words:
                word_counter[word] += 1
                for char in list(word):
                    char_counter[char] += 1
    word_dict, char_dict, word_vectors = create_vocabularies(configs, word_counter, char_counter)

    # generate datasets
    train_set = generate_dataset(train_data, feature_shapes, word_dict, char_dict, "train")
    test_set = generate_dataset(test_data, feature_shapes, word_dict, char_dict, "test")

    # save to directory
    write_json(word_dict, save_path=os.path.join(configs.save_dir, "word_dict.json"))
    write_json(char_dict, save_path=os.path.join(configs.save_dir, "char_dict.json"))
    np.savez_compressed(os.path.join(configs.save_dir, "word_vectors.npz"), vectors=word_vectors)
    write_json(train_set, save_path=os.path.join(configs.save_dir, "train_set.json"))
    write_json(test_set, save_path=os.path.join(configs.save_dir, "test_set.json"))
