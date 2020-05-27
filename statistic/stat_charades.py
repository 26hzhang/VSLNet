import os
import json
import argparse
import numpy as np
from nltk.tokenize import word_tokenize


def load_charades_sta_data(charades_sta_file, charades):
    with open(charades_sta_file, mode="r", encoding="utf-8") as f_sta:
        vids, data = [], []
        for line in f_sta:
            line = line.lstrip().rstrip()

            if len(line) == 0:
                continue

            video_info, sentence = line.split("##")
            vid, start_time, end_time = video_info.split(" ")
            words = word_tokenize(sentence.lower(), language="english")
            start_time, end_time = float(start_time), float(end_time)
            duration = float(charades[vid]["duration"])

            vids.append(vid)
            data.append((vid, start_time, end_time, duration, words))

        return vids, data


def stat_data_info(data):
    query_lengths, moment_lengths, num_words = [], [], []

    for record in data:
        moment_length = record[2] - record[1]
        moment_lengths.append(moment_length)
        query_lengths.append(len(record[-1]))
        num_words.extend(record[-1])

    return query_lengths, moment_lengths, num_words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Charades-STA dataset directory")
    args = parser.parse_args()

    with open(os.path.join(args.dataset_dir, "charades.json"), mode="r", encoding="utf-8") as f:
        charades = json.load(f)

    train_vids, train_data = load_charades_sta_data(os.path.join(args.dataset_dir, "charades_sta_train.txt"), charades)
    test_vids, test_data = load_charades_sta_data(os.path.join(args.dataset_dir, "charades_sta_test.txt"), charades)

    num_train_videos = len(set(train_vids))
    num_test_videos = len(set(test_vids))
    num_train_anns = len(train_data)
    num_test_anns = len(test_data)

    vids = list(set(train_vids + test_vids))
    video_lengths = []
    for vid in vids:
        duration = charades[vid]["duration"]
        video_lengths.append(float(duration))

    train_query_lengths, train_moment_lengths, train_num_words = stat_data_info(train_data)
    test_query_lengths, test_moment_lengths, test_num_words = stat_data_info(test_data)
    query_lengths = train_query_lengths + test_query_lengths
    moment_lengths = train_moment_lengths + test_moment_lengths
    num_words = train_num_words + test_num_words

    print("Training videos:", num_train_videos)
    print("Test videos:", num_test_videos)
    print("Training samples:", num_train_anns)
    print("Test samples:", num_test_anns)
    print("Vocabulary size:", len(set(num_words)))
    print("Average video length:", np.mean(video_lengths))
    print("Average query length:", np.mean(query_lengths))
    print("Average moment length:", np.mean(moment_lengths))
    print("Std. of moment length:", np.std(moment_lengths))


if __name__ == "__main__":
    main()
