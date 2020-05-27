import os
import json
import argparse
import numpy as np
from nltk.tokenize import word_tokenize


def stat_data_info(data):
    num_videos, num_anns, video_lengths, query_lengths, moment_lengths, num_words = 0, 0, list(), list(), list(), list()
    for key, value in data.items():
        num_videos += 1
        num_anns += len(value["timestamps"])
        video_lengths.append(float(value["duration"]))

        for val in value["timestamps"]:
            moment_lengths.append(val[1] - val[0])

        for sentence in value["sentences"]:
            words = word_tokenize(sentence.strip().lower())
            num_words.extend(words)
            query_lengths.append(len(words))

    return num_videos, num_anns, video_lengths, query_lengths, moment_lengths, num_words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="ActivityNet Caption dataset directory")
    args = parser.parse_args()

    with open(os.path.join(args.dataset_dir, "train.json"), mode="r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open(os.path.join(args.dataset_dir, "val_1.json"), mode="r", encoding="utf-8") as f:
        test_data = json.load(f)

    with open(os.path.join(args.dataset_dir, "val_2.json"), mode="r", encoding="utf-8") as f:
        test2_data = json.load(f)

    (train_num_videos, train_num_anns, train_video_lengths, train_query_lengths, train_moment_lengths,
     train_num_words) = stat_data_info(train_data)

    (test_num_videos, test_num_anns, test_video_lengths, test_query_lengths, test_moment_lengths,
     test_num_words) = stat_data_info(test_data)

    (test2_num_videos, test2_num_anns, test2_video_lengths, test2_query_lengths, test2_moment_lengths,
     test2_num_words) = stat_data_info(test2_data)

    video_lengths = train_video_lengths + test_video_lengths + test2_video_lengths
    query_lengths = train_query_lengths + test_query_lengths + test2_query_lengths
    moment_lengths = train_moment_lengths + test_moment_lengths + test2_moment_lengths
    num_words = train_num_words + test_num_words + test2_num_words

    print("Training videos:", train_num_videos)
    print("Test videos:", test_num_videos)
    print("Training samples:", train_num_anns)
    print("Test samples:", test_num_anns)
    print("Vocabulary size:", len(set(num_words)))
    print("Average video length:", np.mean(video_lengths))
    print("Average query length:", np.mean(query_lengths))
    print("Average moment length:", np.mean(moment_lengths))
    print("Std. of moment length:", np.std(moment_lengths))


if __name__ == "__main__":
    main()
