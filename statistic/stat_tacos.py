import os
import json
import glob
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from moviepy.editor import VideoFileClip


def extract_video_to_images(video_dir, video_names, save_dir):
    if not os.path.exists(video_dir):
        raise ValueError("The video directory '{}' does not exist!!!".format(video_dir))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for video_name in tqdm(video_names, total=len(video_names), desc="extract video to images"):
        video_path = os.path.join(video_dir, video_name)
        video_id = video_name[0:-4]
        image_dir = os.path.join(save_dir, video_id)

        if os.path.exists(image_dir):
            continue
        else:
            os.makedirs(image_dir)

        subprocess.call("ffmpeg -hide_banner -loglevel panic -i {} -filter:v fps=fps=29.4 {}/{}-%6d.jpg".format(
            video_path, image_dir, video_id), shell=True)


def load_frames_and_times(image_dir, video_dir, video_names):
    dirs = glob.glob(os.path.join(image_dir, "*/"))
    video_frames = dict()

    for directory in dirs:
        vid = os.path.basename(directory[0:-1])
        num_frames = len(glob.glob(os.path.join(directory, "*.jpg")))
        video_frames[vid] = num_frames

    video_times = dict()
    fps = None

    for video_name in video_names:
        video_id = video_name[0:-4]
        clip = VideoFileClip(os.path.join(video_dir, video_name))
        fps = clip.fps  # all the videos with the same fps
        duration = clip.duration
        video_times[video_id] = duration

    return video_frames, video_times, fps


def load_video_names(dataset_dir):
    video_names = []
    video_files = ["TACoS_train_videos.txt", "TACoS_val_videos.txt", "TACoS_test_videos.txt"]

    for video_file in video_files:
        with open(os.path.join(dataset_dir, video_file), mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if len(line) == 0:
                    continue

                video_names.append(line)

    return video_names


def read_data(filename):
    results = []
    with open(filename, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if len(line) == 0:
                continue

            video, text = line.split(":")

            if text.endswith("#"):
                text = text[0:-1]

            sentences = [sentence.strip().lower() for sentence in text.split("#")]
            vid, start_frame, end_frame = video.split("_")
            vid = vid[0:-4]
            start_frame = int(start_frame)
            end_frame = int(end_frame)

            result = (vid, start_frame, end_frame, sentences)
            results.append(result)

    return results


def reconstruct_tacos_dataset(dataset, video_frames, fps):
    temp_data = dict()
    for data in dataset:
        vid, start_frame, end_frame, sentences = data
        temp_data[vid] = temp_data.get(vid, []) + [(start_frame, end_frame, sentences)]

    new_dataset = dict()
    for vid, records in temp_data.items():
        num_frames = video_frames[vid]
        timestamps, sentences = [], []

        for record in records:
            start_frame, end_frame, sents = record

            for sent in sents:
                timestamps.append([start_frame, end_frame])
                sentences.append(sent)

        new_dataset[vid] = {"timestamps": timestamps, "sentences": sentences, "fps": fps, "num_frames": num_frames}
    return new_dataset


def stat_data_info(data, fps):
    num_samples, query_lengths, num_words, moment_lengths = 0, [], [], []
    for record in data:
        moment_length = float(record[2] - record[1]) / fps
        num_samples += len(record[-1])

        for sentence in record[-1]:
            words = word_tokenize(sentence)
            query_lengths.append(len(words))
            num_words.extend(words)

        moment_lengths.append(moment_length)
    return num_samples, query_lengths, num_words, moment_lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True, help="TACoS video directory")
    parser.add_argument("--dataset_dir", type=str, required=True, help="TACoS dataset directory")
    parser.add_argument("--save_dir", type=str, required=True, help="directory to save extracted images")
    args = parser.parse_args()

    # load video ids
    video_names = load_video_names(args.dataset_dir)

    # extract video information
    extract_video_to_images(args.video_dir, video_names, args.save_dir)
    video_frames, video_times, fps = load_frames_and_times(args.save_dir, args.video_dir, video_names)

    # load TACoS datasets
    train_data = read_data(os.path.join(args.dataset_dir, "TACoS_train_samples.txt"))
    val_data = read_data(os.path.join(args.dataset_dir, "TACoS_val_samples.txt"))
    test_data = read_data(os.path.join(args.dataset_dir, "TACoS_test_samples.txt"))

    train_set = reconstruct_tacos_dataset(train_data, video_frames, fps)
    val_set = reconstruct_tacos_dataset(val_data, video_frames, fps)
    test_set = reconstruct_tacos_dataset(test_data, video_frames, fps)

    with open(os.path.join(args.dataset_dir, "train.json"), mode="w", encoding="utf-8") as f:
        json.dump(train_set, f)

    with open(os.path.join(args.dataset_dir, "val.json"), mode="w", encoding="utf-8") as f:
        json.dump(val_set, f)

    with open(os.path.join(args.dataset_dir, "test.json"), mode="w", encoding="utf-8") as f:
        json.dump(test_set, f)

    # statistics
    train_samples, train_query_lengths, train_num_words, train_moment_lengths = stat_data_info(train_data, fps)
    val_samples, val_query_lengths, val_num_words, val_moment_lengths = stat_data_info(val_data, fps)
    test_samples, test_query_lengths, test_num_words, test_moment_lengths = stat_data_info(test_data, fps)
    query_lengths = train_query_lengths + val_query_lengths + test_query_lengths
    num_words = train_num_words + val_num_words + test_num_words
    moment_lengths = train_moment_lengths + val_moment_lengths + test_moment_lengths
    durations = list(video_times.values())

    # print
    print("Training samples:", train_samples)
    print("Validation samples:", val_samples)
    print("Test samples:", test_samples)
    print("Vocabulary size:", len(set(num_words)))
    print("Average video length:", np.mean(durations))
    print("Average query length:", np.mean(query_lengths))
    print("Average moment length:", np.mean(moment_lengths))
    print("Std. of moment length:", np.std(moment_lengths))


if __name__ == "__main__":
    main()
