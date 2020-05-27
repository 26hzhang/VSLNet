"""
Codes are modified from https://github.com/waybarrios/Anet_tools2.0
"""
import os
import glob
import json
from argparse import ArgumentParser


def crosscheck_videos(video_path, all_video_ids):
    # Get existing videos
    existing_videos = glob.glob("%s/*.mp4" % video_path)
    for idx, vid in enumerate(existing_videos):
        basename = os.path.basename(vid).split(".mp4")[0]
        if len(basename) == 13:
            existing_videos[idx] = basename[2:]
        elif len(basename) == 11:
            existing_videos[idx] = basename
        else:
            raise RuntimeError("Unknown filename format: %s", vid)

    non_existing_videos = []
    for vid in all_video_ids:
        if vid in existing_videos:
            continue
        else:
            non_existing_videos.append(vid)

    return non_existing_videos


def main(video_dir, dataset_dir, bash_file):
    with open(os.path.join(dataset_dir, "train.json"), mode="r", encoding="utf-8") as f:
        train_ids = list(json.load(f).keys())
        train_ids = [vid[2:] if len(vid) == 13 else vid for vid in train_ids]

    with open(os.path.join(dataset_dir, "val_1.json"), mode="r", encoding="utf-8") as f:
        val_ids = list(json.load(f).keys())
        val_ids = [vid[2:] if len(vid) == 13 else vid for vid in val_ids]

    with open(os.path.join(dataset_dir, "val_2.json"), mode="r", encoding="utf-8") as f:
        test_ids = list(json.load(f).keys())
        test_ids = [vid[2:] if len(vid) == 13 else vid for vid in test_ids]

    all_video_ids = list(set(train_ids + val_ids + test_ids))
    print("train_video_ids", len(train_ids))
    print("val_1_video_ids", len(val_ids))
    print("val_2_video_ids", len(test_ids))
    print("all_video_ids", len(all_video_ids))

    non_existing_videos = crosscheck_videos(video_dir, all_video_ids)

    # save command to bash file
    with open(bash_file + '.sh', mode="w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n\n")  # write bash file header
        filename = os.path.join(video_dir, "v_%s.mp4")
        cmd_base = "youtube-dl -f best -f mp4 "
        cmd_base += '"https://www.youtube.com/watch?v=%s" '
        cmd_base += '-o "%s"' % filename

        for vid in non_existing_videos:
            cmd = cmd_base % (vid, vid)
            f.write("%s\n" % cmd)


if __name__ == "__main__":
    parser = ArgumentParser(description="Script to double check video content.")
    parser.add_argument("--video_dir", type=str, required=True, help="where to save the downloaded videos")
    parser.add_argument("--dataset_dir", type=str, required=True, help="where are the annotation files")
    parser.add_argument("--bash_file", type=str, required=True, help="where to save command list script")

    args = vars(parser.parse_args())
    main(**args)
    """
    After running this python file, it will generate an script file. Using the terminal to run this script, it will 
    automatically download all the required videos from YouTube.
    """
