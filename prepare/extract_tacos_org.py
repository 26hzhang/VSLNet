import os
import json
import argparse
import numpy as np
from tqdm import tqdm

# 1. step download pre-trained C3D features from https://github.com/jiyanggao/TALL
# 2. convert the features

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, default="tacos dataset")
parser.add_argument("--feature_path", type=str, required=True, help="pre-trained C3D features")
parser.add_argument("--save_dir", type=str, required=True, help="extracted feature save path")
parser.add_argument("--sample_rate", type=int, default=64, help="sample rate [64 | 128 | 256 | 512]")
args = parser.parse_args()

stride = args.sample_rate // 5  # due to 0.8 overlap of the pre-trained C3D features

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

with open(os.path.join(args.data_path, "train.json"), mode="r", encoding="utf-8") as f:
    dataset = json.load(f)
with open(os.path.join(args.data_path, "val.json"), mode="r", encoding="utf-8") as f:
    dataset.update(json.load(f))
with open(os.path.join(args.data_path, "test.json"), mode="r", encoding="utf-8") as f:
    dataset.update(json.load(f))

feature_shapes = dict()
for video_id, annotations in tqdm(dataset.items(), total=len(dataset), desc=""):
    video_features = []
    num_frames = annotations["num_frames"] - 16  # trick from 2D-TAN
    for idx in range(0, (num_frames - args.sample_rate) // stride + 1):
        s_idx = idx * stride + 1
        e_idx = s_idx + args.sample_rate
        feature_path = os.path.join(args.feature_path, "{}.avi_{}_{}.npy".format(video_id, s_idx, e_idx))
        feature = np.load(feature_path)
        video_features.append(feature)
    video_features = np.stack(video_features, axis=0)
    np.save(os.path.join(args.save_dir, video_id), arr=video_features)
    feature_shapes[video_id] = video_features.shape[0]

with open(os.path.join(args.save_dir, "feature_shapes.json"), mode="w", encoding="utf-8") as f:
    json.dump(feature_shapes, f)
