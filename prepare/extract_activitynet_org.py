import os
import h5py
import json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=True, help="dataset path")
parser.add_argument("--hdf5_file", type=str, required=True, help="downloaded activitynet features")
parser.add_argument("--save_dir", type=str, required=True, help="save dir")
args = parser.parse_args()

with open(os.path.join(args.dataset_dir, "train.json"), mode="r", encoding="utf-8") as f:
    train_data = json.load(f)
with open(os.path.join(args.dataset_dir, "val_1.json"), mode="r", encoding="utf-8") as f:
    val_data = json.load(f)
with open(os.path.join(args.dataset_dir, "val_2.json"), mode="r", encoding="utf-8") as f:
    test_data = json.load(f)

video_ids = list(set(list(train_data.keys()) + list(val_data.keys()) + list(test_data.keys())))
print(video_ids)
print(len(video_ids))

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

feature_shapes = dict()
with h5py.File(args.hdf5_file, mode="r") as f:
    group_key = list(f.keys())
    for key in tqdm(group_key, total=len(group_key), desc="extract features"):
        video_id = key
        if video_id not in video_ids:
            continue
        data = f[key]["c3d_features"][()]
        feature_shapes[video_id] = data.shape[0]
        np.save(os.path.join(args.save_dir, video_id), arr=data)

with open(os.path.join(args.save_dir, "feature_shapes.json"), mode="w", encoding="utf-8") as f:
    json.dump(feature_shapes, f)
