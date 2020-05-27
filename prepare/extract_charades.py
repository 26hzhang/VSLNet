import os
import cv2
import json
import torch
import argparse
import subprocess
import numpy as np
from . import videotransforms
from .feature_extractor import InceptionI3d
from torchvision import transforms
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_idx", type=str, default="0", help="gpu index")
parser.add_argument("--use_finetuned", action="store_true", help="whether use fine-tuned feature extractor")
parser.add_argument("--load_model", type=str, required=True, help="pre-trained model")
parser.add_argument("--video_dir", type=str, required=True, help="where are located the videos")
parser.add_argument("--dataset_dir", type=str, required=True, help="where are located the dataset files")
parser.add_argument("--images_dir", type=str, required=True, help="where to save extracted images")
parser.add_argument("--save_dir", type=str, required=True, help="where to save extracted features")
parser.add_argument("--fps", type=int, default=24, help="frames per second")
parser.add_argument("--video_format", type=str, default="mp4", help="video format")
parser.add_argument("--strides", type=int, default=24, help="window size")
parser.add_argument("--remove_images", action="store_true", help="whether remove extract images to release space")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx


if not os.path.exists(args.video_dir):
    raise ValueError("The video directory '{}' does not exist!!!".format(args.video_dir))

if not os.path.exists(args.images_dir):
    os.makedirs(args.images_dir)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# create I3D model and load pre-trained model
i3d_model = InceptionI3d(400, in_channels=3)
if args.use_fine_tuned:
    i3d_model.replace_logits(157)  # charades has 157 activity types
i3d_model.load_state_dict(torch.load(args.load_model))
i3d_model.cuda()
i3d_model.train(False)
video_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

# load video ids
video_ids = []
for filename in ["charades_sta_train.txt", "charades_sta_test.txt"]:
    with open(os.path.join(args.dataset_dir, filename), mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.lstrip().rstrip()
            if len(line) == 0:
                continue
            vid = line.split("##")[0].split(" ")[0]
            video_ids.append(vid)
video_ids = list(set(video_ids))

# extract images and features
feature_shapes = dict()
for idx, video_id in enumerate(video_ids):
    video_path = os.path.join(args.video_dir, "{}.mp4".format(video_id))
    image_dir = os.path.join(args.images_dir, video_id)

    print("{} / {}: extract features for video {}".format(idx + 1, len(video_ids), video_id), flush=True)

    if os.path.exists(os.path.join(args.save_dir, "{}.npy".format(video_id))):
        print("the visual features for video {} are exist in {}...\n".format(video_id, args.save_dir), flush=True)
        continue

    # extract images
    if os.path.exists(image_dir):
        print("the images for video {} already are exist in {}...".format(video_id, args.images_dir))
    else:
        os.makedirs(image_dir)
        print("extract images with fps={}...".format(args.fps), flush=True)
        if args.fps is None or args.fps <= 0:
            subprocess.call("ffmpeg -hide_banner -loglevel panic -i {} {}/{}-%6d.jpg".format(
                video_path, image_dir, video_id), shell=True)
        else:
            subprocess.call("ffmpeg -hide_banner -loglevel panic -i {} -filter:v fps=fps={} {}/{}-%6d.jpg".format(
                video_path, args.fps, image_dir, video_id), shell=True)

    # process extracted images
    print("load RGB frames...", flush=True)
    num_frames = len(os.listdir(image_dir))
    frames, raw_w, raw_h = [], None, None
    for i in range(1, num_frames + 1):
        # cv2.imread() read image with BGR format by default, so we convert it to RGB format
        img = cv2.imread(os.path.join(image_dir, "{}-{}.jpg".format(video_id, str(i).zfill(6))))[:, :, [2, 1, 0]]
        w, h, c = img.shape
        raw_w, raw_h = w, h
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        frames.append(img)
    frames = np.asarray(frames, dtype=np.float32)
    imgs = video_transforms(frames)
    img_tensor = torch.from_numpy(np.expand_dims(imgs.transpose([3, 0, 1, 2]), axis=0))
    print("process images:", (frames.shape[0], raw_w, raw_h, frames.shape[-1]), "-->", frames.shape, "-->",
          imgs.shape, "-->", tuple(img_tensor.size()), flush=True)

    if args.remove_images:
        # remove extract images to release memory space
        subprocess.call("rm -rf {}".format(image_dir), shell=True)

    print("extract visual visual features...", flush=True)
    b, c, t, h, w = img_tensor.shape
    features = []
    for start in range(0, t, args.strides):
        end = min(t - 1, start + args.strides)
        if end - start < args.strides:
            start = max(0, end - args.strides)
        ip = Variable(torch.from_numpy(img_tensor.numpy()[:, :, start:end]).cuda(), volatile=True)
        feature = i3d_model.extract_features(ip).data.cpu().numpy()
        features.append(feature)
    features = np.concatenate(features, axis=0)
    np.save(os.path.join(args.save_dir, video_id), arr=features)
    print("extracted feature shape: {}\n".format(features.shape), flush=True)
    feature_shapes[video_id] = features.shape[0]

with open(os.path.join(args.save_dir, "feature_shapes.json"), mode="w", encoding="utf-8") as f:
    json.dump(feature_shapes, f)
