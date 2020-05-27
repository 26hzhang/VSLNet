import os
import cv2
import glob
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
parser.add_argument("--load_model", type=str, required=True, help="pre-trained model")
parser.add_argument("--video_dir", type=str, required=True, help="where are located the videos")
parser.add_argument("--images_dir", type=str, required=True, help="where to save extracted images")
parser.add_argument("--save_dir", type=str, required=True, help="where to save extracted features")
parser.add_argument("--fps", type=int, default=None, help="frames per second")
parser.add_argument("--video_format", type=str, default="mp4", help="video format")
parser.add_argument("--strides", type=int, default=16, help="window size")
parser.add_argument("--remove_images", action="store_true", help="whether remove extract images to release space")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx


def load_images(img_dir, vid, start_frame, lengths):
    img_frames, raw_height, raw_width = [], None, None
    for x in range(start_frame, start_frame + lengths):
        image = cv2.imread(os.path.join(img_dir, "{}-{}.jpg".format(vid, str(x).zfill(6))))[:, :, [2, 1, 0]]
        width, height, channel = image.shape
        raw_width, raw_height = width, height
        # resize image
        scale = 1 + (224.0 - min(width, height)) / min(width, height)
        image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale)
        # normalize image to [0, 1]
        image = (image / 255.0) * 2 - 1
        img_frames.append(image)
    return img_frames, raw_width, raw_height


def extract_features(image_tensor, model, strides):
    b, c, t, h, w = image_tensor.shape
    extracted_features = []
    for start in range(0, t, strides):
        end = min(t - 1, start + strides)
        if end - start < strides:
            start = max(0, end - strides)
        ip = Variable(torch.from_numpy(image_tensor.numpy()[:, :, start:end]).cuda(), volatile=True)
        feature = model.extract_features(ip).data.cpu().numpy()
        extracted_features.append(feature)
    extracted_features = np.concatenate(extracted_features, axis=0)
    return extracted_features


if not os.path.exists(args.video_dir):
    raise ValueError("The video directory '{}' does not exist!!!".format(args.video_dir))

if not os.path.exists(args.images_dir):
    os.makedirs(args.images_dir)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# create I3D model and load pre-trained model
i3d_model = InceptionI3d(400, in_channels=3)
i3d_model.load_state_dict(torch.load(args.load_model))
i3d_model.cuda()
i3d_model.train(False)
video_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

# extract images and features
feature_shapes = dict()
video_paths = glob.glob(os.path.join(args.video_dir, "*.{}".format(args.video_format)))
for idx, video_path in enumerate(video_paths):
    video_id = os.path.basename(video_path)[0:-4]  # remove suffix
    image_dir = os.path.join(args.images_dir, video_id)

    print("{} / {}: extract features for video {}".format(idx + 1, len(video_paths), video_id), flush=True)

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

    if num_frames < 10000:
        frames, raw_w, raw_h = load_images(image_dir, video_id, 1, num_frames)
        frames = np.asarray(frames, dtype=np.float32)
        imgs = video_transforms(frames)
        img_tensor = torch.from_numpy(np.expand_dims(imgs.transpose([3, 0, 1, 2]), axis=0))
        print("process images:", (frames.shape[0], raw_w, raw_h, frames.shape[-1]), "-->", frames.shape, "-->",
              imgs.shape, "-->", tuple(img_tensor.size()), flush=True)

        print("extract visual features...", flush=True)
        features = extract_features(img_tensor, i3d_model, args.strides)
        np.save(os.path.join(args.save_dir, video_id), arr=features)
        print("extracted features shape: {}".format(features.shape), flush=True)
        feature_shapes[video_id] = features.shape[0]

    else:
        all_features = []
        for start_idx in range(1, num_frames, 10000):
            end_idx = min(start_idx + 10000, num_frames + 1)
            cur_num_frames = end_idx - start_idx
            if cur_num_frames < args.strides:
                cur_num_frames = args.strides
                start_idx = end_idx - cur_num_frames
            frames, raw_w, raw_h = load_images(image_dir, video_id, start_idx, cur_num_frames)
            frames = np.asarray(frames, dtype=np.float32)
            imgs = video_transforms(frames)
            img_tensor = torch.from_numpy(np.expand_dims(imgs.transpose([3, 0, 1, 2]), axis=0))
            print("process images:", (frames.shape[0], raw_w, raw_h, frames.shape[-1]), "-->", frames.shape, "-->",
                  imgs.shape, "-->", tuple(img_tensor.size()), flush=True)
            print("extract visual features...", flush=True)
            features = extract_features(img_tensor, i3d_model, args.strides)
            all_features.append(features)
        all_features = np.concatenate(all_features, axis=0)
        np.save(os.path.join(args.save_dir, video_id), arr=all_features)
        print("extracted features shape: {}".format(all_features.shape), flush=True)
        feature_shapes[video_id] = all_features.shape[0]

    if args.remove_images:
        # remove extract images to release memory space
        subprocess.call("rm -rf {}".format(image_dir), shell=True)

with open(os.path.join(args.save_dir, "feature_shapes.json"), mode="w", encoding="utf-8") as f:
    json.dump(feature_shapes, f)
