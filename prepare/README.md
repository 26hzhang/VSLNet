# Extract Features

- We use the pre-trained 3D ConvNets ([here](https://github.com/piergiaj/pytorch-i3d)) to prepare the visual features, the 
extraction codes are placed in this folder. Please download the pre-trained weights [`rgb_charades.pt`](
https://github.com/piergiaj/pytorch-i3d/blob/master/models/rgb_charades.pt) and [`rgb_imagenet.pt`](
https://github.com/piergiaj/pytorch-i3d/blob/master/models/rgb_imagenet.pt). 
- The pre-trained GloVe embedding is available at [here](https://nlp.stanford.edu/projects/glove/), please download
`glove.840B.300d.zip`, unzip and put it under `data/` folder.

## Charades STA
The train/test datasets of Charades-STA are available at [[jiyanggao/TALL]](https://github.com/jiyanggao/TALL) 
([`charades_sta_train.txt`](https://drive.google.com/file/d/1ZjG7wJpPSMIBYnW7BAG2u9VVEoNvFm5c/view) and 
[`charades_sta_test.txt`](https://drive.google.com/file/d/1QG4MXFkoj6JFU0YK5olTY75xTARKSW5e/view)).

The `charades.json` file is required ([here](https://github.com/piergiaj/super-events-cvpr18/blob/master/data/charades.json)), 
which contains the video length information. Download and place it into the same directory of the train/test datasets.

The videos/images for Charades-STA dataset is available at [here](https://allenai.org/plato/charades/), please download 
either `RGB frames at 24fps (76 GB)` (image frames) or `Data (original size) (55 GB)` (videos). For the second one, the 
extractor will automatically decompose the video into images.
```shell script
# download RGB frames
wget http://ai2-website.s3.amazonaws.com/data/Charades_v1_rgb.tar
# or, download videos
wget http://ai2-website.s3.amazonaws.com/data/Charades_v1.zip
```

Extract visual features for Charades-STA:
```shell script
# use the weights fine-tuned on Charades or the weights pre-trained on ImageNet
python3 extract_charades.py --use_finetuned --load_model <path to>/rgb_charades.pt  \  # rgb_imagenet.pt
      --video_dir <path to video dir>  \
      --dataset_dir <path to charades-sta dataset dir>  \
      --images_dir <path to images dir>  \  # if images not exist, decompose video into images
      --save_dir <path to save extracted visual features>  \
      --fps 24 --strides 24 --remove_images  # whether remove extract images to release space
```

## TACoS
TACoS dataset is from [[jiyanggao/TALL]](https://github.com/jiyanggao/TALL), while the videos of TACoS is from MPII 
Cooking Composite Activities dataset, which can be download [here](
https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/human-activity-recognition/mpii-cooking-composite-activities/).
Note that we also use the processed TACoS dataset in [[microsoft/2D-TAN]](https://github.com/microsoft/2D-TAN). 

Extract visual features for TACoS:
```shell script
python3 extract_tacos.py --load_model <path to>/rgb_imagenet.pt  \
      --video_dir <path to video dir>  \
      --dataset_dir <path to charades-sta dataset dir>  \
      --images_dir <path to images dir>  \  # if images not exist, decompose video into images
      --save_dir <path to save extracted visual features>  \
      --strides 16 --remove_images  # whether remove extracted images to release space
```

(Optional) Convert the pre-trained C3D visual features from [[jiyanggao/TALL]](https://github.com/jiyanggao/TALL) 
([Interval64_128_256_512_overlap0.8_c3d_fc6.tar](https://drive.google.com/file/d/1zQp0aYGFCm8PqqHOh4UtXfy2U3pJMBeu/view), 
[Interval128_256_overlap0.8_c3d_fc6.tar](https://drive.google.com/file/d/1zC-UrspRf42Qiu5prQw4fQrbgLQfJN-P/view)):
```shell script
python3 extract_tacos_org.py --data_path <path to tacos annotation dataset>  \
      --feature_path <path to downloaded C3D features>  \
      --save_dir <path to save extracted visual features>  \
      --sample_rate 64  # sliding windows
```

## ActivityNet Captions
The train/test sets of ActivityNet Caption are available at [here](
https://cs.stanford.edu/people/ranjaykrishna/densevid/). The videos can be downloaded using:
```shell script
python3 download_activitynet_video.py --video_dir <path to save videos>  \
      --dataset_dir <path to activitynet caption datasets>  \
      --bash_file <path to save generated bash file for downloading videos>
```
It will generate a bash file which contains the commands to download all the videos. Suppose the generated bash file is 
`video_downloader.sh`, then simply run `bash video_downloader.sh`, it will download the videos and save them into 
`video_dir` automatically.

Extract visual features for ActivityNet Captions:
```shell script
python3 extract_activitynet.py --load_model <path to>/rgb_imagenet.pt  \
      --video_dir <path to video dir>  \
      --dataset_dir <path to charades-sta dataset dir>  \
      --images_dir <path to images dir>  \  # if images not exist, decompose video into images
      --save_dir <path to save extracted visual features>  \
      --strides 16 --remove_images  # whether remove extracted images to release space
```

(Optional) We also have the codes to convert the C3D visual features provided in [ActivityNet official website](
http://activity-net.org/challenges/2016/download.html):

- download the C3D visual features
```shell script
wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-00
wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-01
wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-02
wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-03
wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-04
wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-05
cat activitynet_v1-3.part-* > features.zip && unzip features.zip
rm features.zip
rm activitynet_v1-3.part-*
```
- convert the features as
```shell script
python3 extract_activitynet_org.py --dataset_dir <path to activitynet caption annotation dataset>  \
      --hdf5_file <path to downloaded C3D features>  \
      --save_dir <path to save extracted features>
```
