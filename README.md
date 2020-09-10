# Span-based Localizing Network for Natural Language Video Localization

This is **TensorFlow** implementation for the paper "Span-based Localizing Network for Natural Language Video 
Localization" (**ACL 2020**, long paper): 
[https://www.aclweb.org/anthology/2020.acl-main.585.pdf](https://www.aclweb.org/anthology/2020.acl-main.585.pdf), 
[https://arxiv.org/abs/2004.13931](https://arxiv.org/abs/2004.13931).

![overview](/figures/overview.jpg)

## Prerequisites
- python 3.x with tensorflow (`1.13.1`), pytorch (`<=1.1.0`), torchvision, opencv-python, moviepy, tqdm, nltk
- youtube-dl
- cuda10, cudnn

If you have [Anaconda](https://www.anaconda.com/distribution/) installed, the conda environment of VSLNet can be built 
as follow (take python 3.7 as an example):
```shell script
# preparing environment
conda create --name vslnet python=3.7
conda activate vslnet
conda install -c anaconda cudatoolkit=10.0 cudnn
conda install tensorflow-gpu==1.13.1
conda install -c anaconda nltk pillow=6.2.1
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
conda install -c conda-forge opencv moviepy tqdm youtube-dl

# download punkt for word tokenizer
python3.7 -m nltk.downloader punkt
```

## Preparation
The details about how to prepare the `Charades-STA`, `ActivityNet Captions` and `TACoS` datasets are summarized 
here: [[data preparation]](/prepare). Alternatively, you can download the prepared visual features, word embeddings and 
data from [Box Drive](https://app.box.com/s/anywugpxlt134r9hzqf5v3v5xohxliwu), and save them to the `./data/` folder.

## Quick Start
**Data Pre-processing**  
```shell script
# pre-processing the Charades-STA dataset, `feature` argument indicates whether to use 
# visual features finetuned on Charades dataset (`finetune`) or w/o finetune (`raw`).
python run_charades.py --mode prepro --feature finetune

# pre-processing the ActivityNet Captions dataset
python run_activitynet.py --mode prepro

# pre-processing the TACoS dataset
python run_tacos.py --mode prepro
```
You can download the pre-processed datasets from [Box Drive](https://app.box.com/s/qhccf6f1dm4llcto3vh34xciz3sbys92), 
and save them to the `./datasets/` folder.

**Train**
```shell script
# train VSLNet on Charades-STA dataset
python run_charades.py --mode train

# train VSLNet on ActivityNet Captions dataset
python run_activitynet.py --mode train

# train VSLNet on TACoS dataset
python run_tacos.py --mode train
```
Please refer each python file for more parameter settings. You can also download the trained weights for each task from 
[Box Drive](https://app.box.com/s/40wn9kh2eqpnel5ofjcr2qqsu9uyn6ba), and save them to the `./ckpt/` folder.

**Test**
```shell script
# test VSLNet on Charades-STA dataset
python run_charades.py --mode test

# test VSLNet on ActivityNet Captions dataset
python run_activitynet.py --mode test

# test VSLNet on TACoS dataset
python run_tacos.py --mode test
```

## Citation
If you feel this project helpful to your research, please cite our work.
```
@inproceedings{zhang2020span,
    title = "Span-based Localizing Network for Natural Language Video Localization",
    author = "Zhang, Hao  and Sun, Aixin  and Jing, Wei  and Zhou, Joey Tianyi",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.585",
    pages = "6543--6554",
}
```
