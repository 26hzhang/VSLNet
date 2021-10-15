# Span-based Localizing Network for Natural Language Video Localization

This is implementation for the paper "Span-based Localizing Network for Natural Language Video 
Localization" (**ACL 2020**, long paper): [ACL version](https://www.aclweb.org/anthology/2020.acl-main.585.pdf), 
[ArXiv version](https://arxiv.org/abs/2004.13931).

![overview](/figures/overview.jpg)

## Updates
- 2021/06/06: rewrite and optimize the codes, and upload complete visual features to the Box drive. Add the stacked
transformers predictor head (VSLNet with transformer head performs better than that of rnn head in general).
- 2021/07/21: add support to TensorFlow 2.x (test on Tensorflow `2.5.0` with cuda `11.2` and cudnn `8.2`).
```shell
# preparing environment for TensorFlow 2.5.0
conda create --name vslnet_tf2 python=3.9
conda activate vslnet_tf2
conda install -c conda-forge cudnn  # will install cuda 11.2 automatically
pip install tensorflow-gpu==2.5.0
pip install nltk
pip install torch torchvision torchaudio
python3.9 -m nltk.downloader punkt
```

## Prerequisites
- python 3.x with tensorflow (`1.13.1`), pytorch (`1.1.0`), torchvision, opencv-python, moviepy, tqdm, nltk, 
  transformers
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
conda install -c conda-forge transformers opencv moviepy tqdm youtube-dl
# download punkt for word tokenizer
python3.7 -m nltk.downloader punkt
```

## Preparation
The details about how to prepare the `Charades-STA`, `ActivityNet Captions` and `TACoS` features are summarized 
here: [[data preparation]](/prepare). Alternatively, you can download the prepared visual features from 
[Box Drive](https://app.box.com/s/h0sxa5klco6qve5ahnz50ly2nksmuedw), and place them to the `./data/` directory.
Download the word embeddings from [here](http://nlp.stanford.edu/data/glove.840B.300d.zip) and place it to 
`./data/features/` directory.

## Quick Start
### TensorFlow version
**Train** and **Test**
```shell script
# processed dataset will be automatically generated or loaded if exist
# set `--mode test` for evaluation
# set `--predictor transformer` to change the answer predictor from stacked lstms to stacked transformers
# train VSLNet on Charades-STA dataset
python main.py --task charades --predictor rnn --mode train
# train VSLNet on ActivityNet Captions dataset
python main.py --task activitynet --predictor rnn --mode train
# train VSLNet on TACoS dataset
python main.py --task tacos --predictor rnn --mode train
```
Please refer each python file for more parameter settings. You can also download the checkpoints for each task 
from [here](https://app.box.com/s/f20aeutwp2wg8c5laaqtbfdg864g8mj0) and the corresponding processed dataset from
[here](https://app.box.com/s/065efky2sjjgc2xxzyelast15y7tsehs), and save them to the `./ckpt/` and `./datasets/` 
directories, respectively. More hyper-parameter settings are in the `main.py`.

### Pytorch Version
**Train** and **Test**
```shell script
# the same as the usage of tf version
# train VSLNet on Charades-STA dataset
python main.py --task charades --predictor rnn --mode train
# train VSLNet on ActivityNet Captions dataset
python main.py --task activitynet --predictor rnn --mode train
# train VSLNet on TACoS dataset
python main.py --task tacos --predictor rnn --mode train
```
> For unknown reasons, the performance of PyTorch codes is inferior to that of TensorFlow codes on some datasets.

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
    pages = "6543--6554"
}
```
and
```
@article{zhang2021natural,
    author={H. {Zhang} and A. {Sun} and W. {Jing} and L. {Zhen} and J. T. {Zhou} and R. S. M. {Goh}},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
    title={Natural Language Video Localization: A Revisit in Span-based Question Answering Framework}, 
    year={2021},
    doi={10.1109/TPAMI.2021.3060449}
}
```
