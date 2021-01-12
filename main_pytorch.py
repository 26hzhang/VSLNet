import os
import glob
import json
import math
from argparse import ArgumentParser

import torch.backends.cudnn
import torch.nn as nn
from vslnet_pytorch.models import LocalizerNetwork, build_optimizer_and_scheduler
from vslnet_pytorch.prepare_dataset import prepare_datasets
from vslnet_pytorch.data_utils import get_data_loader
from vslnet_pytorch.runner_utils import set_random_seed, convert_length_to_mask, evaluate_predictor

parser = ArgumentParser()
# data parameters
parser.add_argument('--save_dir', type=str, default='datasets/', help='path to save processed dataset')
parser.add_argument('--task', type=str, default='charades', help='target task')
parser.add_argument('--max_pos_len', type=int, default=128, help='maximal position sequence length allowed')
# model parameters
parser.add_argument('--num_words', type=int, default=None, help='word dictionary size')
parser.add_argument('--num_chars', type=int, default=None, help='character dictionary size')
parser.add_argument('--word_dim', type=int, default=300, help='word embedding dimension')
parser.add_argument('--char_dim', type=int, default=50, help='character embedding dimension')
parser.add_argument('--visual_dim', type=int, default=1024, help='video feature dimension [i3d: 1024 | c3d: 4096]')
parser.add_argument('--dim', type=int, default=128, help='hidden size of the model')
parser.add_argument('--num_heads', type=int, default=8, help='number of heads in transformer block')
parser.add_argument('--drop_rate', type=float, default=0.2, help='dropout rate')
parser.add_argument('--highlight_lambda', type=float, default=5.0, help='lambda for highlight region')
# training/evaluation parameters
parser.add_argument('--seed', type=int, default=12345, help='random seed')
parser.add_argument('--mode', type=str, default='train', help='[train | test]')
parser.add_argument('--gpu_idx', type=str, default='0', help='indicate which gpu is used')
parser.add_argument('--model_dir', type=str, default='ckpt_t7', help='path to save trained model weights')
parser.add_argument('--model_name', type=str, default='vslnet', help='model name')
parser.add_argument('--epochs', type=int, default=100, help='maximal training epochs')
parser.add_argument('--num_train_steps', type=int, default=None, help='maximal training steps')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument("--clip_norm", type=float, default=1.0, help="gradient clip norm")
parser.add_argument("--init_lr", type=float, default=0.0005, help="initial learning rate")
parser.add_argument("--warmup_proportion", type=float, default=0.0, help="warmup proportion")
parser.add_argument("--period", type=int, default=100, help="training loss print period")
parser.add_argument("--eval_period", type=int, default=500, help="evaluation period")
configs = parser.parse_args()

# torch config
set_random_seed(configs.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# check if dataset is processed
data_path = os.path.join(configs.save_dir, '_'.join([configs.task, str(configs.max_pos_len)]) + '.pkl')
if not os.path.exists(data_path):
    prepare_datasets(configs)

# Device configuration
cuda_str = 'cuda' if configs.gpu_idx is None else 'cuda:{}'.format(configs.gpu_idx)
device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')

# preset model save dir
model_dir = os.path.join(configs.model_dir, '_'.join([configs.model_name.lower(), configs.task,
                                                      str(configs.max_pos_len)]))

if configs.mode == 'train':
    train_loader, test_loader, num_words, num_chars, word_vectors, train_samples, test_samples = get_data_loader(
        configs)
    configs.num_words = num_words
    configs.num_chars = num_chars
    configs.num_train_steps = math.ceil(train_samples / configs.batch_size) * configs.epochs
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(os.path.join(model_dir, 'configs.json'), mode='w', encoding='utf-8') as f:
        json.dump(vars(configs), f, indent=4, sort_keys=True)
    # build model
    model = LocalizerNetwork(configs=configs, word_vectors=word_vectors).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfgs=configs)

    best_r1i7 = -1.0
    score_writer = open(os.path.join(model_dir, "eval_results.txt"), mode="w", encoding="utf-8")
    print('start training...', flush=True)
    step_counter = 0
    for epoch in range(configs.epochs):
        model.train()
        for video_features, word_ids, char_ids, _, _, s_labels, e_labels, h_labels, num_units, _ in train_loader:
            step_counter += 1
            # prepare features
            video_features, num_units = video_features.to(device), num_units.to(device)
            word_ids, char_ids = word_ids.to(device), char_ids.to(device)
            s_labels, e_labels, h_labels = s_labels.to(device), e_labels.to(device), h_labels.to(device)
            # generate mask
            query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(device)
            video_mask = convert_length_to_mask(num_units).to(device)
            # compute logits
            h_score, start_logits, end_logits = model(word_ids, char_ids, video_features, video_mask, query_mask)
            # compute loss
            highlight_loss = model.compute_highlight_loss(h_score, h_labels, video_mask)
            loc_loss = model.compute_loss(start_logits, end_logits, s_labels, e_labels)
            total_loss = loc_loss + configs.highlight_lambda * highlight_loss
            # compute and apply gradients
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), configs.clip_norm)  # clip gradient
            optimizer.step()
            scheduler.step()
            if step_counter % configs.period == 0:
                print(' epoch: %d | step: %d | loss | loc: %.6f | highlight: %.6f' % (
                    epoch + 1, step_counter, loc_loss.item(), highlight_loss.item()), flush=True)
            if step_counter % configs.eval_period == 0:
                model.eval()
                r1i3, r1i5, r1i7, mi = evaluate_predictor(model, test_loader, device, epoch, test_samples, configs)
                print('\n epoch: %d | step: %d | evaluation (Rank@1) | IoU=0.3: %.2f | IoU=0.5: %.2f | IoU=0.7: %.2f | '
                      'mIoU: %.2f\n' % (epoch + 1, step_counter, r1i3, r1i5, r1i7, mi), flush=True)
                score_str = 'epoch: %d | step: %d | Rank@1 | IoU=0.3: %.2f | IoU=0.5: %.2f | IoU=0.7: %.2f | ' \
                            'mIoU: %.2f\n' % (epoch + 1, step_counter, r1i3, r1i5, r1i7, mi)
                score_writer.write(score_str)
                score_writer.flush()
                if r1i7 >= best_r1i7:
                    best_r1i7 = r1i7
                    torch.save(model.state_dict(), os.path.join(model_dir, '{}_epoch_{}_step_{}_model_{:.2f}.t7'.format(
                        configs.model_name, epoch + 1, step_counter, r1i7)))
                model.train()
        model.eval()
        r1i3, r1i5, r1i7, mi = evaluate_predictor(model, test_loader, device, epoch, test_samples, configs)
        print('\n epoch: %d | step: %d | evaluation (Rank@1) | IoU=0.3: %.2f | IoU=0.5: %.2f | IoU=0.7: %.2f | '
              'mIoU: %.2f\n' % (epoch + 1, step_counter, r1i3, r1i5, r1i7, mi), flush=True)
        score_str = 'epoch: %d | step: %d | Rank@1 | IoU=0.3: %.2f | IoU=0.5: %.2f | IoU=0.7: %.2f | mIoU: ' \
                    '%.2f\n' % (epoch + 1, step_counter, r1i3, r1i5, r1i7, mi)
        score_writer.write(score_str)
        score_writer.flush()
        if r1i7 >= best_r1i7:
            best_r1i7 = r1i7
            torch.save(model.state_dict(), os.path.join(model_dir, '{}_epoch_{}_step_{}_model_{:.2f}.t7'.format(
                configs.model_name, epoch + 1, step_counter, r1i7)))
    score_writer.close()

elif configs.mode == 'test':
    if not os.path.exists(model_dir):
        raise ValueError('No pre-trained weights exist')
    with open(os.path.join(model_dir, 'configs.json'), mode='r', encoding='utf-8') as f:
        pre_configs = json.load(f)
    parser.set_defaults(**pre_configs)
    configs = parser.parse_args()
    # load dataset
    _, test_loader, _, _, word_vectors, _, test_samples = get_data_loader(configs)
    # build model
    model = LocalizerNetwork(configs=configs, word_vectors=word_vectors).to(device)
    # testing
    filenames = glob.glob(os.path.join(model_dir, '*.t7'))
    filenames.sort()
    for filename in filenames:
        epoch = int(os.path.basename(filename).split('_')[2])
        step = int(os.path.basename(filename).split('_')[4])
        model.load_state_dict(torch.load(filename))
        model.eval()
        r1i3, r1i5, r1i7, mi = evaluate_predictor(model, test_loader, device, epoch, test_samples, configs)
        print(' epoch: %d | step: %d | evaluation (Rank@1) | IoU=0.3: %.2f | IoU=0.5: %.2f | IoU=0.7: %.2f | '
              'mIoU: %.2f' % (epoch + 1, step, r1i3, r1i5, r1i7, mi), flush=True)

else:
    raise ValueError('Unknown mode, only support [train | test]!!!')
