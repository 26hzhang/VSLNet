import random
import numpy as np
import torch
import math
import torch.utils.data
from tqdm import tqdm


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def convert_length_to_mask(lengths):
    max_len = lengths.max().item()
    mask = torch.arange(max_len, device=lengths.device).expand(lengths.size()[0], max_len) < lengths.unsqueeze(1)
    mask = mask.float()
    return mask


def index_to_time(start_index, end_index, num_units, duration):
    s_times = np.arange(0, num_units).astype(np.float32) * duration / float(num_units)
    e_times = np.arange(1, num_units + 1).astype(np.float32) * duration / float(num_units)
    start_time = s_times[start_index]
    end_time = e_times[end_index]
    return start_time, end_time


def calculate_iou(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
    return max(0.0, iou)


def calculate_iou_accuracy(ious, threshold):
    count = 0
    for iou in ious:
        if iou >= threshold:
            count += 1
    return float(count) / float(len(ious)) * 100.0


def index_to_time_batch(start_indices, end_indices, num_units, durations):
    start_indices = start_indices.cpu().numpy()
    end_indices = end_indices.cpu().numpy()
    num_units = num_units.cpu().numpy()
    start_times, end_times = list(), list()
    for start_index, end_index, num_unit, duration in zip(start_indices, end_indices, num_units, durations):
        start_time, end_time = index_to_time(start_index, end_index, num_unit, duration)
        start_times.append(start_time)
        end_times.append(end_time)
    start_times = np.array(start_times, dtype=np.float32)
    end_times = np.array(end_times, dtype=np.float32)
    return start_times, end_times


def evaluate_predictor(model, data_loader, device, epoch, num_samples, configs):
    ious, ious_time = [], []
    with torch.no_grad():
        for idx, (video_features, word_ids, char_ids, s_times, e_times, _, _, _, num_units, durations) in tqdm(
                enumerate(data_loader), total=math.ceil(num_samples / configs.batch_size), desc='epoch %d' % (epoch+1)):
            # prepare features
            video_features, num_units = video_features.to(device), num_units.to(device)
            word_ids, char_ids = word_ids.to(device), char_ids.to(device)
            # generate mask
            query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(device)
            video_mask = convert_length_to_mask(num_units).to(device)
            # compute predicted results
            _, start_logits, end_logits = model(word_ids, char_ids, video_features, video_mask, query_mask)
            start_indices, end_indices = model.extract_index(start_logits, end_logits)
            # convert to time
            ps_times, pe_times = index_to_time_batch(start_indices, end_indices, num_units, durations)
            # compute ious
            for ps_time, pe_time, s_time, e_time in zip(ps_times, pe_times, s_times.cpu().numpy(),
                                                        e_times.cpu().numpy()):
                iou = calculate_iou([ps_time, pe_time], [s_time, e_time])
                ious.append(iou)
    # compute score
    r1i3 = calculate_iou_accuracy(ious, threshold=0.3)
    r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
    r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
    mi = np.mean(ious) * 100.0
    return r1i3, r1i5, r1i7, mi
