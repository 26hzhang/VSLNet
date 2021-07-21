import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from util.data_util import index_to_time

if tf.__version__.startswith('2'):
    tf = tf.compat.v1
    tf.disable_v2_behavior()
    tf.disable_eager_execution()


def set_tf_config(seed, gpu_idx):
    # os environment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
    # random seed
    np.random.seed(seed)
    tf.set_random_seed(seed)
    tf.random.set_random_seed(seed)


def write_tf_summary(writer, value_pairs, global_step):
    for tag, value in value_pairs:
        summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        writer.add_summary(summ, global_step=global_step)
    writer.flush()


def calculate_iou_accuracy(ious, threshold):
    total_size = float(len(ious))
    count = 0
    for iou in ious:
        if iou >= threshold:
            count += 1
    return float(count) / total_size * 100.0


def calculate_iou(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
    return max(0.0, iou)


def get_feed_dict(batch_data, model, drop_rate=None, mode='train'):
    if mode == 'train':  # training
        (_, vfeats, vfeat_lens, word_ids, char_ids, s_labels, e_labels, h_labels) = batch_data
        feed_dict = {model.video_inputs: vfeats, model.video_seq_length: vfeat_lens, model.word_ids: word_ids,
                     model.char_ids: char_ids, model.y1: s_labels, model.y2: e_labels, model.drop_rate: drop_rate,
                     model.highlight_labels: h_labels}
        return feed_dict
    else:  # eval
        raw_data, vfeats, vfeat_lens, word_ids, char_ids = batch_data
        feed_dict = {model.video_inputs: vfeats, model.video_seq_length: vfeat_lens, model.word_ids: word_ids,
                     model.char_ids: char_ids}
        return raw_data, feed_dict


def eval_test(sess, model, data_loader, epoch=None, global_step=None, mode="test"):
    ious = list()
    for data in tqdm(data_loader.test_iter(mode), total=data_loader.num_batches(mode), desc="evaluate {}".format(mode)):
        raw_data, feed_dict = get_feed_dict(data, model, mode=mode)
        start_indexes, end_indexes = sess.run([model.start_index, model.end_index], feed_dict=feed_dict)
        for record, start_index, end_index in zip(raw_data, start_indexes, end_indexes):
            start_time, end_time = index_to_time(start_index, end_index, record["v_len"], record["duration"])
            iou = calculate_iou(i0=[start_time, end_time], i1=[record["s_time"], record["e_time"]])
            ious.append(iou)
    r1i3 = calculate_iou_accuracy(ious, threshold=0.3)
    r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
    r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
    mi = np.mean(ious) * 100.0
    value_pairs = [("{}/Rank@1, IoU=0.3".format(mode), r1i3), ("{}/Rank@1, IoU=0.5".format(mode), r1i5),
                   ("{}/Rank@1, IoU=0.7".format(mode), r1i7), ("{}/mean IoU".format(mode), mi)]
    # write the scores
    score_str = "Epoch {}, Step {}:\n".format(epoch, global_step)
    score_str += "Rank@1, IoU=0.3: {:.2f}\t".format(r1i3)
    score_str += "Rank@1, IoU=0.5: {:.2f}\t".format(r1i5)
    score_str += "Rank@1, IoU=0.7: {:.2f}\t".format(r1i7)
    score_str += "mean IoU: {:.2f}\n".format(mi)
    return r1i3, r1i5, r1i7, mi, value_pairs, score_str
