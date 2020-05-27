import math
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils.data_utils import batch_iter


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


def convert_to_time(start_index, end_index, num_features, duration):
    s_times = np.arange(0, num_features).astype(np.float32) * duration / float(num_features)
    e_times = np.arange(1, num_features + 1).astype(np.float32) * duration / float(num_features)

    start_time = s_times[start_index]
    end_time = e_times[end_index]

    return start_time, end_time


def get_feed_dict(batch_data, model, drop_rate=None, mode='train'):
    if mode == 'train':  # training
        (_, video_features, word_ids, char_ids, video_seq_length, start_label, end_label, highlight_labels) = batch_data

        feed_dict = {model.video_inputs: video_features, model.video_seq_length: video_seq_length,
                     model.word_ids: word_ids, model.char_ids: char_ids, model.y1: start_label, model.y2: end_label,
                     model.drop_rate: drop_rate, model.highlight_labels: highlight_labels}

        return feed_dict

    else:  # eval
        raw_data, video_features, word_ids, char_ids, video_seq_length, *_ = batch_data

        feed_dict = {model.video_inputs: video_features, model.video_seq_length: video_seq_length,
                     model.word_ids: word_ids, model.char_ids: char_ids}

        return raw_data, feed_dict


def eval_test(sess, model, dataset, video_features, configs, epoch=None, global_step=None, name="test"):
    num_test_batches = math.ceil(len(dataset) / configs.batch_size)
    ious = list()

    for data in tqdm(batch_iter(dataset, video_features, configs.batch_size, configs.extend, False),
                     total=num_test_batches, desc="evaluate {}".format(name)):

        raw_data, feed_dict = get_feed_dict(data, model, mode=name)
        start_indexes, end_indexes = sess.run([model.start_index, model.end_index], feed_dict=feed_dict)

        for record, start_index, end_index in zip(raw_data, start_indexes, end_indexes):
            start_time, end_time = convert_to_time(start_index, end_index, record["feature_shape"], record["duration"])
            iou = calculate_iou(i0=[start_time, end_time], i1=[record["start_time"], record["end_time"]])
            ious.append(iou)

    r1i3 = calculate_iou_accuracy(ious, threshold=0.3)
    r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
    r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
    mi = np.mean(ious) * 100.0

    value_pairs = [("{}/Rank@1, IoU=0.3".format(name), r1i3), ("{}/Rank@1, IoU=0.5".format(name), r1i5),
                   ("{}/Rank@1, IoU=0.7".format(name), r1i7), ("{}/mean IoU".format(name), mi)]

    # write the scores
    score_str = "Epoch {}, Step {}:\n".format(epoch, global_step)
    score_str += "Rank@1, IoU=0.3: {:.2f}\t".format(r1i3)
    score_str += "Rank@1, IoU=0.5: {:.2f}\t".format(r1i5)
    score_str += "Rank@1, IoU=0.7: {:.2f}\t".format(r1i7)
    score_str += "mean IoU: {:.2f}\n".format(mi)
    return r1i3, r1i5, r1i7, mi, value_pairs, score_str
