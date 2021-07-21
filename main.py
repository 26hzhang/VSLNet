import os
import argparse
import tensorflow as tf
from tqdm import tqdm
from model.VSLNet import VSLNet
from util.data_gen import gen_or_load_dataset
from util.data_util import load_video_features, save_json, load_json
from util.data_loader import TrainLoader, TestLoader
from util.runner_utils import set_tf_config, get_feed_dict, write_tf_summary, eval_test

if tf.__version__.startswith('2'):
    tf = tf.compat.v1
    tf.disable_v2_behavior()
    tf.disable_eager_execution()

parser = argparse.ArgumentParser()
# data parameters
parser.add_argument('--save_dir', type=str, default='datasets', help='path to save processed dataset')
parser.add_argument('--task', type=str, default='charades', help='target task')
parser.add_argument('--fv', type=str, default='new', help='[new | org] for visual features')
parser.add_argument('--max_pos_len', type=int, default=128, help='maximal position sequence length allowed')
# model parameters
parser.add_argument("--char_size", type=int, default=None, help="number of characters")
parser.add_argument("--word_dim", type=int, default=300, help="word embedding dimension")
parser.add_argument("--video_feature_dim", type=int, default=1024, help="video feature input dimension")
parser.add_argument("--char_dim", type=int, default=50, help="character dimension, set to 100 for activitynet")
parser.add_argument("--hidden_size", type=int, default=128, help="hidden size")
parser.add_argument("--highlight_lambda", type=float, default=5.0, help="lambda for highlight region")
parser.add_argument("--num_heads", type=int, default=8, help="number of heads")
parser.add_argument("--drop_rate", type=float, default=0.2, help="dropout rate")
parser.add_argument('--predictor', type=str, default='rnn', help='[rnn | transformer]')
# training/evaluation parameters
parser.add_argument("--gpu_idx", type=str, default="0", help="GPU index")
parser.add_argument("--seed", type=int, default=12345, help="random seed")
parser.add_argument("--mode", type=str, default="train", help="[train | test]")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--num_train_steps", type=int, default=None, help="number of training steps")
parser.add_argument("--init_lr", type=float, default=0.0001, help="initial learning rate")
parser.add_argument("--clip_norm", type=float, default=1.0, help="gradient clip norm")
parser.add_argument("--warmup_proportion", type=float, default=0.0, help="warmup proportion")
parser.add_argument("--extend", type=float, default=0.1, help="highlight region extension")
parser.add_argument("--period", type=int, default=100, help="training loss print period")
parser.add_argument('--model_dir', type=str, default='ckpt', help='path to save trained model weights')
parser.add_argument('--model_name', type=str, default='vslnet', help='model name')
parser.add_argument('--suffix', type=str, default=None, help='set to the last `_xxx` in ckpt repo to eval results')
configs = parser.parse_args()

# set tensorflow configs
set_tf_config(configs.seed, configs.gpu_idx)

# prepare or load dataset
if tf.__version__.startswith('2'):
    configs.save_dir = 'datasets_tf2'  # avoid `ValueError: unsupported pickle protocol: 5`
    configs.model_dir = 'ckpt_tf2'
dataset = gen_or_load_dataset(configs)
configs.char_size = dataset['n_chars']

# get train and test loader
visual_features = load_video_features(os.path.join('data', 'features', configs.task, configs.fv), configs.max_pos_len)
train_loader = TrainLoader(dataset=dataset['train_set'], visual_features=visual_features, configs=configs)
test_loader = TestLoader(datasets=dataset, visual_features=visual_features, configs=configs)
configs.num_train_steps = train_loader.num_batches() * configs.epochs
num_train_batches = train_loader.num_batches()

# create model dir
home_dir = os.path.join(configs.model_dir, '_'.join([configs.model_name, configs.task, configs.fv,
                                                     str(configs.max_pos_len), configs.predictor]))
if configs.suffix is not None:
    home_dir = home_dir + '_' + configs.suffix
log_dir = os.path.join(home_dir, "event")
model_dir = os.path.join(home_dir, "model")

# train and test
if configs.mode.lower() == 'train':
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    eval_period = num_train_batches // 2
    save_json(vars(configs), os.path.join(model_dir, 'configs.json'), sort_keys=True, save_pretty=True)
    with tf.Graph().as_default() as graph:
        model = VSLNet(configs, graph=graph, vectors=dataset['word_vector'])
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            saver = tf.train.Saver(max_to_keep=3)
            writer = tf.summary.FileWriter(log_dir)
            sess.run(tf.global_variables_initializer())
            best_r1i7 = -1.0
            score_writer = open(os.path.join(model_dir, "eval_results.txt"), mode="w", encoding="utf-8")
            for epoch in range(configs.epochs):
                for data in tqdm(train_loader.batch_iter(), total=num_train_batches, desc='Epoch %3d / 3%d' % (
                        epoch + 1, configs.epochs)):
                    # run the model
                    feed_dict = get_feed_dict(data, model, drop_rate=configs.drop_rate)
                    _, loss, h_loss, global_step = sess.run([model.train_op, model.loss, model.highlight_loss,
                                                             model.global_step], feed_dict=feed_dict)
                    if global_step % configs.period == 0:
                        write_tf_summary(writer, [("train/loss", loss), ("train/highlight_loss", h_loss)], global_step)
                    # evaluate
                    if global_step % eval_period == 0 or global_step % num_train_batches == 0:
                        r1i3, r1i5, r1i7, mi, value_pairs, score_str = eval_test(
                            sess=sess, model=model, data_loader=test_loader, epoch=epoch + 1, global_step=global_step,
                            mode="test")
                        print('\nEpoch: %2d | Step: %5d | r1i3: %.2f | r1i5: %.2f | r1i7: %.2f | mIoU: %.2f' % (
                            epoch + 1, global_step, r1i3, r1i5, r1i7, mi), flush=True)
                        write_tf_summary(writer, value_pairs, global_step)
                        score_writer.write(score_str)
                        score_writer.flush()
                        if r1i7 > best_r1i7:
                            best_r1i7 = r1i7
                            filename = os.path.join(model_dir, "{}_{}.ckpt".format(configs.model_name, global_step))
                            saver.save(sess, filename)
            score_writer.close()

elif configs.mode.lower() == 'test':
    if not os.path.exists(model_dir):
        raise ValueError('No pre-trained weights exist')
    # load previous configs
    pre_configs = load_json(os.path.join(model_dir, "configs.json"))
    parser.set_defaults(**pre_configs)
    configs = parser.parse_args()
    with tf.Graph().as_default() as graph:
        model = VSLNet(configs, graph=graph, vectors=dataset['word_vector'])
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))
            r1i3, r1i5, r1i7, mi, *_ = eval_test(sess, model, data_loader=test_loader, mode="test")
            print("\n" + "\x1b[1;31m" + "Rank@1, IoU=0.3:\t{:.2f}".format(r1i3) + "\x1b[0m", flush=True)
            print("\x1b[1;31m" + "Rank@1, IoU=0.5:\t{:.2f}".format(r1i5) + "\x1b[0m", flush=True)
            print("\x1b[1;31m" + "Rank@1, IoU=0.7:\t{:.2f}".format(r1i7) + "\x1b[0m", flush=True)
            print("\x1b[1;31m" + "{}:\t{:.2f}".format("mean IoU".ljust(15), mi) + "\x1b[0m", flush=True)

else:
    raise ValueError("Unknown mode {}!!!".format(configs.mode))
