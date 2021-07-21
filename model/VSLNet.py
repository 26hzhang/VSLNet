import tensorflow as tf
from model.ops import create_optimizer, count_params
from model.layers import word_embedding_lookup, char_embedding_lookup, conv1d, video_query_attention, highlight_layer
from model.layers import context_query_concat, feature_encoder, conditioned_predictor, localization_loss

if tf.__version__.startswith('2'):
    tf = tf.compat.v1
    tf.disable_v2_behavior()
    tf.disable_eager_execution()


class VSLNet:
    def __init__(self, configs, graph, vectors):
        self.configs = configs
        graph = graph if graph is not None else tf.Graph()
        with graph.as_default():
            self.global_step = tf.train.create_global_step()
            self._add_placeholders()
            self._build_model(vectors)
            if configs.mode == 'train':
                print('\x1b[1;33m' + 'Total trainable parameters: {}'.format(count_params()) + '\x1b[0m', flush=True)
            else:
                print('\x1b[1;33m' + 'Total parameters: {}'.format(count_params()) + '\x1b[0m', flush=True)

    def _add_placeholders(self):
        self.video_inputs = tf.placeholder(dtype=tf.float32, shape=[None, None, self.configs.video_feature_dim],
                                           name='video_inputs')
        self.video_seq_length = tf.placeholder(dtype=tf.int32, shape=[None], name='video_sequence_length')
        self.word_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='word_ids')
        self.char_ids = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='char_ids')
        self.highlight_labels = tf.placeholder(dtype=tf.int32, shape=[None, None], name='highlight_labels')
        self.y1 = tf.placeholder(dtype=tf.int32, shape=[None, None], name='start_indexes')
        self.y2 = tf.placeholder(dtype=tf.int32, shape=[None, None], name='end_indexes')
        # hyper-parameters
        self.drop_rate = tf.placeholder_with_default(input=0.0, shape=[], name='dropout_rate')
        # create mask
        self.v_mask = tf.sequence_mask(lengths=self.video_seq_length, maxlen=tf.reduce_max(self.video_seq_length),
                                       dtype=tf.int32)
        self.q_mask = tf.cast(tf.cast(self.word_ids, dtype=tf.bool), dtype=tf.int32)

    def _build_model(self, vectors):
        # word embedding & visual features
        word_emb = word_embedding_lookup(self.word_ids, dim=self.configs.word_dim, drop_rate=self.drop_rate,
                                         vectors=vectors, finetune=False, reuse=False, name='word_embeddings')
        char_emb = char_embedding_lookup(self.char_ids, char_size=self.configs.char_size, dim=self.configs.char_dim,
                                         kernels=[1, 2, 3, 4], filters=[10, 20, 30, 40], drop_rate=self.drop_rate,
                                         activation=tf.nn.relu, reuse=False, name='char_embeddings')
        word_emb = tf.concat([word_emb, char_emb], axis=-1)
        video_features = tf.nn.dropout(self.video_inputs, rate=self.drop_rate)
        # feature projection (map both word and video feature to the same dimension)
        vfeats = conv1d(video_features, dim=self.configs.hidden_size, use_bias=True, reuse=False, name='video_conv1d')
        qfeats = conv1d(word_emb, dim=self.configs.hidden_size, use_bias=True, reuse=False, name='query_conv1d')
        # feature encoder
        vfeats = feature_encoder(vfeats, hidden_size=self.configs.hidden_size, num_heads=self.configs.num_heads,
                                 max_position_length=self.configs.max_pos_len, drop_rate=self.drop_rate,
                                 mask=self.v_mask, reuse=False, name='feature_encoder')
        qfeats = feature_encoder(qfeats, hidden_size=self.configs.hidden_size, num_heads=self.configs.num_heads,
                                 max_position_length=self.configs.max_pos_len, drop_rate=self.drop_rate,
                                 mask=self.q_mask, reuse=True, name='feature_encoder')
        # video query attention
        outputs, self.vq_score = video_query_attention(vfeats, qfeats, self.v_mask, self.q_mask, reuse=False,
                                                       drop_rate=self.drop_rate, name='video_query_attention')
        # weighted pooling and concatenation
        outputs = context_query_concat(outputs, qfeats, q_mask=self.q_mask, reuse=False, name='context_query_concat')
        # highlighting layer
        self.highlight_loss, self.highlight_scores = highlight_layer(outputs, self.highlight_labels, mask=self.v_mask,
                                                                     reuse=False, name='highlighting_layer')
        outputs = tf.multiply(outputs, tf.expand_dims(self.highlight_scores, axis=-1))
        # prediction layer
        start_logits, end_logits = conditioned_predictor(outputs, hidden_size=self.configs.hidden_size,
                                                         seq_len=self.video_seq_length, mask=self.v_mask,
                                                         num_heads=self.configs.num_heads, drop_rate=self.drop_rate,
                                                         max_position_length=self.configs.max_pos_len, reuse=False,
                                                         mode=self.configs.predictor, name='conditioned_predictor')
        # compute localization loss
        self.start_prob, self.end_prob, self.start_index, self.end_index, self.loss = localization_loss(
            start_logits, end_logits, self.y1, self.y2)
        # add l2 regularizer loss (uncomment if required)
        l2_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += tf.reduce_sum(l2_losses)
        # collect regularization losses
        self.total_loss = self.loss + self.configs.highlight_lambda * self.highlight_loss
        # create optimizer
        if self.configs.warmup_proportion > 1.0:
            num_warmup_steps = int(self.configs.warmup_proportion)
        else:
            num_warmup_steps = int(self.configs.num_train_steps * self.configs.warmup_proportion)
        self.train_op = create_optimizer(self.total_loss, self.configs.init_lr, self.configs.num_train_steps,
                                         num_warmup_steps, clip_norm=self.configs.clip_norm)
