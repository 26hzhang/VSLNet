import math
import tensorflow as tf
from model.ops import get_shape_list, mask_logits, trilinear_attention, regularizer

if tf.__version__.startswith('2'):
    tf = tf.compat.v1
    tf.disable_v2_behavior()
    tf.disable_eager_execution()


def layer_norm(inputs, epsilon=1e-6, reuse=None, name='layer_norm'):
    """Layer normalize the tensor x, averaging over the last dimension."""
    with tf.variable_scope(name, default_name="layer_norm", values=[inputs], reuse=reuse):
        dim = get_shape_list(inputs)[-1]
        scale = tf.get_variable("layer_norm_scale", [dim], initializer=tf.ones_initializer(), regularizer=regularizer)
        bias = tf.get_variable("layer_norm_bias", [dim], initializer=tf.zeros_initializer(), regularizer=regularizer)
        mean = tf.reduce_mean(inputs, axis=[-1], keep_dims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=[-1], keep_dims=True)
        norm_inputs = (inputs - mean) * tf.rsqrt(variance + epsilon)
        result = norm_inputs * scale + bias
        return result


def word_embedding_lookup(word_ids, dim, vectors, drop_rate=0.0, finetune=False, reuse=None, name='word_embeddings'):
    with tf.variable_scope(name, reuse=reuse):
        table = tf.Variable(vectors, name='word_table', dtype=tf.float32, trainable=finetune)
        unk = tf.get_variable(name='unk', shape=[1, dim], dtype=tf.float32, trainable=True)
        zero = tf.zeros(shape=[1, dim], dtype=tf.float32)
        word_table = tf.concat([zero, unk, table], axis=0)
        word_emb = tf.nn.embedding_lookup(word_table, word_ids)
        word_emb = tf.nn.dropout(word_emb, rate=drop_rate)
        return word_emb


def char_embedding_lookup(char_ids, char_size, dim, kernels, filters, drop_rate=0.0, activation=tf.nn.relu,
                          padding='VALID', reuse=None, name='char_embeddings'):
    with tf.variable_scope(name, reuse=reuse):
        # char embeddings lookup
        table = tf.get_variable(name='char_table', shape=[char_size - 1, dim], dtype=tf.float32, trainable=True)
        zero = tf.zeros(shape=[1, dim], dtype=tf.float32)
        char_table = tf.concat([zero, table], axis=0)
        char_emb = tf.nn.embedding_lookup(char_table, char_ids)
        char_emb = tf.nn.dropout(char_emb, rate=drop_rate)
        # char-level cnn
        outputs = []
        for i, (kernel, channel) in enumerate(zip(kernels, filters)):
            weight = tf.get_variable('filter_%d' % i, shape=[1, kernel, dim, channel], dtype=tf.float32,
                                     regularizer=regularizer)
            bias = tf.get_variable('bias_%d' % i, shape=[channel], dtype=tf.float32, initializer=tf.zeros_initializer(),
                                   regularizer=regularizer)
            output = tf.nn.conv2d(char_emb, weight, strides=[1, 1, 1, 1], padding=padding, name='conv_%d' % i)
            output = tf.nn.bias_add(output, bias=bias)
            output = tf.reduce_max(activation(output), axis=2)
            outputs.append(output)
        outputs = tf.concat(values=outputs, axis=-1)
        return outputs


def conv1d(inputs, dim, kernel_size=1, use_bias=False, activation=None, padding='VALID', reuse=None, name='conv1d'):
    with tf.variable_scope(name, reuse=reuse):
        shapes = get_shape_list(inputs)
        kernel = tf.get_variable(name='kernel', shape=[kernel_size, shapes[-1], dim], dtype=tf.float32,
                                 regularizer=regularizer)
        outputs = tf.nn.conv1d(inputs, filters=kernel, stride=1, padding=padding)
        if use_bias:
            bias = tf.get_variable(name='bias', shape=[1, 1, dim], dtype=tf.float32, initializer=tf.zeros_initializer(),
                                   regularizer=regularizer)
            outputs += bias
        if activation is not None:
            return activation(outputs)
        else:
            return outputs


def depthwise_separable_conv(inputs, kernel_size, dim, use_bias=True, reuse=None, activation=tf.nn.relu,
                             name='depthwise_separable_conv'):
    with tf.variable_scope(name, reuse=reuse):
        shapes = get_shape_list(inputs)
        depthwise_filter = tf.get_variable(name='depthwise_filter', dtype=tf.float32, regularizer=regularizer,
                                           shape=[kernel_size[0], kernel_size[1], shapes[-1], 1])
        pointwise_filter = tf.get_variable(name='pointwise_filter', shape=[1, 1, shapes[-1], dim], dtype=tf.float32,
                                           regularizer=regularizer)
        outputs = tf.nn.separable_conv2d(inputs, depthwise_filter, pointwise_filter, strides=[1, 1, 1, 1],
                                         padding='SAME')
        if use_bias:
            b = tf.get_variable('bias', outputs.shape[-1], initializer=tf.zeros_initializer(), regularizer=regularizer)
            outputs += b
        outputs = activation(outputs)
        return outputs


def add_positional_embedding(inputs, max_position_length, reuse=None, name='positional_embedding'):
    with tf.variable_scope(name, reuse=reuse):
        batch_size, seq_length, dim = get_shape_list(inputs)
        assert_op = tf.assert_less_equal(seq_length, max_position_length)
        with tf.control_dependencies([assert_op]):
            full_position_embeddings = tf.get_variable(name='position_embeddings', shape=[max_position_length, dim],
                                                       dtype=tf.float32)
            position_embeddings = tf.slice(full_position_embeddings, [0, 0], [seq_length, -1])
            num_dims = len(inputs.shape.as_list())
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length, dim])
            position_embeddings = tf.reshape(position_embeddings, shape=position_broadcast_shape)
            outputs = inputs + position_embeddings
        return outputs


def conv_block(inputs, kernel_size, dim, num_layers, drop_rate=0.0, reuse=None, name='conv_block'):
    with tf.variable_scope(name, reuse=reuse):
        outputs = tf.expand_dims(inputs, axis=2)
        for layer_idx in range(num_layers):
            residual = outputs
            outputs = layer_norm(outputs, reuse=reuse, name='layer_norm_%d' % layer_idx)
            outputs = depthwise_separable_conv(outputs, kernel_size=(kernel_size, 1), dim=dim, use_bias=True,
                                               activation=tf.nn.relu, name='depthwise_conv_layers_%d' % layer_idx,
                                               reuse=reuse)
            outputs = tf.nn.dropout(outputs, rate=drop_rate) + residual
        return tf.squeeze(outputs, 2)


def multihead_attention(inputs, dim, num_heads, mask=None, drop_rate=0.0, reuse=None, name='multihead_attention'):
    with tf.variable_scope(name, reuse=reuse):
        if dim % num_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the attention heads (%d)' % (dim, num_heads))
        batch_size, seq_length, _ = get_shape_list(inputs)
        head_size = dim // num_heads

        def transpose_for_scores(input_tensor, batch_size_, seq_length_, num_heads_, head_size_):
            output_tensor = tf.reshape(input_tensor, shape=[batch_size_, seq_length_, num_heads_, head_size_])
            output_tensor = tf.transpose(output_tensor, perm=[0, 2, 1, 3])
            return output_tensor

        # projection
        query = conv1d(inputs, dim=dim, use_bias=True, reuse=reuse, name='query')
        key = conv1d(inputs, dim=dim, use_bias=True, reuse=reuse, name='key')
        value = conv1d(inputs, dim=dim, use_bias=True, reuse=reuse, name='value')
        # reshape & transpose: (batch_size, seq_length, dim) --> (batch_size, num_heads, seq_length, head_size)
        query = transpose_for_scores(query, batch_size, seq_length, num_heads, head_size)
        key = transpose_for_scores(key, batch_size, seq_length, num_heads, head_size)
        value = transpose_for_scores(value, batch_size, seq_length, num_heads, head_size)
        # compute attention score
        query = tf.multiply(query, 1.0 / math.sqrt(float(head_size)))
        attention_score = tf.matmul(query, key, transpose_b=True)
        if mask is not None:
            shapes = get_shape_list(attention_score)
            mask = tf.cast(tf.reshape(mask, shape=[shapes[0], 1, 1, shapes[-1]]), dtype=tf.float32)
            attention_score += (1.0 - mask) * -1e30
        attention_score = tf.nn.softmax(attention_score)  # shape = (batch_size, num_heads, seq_length, seq_length)
        attention_score = tf.nn.dropout(attention_score, rate=drop_rate)
        # compute value
        value = tf.matmul(attention_score, value)  # shape = (batch_size, num_heads, seq_length, head_size)
        value = tf.transpose(value, perm=[0, 2, 1, 3])
        value = tf.reshape(value, shape=[batch_size, seq_length, num_heads * head_size])
        return value


def multihead_attention_block(inputs, dim, num_heads, mask=None, use_bias=True, drop_rate=0.0, reuse=None,
                              name='multihead_attention_block'):
    with tf.variable_scope(name, reuse=reuse):
        # multihead attention layer
        outputs = layer_norm(inputs, reuse=reuse, name='layer_norm_1')
        outputs = tf.nn.dropout(outputs, rate=drop_rate)
        outputs = multihead_attention(outputs, dim=dim, num_heads=num_heads, mask=mask, drop_rate=drop_rate,
                                      name='multihead_attention')
        outputs = tf.nn.dropout(outputs, rate=drop_rate)
        residual = outputs + inputs
        # feed forward layer
        outputs = layer_norm(residual, reuse=reuse, name='layer_norm_2')
        outputs = tf.nn.dropout(outputs, rate=drop_rate)
        outputs = conv1d(outputs, dim=dim, use_bias=use_bias, activation=None, reuse=reuse, name='dense')
        outputs = tf.nn.dropout(outputs, rate=drop_rate)
        outputs = outputs + residual
        return outputs


def feature_encoder(inputs, hidden_size, num_heads, max_position_length, drop_rate, mask, reuse=None,
                    name='feature_encoder'):
    with tf.variable_scope(name, reuse=reuse):
        features = add_positional_embedding(inputs, max_position_length=max_position_length, reuse=reuse,
                                            name='positional_embedding')
        features = conv_block(features, kernel_size=7, dim=hidden_size, num_layers=4, reuse=reuse, drop_rate=drop_rate,
                              name='conv_block')
        features = multihead_attention_block(features, dim=hidden_size, num_heads=num_heads, mask=mask, use_bias=True,
                                             drop_rate=drop_rate, reuse=False, name='multihead_attention_block')
        return features


def video_query_attention(video_features, query_features, v_mask, q_mask, drop_rate=0.0, reuse=None,
                          name='video_query_attention'):
    with tf.variable_scope(name, reuse=reuse):
        dim = get_shape_list(video_features)[-1]
        v_maxlen = tf.reduce_max(tf.reduce_sum(v_mask, axis=1))
        q_maxlen = tf.reduce_max(tf.reduce_sum(q_mask, axis=1))
        score = trilinear_attention([video_features, query_features], v_maxlen=v_maxlen, q_maxlen=q_maxlen,
                                    drop_rate=drop_rate, reuse=reuse, name='efficient_trilinear')
        mask_q = tf.expand_dims(q_mask, 1)
        score_ = tf.nn.softmax(mask_logits(score, mask=mask_q))
        mask_v = tf.expand_dims(v_mask, 2)
        score_t = tf.transpose(tf.nn.softmax(mask_logits(score, mask=mask_v), dim=1), perm=[0, 2, 1])
        v2q = tf.matmul(score_, query_features)
        q2v = tf.matmul(tf.matmul(score_, score_t), video_features)
        attention_outputs = tf.concat([video_features, v2q, video_features * v2q, video_features * q2v], axis=-1)
        outputs = conv1d(attention_outputs, dim=dim, use_bias=False, activation=None, reuse=reuse, name='dense')
        return outputs, score


def context_query_concat(inputs, qfeats, q_mask, reuse=None, name='context_query_concat'):
    with tf.variable_scope(name, reuse=reuse):
        dim = get_shape_list(qfeats)[-1]
        # compute pooled query feature
        weight = tf.get_variable(name='weight', shape=[dim, 1], dtype=tf.float32, regularizer=regularizer)
        x = tf.tensordot(qfeats, weight, axes=1)  # shape = (batch_size, seq_length, 1)
        q_mask = tf.expand_dims(q_mask, axis=-1)  # shape = (batch_size, seq_length, 1)
        x = mask_logits(x, mask=q_mask)
        alphas = tf.nn.softmax(x, axis=1)
        q_pooled = tf.matmul(tf.transpose(qfeats, perm=[0, 2, 1]), alphas)
        q_pooled = tf.squeeze(q_pooled, axis=-1)  # shape = (batch_size, dim)
        # concatenation
        q_pooled = tf.tile(tf.expand_dims(q_pooled, axis=1), multiples=[1, tf.shape(inputs)[1], 1])
        outputs = tf.concat([inputs, q_pooled], axis=-1)
        outputs = conv1d(outputs, dim=dim, use_bias=True, reuse=False, name='dense')
        return outputs


def highlight_layer(inputs, labels, mask, epsilon=1e-12, reuse=None, name='highlight_layer'):
    with tf.variable_scope(name, reuse=reuse):
        logits = conv1d(inputs, dim=1, use_bias=True, padding='VALID', reuse=reuse, name='dense')
        logits = tf.squeeze(logits, axis=-1)  # (batch_size, seq_length)
        logits = mask_logits(logits, mask=mask)
        # prepare labels and weights
        labels = tf.cast(labels, dtype=logits.dtype)
        weights = tf.where(tf.equal(labels, 0.0), x=labels + 1.0, y=labels * 2.0)
        # binary cross entropy with sigmoid activation
        loss_per_location = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss_per_location = loss_per_location * weights
        mask = tf.cast(mask, dtype=logits.dtype)
        loss = tf.reduce_sum(loss_per_location * mask) / (tf.reduce_sum(mask) + epsilon)
        # compute scores
        scores = tf.sigmoid(logits)
        return loss, scores


def dynamic_rnn(inputs, seq_len, dim, reuse=None, name='dynamic_rnn'):
    with tf.variable_scope(name, reuse=reuse):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=dim, use_peepholes=False, name='lstm_cell')
        outputs, _ = tf.nn.dynamic_rnn(cell, inputs, sequence_length=seq_len, dtype=tf.float32)
        return outputs


def conditioned_predictor(inputs, hidden_size, seq_len, mask, num_heads, max_position_length, drop_rate, mode='rnn',
                          reuse=None, name='conditioned_predictor'):
    with tf.variable_scope(name, reuse=reuse):
        if mode == 'rnn':
            start_features = dynamic_rnn(inputs, seq_len, dim=hidden_size, reuse=False, name='start_rnn')
            end_features = dynamic_rnn(start_features, seq_len, dim=hidden_size, reuse=False, name='end_rnn')
        else:
            start_features = feature_encoder(inputs, hidden_size=hidden_size, num_heads=num_heads, mask=mask,
                                             max_position_length=max_position_length, drop_rate=drop_rate, reuse=False,
                                             name='feature_encoder')
            end_features = feature_encoder(start_features, hidden_size=hidden_size, num_heads=num_heads, mask=mask,
                                           max_position_length=max_position_length, drop_rate=drop_rate, reuse=True,
                                           name='feature_encoder')
            start_features = layer_norm(start_features, reuse=False, name='s_layer_norm')
            end_features = layer_norm(end_features, reuse=False, name='e_layer_norm')
        start_features = conv1d(tf.concat([start_features, inputs], axis=-1), dim=hidden_size, use_bias=True,
                                reuse=False, activation=tf.nn.relu, name='start_hidden')
        end_features = conv1d(tf.concat([end_features, inputs], axis=-1), dim=hidden_size, use_bias=True, reuse=False,
                              activation=tf.nn.relu, name='end_hidden')
        start_logits = conv1d(start_features, dim=1, use_bias=True, reuse=reuse, name='start_dense')
        end_logits = conv1d(end_features, dim=1, use_bias=True, reuse=reuse, name='end_dense')
        start_logits = mask_logits(tf.squeeze(start_logits, axis=-1), mask=mask)  # shape = (batch_size, seq_length)
        end_logits = mask_logits(tf.squeeze(end_logits, axis=-1), mask=mask)  # shape = (batch_size, seq_length)
        return start_logits, end_logits


def localization_loss(start_logits, end_logits, y1, y2):
    start_prob = tf.nn.softmax(start_logits, axis=1)
    end_prob = tf.nn.softmax(end_logits, axis=1)
    outer = tf.matmul(tf.expand_dims(start_prob, axis=2), tf.expand_dims(end_prob, axis=1))
    outer = tf.matrix_band_part(outer, num_lower=0, num_upper=-1)
    start_index = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
    end_index = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
    start_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=start_logits, labels=y1)
    end_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=end_logits, labels=y2)
    loss = tf.reduce_mean(start_losses + end_losses)
    return start_prob, end_prob, start_index, end_index, loss
