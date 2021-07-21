import re
import numpy as np
import tensorflow as tf

if tf.__version__.startswith('2'):
    tf = tf.compat.v1
    tf.disable_v2_behavior()
    tf.disable_eager_execution()
    regularizer = tf.keras.regularizers.l2(l2=3e-7)
else:
    regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)


def count_params(scope=None):
    if scope is None:
        return int(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    else:
        return int(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables(scope)]))


def get_shape_list(tensor):
    shape = tensor.shape.as_list()
    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)
    if not non_static_indexes:
        return shape
    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = tf.cast(mask, tf.float32)
    return inputs * mask + mask_value * (1.0 - mask)


def trilinear_attention(args, v_maxlen, q_maxlen, drop_rate=0.0, reuse=None, name='efficient_trilinear'):
    assert len(args) == 2, 'just use for computing attention with two input'
    arg0_shape = args[0].get_shape().as_list()
    arg1_shape = args[1].get_shape().as_list()
    if len(arg0_shape) != 3 or len(arg1_shape) != 3:
        raise ValueError('`args` must be 3 dims (batch_size, len, dimension)')
    if arg0_shape[2] != arg1_shape[2]:
        raise ValueError('the last dimension of `args` must equal')
    arg_size = arg0_shape[2]
    dtype = args[0].dtype
    drop_args = [tf.nn.dropout(arg, rate=drop_rate) for arg in args]
    with tf.variable_scope(name, reuse=reuse):
        weights4arg0 = tf.get_variable('linear_kernel4arg0', [arg_size, 1], dtype=dtype, regularizer=regularizer)
        weights4arg1 = tf.get_variable('linear_kernel4arg1', [arg_size, 1], dtype=dtype, regularizer=regularizer)
        weights4mlu = tf.get_variable('linear_kernel4mul', [1, 1, arg_size], dtype=dtype, regularizer=regularizer)
        # compute results
        weights4arg0 = tf.tile(tf.expand_dims(weights4arg0, axis=0), multiples=[tf.shape(args[0])[0], 1, 1])
        subres0 = tf.tile(tf.matmul(drop_args[0], weights4arg0), [1, 1, q_maxlen])
        weights4arg1 = tf.tile(tf.expand_dims(weights4arg1, axis=0), multiples=[tf.shape(args[1])[0], 1, 1])
        subres1 = tf.tile(tf.transpose(tf.matmul(drop_args[1], weights4arg1), perm=(0, 2, 1)), [1, v_maxlen, 1])
        subres2 = tf.matmul(drop_args[0] * weights4mlu, tf.transpose(drop_args[1], perm=(0, 2, 1)))
        res = subres0 + subres1 + subres2
        return res


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, clip_norm=1.0):
    """Creates an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
    learning_rate = tf.train.polynomial_decay(learning_rate, global_step, num_train_steps, end_learning_rate=0.0,
                                              power=1.0, cycle=False)
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done
        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
    optimizer = AdamWeightDecayOptimizer(learning_rate=learning_rate, weight_decay_rate=0.01, beta_1=0.9, beta_2=0.999,
                                         epsilon=1e-6, exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias'])
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=clip_norm)
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
    # Normally the global step update is done inside of `apply_gradients`. However, `AdamWeightDecayOptimizer` doesn't
    # do this. But if you use a different optimizer, you should probably take this line out.
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op


class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self, learning_rate, weight_decay_rate=0.0, beta_1=0.9, beta_2=0.999, epsilon=1e-6,
                 exclude_from_weight_decay=None, name='AdamWeightDecayOptimizer'):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)
        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue
            param_name = self._get_variable_name(param.name)
            m = tf.get_variable(name=param_name + '/adam_m', shape=param.shape.as_list(), dtype=tf.float32,
                                trainable=False, initializer=tf.zeros_initializer())
            v = tf.get_variable(name=param_name + '/adam_v', shape=param.shape.as_list(), dtype=tf.float32,
                                trainable=False, initializer=tf.zeros_initializer())
            next_m = (tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2, tf.square(grad)))
            update = next_m / (tf.sqrt(next_v) + self.epsilon)
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param
            update_with_lr = self.learning_rate * update
            next_param = param - update_with_lr
            assignments.extend([param.assign(next_param), m.assign(next_m), v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    @staticmethod
    def _get_variable_name(param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name

    def _apply_dense(self, grad, var):
        pass

    def _resource_apply_dense(self, grad, handle):
        pass

    def _resource_apply_sparse(self, grad, handle, indices):
        pass

    def _apply_sparse(self, grad, var):
        pass
