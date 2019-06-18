import tensorflow as tf
import numpy as np
from math import ceil


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.

      Generates moving average for all losses and associated summaries for
      visualizing the performance of the network.

      Args:
        total_loss: Total loss from loss().
      Returns:
        loss_averages_op: op for generating moving averages of losses.
      """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

      Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable

      Returns:
        Variable Tensor
      """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, initializer, wd):
    """Helper to create an initialized Variable with weighted decay.

      Note that the Variable is initialized with a truncated normal distribution.
      A weighted decay is added only if one is specified.

      Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weighted decay multiplied by this float. If None, weighted
            decay is not added for this Variable.

      Returns:
        Variable Tensor
      """
    var = _variable_on_cpu(
        name,
        shape,
        initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def fast_hist(a, b, n):
    """
    computer confusion matrix
    
    Parameters
    ----------
    a: vector of length image_h x image_w
        ground truth label vector
    b: vector of length image_h x image_w
        prediction vector
    
    Returns
    -------
    Confusion matrix
    """
    
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def get_hist(predictions, labels):
    """
    get histogram
    
    Parameters
    ----------
    predictions: array of size (batch_size, image_h, image_w, num_classes)
        predictions made by our model
    labels: array of size (batch_size, image_h, image_w, 1)
        ground truth labels
    
    Returns
    -------
    Confusion matrix of this batch
    """
    
    num_class = predictions.shape[3]
    batch_size = predictions.shape[0]
    hist = np.zeros((num_class, num_class))
    for i in range(batch_size):
        hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    return hist


def print_hist_summery(hist):
    acc_total = np.diag(hist).sum() / hist.sum()
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    for ii in range(hist.shape[0]):
        if float(hist.sum(1)[ii]) == 0:
            acc = 0.0
        else:
            acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
        print("    class # %d accuracy = %f " % (ii, acc))
    return np.nanmean(acc_total), np.nanmean(iu)  # use nanmean to ignore classes that didn't appear


def per_class_acc(output, predictions, label_tensor):
    labels = label_tensor
    size = predictions.shape[0]
    num_class = predictions.shape[3]
    hist = np.zeros((num_class, num_class))
    for i in range(size):
        hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)

    acc_total = np.diag(hist).sum() / hist.sum()
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    for ii in range(num_class):
        if float(hist.sum(1)[ii]) == 0:
            acc = 0.0
        else:
            acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
        print("    class # %d accuracy = %f " % (ii, acc))

    return np.nanmean(acc_total), np.nanmean(iu)


def get_deconv_filter(f_shape):
    """
        reference: https://github.com/MarvinTeichmann/tensorflow-fcn
    """
    width = f_shape[0]
    heigh = f_shape[0]
    f = ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init, shape=weights.shape)


def deconv_layer(inputT, f_shape, output_shape, stride=2, name=None):
    # output_shape = [b, w, h, c]
    # sess_temp = tf.InteractiveSession()
    sess_temp = tf.global_variables_initializer()
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        weights = get_deconv_filter(f_shape)
        deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape, strides=strides, padding='SAME')
    return deconv


def batch_norm_layer(inputT, is_training, scope):
    scope = scope.split('/')[-1]  # to fix scope bug
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                                                        center=False, updates_collections=None, scope=scope+"_bn"),
                   lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                                                        updates_collections=None, center=False, scope=scope+"_bn", reuse = True))


def orthogonal_initializer(scale = 1.1):
    """
    From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    """

    def _initializer(shape, dtype=tf.float32, partition_info=None):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)  # this needs to be corrected to float32
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)

    return _initializer


def conv_layer_with_bn(inputT, shape, train_phase, activation=True, name=None):
    in_channel = shape[2]
    out_channel = shape[3]
    k_size = shape[0]
    with tf.variable_scope(name) as scope:
        kernel = _variable_with_weight_decay('ort_weights', shape=shape, initializer=orthogonal_initializer(), wd=None)
        conv = tf.nn.conv2d(inputT, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        if activation is True:
            conv_out = tf.nn.relu(batch_norm_layer(bias, train_phase, scope.name))
        else:
            conv_out = batch_norm_layer(bias, train_phase, scope.name)
    return conv_out


encode_dict = {
    0: 0,
    8388608: 1,
    32768: 2,
    8421376: 3,
    128: 4,
    8388736: 5,
    32896: 6,
    8421504: 7,
    4194304: 8,
    12582912: 9,
    4227072: 10,
    12615680: 11,
    4194432: 12,
    12583040: 13,
    4227200: 14,
    12615808: 15,
    16384: 16,
    8404992: 17,
    49152: 18,
    8437760: 19,
    16512: 20,
}

color_dict = {
    0:[0, 0, 0], 
    1:[128, 0, 0], 
    2:[0, 128, 0], 
    3:[128, 128, 0], 
    4:[0, 0, 128], 
    5:[128, 0, 128],
    6:[0, 128, 128], 
    7:[128, 128, 128], 
    8:[64, 0, 0], 
    9:[192, 0, 0], 
    10:[64, 128, 0],
    11:[192, 128, 0], 
    12:[64, 0, 128], 
    13:[192, 0, 128], 
    14:[64, 128, 128], 
    15:[192, 128, 128],
    16:[0, 64, 0], 
    17:[128, 64, 0], 
    18:[0, 192, 0], 
    19:[128, 192, 0], 
    20:[0, 64, 128],
    
    14737600: 21,  # edge
}


def rgb2int(rgb):
    r, g, b = rgb
    return (r << 16) + (g << 8) + b


def encode(rgb):
    return encode_dict[rgb2int(rgb)]


def decode(c):
    return color_dict[c]
