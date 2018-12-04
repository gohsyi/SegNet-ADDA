import tensorflow as tf
import numpy as np
import pickle as pkl
from sklearn.manifold import TSNE

from flip_gradient import flip_gradient
from Utils import *

batch_size = 64


class MNISTModel(object):
    """Simple MNIST domain adaptation model."""

    def __init__(self):
        self._build_model()

    def _build_model(self):
        self.X = tf.placeholder(tf.uint8, [None, 28, 28, 3])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.domain = tf.placeholder(tf.float32, [None, 2])
        self.l = tf.placeholder(tf.float32, [])
        self.train = tf.placeholder(tf.bool, [])

        # X_input = (tf.cast(self.X, tf.float32) - pixel_mean) / 255.  # TODO
        X_input = (tf.cast(self.X, tf.float32)) / 255.

        # CNN model for feature extraction
        with tf.variable_scope('feature_extractor'):
            W_conv0 = weight_variable([5, 5, 3, 32])
            b_conv0 = bias_variable([32])
            h_conv0 = tf.nn.relu(conv2d(X_input, W_conv0) + b_conv0)
            h_pool0 = max_pool_2x2(h_conv0)

            W_conv1 = weight_variable([5, 5, 32, 48])
            b_conv1 = bias_variable([48])
            h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

            # The domain-invariant feature
            self.feature = tf.reshape(h_pool1, [-1, 7 * 7 * 48])

        # Small MLP for domain prediction with adversarial loss
        with tf.variable_scope('domain_predictor'):
            # Flip the gradient when backpropagating through this operation
            feat = flip_gradient(self.feature, self.l)

            d_W_fc0 = weight_variable([7 * 7 * 48, 100])
            d_b_fc0 = bias_variable([100])
            d_h_fc0 = tf.nn.relu(tf.matmul(feat, d_W_fc0) + d_b_fc0)

            d_W_fc1 = weight_variable([100, 2])
            d_b_fc1 = bias_variable([2])
            d_logits = tf.matmul(d_h_fc0, d_W_fc1) + d_b_fc1

            self.domain_pred = tf.nn.softmax(d_logits)
            self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=d_logits, labels=self.domain)
