from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class LayerNorm(tf.keras.Model):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = tf.ones(features)
        self.beta = tf.zeros(features)
        self.eps = eps

    def forward(self, x):
        mean = tf.math.reduce_mean(x, -1, True)
        std = tf.math.reduce_std(x, -1, True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta