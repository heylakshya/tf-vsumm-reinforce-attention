# import torch
# import torch.nn as nn
# import torch.nn.functional as F

import tensorflow as tf
from config import  *
from layer_norm import  *
from tensorflow.python.keras.backend import dtype

class Net(tf.keras.Model):

    def __init__(self, apperture=-1, ignore_itself=False, max_seqlen=1024, output_size=1024, features = 1024):
        super(Net, self).__init__()

        self.apperture = apperture
        self.ignore_itself = ignore_itself

        self.output_size = output_size
        self.m = 1024 # cnn features size
        self.hidden_size = 1024
        self.max_seqlen = max_seqlen


        self.K = tf.keras.layers.Dense(units=self.output_size, use_bias=False)
        self.Q = tf.keras.layers.Dense(units=self.output_size, use_bias=False)
        self.V = tf.keras.layers.Dense(units=self.output_size, use_bias=False)
        self.output_linear = tf.keras.layers.Dense(units=self.output_size, use_bias=False)
        self.softmax = tf.keras.layers.Softmax(axis=0)
        self.drop50 = tf.keras.layers.Dropout(0.5)

        self.ka = tf.keras.layers.Dense(units=self.output_size)
        # self.kb = tf.keras.layers.Dense(input_shape=(self.ka.out_features,), units=1024)
        # self.kc = tf.keras.layers.Dense(input_shape=(self.kb.out_features,), units=1024)
        self.kd = tf.keras.layers.Dense(units=1)

        self.sig = tf.keras.layers.Activation("sigmoid")
        self.relu = tf.keras.layers.ReLU()
        self.drop50 = tf.keras.layers.Dropout(0.5)
        # self.softmax = tf.keras.layers.Softmax()
        self.layer_norm_y = tf.keras.layers.LayerNormalization()
        self.layer_norm_ka = tf.keras.layers.LayerNormalization()

    def call(self, x):
        m = x.shape[2] # Feature size

        # Place the video frames to the batch dimension to allow for batch arithm. operations.
        # Assumes input batch size = 1.
        x = tf.reshape(x, (-1, m))
        seqlen = x.shape[0]

        if seqlen > self.max_seqlen:
            x = x[:self.max_seqlen][...]
        else:
            pad_len = self.max_seqlen - seqlen
            padding = tf.constant([[0, pad_len], [0, 0]])
            x = tf.pad(x, padding)
        
        n = self.max_seqlen
        
        K = self.K(x)  # ENC (N x D) => (N x H) H= hidden size
        Q = self.Q(x)  # ENC (N x D) => (N x H) H= hidden size
        V = self.V(x)

        Q *= 0.06
        # logits = torch.matmul(Q, K.transpose(1,0))
        logits = tf.linalg.matmul(Q, K, False, True) #(N X N)

        if self.ignore_itself:
            # Zero the diagonal activations (a distance of each frame with itself)
            logits = tf.linalg.set_diag(logits, tf.zeros(n))

        if self.apperture > 0:
            # Set attention to zero to frames further than +/- apperture from the current one
            onesmask = tf.ones((n, n), dtype=tf.bool)
            diagonals = tf.zeros((n, n), dtype=tf.bool)
            trimask = tf.linalg.set_diag(onesmask, diagonals, k=(-self.apperture, self.apperture))
            trimask = trimask.numpy()
            logits = logits.numpy()
            logits[trimask] = -tf.float("Inf")
            logits = tf.convert_to_tensor(logits)

        att_weights_ = self.softmax(logits)
        weights = self.drop50(att_weights_)
        y = tf.transpose(tf.linalg.matmul(V, weights, True, False))
        y = self.output_linear(y)

        # print("y shape:{}\nx shape:{}".format(y.shape, x.shape))

        y = y + x
        y = self.drop50(y)
        y = self.layer_norm_y(y)

        # Frame level importance score regression
        # Two layer NN
        y = self.ka(y)
        y = self.relu(y)
        y = self.drop50(y)
        y = self.layer_norm_ka(y)

        y = self.kd(y)
        y = self.sig(y)
        y = tf.reshape(y, (1, -1))
        y = y[:, :seqlen]

        return y
