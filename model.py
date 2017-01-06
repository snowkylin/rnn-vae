# coding=utf8

import tensorflow as tf
import numpy as np

class RNN_VAE_Model():
    def __init__(self, args):
        self.args = args
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, args.width, args.height, 1])
        x_reshape = tf.reshape(self.x, [-1, args.width * args.height])
        cell_enc = tf.nn.rnn_cell.LSTMCell(args.enc_size)
        cell_dec = tf.nn.rnn_cell.LSTMCell(args.dec_size)
        enc_state = cell_enc.zero_state(args.batch_size, tf.float32)
        dec_state = cell_dec.zero_state(args.batch_size, tf.float32)
        self.e = tf.random_normal((args.batch_size, args.z_size))
        self.c, mu, logsigma, sigma, z = [0] * args.T, [0] * args.T, [0] * args.T, \
                                    [0] * args.T, [0] * args.T
        w_mu = weight_variable([args.enc_size, args.z_size])
        b_mu = bias_variable([args.z_size])
        w_sigma = weight_variable([args.enc_size, args.z_size])
        b_sigma = bias_variable([args.z_size])
        w_h_dec = weight_variable([args.dec_size, args.x_size])
        b_h_dec = bias_variable([args.x_size])
        DO_SHARE = False
        for t in range(args.T):
            if t == 0:
                c_prev = tf.zeros((args.batch_size, args.width * args.height))
                h_dec_prev = tf.zeros((args.batch_size, args.dec_size))
            else:
                c_prev = self.c[t - 1]
            c_sigmoid = tf.sigmoid(c_prev)
            x_hat = x_reshape - c_sigmoid
            with tf.variable_scope("encoder", reuse=DO_SHARE):
                if args.mode == 'only c_prev':
                    h_enc, enc_state = cell_enc(c_prev, enc_state)
                if args.mode == 'only c_sigmoid':
                    h_enc, enc_state = cell_enc(c_sigmoid, enc_state)
                if args.mode == 'only x_hat':
                    h_enc, enc_state = cell_enc(x_hat, enc_state)
                if args.mode == 'x with x_hat':
                    h_enc, enc_state = cell_enc(tf.concat(1, [x_reshape, x_hat]), enc_state)
                if args.mode == 'x with x_hat and h_dec_prev':
                    h_enc, enc_state = cell_enc(tf.concat(1, [x_reshape, x_hat, h_dec_prev]), enc_state)

            mu[t] = tf.matmul(h_enc, w_mu) + b_mu  # [z_size]
            logsigma[t] = tf.matmul(h_enc, w_sigma) + b_sigma  # [z_size]
            sigma[t] = tf.exp(logsigma[t])

            z[t] = mu[t] + sigma[t] * self.e
            with tf.variable_scope("decoder", reuse=DO_SHARE):
                h_dec, dec_state = cell_dec(z[t], dec_state)
            h_dec_prev = h_dec
            self.c[t] = tf.matmul(h_dec, w_h_dec) + b_h_dec
            DO_SHARE = True

        self.decoded = tf.nn.sigmoid(self.c[args.T - 1])
        self.re_loss = tf.reduce_mean(-tf.reduce_sum(cross_entropy(x_reshape, self.decoded), 1))
        kl_loss = [0] * args.T
        for t in range(args.T):
            kl_loss[t] = 0.5 * (tf.reduce_sum(sigma[t], 1) + tf.reduce_sum(tf.square(mu[t]), 1)
                         - tf.reduce_sum(logsigma[t] + 1, 1))
        self.kl_loss = tf.reduce_mean(tf.add_n(kl_loss))

        self.loss = args.alpha * self.re_loss + (1 - args.alpha) * self.kl_loss
        # self.loss = self.re_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate) \
            .minimize(self.loss)

class CNN_VAE_Model():
    def __init__(self, args):
        self.args = args
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, args.width, args.height, 1]) # [28 * 28]

        h_conv1 = tf.nn.relu(conv2d(self.x, weight_variable([5, 5, 1, 16]))
                             + bias_variable([16]))  # [28 * 28 * 32]
        h_pool1 = max_pool_2x2(h_conv1)  # [14 * 14 * 32]

        h_conv2 = tf.nn.relu(conv2d(h_pool1, weight_variable([5, 5, 16, 8]))
                             + bias_variable([8]))  # [14 * 14 * 64]
        h_pool2 = max_pool_2x2(h_conv2)  # [7 * 7 * 64]

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 8])  # [3136 (7*7*64)]
        # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weight_variable([7 * 7 * 64, 1024]))
        #                    + bias_variable([1024]))  # [1024]

        # self.keep_prob = tf.placeholder(dtype=tf.float32)
        # # 训练时启用dropout减少过拟合，测试时关掉
        # h_fc1_drop = tf.nn.dropout(h_pool2_flat, self.keep_prob)

        mu = tf.matmul(h_pool2_flat, weight_variable([7 * 7 * 8, args.z_size])) \
                + bias_variable([args.z_size])  # [z_size]
        logsigma = tf.matmul(h_pool2_flat, weight_variable([7 * 7 * 8, args.z_size])) \
                            + bias_variable([args.z_size])  # [z_size]
        sigma = tf.exp(logsigma)

        e = tf.random_normal((args.batch_size, args.z_size))
        z = mu + sigma * e

        h_fc1 = tf.matmul(z, weight_variable([args.z_size, 7 * 7 * 8])
                          + bias_variable([7 * 7 * 8]))
        h_fc2_2dim = tf.reshape(h_fc1, [-1, 7, 7, 8])

        h_conv3 = tf.nn.relu(conv2d(h_fc2_2dim, weight_variable([5, 5, 8, 8]))
        # h_conv3 = tf.nn.relu(conv2d(h_pool2, weight_variable([5, 5, 64, 32]))
                             + bias_variable([8]))     # [7 * 7 * 64]
        h_pool3 = tf.image.resize_images(h_conv3, [14, 14])   # [14 * 14 * 64]

        h_conv4 = tf.nn.relu(conv2d(h_pool3, weight_variable([5, 5, 8, 16]))
                             + bias_variable([16]))  # [14 * 14 * 32]
        h_pool4 = tf.image.resize_images(h_conv4, [28, 28])  # [28 * 28 * 32]

        self.decoded = tf.nn.sigmoid(conv2d(h_pool4, weight_variable([5, 5, 16, 1]))
                             + bias_variable([1]))  # [28 * 28 * 1]

        self.re_loss = tf.reduce_mean(-tf.reduce_sum(cross_entropy(self.x, self.decoded), 1))
        self.kl_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(sigma, 1) + tf.reduce_sum(tf.square(mu), 1)
                              - tf.reduce_sum(logsigma + 1, 1))

        self.loss = args.alpha * self.re_loss + (1 - args.alpha) * self.kl_loss
        # self.loss = self.re_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)\
            .minimize(self.loss)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def cross_entropy(a, b):
    eps = 1e-8
    return a * tf.log(b + eps) + (1. - a) * tf.log(1. - b + eps)







