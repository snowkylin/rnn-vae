# coding=utf8

import tensorflow as tf
import numpy as np
import input_data
import model as m
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='RNN',
                    help='CNN or RNN')
parser.add_argument('--mode', type=str, default='x with x_hat and h_dec_prev',
                    help='(Only for RNN) only c_prev, only x_hat, x with x_hat, x with x_hat and h_dec_prev')
parser.add_argument('--x_size', type=int, default=784,
                    help='size of x')
parser.add_argument('--z_size', type=int, default=50,
                    help='size of latent variable')
parser.add_argument('--enc_size', type=int, default=256,
                    help='size of encoder latent variable (only for rnn vae)')
parser.add_argument('--dec_size', type=int, default=256,
                    help='size of decoder latent variable (only for rnn vae)')
parser.add_argument('--T', type=int, default=10,
                    help='size of time T (only for rnn vae)')
parser.add_argument('--batch_size', type=int, default=100,
                    help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--alpha', type=float, default=0.95,
                    help='ratio between re_loss and kl_loss '
                         '(args.alpha * self.re_loss + (1 - args.alpha) * self.kl_loss)')
parser.add_argument('--num_epochs', type=int, default=20,
                     help='number of epochs')
parser.add_argument('--num_batches', type=int, default=1000,
                     help='number of batches')
parser.add_argument('--width', type=int, default=28,
                     help='image width')
parser.add_argument('--height', type=int, default=28,
                     help='image height')
args=parser.parse_args()

if args.type == 'RNN':
    model = m.RNN_VAE_Model(args=args)
else:
    model = m.CNN_VAE_Model(args=args)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for e in range(args.num_epochs):
        print "epoch %d" % e
        for b in range(args.num_batches):
            batch = mnist.train.next_batch(args.batch_size)
            x = np.reshape(batch[0], [args.batch_size, args.width, args.height, 1])
            [re_loss, kl_loss, loss] = sess.run([model.re_loss, model.kl_loss, model.loss],
                                                feed_dict={model.x: x})
            sess.run(model.optimizer, feed_dict={model.x: x})
            if b % 100 == 0:
                print 'batches %d, loss %g (re: %g, kl: %g)' % (b, loss, re_loss, kl_loss)
    batch = mnist.train.next_batch(args.batch_size)
    x = np.reshape(batch[0], [args.batch_size, args.width, args.height, 1])
    if args.type == 'CNN':
        decoded = sess.run(model.decoded, feed_dict={model.x: x})
        n = args.batch_size
        for i in range(n):
            axes = plt.subplot(2, n, i + 1)
            axes.set_xticks([])
            axes.set_yticks([])
            plt.imshow(x[i].reshape(args.width, args.height), cmap="gray", interpolation='nearest')

            axes = plt.subplot(2, n, n + i + 1)
            axes.set_xticks([])
            axes.set_yticks([])
            plt.imshow(decoded[i].reshape(args.width, args.height), cmap="gray", interpolation='nearest')
        plt.show()
    if args.type == 'RNN':
        c = sess.run(model.c, feed_dict={model.x: x})
        n = 10
        for i in range(n):
            axes = plt.subplot(args.T + 1, n, i + 1)
            axes.set_xticks([])
            axes.set_yticks([])
            plt.imshow(x[i].reshape(args.width, args.height), cmap="gray", interpolation='nearest')
            for t in range(args.T):
                axes = plt.subplot(args.T + 1, n, (t + 1) * n + i + 1)
                axes.set_xticks([])
                axes.set_yticks([])
                plt.imshow(c[t][i].reshape(args.width, args.height), cmap="gray", interpolation='nearest')
        plt.show()