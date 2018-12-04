from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import pickle as pkl
from sklearn.manifold import TSNE
from PIL import Image
from flip_gradient import flip_gradient
from Utils import *
from dann import MNISTModel
from tensorflow.examples.tutorials.mnist import input_data


def preprocess(img_path, save_path, save_image=False):
    glaucoma, glaucoma_ = [], []
    with open(img_path, 'r') as f:
        for line in f.readlines():
            [image, label] = line.strip('\n').split(' ')
            # print(image, label)
            name = image.split('/')[-1]
            img = Image.open(image)

            glaucoma.append(np.array(img))
            glaucoma_.append(np.array(img.transpose(Image.ROTATE_90)))

            if save_image:
                print('processing {}'.format(image))
                img.transpose(Image.ROTATE_90).save('{}_{}/{}'.format(save_path, 90, name))
                img.transpose(Image.ROTATE_180).save('{}_{}/{}'.format(save_path, 180, name))
                img.transpose(Image.ROTATE_270).save('{}_{}/{}'.format(save_path, 270, name))

    return glaucoma, glaucoma_


if __name__ == '__main__':
    glaucoma, glaucoma_ = preprocess('test.txt', './glaucoma/Rotate')

    # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Process MNIST
    # mnist_train = (mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.uint8) * 255
    # mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
    # mnist_test = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
    # mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
    #
    # # Load MNIST-M
    # mnistm = pkl.load(open('mnistm_data.pkl', 'rb'))
    # mnistm_train = mnistm['train']
    # mnistm_test = mnistm['test']
    # mnistm_valid = mnistm['valid']

    # Compute pixel mean for normalizing data
    # pixel_mean = np.vstack([mnist_train, mnistm_train]).mean((0, 1, 2))

    # Create a mixed dataset for TSNE visualization

    num_test = 80
    combined_test_imgs = np.vstack([glaucoma[:num_test], glaucoma_[:num_test]])
    # combined_test_labels = np.vstack([mnist.test.labels[:num_test], mnist.test.labels[:num_test]])
    combined_test_domain = np.vstack([np.tile([1., 0.], [num_test, 1]),
                                      np.tile([0., 1.], [num_test, 1])])

    imshow_grid(glaucoma)
    imshow_grid(glaucoma_)

    # Build the model graph
    graph = tf.get_default_graph()
    with graph.as_default():
        model = MNISTModel()

        learning_rate = tf.placeholder(tf.float32, [])

        pred_loss = tf.reduce_mean(model.pred_loss)
        domain_loss = tf.reduce_mean(model.domain_loss)
        total_loss = pred_loss + domain_loss

        regular_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)
        dann_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)

        # Evaluation
        correct_label_pred = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred, 1))
        label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
        correct_domain_pred = tf.equal(tf.argmax(model.domain, 1), tf.argmax(model.domain_pred, 1))
        domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))


    def train_and_evaluate(training_mode, graph, model, num_steps=8600, batch_size=64, verbose=False):
        """Helper to run the model with different training modes."""

        with tf.Session(graph=graph) as sess:
            tf.global_variables_initializer().run()

            # Batch generators
            gen_source_batch = batch_generator(
                [mnist_train, mnist.train.labels], batch_size // 2)
            gen_target_batch = batch_generator(
                [mnistm_train, mnist.train.labels], batch_size // 2)
            gen_source_only_batch = batch_generator(
                [mnist_train, mnist.train.labels], batch_size)
            gen_target_only_batch = batch_generator(
                [mnistm_train, mnist.train.labels], batch_size)

            domain_labels = np.vstack([np.tile([1., 0.], [batch_size // 2, 1]),
                                       np.tile([0., 1.], [batch_size // 2, 1])])

            # Training loop
            for i in range(num_steps):

                # Adaptation param and learning rate schedule as described in the paper
                p = float(i) / num_steps
                l = 2. / (1. + np.exp(-10. * p)) - 1
                lr = 0.01 / (1. + 10 * p) ** 0.75

                # Training step
                if training_mode == 'dann':

                    X0, y0 = next(gen_source_batch)
                    X1, y1 = next(gen_target_batch)
                    X = np.vstack([X0, X1])
                    y = np.vstack([y0, y1])

                    _, batch_loss, dloss, ploss, d_acc, p_acc = sess.run(
                        [dann_train_op, total_loss, domain_loss, pred_loss, domain_acc, label_acc],
                        feed_dict={model.X: X, model.y: y, model.domain: domain_labels,
                                   model.train: True, model.l: l, learning_rate: lr})

                    if verbose and i % 100 == 0:
                        print('loss: {}  d_acc: {}  p_acc: {}  p: {}  l: {}  lr: {}'.format(
                            batch_loss, d_acc, p_acc, p, l, lr))

                elif training_mode == 'source':
                    X, y = next(gen_source_only_batch)
                    _, batch_loss = sess.run([regular_train_op, pred_loss],
                                             feed_dict={model.X: X, model.y: y, model.train: False,
                                                        model.l: l, learning_rate: lr})

                elif training_mode == 'target':
                    X, y = next(gen_target_only_batch)
                    _, batch_loss = sess.run([regular_train_op, pred_loss],
                                             feed_dict={model.X: X, model.y: y, model.train: False,
                                                        model.l: l, learning_rate: lr})

            # Compute final evaluation on test data
            source_acc = sess.run(label_acc,
                                  feed_dict={model.X: mnist_test, model.y: mnist.test.labels,
                                             model.train: False})

            target_acc = sess.run(label_acc,
                                  feed_dict={model.X: mnistm_test, model.y: mnist.test.labels,
                                             model.train: False})

            test_domain_acc = sess.run(domain_acc,
                                       feed_dict={model.X: combined_test_imgs,
                                                  model.domain: combined_test_domain, model.l: 1.0})

            test_emb = sess.run(model.feature, feed_dict={model.X: combined_test_imgs})

        return source_acc, target_acc, test_domain_acc, test_emb


    print('\nSource only training')
    source_acc, target_acc, _, source_only_emb = train_and_evaluate('source', graph, model)
    print('Source (MNIST) accuracy:', source_acc)
    print('Target (MNIST-M) accuracy:', target_acc)

    print('\nDomain adaptation training')
    source_acc, target_acc, d_acc, dann_emb = train_and_evaluate('dann', graph, model)
    # print('Source (MNIST) accuracy:', source_acc)
    # print('Target (MNIST-M) accuracy:', target_acc)
    # print('Domain accuracy:', d_acc)

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    source_only_tsne = tsne.fit_transform(source_only_emb)

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    dann_tsne = tsne.fit_transform(dann_emb)

    plot_embedding(source_only_tsne, combined_test_labels.argmax(1), combined_test_domain.argmax(1), 'Source only')
    plot_embedding(dann_tsne, combined_test_labels.argmax(1), combined_test_domain.argmax(1), 'Domain Adaptation')
