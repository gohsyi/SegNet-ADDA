import os
import numpy as np
import tensorflow as tf

from PIL import Image
from model.segnet import SegNet
from util.utils import get_hist
from util.loss_functions import adv_loss, adv_loss_v2


class ADDA(SegNet):

    def __init__(self, args):
        SegNet.__init__(self, args)

        self.src_image_path = 'train.txt'
        self.tar_image_path = 'test.txt'
        self.ckpt_dir = args.transfer
        self.src_ckpt_path = self.ckpt_dir + 'src_model.ckpt'
        self.tar_ckpt_path = self.ckpt_dir + 'tar_model.ckpt'
        self.scope_s = 's'  # source scope
        self.scope_t = 't'  # target scope
        self.scope_d = 'd'  # discriminator scope
        self.scope_g = 'g'  # generator scope
        self.lr_g = 1e-3
        self.lr_d = 1e-3
        self.save_image = args.save_image
        self.generate_src_tar_model()


    def generate_src_tar_model(self):
        vars = []
        with tf.Session(config=self.sess_config) as sess:
            tar_vars = []
            src_vars = []
            for name, shape in tf.contrib.framework.list_variables(self.ckpt_dir):
                var = tf.contrib.framework.load_variable(self.ckpt_dir, name)
                vars.append(tf.Variable(var, name=name))
                tar_vars.append(tf.Variable(var, name='t/' + name))
                src_vars.append(tf.Variable(var, name='s/' + name))

            src_saver = tf.train.Saver(src_vars)
            tar_saver = tf.train.Saver(tar_vars)
            tmp_saver = tf.train.Saver(vars)

            sess.run(tf.global_variables_initializer())

            src_saver.save(sess, self.src_ckpt_path)
            print('model saved to {}'.format(self.src_ckpt_path))
            tar_saver.save(sess, self.tar_ckpt_path)
            print('model saved to {}'.format(self.tar_ckpt_path))
            tmp_saver.save(sess, self.ckpt_dir + 'model.ckpt')  # rewrite the checkpoint

        tf.reset_default_graph()  # to avoid variables naming violence


    def discriminator(self, inputs, trainable=True):
        flat = tf.layers.Flatten()(inputs)

        fc1 = tf.layers.dense(flat, 576, activation=tf.nn.leaky_relu, trainable=trainable, name='fc1')
        fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.leaky_relu, trainable=trainable, name='fc2')
        fc3 = tf.layers.dense(fc2, 1, activation=None, trainable=trainable, name='fc3')

        return fc3


    def train(self):
        image_h = self.image_h
        image_w = self.image_w
        image_c = self.image_c
        batch_size = self.batch_size

        src_generator = self.dataset.batch(batch_size=batch_size, path=self.src_image_path)
        tar_generator = self.dataset.batch(batch_size=batch_size, path=self.tar_image_path)

        # src_x_test, src_y_test = self.dataset.batch(batch_size=1, path=self.src_image_path)
        # tar_x_test, tar_y_test = self.dataset.batch(batch_size=1, path=self.tar_image_path)

        src_images = tf.placeholder(tf.float32, shape=[None, image_h, image_w, image_c], name='src_images')
        src_labels = tf.placeholder(tf.int64, shape=[None, image_h, image_w, 1], name='src_labels')

        tar_images = tf.placeholder(tf.float32, shape=[None, image_h, image_w, image_c], name='tar_images')
        tar_labels = tf.placeholder(tf.int64, shape=[None, image_h, image_w, 1], name='tar_labels')

        phase_train = tf.placeholder(tf.bool, name='phase_train')

        # for source domain
        with tf.variable_scope(self.scope_s):
            src_encode_output = self.encoder(src_images, tf.constant(False))
            src_decode_output = self.decoder(src_encode_output, batch_size, tf.constant(False))
            src_logits, src_cls_loss = self.classifier(src_decode_output, src_labels, self.loss_func)
        with tf.variable_scope(self.scope_d):
            dis_src = self.discriminator(src_encode_output)
            # dis_src = tf.Print(dis_src, [dis_src], summarize=batch_size, message="D_s: ")

        # for target domain
        with tf.variable_scope(self.scope_t):
            tar_encode_output = self.encoder(tar_images, phase_train=phase_train)
            tar_decode_output = self.decoder(tar_encode_output, batch_size, tf.constant(False))
            tar_logits, tar_cls_loss = self.classifier(tar_decode_output, tar_labels, self.loss_func)
        with tf.variable_scope(self.scope_d, reuse=True):
            dis_tar = self.discriminator(tar_encode_output)
            # dis_tar = tf.Print(dis_tar, [dis_tar], summarize=batch_size, message="D_t: ")

        # build loss
        g_loss, d_loss = adv_loss(dis_src, dis_tar)

        # create optimizer for two task
        var_tar = tf.trainable_variables(self.scope_t)
        optim_g = tf.train.AdamOptimizer(self.lr_g).minimize(g_loss, var_list=var_tar)

        var_d = tf.trainable_variables(self.scope_d)
        optim_d = tf.train.AdamOptimizer(self.lr_d).minimize(d_loss, var_list=var_d)

        with tf.Session(config=self.sess_config) as sess:
            # print(tf.trainable_variables())
            sess.run(tf.global_variables_initializer())

            self.src_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_s)
            self.tar_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_t)

            src_saver = tf.train.Saver(self.src_vars)
            tar_saver = tf.train.Saver(self.tar_vars)

            src_saver.restore(sess, self.src_ckpt_path)
            print("src model restored successfully!")
            tar_saver.restore(sess, self.tar_ckpt_path)
            print("tar model restored succesfully!")

            for i in range(self.max_steps):
                src_data = sess.run(src_generator)
                tar_data = sess.run(tar_generator)

                _, d_loss_, = sess.run([optim_d, d_loss], feed_dict={
                    src_images: src_data[0],
                    tar_images: tar_data[0],
                    phase_train: True
                })
                _, g_loss_ = sess.run([optim_g, g_loss], feed_dict={
                    src_images: src_data[0],
                    tar_images: tar_data[0],
                    phase_train: True
                })

                if i % 100 == 0:
                    self.output.write("step:{}, g_loss:{:.4f}, d_loss:{:.4f}".format(i, g_loss_, d_loss_))
                if i % 100 == 0:
                    print("testing ...")  # TODO finish testing
                    pred = tf.argmax(tar_logits, axis=3)
                    hist = np.zeros((self.num_classes, self.num_classes))
                    image_filenames, label_filenames = self.dataset.get_filename_list(self.test_path)
                    images, labels = self.dataset.get_all_test_data(image_filenames, label_filenames)
                    images = [image[0] for image in images]
                    labels = [label[0] for label in labels]
                    images = [images[i: i+batch_size] for i in range(0, len(images), batch_size)]
                    labels = [labels[i: i+batch_size] for i in range(0, len(labels), batch_size)]
                    names = [image_filenames[i: i+batch_size] for i in range(0, len(image_filenames), batch_size)]

                    for image_batch, label_batch, name_batch in zip(images, labels, names):
                        logits, pred_image = sess.run([tar_logits, pred], feed_dict={
                            tar_images: image_batch,
                            phase_train: False
                        })
                        if self.save_image:
                            for image, name in zip(pred_image, name_batch):
                                image[image == 0] = 0
                                image[image == 1] = 128
                                image[image == 2] = 255
                                image = Image.fromarray(np.uint8(image))
                                save_path = '{}{}_{}'.format(self.log_dir, i, name.split('/')[-1])
                                # image = image.resize((self.image_w_origin, self.image_h_origin))
                                image.save(save_path)
                                # print("image saved to {}".format(save_path))
                        
                        hist += get_hist(logits, label_batch)

                    acc_total = np.diag(hist).sum() / hist.sum()
                    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
                    self.output.write('acc: {}'.format(acc_total))
                    self.output.write('mean IU: {}'.format(np.nanmean(iu)))

