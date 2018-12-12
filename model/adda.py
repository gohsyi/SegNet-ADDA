import numpy as np
import tensorflow as tf

from PIL import Image
from model.segnet import SegNet
from util.loss_functions import adv_loss, adv_loss_v2


class ADDA(SegNet):

    def __init__(self, args):
        SegNet.__init__(self, args)

        self.src_image_path = 'train.txt'
        # self.src_val_path = 'val.txt'
        self.tar_image_path = 'validation400.txt'
        # self.tar_val_path = 'rotate_180_val.txt'
        self.ckpt_dir = 'logs/dice/'
        self.src_ckpt_path = self.ckpt_dir + 'src_model.ckpt'
        self.tar_ckpt_path = self.ckpt_dir + 'tar_model.ckpt'
        self.scope_s = 's'  # source scope
        self.scope_t = 't'  # target scope
        self.scope_d = 'd'  # discriminator scope
        self.scope_g = 'g'  # generator scope
        self.lr_g = 1e-3
        self.lr_d = 1e-3


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
            tmp_saver.save(sess, self.ckpt_dir + 'model.ckpt')


    def discriminator(self, inputs, trainable=True):
        flat = tf.layers.Flatten()(inputs)

        fc1 = tf.layers.dense(flat, 576, activation=tf.nn.leaky_relu, trainable=trainable, name='fc1')
        # fc2 = tf.layers.dense(fc1, 576, activation=tf.nn.leaky_relu, trainable=trainable, name='fc2')
        fc3 = tf.layers.dense(flat, 1, activation=None, trainable=trainable, name='fc3')

        return tf.nn.sigmoid(fc3)


    def train(self):
        image_h = self.image_h
        image_w = self.image_w
        image_c = self.image_c
        batch_size = self.batch_size

        src_image, src_label = self.dataset.batch(batch_size=batch_size, path=self.src_image_path)
        tar_image, tar_label = self.dataset.batch(batch_size=batch_size, path=self.tar_image_path)

        # src_x_test, src_y_test = self.dataset.batch(batch_size=1, path=self.src_image_path)
        # tar_x_test, tar_y_test = self.dataset.batch(batch_size=1, path=self.tar_image_path)

        # src_image = tf.placeholder(tf.float32, shape=[batch_size, image_h, image_w, image_c])
        # src_label = tf.placeholder(tf.int64, shape=[batch_size, image_h, image_w, 1])
        #
        # tar_image = tf.placeholder(tf.float32, shape=[batch_size, image_h, image_w, image_c])
        # tar_label = tf.placeholder(tf.int64, shape=[batch_size, image_h, image_w, 1])

        # for source domain
        # imitate inference, for restoring SegNet model
        with tf.variable_scope(self.scope_s):
            src_encode_output = self.encoder(src_image, tf.constant(False))
            src_decode_output = self.decoder(src_encode_output, batch_size, tf.constant(False))
            src_logits, src_cls_loss = self.classifier(src_decode_output, src_label, self.loss_func)
        with tf.variable_scope(self.scope_d):
            dis_src = self.discriminator(src_encode_output)
            # dis_src = tf.Print(dis_src, [dis_src], summarize=batch_size, message="D_s: ")

        # for target domain
        with tf.variable_scope(self.scope_t):
            tar_encode_output = self.encoder(tar_image, phase_train=tf.constant(True))
            tar_decode_output = self.decoder(tar_encode_output, batch_size, tf.constant(False))
            tar_logits, tar_cls_loss = self.classifier(tar_decode_output, tar_label, self.loss_func)
        with tf.variable_scope(self.scope_d, reuse=True):
            dis_tar = self.discriminator(tar_encode_output)
            # dis_tar = tf.Print(dis_tar, [dis_tar], summarize=batch_size, message="D_t: ")

        # build loss
        g_loss, d_loss = adv_loss_v2(dis_src, dis_tar)

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

            print(self.src_vars)
            print(self.tar_vars)

            src_saver = tf.train.Saver(self.src_vars)
            tar_saver = tf.train.Saver(self.tar_vars)

            src_saver.restore(sess, self.src_ckpt_path)
            print("src model restored successfully!")
            tar_saver.restore(sess, self.tar_ckpt_path)
            print("tar model restored succesfully!")

            # filewriter = tf.summary.FileWriter(logdir=self.log_dir, graph=sess.graph)

            flat_src = tf.layers.Flatten()(src_encode_output)
            flat_tar = tf.layers.Flatten()(tar_encode_output)

            for i in range(self.max_steps):
                # _src_image, _tar_image = sess.run([src_image, tar_image])
                # feed_dict_train = {
                #     src_image: _src_image,
                #     tar_image: _tar_image
                # }
                _, d_loss_, = sess.run([optim_d, d_loss])
                _, g_loss_ = sess.run([optim_g, g_loss])

                # _src_output, _tar_output = sess.run([flat_src, flat_tar], feed_dict)
                #
                # np.savetxt('src_output_{}.csv'.format(i), _src_output, delimiter=',')
                # np.savetxt('tar_output_{}.csv'.format(i), _tar_output, delimiter=',')

                if i % 10 == 0:
                    self.output.write("step:{}, g_loss:{:.4f}, d_loss:{:.4f}".format(i, g_loss_, d_loss_))
                if i % 1000 == 0:
                    print("testing ...")
                    # _tar_image, _tar_label = sess.run([tar_x_test, tar_y_test])
                    # feed_dict_test = {
                    #     tar_image: _tar_image,
                    #     tar_label: _tar_label
                    # }
                    pred = tf.argmax(tar_logits, axis=3)
                    _, pred_image = sess.run([src_logits, pred])
                    pred_image = pred_image[0]
                    pred_image[pred_image == 0] = 0
                    pred_image[pred_image == 1] = 128
                    pred_image[pred_image == 2] = 255
                    pred_image = Image.fromarray(np.uint8(pred_image))
                    save_path = self.ckpt_dir + '{}.bmp'.format(i)
                    # pred_image = pred_image.resize((self.image_w_origin, self.image_h_origin))
                    pred_image.save(save_path)
                    print("image saved to {}".format(save_path))
