import tensorflow as tf
from model.segnet import SegNet
from util.loss_functions import adv_loss
from util.Inputs import get_filename_list, generate_batch, get_all_test_data


class ADDA(SegNet):
    def __init__(self, args):
        SegNet.__init__(self, args)
        self.src_image_path = 'train.txt'
        # self.src_val_path = 'val.txt'
        self.tar_image_path = 'rotate_180.txt'
        # self.tar_val_path = 'rotate_180_val.txt'
        self.ckpt_path = args.transfer
        self.scope_s = 's'  # source scope
        self.scope_t = 't'  # target scope
        self.scope_d = 'd'  # discriminator scope
        self.scope_g = 'g'  # generator scope
        self.lr_g = 1e-3
        self.lr_d = 1e-3


    def discriminator(self, inputs, reuse=False, trainable=True):
        flat = tf.layers.Flatten()(inputs)
        with tf.variable_scope(self.scope_d, reuse=reuse):
            fc1 = tf.layers.dense(flat, 576, activation=tf.nn.leaky_relu, trainable=trainable, name='fc1')
            fc2 = tf.layers.dense(fc1, 576, activation=tf.nn.leaky_relu, trainable=trainable, name='fc2')
            fc3 = tf.layers.dense(fc2, 1, activation=None, trainable=trainable, name='fc3')
        return fc3


    def train(self):
        image_h = self.image_h
        image_w = self.image_w
        image_c = self.image_c
        batch_size = self.batch_size
        src_x_filenames, src_y_filenames = get_filename_list(self.src_image_path)
        tar_x_filenames, tar_y_filenames = get_filename_list(self.tar_image_path)
        # src_val_x_filenames, src_val_y_filenames = get_filename_list(self.src_val_path)
        # tar_val_x_filenames, tar_val_y_filenames = get_filename_list(self.tar_val_path)

        src_x_train = tf.placeholder(tf.float32, shape=[self.batch_size, self.image_h, self.image_w, self.image_c])
        tar_x_train = tf.placeholder(tf.float32, shape=[self.batch_size, self.image_h, self.image_w, self.image_c])
        src_y_train = tf.placeholder(tf.float32, shape=[self.batch_size, self.image_h, self.image_w, 1])
        phase_train = tf.placeholder(tf.bool, name='phase_train')

        # for source domain
        # imitate inference, for restoring SegNet model
        src_encode_output = self.encoder(src_x_train, tf.constant(False))
        src_decode_output = self.decoder(src_encode_output, batch_size, tf.constant(False))
        logit_src, cls_loss_src = self.classifier(src_decode_output, src_y_train, self.loss_func)

        # variables to restore
        variable_averages = tf.train.ExponentialMovingAverage(self.moving_average_decay)
        src_variables = variable_averages.variables_to_restore()

        dis_src = self.discriminator(src_encode_output, reuse=False)

        # for target domain
        with tf.variable_scope(self.scope_t):
            tar_encode_output = self.encoder(tar_x_train, phase_train=tf.constant(True))
        dis_tar = self.discriminator(tar_encode_output, reuse=True)

        # build loss
        g_loss, d_loss = adv_loss(dis_src, dis_tar)

        # create optimizer for two task
        var_tar = tf.trainable_variables(self.scope_t)
        optim_g = tf.train.AdamOptimizer(self.lr_g).minimize(g_loss, var_list=var_tar)

        var_d = tf.trainable_variables(self.scope_d)
        optim_d = tf.train.AdamOptimizer(self.lr_d).minimize(d_loss, var_list=var_d)

        src_images, _ = generate_batch(src_x_filenames, src_y_filenames,
                                       batch_size, image_h, image_w, image_c)
        tar_images, _ = generate_batch(tar_x_filenames, tar_y_filenames,
                                       batch_size, image_h, image_w, image_c)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            src_saver = tf.train.Saver(src_variables)
            src_saver.restore(sess, self.ckpt_path)

            print("model restored successfully!")
            filewriter = tf.summary.FileWriter(logdir=self.log_dir, graph=sess.graph)

            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(self.max_steps):
                src_batch, tar_batch = sess.run([src_images, tar_images])
                feed_dict = {
                    src_x_train: src_batch,
                    tar_x_train: tar_batch,
                }
                _, d_loss_, = sess.run([optim_d, d_loss], feed_dict=feed_dict)
                _, g_loss_ = sess.run([optim_g, g_loss], feed_dict=feed_dict)
                if i % 10 == 0:
                    print("step:{}, g_loss:{:.4f}, d_loss:{:.4f}".format(i, g_loss_, d_loss_))

